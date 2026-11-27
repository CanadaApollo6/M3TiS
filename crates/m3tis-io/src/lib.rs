//! Zero-thread, lock-free Linux Dual-Ring `io_uring` + `O_DIRECT` engine for M3TiS.
//!
//! Architecture:
//! - Single-owner design: `DualRingIo` owns two independent `IoRing` instances (`demand` and
//!   `prefetch`) using `IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_COOP_TASKRUN` (or optional
//!   `IORING_SETUP_SQPOLL`). Zero host mutexes, zero worker thread spawns, zero ring contention.
//! - Direct zero-copy DMA from NVMe/dm-crypt straight into 4K-aligned CUDA pinned Host RAM
//!   (`cuMemHostAlloc`), with optional `IORING_REGISTER_BUFFERS` (`read_fixed`) to bypass
//!   per-I/O kernel `pin_user_pages` overhead.
//! - Tracks live NVMe bandwidth EMA (`bw_gbps_ema`) so callers auto-tune speculative prefetch
//!   depth `P = floor(gpu_window_s * bw_bytes_per_s / expert_bytes)` dynamically.

use std::ffi::{CString, c_int, c_uint, c_void};
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::RawFd;
use std::path::Path;
use std::time::Instant;

pub const SECTOR_BYTES: usize = 4096;
pub const MAX_BATCH_SQES: usize = 64;

const O_RDONLY: c_int = 0;
const O_CLOEXEC: c_int = 0x80000;
const O_DIRECT: c_int = 0x4000;

#[repr(C)]
struct M3tisUring {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    fn open(path: *const std::ffi::c_char, oflag: c_int, ...) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn pread(fd: c_int, buf: *mut c_void, count: usize, offset: i64) -> isize;

    fn m3tis_uring_init(entries: c_uint, want_sqpoll: c_int, out_errno: *mut c_int) -> *mut M3tisUring;
    fn m3tis_uring_is_sqpoll(h: *const M3tisUring) -> c_int;
    fn m3tis_uring_destroy(h: *mut M3tisUring);
    fn m3tis_uring_register_buffers(
        h: *mut M3tisUring,
        ptrs: *const *mut c_void,
        lens: *const usize,
        count: c_uint,
    ) -> c_int;
    fn m3tis_uring_queue_read(
        h: *mut M3tisUring,
        fd: c_int,
        buf: *mut c_void,
        len: c_uint,
        offset: u64,
        user_data: u64,
        fixed_buf_idx: c_int,
    ) -> c_int;
    fn m3tis_uring_submit(h: *mut M3tisUring) -> c_int;
    fn m3tis_uring_submit_and_wait(h: *mut M3tisUring, wait_nr: c_uint) -> c_int;
    fn m3tis_uring_reap(
        h: *mut M3tisUring,
        out_ud: *mut u64,
        out_res: *mut i32,
        max_cqes: c_uint,
        wait_nr: c_uint,
    ) -> c_int;
}

/// Owned `O_DIRECT` (with safe buffered fallback on tmpfs) read-only file descriptor.
pub struct DirectFile {
    fd: RawFd,
    pub len: u64,
    pub direct: bool,
}

impl DirectFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let meta = std::fs::metadata(path)?;
        let len = meta.len();
        let cpath = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "interior NUL in path"))?;

        let mut fd = unsafe { open(cpath.as_ptr(), O_RDONLY | O_CLOEXEC | O_DIRECT) };
        let mut direct = true;
        if fd < 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(22 /* EINVAL */) {
                direct = false;
                fd = unsafe { open(cpath.as_ptr(), O_RDONLY | O_CLOEXEC) };
            } else {
                return Err(err);
            }
        }
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        // Probe one 4K aligned read to verify filesystem truly permits O_DIRECT.
        if direct && len >= SECTOR_BYTES as u64 {
            #[repr(align(4096))]
            struct Page([u8; SECTOR_BYTES]);
            let mut probe = Page([0u8; SECTOR_BYTES]);
            let n = unsafe { pread(fd, probe.0.as_mut_ptr() as *mut c_void, SECTOR_BYTES, 0) };
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(22 /* EINVAL */) {
                    unsafe { close(fd) };
                    direct = false;
                    fd = unsafe { open(cpath.as_ptr(), O_RDONLY | O_CLOEXEC) };
                    if fd < 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
            }
        }
        Ok(Self { fd, len, direct })
    }

    #[inline]
    pub fn fd(&self) -> RawFd {
        self.fd
    }
}

impl Drop for DirectFile {
    fn drop(&mut self) {
        unsafe { close(self.fd) };
    }
}

/// One zero-copy 4K-aligned read request targeting pinned Host RAM.
#[derive(Debug, Clone, Copy)]
pub struct ReadOp {
    pub fd: RawFd,
    pub offset: u64,
    pub dst: *mut u8,
    pub len: usize,
    pub user_data: u64,
    /// Optional registered fixed-buffer index (`>= 0` uses `io_uring_prep_read_fixed`).
    pub fixed_buf_idx: i32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct IoStats {
    pub bytes: u64,
    pub ops: u64,
    pub submissions: u64,
}

/// One dedicated Linux `io_uring` instance (single-owner, zero locks).
pub struct IoRing {
    raw: *mut M3tisUring,
    pub entries: u32,
    pub sqpoll: bool,
    pub registered_bufs: bool,
}

impl IoRing {
    pub fn new(entries: u32, want_sqpoll: bool) -> io::Result<Self> {
        let mut err_no: c_int = 0;
        let raw = unsafe { m3tis_uring_init(entries, if want_sqpoll { 1 } else { 0 }, &mut err_no) };
        if raw.is_null() {
            return Err(io::Error::from_raw_os_error(err_no));
        }
        let sqpoll = unsafe { m3tis_uring_is_sqpoll(raw) != 0 };
        Ok(Self {
            raw,
            entries,
            sqpoll,
            registered_bufs: false,
        })
    }

    /// Register pinned host slab buffers with the kernel (`IORING_REGISTER_BUFFERS`) so `read_fixed`
    /// skips per-I/O `pin_user_pages` overhead. Note: Linux limits single iovec length to <= 1 GiB
    /// on older kernels; callers can chunk large slabs into <= 1 GiB registered buffers.
    pub fn register_buffers(&mut self, slices: &[(*mut u8, usize)]) -> io::Result<()> {
        if slices.is_empty() {
            return Ok(());
        }
        let ptrs: Vec<*mut c_void> = slices.iter().map(|&(p, _)| p as *mut c_void).collect();
        let lens: Vec<usize> = slices.iter().map(|&(_, l)| l).collect();
        let ret = unsafe {
            m3tis_uring_register_buffers(
                self.raw,
                ptrs.as_ptr(),
                lens.as_ptr(),
                slices.len() as c_uint,
            )
        };
        if ret < 0 {
            return Err(io::Error::from_raw_os_error(-ret));
        }
        self.registered_bufs = true;
        Ok(())
    }

    /// Submit a batch of 4K-aligned reads and block until every SQE completes (demand path).
    pub fn read_batch_sync(&mut self, ops: &[ReadOp], stats: &mut IoStats) -> io::Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        let max_chunk = (self.entries as usize).min(MAX_BATCH_SQES);
        for chunk in ops.chunks(max_chunk) {
            for op in chunk {
                debug_assert_eq!(op.offset % SECTOR_BYTES as u64, 0);
                debug_assert_eq!((op.dst as usize) % SECTOR_BYTES, 0);
                debug_assert_eq!(op.len % SECTOR_BYTES, 0);
                let fixed = if self.registered_bufs { op.fixed_buf_idx } else { -1 };
                let ret = unsafe {
                    m3tis_uring_queue_read(
                        self.raw,
                        op.fd,
                        op.dst as *mut c_void,
                        op.len as c_uint,
                        op.offset,
                        op.user_data,
                        fixed,
                    )
                };
                if ret < 0 {
                    return Err(io::Error::from_raw_os_error(-ret));
                }
            }
            let sub = unsafe { m3tis_uring_submit_and_wait(self.raw, chunk.len() as c_uint) };
            if sub < 0 {
                return Err(io::Error::from_raw_os_error(-sub));
            }
            stats.submissions += 1;
            let mut remaining = chunk.len();
            let mut ud = [0u64; MAX_BATCH_SQES];
            let mut res = [0i32; MAX_BATCH_SQES];
            let mut first_err: Option<io::Error> = None;
            while remaining > 0 {
                let got = unsafe {
                    m3tis_uring_reap(
                        self.raw,
                        ud.as_mut_ptr(),
                        res.as_mut_ptr(),
                        remaining as c_uint,
                        remaining as c_uint,
                    )
                };
                if got < 0 {
                    return Err(io::Error::from_raw_os_error(-got));
                }
                for i in 0..got as usize {
                    if res[i] < 0 {
                        if first_err.is_none() {
                            first_err = Some(io::Error::from_raw_os_error(-res[i]));
                        }
                    } else {
                        stats.bytes += res[i] as u64;
                        stats.ops += 1;
                    }
                }
                remaining -= got as usize;
            }
            if let Some(e) = first_err {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Submit up to `entries` reads NON-BLOCKING (returns in ~5 µs after `io_uring_enter`).
    pub fn submit_async(&mut self, ops: &[ReadOp], stats: &mut IoStats) -> io::Result<usize> {
        if ops.is_empty() {
            return Ok(0);
        }
        assert!(ops.len() <= (self.entries as usize).min(MAX_BATCH_SQES));
        for op in ops {
            debug_assert_eq!(op.offset % SECTOR_BYTES as u64, 0);
            debug_assert_eq!((op.dst as usize) % SECTOR_BYTES, 0);
            debug_assert_eq!(op.len % SECTOR_BYTES, 0);
            let fixed = if self.registered_bufs { op.fixed_buf_idx } else { -1 };
            let ret = unsafe {
                m3tis_uring_queue_read(
                    self.raw,
                    op.fd,
                    op.dst as *mut c_void,
                    op.len as c_uint,
                    op.offset,
                    op.user_data,
                    fixed,
                )
            };
            if ret < 0 {
                return Err(io::Error::from_raw_os_error(-ret));
            }
        }
        let sub = unsafe { m3tis_uring_submit(self.raw) };
        if sub < 0 {
            return Err(io::Error::from_raw_os_error(-sub));
        }
        stats.submissions += 1;
        Ok(ops.len())
    }

    /// Reap completions from an async batch.
    /// `wait_nr = 0`: strictly non-blocking user-space CQ peek.
    /// `wait_nr > 0`: blocks until at least `wait_nr` completions are ready.
    pub fn reap_async(
        &mut self,
        out_ud: &mut [u64],
        out_res: &mut [i32],
        wait_nr: u32,
        stats: &mut IoStats,
    ) -> io::Result<usize> {
        let cap = out_ud.len().min(out_res.len()).min(MAX_BATCH_SQES);
        let got = unsafe {
            m3tis_uring_reap(
                self.raw,
                out_ud.as_mut_ptr(),
                out_res.as_mut_ptr(),
                cap as c_uint,
                wait_nr,
            )
        };
        if got < 0 {
            return Err(io::Error::from_raw_os_error(-got));
        }
        let n = got as usize;
        for &r in &out_res[..n] {
            if r > 0 {
                stats.bytes += r as u64;
                stats.ops += 1;
            }
        }
        Ok(n)
    }
}

impl Drop for IoRing {
    fn drop(&mut self) {
        unsafe { m3tis_uring_destroy(self.raw) };
    }
}

/// Dual-Ring async I/O engine (`demand` + `prefetch`) with live NVMe bandwidth EMA for
/// auto-scaling speculative prefetch depth `P = floor(gpu_window_s * bw_ema / expert_bytes)`.
pub struct DualRingIo {
    pub demand: IoRing,
    pub prefetch: IoRing,
    pub demand_stats: IoStats,
    pub prefetch_stats: IoStats,
    /// Exponential moving average of effective NVMe read bandwidth (bytes/sec).
    pub bw_bps_ema: f64,
}

impl DualRingIo {
    pub fn new(entries: u32, want_sqpoll: bool) -> io::Result<Self> {
        let demand = IoRing::new(entries, want_sqpoll)?;
        let prefetch = IoRing::new(entries, want_sqpoll)?;
        Ok(Self {
            demand,
            prefetch,
            demand_stats: IoStats::default(),
            prefetch_stats: IoStats::default(),
            bw_bps_ema: 6.0e9, // calibrated prior for 990 PRO LUKS2, updated live
        })
    }

    pub fn read_demand(&mut self, ops: &[ReadOp]) -> io::Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        let b0 = self.demand_stats.bytes;
        let t0 = Instant::now();
        self.demand.read_batch_sync(ops, &mut self.demand_stats)?;
        let dt = t0.elapsed().as_secs_f64();
        let read_b = self.demand_stats.bytes.saturating_sub(b0);
        if read_b >= 4 * 1024 * 1024 && dt > 1e-5 {
            let sample = (read_b as f64) / dt;
            self.bw_bps_ema = 0.85 * self.bw_bps_ema + 0.15 * sample;
        }
        Ok(())
    }

    /// Compute optimal speculative prefetch count `P` matched to the measured GPU compute window `W`.
    pub fn optimal_prefetch_depth(&self, gpu_window_secs: f64, expert_bytes: usize, max_cap: usize) -> usize {
        if gpu_window_secs <= 0.0 || expert_bytes == 0 || max_cap == 0 {
            return 0;
        }
        let est = (gpu_window_secs * self.bw_bps_ema / (expert_bytes as f64)).floor() as usize;
        est.clamp(1, max_cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn dual_ring_sync_and_nonblocking_async_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("m3tis_io_test_{}.bin", std::process::id()));
        let mut data = vec![0u8; 4 * SECTOR_BYTES];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i * 31 + 7) & 0xFF) as u8;
        }
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(&data).unwrap();
            f.sync_all().unwrap();
        }
        let df = DirectFile::open(&tmp).unwrap();
        let mut io = DualRingIo::new(32, false).unwrap();

        #[repr(align(4096))]
        struct AlignedBuf([u8; 4 * SECTOR_BYTES]);
        let mut dst1 = Box::new(AlignedBuf([0u8; 4 * SECTOR_BYTES]));
        let mut dst2 = Box::new(AlignedBuf([0u8; 4 * SECTOR_BYTES]));

        // 1. Synchronous demand batch
        let ops1 = [
            ReadOp { fd: df.fd(), offset: 0, dst: dst1.0.as_mut_ptr(), len: 2 * SECTOR_BYTES, user_data: 10, fixed_buf_idx: -1 },
            ReadOp { fd: df.fd(), offset: (2 * SECTOR_BYTES) as u64, dst: unsafe { dst1.0.as_mut_ptr().add(2 * SECTOR_BYTES) }, len: 2 * SECTOR_BYTES, user_data: 11, fixed_buf_idx: -1 },
        ];
        io.read_demand(&ops1).unwrap();
        assert_eq!(&dst1.0[..], &data[..]);

        // 2. Non-blocking async prefetch submit + drain
        let ops2 = [
            ReadOp { fd: df.fd(), offset: 0, dst: dst2.0.as_mut_ptr(), len: 4 * SECTOR_BYTES, user_data: 99, fixed_buf_idx: -1 },
        ];
        let submitted = io.prefetch.submit_async(&ops2, &mut io.prefetch_stats).unwrap();
        assert_eq!(submitted, 1);
        let mut ud = [0u64; 8];
        let mut res = [0i32; 8];
        let got = io.prefetch.reap_async(&mut ud, &mut res, 1, &mut io.prefetch_stats).unwrap();
        assert_eq!(got, 1);
        assert_eq!(ud[0], 99);
        assert_eq!(res[0] as usize, 4 * SECTOR_BYTES);
        assert_eq!(&dst2.0[..], &data[..]);

        let _ = std::fs::remove_file(&tmp);
    }
}
