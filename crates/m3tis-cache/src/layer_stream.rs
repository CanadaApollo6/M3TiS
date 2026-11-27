//! Milestone 6: Lock-Free Asynchronous Layer-Ahead NVMe Streamer for Expert-Major Prefill.
//!
//! Runs one dedicated background OS thread owning its own `O_DIRECT` fd + `SINGLE_ISSUER` `io_uring`,
//! streaming all cold NVMe experts for Layers `3..44` in strict ascending pack-offset order at full
//! NVMe sequential bandwidth (~6.8 GB/s) into a circular pinned-Host-RAM ring buffer (`AHEAD_RING_SLOTS`)
//! starting at `t = 0` — overlapping 100% of NVMe I/O with Dense Layers 0..2, Attention (KDA/MLA/DSA),
//! and Tier-1/Tier-2 MoE execution!
//!
//! Synchronization:
//! - Single-producer (`head`), single-consumer (`tail`) lock-free atomics (`Acquire`/`Release`) in the hot loop.
//! - Zero deadlocks even when a layer has more cold experts than `AHEAD_RING_SLOTS`, because the consumer
//!   advances `tail` incrementally every 8-expert wave!
//! - Thread sleeps on a `Condvar` with 0% CPU between prefills and joins cleanly before freeing pinned buffers.

use m3tis_cuda::{CudaContext, PinnedHostBuf};
use m3tis_format::PackCatalog;
use m3tis_io::{DirectFile, IoRing, IoStats, ReadOp};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Circular pinned-Host-RAM ring capacity (`240` experts = ~2.1 GiB at 9 MiB hot stride, >1 full layer).
pub const AHEAD_RING_SLOTS: usize = 240;
/// NVMe `io_uring` burst size (`16` experts x 3 projs = `48` SQEs <= `64` -> saturates Samsung 990 PRO!).
const AHEAD_IO_BATCH: usize = 16;

#[derive(Clone, Copy)]
struct AheadItem {
    offsets: [u64; 3],
    lens: [usize; 3],
}

struct WorkerCmd {
    epoch: u64,
    items: Vec<AheadItem>,
}

struct Shared {
    ring_ptrs: [usize; 3], // raw `*mut u8` as usize (Send + Sync; lifetime guarded by JoinHandle in Drop!)
    strides: [usize; 3],
    ring_slots: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
    cancel: AtomicBool,
    shutdown: AtomicBool,
    done_gen: AtomicU64,
    bytes_read: AtomicU64,
    experts_read: AtomicU64,
    cmd: Mutex<WorkerCmd>,
    cv: Condvar,
    last_err: Mutex<Option<String>>,
}

pub struct LayerAheadStreamer {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
    /// Kept alive strictly longer than `worker` (dropped AFTER `worker.join()` in `Drop`)!
    _ring_slabs: [PinnedHostBuf; 3],
    active_gen: u64,
    active_len: usize,
}

impl LayerAheadStreamer {
    pub fn open(
        ctx: &Arc<CudaContext>,
        pack_path: &Path,
        strides: [usize; 3],
        ring_slots: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ring_slots = ring_slots.max(32);
        let mut slabs_v = Vec::with_capacity(3);
        let mut ring_ptrs = [0usize; 3];
        for p in 0..3 {
            let buf = ctx.alloc_pinned(ring_slots * strides[p])?;
            ring_ptrs[p] = buf.as_ptr() as usize;
            slabs_v.push(buf);
        }
        let ring_slabs: [PinnedHostBuf; 3] = slabs_v.try_into().ok().unwrap();

        let shared = Arc::new(Shared {
            ring_ptrs,
            strides,
            ring_slots,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            done_gen: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            experts_read: AtomicU64::new(0),
            cmd: Mutex::new(WorkerCmd {
                epoch: 0,
                items: Vec::new(),
            }),
            cv: Condvar::new(),
            last_err: Mutex::new(None),
        });

        let path_owned = pack_path.to_path_buf();
        let sh = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("m3tis-nvme-ahead".to_string())
            .spawn(move || worker_loop(path_owned, sh))?;

        Ok(Self {
            shared,
            worker: Some(worker),
            _ring_slabs: ring_slabs,
            active_gen: 0,
            active_len: 0,
        })
    }

    #[inline]
    pub fn ring_slots(&self) -> usize {
        self.shared.ring_slots
    }

    #[inline]
    pub fn pinned_ram_bytes(&self) -> usize {
        self._ring_slabs[0].len() + self._ring_slabs[1].len() + self._ring_slabs[2].len()
    }

    /// Start streaming `plan_keys` in exact order into the circular pinned RAM ring (non-blocking!).
    pub fn start_plan(&mut self, catalog: &PackCatalog, plan_keys: &[usize]) {
        self.finish_and_drain();
        if plan_keys.is_empty() {
            return;
        }
        let mut items = Vec::with_capacity(plan_keys.len());
        for &gkey in plan_keys {
            let e = &catalog.entries[gkey];
            items.push(AheadItem {
                offsets: [e.components[0].offset, e.components[1].offset, e.components[2].offset],
                lens: [
                    e.components[0].aligned_len,
                    e.components[1].aligned_len,
                    e.components[2].aligned_len,
                ],
            });
        }
        self.shared.head.store(0, Ordering::Release);
        self.shared.tail.store(0, Ordering::Release);
        self.shared.cancel.store(false, Ordering::Release);
        *self.shared.last_err.lock().unwrap() = None;
        self.active_gen += 1;
        self.active_len = items.len();
        {
            let mut guard = self.shared.cmd.lock().unwrap();
            guard.items = items;
            guard.epoch = self.active_gen;
        }
        self.shared.cv.notify_one();
    }

    /// Block until plan index `pidx` (`0..active_len`) has been completely DMA'd into pinned RAM.
    #[inline]
    pub fn wait_ready(&self, pidx: usize) -> io::Result<()> {
        debug_assert!(pidx < self.active_len);
        let mut spins = 0u32;
        while self.shared.head.load(Ordering::Acquire) <= pidx {
            if self.shared.done_gen.load(Ordering::Acquire) >= self.active_gen {
                if let Some(msg) = self.shared.last_err.lock().unwrap().as_ref() {
                    return Err(io::Error::new(io::ErrorKind::Other, msg.clone()));
                }
                if self.shared.head.load(Ordering::Acquire) > pidx {
                    break;
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "layer-ahead streamer stopped before requested expert",
                ));
            }
            if spins < 2000 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                thread::sleep(Duration::from_micros(25));
            }
        }
        Ok(())
    }

    /// Return pinned host pointer for component `comp` (`0..3`) of ready plan index `pidx`.
    #[inline]
    pub fn slot_ptr(&self, comp: usize, pidx: usize) -> *const u8 {
        let rslot = pidx % self.shared.ring_slots;
        let base = self.shared.ring_ptrs[comp] as *const u8;
        unsafe { base.add(rslot * self.shared.strides[comp]) }
    }

    /// Release all ring slots strictly before `new_tail` so the NVMe worker can reuse them immediately.
    #[inline]
    pub fn release_up_to(&self, new_tail: usize) {
        let cur = self.shared.tail.load(Ordering::Relaxed);
        if new_tail > cur {
            self.shared.tail.store(new_tail.min(self.active_len), Ordering::Release);
        }
    }

    /// Drain stats delta `(bytes, experts)` accumulated by the background NVMe streamer.
    pub fn take_stats(&self) -> (u64, u64) {
        let b = self.shared.bytes_read.swap(0, Ordering::Relaxed);
        let e = self.shared.experts_read.swap(0, Ordering::Relaxed);
        (b, e)
    }

    /// Ensure any active background job stops cleanly (called on end of prefill or error unwind).
    pub fn finish_and_drain(&mut self) {
        if self.active_gen == 0 {
            return;
        }
        if self.shared.done_gen.load(Ordering::Acquire) < self.active_gen {
            self.shared.cancel.store(true, Ordering::Release);
            self.shared.tail.store(self.active_len, Ordering::Release);
            while self.shared.done_gen.load(Ordering::Acquire) < self.active_gen {
                thread::sleep(Duration::from_micros(50));
            }
        }
        self.active_len = 0;
    }
}

impl Drop for LayerAheadStreamer {
    fn drop(&mut self) {
        self.shared.cancel.store(true, Ordering::Release);
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.tail.store(usize::MAX / 2, Ordering::Release);
        {
            let _g = self.shared.cmd.lock().unwrap();
            self.shared.cv.notify_all();
        }
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
        // `_ring_slabs` drops AFTER `handle.join()`, guaranteeing zero dangling DMA/pointer access!
    }
}

fn worker_loop(pack_path: std::path::PathBuf, shared: Arc<Shared>) {
    let open_res = (|| -> io::Result<(DirectFile, IoRing)> {
        let file = DirectFile::open(&pack_path)?;
        let ring = IoRing::new(64, false)?;
        Ok((file, ring))
    })();
    let (file, mut ring) = match open_res {
        Ok(pair) => pair,
        Err(e) => {
            *shared.last_err.lock().unwrap() = Some(format!("layer-ahead init failed: {e}"));
            loop {
                let mut g = shared.cmd.lock().unwrap();
                if shared.shutdown.load(Ordering::Acquire) {
                    return;
                }
                let ep = g.epoch;
                shared.done_gen.store(ep, Ordering::Release);
                g = shared.cv.wait(g).unwrap();
                drop(g);
            }
        }
    };

    let fd = file.fd();
    let ring_slots = shared.ring_slots;
    let mut ops = Vec::with_capacity(AHEAD_IO_BATCH * 3);
    let mut seen_gen = 0u64;

    loop {
        let (job_gen, items) = {
            let mut g = shared.cmd.lock().unwrap();
            while g.epoch == seen_gen && !shared.shutdown.load(Ordering::Acquire) {
                g = shared.cv.wait(g).unwrap();
            }
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            (g.epoch, std::mem::take(&mut g.items))
        };
        seen_gen = job_gen;

        let mut cursor = 0usize;
        let total = items.len();
        let mut io_stats = IoStats::default();
        while cursor < total && !shared.cancel.load(Ordering::Acquire) {
            let tail = shared.tail.load(Ordering::Acquire);
            let inflight = cursor.saturating_sub(tail);
            if inflight >= ring_slots {
                thread::sleep(Duration::from_micros(30));
                continue;
            }
            let avail = ring_slots - inflight;
            let batch = avail.min(AHEAD_IO_BATCH).min(total - cursor);
            ops.clear();
            let mut batch_bytes = 0u64;
            for i in 0..batch {
                let idx = cursor + i;
                let rslot = idx % ring_slots;
                let it = items[idx];
                for p in 0..3 {
                    let dst = (shared.ring_ptrs[p] as *mut u8).wrapping_add(rslot * shared.strides[p]);
                    ops.push(ReadOp {
                        fd,
                        offset: it.offsets[p],
                        dst,
                        len: it.lens[p],
                        user_data: (i * 3 + p) as u64,
                        fixed_buf_idx: -1,
                    });
                    batch_bytes += it.lens[p] as u64;
                }
            }
            if let Err(e) = ring.read_batch_sync(&ops, &mut io_stats) {
                *shared.last_err.lock().unwrap() = Some(format!("layer-ahead io_uring read failed: {e}"));
                break;
            }
            cursor += batch;
            shared.bytes_read.fetch_add(batch_bytes, Ordering::Relaxed);
            shared.experts_read.fetch_add(batch as u64, Ordering::Relaxed);
            shared.head.store(cursor, Ordering::Release);
        }
        shared.done_gen.store(job_gen, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use m3tis_format::{ComponentSpan, ExpertEntry, PackVersion};
    use std::io::Write;

    #[test]
    fn layer_ahead_streamer_wraparound_byte_exact_and_clean_abort() {
        let ctx = CudaContext::new(0).expect("RTX 3090");
        let tmp = std::env::temp_dir().join(format!("m3tis_ahead_test_{}.m3tis", std::process::id()));

        let n_exp = 96usize; // > 3x ring_slots(32) -> tests multiple wraparounds!
        let comp_len = 4096usize;
        let mut f = std::fs::File::create(&tmp).unwrap();
        let mut entries = Vec::with_capacity(n_exp);
        let mut expected = Vec::with_capacity(n_exp);
        let mut off = 0u64;
        for e in 0..n_exp {
            let mut c_bytes = Vec::new();
            let mut spans = [ComponentSpan { offset: 0, len: comp_len, aligned_len: comp_len, bits: 2 }; 3];
            for p in 0..3 {
                let mut buf = vec![0u8; comp_len];
                for (j, b) in buf.iter_mut().enumerate() {
                    *b = ((e * 37 + p * 113 + j * 7) & 0xFF) as u8;
                }
                f.write_all(&buf).unwrap();
                spans[p] = ComponentSpan { offset: off, len: comp_len, aligned_len: comp_len, bits: 2 };
                off += comp_len as u64;
                c_bytes.push(buf);
            }
            entries.push(ExpertEntry {
                key: e,
                layer: 0,
                expert: e as u16,
                zipf_rank: e as u32,
                components: spans,
            });
            expected.push(c_bytes);
        }
        drop(f);

        let catalog = PackCatalog {
            version: PackVersion::V2MixedZipf,
            layers: 1,
            experts_per_layer: n_exp,
            max_strides: [comp_len; 3],
            file_bytes: off,
            entries,
        };
        let mut streamer = LayerAheadStreamer::open(&ctx, &tmp, [comp_len; 3], 32).unwrap();
        let plan: Vec<usize> = (0..n_exp).collect();
        streamer.start_plan(&catalog, &plan);

        // Consume in waves of 8, releasing ring slots incrementally across 3 full wraparounds!
        for wave_start in (0..n_exp).step_by(8) {
            let wave_end = (wave_start + 8).min(n_exp);
            streamer.wait_ready(wave_end - 1).unwrap();
            for idx in wave_start..wave_end {
                for p in 0..3 {
                    let got = unsafe { std::slice::from_raw_parts(streamer.slot_ptr(p, idx), comp_len) };
                    assert_eq!(got, &expected[idx][p][..], "mismatch at expert {idx} comp {p}");
                }
            }
            streamer.release_up_to(wave_end);
        }
        let (bytes, exps) = streamer.take_stats();
        assert_eq!(exps, n_exp as u64);
        assert_eq!(bytes, (n_exp * 3 * comp_len) as u64);

        // Also test clean mid-flight abort + drop
        streamer.start_plan(&catalog, &plan);
        streamer.wait_ready(0).unwrap();
        streamer.finish_and_drain();
        drop(streamer);
        let _ = std::fs::remove_file(&tmp);
    }
}
