//! Zero-copy AVX2+FMA+F16C Hybrid CPU EXL3 MoE Execution Engine.
//!
//! Executes Tier-2 Pinned-RAM-resident EXL3 experts directly on host CPU cores (zero PCIe expert
//! transfers, zero Tier-1 VRAM evictions!) concurrently while the RTX 3090 runs Tier-1 VRAM hits.

use std::ffi::c_void;

pub const MAX_CPU_JOB_ROWS: usize = 8;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CpuExl3ExpertJob {
    pub gate_t: *const u32,
    pub up_t: *const u32,
    pub down_t: *const u32,
    pub suh_g: *const u16,
    pub suh_u: *const u16,
    pub svh_g: *const u16,
    pub svh_u: *const u16,
    pub suh_d: *const u16,
    pub svh_d: *const u16,
    pub k2_g: i32,
    pub k2_u: i32,
    pub k2_d: i32,
    pub num_rows: i32,
    pub row_idx: [i32; MAX_CPU_JOB_ROWS],
    pub row_wt: [f32; MAX_CPU_JOB_ROWS],
}

unsafe extern "C" {
    fn m3tis_cpu_exl3_create(requested_threads: i32) -> *mut c_void;
    fn m3tis_cpu_exl3_destroy(handle: *mut c_void);
    fn m3tis_cpu_exl3_num_threads(handle: *const c_void) -> i32;
    fn m3tis_cpu_exl3_submit_async(
        handle: *mut c_void,
        x_bf16: *const u16,
        total_rows: i32,
        dims: i32,
        inter: i32,
        swiglu_limit: f32,
        jobs: *const CpuExl3ExpertJob,
        num_jobs: i32,
        out_f32: *mut f32,
    );
    fn m3tis_cpu_exl3_wait_async(handle: *mut c_void);
    fn m3tis_cpu_exl3_run_batch(
        handle: *mut c_void,
        x_bf16: *const u16,
        total_rows: i32,
        dims: i32,
        inter: i32,
        swiglu_limit: f32,
        jobs: *const CpuExl3ExpertJob,
        num_jobs: i32,
        out_f32: *mut f32,
    );
}

pub struct CpuExl3Engine {
    handle: *mut c_void,
    num_threads: usize,
}

unsafe impl Send for CpuExl3Engine {}
unsafe impl Sync for CpuExl3Engine {}

impl CpuExl3Engine {
    /// Create persistent AVX2+FMA+F16C worker pool (`requested_threads == 0` auto-selects optimal P-core count).
    pub fn new(requested_threads: usize) -> Self {
        let handle = unsafe { m3tis_cpu_exl3_create(requested_threads as i32) };
        assert!(!handle.is_null(), "m3tis_cpu_exl3_create returned null");
        let num_threads = unsafe { m3tis_cpu_exl3_num_threads(handle) } as usize;
        Self { handle, num_threads }
    }

    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Launch a non-blocking CPU EXL3 batch (<1 us return!) so host P-cores run concurrently with NVMe + RTX 3090!
    ///
    /// # Safety
    /// `x_bf16`, `out_f32`, and all trellis/scale pointers inside `jobs` must remain valid until `wait_async()` returns.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit_async(
        &self,
        x_bf16: &[u16],
        total_rows: usize,
        dims: usize,
        inter: usize,
        swiglu_limit: f32,
        jobs: &[CpuExl3ExpertJob],
        out_f32: &mut [f32],
    ) {
        if jobs.is_empty() || total_rows == 0 {
            return;
        }
        debug_assert!(x_bf16.len() >= total_rows * dims);
        debug_assert!(out_f32.len() >= total_rows * dims);
        unsafe {
            m3tis_cpu_exl3_submit_async(
                self.handle,
                x_bf16.as_ptr(),
                total_rows as i32,
                dims as i32,
                inter as i32,
                swiglu_limit,
                jobs.as_ptr(),
                jobs.len() as i32,
                out_f32.as_mut_ptr(),
            );
        }
    }

    /// Wait for any in-flight `submit_async` batch to finish writing `out_f32`.
    #[inline]
    pub fn wait_async(&self) {
        unsafe { m3tis_cpu_exl3_wait_async(self.handle) };
    }

    /// Synchronous convenience wrapper (`submit_async` + `wait_async`).
    ///
    /// # Safety
    /// All trellis/scale pointers inside `jobs` must remain valid and immutable in RAM for the duration of this call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_batch(
        &self,
        x_bf16: &[u16],
        total_rows: usize,
        dims: usize,
        inter: usize,
        swiglu_limit: f32,
        jobs: &[CpuExl3ExpertJob],
        out_f32: &mut [f32],
    ) {
        if jobs.is_empty() || total_rows == 0 {
            return;
        }
        debug_assert!(x_bf16.len() >= total_rows * dims);
        debug_assert!(out_f32.len() >= total_rows * dims);
        unsafe {
            m3tis_cpu_exl3_run_batch(
                self.handle,
                x_bf16.as_ptr(),
                total_rows as i32,
                dims as i32,
                inter as i32,
                swiglu_limit,
                jobs.as_ptr(),
                jobs.len() as i32,
                out_f32.as_mut_ptr(),
            );
        }
    }
}

impl Drop for CpuExl3Engine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { m3tis_cpu_exl3_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use m3tis_format::exl3_ref::{f32_to_bf16, reference_exl3_expert_mlp};
    use std::time::Instant;

    fn make_f16(v: f32) -> u16 {
        // Quick exact f32 -> f16 for simple positive/negative normal values in tests
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let frac = ((bits >> 13) & 0x3FF) as u16;
        if exp <= 0 { sign } else { sign | ((exp as u16) << 10) | frac }
    }

    #[test]
    fn cpu_avx2_exl3_matches_fp64_reference_2b_3b_4b_and_benchmarks_real_dims() {
        let eng = CpuExl3Engine::new(0);
        // 1. Strict mathematical parity on 2b (k2=4), 3b (k2=6), 4b (k2=8) across multi-row + multi-job!
        let dims = 256usize;
        let inter = 256usize;
        let total_rows = 5usize;
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next_u32 = || -> u32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u32
        };

        let mut x_bf16 = vec![0u16; total_rows * dims];
        let mut x_f32 = vec![0.0f32; total_rows * dims];
        for i in 0..total_rows * dims {
            let v = ((next_u32() as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32 * 0.4;
            let b = f32_to_bf16(v);
            x_bf16[i] = b;
            x_f32[i] = f32::from_bits((b as u32) << 16);
        }

        let k2s = [4usize, 6usize, 8usize]; // 2b, 3b, 4b
        let mut trellises: Vec<[Vec<u8>; 3]> = Vec::new();
        let mut scales_vec: Vec<[Vec<u16>; 6]> = Vec::new();
        for &k2 in &k2s {
            let tw_bytes = (dims / 16) * (inter / 16) * 4 * k2 * 4;
            let mut mk = || {
                let mut v = vec![0u8; tw_bytes];
                for ch in v.chunks_exact_mut(4) {
                    ch.copy_from_slice(&next_u32().to_le_bytes());
                }
                v
            };
            trellises.push([mk(), mk(), mk()]);
            let mut mks = |n: usize| {
                (0..n)
                    .map(|_| {
                        let s = 0.08 + ((next_u32() & 0xFF) as f32 / 255.0) * 0.12;
                        make_f16(if (next_u32() & 1) == 0 { s } else { -s })
                    })
                    .collect::<Vec<u16>>()
            };
            scales_vec.push([mks(dims), mks(dims), mks(inter), mks(inter), mks(inter), mks(dims)]);
        }

        let mut jobs = Vec::new();
        let mut ref_out = vec![0.0f32; total_rows * dims];
        let wts = [[0.55f32, 0.30, 0.25, 0.40, 0.15], [0.20, 0.45, 0.35, 0.10, 0.60], [0.25, 0.25, 0.40, 0.50, 0.25]];
        for e in 0..3 {
            let mut job = CpuExl3ExpertJob {
                gate_t: trellises[e][0].as_ptr() as *const u32,
                up_t: trellises[e][1].as_ptr() as *const u32,
                down_t: trellises[e][2].as_ptr() as *const u32,
                suh_g: scales_vec[e][0].as_ptr(),
                suh_u: scales_vec[e][1].as_ptr(),
                svh_g: scales_vec[e][2].as_ptr(),
                svh_u: scales_vec[e][3].as_ptr(),
                suh_d: scales_vec[e][4].as_ptr(),
                svh_d: scales_vec[e][5].as_ptr(),
                k2_g: k2s[e] as i32,
                k2_u: k2s[e] as i32,
                k2_d: k2s[e] as i32,
                num_rows: total_rows as i32,
                row_idx: [0; MAX_CPU_JOB_ROWS],
                row_wt: [0.0; MAX_CPU_JOB_ROWS],
            };
            for r in 0..total_rows {
                job.row_idx[r] = r as i32;
                job.row_wt[r] = wts[e][r];
                let y = reference_exl3_expert_mlp(
                    &x_f32[r * dims..(r + 1) * dims],
                    &trellises[e][0],
                    &trellises[e][1],
                    &trellises[e][2],
                    k2s[e],
                    k2s[e],
                    k2s[e],
                    &scales_vec[e][0],
                    &scales_vec[e][1],
                    &scales_vec[e][2],
                    &scales_vec[e][3],
                    &scales_vec[e][4],
                    &scales_vec[e][5],
                    dims,
                    inter,
                    10.0,
                );
                for c in 0..dims {
                    ref_out[r * dims + c] += wts[e][r] * y[c];
                }
            }
            jobs.push(job);
        }

        let mut cpu_out = vec![0.0f32; total_rows * dims];
        unsafe {
            eng.run_batch(&x_bf16, total_rows, dims, inter, 10.0, &jobs, &mut cpu_out);
        }

        let mut dot = 0.0f64;
        let mut n1 = 0.0f64;
        let mut n2 = 0.0f64;
        let mut max_abs = 0.0f32;
        for i in 0..total_rows * dims {
            let a = cpu_out[i] as f64;
            let b = ref_out[i] as f64;
            dot += a * b;
            n1 += a * a;
            n2 += b * b;
            max_abs = max_abs.max((cpu_out[i] - ref_out[i]).abs());
        }
        let cos = dot / (n1.sqrt() * n2.sqrt()).max(1e-30);
        println!(
            "[CPU AVX2 EXL3 2b+3b+4b 5-row parity | threads={}] cos_sim={cos:.8} max_abs={max_abs:.6}",
            eng.num_threads()
        );
        assert!(cos > 0.99995, "CPU EXL3 parity failed: cos={cos} max_abs={max_abs}");

        // 2. Real GLM-5.3 expert size microbenchmark (dims=4096, inter=2048 -> 6 MiB 2b & 9 MiB 3b!)
        let gd = 4096usize;
        let gi = 2048usize;
        let xb = vec![f32_to_bf16(0.1); 4 * gd];
        let sc_d = vec![make_f16(0.1); gd];
        let sc_i = vec![make_f16(0.1); gi];
        let mut out_b = vec![0.0f32; 4 * gd];
        for &(label, k2) in &[("2.0bpw (6 MiB)", 4usize), ("3.0bpw (9 MiB)", 6usize)] {
            let mat_u32 = (gd / 16) * (gi / 16) * 4 * k2;
            let g_t = vec![0x12345678u32; mat_u32];
            let u_t = vec![0x9ABCDEF0u32; mat_u32];
            let d_t = vec![0x0F1E2D3Cu32; mat_u32];
            for &nr in &[1usize, 4usize] {
                let job = CpuExl3ExpertJob {
                    gate_t: g_t.as_ptr(),
                    up_t: u_t.as_ptr(),
                    down_t: d_t.as_ptr(),
                    suh_g: sc_d.as_ptr(),
                    suh_u: sc_d.as_ptr(),
                    svh_g: sc_i.as_ptr(),
                    svh_u: sc_i.as_ptr(),
                    suh_d: sc_i.as_ptr(),
                    svh_d: sc_d.as_ptr(),
                    k2_g: k2 as i32,
                    k2_u: k2 as i32,
                    k2_d: k2 as i32,
                    num_rows: nr as i32,
                    row_idx: [0, 1, 2, 3, 0, 0, 0, 0],
                    row_wt: [0.5; MAX_CPU_JOB_ROWS],
                };
                // Warmup
                for _ in 0..3 {
                    unsafe { eng.run_batch(&xb, nr, gd, gi, 10.0, &[job], &mut out_b); }
                }
                let iters = 20;
                let t0 = Instant::now();
                for _ in 0..iters {
                    unsafe { eng.run_batch(&xb, nr, gd, gi, 10.0, &[job], &mut out_b); }
                }
                let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
                let bytes = (mat_u32 * 4 * 3) as f64;
                let gbs = bytes / (us * 1e-6) / 1e9;
                std::hint::black_box(out_b[0]);
                println!("  [CPU AVX2 GLM-5.3 {label} rows={nr}] {us:7.1} us ({gbs:5.1} GB/s effective!)");
            }
        }
    }
}
