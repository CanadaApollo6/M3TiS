//! Clean-room Rust host wrapper for `sm_86` on-GPU EXL3 3-bit -> 2-bit MCG trellis requantization.

use crate::{CuDevicePtr, CudaContext, CudaFunction, CudaModule, Result, Stream};
use std::ffi::c_void;
use std::sync::Arc;

const EXL3_REQUANT_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/exl3_requant.cubin"));

pub struct Exl3RequantKernels {
    _module: CudaModule,
    requant_3b_to_2b: CudaFunction,
}

impl Exl3RequantKernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = ctx.load_module(EXL3_REQUANT_CUBIN)?;
        Ok(Self {
            requant_3b_to_2b: module.function("m3tis_exl3_requant_3b_to_2b")?,
            _module: module,
        })
    }

    /// Requantize `num_tiles` 3-bit (`k2 = 6`, 96 B/tile) MCG tiles -> 2-bit (`k2 = 4`, 64 B/tile) on GPU.
    pub fn requant_3b_to_2b(
        &self,
        stream: &Stream,
        mut in_trellis: CuDevicePtr,
        mut out_trellis: CuDevicePtr,
        num_tiles: usize,
    ) -> Result<()> {
        if num_tiles == 0 {
            return Ok(());
        }
        let mut nt_i = num_tiles as i32;
        let mut args: [*mut c_void; 3] = [
            &mut in_trellis as *mut _ as *mut c_void,
            &mut out_trellis as *mut _ as *mut c_void,
            &mut nt_i as *mut _ as *mut c_void,
        ];
        let grid_x = ((num_tiles + 1) / 2) as u32;
        unsafe {
            self.requant_3b_to_2b
                .launch((grid_x, 1, 1), (64, 1, 1), 0, stream, &mut args)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use m3tis_format::exl3_ref::decode_mcg_tile_16x16;

    #[test]
    fn gpu_exl3_requant_3b_to_2b_quality_and_speed() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.stream().unwrap();
        let req = Exl3RequantKernels::load(&ctx).unwrap();

        // Generate 32,768 synthetic 3-bit tiles (= 1 full GLM-5.3 4096x2048 expert matrix = 3 MiB -> 2 MiB!)
        let num_tiles = 32_768usize;
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut in_words = vec![0u32; num_tiles * 24];
        for w in &mut in_words {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *w = seed as u32;
        }
        let d_in = ctx.alloc_device(in_words.len() * 4).unwrap();
        let d_out = ctx.alloc_device(num_tiles * 16 * 4).unwrap();
        unsafe {
            stream
                .htod_async(d_in.ptr(), in_words.as_ptr() as *const u8, in_words.len() * 4)
                .unwrap();
        }
        stream.synchronize().unwrap();

        let t0 = std::time::Instant::now();
        req.requant_3b_to_2b(&stream, d_in.ptr(), d_out.ptr(), num_tiles)
            .unwrap();
        stream.synchronize().unwrap();
        let dt_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut out_words = vec![0u32; num_tiles * 16];
        unsafe {
            stream
                .dtoh_async(out_words.as_mut_ptr() as *mut u8, d_out.ptr(), out_words.len() * 4)
                .unwrap();
        }
        stream.synchronize().unwrap();

        // Measure cosine similarity & SNR on first 64 tiles via our exact CPU MCG decoder (verifying wraparound too!)
        let mut dot = 0.0f64;
        let mut n3 = 0.0f64;
        let mut n2 = 0.0f64;
        let mut err = 0.0f64;
        for t in 0..64 {
            let w3 = decode_mcg_tile_16x16(&in_words[t * 24..(t + 1) * 24], 6);
            let w2 = decode_mcg_tile_16x16(&out_words[t * 16..(t + 1) * 16], 4);
            for i in 0..256 {
                let a = w3[i] as f64;
                let b = w2[i] as f64;
                dot += a * b;
                n3 += a * a;
                n2 += b * b;
                err += (a - b) * (a - b);
            }
        }
        let cos = dot / (n3.sqrt() * n2.sqrt());
        let snr_db = 10.0 * (n3 / err.max(1e-30)).log10();
        println!(
            "[RTX 3090 EXL3 3b->2b 32-beam Viterbi] 32,768 tiles (1 matrix) in {dt_ms:.2} ms | cos_sim={cos:.5} | SQNR={snr_db:.2} dB"
        );
        assert!(cos > 0.94, "2-bit MCG Viterbi quality too low: cos={cos:.5}");
    }
}
