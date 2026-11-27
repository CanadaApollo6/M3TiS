//! Clean-room Rust host wrappers for M3TiS `sm_86` dense linear & Q4 packing kernels (`linear_q4.cu`).

use crate::{CuDevicePtr, CudaContext, CudaFunction, CudaModule, DeviceBuf, Result, Stream};
use std::ffi::c_void;
use std::sync::Arc;

const LINEAR_Q4_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_q4.cubin"));

/// On-GPU affine 4-bit quantized weight matrix (`group_size = 64`, 4.5 bits/weight -> 3.55x VRAM savings vs BF16).
pub struct Q4Weight {
    pub qdata: DeviceBuf,  // [N, K / 8] u32
    pub scales: DeviceBuf, // [N, K / 64] half2 (.x = scale, .y = zero)
    pub n: usize,
    pub k: usize,
}

impl Q4Weight {
    #[inline]
    pub fn vram_bytes(&self) -> usize {
        self.qdata.len() + self.scales.len()
    }
}

/// On-GPU affine 3-bit quantized weight matrix (`group_size = 64`, 3.50 bits/weight -> -22.2% VRAM & HBM bytes vs Q4!).
pub struct Q3Weight {
    pub qdata: DeviceBuf,  // [N, (K / 64) * 6] u32
    pub scales: DeviceBuf, // [N, K / 64] half2 (.x = scale, .y = zero)
    pub n: usize,
    pub k: usize,
}

impl Q3Weight {
    #[inline]
    pub fn vram_bytes(&self) -> usize {
        self.qdata.len() + self.scales.len()
    }
}

pub struct LinearKernels {
    _module: CudaModule,
    pack_bf16_to_q4: CudaFunction,
    linear_q4: CudaFunction,
    linear_q4_tc: CudaFunction,
    linear_q4_tc64: CudaFunction,
    pack_bf16_to_q3: CudaFunction,
    linear_q3: CudaFunction,
    linear_q3_tc: CudaFunction,
    linear_q3_tc64: CudaFunction,
    linear_bf16: CudaFunction,
    linear_bf16_tc16: CudaFunction,
    linear_bf16_tc64: CudaFunction,
    swiglu_bf16: CudaFunction,
    swiglu_split_bf16: CudaFunction,
    put_shared_slot_f32: CudaFunction,
    add_f32_inplace: CudaFunction,
}

impl LinearKernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = ctx.load_module(LINEAR_Q4_CUBIN)?;
        Ok(Self {
            pack_bf16_to_q4: module.function("m3tis_pack_bf16_to_q4")?,
            linear_q4: module.function("m3tis_linear_q4")?,
            linear_q4_tc: module.function("m3tis_linear_q4_tc")?,
            linear_q4_tc64: module.function("m3tis_linear_q4_tc64")?,
            pack_bf16_to_q3: module.function("m3tis_pack_bf16_to_q3")?,
            linear_q3: module.function("m3tis_linear_q3")?,
            linear_q3_tc: module.function("m3tis_linear_q3_tc")?,
            linear_q3_tc64: module.function("m3tis_linear_q3_tc64")?,
            linear_bf16: module.function("m3tis_linear_bf16")?,
            linear_bf16_tc16: module.function("m3tis_linear_bf16_tc16")?,
            linear_bf16_tc64: module.function("m3tis_linear_bf16_tc64")?,
            swiglu_bf16: module.function("m3tis_swiglu_bf16")?,
            swiglu_split_bf16: module.function("m3tis_swiglu_split_bf16")?,
            put_shared_slot_f32: module.function("m3tis_put_shared_slot_f32")?,
            add_f32_inplace: module.function("m3tis_add_f32_inplace")?,
            _module: module,
        })
    }

    /// Pack a device-resident row-major BF16 weight matrix `[N, K]` (`K % 64 == 0`) into affine Q4 (`gs = 64`).
    pub fn pack_bf16_to_q4(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Stream,
        mut d_bf16: CuDevicePtr,
        n: usize,
        k: usize,
    ) -> Result<Q4Weight> {
        assert!(n > 0 && k > 0 && k % 64 == 0, "Q4 pack requires K % 64 == 0 (got N={n}, K={k})");
        let qdata = ctx.alloc_device(n * (k / 8) * 4)?;
        let scales = ctx.alloc_device(n * (k / 64) * 4)?;
        let mut q_ptr = qdata.ptr();
        let mut s_ptr = scales.ptr();
        let mut n_i = n as i32;
        let mut k_i = k as i32;
        let mut args: [*mut c_void; 5] = [
            &mut d_bf16 as *mut _ as *mut c_void,
            &mut q_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.pack_bf16_to_q4.launch(
                (n as u32, (k / 64) as u32, 1),
                (32, 1, 1),
                0,
                stream,
                &mut args,
            )?;
        }
        Ok(Q4Weight { qdata, scales, n, k })
    }

    /// Compute `Y[rows, N] = X[rows, K] @ W_q4[N, K]^T` into BF16 or FP32 output with arbitrary row strides.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_q4(
        &self,
        stream: &Stream,
        mut x_ptr: CuDevicePtr,
        x_stride: usize,
        w: &Q4Weight,
        mut out_ptr: CuDevicePtr,
        out_stride: usize,
        rows: usize,
        out_f32: bool,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let mut xs_i = x_stride as i32;
        let mut q_ptr = w.qdata.ptr();
        let mut s_ptr = w.scales.ptr();
        let mut os_i = out_stride as i32;
        let mut r_i = rows as i32;
        let mut n_i = w.n as i32;
        let mut k_i = w.k as i32;
        let mut f32_i: i32 = if out_f32 { 1 } else { 0 };
        let mut args: [*mut c_void; 10] = [
            &mut x_ptr as *mut _ as *mut c_void,
            &mut xs_i as *mut _ as *mut c_void,
            &mut q_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut os_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut f32_i as *mut _ as *mut c_void,
        ];
        // Dispatch Ampere sm_86 Tensor Core Q4 GEMM:
        // - rows >= 32: 64x128 8-warp native BF16 TC tile (`linear_q4_tc64`)
        // - rows >= 16 || (rows >= 2 && N >= 256): 16x64 coalesced K=128 BF16 TC tile (`linear_q4_tc`)
        // - rows == 1 (or tiny N): 16-group 128-bit warp GEMV (`linear_q4`)
        if rows >= 32 && x_stride % 8 == 0 {
            let gx = ((w.n + 127) / 128) as u32;
            let gy = ((rows + 63) / 64) as u32;
            unsafe {
                self.linear_q4_tc64
                    .launch((gx, gy, 1), (256, 1, 1), 0, stream, &mut args)
            }
        } else if (rows >= 16 || (rows >= 2 && w.n >= 256)) && x_stride % 8 == 0 {
            let gx = ((w.n + 63) / 64) as u32;
            let gy = ((rows + 15) / 16) as u32;
            unsafe {
                self.linear_q4_tc
                    .launch((gx, gy, 1), (128, 1, 1), 0, stream, &mut args)
            }
        } else {
            let gx = ((w.n + 3) / 4) as u32;
            let gy = ((rows + 3) / 4) as u32;
            unsafe {
                self.linear_q4
                    .launch((gx, gy, 1), (128, 1, 1), 0, stream, &mut args)
            }
        }
    }

    /// Pack a device-resident row-major BF16 weight matrix `[N, K]` (`K % 64 == 0`) into affine Q3 (`gs = 64`, 3.50 bpw).
    pub fn pack_bf16_to_q3(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Stream,
        mut d_bf16: CuDevicePtr,
        n: usize,
        k: usize,
    ) -> Result<Q3Weight> {
        assert!(n > 0 && k > 0 && k % 64 == 0, "Q3 pack requires K % 64 == 0 (got N={n}, K={k})");
        let qdata = ctx.alloc_device(n * (k / 64) * 6 * 4)?;
        let scales = ctx.alloc_device(n * (k / 64) * 4)?;
        let mut q_ptr = qdata.ptr();
        let mut s_ptr = scales.ptr();
        let mut n_i = n as i32;
        let mut k_i = k as i32;
        let mut args: [*mut c_void; 5] = [
            &mut d_bf16 as *mut _ as *mut c_void,
            &mut q_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.pack_bf16_to_q3.launch(
                (n as u32, (k / 64) as u32, 1),
                (32, 1, 1),
                0,
                stream,
                &mut args,
            )?;
        }
        Ok(Q3Weight { qdata, scales, n, k })
    }

    /// Compute `Y[rows, N] = X[rows, K] @ W_q3[N, K]^T` into BF16 or FP32 output with arbitrary row strides.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_q3(
        &self,
        stream: &Stream,
        mut x_ptr: CuDevicePtr,
        x_stride: usize,
        w: &Q3Weight,
        mut out_ptr: CuDevicePtr,
        out_stride: usize,
        rows: usize,
        out_f32: bool,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let mut xs_i = x_stride as i32;
        let mut q_ptr = w.qdata.ptr();
        let mut s_ptr = w.scales.ptr();
        let mut os_i = out_stride as i32;
        let mut r_i = rows as i32;
        let mut n_i = w.n as i32;
        let mut k_i = w.k as i32;
        let mut f32_i: i32 = if out_f32 { 1 } else { 0 };
        let mut args: [*mut c_void; 10] = [
            &mut x_ptr as *mut _ as *mut c_void,
            &mut xs_i as *mut _ as *mut c_void,
            &mut q_ptr as *mut _ as *mut c_void,
            &mut s_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut os_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut f32_i as *mut _ as *mut c_void,
        ];
        if rows >= 32 && x_stride % 8 == 0 {
            let gx = ((w.n + 127) / 128) as u32;
            let gy = ((rows + 63) / 64) as u32;
            unsafe {
                self.linear_q3_tc64
                    .launch((gx, gy, 1), (256, 1, 1), 0, stream, &mut args)
            }
        } else if (rows >= 16 || (rows >= 2 && w.n >= 256)) && x_stride % 8 == 0 {
            let gx = ((w.n + 63) / 64) as u32;
            let gy = ((rows + 15) / 16) as u32;
            unsafe {
                self.linear_q3_tc
                    .launch((gx, gy, 1), (128, 1, 1), 0, stream, &mut args)
            }
        } else {
            let gx = ((w.n + 3) / 4) as u32;
            let gy = ((rows + 3) / 4) as u32;
            unsafe {
                self.linear_q3
                    .launch((gx, gy, 1), (128, 1, 1), 0, stream, &mut args)
            }
        }
    }

    /// Compute `Y[rows, N] = X[rows, K] @ W_bf16[N, K]^T` into BF16 or FP32 output with arbitrary row strides.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16(
        &self,
        stream: &Stream,
        mut x_ptr: CuDevicePtr,
        x_stride: usize,
        mut w_ptr: CuDevicePtr,
        mut out_ptr: CuDevicePtr,
        out_stride: usize,
        rows: usize,
        n: usize,
        k: usize,
        out_f32: bool,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let mut xs_i = x_stride as i32;
        let mut os_i = out_stride as i32;
        let mut r_i = rows as i32;
        let mut n_i = n as i32;
        let mut k_i = k as i32;
        let mut f32_i: i32 = if out_f32 { 1 } else { 0 };
        let mut args: [*mut c_void; 9] = [
            &mut x_ptr as *mut _ as *mut c_void,
            &mut xs_i as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut os_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut f32_i as *mut _ as *mut c_void,
        ];
        if rows >= 16 && k % 64 == 0 && x_stride % 8 == 0 {
            let gx = ((n + 127) / 128) as u32;
            let gy = ((rows + 63) / 64) as u32;
            unsafe {
                self.linear_bf16_tc64
                    .launch((gx, gy, 1), (256, 1, 1), 0, stream, &mut args)
            }
        } else if rows >= 2 && n >= 1024 && k % 64 == 0 && x_stride % 8 == 0 {
            let gx = ((n + 63) / 64) as u32;
            let gy = ((rows + 15) / 16) as u32;
            unsafe {
                self.linear_bf16_tc16
                    .launch((gx, gy, 1), (128, 1, 1), 0, stream, &mut args)
            }
        } else {
            let gx = ((n + 3) / 4) as u32;
            let gy = ((rows + 3) / 4) as u32;
            unsafe {
                self.linear_bf16
                    .launch((gx, gy, 1), (128, 1, 1), 0, stream, &mut args)
            }
        }
    }

    /// Clamped GLM-5.3 SwiGLU: `out = bf16(bf16(silu(clamp_hi(gate, lim))) * clamp(up, -lim, lim))`.
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu_bf16(
        &self,
        stream: &Stream,
        mut gate_ptr: CuDevicePtr,
        mut up_ptr: CuDevicePtr,
        mut out_ptr: CuDevicePtr,
        rows: usize,
        n: usize,
        mut limit: f32,
    ) -> Result<()> {
        let total = rows * n;
        if total == 0 {
            return Ok(());
        }
        let mut r_i = rows as i32;
        let mut n_i = n as i32;
        let mut args: [*mut c_void; 6] = [
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut up_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
            &mut limit as *mut _ as *mut c_void,
        ];
        let blocks = ((total + 255) / 256) as u32;
        unsafe {
            self.swiglu_bf16
                .launch((blocks, 1, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }

    /// Clamped GLM-5.3 SwiGLU on fused `[gate | up]` (`gu[rows, 2 * N] -> out[rows, N]`).
    pub fn swiglu_split_bf16(
        &self,
        stream: &Stream,
        mut gu_ptr: CuDevicePtr,
        mut out_ptr: CuDevicePtr,
        rows: usize,
        n: usize,
        mut limit: f32,
    ) -> Result<()> {
        let total = rows * n;
        if total == 0 {
            return Ok(());
        }
        let mut r_i = rows as i32;
        let mut n_i = n as i32;
        let mut args: [*mut c_void; 5] = [
            &mut gu_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
            &mut limit as *mut _ as *mut c_void,
        ];
        let blocks = ((total + 255) / 256) as u32;
        unsafe {
            self.swiglu_split_bf16
                .launch((blocks, 1, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }

    /// Copy shared expert FP32 output `sy[rows, D]` into the last slot (`slots - 1`) of `slot_y[rows, slots, D]`.
    pub fn put_shared_slot_f32(
        &self,
        stream: &Stream,
        mut sy_ptr: CuDevicePtr,
        mut slot_y_ptr: CuDevicePtr,
        rows: usize,
        d: usize,
        slots: usize,
    ) -> Result<()> {
        let total = rows * d;
        if total == 0 {
            return Ok(());
        }
        let mut r_i = rows as i32;
        let mut d_i = d as i32;
        let mut s_i = slots as i32;
        let mut args: [*mut c_void; 5] = [
            &mut sy_ptr as *mut _ as *mut c_void,
            &mut slot_y_ptr as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut d_i as *mut _ as *mut c_void,
            &mut s_i as *mut _ as *mut c_void,
        ];
        let blocks = ((total + 255) / 256) as u32;
        unsafe {
            self.put_shared_slot_f32
                .launch((blocks, 1, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }

    pub fn add_f32_inplace(
        &self,
        stream: &Stream,
        mut dst_ptr: CuDevicePtr,
        mut src_ptr: CuDevicePtr,
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let mut c_i = count as i32;
        let mut args: [*mut c_void; 3] = [
            &mut dst_ptr as *mut _ as *mut c_void,
            &mut src_ptr as *mut _ as *mut c_void,
            &mut c_i as *mut _ as *mut c_void,
        ];
        let blocks = ((count + 255) / 256) as u32;
        unsafe {
            self.add_f32_inplace
                .launch((blocks, 1, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use m3tis_format::exl3_ref::f32_to_bf16;

    fn bf16_to_f32(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += (x as f64) * (y as f64);
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn gpu_pack_bf16_to_q4_and_linear_q4_bf16_parity() {
        let ctx = CudaContext::new(0).expect("RTX 3090");
        let stream = ctx.stream().unwrap();
        let lin = LinearKernels::load(&ctx).expect("load linear_q4.cubin");

        let rows = 5usize; // exercises both full 4-row tile + 1-row tail tile!
        let n = 131usize;  // non-multiple of 4 to test N tail!
        let k = 320usize;  // 5 groups of 64 -> exercises 4-group unroll + 1 tail group!

        let mut x_bf16 = vec![0u16; rows * k];
        let mut x_f32 = vec![0.0f32; rows * k];
        for i in 0..rows * k {
            let v = (((i * 29 + 7) % 199) as f32 - 99.0) * 0.015;
            let b = f32_to_bf16(v);
            x_bf16[i] = b;
            x_f32[i] = bf16_to_f32(b);
        }

        let mut w_bf16 = vec![0u16; n * k];
        let mut w_f32 = vec![0.0f32; n * k];
        for i in 0..n * k {
            let v = (((i * 53 + 13) % 251) as f32 - 125.0) * 0.008;
            let b = f32_to_bf16(v);
            w_bf16[i] = b;
            w_f32[i] = bf16_to_f32(b);
        }

        // CPU FP64 ground truth
        let mut y_ref = vec![0.0f32; rows * n];
        for r in 0..rows {
            for col in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += (x_f32[r * k + kk] as f64) * (w_f32[col * k + kk] as f64);
                }
                y_ref[r * n + col] = acc as f32;
            }
        }

        let d_x = ctx.alloc_device(x_bf16.len() * 2).unwrap();
        let d_w = ctx.alloc_device(w_bf16.len() * 2).unwrap();
        let d_out_bf16 = ctx.alloc_device(rows * n * 4).unwrap();
        let d_out_q4 = ctx.alloc_device(rows * n * 4).unwrap();
        unsafe {
            stream.htod_async(d_x.ptr(), x_bf16.as_ptr() as *const u8, x_bf16.len() * 2).unwrap();
            stream.htod_async(d_w.ptr(), w_bf16.as_ptr() as *const u8, w_bf16.len() * 2).unwrap();
        }

        let w_q4 = lin.pack_bf16_to_q4(&ctx, &stream, d_w.ptr(), n, k).unwrap();
        lin.linear_bf16(&stream, d_x.ptr(), k, d_w.ptr(), d_out_bf16.ptr(), n, rows, n, k, true).unwrap();
        lin.linear_q4(&stream, d_x.ptr(), k, &w_q4, d_out_q4.ptr(), n, rows, true).unwrap();

        let mut h_bf16 = vec![0.0f32; rows * n];
        let mut h_q4 = vec![0.0f32; rows * n];
        unsafe {
            stream.dtoh_async(h_bf16.as_mut_ptr() as *mut u8, d_out_bf16.ptr(), rows * n * 4).unwrap();
            stream.dtoh_async(h_q4.as_mut_ptr() as *mut u8, d_out_q4.ptr(), rows * n * 4).unwrap();
        }
        stream.synchronize().unwrap();

        let cos_bf16 = cos_sim(&h_bf16, &y_ref);
        let cos_q4 = cos_sim(&h_q4, &y_ref);

        // Also compute exact CPU Q4 dequant reference to prove GPU pack + GEMM is bit-accurate!
        let mut y_q4_cpu = vec![0.0f32; rows * n];
        let mut w_deq = vec![0.0f32; n * k];
        for row in 0..n {
            for grp in 0..(k / 64) {
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for j in 0..64 {
                    let v = w_f32[row * k + grp * 64 + j];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let sc = if mx - mn > 1e-12 { (mx - mn) / 15.0 } else { 0.0 };
                for j in 0..64 {
                    let v = w_f32[row * k + grp * 64 + j];
                    let q = if sc > 0.0 { ((v - mn) / sc).round().clamp(0.0, 15.0) } else { 0.0 };
                    w_deq[row * k + grp * 64 + j] = q * sc + mn;
                }
            }
        }
        for r in 0..rows {
            for col in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += (x_f32[r * k + kk] as f64) * (w_deq[col * k + kk] as f64);
                }
                y_q4_cpu[r * n + col] = acc as f32;
            }
        }
        let cos_q4_vs_cpu_q4 = cos_sim(&h_q4, &y_q4_cpu);
        let cos_cpu_q4_vs_bf16 = cos_sim(&y_q4_cpu, &y_ref);
        println!(
            "[RTX 3090 linear] bf16={cos_bf16:.8}, q4_vs_bf16={cos_q4:.8}, cpu_q4_vs_bf16={cos_cpu_q4_vs_bf16:.8}, gpu_q4_vs_cpu_q4={cos_q4_vs_cpu_q4:.8}"
        );
        println!(
            "  sample y_ref[0..4]={:?} h_q4[0..4]={:?} y_q4_cpu[0..4]={:?}",
            &y_ref[0..4],
            &h_q4[0..4],
            &y_q4_cpu[0..4]
        );
        assert!(cos_bf16 > 0.999999, "BF16 linear mismatch: {cos_bf16}");
        assert!(
            cos_q4_vs_cpu_q4 > 0.999,
            "GPU Q4 != CPU Q4 oracle: {cos_q4_vs_cpu_q4}"
        );
        assert!(
            cos_q4 >= cos_cpu_q4_vs_bf16 - 0.001,
            "GPU Q4 quality below CPU Q4 oracle: gpu={cos_q4} cpu={cos_cpu_q4_vs_bf16}"
        );

        // Also test clamped SwiGLU (limit = 10.0)
        let g_h = [f32_to_bf16(15.0), f32_to_bf16(-2.0), f32_to_bf16(0.5)];
        let u_h = [f32_to_bf16(20.0), f32_to_bf16(-20.0), f32_to_bf16(2.0)];
        let d_g = ctx.alloc_device(6).unwrap();
        let d_u = ctx.alloc_device(6).unwrap();
        let d_o = ctx.alloc_device(6).unwrap();
        unsafe {
            stream.htod_async(d_g.ptr(), g_h.as_ptr() as *const u8, 6).unwrap();
            stream.htod_async(d_u.ptr(), u_h.as_ptr() as *const u8, 6).unwrap();
        }
        lin.swiglu_bf16(&stream, d_g.ptr(), d_u.ptr(), d_o.ptr(), 1, 3, 10.0).unwrap();
        let mut o_h = [0u16; 3];
        unsafe {
            stream.dtoh_async(o_h.as_mut_ptr() as *mut u8, d_o.ptr(), 6).unwrap();
        }
        stream.synchronize().unwrap();
        let o0 = bf16_to_f32(o_h[0]);
        let exp0 = (10.0f32 / (1.0 + (-10.0f32).exp())) * 10.0;
        assert!((o0 - exp0).abs() < 1.0, "SwiGLU clamp failed: got {o0}, expected {exp0}");

        // Also test sm_86 Tensor Core Q4 prefill path (rows = 21 >= 16, K = 1088 exercises 16-group + tail!)
        let tc_rows = 21usize;
        let tc_n = 131usize;
        let tc_k = 1088usize;
        let mut tx_bf16 = vec![0u16; tc_rows * tc_k];
        let mut tx_f32 = vec![0.0f32; tc_rows * tc_k];
        for i in 0..tc_rows * tc_k {
            let b = f32_to_bf16((((i * 31 + 17) % 197) as f32 - 98.0) * 0.012);
            tx_bf16[i] = b;
            tx_f32[i] = bf16_to_f32(b);
        }
        let mut tw_bf16 = vec![0u16; tc_n * tc_k];
        let mut tw_f32 = vec![0.0f32; tc_n * tc_k];
        for i in 0..tc_n * tc_k {
            let b = f32_to_bf16((((i * 67 + 5) % 241) as f32 - 120.0) * 0.009);
            tw_bf16[i] = b;
            tw_f32[i] = bf16_to_f32(b);
        }
        let d_tx = ctx.alloc_device(tx_bf16.len() * 2).unwrap();
        let d_tw = ctx.alloc_device(tw_bf16.len() * 2).unwrap();
        let d_ty_tc = ctx.alloc_device(tc_rows * tc_n * 4).unwrap();
        let d_ty_bf = ctx.alloc_device(tc_rows * tc_n * 4).unwrap();
        let d_ty_q4_ref = ctx.alloc_device(tc_rows * tc_n * 4).unwrap();
        unsafe {
            stream.htod_async(d_tx.ptr(), tx_bf16.as_ptr() as *const u8, tx_bf16.len() * 2).unwrap();
            stream.htod_async(d_tw.ptr(), tw_bf16.as_ptr() as *const u8, tw_bf16.len() * 2).unwrap();
        }
        let tw_q4 = lin.pack_bf16_to_q4(&ctx, &stream, d_tw.ptr(), tc_n, tc_k).unwrap();
        lin.linear_q4(&stream, d_tx.ptr(), tc_k, &tw_q4, d_ty_tc.ptr(), tc_n, tc_rows, true).unwrap();
        // Compare TC Q4 against scalar Q4 (15 rows + 6 rows) AND unquantized BF16!
        lin.linear_q4(&stream, d_tx.ptr(), tc_k, &tw_q4, d_ty_q4_ref.ptr(), tc_n, 15, true).unwrap();
        lin.linear_q4(&stream, d_tx.at(15 * tc_k * 2), tc_k, &tw_q4, d_ty_q4_ref.at(15 * tc_n * 4), tc_n, 6, true).unwrap();
        lin.linear_bf16(&stream, d_tx.ptr(), tc_k, d_tw.ptr(), d_ty_bf.ptr(), tc_n, tc_rows, tc_n, tc_k, true).unwrap();
        let mut h_ty_tc = vec![0.0f32; tc_rows * tc_n];
        let mut h_ty_q4 = vec![0.0f32; tc_rows * tc_n];
        let mut h_ty_bf = vec![0.0f32; tc_rows * tc_n];
        unsafe {
            stream.dtoh_async(h_ty_tc.as_mut_ptr() as *mut u8, d_ty_tc.ptr(), tc_rows * tc_n * 4).unwrap();
            stream.dtoh_async(h_ty_q4.as_mut_ptr() as *mut u8, d_ty_q4_ref.ptr(), tc_rows * tc_n * 4).unwrap();
            stream.dtoh_async(h_ty_bf.as_mut_ptr() as *mut u8, d_ty_bf.ptr(), tc_rows * tc_n * 4).unwrap();
        }
        stream.synchronize().unwrap();
        let cos_tc_vs_q4 = cos_sim(&h_ty_tc, &h_ty_q4);
        let cos_tc_vs_bf = cos_sim(&h_ty_tc, &h_ty_bf);
        println!(
            "[RTX 3090 linear_q4_tc mma.sync.m16n8k16] rows={tc_rows} n={tc_n} k={tc_k} vs_q4={cos_tc_vs_q4:.8} vs_bf16={cos_tc_vs_bf:.8}"
        );
        assert!(cos_tc_vs_q4 > 0.99999, "Tensor Core Q4 != Scalar Q4: {cos_tc_vs_q4}");
        assert!(cos_tc_vs_bf > 0.990, "Tensor Core Q4 vs BF16 too low: {cos_tc_vs_bf}");

        // Test Q3 (3.50 bpw, gs=64) packer + 16-group decode + Tensor Core m16n8k16 parity!
        let tw_q3 = lin.pack_bf16_to_q3(&ctx, &stream, d_tw.ptr(), tc_n, tc_k).unwrap();
        let d_ty_q3_tc = ctx.alloc_device(tc_rows * tc_n * 4).unwrap();
        let d_ty_q3_ref = ctx.alloc_device(tc_rows * tc_n * 4).unwrap();
        lin.linear_q3(&stream, d_tx.ptr(), tc_k, &tw_q3, d_ty_q3_tc.ptr(), tc_n, tc_rows, true).unwrap();
        // Exercise BOTH nr==1 fast 16-group path (row 0) AND multi-row scalar path (rows 1..21)!
        lin.linear_q3(&stream, d_tx.ptr(), tc_k, &tw_q3, d_ty_q3_ref.ptr(), tc_n, 1, true).unwrap();
        lin.linear_q3(&stream, d_tx.at(tc_k * 2), tc_k, &tw_q3, d_ty_q3_ref.at(tc_n * 4), tc_n, 15, true).unwrap();
        lin.linear_q3(&stream, d_tx.at(16 * tc_k * 2), tc_k, &tw_q3, d_ty_q3_ref.at(16 * tc_n * 4), tc_n, 5, true).unwrap();
        let mut h_q3_tc = vec![0.0f32; tc_rows * tc_n];
        let mut h_q3_ref = vec![0.0f32; tc_rows * tc_n];
        unsafe {
            stream.dtoh_async(h_q3_tc.as_mut_ptr() as *mut u8, d_ty_q3_tc.ptr(), tc_rows * tc_n * 4).unwrap();
            stream.dtoh_async(h_q3_ref.as_mut_ptr() as *mut u8, d_ty_q3_ref.ptr(), tc_rows * tc_n * 4).unwrap();
        }
        stream.synchronize().unwrap();
        let cos_q3_tc_vs_ref = cos_sim(&h_q3_tc, &h_q3_ref);
        let cos_q3_vs_bf = cos_sim(&h_q3_tc, &h_ty_bf);
        println!(
            "[RTX 3090 linear_q3 3.50bpw] tc_vs_decode={cos_q3_tc_vs_ref:.8} vs_bf16={cos_q3_vs_bf:.8}"
        );
        assert!(cos_q3_tc_vs_ref > 0.99999, "Tensor Core Q3 != Decode Q3: {cos_q3_tc_vs_ref}");
        assert!(cos_q3_vs_bf > 0.975, "Q3 vs BF16 too low: {cos_q3_vs_bf}");

        // Milestone 6: Test 64-row TC64 kernels (rows = 69 -> 1 full 64-row tile + 5-row tail!) for BOTH Q4 & Q3!
        let r64 = 69usize;
        let x64: Vec<u16> = (0..r64 * tc_k)
            .map(|i| f32_to_bf16((((i * 29 + 11) % 199) as f32 - 99.0) * 0.011))
            .collect();
        let d_x64 = ctx.alloc_device(x64.len() * 2).unwrap();
        let d_y64_q4 = ctx.alloc_device(r64 * tc_n * 4).unwrap();
        let d_y64_q4_ref = ctx.alloc_device(r64 * tc_n * 4).unwrap();
        let d_y64_q3 = ctx.alloc_device(r64 * tc_n * 4).unwrap();
        let d_y64_q3_ref = ctx.alloc_device(r64 * tc_n * 4).unwrap();
        unsafe {
            stream.htod_async(d_x64.ptr(), x64.as_ptr() as *const u8, x64.len() * 2).unwrap();
        }
        lin.linear_q4(&stream, d_x64.ptr(), tc_k, &tw_q4, d_y64_q4.ptr(), tc_n, r64, true).unwrap();
        lin.linear_q3(&stream, d_x64.ptr(), tc_k, &tw_q3, d_y64_q3.ptr(), tc_n, r64, true).unwrap();
        for off in (0..r64).step_by(16) {
            let m = (r64 - off).min(16);
            lin.linear_q4(&stream, d_x64.at(off * tc_k * 2), tc_k, &tw_q4, d_y64_q4_ref.at(off * tc_n * 4), tc_n, m, true).unwrap();
            lin.linear_q3(&stream, d_x64.at(off * tc_k * 2), tc_k, &tw_q3, d_y64_q3_ref.at(off * tc_n * 4), tc_n, m, true).unwrap();
        }
        let mut h64_q4 = vec![0.0f32; r64 * tc_n];
        let mut h64_q4_ref = vec![0.0f32; r64 * tc_n];
        let mut h64_q3 = vec![0.0f32; r64 * tc_n];
        let mut h64_q3_ref = vec![0.0f32; r64 * tc_n];
        unsafe {
            stream.dtoh_async(h64_q4.as_mut_ptr() as *mut u8, d_y64_q4.ptr(), r64 * tc_n * 4).unwrap();
            stream.dtoh_async(h64_q4_ref.as_mut_ptr() as *mut u8, d_y64_q4_ref.ptr(), r64 * tc_n * 4).unwrap();
            stream.dtoh_async(h64_q3.as_mut_ptr() as *mut u8, d_y64_q3.ptr(), r64 * tc_n * 4).unwrap();
            stream.dtoh_async(h64_q3_ref.as_mut_ptr() as *mut u8, d_y64_q3_ref.ptr(), r64 * tc_n * 4).unwrap();
        }
        stream.synchronize().unwrap();
        let c_q4_64 = cos_sim(&h64_q4, &h64_q4_ref);
        let c_q3_64 = cos_sim(&h64_q3, &h64_q3_ref);
        // Also test BF16 TC64 (rows = 69) vs exact scalar BF16 (4-row slices)!
        let d_y64_bf = ctx.alloc_device(r64 * tc_n * 4).unwrap();
        let d_y64_bf_ref = ctx.alloc_device(r64 * tc_n * 4).unwrap();
        lin.linear_bf16(&stream, d_x64.ptr(), tc_k, d_tw.ptr(), d_y64_bf.ptr(), tc_n, r64, tc_n, tc_k, true).unwrap();
        for off in (0..r64).step_by(4) {
            let m = (r64 - off).min(4);
            lin.linear_bf16(&stream, d_x64.at(off * tc_k * 2), tc_k, d_tw.ptr(), d_y64_bf_ref.at(off * tc_n * 4), tc_n, m, tc_n, tc_k, true).unwrap();
        }
        let mut h64_bf = vec![0.0f32; r64 * tc_n];
        let mut h64_bf_ref = vec![0.0f32; r64 * tc_n];
        unsafe {
            stream.dtoh_async(h64_bf.as_mut_ptr() as *mut u8, d_y64_bf.ptr(), r64 * tc_n * 4).unwrap();
            stream.dtoh_async(h64_bf_ref.as_mut_ptr() as *mut u8, d_y64_bf_ref.ptr(), r64 * tc_n * 4).unwrap();
        }
        stream.synchronize().unwrap();
        let c_bf_64 = cos_sim(&h64_bf, &h64_bf_ref);
        println!("[RTX 3090 linear_q4/q3/bf16 tc64 64x128] rows={r64} q4_cos={c_q4_64:.8} q3_cos={c_q3_64:.8} bf16_cos={c_bf_64:.8}");
        assert!(c_q4_64 > 0.99998, "Q4 tc64 mismatch: {c_q4_64}");
        assert!(c_q3_64 > 0.99998, "Q3 tc64 mismatch: {c_q3_64}");
        assert!(c_bf_64 > 0.999999, "BF16 tc64 mismatch: {c_bf_64}");

        // Milestone 8: Test 16x64 native BF16 TC16 path (rows = 11, N = 1024, K = 128 -> KDA f_b/g_b shape!)
        let (r16, n16, k16) = (11usize, 1024usize, 128usize);
        let x16: Vec<u16> = (0..r16 * k16)
            .map(|i| f32_to_bf16((((i * 41 + 9) % 193) as f32 - 96.0) * 0.014))
            .collect();
        let w16: Vec<u16> = (0..n16 * k16)
            .map(|i| f32_to_bf16((((i * 73 + 3) % 239) as f32 - 119.0) * 0.008))
            .collect();
        let d_x16 = ctx.alloc_device(x16.len() * 2).unwrap();
        let d_w16 = ctx.alloc_device(w16.len() * 2).unwrap();
        let d_y16_tc = ctx.alloc_device(r16 * n16 * 4).unwrap();
        let d_y16_ref = ctx.alloc_device(r16 * n16 * 4).unwrap();
        unsafe {
            stream.htod_async(d_x16.ptr(), x16.as_ptr() as *const u8, x16.len() * 2).unwrap();
            stream.htod_async(d_w16.ptr(), w16.as_ptr() as *const u8, w16.len() * 2).unwrap();
        }
        lin.linear_bf16(&stream, d_x16.ptr(), k16, d_w16.ptr(), d_y16_tc.ptr(), n16, r16, n16, k16, true).unwrap();
        for off in (0..r16).step_by(4) {
            let m = (r16 - off).min(4);
            lin.linear_bf16(&stream, d_x16.at(off * k16 * 2), k16, d_w16.ptr(), d_y16_ref.at(off * n16 * 4), n16, m, n16, k16, true).unwrap();
        }
        let mut h16_tc = vec![0.0f32; r16 * n16];
        let mut h16_ref = vec![0.0f32; r16 * n16];
        unsafe {
            stream.dtoh_async(h16_tc.as_mut_ptr() as *mut u8, d_y16_tc.ptr(), r16 * n16 * 4).unwrap();
            stream.dtoh_async(h16_ref.as_mut_ptr() as *mut u8, d_y16_ref.ptr(), r16 * n16 * 4).unwrap();
        }
        stream.synchronize().unwrap();
        let c_bf_16 = cos_sim(&h16_tc, &h16_ref);
        println!("[RTX 3090 linear_bf16_tc16 16x64] rows={r16} n={n16} k={k16} bf16_cos={c_bf_16:.8}");
        assert!(c_bf_16 > 0.999999, "BF16 tc16 mismatch: {c_bf_16}");
    }
}
