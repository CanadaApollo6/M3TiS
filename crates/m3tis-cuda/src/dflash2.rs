//! Clean-room Rust host wrappers for M3TiS `sm_86` DFlash2 speculative drafter kernels (`dflash2.cu`).

use crate::{CuDevicePtr, CudaContext, CudaFunction, CudaModule, Result, Stream};
use std::ffi::c_void;
use std::sync::Arc;

const DFLASH2_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dflash2.cubin"));

pub struct DFlash2Kernels {
    _module: CudaModule,
    ctx_kv: CudaFunction,
    block_qk_rope: CudaFunction,
    conv: CudaFunction,
    attn_part: CudaFunction,
    attn_merge: CudaFunction,
    top16_part: CudaFunction,
    top16_merge: CudaFunction,
}

impl DFlash2Kernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = ctx.load_module(DFLASH2_CUBIN)?;
        Ok(Self {
            ctx_kv: module.function("m3tis_df_ctx_kv")?,
            block_qk_rope: module.function("m3tis_df_block_qk_rope")?,
            conv: module.function("m3tis_df_conv")?,
            attn_part: module.function("m3tis_df_attn_part")?,
            attn_merge: module.function("m3tis_df_attn_merge")?,
            top16_part: module.function("m3tis_df_top16_part_f32")?,
            top16_merge: module.function("m3tis_df_top16_merge_f32")?,
            _module: module,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ctx_kv(
        &self,
        stream: &Stream,
        mut k_in: CuDevicePtr,
        mut v_in: CuDevicePtr,
        mut kn: CuDevicePtr,
        mut ring_k: CuDevicePtr,
        mut ring_v: CuDevicePtr,
        mut pos0: i64,
        mut eps: f32,
        mut theta: f32,
        n_rows: usize,
    ) -> Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let mut args: [*mut c_void; 8] = [
            &mut k_in as *mut _ as *mut c_void,
            &mut v_in as *mut _ as *mut c_void,
            &mut kn as *mut _ as *mut c_void,
            &mut ring_k as *mut _ as *mut c_void,
            &mut ring_v as *mut _ as *mut c_void,
            &mut pos0 as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut theta as *mut _ as *mut c_void,
        ];
        unsafe {
            self.ctx_kv
                .launch((n_rows as u32, 8, 1), (64, 1, 1), 0, stream, &mut args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn block_qk_rope(
        &self,
        stream: &Stream,
        mut q_in: CuDevicePtr,
        mut k_in: CuDevicePtr,
        mut qn: CuDevicePtr,
        mut kn: CuDevicePtr,
        mut pos0: i64,
        mut eps: f32,
        mut theta: f32,
        mut q_out: CuDevicePtr,
        mut k_out: CuDevicePtr,
        rows: usize,
    ) -> Result<()> {
        let mut args: [*mut c_void; 9] = [
            &mut q_in as *mut _ as *mut c_void,
            &mut k_in as *mut _ as *mut c_void,
            &mut qn as *mut _ as *mut c_void,
            &mut kn as *mut _ as *mut c_void,
            &mut pos0 as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut theta as *mut _ as *mut c_void,
            &mut q_out as *mut _ as *mut c_void,
            &mut k_out as *mut _ as *mut c_void,
        ];
        unsafe {
            self.block_qk_rope
                .launch((rows as u32, 40, 1), (64, 1, 1), 0, stream, &mut args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv(
        &self,
        stream: &Stream,
        mut z: CuDevicePtr,
        mut dyn_w: CuDevicePtr,
        mut which: i32,
        mut base: CuDevicePtr,
        mut out: CuDevicePtr,
        add: bool,
        rows: usize,
    ) -> Result<()> {
        let mut add_i: i32 = if add { 1 } else { 0 };
        let mut args: [*mut c_void; 6] = [
            &mut z as *mut _ as *mut c_void,
            &mut dyn_w as *mut _ as *mut c_void,
            &mut which as *mut _ as *mut c_void,
            &mut base as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut add_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.conv
                .launch((rows as u32, 4, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn(
        &self,
        stream: &Stream,
        mut q: CuDevicePtr,
        mut kb: CuDevicePtr,
        mut vb: CuDevicePtr,
        mut ring_k: CuDevicePtr,
        mut ring_v: CuDevicePtr,
        mut start_pos: i64,
        mut from_pos: i64,
        mut scale: f32,
        mut pm: CuDevicePtr,
        mut pl: CuDevicePtr,
        mut pacc: CuDevicePtr,
        mut out: CuDevicePtr,
    ) -> Result<()> {
        {
            let mut args: [*mut c_void; 11] = [
                &mut q as *mut _ as *mut c_void,
                &mut kb as *mut _ as *mut c_void,
                &mut vb as *mut _ as *mut c_void,
                &mut ring_k as *mut _ as *mut c_void,
                &mut ring_v as *mut _ as *mut c_void,
                &mut start_pos as *mut _ as *mut c_void,
                &mut from_pos as *mut _ as *mut c_void,
                &mut scale as *mut _ as *mut c_void,
                &mut pm as *mut _ as *mut c_void,
                &mut pl as *mut _ as *mut c_void,
                &mut pacc as *mut _ as *mut c_void,
            ];
            unsafe {
                self.attn_part
                    .launch((8, 16, 1), (256, 1, 1), 0, stream, &mut args)?;
            }
        }
        {
            let mut args: [*mut c_void; 4] = [
                &mut pm as *mut _ as *mut c_void,
                &mut pl as *mut _ as *mut c_void,
                &mut pacc as *mut _ as *mut c_void,
                &mut out as *mut _ as *mut c_void,
            ];
            unsafe {
                self.attn_merge
                    .launch((8, 32, 1), (128, 1, 1), 0, stream, &mut args)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn top16_f32(
        &self,
        stream: &Stream,
        mut logits: CuDevicePtr,
        cols: usize,
        rows: usize,
        mut pv: CuDevicePtr,
        mut pi: CuDevicePtr,
        mut out_ids: CuDevicePtr,
        mut out_vals: CuDevicePtr,
    ) -> Result<()> {
        let chunks = (cols + 4095) / 4096;
        assert!(chunks <= 64);
        let mut cols_i = cols as i32;
        {
            let mut args: [*mut c_void; 4] = [
                &mut logits as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut pv as *mut _ as *mut c_void,
                &mut pi as *mut _ as *mut c_void,
            ];
            unsafe {
                self.top16_part.launch(
                    (rows as u32, chunks as u32, 1),
                    (256, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }
        let mut ch_i = chunks as i32;
        {
            let mut args: [*mut c_void; 5] = [
                &mut pv as *mut _ as *mut c_void,
                &mut pi as *mut _ as *mut c_void,
                &mut ch_i as *mut _ as *mut c_void,
                &mut out_ids as *mut _ as *mut c_void,
                &mut out_vals as *mut _ as *mut c_void,
            ];
            unsafe {
                self.top16_merge
                    .launch((rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args)
            }
        }
    }
}
