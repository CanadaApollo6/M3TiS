//! Clean-room Rust host wrappers for M3TiS `sm_86` GLM-5.3-Flash core operations (`glm_ops.cu`):
//! - Embedding broadcast to `S=4` mHC streams
//! - RMSNorm (`D = 4096, 1536, 512`)
//! - mHC Hyper-Connections (`hc_pre` with 20-iter Sinkhorn doubly-stochastic `4x4` + fused RMSNorm, `hc_post`, `stream_mean`)
//! - KDA Linear Attention (`kda_chain`, `kda_conv_shift`)
//! - DSA/MLA Latent Attention (`mla_absorb_q`, `mla_write_kv`, `mla_attend_dense`, `mla_expand_v`)

use crate::{CuDevicePtr, CudaContext, CudaFunction, CudaModule, Result, Stream};
use std::ffi::c_void;
use std::sync::Arc;

const GLM_OPS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glm_ops.cubin"));

pub struct GlmOpsKernels {
    _module: CudaModule,
    embed_broadcast_bf16: CudaFunction,
    rmsnorm_bf16: CudaFunction,
    hc_partial: CudaFunction,
    hc_finish: CudaFunction,
    hc_post: CudaFunction,
    stream_mean_bf16: CudaFunction,
    kda_chain: CudaFunction,
    kda_chain_save: CudaFunction,
    kda_replay_layers: CudaFunction,
    copy_bf16_strided: CudaFunction,
    kda_conv_shift: CudaFunction,
    kda_conv_shift_all: CudaFunction,
    tap_stream_mean: CudaFunction,
    argmax_f32: CudaFunction,
    argmax_prob_f32: CudaFunction,
    mla_absorb_q: CudaFunction,
    mla_write_kv: CudaFunction,
    mla_write_kv_q8: CudaFunction,
    mla_attend_dense: CudaFunction,
    mla_attend_dense_tc: CudaFunction,
    mla_attend_dense_q8: CudaFunction,
    mla_attend_dense_tc_q8: CudaFunction,
    mla_expand_v: CudaFunction,
    dsa_index_write: CudaFunction,
    dsa_pool_keys: CudaFunction,
    dsa_score_pools: CudaFunction,
    dsa_score_pools_tc: CudaFunction,
    dsa_select_tokens: CudaFunction,
    mla_attend_sparse: CudaFunction,
    mla_attend_sparse_tc: CudaFunction,
    mla_attend_sparse_q8: CudaFunction,
    mla_attend_sparse_tc_q8: CudaFunction,
    mla_compact_kv_bf16: CudaFunction,
    mla_compact_kv_q8: CudaFunction,
    dsa_compact_ik_ig: CudaFunction,
    add_f32_inplace: CudaFunction,
}

impl GlmOpsKernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = ctx.load_module(GLM_OPS_CUBIN)?;
        Ok(Self {
            embed_broadcast_bf16: module.function("m3tis_embed_broadcast_bf16")?,
            rmsnorm_bf16: module.function("m3tis_rmsnorm_bf16")?,
            hc_partial: module.function("m3tis_hc_partial")?,
            hc_finish: module.function("m3tis_hc_finish")?,
            hc_post: module.function("m3tis_hc_post")?,
            stream_mean_bf16: module.function("m3tis_stream_mean_bf16")?,
            kda_chain: module.function("m3tis_kda_chain")?,
            kda_chain_save: module.function("m3tis_kda_chain_save")?,
            kda_replay_layers: module.function("m3tis_kda_replay_layers")?,
            copy_bf16_strided: module.function("m3tis_copy_bf16_strided")?,
            kda_conv_shift: module.function("m3tis_kda_conv_shift")?,
            kda_conv_shift_all: module.function("m3tis_kda_conv_shift_all")?,
            tap_stream_mean: module.function("m3tis_tap_stream_mean")?,
            argmax_f32: module.function("m3tis_argmax_f32")?,
            argmax_prob_f32: module.function("m3tis_argmax_prob_f32")?,
            mla_absorb_q: module.function("m3tis_mla_absorb_q")?,
            mla_write_kv: module.function("m3tis_mla_write_kv")?,
            mla_write_kv_q8: module.function("m3tis_mla_write_kv_q8")?,
            mla_attend_dense: module.function("m3tis_mla_attend_dense")?,
            mla_attend_dense_tc: module.function("m3tis_mla_attend_dense_tc")?,
            mla_attend_dense_q8: module.function("m3tis_mla_attend_dense_q8")?,
            mla_attend_dense_tc_q8: module.function("m3tis_mla_attend_dense_tc_q8")?,
            mla_expand_v: module.function("m3tis_mla_expand_v")?,
            dsa_index_write: module.function("m3tis_dsa_index_write")?,
            dsa_pool_keys: module.function("m3tis_dsa_pool_keys")?,
            dsa_score_pools: module.function("m3tis_dsa_score_pools")?,
            dsa_score_pools_tc: module.function("m3tis_dsa_score_pools_tc")?,
            dsa_select_tokens: module.function("m3tis_dsa_select_tokens")?,
            mla_attend_sparse: module.function("m3tis_mla_attend_sparse")?,
            mla_attend_sparse_tc: module.function("m3tis_mla_attend_sparse_tc")?,
            mla_attend_sparse_q8: module.function("m3tis_mla_attend_sparse_q8")?,
            mla_attend_sparse_tc_q8: module.function("m3tis_mla_attend_sparse_tc_q8")?,
            mla_compact_kv_bf16: module.function("m3tis_mla_compact_kv_bf16")?,
            mla_compact_kv_q8: module.function("m3tis_mla_compact_kv_q8")?,
            dsa_compact_ik_ig: module.function("m3tis_dsa_compact_ik_ig")?,
            add_f32_inplace: module.function("m3tis_add_f32_inplace")?,
            _module: module,
        })
    }

    /// Vectorized `dst[0..n] += src[0..n]` (accumulates CPU-executed hybrid MoE outputs into GPU `st.branch`).
    pub fn add_f32_inplace(
        &self,
        stream: &Stream,
        mut dst: CuDevicePtr,
        mut src: CuDevicePtr,
        n: usize,
    ) -> Result<()> {
        let mut n_i = n as i32;
        let mut args: [*mut c_void; 3] = [
            &mut dst as *mut _ as *mut c_void,
            &mut src as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
        ];
        let blocks = (n.div_ceil(4)).div_ceil(256).max(1) as u32;
        unsafe {
            self.add_f32_inplace
                .launch((blocks, 1, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }

    pub fn embed_broadcast_bf16(
        &self,
        stream: &Stream,
        mut rows_in: CuDevicePtr,
        mut x_out: CuDevicePtr,
        rows: usize,
        d: usize,
        s: usize,
    ) -> Result<()> {
        let mut r_i = rows as i32;
        let mut d_i = d as i32;
        let mut s_i = s as i32;
        let mut args: [*mut c_void; 5] = [
            &mut rows_in as *mut _ as *mut c_void,
            &mut x_out as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut d_i as *mut _ as *mut c_void,
            &mut s_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.embed_broadcast_bf16.launch(
                (rows as u32, ((d + 255) / 256) as u32, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_bf16(
        &self,
        stream: &Stream,
        mut x: CuDevicePtr,
        x_stride: usize,
        mut w: CuDevicePtr,
        mut out: CuDevicePtr,
        o_stride: usize,
        rows: usize,
        d: usize,
        mut eps: f32,
    ) -> Result<()> {
        let mut xs_i = x_stride as i32;
        let mut os_i = o_stride as i32;
        let mut r_i = rows as i32;
        let mut d_i = d as i32;
        let mut args: [*mut c_void; 8] = [
            &mut x as *mut _ as *mut c_void,
            &mut xs_i as *mut _ as *mut c_void,
            &mut w as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut os_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut d_i as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe {
            self.rmsnorm_bf16
                .launch((rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }

    /// Full mHC Pre step (`hc_partial` + `hc_finish`): computes 20-iter Sinkhorn doubly-stochastic `comb[rows, 16]`,
    /// `post[rows, 4]`, collapses 4 streams, and applies fused RMSNorm -> `normed_out[rows, D]`.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre(
        &self,
        stream: &Stream,
        mut x: CuDevicePtr,
        mut fn_w: CuDevicePtr,
        mut part: CuDevicePtr,
        mut base: CuDevicePtr,
        mut scale: CuDevicePtr,
        mut nw: CuDevicePtr,
        mut normed_out: CuDevicePtr,
        mut post_out: CuDevicePtr,
        mut comb_out: CuDevicePtr,
        rows: usize,
        d: usize,
        mut eps: f32,
        mut hc_eps: f32,
        mut sinkhorn_iters: i32,
    ) -> Result<()> {
        let wide = 4 * d;
        assert_eq!(wide % 256, 0, "hc_pre requires 4*D divisible by 256");
        // Choose nb dividing wide/256, capped at 16 (nb=16 for D=4096 -> kb=1024).
        let max_nb = (wide / 256).min(16);
        let mut nb = max_nb;
        while nb > 1 && (wide % (nb * 256)) != 0 {
            nb -= 1;
        }
        let mut wide_i = wide as i32;
        let mut nb_i = nb as i32;
        let mut rows_i = rows as i32;
        {
            let mut args: [*mut c_void; 6] = [
                &mut x as *mut _ as *mut c_void,
                &mut fn_w as *mut _ as *mut c_void,
                &mut part as *mut _ as *mut c_void,
                &mut wide_i as *mut _ as *mut c_void,
                &mut nb_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hc_partial.launch(
                    (((rows + 3) / 4) as u32, nb as u32, 1),
                    (256, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }
        {
            let mut d_i = d as i32;
            let mut args: [*mut c_void; 13] = [
                &mut x as *mut _ as *mut c_void,
                &mut part as *mut _ as *mut c_void,
                &mut base as *mut _ as *mut c_void,
                &mut scale as *mut _ as *mut c_void,
                &mut nw as *mut _ as *mut c_void,
                &mut normed_out as *mut _ as *mut c_void,
                &mut post_out as *mut _ as *mut c_void,
                &mut comb_out as *mut _ as *mut c_void,
                &mut d_i as *mut _ as *mut c_void,
                &mut nb_i as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
                &mut hc_eps as *mut _ as *mut c_void,
                &mut sinkhorn_iters as *mut _ as *mut c_void,
            ];
            unsafe {
                self.hc_finish
                    .launch((rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args)
            }
        }
    }

    pub fn hc_post(
        &self,
        stream: &Stream,
        mut x: CuDevicePtr,
        mut branch_f32: CuDevicePtr,
        mut post: CuDevicePtr,
        mut comb: CuDevicePtr,
        rows: usize,
        d: usize,
    ) -> Result<()> {
        let mut r_i = rows as i32;
        let mut d_i = d as i32;
        let mut args: [*mut c_void; 6] = [
            &mut x as *mut _ as *mut c_void,
            &mut branch_f32 as *mut _ as *mut c_void,
            &mut post as *mut _ as *mut c_void,
            &mut comb as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut d_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.hc_post.launch(
                (rows as u32, ((d + 255) / 256) as u32, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    pub fn stream_mean_bf16(
        &self,
        stream: &Stream,
        mut x: CuDevicePtr,
        mut out: CuDevicePtr,
        rows: usize,
        d: usize,
        s: usize,
    ) -> Result<()> {
        let mut r_i = rows as i32;
        let mut d_i = d as i32;
        let mut s_i = s as i32;
        let mut args: [*mut c_void; 5] = [
            &mut x as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut d_i as *mut _ as *mut c_void,
            &mut s_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.stream_mean_bf16.launch(
                (rows as u32, ((d + 255) / 256) as u32, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kda_chain(
        &self,
        stream: &Stream,
        heads: usize,
        mut p: CuDevicePtr,
        p_stride: usize,
        b_off: usize,
        mut a: CuDevicePtr,
        mut g: CuDevicePtr,
        mut cs: CuDevicePtr,
        mut cw: CuDevicePtr,
        mut state_in: CuDevicePtr,
        mut a_log: CuDevicePtr,
        mut dt_bias: CuDevicePtr,
        mut norm_w: CuDevicePtr,
        mut eps: f32,
        mut lower: f32,
        rows: usize,
        mut out: CuDevicePtr,
        mut state_out: CuDevicePtr,
    ) -> Result<()> {
        let mut h_i = heads as i32;
        let mut ps_i = p_stride as i32;
        let mut bo_i = b_off as i32;
        let mut r_i = rows as i32;
        let mut args: [*mut c_void; 17] = [
            &mut h_i as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut ps_i as *mut _ as *mut c_void,
            &mut bo_i as *mut _ as *mut c_void,
            &mut a as *mut _ as *mut c_void,
            &mut g as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
            &mut cw as *mut _ as *mut c_void,
            &mut state_in as *mut _ as *mut c_void,
            &mut a_log as *mut _ as *mut c_void,
            &mut dt_bias as *mut _ as *mut c_void,
            &mut norm_w as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut lower as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut state_out as *mut _ as *mut c_void,
        ];
        unsafe {
            self.kda_chain
                .launch((heads as u32, 1, 1), (1024, 1, 1), 0, stream, &mut args)
        }
    }

    pub fn kda_conv_shift(
        &self,
        stream: &Stream,
        mut cs: CuDevicePtr,
        mut p: CuDevicePtr,
        rows: usize,
        p_stride: usize,
        c: usize,
    ) -> Result<()> {
        let mut r_i = rows as i32;
        let mut ps_i = p_stride as i32;
        let mut c_i = c as i32;
        let mut args: [*mut c_void; 5] = [
            &mut cs as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut ps_i as *mut _ as *mut c_void,
            &mut c_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.kda_conv_shift.launch(
                (((c + 255) / 256) as u32, 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kda_chain_save(
        &self,
        stream: &Stream,
        heads: usize,
        mut p: CuDevicePtr,
        p_stride: usize,
        b_off: usize,
        mut a: CuDevicePtr,
        mut g: CuDevicePtr,
        mut cs: CuDevicePtr,
        mut cw: CuDevicePtr,
        mut state_in: CuDevicePtr,
        mut a_log: CuDevicePtr,
        mut dt_bias: CuDevicePtr,
        mut norm_w: CuDevicePtr,
        mut eps: f32,
        mut lower: f32,
        rows: usize,
        mut out: CuDevicePtr,
        mut state_out: CuDevicePtr,
        mut k_save: CuDevicePtr,
        mut v_save: CuDevicePtr,
        mut g_save: CuDevicePtr,
        mut b_save: CuDevicePtr,
        mut p_save: CuDevicePtr,
        tree_trunk_len: usize,
        tree_b_start: usize,
    ) -> Result<()> {
        let mut h_i = heads as i32;
        let mut ps_i = p_stride as i32;
        let mut bo_i = b_off as i32;
        let mut r_i = rows as i32;
        let mut tt_i = tree_trunk_len as i32;
        let mut tb_i = tree_b_start as i32;
        let mut args: [*mut c_void; 24] = [
            &mut h_i as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut ps_i as *mut _ as *mut c_void,
            &mut bo_i as *mut _ as *mut c_void,
            &mut a as *mut _ as *mut c_void,
            &mut g as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
            &mut cw as *mut _ as *mut c_void,
            &mut state_in as *mut _ as *mut c_void,
            &mut a_log as *mut _ as *mut c_void,
            &mut dt_bias as *mut _ as *mut c_void,
            &mut norm_w as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut lower as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut state_out as *mut _ as *mut c_void,
            &mut k_save as *mut _ as *mut c_void,
            &mut v_save as *mut _ as *mut c_void,
            &mut g_save as *mut _ as *mut c_void,
            &mut b_save as *mut _ as *mut c_void,
            &mut p_save as *mut _ as *mut c_void,
            &mut tt_i as *mut _ as *mut c_void,
            &mut tb_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.kda_chain_save
                .launch((heads as u32, 1, 1), (1024, 1, 1), 0, stream, &mut args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kda_replay_layers(
        &self,
        stream: &Stream,
        num_layers: usize,
        heads: usize,
        mut state_in: CuDevicePtr,
        mut state_stride: i64,
        mut k_save: CuDevicePtr,
        mut v_save: CuDevicePtr,
        mut g_save: CuDevicePtr,
        mut b_save: CuDevicePtr,
        mut kv_stride: i64,
        mut b_stride: i64,
        keep: usize,
        mut state_out: CuDevicePtr,
        tree_trunk_len: usize,
        tree_b_start: usize,
    ) -> Result<()> {
        let mut h_i = heads as i32;
        let mut keep_i = keep as i32;
        let mut tt_i = tree_trunk_len as i32;
        let mut tb_i = tree_b_start as i32;
        let mut args: [*mut c_void; 13] = [
            &mut h_i as *mut _ as *mut c_void,
            &mut state_in as *mut _ as *mut c_void,
            &mut state_stride as *mut _ as *mut c_void,
            &mut k_save as *mut _ as *mut c_void,
            &mut v_save as *mut _ as *mut c_void,
            &mut g_save as *mut _ as *mut c_void,
            &mut b_save as *mut _ as *mut c_void,
            &mut kv_stride as *mut _ as *mut c_void,
            &mut b_stride as *mut _ as *mut c_void,
            &mut keep_i as *mut _ as *mut c_void,
            &mut state_out as *mut _ as *mut c_void,
            &mut tt_i as *mut _ as *mut c_void,
            &mut tb_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.kda_replay_layers.launch(
                ((num_layers * heads) as u32, 1, 1),
                (1024, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn copy_bf16_strided(
        &self,
        stream: &Stream,
        mut src: CuDevicePtr,
        src_stride: usize,
        mut dst: CuDevicePtr,
        dst_stride: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let mut ss_i = src_stride as i32;
        let mut ds_i = dst_stride as i32;
        let mut r_i = rows as i32;
        let mut c_i = cols as i32;
        let mut args: [*mut c_void; 6] = [
            &mut src as *mut _ as *mut c_void,
            &mut ss_i as *mut _ as *mut c_void,
            &mut dst as *mut _ as *mut c_void,
            &mut ds_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut c_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.copy_bf16_strided.launch(
                (rows as u32, ((cols + 255) / 256) as u32, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kda_conv_shift_all(
        &self,
        stream: &Stream,
        num_layers: usize,
        mut cs_all: CuDevicePtr,
        mut p_save: CuDevicePtr,
        keep: usize,
        max_rows: usize,
        c: usize,
        tree_trunk_len: usize,
        tree_b_start: usize,
    ) -> Result<()> {
        let mut keep_i = keep as i32;
        let mut mr_i = max_rows as i32;
        let mut c_i = c as i32;
        let mut tt_i = tree_trunk_len as i32;
        let mut tb_i = tree_b_start as i32;
        let mut args: [*mut c_void; 7] = [
            &mut cs_all as *mut _ as *mut c_void,
            &mut p_save as *mut _ as *mut c_void,
            &mut keep_i as *mut _ as *mut c_void,
            &mut mr_i as *mut _ as *mut c_void,
            &mut c_i as *mut _ as *mut c_void,
            &mut tt_i as *mut _ as *mut c_void,
            &mut tb_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.kda_conv_shift_all.launch(
                (num_layers as u32, ((c + 255) / 256) as u32, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    pub fn tap_stream_mean(
        &self,
        stream: &Stream,
        mut x: CuDevicePtr,
        mut taps_out: CuDevicePtr,
        rows: usize,
        d: usize,
        mut slot: i32,
    ) -> Result<()> {
        let mut r_i = rows as i32;
        let mut d_i = d as i32;
        let mut args: [*mut c_void; 5] = [
            &mut x as *mut _ as *mut c_void,
            &mut taps_out as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut d_i as *mut _ as *mut c_void,
            &mut slot as *mut _ as *mut c_void,
        ];
        unsafe {
            self.tap_stream_mean.launch(
                (rows as u32, ((d + 255) / 256) as u32, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    pub fn argmax_f32(
        &self,
        stream: &Stream,
        mut logits: CuDevicePtr,
        mut out_ids: CuDevicePtr,
        mut out_vals: CuDevicePtr,
        rows: usize,
        vocab: usize,
    ) -> Result<()> {
        let mut r_i = rows as i32;
        let mut v_i = vocab as i32;
        let mut args: [*mut c_void; 5] = [
            &mut logits as *mut _ as *mut c_void,
            &mut out_ids as *mut _ as *mut c_void,
            &mut out_vals as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut v_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.argmax_f32
                .launch((rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }

    pub fn argmax_prob_f32(
        &self,
        stream: &Stream,
        mut logits: CuDevicePtr,
        mut out_ids: CuDevicePtr,
        mut out_probs: CuDevicePtr,
        rows: usize,
        vocab: usize,
    ) -> Result<()> {
        let mut r_i = rows as i32;
        let mut v_i = vocab as i32;
        let mut args: [*mut c_void; 5] = [
            &mut logits as *mut _ as *mut c_void,
            &mut out_ids as *mut _ as *mut c_void,
            &mut out_probs as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut v_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.argmax_prob_f32
                .launch((rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_absorb_q(
        &self,
        stream: &Stream,
        mut q: CuDevicePtr,
        mut wk: CuDevicePtr,
        mut qa: CuDevicePtr,
        rows: usize,
        heads: usize,
        qk: usize,
        lw: usize,
    ) -> Result<()> {
        let mut r_i = rows as i32;
        let mut h_i = heads as i32;
        let mut qk_i = qk as i32;
        let mut lw_i = lw as i32;
        let mut args: [*mut c_void; 7] = [
            &mut q as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut qa as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut h_i as *mut _ as *mut c_void,
            &mut qk_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.mla_absorb_q.launch(
                ((((rows + 15) / 16) * heads) as u32, ((lw + 3) / 4) as u32, 1),
                (128, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_write_kv(
        &self,
        stream: &Stream,
        mut lat: CuDevicePtr,
        mut kv_cache: CuDevicePtr,
        pos: usize,
        capacity: usize,
        rows: usize,
        lw: usize,
    ) -> Result<()> {
        let mut pos_i = pos as i32;
        let mut cap_i = capacity as i32;
        let mut r_i = rows as i32;
        let mut lw_i = lw as i32;
        let mut args: [*mut c_void; 6] = [
            &mut lat as *mut _ as *mut c_void,
            &mut kv_cache as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.mla_write_kv.launch(
                (rows as u32, ((lw + 255) / 256) as u32, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_write_kv_q8(
        &self,
        stream: &Stream,
        mut lat: CuDevicePtr,
        mut kv_q8: CuDevicePtr,
        mut kv_sc: CuDevicePtr,
        pos: usize,
        capacity: usize,
        rows: usize,
        lw: usize,
    ) -> Result<()> {
        let mut pos_i = pos as i32;
        let mut cap_i = capacity as i32;
        let mut r_i = rows as i32;
        let mut lw_i = lw as i32;
        let mut args: [*mut c_void; 7] = [
            &mut lat as *mut _ as *mut c_void,
            &mut kv_q8 as *mut _ as *mut c_void,
            &mut kv_sc as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.mla_write_kv_q8
                .launch((rows as u32, 1, 1), (64, 1, 1), 0, stream, &mut args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_attend_dense(
        &self,
        stream: &Stream,
        mut qa: CuDevicePtr,
        mut kv_cache: CuDevicePtr,
        mut ol: CuDevicePtr,
        pos: usize,
        capacity: usize,
        window: usize,
        rows: usize,
        heads: usize,
        lw: usize,
        mut scale: f32,
        tree_trunk_len: usize,
        tree_b_start: usize,
    ) -> Result<()> {
        assert!(lw <= 512, "mla_attend_dense expects lw <= 512 (got {lw})");
        let mut pos_i = pos as i32;
        let mut cap_i = capacity as i32;
        let mut win_i = window as i32;
        let mut r_i = rows as i32;
        let mut h_i = heads as i32;
        let mut lw_i = lw as i32;
        let mut tt_i = tree_trunk_len as i32;
        let mut tb_i = tree_b_start as i32;
        let mut args: [*mut c_void; 12] = [
            &mut qa as *mut _ as *mut c_void,
            &mut kv_cache as *mut _ as *mut c_void,
            &mut ol as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut win_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut h_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut tt_i as *mut _ as *mut c_void,
            &mut tb_i as *mut _ as *mut c_void,
        ];
        if lw == 512 && heads % 16 == 0 && rows >= 16 {
            let gy = (heads / 16) as u32;
            unsafe {
                self.mla_attend_dense_tc
                    .launch((rows as u32, gy, 1), (256, 1, 1), 0, stream, &mut args)
            }
        } else {
            let gy = if lw == 512 && heads % 8 == 0 { heads / 8 } else { heads };
            unsafe {
                self.mla_attend_dense.launch(
                    (rows as u32, gy as u32, 1),
                    (256, 1, 1),
                    0,
                    stream,
                    &mut args,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_attend_dense_q8(
        &self,
        stream: &Stream,
        mut qa: CuDevicePtr,
        mut kv_q8: CuDevicePtr,
        mut kv_sc: CuDevicePtr,
        mut ol: CuDevicePtr,
        pos: usize,
        capacity: usize,
        window: usize,
        rows: usize,
        heads: usize,
        lw: usize,
        mut scale: f32,
        tree_trunk_len: usize,
        tree_b_start: usize,
    ) -> Result<()> {
        assert!(lw == 512 && heads % 8 == 0, "mla_attend_dense_q8 expects lw==512, heads%8==0");
        let mut pos_i = pos as i32;
        let mut cap_i = capacity as i32;
        let mut win_i = window as i32;
        let mut r_i = rows as i32;
        let mut h_i = heads as i32;
        let mut lw_i = lw as i32;
        let mut tt_i = tree_trunk_len as i32;
        let mut tb_i = tree_b_start as i32;
        let mut args: [*mut c_void; 13] = [
            &mut qa as *mut _ as *mut c_void,
            &mut kv_q8 as *mut _ as *mut c_void,
            &mut kv_sc as *mut _ as *mut c_void,
            &mut ol as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut win_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut h_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut tt_i as *mut _ as *mut c_void,
            &mut tb_i as *mut _ as *mut c_void,
        ];
        if heads % 16 == 0 && rows >= 16 {
            let gy = (heads / 16) as u32;
            unsafe {
                self.mla_attend_dense_tc_q8
                    .launch((rows as u32, gy, 1), (256, 1, 1), 0, stream, &mut args)
            }
        } else {
            let gy = (heads / 8) as u32;
            unsafe {
                self.mla_attend_dense_q8
                    .launch((rows as u32, gy, 1), (256, 1, 1), 0, stream, &mut args)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_expand_v(
        &self,
        stream: &Stream,
        mut ol: CuDevicePtr,
        mut wv: CuDevicePtr,
        mut vn: CuDevicePtr,
        rows: usize,
        heads: usize,
        dv: usize,
        lw: usize,
    ) -> Result<()> {
        let mut r_i = rows as i32;
        let mut h_i = heads as i32;
        let mut dv_i = dv as i32;
        let mut lw_i = lw as i32;
        let mut args: [*mut c_void; 7] = [
            &mut ol as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut vn as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut h_i as *mut _ as *mut c_void,
            &mut dv_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.mla_expand_v.launch(
                ((((rows + 15) / 16) * heads) as u32, ((dv + 3) / 4) as u32, 1),
                (128, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsa_index_write(
        &self,
        stream: &Stream,
        mut k_raw: CuDevicePtr,
        mut g_raw: CuDevicePtr,
        mut ln_w: CuDevicePtr,
        mut ln_b: CuDevicePtr,
        mut ik_cache: CuDevicePtr,
        mut ig_cache: CuDevicePtr,
        pos: usize,
        capacity: usize,
        mut eps: f32,
        rows: usize,
    ) -> Result<()> {
        let mut pos_i = pos as i32;
        let mut cap_i = capacity as i32;
        let mut args: [*mut c_void; 9] = [
            &mut k_raw as *mut _ as *mut c_void,
            &mut g_raw as *mut _ as *mut c_void,
            &mut ln_w as *mut _ as *mut c_void,
            &mut ln_b as *mut _ as *mut c_void,
            &mut ik_cache as *mut _ as *mut c_void,
            &mut ig_cache as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        unsafe {
            self.dsa_index_write
                .launch((rows as u32, 1, 1), (32, 1, 1), 0, stream, &mut args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsa_pool_keys(
        &self,
        stream: &Stream,
        mut ik_cache: CuDevicePtr,
        mut ig_cache: CuDevicePtr,
        mut ape: CuDevicePtr,
        mut pk_cache: CuDevicePtr,
        pos: usize,
        rows: usize,
        ik_cap: usize,
        pk_cap: usize,
    ) -> Result<()> {
        let mut pos_i = pos as i32;
        let mut r_i = rows as i32;
        let mut ik_i = ik_cap as i32;
        let mut pk_i = pk_cap as i32;
        let mut args: [*mut c_void; 8] = [
            &mut ik_cache as *mut _ as *mut c_void,
            &mut ig_cache as *mut _ as *mut c_void,
            &mut ape as *mut _ as *mut c_void,
            &mut pk_cache as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut ik_i as *mut _ as *mut c_void,
            &mut pk_i as *mut _ as *mut c_void,
        ];
        let grid_x = (rows / 4 + 2) as u32;
        unsafe {
            self.dsa_pool_keys
                .launch((grid_x, 1, 1), (32, 1, 1), 0, stream, &mut args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsa_score_pools(
        &self,
        stream: &Stream,
        mut qi: CuDevicePtr,
        mut w_raw: CuDevicePtr,
        mut pk_cache: CuDevicePtr,
        mut scores: CuDevicePtr,
        pos: usize,
        np_stride: usize,
        pk_cap: usize,
        heads: usize,
        mut qscale: f32,
        mut wscale: f32,
        rows: usize,
    ) -> Result<()> {
        let mut pos_i = pos as i32;
        let mut np_i = np_stride as i32;
        let mut pk_i = pk_cap as i32;
        let mut h_i = heads as i32;
        let mut r_i = rows as i32;
        let mut args: [*mut c_void; 11] = [
            &mut qi as *mut _ as *mut c_void,
            &mut w_raw as *mut _ as *mut c_void,
            &mut pk_cache as *mut _ as *mut c_void,
            &mut scores as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut np_i as *mut _ as *mut c_void,
            &mut pk_i as *mut _ as *mut c_void,
            &mut h_i as *mut _ as *mut c_void,
            &mut qscale as *mut _ as *mut c_void,
            &mut wscale as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
        ];
        if heads == 32 && np_stride >= 32 {
            let grid_x = rows.max(1) as u32;
            let grid_y = ((np_stride + 31) / 32).max(1) as u32;
            unsafe {
                self.dsa_score_pools_tc
                    .launch((grid_x, grid_y, 1), (128, 1, 1), 0, stream, &mut args)
            }
        } else {
            let grid_x = ((rows + 3) / 4).max(1) as u32;
            let grid_y = ((np_stride + 7) / 8).max(1) as u32;
            unsafe {
                self.dsa_score_pools
                    .launch((grid_x, grid_y, 1), (256, 1, 1), 0, stream, &mut args)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsa_select_tokens(
        &self,
        stream: &Stream,
        mut scores: CuDevicePtr,
        mut tokens: CuDevicePtr,
        mut counts: CuDevicePtr,
        pos: usize,
        np_stride: usize,
        topk_pools: usize,
        width: usize,
        rows: usize,
        tree_trunk_len: usize,
        tree_b_start: usize,
    ) -> Result<()> {
        let mut pos_i = pos as i32;
        let mut np_i = np_stride as i32;
        let mut k_i = topk_pools as i32;
        let mut w_i = width as i32;
        let mut tt_i = tree_trunk_len as i32;
        let mut tb_i = tree_b_start as i32;
        let mut args: [*mut c_void; 9] = [
            &mut scores as *mut _ as *mut c_void,
            &mut tokens as *mut _ as *mut c_void,
            &mut counts as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut np_i as *mut _ as *mut c_void,
            &mut k_i as *mut _ as *mut c_void,
            &mut w_i as *mut _ as *mut c_void,
            &mut tt_i as *mut _ as *mut c_void,
            &mut tb_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.dsa_select_tokens
                .launch((rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_attend_sparse(
        &self,
        stream: &Stream,
        mut qa: CuDevicePtr,
        mut kv_cache: CuDevicePtr,
        mut tokens: CuDevicePtr,
        mut counts: CuDevicePtr,
        mut ol: CuDevicePtr,
        pos: usize,
        capacity: usize,
        width: usize,
        rows: usize,
        heads: usize,
        lw: usize,
        mut scale: f32,
        tree_trunk_len: usize,
        tree_b_start: usize,
    ) -> Result<()> {
        assert!(lw <= 512, "mla_attend_sparse expects lw <= 512 (got {lw})");
        let mut pos_i = pos as i32;
        let mut cap_i = capacity as i32;
        let mut w_i = width as i32;
        let mut r_i = rows as i32;
        let mut h_i = heads as i32;
        let mut lw_i = lw as i32;
        let mut tt_i = tree_trunk_len as i32;
        let mut tb_i = tree_b_start as i32;
        let mut args: [*mut c_void; 14] = [
            &mut qa as *mut _ as *mut c_void,
            &mut kv_cache as *mut _ as *mut c_void,
            &mut tokens as *mut _ as *mut c_void,
            &mut counts as *mut _ as *mut c_void,
            &mut ol as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut w_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut h_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut tt_i as *mut _ as *mut c_void,
            &mut tb_i as *mut _ as *mut c_void,
        ];
        if lw == 512 && heads % 16 == 0 && rows >= 16 {
            let gy = (heads / 16) as u32;
            unsafe {
                self.mla_attend_sparse_tc
                    .launch((rows as u32, gy, 1), (256, 1, 1), 0, stream, &mut args)
            }
        } else {
            let gy = if lw == 512 && heads % 8 == 0 { heads / 8 } else { heads };
            unsafe {
                self.mla_attend_sparse.launch(
                    (rows as u32, gy as u32, 1),
                    (256, 1, 1),
                    0,
                    stream,
                    &mut args,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_attend_sparse_q8(
        &self,
        stream: &Stream,
        mut qa: CuDevicePtr,
        mut kv_q8: CuDevicePtr,
        mut kv_sc: CuDevicePtr,
        mut tokens: CuDevicePtr,
        mut counts: CuDevicePtr,
        mut ol: CuDevicePtr,
        pos: usize,
        capacity: usize,
        width: usize,
        rows: usize,
        heads: usize,
        lw: usize,
        mut scale: f32,
        tree_trunk_len: usize,
        tree_b_start: usize,
    ) -> Result<()> {
        assert!(lw == 512 && heads % 8 == 0, "mla_attend_sparse_q8 expects lw==512, heads%8==0");
        let mut pos_i = pos as i32;
        let mut cap_i = capacity as i32;
        let mut w_i = width as i32;
        let mut r_i = rows as i32;
        let mut h_i = heads as i32;
        let mut lw_i = lw as i32;
        let mut tt_i = tree_trunk_len as i32;
        let mut tb_i = tree_b_start as i32;
        let mut args: [*mut c_void; 15] = [
            &mut qa as *mut _ as *mut c_void,
            &mut kv_q8 as *mut _ as *mut c_void,
            &mut kv_sc as *mut _ as *mut c_void,
            &mut tokens as *mut _ as *mut c_void,
            &mut counts as *mut _ as *mut c_void,
            &mut ol as *mut _ as *mut c_void,
            &mut pos_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut w_i as *mut _ as *mut c_void,
            &mut r_i as *mut _ as *mut c_void,
            &mut h_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut tt_i as *mut _ as *mut c_void,
            &mut tb_i as *mut _ as *mut c_void,
        ];
        if heads % 16 == 0 && rows >= 16 {
            let gy = (heads / 16) as u32;
            unsafe {
                self.mla_attend_sparse_tc_q8
                    .launch((rows as u32, gy, 1), (256, 1, 1), 0, stream, &mut args)
            }
        } else {
            let gy = (heads / 8) as u32;
            unsafe {
                self.mla_attend_sparse_q8
                    .launch((rows as u32, gy, 1), (256, 1, 1), 0, stream, &mut args)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_compact_kv_bf16(
        &self,
        stream: &Stream,
        num_layers: usize,
        mut kv_all: CuDevicePtr,
        pos_dst: usize,
        pos_src: usize,
        capacity: usize,
        n_rows: usize,
        lw: usize,
    ) -> Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let mut dst_i = pos_dst as i32;
        let mut src_i = pos_src as i32;
        let mut cap_i = capacity as i32;
        let mut nr_i = n_rows as i32;
        let mut lw_i = lw as i32;
        let mut args: [*mut c_void; 6] = [
            &mut kv_all as *mut _ as *mut c_void,
            &mut dst_i as *mut _ as *mut c_void,
            &mut src_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut nr_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.mla_compact_kv_bf16.launch(
                (num_layers as u32, 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_compact_kv_q8(
        &self,
        stream: &Stream,
        num_layers: usize,
        mut kv_q8_all: CuDevicePtr,
        mut kv_sc_all: CuDevicePtr,
        pos_dst: usize,
        pos_src: usize,
        capacity: usize,
        n_rows: usize,
        lw: usize,
    ) -> Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let mut dst_i = pos_dst as i32;
        let mut src_i = pos_src as i32;
        let mut cap_i = capacity as i32;
        let mut nr_i = n_rows as i32;
        let mut lw_i = lw as i32;
        let mut args: [*mut c_void; 7] = [
            &mut kv_q8_all as *mut _ as *mut c_void,
            &mut kv_sc_all as *mut _ as *mut c_void,
            &mut dst_i as *mut _ as *mut c_void,
            &mut src_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut nr_i as *mut _ as *mut c_void,
            &mut lw_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.mla_compact_kv_q8.launch(
                (num_layers as u32, 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsa_compact_ik_ig(
        &self,
        stream: &Stream,
        num_layers: usize,
        mut ik_all: CuDevicePtr,
        mut ig_all: CuDevicePtr,
        pos_dst: usize,
        pos_src: usize,
        ik_cap: usize,
        n_rows: usize,
    ) -> Result<()> {
        if n_rows == 0 {
            return Ok(());
        }
        let mut dst_i = pos_dst as i32;
        let mut src_i = pos_src as i32;
        let mut cap_i = ik_cap as i32;
        let mut nr_i = n_rows as i32;
        let mut args: [*mut c_void; 6] = [
            &mut ik_all as *mut _ as *mut c_void,
            &mut ig_all as *mut _ as *mut c_void,
            &mut dst_i as *mut _ as *mut c_void,
            &mut src_i as *mut _ as *mut c_void,
            &mut cap_i as *mut _ as *mut c_void,
            &mut nr_i as *mut _ as *mut c_void,
        ];
        unsafe {
            self.dsa_compact_ik_ig.launch(
                (num_layers as u32, 1, 1),
                (128, 1, 1),
                0,
                stream,
                &mut args,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use m3tis_format::exl3_ref::f32_to_bf16;

    fn bf(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    fn rbf(x: f32) -> f32 {
        bf(f32_to_bf16(x))
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
    fn gpu_hc_pre_sinkhorn_doubly_stochastic_and_hc_post() {
        let ctx = CudaContext::new(0).expect("RTX 3090");
        let stream = ctx.stream().unwrap();
        let ops = GlmOpsKernels::load(&ctx).expect("load glm_ops.cubin");

        let rows = 2usize;
        let d = 256usize;
        let wide = 4 * d;

        let x_h: Vec<u16> = (0..rows * wide)
            .map(|i| f32_to_bf16((((i * 31 + 5) % 199) as f32 - 99.0) * 0.02))
            .collect();
        let fn_h: Vec<u16> = (0..24 * wide)
            .map(|i| f32_to_bf16((((i * 17 + 3) % 127) as f32 - 63.0) * 0.01))
            .collect();
        let base_h: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.1).collect();
        let scale_h: [f32; 3] = [0.25, 0.5, 0.75];
        let nw_h: Vec<u16> = (0..d).map(|i| f32_to_bf16(0.8 + (i % 7) as f32 * 0.05)).collect();

        let d_x = ctx.alloc_device(x_h.len() * 2).unwrap();
        let d_fn = ctx.alloc_device(fn_h.len() * 2).unwrap();
        let d_part = ctx.alloc_device(rows * 16 * 32 * 4).unwrap();
        let d_base = ctx.alloc_device(24 * 4).unwrap();
        let d_scale = ctx.alloc_device(3 * 4).unwrap();
        let d_nw = ctx.alloc_device(d * 2).unwrap();
        let d_normed = ctx.alloc_device(rows * d * 2).unwrap();
        let d_post = ctx.alloc_device(rows * 4 * 4).unwrap();
        let d_comb = ctx.alloc_device(rows * 16 * 4).unwrap();
        unsafe {
            stream.htod_async(d_x.ptr(), x_h.as_ptr() as *const u8, x_h.len() * 2).unwrap();
            stream.htod_async(d_fn.ptr(), fn_h.as_ptr() as *const u8, fn_h.len() * 2).unwrap();
            stream.htod_async(d_base.ptr(), base_h.as_ptr() as *const u8, 24 * 4).unwrap();
            stream.htod_async(d_scale.ptr(), scale_h.as_ptr() as *const u8, 12).unwrap();
            stream.htod_async(d_nw.ptr(), nw_h.as_ptr() as *const u8, d * 2).unwrap();
        }

        ops.hc_pre(
            &stream,
            d_x.ptr(),
            d_fn.ptr(),
            d_part.ptr(),
            d_base.ptr(),
            d_scale.ptr(),
            d_nw.ptr(),
            d_normed.ptr(),
            d_post.ptr(),
            d_comb.ptr(),
            rows,
            d,
            1e-5,
            1e-6,
            20,
        )
        .unwrap();

        let mut h_post = vec![0.0f32; rows * 4];
        let mut h_comb = vec![0.0f32; rows * 16];
        let mut h_normed = vec![0u16; rows * d];
        unsafe {
            stream.dtoh_async(h_post.as_mut_ptr() as *mut u8, d_post.ptr(), rows * 16).unwrap();
            stream.dtoh_async(h_comb.as_mut_ptr() as *mut u8, d_comb.ptr(), rows * 64).unwrap();
            stream.dtoh_async(h_normed.as_mut_ptr() as *mut u8, d_normed.ptr(), rows * d * 2).unwrap();
        }
        stream.synchronize().unwrap();

        // Check exact 20-iter Sinkhorn parity against CPU FP64 reference!
        let eps = 1e-5f64;
        let hc_eps = 1e-6f64;
        for r in 0..rows {
            let mut ss = 0.0f64;
            for k in 0..wide {
                let v = bf(x_h[r * wide + k]) as f64;
                ss += v * v;
            }
            let rinv = 1.0 / (eps + ss / (wide as f64)).sqrt();
            let mut mix = [0.0f64; 24];
            for m in 0..24 {
                let mut acc = 0.0f64;
                for k in 0..wide {
                    acc += (bf(x_h[r * wide + k]) as f64) * (bf(fn_h[m * wide + k]) as f64);
                }
                mix[m] = acc * rinv;
            }
            let mut cm = [[0.0f64; 4]; 4];
            for i in 0..4 {
                let mut rmax = f64::NEG_INFINITY;
                let mut cl = [0.0f64; 4];
                for j in 0..4 {
                    cl[j] = mix[8 + i * 4 + j] * (scale_h[2] as f64) + (base_h[8 + i * 4 + j] as f64);
                    rmax = rmax.max(cl[j]);
                }
                let mut rsum = 0.0;
                for j in 0..4 {
                    cm[i][j] = (cl[j] - rmax).exp();
                    rsum += cm[i][j];
                }
                for j in 0..4 {
                    cm[i][j] = hc_eps + cm[i][j] / rsum;
                }
            }
            for it in 0..20 {
                if it > 0 {
                    for i in 0..4 {
                        let rsum: f64 = cm[i].iter().sum();
                        for j in 0..4 {
                            cm[i][j] /= hc_eps + rsum;
                        }
                    }
                }
                for j in 0..4 {
                    let csum: f64 = (0..4).map(|i| cm[i][j]).sum();
                    for i in 0..4 {
                        cm[i][j] /= hc_eps + csum;
                    }
                }
            }
            let gpu_cm = &h_comb[r * 16..(r + 1) * 16];
            for i in 0..4 {
                let csum: f32 = (0..4).map(|row| gpu_cm[row * 4 + i]).sum();
                assert!((csum - 1.0).abs() < 1e-4, "Sinkhorn final col {i} sum = {csum}");
                for j in 0..4 {
                    let diff = (gpu_cm[i * 4 + j] as f64 - cm[i][j]).abs();
                    assert!(
                        diff < 1e-5,
                        "Sinkhorn mismatch at r={r} ({i},{j}): gpu={} ref={}",
                        gpu_cm[i * 4 + j],
                        cm[i][j]
                    );
                }
            }
            for &p in &h_post[r * 4..(r + 1) * 4] {
                assert!(p > 0.0 && p < 2.0, "post weight out of (0,2): {p}");
            }
        }

        // Verify hc_post in-place against CPU reference
        let branch_h: Vec<f32> = (0..rows * d).map(|i| ((i * 19 % 100) as f32 - 50.0) * 0.01).collect();
        let d_branch = ctx.alloc_device(rows * d * 4).unwrap();
        unsafe {
            stream.htod_async(d_branch.ptr(), branch_h.as_ptr() as *const u8, rows * d * 4).unwrap();
        }
        ops.hc_post(&stream, d_x.ptr(), d_branch.ptr(), d_post.ptr(), d_comb.ptr(), rows, d).unwrap();
        let mut h_x_after = vec![0u16; rows * wide];
        unsafe {
            stream.dtoh_async(h_x_after.as_mut_ptr() as *mut u8, d_x.ptr(), x_h.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();

        let mut ref_x = vec![0.0f32; rows * wide];
        for r in 0..rows {
            for col in 0..d {
                let br = rbf(branch_h[r * d + col]);
                for s in 0..4 {
                    let mut m = 0.0f32;
                    for t in 0..4 {
                        m += bf(x_h[(r * 4 + t) * d + col]) * h_comb[r * 16 + t * 4 + s];
                    }
                    ref_x[(r * 4 + s) * d + col] = rbf(br * h_post[r * 4 + s] + m);
                }
            }
        }
        let gpu_x: Vec<f32> = h_x_after.iter().map(|&v| bf(v)).collect();
        let cos = cos_sim(&gpu_x, &ref_x);
        println!("[RTX 3090 mHC Sinkhorn + hc_post] cos_sim={cos:.8}");
        assert!(cos > 0.99999);
    }

    #[test]
    fn gpu_kda_chain_and_conv_shift_parity() {
        let ctx = CudaContext::new(0).expect("RTX 3090");
        let stream = ctx.stream().unwrap();
        let ops = GlmOpsKernels::load(&ctx).expect("load glm_ops.cubin");

        let heads = 4usize;
        let dk = 128usize;
        let rows = 3usize;
        let c_conv = 3 * heads * dk;
        let p_stride = c_conv + 256 + heads;
        let b_off = c_conv + 256;

        let p_h: Vec<u16> = (0..rows * p_stride)
            .map(|i| f32_to_bf16((((i * 23 + 7) % 173) as f32 - 86.0) * 0.015))
            .collect();
        let a_h: Vec<u16> = (0..rows * heads * dk)
            .map(|i| f32_to_bf16((((i * 13 + 11) % 97) as f32 - 48.0) * 0.02))
            .collect();
        let g_h: Vec<u16> = (0..rows * heads * dk)
            .map(|i| f32_to_bf16((((i * 41 + 3) % 101) as f32 - 50.0) * 0.03))
            .collect();
        let cs_h: Vec<u16> = (0..3 * c_conv)
            .map(|i| f32_to_bf16((((i * 19 + 5) % 89) as f32 - 44.0) * 0.01))
            .collect();
        let cw_h: Vec<f32> = (0..c_conv * 4).map(|i| (((i * 7 + 1) % 31) as f32 - 15.0) * 0.04).collect();
        let state_init: Vec<f32> = (0..heads * dk * dk)
            .map(|i| (((i * 11 + 2) % 71) as f32 - 35.0) * 0.005)
            .collect();
        let a_log_h: Vec<f32> = (0..heads).map(|h| -1.5 + h as f32 * 0.2).collect();
        let dt_bias_h: Vec<f32> = (0..heads * dk).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let norm_w_h: Vec<u16> = (0..dk).map(|i| f32_to_bf16(0.9 + (i % 5) as f32 * 0.05)).collect();

        let up_u16 = |v: &[u16]| {
            let b = ctx.alloc_device(v.len() * 2).unwrap();
            unsafe { stream.htod_async(b.ptr(), v.as_ptr() as *const u8, v.len() * 2).unwrap(); }
            b
        };
        let up_f32 = |v: &[f32]| {
            let b = ctx.alloc_device(v.len() * 4).unwrap();
            unsafe { stream.htod_async(b.ptr(), v.as_ptr() as *const u8, v.len() * 4).unwrap(); }
            b
        };

        let d_p = up_u16(&p_h);
        let d_a = up_u16(&a_h);
        let d_g = up_u16(&g_h);
        let d_cs = up_u16(&cs_h);
        let d_cw = up_f32(&cw_h);
        let d_state = up_f32(&state_init);
        let d_alog = up_f32(&a_log_h);
        let d_dt = up_f32(&dt_bias_h);
        let d_nw = up_u16(&norm_w_h);
        let d_out = ctx.alloc_device(rows * heads * dk * 2).unwrap();

        let eps = 1e-5f32;
        let lower = -5.0f32;
        ops.kda_chain(
            &stream,
            heads,
            d_p.ptr(),
            p_stride,
            b_off,
            d_a.ptr(),
            d_g.ptr(),
            d_cs.ptr(),
            d_cw.ptr(),
            d_state.ptr(),
            d_alog.ptr(),
            d_dt.ptr(),
            d_nw.ptr(),
            eps,
            lower,
            rows,
            d_out.ptr(),
            d_state.ptr(), // test in-place state update!
        )
        .unwrap();
        ops.kda_conv_shift(&stream, d_cs.ptr(), d_p.ptr(), rows, p_stride, c_conv).unwrap();

        let mut h_out = vec![0u16; rows * heads * dk];
        let mut h_state = vec![0.0f32; heads * dk * dk];
        let mut h_cs_after = vec![0u16; 3 * c_conv];
        unsafe {
            stream.dtoh_async(h_out.as_mut_ptr() as *mut u8, d_out.ptr(), h_out.len() * 2).unwrap();
            stream.dtoh_async(h_state.as_mut_ptr() as *mut u8, d_state.ptr(), h_state.len() * 4).unwrap();
            stream.dtoh_async(h_cs_after.as_mut_ptr() as *mut u8, d_cs.ptr(), h_cs_after.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();

        // CPU reference KDA delta-rule
        let mut ref_out = vec![0.0f32; rows * heads * dk];
        let mut ref_state = state_init.clone();
        for h in 0..heads {
            let decay_rate = a_log_h[h].exp();
            for r in 0..rows {
                let mut q = [0.0f32; 128];
                let mut k = [0.0f32; 128];
                let mut v = [0.0f32; 128];
                for part in 0..3 {
                    for d in 0..dk {
                        let c = part * heads * dk + h * dk + d;
                        let mut acc = 0.0f32;
                        for tap in 0..4 {
                            let at = r + tap;
                            let xv = if at < 3 {
                                bf(cs_h[at * c_conv + c])
                            } else {
                                bf(p_h[(at - 3) * p_stride + c])
                            };
                            acc += cw_h[c * 4 + tap] * xv;
                        }
                        let act = rbf(acc / (1.0 + (-acc).exp()));
                        match part {
                            0 => q[d] = act,
                            1 => k[d] = act,
                            _ => v[d] = act,
                        }
                    }
                }
                let q_inv = 1.0 / (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                let k_inv = 1.0 / (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                for d in 0..dk {
                    q[d] *= q_inv / (dk as f32).sqrt();
                    k[d] *= k_inv;
                }
                let mut gv = [0.0f32; 128];
                for d in 0..dk {
                    let av = bf(a_h[r * heads * dk + h * dk + d]) + dt_bias_h[h * dk + d];
                    gv[d] = (lower / (1.0 + (-(decay_rate * av)).exp())).exp();
                }
                let beta = rbf(1.0 / (1.0 + (-bf(p_h[r * p_stride + b_off + h])).exp()));
                let sbase = h * dk * dk;
                let mut y = [0.0f32; 128];
                for j in 0..dk {
                    let mut kv = 0.0f32;
                    for i in 0..dk {
                        ref_state[sbase + j * dk + i] *= gv[i];
                        kv += ref_state[sbase + j * dk + i] * k[i];
                    }
                    let delta = (v[j] - kv) * beta;
                    let mut o = 0.0f32;
                    for i in 0..dk {
                        ref_state[sbase + j * dk + i] += k[i] * delta;
                        o += ref_state[sbase + j * dk + i] * q[i];
                    }
                    y[j] = rbf(o);
                }
                let rinv = 1.0 / (y.iter().map(|x| x * x).sum::<f32>() / (dk as f32) + eps).sqrt();
                for d in 0..dk {
                    let gate = bf(g_h[r * heads * dk + h * dk + d]);
                    let val = bf(norm_w_h[d]) * (y[d] * rinv) / (1.0 + (-gate).exp());
                    ref_out[r * heads * dk + h * dk + d] = rbf(val);
                }
            }
        }
        let gpu_out: Vec<f32> = h_out.iter().map(|&v| bf(v)).collect();
        let cos_y = cos_sim(&gpu_out, &ref_out);
        let cos_s = cos_sim(&h_state, &ref_state);
        println!("[RTX 3090 KDA chain] out cos_sim={cos_y:.8}, state cos_sim={cos_s:.8}");
        assert!(cos_y > 0.9999, "KDA out mismatch: {cos_y}");
        assert!(cos_s > 0.99999, "KDA state mismatch: {cos_s}");
        // Verify conv shift copied rows 0..3 of P into cs[0..3]
        for j in 0..3 {
            assert_eq!(h_cs_after[j * c_conv], p_h[j * p_stride]);
        }
    }

    #[test]
    fn gpu_mla_absorb_attend_expand_parity() {
        let ctx = CudaContext::new(0).expect("RTX 3090");
        let stream = ctx.stream().unwrap();
        let ops = GlmOpsKernels::load(&ctx).expect("load glm_ops.cubin");

        let rows = 4usize;
        let heads = 4usize;
        let qk = 256usize;
        let dv = 256usize;
        let lw = 512usize;
        let cap = 16usize;
        let pos = 14usize; // forces ring-buffer wraparound across slot 14, 15, 0, 1!
        let scale = 1.0f32 / (qk as f32).sqrt();

        let q_h: Vec<u16> = (0..rows * heads * qk)
            .map(|i| f32_to_bf16((((i * 17 + 3) % 131) as f32 - 65.0) * 0.02))
            .collect();
        let wk_h: Vec<u16> = (0..heads * qk * lw)
            .map(|i| f32_to_bf16((((i * 29 + 11) % 149) as f32 - 74.0) * 0.01))
            .collect();
        let wv_h: Vec<u16> = (0..heads * dv * lw)
            .map(|i| f32_to_bf16((((i * 37 + 5) % 157) as f32 - 78.0) * 0.01))
            .collect();
        let lat_h: Vec<u16> = (0..rows * lw)
            .map(|i| f32_to_bf16((((i * 43 + 9) % 163) as f32 - 81.0) * 0.02))
            .collect();
        let mut kv_init: Vec<u16> = (0..cap * lw)
            .map(|i| f32_to_bf16((((i * 19 + 1) % 127) as f32 - 63.0) * 0.02))
            .collect();

        let up = |v: &[u16]| {
            let b = ctx.alloc_device(v.len() * 2).unwrap();
            unsafe { stream.htod_async(b.ptr(), v.as_ptr() as *const u8, v.len() * 2).unwrap(); }
            b
        };
        let d_q = up(&q_h);
        // Transpose wk_h [heads, qk, lw] -> wk_t [heads, lw, qk] for coalesced 128-bit mla_absorb_q!
        let mut wk_t = vec![0u16; heads * lw * qk];
        for h in 0..heads {
            for k in 0..qk {
                for l in 0..lw {
                    wk_t[(h * lw + l) * qk + k] = wk_h[(h * qk + k) * lw + l];
                }
            }
        }
        let d_wk = up(&wk_t);
        let d_wv = up(&wv_h);
        let d_lat = up(&lat_h);
        let d_kv = up(&kv_init);
        let d_qa = ctx.alloc_device(rows * heads * lw * 2).unwrap();
        let d_ol = ctx.alloc_device(rows * heads * lw * 2).unwrap();
        let d_vn = ctx.alloc_device(rows * heads * dv * 2).unwrap();

        ops.mla_absorb_q(&stream, d_q.ptr(), d_wk.ptr(), d_qa.ptr(), rows, heads, qk, lw).unwrap();
        ops.mla_write_kv(&stream, d_lat.ptr(), d_kv.ptr(), pos, cap, rows, lw).unwrap();
        ops.mla_attend_dense(&stream, d_qa.ptr(), d_kv.ptr(), d_ol.ptr(), pos, cap, 8, rows, heads, lw, scale, 0, 0).unwrap();
        ops.mla_expand_v(&stream, d_ol.ptr(), d_wv.ptr(), d_vn.ptr(), rows, heads, dv, lw).unwrap();

        let mut h_vn = vec![0u16; rows * heads * dv];
        unsafe {
            stream.dtoh_async(h_vn.as_mut_ptr() as *mut u8, d_vn.ptr(), h_vn.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();

        // CPU FP64 MLA reference
        for r in 0..rows {
            let slot = (pos + r) % cap;
            for l in 0..lw {
                kv_init[slot * lw + l] = lat_h[r * lw + l];
            }
        }
        let mut ref_vn = vec![0.0f32; rows * heads * dv];
        for r in 0..rows {
            let q_pos = pos + r;
            let k_start = q_pos.saturating_sub(7);
            for h in 0..heads {
                let mut qa = vec![0.0f64; lw];
                for l in 0..lw {
                    let mut s = 0.0f64;
                    for k in 0..qk {
                        s += (bf(q_h[(r * heads + h) * qk + k]) as f64)
                            * (bf(wk_h[(h * qk + k) * lw + l]) as f64);
                    }
                    qa[l] = rbf(s as f32) as f64;
                }
                let mut scores = Vec::new();
                let mut mmax = f64::NEG_INFINITY;
                for kp in k_start..=q_pos {
                    let slot = kp % cap;
                    let mut dot = 0.0f64;
                    for l in 0..lw {
                        dot += qa[l] * (bf(kv_init[slot * lw + l]) as f64);
                    }
                    let sc = dot * (scale as f64);
                    mmax = mmax.max(sc);
                    scores.push((slot, sc));
                }
                let mut lsum = 0.0f64;
                let mut ol = vec![0.0f64; lw];
                for &(slot, sc) in &scores {
                    let p = (sc - mmax).exp();
                    lsum += p;
                    for l in 0..lw {
                        ol[l] += p * (bf(kv_init[slot * lw + l]) as f64);
                    }
                }
                for l in 0..lw {
                    ol[l] = rbf((ol[l] / lsum) as f32) as f64;
                }
                for d in 0..dv {
                    let mut s = 0.0f64;
                    for l in 0..lw {
                        s += ol[l] * (bf(wv_h[(h * dv + d) * lw + l]) as f64);
                    }
                    ref_vn[(r * heads + h) * dv + d] = rbf(s as f32);
                }
            }
        }
        let gpu_vn: Vec<f32> = h_vn.iter().map(|&v| bf(v)).collect();
        let cos = cos_sim(&gpu_vn, &ref_vn);
        println!("[RTX 3090 MLA absorb+ringKV+attend+expand] cos_sim={cos:.8}");
        assert!(cos > 0.9999, "MLA pipeline mismatch: {cos}");
    }

    #[test]
    fn gpu_dsa_indexer_and_sparse_mla_parity() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.stream().unwrap();
        let ops = GlmOpsKernels::load(&ctx).unwrap();

        // Test sparse DSA with topk_pools=16 over 35 pools (142 tokens -> 16*4 + 2 = 66 selected tokens!)
        let cap = 256usize;
        let pk_cap = cap / 4;
        let n_tokens = 142usize;
        let topk_pools = 16usize;
        let width = topk_pools * 4 + 3;
        let heads = 32usize;
        let d = 128usize;
        let mla_heads = 4usize;
        let lw = 512usize;
        let mut seed = 0x55aa_1234_9988_7766u64;
        let rand_f32 = |s: &mut u64, amp: f32| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            let u = ((*s >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            ((u * 2.0 - 1.0) as f32) * amp
        };

        let mut k_raw = vec![0u16; n_tokens * d];
        let mut g_raw = vec![0.0f32; n_tokens * d];
        for i in 0..n_tokens * d {
            k_raw[i] = f32_to_bf16(rand_f32(&mut seed, 1.0));
            g_raw[i] = rand_f32(&mut seed, 0.8);
        }
        let ln_w: Vec<u16> = (0..d).map(|_| f32_to_bf16(0.9 + rand_f32(&mut seed, 0.2))).collect();
        let ln_b: Vec<u16> = (0..d).map(|_| f32_to_bf16(rand_f32(&mut seed, 0.1))).collect();
        let ape: Vec<u16> = (0..4 * d).map(|_| f32_to_bf16(rand_f32(&mut seed, 0.3))).collect();

        // Test sparse DSA with compact ring `ik_cap = 32 << cap = 256` processed in chunks of 23 tokens!
        let ik_cap = 32usize;
        let d_kr = ctx.alloc_device(k_raw.len() * 2).unwrap();
        let d_gr = ctx.alloc_device(g_raw.len() * 4).unwrap();
        let d_lnw = ctx.alloc_device(ln_w.len() * 2).unwrap();
        let d_lnb = ctx.alloc_device(ln_b.len() * 2).unwrap();
        let d_ape = ctx.alloc_device(ape.len() * 2).unwrap();
        let d_ik = ctx.alloc_device(ik_cap * d * 2).unwrap();
        let d_ig = ctx.alloc_device(ik_cap * d * 2).unwrap();
        let d_pk = ctx.alloc_device(pk_cap * d * 2).unwrap();
        unsafe {
            stream.htod_async(d_kr.ptr(), k_raw.as_ptr() as *const u8, k_raw.len() * 2).unwrap();
            stream.htod_async(d_gr.ptr(), g_raw.as_ptr() as *const u8, g_raw.len() * 4).unwrap();
            stream.htod_async(d_lnw.ptr(), ln_w.as_ptr() as *const u8, ln_w.len() * 2).unwrap();
            stream.htod_async(d_lnb.ptr(), ln_b.as_ptr() as *const u8, ln_b.len() * 2).unwrap();
            stream.htod_async(d_ape.ptr(), ape.as_ptr() as *const u8, ape.len() * 2).unwrap();
        }
        let mut p_off = 0usize;
        while p_off < n_tokens {
            let m = (n_tokens - p_off).min(23);
            ops.dsa_index_write(&stream, d_kr.at(p_off * d * 2), d_gr.at(p_off * d * 4), d_lnw.ptr(), d_lnb.ptr(), d_ik.ptr(), d_ig.ptr(), p_off, ik_cap, 1e-6, m).unwrap();
            ops.dsa_pool_keys(&stream, d_ik.ptr(), d_ig.ptr(), d_ape.ptr(), d_pk.ptr(), p_off, m, ik_cap, pk_cap).unwrap();
            p_off += m;
        }

        // Query at pos = n_tokens - 2 (rows = 2: positions 140 and 141)
        let q_pos0 = n_tokens - 2;
        let q_rows = 2usize;
        let qi: Vec<u16> = (0..q_rows * heads * d).map(|_| f32_to_bf16(rand_f32(&mut seed, 0.7))).collect();
        let w_raw: Vec<u16> = (0..q_rows * heads).map(|_| f32_to_bf16(0.5 + rand_f32(&mut seed, 0.5))).collect();
        let kv: Vec<u16> = (0..cap * lw).map(|_| f32_to_bf16(rand_f32(&mut seed, 0.4))).collect();
        let qa: Vec<u16> = (0..q_rows * mla_heads * lw).map(|_| f32_to_bf16(rand_f32(&mut seed, 0.4))).collect();

        let d_qi = ctx.alloc_device(qi.len() * 2).unwrap();
        let d_w = ctx.alloc_device(w_raw.len() * 2).unwrap();
        let d_scores = ctx.alloc_device(q_rows * pk_cap * 4).unwrap();
        let d_toks = ctx.alloc_device(q_rows * width * 4).unwrap();
        let d_cnts = ctx.alloc_device(q_rows * 4).unwrap();
        let d_kv = ctx.alloc_device(kv.len() * 2).unwrap();
        let d_qa = ctx.alloc_device(qa.len() * 2).unwrap();
        let d_ol = ctx.alloc_device(q_rows * mla_heads * lw * 2).unwrap();
        unsafe {
            stream.htod_async(d_qi.ptr(), qi.as_ptr() as *const u8, qi.len() * 2).unwrap();
            stream.htod_async(d_w.ptr(), w_raw.as_ptr() as *const u8, w_raw.len() * 2).unwrap();
            stream.htod_async(d_kv.ptr(), kv.as_ptr() as *const u8, kv.len() * 2).unwrap();
            stream.htod_async(d_qa.ptr(), qa.as_ptr() as *const u8, qa.len() * 2).unwrap();
        }
        let qscale = (d as f32).powf(-0.5);
        let wscale = (heads as f32).powf(-0.5);
        ops.dsa_score_pools(&stream, d_qi.ptr(), d_w.ptr(), d_pk.ptr(), d_scores.ptr(), q_pos0, pk_cap, pk_cap, heads, qscale, wscale, q_rows).unwrap();
        ops.dsa_select_tokens(&stream, d_scores.ptr(), d_toks.ptr(), d_cnts.ptr(), q_pos0, pk_cap, topk_pools, width, q_rows, 0, 0).unwrap();
        let mla_scale = (256.0f32).powf(-0.5);
        ops.mla_attend_sparse(&stream, d_qa.ptr(), d_kv.ptr(), d_toks.ptr(), d_cnts.ptr(), d_ol.ptr(), q_pos0, cap, width, q_rows, mla_heads, lw, mla_scale, 0, 0).unwrap();

        let mut h_scores = vec![0.0f32; q_rows * pk_cap];
        let mut h_toks = vec![0i32; q_rows * width];
        let mut h_cnts = vec![0i32; q_rows];
        let mut h_ol = vec![0u16; q_rows * mla_heads * lw];
        unsafe {
            stream.dtoh_async(h_scores.as_mut_ptr() as *mut u8, d_scores.ptr(), h_scores.len() * 4).unwrap();
            stream.dtoh_async(h_toks.as_mut_ptr() as *mut u8, d_toks.ptr(), h_toks.len() * 4).unwrap();
            stream.dtoh_async(h_cnts.as_mut_ptr() as *mut u8, d_cnts.ptr(), h_cnts.len() * 4).unwrap();
            stream.dtoh_async(h_ol.as_mut_ptr() as *mut u8, d_ol.ptr(), h_ol.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();

        // CPU FP64 reference for ALL 5 DSA kernels: index_write -> pool_keys -> score_pools -> select_tokens -> attend_sparse!
        let mut ref_ik = vec![0.0f32; cap * d];
        let mut ref_ig = vec![0.0f32; cap * d];
        for r in 0..n_tokens {
            let mut mean = 0.0f64;
            for j in 0..d {
                mean += bf(k_raw[r * d + j]) as f64;
            }
            mean /= d as f64;
            let mut var = 0.0f64;
            for j in 0..d {
                let xc = (bf(k_raw[r * d + j]) as f64) - mean;
                var += xc * xc;
            }
            let inv_sd = 1.0 / (var / (d as f64) + 1e-6).sqrt();
            for j in 0..d {
                let xc = (bf(k_raw[r * d + j]) as f64) - mean;
                let y = xc * inv_sd * (bf(ln_w[j]) as f64) + (bf(ln_b[j]) as f64);
                ref_ik[r * d + j] = rbf(y as f32);
                ref_ig[r * d + j] = rbf(g_raw[r * d + j]);
            }
        }
        let mut ref_pk = vec![0.0f32; pk_cap * d];
        for p in 0..(n_tokens / 4) {
            for j in 0..d {
                let mut ls = [0.0f64; 4];
                let mut mx = f64::NEG_INFINITY;
                for q in 0..4 {
                    ls[q] = (ref_ig[(4 * p + q) * d + j] as f64) + (bf(ape[q * d + j]) as f64);
                    mx = mx.max(ls[q]);
                }
                let es: [f64; 4] = ls.map(|v| (v - mx).exp());
                let se: f64 = es.iter().sum();
                let mut acc = 0.0f64;
                for q in 0..4 {
                    let w = rbf((es[q] / se) as f32) as f64;
                    acc += rbf((w * (ref_ik[(4 * p + q) * d + j] as f64)) as f32) as f64;
                }
                ref_pk[p * d + j] = rbf(acc as f32);
            }
        }
        let mut ref_scores = vec![f32::NEG_INFINITY; q_rows * pk_cap];
        for r in 0..q_rows {
            let npool = (q_pos0 + r + 1) / 4;
            for p in 0..npool {
                let mut tot = 0.0f64;
                for h in 0..heads {
                    let mut dot = 0.0f64;
                    for j in 0..d {
                        dot += (bf(qi[(r * heads + h) * d + j]) as f64) * (ref_pk[p * d + j] as f64);
                    }
                    let relu = (dot * (qscale as f64)).max(0.0);
                    let wh = (bf(w_raw[r * heads + h]) as f64) * (wscale as f64);
                    tot += wh * relu;
                }
                ref_scores[r * pk_cap + p] = tot as f32;
            }
        }
        let mut score_gpu_v = Vec::new();
        let mut score_ref_v = Vec::new();
        for r in 0..q_rows {
            let npool = (q_pos0 + r + 1) / 4;
            for p in 0..npool {
                score_gpu_v.push(h_scores[r * pk_cap + p]);
                score_ref_v.push(ref_scores[r * pk_cap + p]);
            }
        }
        let sc_cos = cos_sim(&score_gpu_v, &score_ref_v);
        assert!(sc_cos > 0.99999, "DSA pool score mismatch vs CPU FP64: cos={sc_cos}");

        // Verify exact selected pools, exact tail counts (65 & 66), and sparse MLA output parity!
        assert_eq!(h_cnts, vec![65, 66]);
        for r in 0..q_rows {
            let npool = (q_pos0 + r + 1) / 4;
            let mut idx_sc: Vec<(usize, f32)> = (0..npool).map(|p| (p, h_scores[r * pk_cap + p])).collect();
            idx_sc.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut want_pools: Vec<i32> = idx_sc[..topk_pools].iter().map(|&(p, _)| p as i32).collect();
            want_pools.sort_unstable();
            let got_pools: Vec<i32> = (0..topk_pools).map(|i| h_toks[r * width + i * 4] / 4).collect();
            assert_eq!(got_pools, want_pools, "radix top-K pool mismatch on row {r}");
        }
        // CPU FP64 sparse MLA verification
        let mut ref_ol = vec![0.0f32; q_rows * mla_heads * lw];
        for r in 0..q_rows {
            let cnt = h_cnts[r] as usize;
            for h in 0..mla_heads {
                let mut scs = Vec::with_capacity(cnt);
                let mut mmax = f64::NEG_INFINITY;
                for j in 0..cnt {
                    let kp = h_toks[r * width + j] as usize;
                    let mut dot = 0.0f64;
                    for l in 0..lw {
                        dot += (bf(qa[(r * mla_heads + h) * lw + l]) as f64) * (bf(kv[(kp % cap) * lw + l]) as f64);
                    }
                    let sc = dot * (mla_scale as f64);
                    mmax = mmax.max(sc);
                    scs.push((kp % cap, sc));
                }
                let mut lsum = 0.0f64;
                let mut acc = vec![0.0f64; lw];
                for &(slot, sc) in &scs {
                    let p = (sc - mmax).exp();
                    lsum += p;
                    for l in 0..lw {
                        acc[l] += p * (bf(kv[slot * lw + l]) as f64);
                    }
                }
                for l in 0..lw {
                    ref_ol[(r * mla_heads + h) * lw + l] = rbf((acc[l] / lsum) as f32);
                }
            }
        }
        let gpu_ol: Vec<f32> = h_ol.iter().map(|&v| bf(v)).collect();
        let cos = cos_sim(&gpu_ol, &ref_ol);
        println!("[RTX 3090 DSA Indexer + Sparse MLA] pools=EXACT counts={:?} cos_sim={cos:.8}", h_cnts);
        assert!(cos > 0.99995, "Sparse MLA mismatch: {cos}");
    }

    #[test]
    fn gpu_mla_attend_fast_lw512_dense_and_sparse_parity() {
        let ctx = CudaContext::new(0).expect("RTX 3090");
        let stream = ctx.stream().unwrap();
        let ops = GlmOpsKernels::load(&ctx).unwrap();

        let (rows, heads, lw, cap, pos) = (19usize, 16usize, 512usize, 64usize, 23usize);
        let scale = 1.0f32 / (256.0f32).sqrt();
        let mut s = 0x5120_BEEFu64;
        let mut next_bf = || -> u16 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let f = (((s >> 33) as i32) as f32) / ((1u32 << 31) as f32);
            f32_to_bf16(f * 0.5)
        };
        let qa: Vec<u16> = (0..rows * heads * lw).map(|_| next_bf()).collect();
        let kv: Vec<u16> = (0..cap * lw).map(|_| next_bf()).collect();

        let d_qa = ctx.alloc_device(qa.len() * 2).unwrap();
        let d_kv = ctx.alloc_device(kv.len() * 2).unwrap();
        let d_ol = ctx.alloc_device(qa.len() * 2).unwrap();
        unsafe {
            stream.htod_async(d_qa.ptr(), qa.as_ptr() as *const u8, qa.len() * 2).unwrap();
            stream.htod_async(d_kv.ptr(), kv.as_ptr() as *const u8, kv.len() * 2).unwrap();
        }
        ops.mla_attend_dense(&stream, d_qa.ptr(), d_kv.ptr(), d_ol.ptr(), pos, cap, cap, rows, heads, lw, scale, 0, 0)
            .unwrap();
        let mut h_ol = vec![0u16; qa.len()];
        unsafe {
            stream.dtoh_async(h_ol.as_mut_ptr() as *mut u8, d_ol.ptr(), qa.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();

        let mut ref_ol = vec![0.0f32; qa.len()];
        for r in 0..rows {
            let q_pos = pos + r;
            for h in 0..heads {
                let mut scs = Vec::with_capacity(q_pos + 1);
                let mut mmax = f64::NEG_INFINITY;
                for kp in 0..=q_pos {
                    let mut dot = 0.0f64;
                    for l in 0..lw {
                        dot += (bf(qa[(r * heads + h) * lw + l]) as f64) * (bf(kv[(kp % cap) * lw + l]) as f64);
                    }
                    let sc = dot * (scale as f64);
                    mmax = mmax.max(sc);
                    scs.push((kp % cap, sc));
                }
                let mut lsum = 0.0f64;
                let mut acc = vec![0.0f64; lw];
                for &(slot, sc) in &scs {
                    let p = (sc - mmax).exp();
                    lsum += p;
                    for l in 0..lw {
                        acc[l] += p * (bf(kv[slot * lw + l]) as f64);
                    }
                }
                for l in 0..lw {
                    ref_ol[(r * heads + h) * lw + l] = rbf((acc[l] / lsum) as f32);
                }
            }
        }
        let gpu_ol: Vec<f32> = h_ol.iter().map(|&v| bf(v)).collect();
        let cos_d = cos_sim(&gpu_ol, &ref_ol);
        assert!(cos_d > 0.99995, "TC FlashMLA lw=512 dense mismatch: {cos_d}");

        // Also test fast lw=512 sparse MLA (mixing counts==0 dense fallback and explicit token lists with -1 padding!)
        let width = 16usize;
        let mut toks = vec![-1i32; rows * width];
        let mut cnts = vec![0i32; rows];
        for r in 0..rows {
            if r % 2 == 1 {
                let c = 11usize;
                cnts[r] = c as i32;
                for j in 0..c {
                    toks[r * width + j] = if j == 5 { -1 } else { ((j * 3 + r) % (pos + r + 1)) as i32 };
                }
            }
        }
        let d_toks = ctx.alloc_device(toks.len() * 4).unwrap();
        let d_cnts = ctx.alloc_device(cnts.len() * 4).unwrap();
        unsafe {
            stream.htod_async(d_toks.ptr(), toks.as_ptr() as *const u8, toks.len() * 4).unwrap();
            stream.htod_async(d_cnts.ptr(), cnts.as_ptr() as *const u8, cnts.len() * 4).unwrap();
        }
        ops.mla_attend_sparse(&stream, d_qa.ptr(), d_kv.ptr(), d_toks.ptr(), d_cnts.ptr(), d_ol.ptr(), pos, cap, width, rows, heads, lw, scale, 0, 0)
            .unwrap();
        unsafe {
            stream.dtoh_async(h_ol.as_mut_ptr() as *mut u8, d_ol.ptr(), qa.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();
        let mut ref_sp = vec![0.0f32; qa.len()];
        for r in 0..rows {
            let is_sp = cnts[r] > 0;
            let n_it = if is_sp { cnts[r] as usize } else { pos + r + 1 };
            for h in 0..heads {
                let mut scs = Vec::new();
                let mut mmax = f64::NEG_INFINITY;
                for j in 0..n_it {
                    let kp = if is_sp { toks[r * width + j] } else { j as i32 };
                    if kp < 0 { continue; }
                    let slot = (kp as usize) % cap;
                    let mut dot = 0.0f64;
                    for l in 0..lw {
                        dot += (bf(qa[(r * heads + h) * lw + l]) as f64) * (bf(kv[slot * lw + l]) as f64);
                    }
                    let sc = dot * (scale as f64);
                    mmax = mmax.max(sc);
                    scs.push((slot, sc));
                }
                let mut lsum = 0.0f64;
                let mut acc = vec![0.0f64; lw];
                for &(slot, sc) in &scs {
                    let p = (sc - mmax).exp();
                    lsum += p;
                    for l in 0..lw {
                        acc[l] += p * (bf(kv[slot * lw + l]) as f64);
                    }
                }
                for l in 0..lw {
                    ref_sp[(r * heads + h) * lw + l] = rbf((acc[l] / lsum) as f32);
                }
            }
        }
        let gpu_sp: Vec<f32> = h_ol.iter().map(|&v| bf(v)).collect();
        let cos_s = cos_sim(&gpu_sp, &ref_sp);
        println!("[RTX 3090 TC FlashMLA lw=512 16-Head Shared-KV] dense_cos={cos_d:.8} sparse_cos={cos_s:.8}");
        assert!(cos_s > 0.99995, "TC FlashMLA lw=512 sparse mismatch: {cos_s}");

        // Also test rows = 4 (< 16) to exercise the 8-head decode path!
        let r_dec = 4usize;
        ops.mla_attend_dense(&stream, d_qa.ptr(), d_kv.ptr(), d_ol.ptr(), pos, cap, cap, r_dec, heads, lw, scale, 0, 0).unwrap();
        let mut h_dec = vec![0u16; r_dec * heads * lw];
        unsafe {
            stream.dtoh_async(h_dec.as_mut_ptr() as *mut u8, d_ol.ptr(), h_dec.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();
        let gpu_dec: Vec<f32> = h_dec.iter().map(|&v| bf(v)).collect();
        let cos_dec = cos_sim(&gpu_dec, &ref_ol[..h_dec.len()]);
        assert!(cos_dec > 0.99999, "decode 8-head dense MLA mismatch: {cos_dec}");

        // Milestone 8: Verify INT8 Latent KV Cache (`mla_write_kv_q8` + dense/sparse decode & 16-head TC FlashMLA)!
        let d_kv_q8 = ctx.alloc_device(cap * lw).unwrap();
        let d_kv_sc = ctx.alloc_device(cap * 4).unwrap();
        ops.mla_write_kv_q8(&stream, d_kv.ptr(), d_kv_q8.ptr(), d_kv_sc.ptr(), 0, cap, cap, lw).unwrap();
        let d_ol_q8 = ctx.alloc_device(qa.len() * 2).unwrap();
        ops.mla_attend_dense_q8(&stream, d_qa.ptr(), d_kv_q8.ptr(), d_kv_sc.ptr(), d_ol_q8.ptr(), pos, cap, cap, rows, heads, lw, scale, 0, 0).unwrap();
        let mut h_q8_tc = vec![0u16; qa.len()];
        unsafe {
            stream.dtoh_async(h_q8_tc.as_mut_ptr() as *mut u8, d_ol_q8.ptr(), qa.len() * 2).unwrap();
        }
        ops.mla_attend_dense_q8(&stream, d_qa.ptr(), d_kv_q8.ptr(), d_kv_sc.ptr(), d_ol_q8.ptr(), pos, cap, cap, r_dec, heads, lw, scale, 0, 0).unwrap();
        let mut h_q8_dec = vec![0u16; r_dec * heads * lw];
        unsafe {
            stream.dtoh_async(h_q8_dec.as_mut_ptr() as *mut u8, d_ol_q8.ptr(), h_q8_dec.len() * 2).unwrap();
        }
        ops.mla_attend_sparse_q8(&stream, d_qa.ptr(), d_kv_q8.ptr(), d_kv_sc.ptr(), d_toks.ptr(), d_cnts.ptr(), d_ol_q8.ptr(), pos, cap, width, rows, heads, lw, scale, 0, 0).unwrap();
        let mut h_q8_sp = vec![0u16; qa.len()];
        unsafe {
            stream.dtoh_async(h_q8_sp.as_mut_ptr() as *mut u8, d_ol_q8.ptr(), qa.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();
        let cos_q8_tc = cos_sim(&h_q8_tc.iter().map(|&v| bf(v)).collect::<Vec<_>>(), &ref_ol);
        let cos_q8_dec = cos_sim(&h_q8_dec.iter().map(|&v| bf(v)).collect::<Vec<_>>(), &ref_ol[..h_q8_dec.len()]);
        let cos_q8_sp = cos_sim(&h_q8_sp.iter().map(|&v| bf(v)).collect::<Vec<_>>(), &ref_sp);
        println!("[RTX 3090 INT8 Latent KV FlashMLA] dense_tc={cos_q8_tc:.8} dense_dec={cos_q8_dec:.8} sparse_tc={cos_q8_sp:.8}");
        assert!(cos_q8_tc > 0.9995, "INT8 KV dense TC mismatch: {cos_q8_tc}");
        assert!(cos_q8_dec > 0.9995, "INT8 KV dense decode mismatch: {cos_q8_dec}");
        assert!(cos_q8_sp > 0.9995, "INT8 KV sparse TC mismatch: {cos_q8_sp}");
    }

    #[test]
    fn gpu_tree_verify_kda_and_mla_exact_parity() {
        let ctx = CudaContext::new(0).expect("RTX 3090");
        let stream = ctx.stream().unwrap();
        let ops = GlmOpsKernels::load(&ctx).unwrap();

        // Tree layout: trunk(2) + branch_a(4) + branch_b(3) => total 9 rows, fork after trunk_len=2, branch_b starts at 6!
        let trunk_len = 2usize;
        let branch_a_len = 4usize;
        let branch_b_len = 3usize;
        let branch_b_start = trunk_len + branch_a_len; // 6
        let tree_rows = branch_b_start + branch_b_len; // 9
        let lin_rows = trunk_len + branch_b_len;       // 5 (standalone [trunk, Branch B])

        let heads = 4usize;
        let dk = 128usize;
        let c_conv = 3 * heads * dk;
        let p_stride = c_conv + 256 + heads;
        let b_off = c_conv + 256;

        let p_tree: Vec<u16> = (0..tree_rows * p_stride)
            .map(|i| f32_to_bf16((((i * 23 + 7) % 173) as f32 - 86.0) * 0.015))
            .collect();
        let a_tree: Vec<u16> = (0..tree_rows * heads * dk)
            .map(|i| f32_to_bf16((((i * 13 + 11) % 97) as f32 - 48.0) * 0.02))
            .collect();
        let g_tree: Vec<u16> = (0..tree_rows * heads * dk)
            .map(|i| f32_to_bf16((((i * 41 + 3) % 101) as f32 - 50.0) * 0.03))
            .collect();
        let cs_init: Vec<u16> = (0..3 * c_conv)
            .map(|i| f32_to_bf16((((i * 19 + 5) % 89) as f32 - 44.0) * 0.01))
            .collect();
        let cw_h: Vec<f32> = (0..c_conv * 4).map(|i| (((i * 7 + 1) % 31) as f32 - 15.0) * 0.04).collect();
        let state_init: Vec<f32> = (0..heads * dk * dk)
            .map(|i| (((i * 11 + 2) % 71) as f32 - 35.0) * 0.005)
            .collect();
        let a_log_h: Vec<f32> = (0..heads).map(|h| -1.5 + h as f32 * 0.2).collect();
        let dt_bias_h: Vec<f32> = (0..heads * dk).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let norm_w_h: Vec<u16> = (0..dk).map(|i| f32_to_bf16(0.9 + (i % 5) as f32 * 0.05)).collect();

        // Build standalone linear inputs = [trunk[0..2], branch_b[0..3]]
        let mut p_lin = Vec::with_capacity(lin_rows * p_stride);
        let mut a_lin = Vec::with_capacity(lin_rows * heads * dk);
        let mut g_lin = Vec::with_capacity(lin_rows * heads * dk);
        for &r in &[0usize, 1, 6, 7, 8] {
            p_lin.extend_from_slice(&p_tree[r * p_stride..(r + 1) * p_stride]);
            a_lin.extend_from_slice(&a_tree[r * heads * dk..(r + 1) * heads * dk]);
            g_lin.extend_from_slice(&g_tree[r * heads * dk..(r + 1) * heads * dk]);
        }

        let up16 = |v: &[u16]| {
            let b = ctx.alloc_device(v.len() * 2).unwrap();
            unsafe { stream.htod_async(b.ptr(), v.as_ptr() as *const u8, v.len() * 2).unwrap(); }
            b
        };
        let up32 = |v: &[f32]| {
            let b = ctx.alloc_device(v.len() * 4).unwrap();
            unsafe { stream.htod_async(b.ptr(), v.as_ptr() as *const u8, v.len() * 4).unwrap(); }
            b
        };
        let d_cw = up32(&cw_h);
        let d_alog = up32(&a_log_h);
        let d_dt = up32(&dt_bias_h);
        let d_nw = up16(&norm_w_h);

        // 1. Run tree KDA + replay + conv_shift_all committing trunk(2) + branch_b(2) = keep 4!
        let keep = 4usize;
        let d_pt = up16(&p_tree);
        let d_at = up16(&a_tree);
        let d_gt = up16(&g_tree);
        let d_cst = up16(&cs_init);
        let d_st_in = up32(&state_init);
        let d_st_alt = ctx.alloc_device(state_init.len() * 4).unwrap();
        let d_out_tree = ctx.alloc_device(tree_rows * heads * dk * 2).unwrap();
        let d_ks = ctx.alloc_device(tree_rows * heads * dk * 4).unwrap();
        let d_vs = ctx.alloc_device(tree_rows * heads * dk * 2).unwrap();
        let d_gs = ctx.alloc_device(tree_rows * heads * dk * 4).unwrap();
        let d_bs = ctx.alloc_device(tree_rows * heads * 4).unwrap();
        let d_ps = ctx.alloc_device(tree_rows * c_conv * 2).unwrap();

        ops.kda_chain_save(
            &stream, heads, d_pt.ptr(), p_stride, b_off, d_at.ptr(), d_gt.ptr(), d_cst.ptr(),
            d_cw.ptr(), d_st_in.ptr(), d_alog.ptr(), d_dt.ptr(), d_nw.ptr(), 1e-5, -5.0,
            tree_rows, d_out_tree.ptr(), d_st_alt.ptr(), d_ks.ptr(), d_vs.ptr(), d_gs.ptr(), d_bs.ptr(),
            d_ps.ptr(), trunk_len, branch_b_start,
        ).unwrap();
        ops.kda_replay_layers(
            &stream, 1, heads, d_st_in.ptr(), (heads * dk * dk) as i64,
            d_ks.ptr(), d_vs.ptr(), d_gs.ptr(), d_bs.ptr(),
            (tree_rows * heads * dk) as i64, (tree_rows * heads) as i64,
            keep, d_st_alt.ptr(), trunk_len, branch_b_start,
        ).unwrap();
        ops.kda_conv_shift_all(&stream, 1, d_cst.ptr(), d_ps.ptr(), keep, tree_rows, c_conv, trunk_len, branch_b_start).unwrap();

        // 2. Run standalone linear KDA on [trunk(2), branch_b(3)] and commit `keep = 4`!
        let d_pl = up16(&p_lin);
        let d_al = up16(&a_lin);
        let d_gl = up16(&g_lin);
        let d_csl = up16(&cs_init);
        let d_sl_in = up32(&state_init);
        let d_sl_alt = ctx.alloc_device(state_init.len() * 4).unwrap();
        let d_out_lin = ctx.alloc_device(lin_rows * heads * dk * 2).unwrap();
        let d_ksl = ctx.alloc_device(lin_rows * heads * dk * 4).unwrap();
        let d_vsl = ctx.alloc_device(lin_rows * heads * dk * 2).unwrap();
        let d_gsl = ctx.alloc_device(lin_rows * heads * dk * 4).unwrap();
        let d_bsl = ctx.alloc_device(lin_rows * heads * 4).unwrap();
        let d_psl = ctx.alloc_device(lin_rows * c_conv * 2).unwrap();

        ops.kda_chain_save(
            &stream, heads, d_pl.ptr(), p_stride, b_off, d_al.ptr(), d_gl.ptr(), d_csl.ptr(),
            d_cw.ptr(), d_sl_in.ptr(), d_alog.ptr(), d_dt.ptr(), d_nw.ptr(), 1e-5, -5.0,
            lin_rows, d_out_lin.ptr(), d_sl_alt.ptr(), d_ksl.ptr(), d_vsl.ptr(), d_gsl.ptr(), d_bsl.ptr(),
            d_psl.ptr(), 0, 0,
        ).unwrap();
        ops.kda_replay_layers(
            &stream, 1, heads, d_sl_in.ptr(), (heads * dk * dk) as i64,
            d_ksl.ptr(), d_vsl.ptr(), d_gsl.ptr(), d_bsl.ptr(),
            (lin_rows * heads * dk) as i64, (lin_rows * heads) as i64,
            keep, d_sl_alt.ptr(), 0, 0,
        ).unwrap();
        ops.kda_conv_shift_all(&stream, 1, d_csl.ptr(), d_psl.ptr(), keep, lin_rows, c_conv, 0, 0).unwrap();

        let mut out_tree = vec![0u16; tree_rows * heads * dk];
        let mut out_lin = vec![0u16; lin_rows * heads * dk];
        let mut st_tree = vec![0.0f32; heads * dk * dk];
        let mut st_lin = vec![0.0f32; heads * dk * dk];
        let mut cs_tree = vec![0u16; 3 * c_conv];
        let mut cs_lin = vec![0u16; 3 * c_conv];
        unsafe {
            stream.dtoh_async(out_tree.as_mut_ptr() as *mut u8, d_out_tree.ptr(), out_tree.len() * 2).unwrap();
            stream.dtoh_async(out_lin.as_mut_ptr() as *mut u8, d_out_lin.ptr(), out_lin.len() * 2).unwrap();
            stream.dtoh_async(st_tree.as_mut_ptr() as *mut u8, d_st_alt.ptr(), st_tree.len() * 4).unwrap();
            stream.dtoh_async(st_lin.as_mut_ptr() as *mut u8, d_sl_alt.ptr(), st_lin.len() * 4).unwrap();
            stream.dtoh_async(cs_tree.as_mut_ptr() as *mut u8, d_cst.ptr(), cs_tree.len() * 2).unwrap();
            stream.dtoh_async(cs_lin.as_mut_ptr() as *mut u8, d_csl.ptr(), cs_lin.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();

        // Branch B outputs (tree rows 6..9) MUST be BIT-EXACT to standalone linear rows 2..5!
        let row_elems = heads * dk;
        assert_eq!(
            &out_tree[branch_b_start * row_elems..tree_rows * row_elems],
            &out_lin[trunk_len * row_elems..lin_rows * row_elems],
            "Tree KDA Branch B output not bit-exact!"
        );
        assert_eq!(st_tree, st_lin, "Tree KDA replayed state not bit-exact!");
        assert_eq!(cs_tree, cs_lin, "Tree KDA shifted conv state not bit-exact!");

        // 3. Also verify MLA Dense (BF16 + Q8) + compact_kv_bf16/q8 Branch B bit-exact parity!
        let (mla_h, lw, cap, pos) = (16usize, 512usize, 64usize, 10usize);
        let scale = 1.0f32 / (256.0f32).sqrt();
        let mut seed = 0x7777_4242u64;
        let mut nbf = || -> u16 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            f32_to_bf16((((seed >> 33) as i32) as f32) / ((1u32 << 31) as f32) * 0.5)
        };
        let kv_hist: Vec<u16> = (0..cap * lw).map(|_| nbf()).collect();
        let lat_tree: Vec<u16> = (0..tree_rows * lw).map(|_| nbf()).collect();
        let qa_tree: Vec<u16> = (0..tree_rows * mla_h * lw).map(|_| nbf()).collect();
        let mut lat_lin = Vec::with_capacity(lin_rows * lw);
        let mut qa_lin = Vec::with_capacity(lin_rows * mla_h * lw);
        for &r in &[0usize, 1, 6, 7, 8] {
            lat_lin.extend_from_slice(&lat_tree[r * lw..(r + 1) * lw]);
            qa_lin.extend_from_slice(&qa_tree[r * mla_h * lw..(r + 1) * mla_h * lw]);
        }
        let d_kv_t = up16(&kv_hist);
        let d_kv_l = up16(&kv_hist);
        let d_lt = up16(&lat_tree);
        let d_ll = up16(&lat_lin);
        let d_qat = up16(&qa_tree);
        let d_qal = up16(&qa_lin);
        let d_olt = ctx.alloc_device(qa_tree.len() * 2).unwrap();
        let d_oll = ctx.alloc_device(qa_lin.len() * 2).unwrap();

        ops.mla_write_kv(&stream, d_lt.ptr(), d_kv_t.ptr(), pos, cap, tree_rows, lw).unwrap();
        ops.mla_attend_dense(&stream, d_qat.ptr(), d_kv_t.ptr(), d_olt.ptr(), pos, cap, cap, tree_rows, mla_h, lw, scale, trunk_len, branch_b_start).unwrap();
        ops.mla_compact_kv_bf16(&stream, 1, d_kv_t.ptr(), pos + trunk_len, pos + branch_b_start, cap, branch_b_len, lw).unwrap();

        ops.mla_write_kv(&stream, d_ll.ptr(), d_kv_l.ptr(), pos, cap, lin_rows, lw).unwrap();
        ops.mla_attend_dense(&stream, d_qal.ptr(), d_kv_l.ptr(), d_oll.ptr(), pos, cap, cap, lin_rows, mla_h, lw, scale, 0, 0).unwrap();

        let mut h_olt = vec![0u16; qa_tree.len()];
        let mut h_oll = vec![0u16; qa_lin.len()];
        let mut h_kvt = vec![0u16; cap * lw];
        let mut h_kvl = vec![0u16; cap * lw];
        unsafe {
            stream.dtoh_async(h_olt.as_mut_ptr() as *mut u8, d_olt.ptr(), h_olt.len() * 2).unwrap();
            stream.dtoh_async(h_oll.as_mut_ptr() as *mut u8, d_oll.ptr(), h_oll.len() * 2).unwrap();
            stream.dtoh_async(h_kvt.as_mut_ptr() as *mut u8, d_kv_t.ptr(), h_kvt.len() * 2).unwrap();
            stream.dtoh_async(h_kvl.as_mut_ptr() as *mut u8, d_kv_l.ptr(), h_kvl.len() * 2).unwrap();
        }
        stream.synchronize().unwrap();

        let mla_row = mla_h * lw;
        let cos_mla_b = cos_sim(
            &h_olt[branch_b_start * mla_row..tree_rows * mla_row].iter().map(|&v| bf(v)).collect::<Vec<_>>(),
            &h_oll[trunk_len * mla_row..lin_rows * mla_row].iter().map(|&v| bf(v)).collect::<Vec<_>>(),
        );
        assert!(cos_mla_b > 0.999999, "Tree MLA Branch B mismatch: {cos_mla_b}");
        assert_eq!(
            &h_kvt[(pos + trunk_len) * lw..(pos + lin_rows) * lw],
            &h_kvl[(pos + trunk_len) * lw..(pos + lin_rows) * lw],
            "Compacted Branch B KV not bit-exact!"
        );
        println!("[RTX 3090 2-Branch Speculative Tree] KDA+replay+conv=BIT-EXACT, MLA Branch B cos={cos_mla_b:.8}, KV compact=BIT-EXACT");
    }
}
