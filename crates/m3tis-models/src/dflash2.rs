//! Clean-room DFlash2 speculative drafter for GLM-5.3-Flash (`GLM-5.3-Flash-DFlash2`, Apache-2.0 weights,
//! algorithm after z-lab/dflash, MIT).
//!
//! Architecture:
//! - 5-layer Qwen3 block-diffusion drafter (`D=4096`, `QH=32`, `KVH=8`, `HD=128`, `FF=12288`, `WIN=2048`, `BLOCK=8`).
//! - Consumes 5 tapped 4-stream-mean target layers (`[5, 14, 24, 33, 42]` -> `fc [4096, 20480]` -> `hidden_norm`).
//! - Pre/post 2-tap dynamic causal grouped conv (`group_size=16`) wrapping attention and MLP.
//! - Shares target's Pinned-RAM `embed_tokens` and VRAM `lm_head` (0 duplicate embedding/head VRAM!).
//! - Keeps `predecessor_codebook` & `successor_codebook` (`[154880, 256]` BF16) in Host RAM (`0` VRAM).
//! - Packs drafter linears to Q4 (`gs=64`) on GPU -> only ~650 MiB total VRAM!

use m3tis_cuda::dflash2::DFlash2Kernels;
use m3tis_cuda::glm_ops::GlmOpsKernels;
use m3tis_cuda::linear::{LinearKernels, Q3Weight, Q4Weight};
use m3tis_cuda::{CuDevicePtr, CudaContext, DeviceBuf, PinnedHostBuf, Stream};
use m3tis_format::safetensors::SafetensorsIndex;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub const DF_BLOCK: usize = 8;
pub const DF_MASK_TOKEN: i32 = 154856;
pub const DF_WIN: usize = 2048;
pub const DF_LAYERS: usize = 5;
pub const DF_D: usize = 4096;
pub const DF_KVW: usize = 1024;
pub const DF_FF: usize = 12288;
pub const DF_RANK: usize = 256;
pub const DF_TOPK: usize = 16;
pub const DF_VOCAB: usize = 154880;
pub const DF_EPS: f32 = 1e-5;
pub const DF_THETA: f32 = 10000.0;
pub const DF_SCALE: f32 = 0.08838834764831845; // 1 / sqrt(128)
pub const DF_TAP_LAYERS: [usize; 5] = [5, 14, 24, 33, 42];

/// 2-Branch DFlash2 speculative draft tree (0 extra GPU draft ms — decoded from Pass #1's own
/// top-16 + predecessor/successor codebook selector on CPU in ~5 us!).
#[derive(Debug, Clone)]
pub struct DraftTree {
    /// Primary branch drafted tokens after `anchor`: `[d_0, d_1, ..., d_{k-1}]`.
    pub primary: Vec<i32>,
    /// Cumulative selector probability for each verify row `[1.0, p_0, p_0*p_1, ...]` (`len = 1 + primary.len()`).
    pub row_cum_probs: Vec<f32>,
    /// Index in `primary` (`0..primary.len()`) where `branch_b` diverges (meaning trunk = `[anchor, primary[..fork_pos]]`).
    pub fork_pos: usize,
    /// Alternative branch drafted tokens starting at `fork_pos`: `[alt_f, b_1, ...]`.
    /// Empty when primary confidence is high enough that no fork is needed (pure linear verify).
    pub branch_b: Vec<i32>,
}

#[inline]
pub fn tap_slot_for_layer(layer: usize) -> Option<i32> {
    match layer {
        5 => Some(0),
        14 => Some(1),
        24 => Some(2),
        33 => Some(3),
        42 => Some(4),
        _ => None,
    }
}

enum DfLinear {
    Q4(Q4Weight),
    Q3(Q3Weight),
    Bf16 { buf: DeviceBuf, n: usize, k: usize },
}

impl DfLinear {
    fn vram_bytes(&self) -> usize {
        match self {
            DfLinear::Q4(q) => q.vram_bytes(),
            DfLinear::Q3(q) => q.vram_bytes(),
            DfLinear::Bf16 { buf, .. } => buf.len(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply(
        &self,
        lin: &LinearKernels,
        stream: &Stream,
        x_ptr: CuDevicePtr,
        x_stride: usize,
        out_ptr: CuDevicePtr,
        out_stride: usize,
        rows: usize,
        out_f32: bool,
    ) -> m3tis_cuda::Result<()> {
        match self {
            DfLinear::Q4(q) => {
                lin.linear_q4(stream, x_ptr, x_stride, q, out_ptr, out_stride, rows, out_f32)
            }
            DfLinear::Q3(q) => {
                lin.linear_q3(stream, x_ptr, x_stride, q, out_ptr, out_stride, rows, out_f32)
            }
            DfLinear::Bf16 { buf, n, k } => lin.linear_bf16(
                stream,
                x_ptr,
                x_stride,
                buf.ptr(),
                out_ptr,
                out_stride,
                rows,
                *n,
                *k,
                out_f32,
            ),
        }
    }
}

struct DfLayer {
    in_norm: DeviceBuf,   // [4096] bf16
    post_norm: DeviceBuf, // [4096] bf16
    q_norm: DeviceBuf,    // [128] bf16
    k_norm: DeviceBuf,    // [128] bf16
    a_base: DeviceBuf,    // [2, 2, 4096] bf16
    m_base: DeviceBuf,    // [2, 2, 4096] bf16
    a_proj: DfLinear,     // [1024, 4096]
    m_proj: DfLinear,     // [1024, 4096]
    q_proj: DfLinear,     // [4096, 4096]
    k_proj: DfLinear,     // [1024, 4096]
    v_proj: DfLinear,     // [1024, 4096]
    o_proj: DfLinear,      // [4096, 4096]
    gate_up_proj: DfLinear,// fused [gate; up] = [24576, 4096] (1 launch instead of 2!)
    down_proj: DfLinear,   // [4096, 12288]
}

impl DfLayer {
    fn vram_bytes(&self) -> usize {
        self.in_norm.len()
            + self.post_norm.len()
            + self.q_norm.len()
            + self.k_norm.len()
            + self.a_base.len()
            + self.m_base.len()
            + self.a_proj.vram_bytes()
            + self.m_proj.vram_bytes()
            + self.q_proj.vram_bytes()
            + self.k_proj.vram_bytes()
            + self.v_proj.vram_bytes()
            + self.o_proj.vram_bytes()
            + self.gate_up_proj.vram_bytes()
            + self.down_proj.vram_bytes()
    }
}

pub struct DFlash2Drafter {
    kernels: DFlash2Kernels,
    fc: DfLinear,        // [4096, 20480]
    hnorm: DeviceBuf,    // [4096] bf16
    norm: DeviceBuf,     // [4096] bf16
    hproj: DeviceBuf,    // [256, 4096] bf16 (exact!)
    layers: Vec<DfLayer>,
    // Host-resident selector codebooks (0 MiB VRAM!)
    pred_book: Vec<u16>, // [VOCAB, 256] bf16
    succ_book: Vec<u16>, // [VOCAB, 256] bf16
    // Single-stream sliding window KV ring: [5, 2048, 1024] bf16 each (40 MiB total)
    ring_k: DeviceBuf,
    ring_v: DeviceBuf,
    pub from_pos: i64,
    // Context ingest scratch (up to 16 rows per chunk)
    max_ingest: usize,
    c1: DeviceBuf,  // [max_ingest, 4096] bf16
    ctx: DeviceBuf, // [max_ingest, 4096] bf16
    ck: DeviceBuf,  // [max_ingest, 1024] bf16
    cv: DeviceBuf,  // [max_ingest, 1024] bf16
    // Block pass scratch (M = 8)
    h_emb: PinnedHostBuf, // [8, 4096] bf16
    x: DeviceBuf,         // [8, 4096] bf16
    y: DeviceBuf,         // [8, 4096] bf16
    y2: DeviceBuf,        // [8, 4096] bf16
    dyn_w: DeviceBuf,     // [8, 1024] bf16
    q: DeviceBuf,         // [8, 4096] bf16
    kb: DeviceBuf,        // [8, 1024] bf16
    vb: DeviceBuf,        // [8, 1024] bf16
    att: DeviceBuf,       // [8, 4096] bf16
    a_buf: DeviceBuf,     // [8, 4096] bf16
    mlp_gu: DeviceBuf,    // [8, 24576] bf16 (fused [gate | up])
    mlp_act: DeviceBuf,   // [8, 12288] bf16
    hn: DeviceBuf,        // [8, 4096] bf16
    g_dev: DeviceBuf,     // [7, 256] f32
    logits: DeviceBuf,    // [7, VOCAB] f32
    pm: DeviceBuf,        // [8, 32, 16] f32
    pl: DeviceBuf,        // [8, 32, 16] f32
    pacc: DeviceBuf,      // [8, 32, 16, 128] f32
    pv: DeviceBuf,        // [7, 64, 16] f32
    pi: DeviceBuf,        // [7, 64, 16] i32
    d_top_ids: DeviceBuf, // [7, 16] i32
    d_top_vals: DeviceBuf,// [7, 16] f32
    h_top_ids: PinnedHostBuf,  // [7, 16] i32
    h_top_vals: PinnedHostBuf, // [7, 16] f32
    h_g: PinnedHostBuf,        // [7, 256] f32
    pub vram_bytes: usize,
    pub load_secs: f64,
}

impl DFlash2Drafter {
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &Stream,
        lin: &LinearKernels,
        drafter_dir: &Path,
        pack_q4: bool,
        pack_q3: bool,
        max_ingest: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let t0 = Instant::now();
        let kernels = DFlash2Kernels::load(ctx)?;
        let st = SafetensorsIndex::open_dir(drafter_dir)?;

        // Reusable staging buffer: largest linear is fused gate_up [24576, 4096] BF16 = 192 MiB
        let max_stage = (DF_D * (5 * DF_D)).max(2 * DF_FF * DF_D) * 2;
        let mut h_stage = vec![0u8; max_stage];
        let d_stage = ctx.alloc_device(max_stage)?;

        let load_raw = |name: &str, hb: &mut Vec<u8>| -> Result<DeviceBuf, Box<dyn Error>> {
            let loc = st.get(name)?;
            if hb.len() < loc.nbytes {
                hb.resize(loc.nbytes, 0);
            }
            st.read_into(name, &mut hb[..loc.nbytes])?;
            let d = ctx.alloc_device(loc.nbytes)?;
            unsafe {
                stream.htod_async(d.ptr(), hb.as_ptr(), loc.nbytes)?;
            }
            stream.synchronize()?;
            Ok(d)
        };

        let load_lin = |name: &str, n: usize, k: usize, as_q4: bool, as_q3: bool, hb: &mut Vec<u8>| -> Result<DfLinear, Box<dyn Error>> {
            let expected = n * k * 2;
            st.read_into(name, &mut hb[..expected])?;
            if !as_q4 && !as_q3 {
                let d = ctx.alloc_device(expected)?;
                unsafe {
                    stream.htod_async(d.ptr(), hb.as_ptr(), expected)?;
                }
                stream.synchronize()?;
                return Ok(DfLinear::Bf16 { buf: d, n, k });
            }
            unsafe {
                stream.htod_async(d_stage.ptr(), hb.as_ptr(), expected)?;
            }
            let out = if as_q3 {
                DfLinear::Q3(lin.pack_bf16_to_q3(ctx, stream, d_stage.ptr(), n, k)?)
            } else {
                DfLinear::Q4(lin.pack_bf16_to_q4(ctx, stream, d_stage.ptr(), n, k)?)
            };
            stream.synchronize()?;
            Ok(out)
        };

        let fc = load_lin("fc.weight", DF_D, 5 * DF_D, pack_q4, pack_q3, &mut h_stage)?;
        let hnorm = load_raw("hidden_norm.weight", &mut h_stage)?;
        let norm = load_raw("norm.weight", &mut h_stage)?;
        let hproj = load_raw("candidate_selector.hidden_projection.weight", &mut h_stage)?;

        let mut layers = Vec::with_capacity(DF_LAYERS);
        let mut w_vram = fc.vram_bytes() + hnorm.len() + norm.len() + hproj.len();
        for i in 0..DF_LAYERS {
            let half_gu = DF_FF * DF_D * 2;
            st.read_into(&format!("layers.{i}.mlp.gate_proj.weight"), &mut h_stage[..half_gu])?;
            st.read_into(&format!("layers.{i}.mlp.up_proj.weight"), &mut h_stage[half_gu..2 * half_gu])?;
            let gate_up_proj = if !pack_q4 && !pack_q3 {
                let d = ctx.alloc_device(2 * half_gu)?;
                unsafe {
                    stream.htod_async(d.ptr(), h_stage.as_ptr(), 2 * half_gu)?;
                }
                stream.synchronize()?;
                DfLinear::Bf16 { buf: d, n: 2 * DF_FF, k: DF_D }
            } else {
                unsafe {
                    stream.htod_async(d_stage.ptr(), h_stage.as_ptr(), 2 * half_gu)?;
                }
                let out = if pack_q3 {
                    DfLinear::Q3(lin.pack_bf16_to_q3(ctx, stream, d_stage.ptr(), 2 * DF_FF, DF_D)?)
                } else {
                    DfLinear::Q4(lin.pack_bf16_to_q4(ctx, stream, d_stage.ptr(), 2 * DF_FF, DF_D)?)
                };
                stream.synchronize()?;
                out
            };
            let lw = DfLayer {
                in_norm: load_raw(&format!("layers.{i}.input_layernorm.weight"), &mut h_stage)?,
                post_norm: load_raw(&format!("layers.{i}.post_attention_layernorm.weight"), &mut h_stage)?,
                q_norm: load_raw(&format!("layers.{i}.self_attn.q_norm.weight"), &mut h_stage)?,
                k_norm: load_raw(&format!("layers.{i}.self_attn.k_norm.weight"), &mut h_stage)?,
                a_base: load_raw(&format!("layers.{i}.attention_conv.base_kernel"), &mut h_stage)?,
                m_base: load_raw(&format!("layers.{i}.mlp_conv.base_kernel"), &mut h_stage)?,
                a_proj: load_lin(
                    &format!("layers.{i}.attention_conv.kernel_projection.weight"),
                    DF_KVW,
                    DF_D,
                    pack_q4,
                    pack_q3,
                    &mut h_stage,
                )?,
                m_proj: load_lin(
                    &format!("layers.{i}.mlp_conv.kernel_projection.weight"),
                    DF_KVW,
                    DF_D,
                    pack_q4,
                    pack_q3,
                    &mut h_stage,
                )?,
                q_proj: load_lin(&format!("layers.{i}.self_attn.q_proj.weight"), DF_D, DF_D, pack_q4, pack_q3, &mut h_stage)?,
                k_proj: load_lin(&format!("layers.{i}.self_attn.k_proj.weight"), DF_KVW, DF_D, pack_q4, pack_q3, &mut h_stage)?,
                v_proj: load_lin(&format!("layers.{i}.self_attn.v_proj.weight"), DF_KVW, DF_D, pack_q4, pack_q3, &mut h_stage)?,
                o_proj: load_lin(&format!("layers.{i}.self_attn.o_proj.weight"), DF_D, DF_D, pack_q4, pack_q3, &mut h_stage)?,
                gate_up_proj,
                down_proj: load_lin(&format!("layers.{i}.mlp.down_proj.weight"), DF_D, DF_FF, pack_q4, pack_q3, &mut h_stage)?,
            };
            w_vram += lw.vram_bytes();
            layers.push(lw);
        }

        drop(d_stage);
        drop(h_stage);

        // Load selector codebooks into normal Host RAM (0 MiB VRAM!)
        let load_u16_vec = |name: &str| -> Result<Vec<u16>, Box<dyn Error>> {
            let raw = st.read_vec(name)?;
            Ok(raw
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect())
        };
        let pred_book = load_u16_vec("candidate_selector.predecessor_codebook")?;
        let succ_book = load_u16_vec("candidate_selector.successor_codebook")?;

        // KV ring + scratch
        let ring_bytes = DF_LAYERS * DF_WIN * DF_KVW * 2;
        let ring_k = ctx.alloc_device(ring_bytes)?;
        let ring_v = ctx.alloc_device(ring_bytes)?;

        let c1 = ctx.alloc_device(max_ingest * DF_D * 2)?;
        let ctx_b = ctx.alloc_device(max_ingest * DF_D * 2)?;
        let ck = ctx.alloc_device(max_ingest * DF_KVW * 2)?;
        let cv = ctx.alloc_device(max_ingest * DF_KVW * 2)?;

        let h_emb = ctx.alloc_pinned(DF_BLOCK * DF_D * 2)?;
        let x = ctx.alloc_device(DF_BLOCK * DF_D * 2)?;
        let y = ctx.alloc_device(DF_BLOCK * DF_D * 2)?;
        let y2 = ctx.alloc_device(DF_BLOCK * DF_D * 2)?;
        let dyn_w = ctx.alloc_device(DF_BLOCK * DF_KVW * 2)?;
        let q = ctx.alloc_device(DF_BLOCK * DF_D * 2)?;
        let kb = ctx.alloc_device(DF_BLOCK * DF_KVW * 2)?;
        let vb = ctx.alloc_device(DF_BLOCK * DF_KVW * 2)?;
        let att = ctx.alloc_device(DF_BLOCK * DF_D * 2)?;
        let a_buf = ctx.alloc_device(DF_BLOCK * DF_D * 2)?;
        let mlp_gu = ctx.alloc_device(DF_BLOCK * 2 * DF_FF * 2)?;
        let mlp_act = ctx.alloc_device(DF_BLOCK * DF_FF * 2)?;
        let hn = ctx.alloc_device(DF_BLOCK * DF_D * 2)?;

        let n_draft = DF_BLOCK - 1;
        let g_dev = ctx.alloc_device(n_draft * DF_RANK * 4)?;
        let logits = ctx.alloc_device(n_draft * DF_VOCAB * 4)?;
        let pm = ctx.alloc_device(DF_BLOCK * 32 * 16 * 4)?;
        let pl = ctx.alloc_device(DF_BLOCK * 32 * 16 * 4)?;
        let pacc = ctx.alloc_device(DF_BLOCK * 32 * 16 * 128 * 4)?;
        let pv = ctx.alloc_device(n_draft * 64 * DF_TOPK * 4)?;
        let pi = ctx.alloc_device(n_draft * 64 * DF_TOPK * 4)?;
        let d_top_ids = ctx.alloc_device(n_draft * DF_TOPK * 4)?;
        let d_top_vals = ctx.alloc_device(n_draft * DF_TOPK * 4)?;
        let h_top_ids = ctx.alloc_pinned(n_draft * DF_TOPK * 4)?;
        let h_top_vals = ctx.alloc_pinned(n_draft * DF_TOPK * 4)?;
        let h_g = ctx.alloc_pinned(n_draft * DF_RANK * 4)?;

        let vram_bytes = w_vram
            + ring_k.len()
            + ring_v.len()
            + c1.len()
            + ctx_b.len()
            + ck.len()
            + cv.len()
            + x.len()
            + y.len()
            + y2.len()
            + dyn_w.len()
            + q.len()
            + kb.len()
            + vb.len()
            + att.len()
            + a_buf.len()
            + mlp_gu.len()
            + mlp_act.len()
            + hn.len()
            + g_dev.len()
            + logits.len()
            + pm.len()
            + pl.len()
            + pacc.len()
            + pv.len()
            + pi.len()
            + d_top_ids.len()
            + d_top_vals.len();

        let mut dr = Self {
            kernels,
            fc,
            hnorm,
            norm,
            hproj,
            layers,
            pred_book,
            succ_book,
            ring_k,
            ring_v,
            from_pos: 0,
            max_ingest,
            c1,
            ctx: ctx_b,
            ck,
            cv,
            h_emb,
            x,
            y,
            y2,
            dyn_w,
            q,
            kb,
            vb,
            att,
            a_buf,
            mlp_gu,
            mlp_act,
            hn,
            g_dev,
            logits,
            pm,
            pl,
            pacc,
            pv,
            pi,
            d_top_ids,
            d_top_vals,
            h_top_ids,
            h_top_vals,
            h_g,
            vram_bytes,
            load_secs: t0.elapsed().as_secs_f64(),
        };
        dr.reset(stream)?;
        Ok(dr)
    }

    pub fn reset(&mut self, stream: &Stream) -> m3tis_cuda::Result<()> {
        self.from_pos = 0;
        stream.memset_d32_async(self.ring_k.ptr(), 0, self.ring_k.len() / 4)?;
        stream.memset_d32_async(self.ring_v.ptr(), 0, self.ring_v.len() / 4)?;
        Ok(())
    }

    /// Snapshot the drafter's sliding-window KV rings into Host RAM for multi-turn Prefix KV Caching.
    pub fn snapshot_rings(&self, stream: &Stream) -> m3tis_cuda::Result<(Vec<u8>, Vec<u8>, i64)> {
        let mut rk = vec![0u8; self.ring_k.len()];
        let mut rv = vec![0u8; self.ring_v.len()];
        unsafe {
            stream.dtoh_async(rk.as_mut_ptr(), self.ring_k.ptr(), rk.len())?;
            stream.dtoh_async(rv.as_mut_ptr(), self.ring_v.ptr(), rv.len())?;
        }
        stream.synchronize()?;
        Ok((rk, rv, self.from_pos))
    }

    /// Restore the drafter's sliding-window KV rings from a Host RAM prefix snapshot.
    pub fn restore_rings(&mut self, stream: &Stream, rk: &[u8], rv: &[u8], from_pos: i64) -> m3tis_cuda::Result<()> {
        self.from_pos = from_pos;
        unsafe {
            stream.htod_async(self.ring_k.ptr(), rk.as_ptr(), rk.len())?;
            stream.htod_async(self.ring_v.ptr(), rv.as_ptr(), rv.len())?;
        }
        Ok(())
    }

    /// Ingest `rows` target tap rows (`taps_ptr: [rows, 5 * 4096]` BF16) at positions `pos0 .. pos0 + rows`
    /// into all 5 drafter sliding-window KV rings.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest(
        &self,
        stream: &Stream,
        lin: &LinearKernels,
        glm: &GlmOpsKernels,
        taps_ptr: CuDevicePtr,
        pos0: usize,
        rows: usize,
    ) -> m3tis_cuda::Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let tap_row_bytes = 5 * DF_D * 2;
        let layer_ring_bytes = DF_WIN * DF_KVW * 2;
        let mut off = 0usize;
        while off < rows {
            let n = (rows - off).min(self.max_ingest);
            let src = taps_ptr + (off * tap_row_bytes) as u64;
            self.fc.apply(lin, stream, src, 5 * DF_D, self.c1.ptr(), DF_D, n, false)?;
            glm.rmsnorm_bf16(stream, self.c1.ptr(), DF_D, self.hnorm.ptr(), self.ctx.ptr(), DF_D, n, DF_D, DF_EPS)?;
            for (li, lw) in self.layers.iter().enumerate() {
                lw.k_proj.apply(lin, stream, self.ctx.ptr(), DF_D, self.ck.ptr(), DF_KVW, n, false)?;
                lw.v_proj.apply(lin, stream, self.ctx.ptr(), DF_D, self.cv.ptr(), DF_KVW, n, false)?;
                self.kernels.ctx_kv(
                    stream,
                    self.ck.ptr(),
                    self.cv.ptr(),
                    lw.k_norm.ptr(),
                    self.ring_k.at(li * layer_ring_bytes),
                    self.ring_v.at(li * layer_ring_bytes),
                    (pos0 + off) as i64,
                    DF_EPS,
                    DF_THETA,
                    n,
                )?;
            }
            off += n;
        }
        Ok(())
    }

    /// Propose up to `max_drafts` (`1..=7`) candidate tokens starting after `anchor` at `pos`,
    /// stopping early if cumulative selector confidence drops below `conf_threshold` (`0.0` = always propose `max_drafts`).
    #[allow(clippy::too_many_arguments)]
    pub fn propose<F>(
        &mut self,
        stream: &Stream,
        lin: &LinearKernels,
        glm: &GlmOpsKernels,
        embed_host: &[u8],
        apply_lm_head: F,
        anchor: i32,
        pos: usize,
        max_drafts: usize,
        conf_threshold: f64,
    ) -> Result<(Vec<i32>, Vec<f32>), Box<dyn Error>>
    where
        F: FnOnce(&Stream, CuDevicePtr, CuDevicePtr, usize) -> m3tis_cuda::Result<()>,
    {
        let max_d = max_drafts.clamp(1, DF_BLOCK - 1);
        let row_bytes = DF_D * 2;
        let emb_dst = self.h_emb.as_mut_slice();
        for t in 0..DF_BLOCK {
            let tok = if t == 0 { anchor } else { DF_MASK_TOKEN } as usize;
            let s_off = tok * row_bytes;
            let d_off = t * row_bytes;
            emb_dst[d_off..d_off + row_bytes].copy_from_slice(&embed_host[s_off..s_off + row_bytes]);
        }
        unsafe {
            stream.htod_async(self.x.ptr(), self.h_emb.as_ptr(), DF_BLOCK * row_bytes)?;
        }

        let layer_ring_bytes = DF_WIN * DF_KVW * 2;
        for (li, lw) in self.layers.iter().enumerate() {
            // 1. Attention wrapped in dynamic 2-tap conv
            glm.rmsnorm_bf16(stream, self.x.ptr(), DF_D, lw.in_norm.ptr(), self.y.ptr(), DF_D, DF_BLOCK, DF_D, DF_EPS)?;
            lw.a_proj.apply(lin, stream, self.y.ptr(), DF_D, self.dyn_w.ptr(), DF_KVW, DF_BLOCK, false)?;
            self.kernels.conv(stream, self.y.ptr(), self.dyn_w.ptr(), 0, lw.a_base.ptr(), self.y2.ptr(), false, DF_BLOCK)?;
            lw.q_proj.apply(lin, stream, self.y2.ptr(), DF_D, self.q.ptr(), DF_D, DF_BLOCK, false)?;
            lw.k_proj.apply(lin, stream, self.y2.ptr(), DF_D, self.kb.ptr(), DF_KVW, DF_BLOCK, false)?;
            lw.v_proj.apply(lin, stream, self.y2.ptr(), DF_D, self.vb.ptr(), DF_KVW, DF_BLOCK, false)?;
            self.kernels.block_qk_rope(
                stream,
                self.q.ptr(),
                self.kb.ptr(),
                lw.q_norm.ptr(),
                lw.k_norm.ptr(),
                pos as i64,
                DF_EPS,
                DF_THETA,
                self.q.ptr(),
                self.kb.ptr(),
                DF_BLOCK,
            )?;
            self.kernels.attn(
                stream,
                self.q.ptr(),
                self.kb.ptr(),
                self.vb.ptr(),
                self.ring_k.at(li * layer_ring_bytes),
                self.ring_v.at(li * layer_ring_bytes),
                pos as i64,
                self.from_pos,
                DF_SCALE,
                self.pm.ptr(),
                self.pl.ptr(),
                self.pacc.ptr(),
                self.att.ptr(),
            )?;
            lw.o_proj.apply(lin, stream, self.att.ptr(), DF_D, self.a_buf.ptr(), DF_D, DF_BLOCK, false)?;
            self.kernels.conv(
                stream,
                self.a_buf.ptr(),
                self.dyn_w.ptr(),
                1,
                lw.a_base.at(2 * DF_D * 2),
                self.x.ptr(),
                true,
                DF_BLOCK,
            )?;

            // 2. MLP wrapped in dynamic 2-tap conv (Qwen3 drafter uses UNCLAMPED SwiGLU, limit = 0.0!)
            glm.rmsnorm_bf16(stream, self.x.ptr(), DF_D, lw.post_norm.ptr(), self.y.ptr(), DF_D, DF_BLOCK, DF_D, DF_EPS)?;
            lw.m_proj.apply(lin, stream, self.y.ptr(), DF_D, self.dyn_w.ptr(), DF_KVW, DF_BLOCK, false)?;
            self.kernels.conv(stream, self.y.ptr(), self.dyn_w.ptr(), 0, lw.m_base.ptr(), self.y2.ptr(), false, DF_BLOCK)?;
            lw.gate_up_proj.apply(lin, stream, self.y2.ptr(), DF_D, self.mlp_gu.ptr(), 2 * DF_FF, DF_BLOCK, false)?;
            lin.swiglu_split_bf16(stream, self.mlp_gu.ptr(), self.mlp_act.ptr(), DF_BLOCK, DF_FF, 0.0)?;
            lw.down_proj.apply(lin, stream, self.mlp_act.ptr(), DF_FF, self.a_buf.ptr(), DF_D, DF_BLOCK, false)?;
            self.kernels.conv(
                stream,
                self.a_buf.ptr(),
                self.dyn_w.ptr(),
                1,
                lw.m_base.at(2 * DF_D * 2),
                self.x.ptr(),
                true,
                DF_BLOCK,
            )?;
        }

        // 3. Final norm + shared target lm_head + top-16 + selector hidden projection (only rows 1..=max_d!)
        let n_draft = max_d;
        glm.rmsnorm_bf16(stream, self.x.ptr(), DF_D, self.norm.ptr(), self.hn.ptr(), DF_D, 1 + n_draft, DF_D, DF_EPS)?;
        let hn_drafts = self.hn.at(DF_D * 2); // rows 1..8
        apply_lm_head(stream, hn_drafts, self.logits.ptr(), n_draft)?;
        self.kernels.top16_f32(
            stream,
            self.logits.ptr(),
            DF_VOCAB,
            n_draft,
            self.pv.ptr(),
            self.pi.ptr(),
            self.d_top_ids.ptr(),
            self.d_top_vals.ptr(),
        )?;
        lin.linear_bf16(
            stream,
            hn_drafts,
            DF_D,
            self.hproj.ptr(),
            self.g_dev.ptr(),
            DF_RANK,
            n_draft,
            DF_RANK,
            DF_D,
            true,
        )?;
        unsafe {
            stream.dtoh_async(self.h_top_ids.as_mut_ptr(), self.d_top_ids.ptr(), n_draft * DF_TOPK * 4)?;
            stream.dtoh_async(self.h_top_vals.as_mut_ptr(), self.d_top_vals.ptr(), n_draft * DF_TOPK * 4)?;
            stream.dtoh_async(self.h_g.as_mut_ptr(), self.g_dev.ptr(), n_draft * DF_RANK * 4)?;
        }
        stream.synchronize()?;

        // 4. Sequential codebook candidate selector on host (~5 us!)
        let ids_slice = &self.h_top_ids.as_i32_mut()[..n_draft * DF_TOPK];
        let vals_slice = &self.h_top_vals.as_f32_mut()[..n_draft * DF_TOPK];
        let g_slice = &self.h_g.as_f32_mut()[..n_draft * DF_RANK];

        #[inline]
        fn bf(u: u16) -> f32 {
            f32::from_bits((u as u32) << 16)
        }

        let mut drafts = Vec::with_capacity(max_d);
        let mut row_cum_probs = Vec::with_capacity(1 + max_d);
        row_cum_probs.push(1.0f32);
        let mut prev = anchor as usize;
        let mut cum_prob = 1.0f64;
        let mut pg = [0.0f32; DF_RANK];
        let mut scores = [0.0f32; DF_TOPK];

        for t in 0..max_d {
            let g_row = &g_slice[t * DF_RANK..(t + 1) * DF_RANK];
            let p_row = &self.pred_book[prev * DF_RANK..(prev + 1) * DF_RANK];
            for j in 0..DF_RANK {
                pg[j] = bf(p_row[j]) * g_row[j];
            }
            let c_ids = &ids_slice[t * DF_TOPK..(t + 1) * DF_TOPK];
            let c_vals = &vals_slice[t * DF_TOPK..(t + 1) * DF_TOPK];
            let mut best = 0usize;
            for m in 0..DF_TOPK {
                let cid = c_ids[m] as usize;
                let s_row = &self.succ_book[cid * DF_RANK..(cid + 1) * DF_RANK];
                let mut dot = 0.0f32;
                for j in 0..DF_RANK {
                    dot += pg[j] * bf(s_row[j]);
                }
                scores[m] = c_vals[m] + dot;
                if scores[m] > scores[best] {
                    best = m;
                }
            }
            let mut z = 0.0f64;
            for &sc in &scores {
                z += ((sc - scores[best]) as f64).exp();
            }
            let prob = 1.0 / z;
            let chosen = c_ids[best];
            // Always propose at least 1 draft; truncate tail when cumulative confidence drops below threshold!
            if t > 0 && conf_threshold > 0.0 && cum_prob * prob < conf_threshold {
                break;
            }
            cum_prob *= prob;
            drafts.push(chosen);
            row_cum_probs.push(cum_prob as f32);
            prev = chosen as usize;
        }
        Ok((drafts, row_cum_probs))
    }

    /// Milestone 8: Zero-Extra-GPU-Pass 2-Branch Speculative Tree Proposal (`0.005 ms` extra CPU selector!).
    /// Runs ONE standard in-distribution DFlash2 block pass, decodes the primary path `primary`, picks the
    /// highest expected-hit-probability uncertain fork `f = argmax_t (P(reach t) * p_2(t))`, and decodes a
    /// coherent alternative continuation `branch_b` directly from the SAME pass's `top-16` + bigram codebook!
    #[allow(clippy::too_many_arguments)]
    pub fn propose_tree<F>(
        &mut self,
        stream: &Stream,
        lin: &LinearKernels,
        glm: &GlmOpsKernels,
        embed_host: &[u8],
        apply_lm_head: F,
        anchor: i32,
        pos: usize,
        max_drafts: usize,
        conf_threshold: f64,
        max_verify_rows: usize,
    ) -> Result<DraftTree, Box<dyn Error>>
    where
        F: FnOnce(&Stream, CuDevicePtr, CuDevicePtr, usize) -> m3tis_cuda::Result<()>,
    {
        let max_d = max_drafts.clamp(1, DF_BLOCK - 1);
        let row_bytes = DF_D * 2;
        let emb_dst = self.h_emb.as_mut_slice();
        for t in 0..DF_BLOCK {
            let tok = if t == 0 { anchor } else { DF_MASK_TOKEN } as usize;
            let s_off = tok * row_bytes;
            let d_off = t * row_bytes;
            emb_dst[d_off..d_off + row_bytes].copy_from_slice(&embed_host[s_off..s_off + row_bytes]);
        }
        unsafe {
            stream.htod_async(self.x.ptr(), self.h_emb.as_ptr(), DF_BLOCK * row_bytes)?;
        }

        let layer_ring_bytes = DF_WIN * DF_KVW * 2;
        for (li, lw) in self.layers.iter().enumerate() {
            glm.rmsnorm_bf16(stream, self.x.ptr(), DF_D, lw.in_norm.ptr(), self.y.ptr(), DF_D, DF_BLOCK, DF_D, DF_EPS)?;
            lw.a_proj.apply(lin, stream, self.y.ptr(), DF_D, self.dyn_w.ptr(), DF_KVW, DF_BLOCK, false)?;
            self.kernels.conv(stream, self.y.ptr(), self.dyn_w.ptr(), 0, lw.a_base.ptr(), self.y2.ptr(), false, DF_BLOCK)?;
            lw.q_proj.apply(lin, stream, self.y2.ptr(), DF_D, self.q.ptr(), DF_D, DF_BLOCK, false)?;
            lw.k_proj.apply(lin, stream, self.y2.ptr(), DF_D, self.kb.ptr(), DF_KVW, DF_BLOCK, false)?;
            lw.v_proj.apply(lin, stream, self.y2.ptr(), DF_D, self.vb.ptr(), DF_KVW, DF_BLOCK, false)?;
            self.kernels.block_qk_rope(
                stream,
                self.q.ptr(),
                self.kb.ptr(),
                lw.q_norm.ptr(),
                lw.k_norm.ptr(),
                pos as i64,
                DF_EPS,
                DF_THETA,
                self.q.ptr(),
                self.kb.ptr(),
                DF_BLOCK,
            )?;
            self.kernels.attn(
                stream,
                self.q.ptr(),
                self.kb.ptr(),
                self.vb.ptr(),
                self.ring_k.at(li * layer_ring_bytes),
                self.ring_v.at(li * layer_ring_bytes),
                pos as i64,
                self.from_pos,
                DF_SCALE,
                self.pm.ptr(),
                self.pl.ptr(),
                self.pacc.ptr(),
                self.att.ptr(),
            )?;
            lw.o_proj.apply(lin, stream, self.att.ptr(), DF_D, self.a_buf.ptr(), DF_D, DF_BLOCK, false)?;
            self.kernels.conv(
                stream,
                self.a_buf.ptr(),
                self.dyn_w.ptr(),
                1,
                lw.a_base.at(2 * DF_D * 2),
                self.x.ptr(),
                true,
                DF_BLOCK,
            )?;

            glm.rmsnorm_bf16(stream, self.x.ptr(), DF_D, lw.post_norm.ptr(), self.y.ptr(), DF_D, DF_BLOCK, DF_D, DF_EPS)?;
            lw.m_proj.apply(lin, stream, self.y.ptr(), DF_D, self.dyn_w.ptr(), DF_KVW, DF_BLOCK, false)?;
            self.kernels.conv(stream, self.y.ptr(), self.dyn_w.ptr(), 0, lw.m_base.ptr(), self.y2.ptr(), false, DF_BLOCK)?;
            lw.gate_up_proj.apply(lin, stream, self.y2.ptr(), DF_D, self.mlp_gu.ptr(), 2 * DF_FF, DF_BLOCK, false)?;
            lin.swiglu_split_bf16(stream, self.mlp_gu.ptr(), self.mlp_act.ptr(), DF_BLOCK, DF_FF, 0.0)?;
            lw.down_proj.apply(lin, stream, self.mlp_act.ptr(), DF_FF, self.a_buf.ptr(), DF_D, DF_BLOCK, false)?;
            self.kernels.conv(
                stream,
                self.a_buf.ptr(),
                self.dyn_w.ptr(),
                1,
                lw.m_base.at(2 * DF_D * 2),
                self.x.ptr(),
                true,
                DF_BLOCK,
            )?;
        }

        let n_draft = max_d;
        glm.rmsnorm_bf16(stream, self.x.ptr(), DF_D, self.norm.ptr(), self.hn.ptr(), DF_D, 1 + n_draft, DF_D, DF_EPS)?;
        let hn_drafts = self.hn.at(DF_D * 2);
        apply_lm_head(stream, hn_drafts, self.logits.ptr(), n_draft)?;
        self.kernels.top16_f32(
            stream,
            self.logits.ptr(),
            DF_VOCAB,
            n_draft,
            self.pv.ptr(),
            self.pi.ptr(),
            self.d_top_ids.ptr(),
            self.d_top_vals.ptr(),
        )?;
        lin.linear_bf16(
            stream,
            hn_drafts,
            DF_D,
            self.hproj.ptr(),
            self.g_dev.ptr(),
            DF_RANK,
            n_draft,
            DF_RANK,
            DF_D,
            true,
        )?;
        unsafe {
            stream.dtoh_async(self.h_top_ids.as_mut_ptr(), self.d_top_ids.ptr(), n_draft * DF_TOPK * 4)?;
            stream.dtoh_async(self.h_top_vals.as_mut_ptr(), self.d_top_vals.ptr(), n_draft * DF_TOPK * 4)?;
            stream.dtoh_async(self.h_g.as_mut_ptr(), self.g_dev.ptr(), n_draft * DF_RANK * 4)?;
        }
        stream.synchronize()?;

        let ids_slice = &self.h_top_ids.as_i32_mut()[..n_draft * DF_TOPK];
        let vals_slice = &self.h_top_vals.as_f32_mut()[..n_draft * DF_TOPK];
        let g_slice = &self.h_g.as_f32_mut()[..n_draft * DF_RANK];

        #[inline]
        fn bf(u: u16) -> f32 {
            f32::from_bits((u as u32) << 16)
        }

        let mut pg = [0.0f32; DF_RANK];
        let mut scores = [0.0f32; DF_TOPK];
        let score_slot = |t: usize, prev: usize, pg: &mut [f32; DF_RANK], scores: &mut [f32; DF_TOPK]| -> (i32, f64, i32, f64) {
            let g_row = &g_slice[t * DF_RANK..(t + 1) * DF_RANK];
            let p_row = &self.pred_book[prev * DF_RANK..(prev + 1) * DF_RANK];
            for j in 0..DF_RANK {
                pg[j] = bf(p_row[j]) * g_row[j];
            }
            let c_ids = &ids_slice[t * DF_TOPK..(t + 1) * DF_TOPK];
            let c_vals = &vals_slice[t * DF_TOPK..(t + 1) * DF_TOPK];
            let mut best = 0usize;
            let mut second = 1usize;
            for m in 0..DF_TOPK {
                let cid = c_ids[m] as usize;
                let s_row = &self.succ_book[cid * DF_RANK..(cid + 1) * DF_RANK];
                let mut dot = 0.0f32;
                for j in 0..DF_RANK {
                    dot += pg[j] * bf(s_row[j]);
                }
                scores[m] = c_vals[m] + dot;
            }
            if scores[1] > scores[0] {
                best = 1;
                second = 0;
            }
            for m in 2..DF_TOPK {
                if scores[m] > scores[best] {
                    second = best;
                    best = m;
                } else if scores[m] > scores[second] {
                    second = m;
                }
            }
            let mut z = 0.0f64;
            for &sc in scores.iter() {
                z += ((sc - scores[best]) as f64).exp();
            }
            let p1 = 1.0 / z;
            let p2 = ((scores[second] - scores[best]) as f64).exp() / z;
            (c_ids[best], p1, c_ids[second], p2)
        };

        // 1. Decode Primary Branch (Branch A) + per-row cumulative probabilities
        let mut primary = Vec::with_capacity(max_d);
        let mut row_cum_probs = Vec::with_capacity(1 + max_d);
        row_cum_probs.push(1.0f32);
        let mut prev = anchor as usize;
        let mut cum_prob = 1.0f64;
        let mut best_fork: Option<(f64, usize, i32)> = None; // (expected_hit_prob, fork_pos, alt_tok)

        for t in 0..max_d {
            let (tok1, p1, tok2, p2) = score_slot(t, prev, &mut pg, &mut scores);
            if t > 0 && conf_threshold > 0.0 && cum_prob * p1 < conf_threshold {
                break;
            }
            let hit_prob = cum_prob * p2;
            if tok2 != tok1 && best_fork.map_or(true, |(bs, _, _)| hit_prob > bs) {
                best_fork = Some((hit_prob, t, tok2));
            }
            cum_prob *= p1;
            primary.push(tok1);
            row_cum_probs.push(cum_prob as f32);
            prev = tok1 as usize;
        }

        // 2. Decode Branch B ONLY on genuine high-EV 2-way ties (capped to <=3 rows so verify stays fast!)
        let mut fork_pos = primary.len();
        let mut branch_b = Vec::new();
        let max_b_rows = max_verify_rows.saturating_sub(1 + primary.len()).min(3);
        if max_b_rows >= 1 {
            if let Some((hit_prob, fpos, alt_tok)) = best_fork {
                if hit_prob >= 0.18 {
                    fork_pos = fpos;
                    branch_b.push(alt_tok);
                    let mut prev_b = alt_tok as usize;
                    let mut cum_b = 1.0f64;
                    for t in (fpos + 1)..max_d {
                        if branch_b.len() >= max_b_rows {
                            break;
                        }
                        let (tok1_b, p1_b, _, _) = score_slot(t, prev_b, &mut pg, &mut scores);
                        if conf_threshold > 0.0 && cum_b * p1_b < (conf_threshold * 0.65).max(0.35) {
                            break;
                        }
                        cum_b *= p1_b;
                        branch_b.push(tok1_b);
                        prev_b = tok1_b as usize;
                    }
                }
            }
        }

        Ok(DraftTree {
            primary,
            row_cum_probs,
            fork_pos,
            branch_b,
        })
    }
}
