//! Clean-room GLM-5.3-Flash (320B-A18B EXL3) forward pass for M3TiS.
//!
//! Architecture highlights:
//! - 45 layers (`hidden=4096`, `vocab=154880`, `hc_mult=4`, `sinkhorn=20`, `swiglu_limit=10.0`).
//! - 34 KDA linear-attention layers (`l % 4 != 3`, 64 heads x 128 dim, 4-tap conv1d + delta-rule).
//! - 11 DSA/MLA latent-attention layers (`l % 4 == 3`, 64 heads, `q_lora=1536`, `kv_lora=512`, `qk_nope=256`, `dv=256`).
//! - 3 Dense MLP layers (`0..3`, `inter=12288`) + 42 MoE layers (`3..45`, `288` EXL3 routed experts, `top_k=8`, `shared_inter=2048`).
//!
//! Memory engineering:
//! - `embed_tokens` (`1.18 GiB`) lives in pinned Host RAM (0 VRAM; 8 KiB HtoD per token).
//! - Large dense/attention/shared/`lm_head` matrices pack on-GPU to affine Q4 (`gs=64`) using a single
//!   reusable 136 MiB staging buffer, fitting the entire 320B non-routed backbone in ~5.1 GiB VRAM!
//! - Remaining VRAM is automatically handed to `TieredExpertPool` (~1,000–1,300 VRAM expert slots on RTX 3090),
//!   backed by Scan-Resistant 2Q/LFU Pinned RAM and dual-ring `io_uring` NVMe with Lever 2 speculative prefetch.

use crate::dflash2::{DFlash2Drafter, DraftTree, DF_BLOCK, tap_slot_for_layer};
use crate::prefix_disk::{fnv1a_u64, DiskPrefixPayload, DiskPrefixStore};
use m3tis_cache::cpu_exl3::{CpuExl3Engine, CpuExl3ExpertJob, MAX_CPU_JOB_ROWS};
use m3tis_cache::scales::{Exl3ScaleBank, PF_SCALE_SLOTS};
use m3tis_cache::{EvictionPolicy, PF_WAVE_SLOTS, PoolStats, TieredExpertPool};
use m3tis_cuda::exl3::{Exl3ExpertsView, Exl3MoeKernels, Exl3MoeScratch, Exl3PrefillScratch, InputDType};
use m3tis_cuda::glm_ops::GlmOpsKernels;
use m3tis_cuda::linear::{LinearKernels, Q3Weight, Q4Weight};
use m3tis_cuda::{
    CuDevicePtr, CudaContext, DeviceBuf, Event, PinnedHostBuf, RouterRemapKernels, Stream,
};
use m3tis_format::safetensors::SafetensorsIndex;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const PREFIX: &str = "model.language_model.";

#[derive(Debug, Clone, Copy)]
pub struct Glm53Config {
    pub num_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub hc_mult: usize,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: i32,
    pub rms_eps: f32,
    pub kda_heads: usize,
    pub kda_dim: usize,
    pub kda_lower: f32,
    pub mla_heads: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_nope: usize,
    pub dv: usize,
    pub dense_inter: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub routed_scale: f32,
    pub swiglu_limit: f32,
    pub eos_tokens: [i32; 3],
}

impl Default for Glm53Config {
    fn default() -> Self {
        Self {
            num_layers: 45,
            hidden: 4096,
            vocab: 154880,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_sinkhorn_iters: 20,
            rms_eps: 1e-5,
            kda_heads: 64,
            kda_dim: 128,
            kda_lower: -5.0,
            mla_heads: 64,
            q_lora: 1536,
            kv_lora: 512,
            qk_nope: 256,
            dv: 256,
            dense_inter: 12288,
            moe_inter: 2048,
            shared_inter: 2048,
            num_experts: 288,
            top_k: 8,
            routed_scale: 2.5,
            swiglu_limit: 10.0,
            eos_tokens: [154820, 154827, 154829],
        }
    }
}

impl Glm53Config {
    #[inline]
    pub fn is_mla(&self, layer: usize) -> bool {
        layer % 4 == 3
    }

    #[inline]
    pub fn is_moe(&self, layer: usize) -> bool {
        layer >= 3
    }

    #[inline]
    pub fn kda_layers(&self) -> usize {
        (0..self.num_layers).filter(|&l| !self.is_mla(l)).count()
    }

    #[inline]
    pub fn mla_layers(&self) -> usize {
        (0..self.num_layers).filter(|&l| self.is_mla(l)).count()
    }

    #[inline]
    pub fn moe_layers(&self) -> usize {
        (0..self.num_layers).filter(|&l| self.is_moe(l)).count()
    }

    #[inline]
    pub fn kda_conv_c(&self) -> usize {
        3 * self.kda_heads * self.kda_dim
    }

    #[inline]
    pub fn kda_p_stride(&self) -> usize {
        self.kda_conv_c() + 2 * self.kda_dim + self.kda_heads
    }

    #[inline]
    pub fn kda_b_off(&self) -> usize {
        self.kda_conv_c() + 2 * self.kda_dim
    }
}

#[derive(Debug, Clone)]
pub struct Glm53LoadOptions {
    /// Pack large dense/attention/shared linears to on-GPU affine Q4 (`gs=64`).
    pub pack_dense_q4: bool,
    /// Pack large dense/attention/shared linears to on-GPU affine Q3 (`gs=64`, 3.50 bpw -> frees ~940 MiB VRAM & -22% dense HBM!).
    pub pack_dense_q3: bool,
    /// Selective Q3: keep sensitive Dense Layers 0..2 + MLA q_a/kv_a + lm_head in Q4, pack MoE-layer KDA/MLA/shared in Q3 (~810 MiB VRAM freed, 0 syntax loss!).
    pub pack_dense_q3_moe: bool,
    /// Pack `lm_head` to on-GPU affine Q4 (`gs=64`) (saves 912 MiB VRAM).
    pub pack_head_q4: bool,
    /// Explicit Tier-1 VRAM budget for expert slabs (GiB); `None` auto-sizes leaving `vram_headroom_gib`.
    pub vram_gib: Option<f64>,
    /// VRAM headroom (GiB) kept free when auto-sizing expert VRAM slots.
    pub vram_headroom_gib: f64,
    /// Tier-2 Pinned Host RAM budget for expert slabs (GiB). Keep <= 20 GiB on 64 GiB hosts!
    pub ram_gib: f64,
    /// Eviction policy for Tier-2 RAM cache (`TwoQLfu` recommended).
    pub ram_policy: EvictionPolicy,
    /// Enable `io_uring` `SQPOLL` kernel polling if available.
    pub sqpoll: bool,
    /// Default EXL3 half-bit width (`6` = 3.0bpw).
    pub exl3_k2: i32,
    /// Max context window for MLA latent KV ring buffer.
    pub max_ctx: usize,
    /// Symmetric per-token INT8 latent KV cache (`516 B/tok` vs `1024 B/tok` -> 2x smaller KV VRAM & HBM BW!).
    pub kv_int8: bool,
    /// Max rows per forward chunk (prefill chunks automatically split to this size).
    pub max_chunk_rows: usize,
    /// Milestone 5B Expert-Major Streaming Prefill super-batch size (`0` = legacy chunk-16 prefill;
    /// default `1024` streams each layer's cold experts ONCE in sequential NVMe offset order with zero cache eviction!).
    pub prefill_batch: usize,
    /// Optional DFlash2 speculative drafter directory (`GLM-5.3-Flash-DFlash2`).
    pub drafter_dir: Option<PathBuf>,
    /// Pack DFlash2 drafter linears to Q4 (`gs=64`, ~650 MiB VRAM vs ~2.2 GiB BF16).
    pub pack_drafter_q4: bool,
    /// Pack DFlash2 drafter linears to Q3 (`gs=64`, ~505 MiB VRAM -> +20 VRAM experts & -22% draft HBM, 100% lossless!).
    pub pack_drafter_q3: bool,
    /// Number of host-RAM multi-turn prefix KV/KDA snapshots to retain (`0` disables prefix caching).
    pub prefix_cache_slots: usize,
    /// Optional persistent NVMe directory for cross-restart `M3TISPFX1` prefix snapshots (`None` = RAM-only).
    pub prefix_cache_dir: Option<PathBuf>,
    /// Maximum disk quota (GiB) for `prefix_cache_dir` before LRU eviction (default `8.0` GiB).
    pub prefix_cache_max_gib: f64,
    /// Boot-time VRAM Pre-Warming: stream top Zipf experts from `.m3tis` into VRAM at load (~2s) to eliminate cold start.
    pub prewarm_vram: bool,
    /// When `prewarm_vram` is true, also prefill the compact cold 2b VRAM slab (fills 100% of VRAM slots).
    pub prewarm_cold: bool,
    /// When `prewarm_vram` is true, also stream the next exclusive Zipf tier into Tier-2 Pinned RAM (0 duplication!).
    pub prewarm_ram: bool,
}

impl Default for Glm53LoadOptions {
    fn default() -> Self {
        Self {
            pack_dense_q4: true,
            pack_dense_q3: false,
            pack_dense_q3_moe: false,
            pack_head_q4: true,
            vram_gib: None,
            vram_headroom_gib: 0.50,
            ram_gib: 18.0,
            ram_policy: EvictionPolicy::TwoQLfu,
            sqpoll: false,
            exl3_k2: 6,
            max_ctx: 2048,
            kv_int8: false,
            max_chunk_rows: 64,
            prefill_batch: 2048,
            drafter_dir: None,
            pack_drafter_q4: true,
            pack_drafter_q3: false,
            prefix_cache_slots: 4,
            prefix_cache_dir: None,
            prefix_cache_max_gib: 8.0,
            prewarm_vram: false,
            prewarm_cold: true,
            prewarm_ram: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ForwardOpts {
    /// Lever 2 speculative next-layer prefetch count (`0` = off, `1..8` experts).
    pub spec_prefetch_p: usize,
    /// Lever 2 lookahead depth (`1` = layer L+1 only, `2` = layers L+1 and L+2).
    pub lookahead_depth: usize,
    /// Lever 4 DynTopK cumulative probability mass (`0.0` = exact top-8, `0.80` = ~5.6 avg experts).
    pub dyn_topk_mass: f32,
    /// Lever 4 DynTopK minimum kept experts (`4` recommended when `dyn_topk_mass > 0`).
    pub dyn_topk_min_k: usize,
    /// Residency-aware router tail tiebreak bonus for VRAM-resident experts (`0.0` = off; e.g. `0.03..0.06`).
    pub residency_bias: f32,
    /// Primary ranks `0..residency_strict_k` never apply `residency_bias`.
    pub residency_strict_k: usize,
    /// Never-Stall tail cap (`0` = fetch all `top_k`; `1..top_k` = only stall NVMe for ranks `< nvme_strict_k`,
    /// while using all VRAM/RAM hits for tail ranks and renormalizing skipped cold tail misses on GPU!).
    pub nvme_strict_k: usize,
    /// Entropy-adaptive Residency-Covered NVMe Mass Gate (`0.0` = disabled; e.g. `0.65..0.75` = only fetch
    /// cold experts from NVMe when VRAM+RAM residents cover `< nvme_mass` of this token's routed weight!).
    pub nvme_mass: f32,
    /// Speculative-Row NVMe Gating (`0` = strict `nvme_strict_k` on all verify rows; `1` = strict Rank-0 on
    /// committed Row 0 while speculative draft rows `r >= 1` always use VRAM+RAM Rank-0 and only stall NVMe
    /// if VRAM+RAM cover `< nvme_mass`!; `2` = strict on Rows 0..1, gated on `r >= 2`).
    pub nvme_tail_row: usize,
    /// 100% Lossless Mid-Forward Speculative Prefix Pruning (`0` = off; `R >= 1` = if any draft row `r >= R`
    /// would stall on a cold NVMe read at any layer, immediately truncate the verify batch to `rows = r`
    /// for the rest of the forward pass -> 0 cold-tail NVMe stalls, 0 wasted compute, 100% exact committed tokens!).
    pub nvme_prune_row: usize,
    /// Confidence-Gated 100% Lossless Mid-Forward NVMe Prefix Pruning (`0.0` = off; e.g. `0.35..0.55` = if any
    /// speculative tail row `r >= 2` with cumulative DFlash2 probability `< nvme_prune_prob` would stall on a
    /// cold NVMe read at any layer, immediately truncate `rows` to `r` -> keeps full 5-7 tok/rd on confident
    /// code while slashing wasted NVMe reads on uncertain speculative tails with ZERO quality loss!).
    pub nvme_prune_prob: f32,
    /// Self-Healing Zero-Sync Warp Streaming floor (`0.0` = off; e.g. `0.25` = run all 45 layers with 0 CPU-GPU
    /// syncs and 0 I/O stalls while asynchronously warming RAM->VRAM + NVMe->RAM in the background; if >= 6 layers
    /// drop below `zero_stall_mass` VRAM coverage, automatically triggers 1 sync heal step).
    pub zero_stall_mass: f32,
    /// Zero-NVMe Resident-Only Self-Speculation draft length (`0` = off, `1..=7` = 100% lossless exact self-spec!).
    pub self_spec_k: usize,
    /// Confidence gate for lossless self-speculation (`0.0` = always draft full `self_spec_k`; e.g. `0.70` stops
    /// drafting early when `DraftResident` top-1 softmax probability drops below threshold).
    pub self_spec_conf: f32,
    /// Hybrid CPU-GPU Tier-2 RAM Expert Offload cap (`0` = pure GPU, `1..=8` = execute up to N RAM-hit experts
    /// asynchronously on host AVX2+FMA+F16C threads concurrently with NVMe + RTX 3090, 0 PCIe / 0 VRAM eviction!).
    pub cpu_hybrid_max: usize,
    /// When true, offload RAM-hit experts to CPU even on 0-NVMe/1-RAM-hit layers; when false (default), uses
    /// cost-aware overlap (offloads when layer has >=1 cold NVMe miss OR >=2 RAM hits).
    pub cpu_hybrid_always: bool,
    /// Online Auto-Adaptive DFlash2 Controller: dynamically throttles drafts on low-acceptance creative prose
    /// (falling back to 35 tok/s serial while keeping DFlash2 KV synced!) and ramps back to full drafts on code/JSON.
    pub adaptive_draft: bool,
    /// Milestone 8: Zero-extra-GPU-pass 2-Branch DFlash2 Speculative Tree Verification (100% lossless greedy parity!).
    pub tree_verify: bool,
}

impl Default for ForwardOpts {
    fn default() -> Self {
        Self {
            spec_prefetch_p: 4,
            lookahead_depth: 1,
            dyn_topk_mass: 0.0,
            dyn_topk_min_k: 4,
            residency_bias: 0.0,
            residency_strict_k: 3,
            nvme_strict_k: 0,
            nvme_mass: 0.0,
            nvme_tail_row: 0,
            nvme_prune_row: 0,
            nvme_prune_prob: 0.0,
            zero_stall_mass: 0.0,
            self_spec_k: 0,
            self_spec_conf: 0.70,
            cpu_hybrid_max: 0,
            cpu_hybrid_always: false,
            adaptive_draft: false,
            tree_verify: true,
        }
    }
}

enum LinearWeight {
    Q4(Q4Weight),
    Q3(Q3Weight),
    Bf16 { buf: DeviceBuf, n: usize, k: usize },
}

impl LinearWeight {
    fn vram_bytes(&self) -> usize {
        match self {
            LinearWeight::Q4(q) => q.vram_bytes(),
            LinearWeight::Q3(q) => q.vram_bytes(),
            LinearWeight::Bf16 { buf, .. } => buf.len(),
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
            LinearWeight::Q4(q) => {
                lin.linear_q4(stream, x_ptr, x_stride, q, out_ptr, out_stride, rows, out_f32)
            }
            LinearWeight::Q3(q) => {
                lin.linear_q3(stream, x_ptr, x_stride, q, out_ptr, out_stride, rows, out_f32)
            }
            LinearWeight::Bf16 { buf, n, k } => lin.linear_bf16(
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

struct HcWeights {
    fn_w: DeviceBuf,  // [24, 16384] bf16
    base: DeviceBuf,  // [24] f32
    scale: DeviceBuf, // [3] f32
}

impl HcWeights {
    fn vram_bytes(&self) -> usize {
        self.fn_w.len() + self.base.len() + self.scale.len()
    }
}

struct KdaWeights {
    qkv_proj: LinearWeight, // fused [q; k; v] = [24576, 4096] (1 launch instead of 3!)
    fgb_proj: DeviceBuf,    // fused [f_a (128); g_a (128); b (64)] = [320, 4096] bf16
    f_b_proj: DeviceBuf,    // [8192, 128] bf16
    g_b_proj: DeviceBuf,    // [8192, 128] bf16
    o_proj: LinearWeight,   // [4096, 8192]
    conv_w: DeviceBuf,      // [24576, 4] f32
    a_log: DeviceBuf,       // [64] f32
    dt_bias: DeviceBuf,     // [8192] f32
    o_norm: DeviceBuf,      // [128] bf16
}

impl KdaWeights {
    fn vram_bytes(&self) -> usize {
        self.qkv_proj.vram_bytes()
            + self.fgb_proj.len()
            + self.f_b_proj.len()
            + self.g_b_proj.len()
            + self.o_proj.vram_bytes()
            + self.conv_w.len()
            + self.a_log.len()
            + self.dt_bias.len()
            + self.o_norm.len()
    }
}

pub const DSA_HEADS: usize = 32;
pub const DSA_DIM: usize = 128;
pub const DSA_KPOOL: usize = 4;
pub const DSA_TOPK_POOLS: usize = 512;
pub const DSA_WIDTH: usize = DSA_TOPK_POOLS * DSA_KPOOL + (DSA_KPOOL - 1); // 2051
pub const DSA_TOKEN_STRIDE: usize = DSA_TOPK_POOLS * DSA_KPOOL + 16; // 2064 (room for 3 trunk tail + Branch B tokens)

struct DsaIndexerWeights {
    wk: DeviceBuf,       // [128, 4096] bf16
    w_proj: DeviceBuf,   // [32, 4096] bf16
    gate: DeviceBuf,     // [128, 4096] bf16
    k_norm_w: DeviceBuf, // [128] bf16
    k_norm_b: DeviceBuf, // [128] bf16
    ape: DeviceBuf,      // [4, 128] bf16
    wq_b: LinearWeight,  // [4096, 1536] Q4 or bf16
}

impl DsaIndexerWeights {
    fn vram_bytes(&self) -> usize {
        self.wk.len()
            + self.w_proj.len()
            + self.gate.len()
            + self.k_norm_w.len()
            + self.k_norm_b.len()
            + self.ape.len()
            + self.wq_b.vram_bytes()
    }
}

struct MlaWeights {
    q_a_proj: LinearWeight,  // [1536, 4096] Q4 or bf16
    q_a_norm: DeviceBuf,     // [1536] bf16
    q_b_proj: LinearWeight,  // [16384, 1536]
    kv_a_proj: LinearWeight, // [512, 4096] Q4 or bf16
    kv_a_norm: DeviceBuf,    // [512] bf16
    wk: DeviceBuf,           // [64, 512, 256] bf16
    wv: DeviceBuf,           // [64, 256, 512] bf16
    o_proj: LinearWeight,    // [4096, 16384]
    indexer: Option<DsaIndexerWeights>,
}

impl MlaWeights {
    fn vram_bytes(&self) -> usize {
        self.q_a_proj.vram_bytes()
            + self.q_a_norm.len()
            + self.q_b_proj.vram_bytes()
            + self.kv_a_proj.vram_bytes()
            + self.kv_a_norm.len()
            + self.wk.len()
            + self.wv.len()
            + self.o_proj.vram_bytes()
            + self.indexer.as_ref().map_or(0, |ix| ix.vram_bytes())
    }
}

enum MixerWeights {
    Kda { idx: usize, w: KdaWeights },
    Mla { idx: usize, w: MlaWeights },
}

struct DenseMlpWeights {
    gate: LinearWeight, // [12288, 4096]
    up: LinearWeight,   // [12288, 4096]
    down: LinearWeight, // [4096, 12288]
}

struct MoeLayerWeights {
    moe_idx: usize,           // 0..42
    router: DeviceBuf,        // [288, 4096] bf16 (exact!)
    bias: DeviceBuf,          // [288] f32
    shared_gu: LinearWeight,   // fused [shared_gate; shared_up] = [4096, 4096] (1 launch instead of 2!)
    shared_down: LinearWeight, // [4096, 2048]
}

enum FfnWeights {
    Dense(DenseMlpWeights),
    Moe(MoeLayerWeights),
}

struct LayerWeights {
    attn_hc: HcWeights,
    ffn_hc: HcWeights,
    in_norm: DeviceBuf,   // [4096] bf16
    post_norm: DeviceBuf, // [4096] bf16
    mixer: MixerWeights,
    ffn: FfnWeights,
}

impl LayerWeights {
    fn vram_bytes(&self) -> usize {
        let m = match &self.mixer {
            MixerWeights::Kda { w, .. } => w.vram_bytes(),
            MixerWeights::Mla { w, .. } => w.vram_bytes(),
        };
        let f = match &self.ffn {
            FfnWeights::Dense(d) => d.gate.vram_bytes() + d.up.vram_bytes() + d.down.vram_bytes(),
            FfnWeights::Moe(e) => {
                e.router.len()
                    + e.bias.len()
                    + e.shared_gu.vram_bytes()
                    + e.shared_down.vram_bytes()
            }
        };
        self.attn_hc.vram_bytes() + self.ffn_hc.vram_bytes() + self.in_norm.len() + self.post_norm.len() + m + f
    }
}

#[derive(Debug, Clone)]
pub struct LoadSummary {
    pub load_secs: f64,
    pub dense_vram_mib: f64,
    pub drafter_vram_mib: f64,
    pub state_vram_mib: f64,
    pub expert_vram_mib: f64,
    pub vram_slots: usize,
    pub vram_hot_slots: usize,
    pub vram_cold_slots: usize,
    pub ram_slots: usize,
    pub ram_hot_slots: usize,
    pub ram_cold_slots: usize,
    pub direct_io: bool,
    pub fixed_io: bool,
    pub prewarm_loaded: usize,
    pub prewarm_ram_loaded: usize,
    pub prewarm_secs: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SpecDecodeStats {
    pub rounds: usize,
    pub total_drafted: usize,
    pub total_accepted: usize,
    pub tree_b_wins: usize,
    pub tree_b_accepted: usize,
    pub draft_secs: f64,
    pub verify_secs: f64,
    pub commit_secs: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkMode {
    Prefill,
    DecodeSerial,
    Verify,
    DraftResident,
}

/// Sparse top-2048 DSA indexer KV caches & scratch (allocated automatically when `max_ctx > 2051`).
/// Milestone 8: `ik` and `ig` use a tiny ring (`ik_cap = (max_rows + 8).next_multiple_of(4)`) since
/// tokens are never read again once their 4-token pool `pk` completes -> 704 MiB -> 1.4 MiB at 128K!
struct DsaState {
    ik_cap: usize,         // (max_rows + 8).next_multiple_of(4)
    pk_cap: usize,         // max_ctx / 4
    ik: DeviceBuf,         // [11, ik_cap, 128] bf16
    ig: DeviceBuf,         // [11, ik_cap, 128] bf16
    pk: DeviceBuf,         // [11, pk_cap, 128] bf16
    k_raw: DeviceBuf,      // [max_rows, 128] bf16
    w_raw: DeviceBuf,      // [max_rows, 32] bf16
    g_raw: DeviceBuf,      // [max_rows, 128] f32
    qi: DeviceBuf,         // [max_rows, 32 * 128] bf16
    scores: DeviceBuf,     // [max_rows, pk_cap] f32
    tokens: DeviceBuf,     // [max_rows, DSA_TOKEN_STRIDE] i32
    counts: DeviceBuf,     // [max_rows] i32
    vram_bytes: usize,
}

impl DsaState {
    fn alloc(ctx: &Arc<CudaContext>, mla_n: usize, max_ctx: usize, max_rows: usize) -> m3tis_cuda::Result<Self> {
        let ik_cap = (max_rows.max(16) + 8).next_multiple_of(DSA_KPOOL);
        let pk_cap = (max_ctx / DSA_KPOOL).max(DSA_TOPK_POOLS + 1);
        let ik = ctx.alloc_device(mla_n * ik_cap * DSA_DIM * 2)?;
        let ig = ctx.alloc_device(mla_n * ik_cap * DSA_DIM * 2)?;
        let pk = ctx.alloc_device(mla_n * pk_cap * DSA_DIM * 2)?;
        let k_raw = ctx.alloc_device(max_rows * DSA_DIM * 2)?;
        let w_raw = ctx.alloc_device(max_rows * DSA_HEADS * 2)?;
        let g_raw = ctx.alloc_device(max_rows * DSA_DIM * 4)?;
        let qi = ctx.alloc_device(max_rows * DSA_HEADS * DSA_DIM * 2)?;
        let scores = ctx.alloc_device(max_rows * pk_cap * 4)?;
        let tokens = ctx.alloc_device(max_rows * DSA_TOKEN_STRIDE * 4)?;
        let counts = ctx.alloc_device(max_rows * 4)?;
        let vram_bytes = ik.len()
            + ig.len()
            + pk.len()
            + k_raw.len()
            + w_raw.len()
            + g_raw.len()
            + qi.len()
            + scores.len()
            + tokens.len()
            + counts.len();
        Ok(Self {
            ik_cap,
            pk_cap,
            ik,
            ig,
            pk,
            k_raw,
            w_raw,
            g_raw,
            qi,
            scores,
            tokens,
            counts,
            vram_bytes,
        })
    }

    fn reset(&self, stream: &Stream) -> m3tis_cuda::Result<()> {
        stream.memset_d32_async(self.ik.ptr(), 0, self.ik.len() / 4)?;
        stream.memset_d32_async(self.ig.ptr(), 0, self.ig.len() / 4)?;
        stream.memset_d32_async(self.pk.ptr(), 0, self.pk.len() / 4)?;
        Ok(())
    }
}

/// Host-RAM snapshot of a committed prefix's exact KDA/MLA/DSA/DFlash2 state (~155 MiB RAM, 0 VRAM).
struct PrefixSnapshot {
    tokens: Vec<i32>,
    pos: usize,
    kda_rec: Vec<u8>,
    kda_conv: Vec<u8>,
    mla_kv_layers: Vec<Vec<u8>>,
    dsa_ik_layers: Vec<Vec<u8>>,
    dsa_ig_layers: Vec<Vec<u8>>,
    dsa_pk_layers: Vec<Vec<u8>>,
    drafter_rings: Option<(Vec<u8>, Vec<u8>, i64)>,
    last_logits: Option<Vec<f32>>,
    last_used: u64,
}

/// Bounded LRU multi-turn Prefix KV/KDA Cache stored in Host RAM + optional persistent NVMe disk store.
pub struct PrefixCache {
    max_entries: usize,
    entries: Vec<PrefixSnapshot>,
    disk: Option<DiskPrefixStore>,
    clock: u64,
    pub hits: u64,
    pub disk_hits: u64,
    pub tokens_reused: u64,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        Self::with_disk(max_entries, None)
    }

    pub fn with_disk(max_entries: usize, disk: Option<DiskPrefixStore>) -> Self {
        Self {
            max_entries,
            entries: Vec::new(),
            disk,
            clock: 0,
            hits: 0,
            disk_hits: 0,
            tokens_reused: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn disk_len(&self) -> usize {
        self.disk.as_ref().map_or(0, DiskPrefixStore::len)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.disk_len() == 0
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn find_best_idx(&self, prompt: &[i32], max_ctx: usize) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None; // (entry_idx, matched_len)
        for (i, snap) in self.entries.iter().enumerate() {
            let n = snap.tokens.len();
            if n == 0 || n > prompt.len() || snap.pos > max_ctx {
                continue;
            }
            if n == prompt.len() && snap.last_logits.is_none() {
                continue;
            }
            if prompt[..n] == snap.tokens[..] {
                if best.map_or(true, |(_, bl)| n > bl) {
                    best = Some((i, n));
                }
            }
        }
        best.map(|(i, _)| i)
    }

    fn insert(&mut self, mut snap: PrefixSnapshot) {
        if self.max_entries == 0 && self.disk.is_none() {
            return;
        }
        if snap.tokens.is_empty() {
            return;
        }
        if let Some(ref mut disk) = self.disk {
            if snap.tokens.len() >= 16 {
                let payload = DiskPrefixPayload {
                    tokens: snap.tokens.clone(),
                    pos: snap.pos,
                    kda_rec: snap.kda_rec.clone(),
                    kda_conv: snap.kda_conv.clone(),
                    mla_kv_layers: snap.mla_kv_layers.clone(),
                    dsa_ik_layers: snap.dsa_ik_layers.clone(),
                    dsa_ig_layers: snap.dsa_ig_layers.clone(),
                    dsa_pk_layers: snap.dsa_pk_layers.clone(),
                    drafter_rings: snap.drafter_rings.clone(),
                    last_logits: snap.last_logits.clone(),
                };
                if let Err(e) = disk.save(&payload) {
                    eprintln!("[prefix-disk] warning: failed to persist snapshot: {e}");
                }
            }
        }
        if self.max_entries == 0 {
            return;
        }
        self.clock += 1;
        snap.last_used = self.clock;
        if let Some(existing) = self.entries.iter_mut().find(|e| e.tokens == snap.tokens) {
            if snap.last_logits.is_none() && existing.last_logits.is_some() {
                snap.last_logits = existing.last_logits.take();
            }
            *existing = snap;
            return;
        }
        if self.entries.len() >= self.max_entries {
            if let Some((evict_idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
            {
                self.entries.swap_remove(evict_idx);
            }
        }
        self.entries.push(snap);
    }

    fn insert_ram_only(&mut self, mut snap: PrefixSnapshot) -> usize {
        self.clock += 1;
        snap.last_used = self.clock;
        if let Some(idx) = self.entries.iter().position(|e| e.tokens == snap.tokens) {
            self.entries[idx] = snap;
            return idx;
        }
        let cap = self.max_entries.max(1);
        if self.entries.len() >= cap {
            if let Some((evict_idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
            {
                self.entries.swap_remove(evict_idx);
            }
        }
        self.entries.push(snap);
        self.entries.len() - 1
    }
}

/// Reusable sequence state + GPU/Pinned scratch buffers for GLM-5.3-Flash.
pub struct Glm53State {
    pub pos: usize,
    pub max_ctx: usize,
    pub max_rows: usize,
    pub max_verify: usize,
    pub kv_int8: bool,
    // Double-buffered KDA recurrent state for lossless speculative verify rollback/replay
    kda_rec: [DeviceBuf; 2], // 2 x [34, 64, 128, 128] f32
    kda_cur: usize,
    kda_conv: DeviceBuf,          // [34, 3, 24576] bf16
    kda_conv_bak: DeviceBuf,      // [34, 3, 24576] bf16 (4.8 MiB scratch backup for zero-NVMe self-speculation!)
    mla_kv: DeviceBuf,            // [11, max_ctx, 512] bf16 OR int8 when `kv_int8`
    mla_kv_sc: Option<DeviceBuf>, // [11, max_ctx] f32 per-token scales when `kv_int8`
    dsa: Option<DsaState>,        // Sparse top-2048 DSA indexer caches when max_ctx > 2051
    // Speculative verify KDA per-row saves + DFlash2 target taps
    kda_k_save: DeviceBuf,   // [34, max_verify, 64, 128] f32
    kda_v_save: DeviceBuf,   // [34, max_verify, 64, 128] bf16
    kda_g_save: DeviceBuf,   // [34, max_verify, 64, 128] f32
    kda_b_save: DeviceBuf,   // [34, max_verify, 64] f32
    kda_p_save: DeviceBuf,   // [34, max_verify, 24576] bf16
    taps_buf: DeviceBuf,     // [max_rows, 5, 4096] bf16
    // Stream & mHC scratch
    h_embed_rows: PinnedHostBuf, // [max_rows, 4096] bf16
    d_embed_rows: DeviceBuf,     // [max_rows, 4096] bf16
    x: DeviceBuf,                // [max_rows, 4, 4096] bf16
    normed: DeviceBuf,           // [max_rows, 4096] bf16
    branch: DeviceBuf,           // [max_rows, 4096] f32
    hc_part: DeviceBuf,          // [max_rows, 16, 32] f32
    post: DeviceBuf,             // [max_rows, 4] f32
    comb: DeviceBuf,             // [max_rows, 16] f32
    // KDA scratch
    kda_p: DeviceBuf,   // [max_rows, 24896] bf16
    kda_ka: DeviceBuf,  // [max_rows, 8192] bf16
    kda_kg: DeviceBuf,  // [max_rows, 8192] bf16
    kda_out: DeviceBuf, // [max_rows, 8192] bf16
    // MLA scratch
    mla_qa_raw: DeviceBuf,  // [max_rows, 1536] bf16
    mla_qr: DeviceBuf,      // [max_rows, 1536] bf16
    mla_kva_raw: DeviceBuf, // [max_rows, 512] bf16
    mla_lat: DeviceBuf,     // [max_rows, 512] bf16
    mla_q: DeviceBuf,       // [max_rows, 16384] bf16
    mla_qa: DeviceBuf,      // [max_rows, 64, 512] bf16
    mla_ol: DeviceBuf,      // [max_rows, 64, 512] bf16
    mla_vn: DeviceBuf,      // [max_rows, 16384] bf16
    // Dense MLP / Shared Expert scratch
    mlp_g: DeviceBuf,   // [max_rows, 12288] bf16 (also reused for shared 2048!)
    mlp_u: DeviceBuf,   // [max_rows, 12288] bf16
    mlp_act: DeviceBuf, // [max_rows, 12288] bf16
    sh_out: DeviceBuf,  // [max_rows, 4096] f32
    // MoE router + EXL3 scratch
    router_logits: DeviceBuf, // [max_rows, 288] f32
    spec_logits: DeviceBuf,   // [max_rows, 288] f32
    d_picks: DeviceBuf,       // [max_rows, 9] i32
    d_mapped: DeviceBuf,      // [max_rows, 9] i32
    d_wts: DeviceBuf,         // [max_rows, 9] f32
    d_wts_renorm: DeviceBuf,  // [max_rows, 9] f32
    d_spec_picks: DeviceBuf,  // [max_rows, 9] i32
    d_spec_wts: DeviceBuf,    // [max_rows, 9] f32
    d_spec2_picks: DeviceBuf, // [max_rows, 9] i32
    h_picks: PinnedHostBuf,       // [max_rows, 9] i32
    h_wts: PinnedHostBuf,         // [max_rows, 9] f32
    h_spec_picks: PinnedHostBuf,  // [max_rows, 9] i32
    h_spec2_picks: PinnedHostBuf, // [max_rows, 9] i32
    // Batched all-42-layer zero-sync router pick/weight buffers (12 KiB!) for WarpZeroSync & Oracle Self-Spec!
    d_all_picks: DeviceBuf,       // [42 * max_verify * 9] i32
    d_all_wts: DeviceBuf,         // [42 * max_verify * 9] f32
    h_all_picks: PinnedHostBuf,   // [42 * max_verify * 9] i32
    h_all_wts: PinnedHostBuf,     // [42 * max_verify * 9] f32
    exl3_sc: Exl3MoeScratch,
    // Final head + multi-row verify argmax scratch
    final_hidden: DeviceBuf,       // [max_verify, 4096] bf16
    final_normed: DeviceBuf,       // [max_verify, 4096] bf16
    d_logits: DeviceBuf,           // [max_verify, vocab] f32
    h_logits: PinnedHostBuf,       // [vocab] f32 (for single-row logit export)
    d_argmax_ids: DeviceBuf,       // [max_verify] i32
    d_argmax_vals: DeviceBuf,      // [max_verify] f32 (also receives softmax top-1 prob in DraftResident)
    h_argmax_ids: PinnedHostBuf,   // [max_verify] i32
    h_argmax_probs: PinnedHostBuf, // [max_verify] f32
    // Self-healing WarpZeroSync heal counter (>0 triggers brief sync demand on cold domain shift)
    warp_heal_steps: usize,
    // Hybrid CPU-GPU Tier-2 RAM EXL3 execution & Split-Wave Async MoE scratch (tiny: ~642 KiB total!)
    h_normed: PinnedHostBuf,    // [max_verify, 4096] bf16
    h_cpu_out: PinnedHostBuf,   // [max_verify, 4096] f32
    d_cpu_out: DeviceBuf,       // [max_verify, 4096] f32
    h_wts_gpu: PinnedHostBuf,   // [max_verify, 9] f32
    h_w1_mapped: PinnedHostBuf, // [max_verify, 9] i32
    h_w2_mapped: PinnedHostBuf, // [max_verify, 9] i32
    d_w2_mapped: DeviceBuf,     // [max_verify, 9] i32
    h_w2_wts: PinnedHostBuf,    // [max_verify, 9] f32
    d_w2_wts: DeviceBuf,        // [max_verify, 9] f32
    // Scratch Vecs on CPU (zero alloc in hot loop)
    active_keys: Vec<usize>,
    strict_keys: Vec<usize>,
    gpu_keys: Vec<usize>,
    cpu_keys: Vec<usize>,
    cpu_jobs: Vec<CpuExl3ExpertJob>,
    skipped_cold: Vec<usize>,
    spec_keys: Vec<usize>,
    spec_ram_keys: Vec<usize>,
    warp_missing: Vec<(usize, f32)>,
    // Milestone 5B: Expert-Major Streaming Prefill workspace (allocated once when prefill_batch > 0)
    pf: Option<PfState>,
    vram_bytes: usize,
}

struct PfState {
    max_rows: usize,
    h_embed: PinnedHostBuf, // [max_rows, 4096] bf16
    x: DeviceBuf,           // [max_rows, 4, 4096] bf16
    normed: DeviceBuf,      // [max_rows, 4096] bf16
    branch: DeviceBuf,      // [max_rows, 4096] f32
    post: DeviceBuf,        // [max_rows, 4] f32
    comb: DeviceBuf,        // [max_rows, 16] f32
    d_picks: DeviceBuf,     // [max_rows, 9] i32
    d_wts: DeviceBuf,       // [max_rows, 9] f32
    h_picks: PinnedHostBuf, // [max_rows, 9] i32
    h_wts: PinnedHostBuf,   // [max_rows, 9] f32
    exl3_sc: Exl3PrefillScratch,
    h_taps_layers: Option<[PinnedHostBuf; 5]>, // 5 * [max_rows, 4096] bf16 in pinned RAM (0 VRAM!)
    h_taps_micro: Option<PinnedHostBuf>,       // [micro, 5, 4096] bf16
    exp_tokens: Vec<Vec<(i32, f32)>>,
    vram_list: Vec<(usize, usize)>,       // (local_e, vram_slot)
    vram_scale_pairs: Vec<(usize, usize)>,// (gkey, vram_slot)
    ram_list: Vec<(usize, usize)>,        // (local_e, gkey)
    nvme_list: Vec<(u64, usize, usize)>,  // (pack_offset, local_e, gkey)
    pair_row: Vec<i32>,
    pair_slot: Vec<i32>,
    pair_wt: Vec<f32>,
    tile_slot: Vec<i32>,
    tile_off: Vec<i32>,
    tile_cnt: Vec<i32>,
    ahead_plan: Vec<usize>,
    ahead_layer_end: Vec<usize>,
    ahead_key_to_pidx: Vec<i32>,
    wave_seq: usize,
    vram_bytes: usize,
}

impl PfState {
    fn alloc(
        ctx: &Arc<CudaContext>,
        cfg: &Glm53Config,
        max_rows: usize,
        micro_rows: usize,
        has_drafter: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let d = cfg.hidden;
        let slots = cfg.top_k + 1;
        let h_embed = ctx.alloc_pinned(max_rows * d * 2)?;
        let x = ctx.alloc_device(max_rows * cfg.hc_mult * d * 2)?;
        let normed = ctx.alloc_device(max_rows * d * 2)?;
        let branch = ctx.alloc_device(max_rows * d * 4)?;
        let post = ctx.alloc_device(max_rows * cfg.hc_mult * 4)?;
        let comb = ctx.alloc_device(max_rows * cfg.hc_mult * cfg.hc_mult * 4)?;
        let d_picks = ctx.alloc_device(max_rows * slots * 4)?;
        let d_wts = ctx.alloc_device(max_rows * slots * 4)?;
        let h_picks = ctx.alloc_pinned(max_rows * slots * 4)?;
        let h_wts = ctx.alloc_pinned(max_rows * slots * 4)?;
        let exl3_sc = Exl3PrefillScratch::alloc(ctx, 1024, 256, d, cfg.moe_inter)?;
        let (h_taps_layers, h_taps_micro) = if has_drafter {
            let layers = [
                ctx.alloc_pinned(max_rows * d * 2)?,
                ctx.alloc_pinned(max_rows * d * 2)?,
                ctx.alloc_pinned(max_rows * d * 2)?,
                ctx.alloc_pinned(max_rows * d * 2)?,
                ctx.alloc_pinned(max_rows * d * 2)?,
            ];
            let micro = ctx.alloc_pinned(micro_rows * 5 * d * 2)?;
            (Some(layers), Some(micro))
        } else {
            (None, None)
        };
        let vram_bytes = x.len()
            + normed.len()
            + branch.len()
            + post.len()
            + comb.len()
            + d_picks.len()
            + d_wts.len()
            + exl3_sc.vram_bytes;
        let mut exp_tokens = Vec::with_capacity(cfg.num_experts);
        for _ in 0..cfg.num_experts {
            exp_tokens.push(Vec::with_capacity(32));
        }
        Ok(Self {
            max_rows,
            h_embed,
            x,
            normed,
            branch,
            post,
            comb,
            d_picks,
            d_wts,
            h_picks,
            h_wts,
            exl3_sc,
            h_taps_layers,
            h_taps_micro,
            exp_tokens,
            vram_list: Vec::with_capacity(cfg.num_experts),
            vram_scale_pairs: Vec::with_capacity(cfg.num_experts),
            ram_list: Vec::with_capacity(cfg.num_experts),
            nvme_list: Vec::with_capacity(cfg.num_experts),
            pair_row: Vec::with_capacity(1024),
            pair_slot: Vec::with_capacity(1024),
            pair_wt: Vec::with_capacity(1024),
            tile_slot: Vec::with_capacity(256),
            tile_off: Vec::with_capacity(256),
            tile_cnt: Vec::with_capacity(256),
            ahead_plan: Vec::with_capacity(cfg.moe_layers() * cfg.num_experts),
            ahead_layer_end: vec![0; cfg.moe_layers()],
            ahead_key_to_pidx: vec![-1; cfg.moe_layers() * cfg.num_experts],
            wave_seq: 0,
            vram_bytes,
        })
    }
}

impl Glm53State {
    pub fn alloc(
        ctx: &Arc<CudaContext>,
        stream: &Stream,
        cfg: &Glm53Config,
        max_ctx: usize,
        max_rows: usize,
        pf_max_rows: usize,
        has_drafter: bool,
        kv_int8: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let kda_n = cfg.kda_layers();
        let mla_n = cfg.mla_layers();
        let slots = cfg.top_k + 1;
        // Milestone 8: Expand `max_verify` from 8 -> 16 so 2-branch DFlash2 trees (<= 15 rows) fit in 1 TC tile!
        let max_verify = (2 * DF_BLOCK).min(max_rows);

        let rec_bytes = kda_n * cfg.kda_heads * cfg.kda_dim * cfg.kda_dim * 4;
        let kda_rec = [ctx.alloc_device(rec_bytes)?, ctx.alloc_device(rec_bytes)?];
        let kda_conv = ctx.alloc_device(kda_n * 3 * cfg.kda_conv_c() * 2)?;
        let kda_conv_bak = ctx.alloc_device(kda_n * 3 * cfg.kda_conv_c() * 2)?;
        let mla_kv = if kv_int8 {
            ctx.alloc_device(mla_n * max_ctx * cfg.kv_lora)?
        } else {
            ctx.alloc_device(mla_n * max_ctx * cfg.kv_lora * 2)?
        };
        let mla_kv_sc = if kv_int8 {
            Some(ctx.alloc_device(mla_n * max_ctx * 4)?)
        } else {
            None
        };
        let dsa = if max_ctx > DSA_WIDTH {
            Some(DsaState::alloc(ctx, mla_n, max_ctx, max_rows)?)
        } else {
            None
        };

        let kv_save_elems = kda_n * max_verify * cfg.kda_heads * cfg.kda_dim;
        let kda_k_save = ctx.alloc_device(kv_save_elems * 4)?;
        let kda_v_save = ctx.alloc_device(kv_save_elems * 2)?;
        let kda_g_save = ctx.alloc_device(kv_save_elems * 4)?;
        let kda_b_save = ctx.alloc_device(kda_n * max_verify * cfg.kda_heads * 4)?;
        let kda_p_save = ctx.alloc_device(kda_n * max_verify * cfg.kda_conv_c() * 2)?;
        let taps_buf = ctx.alloc_device(max_rows * 5 * cfg.hidden * 2)?;

        let h_embed_rows = ctx.alloc_pinned(max_rows * cfg.hidden * 2)?;
        let d_embed_rows = ctx.alloc_device(max_rows * cfg.hidden * 2)?;
        let x = ctx.alloc_device(max_rows * cfg.hc_mult * cfg.hidden * 2)?;
        let normed = ctx.alloc_device(max_rows * cfg.hidden * 2)?;
        let branch = ctx.alloc_device(max_rows * cfg.hidden * 4)?;
        let hc_part = ctx.alloc_device(max_rows * 16 * 32 * 4)?;
        let post = ctx.alloc_device(max_rows * cfg.hc_mult * 4)?;
        let comb = ctx.alloc_device(max_rows * cfg.hc_mult * cfg.hc_mult * 4)?;

        let kda_p = ctx.alloc_device(max_rows * cfg.kda_p_stride() * 2)?;
        let kda_ka = ctx.alloc_device(max_rows * cfg.kda_heads * cfg.kda_dim * 2)?;
        let kda_kg = ctx.alloc_device(max_rows * cfg.kda_heads * cfg.kda_dim * 2)?;
        let kda_out = ctx.alloc_device(max_rows * cfg.kda_heads * cfg.kda_dim * 2)?;

        let mla_qa_raw = ctx.alloc_device(max_rows * cfg.q_lora * 2)?;
        let mla_qr = ctx.alloc_device(max_rows * cfg.q_lora * 2)?;
        let mla_kva_raw = ctx.alloc_device(max_rows * cfg.kv_lora * 2)?;
        let mla_lat = ctx.alloc_device(max_rows * cfg.kv_lora * 2)?;
        let mla_q = ctx.alloc_device(max_rows * cfg.mla_heads * cfg.qk_nope * 2)?;
        let mla_qa = ctx.alloc_device(max_rows * cfg.mla_heads * cfg.kv_lora * 2)?;
        let mla_ol = ctx.alloc_device(max_rows * cfg.mla_heads * cfg.kv_lora * 2)?;
        let mla_vn = ctx.alloc_device(max_rows * cfg.mla_heads * cfg.dv * 2)?;

        let mlp_g = ctx.alloc_device(max_rows * cfg.dense_inter * 2)?;
        let mlp_u = ctx.alloc_device(max_rows * cfg.dense_inter * 2)?;
        let mlp_act = ctx.alloc_device(max_rows * cfg.dense_inter * 2)?;
        let sh_out = ctx.alloc_device(max_rows * cfg.hidden * 4)?;

        let router_logits = ctx.alloc_device(max_rows * cfg.num_experts * 4)?;
        let spec_logits = ctx.alloc_device(max_rows * cfg.num_experts * 4)?;
        let d_picks = ctx.alloc_device(max_rows * slots * 4)?;
        let d_mapped = ctx.alloc_device(max_rows * slots * 4)?;
        let d_wts = ctx.alloc_device(max_rows * slots * 4)?;
        let d_wts_renorm = ctx.alloc_device(max_rows * slots * 4)?;
        let d_spec_picks = ctx.alloc_device(max_rows * slots * 4)?;
        let d_spec_wts = ctx.alloc_device(max_rows * slots * 4)?;
        let d_spec2_picks = ctx.alloc_device(max_rows * slots * 4)?;
        let h_picks = ctx.alloc_pinned(max_rows * slots * 4)?;
        let h_wts = ctx.alloc_pinned(max_rows * slots * 4)?;
        let h_spec_picks = ctx.alloc_pinned(max_rows * slots * 4)?;
        let h_spec2_picks = ctx.alloc_pinned(max_rows * slots * 4)?;
        let all_moe_bytes = cfg.moe_layers() * max_verify * slots * 4;
        let d_all_picks = ctx.alloc_device(all_moe_bytes)?;
        let d_all_wts = ctx.alloc_device(all_moe_bytes)?;
        let h_all_picks = ctx.alloc_pinned(all_moe_bytes)?;
        let h_all_wts = ctx.alloc_pinned(all_moe_bytes)?;
        let exl3_rows = if pf_max_rows > 0 { max_verify } else { max_rows };
        let exl3_sc = Exl3MoeScratch::alloc(ctx, exl3_rows, slots, cfg.hidden, cfg.moe_inter, 4)?;

        let final_hidden = ctx.alloc_device(max_verify * cfg.hidden * 2)?;
        let final_normed = ctx.alloc_device(max_verify * cfg.hidden * 2)?;
        let d_logits = ctx.alloc_device(max_verify * cfg.vocab * 4)?;
        let h_logits = ctx.alloc_pinned(cfg.vocab * 4)?;
        let d_argmax_ids = ctx.alloc_device(max_verify * 4)?;
        let d_argmax_vals = ctx.alloc_device(max_verify * 4)?;
        let h_argmax_ids = ctx.alloc_pinned(max_verify * 4)?;
        let h_argmax_probs = ctx.alloc_pinned(max_verify * 4)?;
        let h_normed = ctx.alloc_pinned(max_verify * cfg.hidden * 2)?;
        let h_cpu_out = ctx.alloc_pinned(max_verify * cfg.hidden * 4)?;
        let d_cpu_out = ctx.alloc_device(max_verify * cfg.hidden * 4)?;
        let h_wts_gpu = ctx.alloc_pinned(max_verify * slots * 4)?;
        let h_w1_mapped = ctx.alloc_pinned(max_verify * slots * 4)?;
        let h_w2_mapped = ctx.alloc_pinned(max_verify * slots * 4)?;
        let d_w2_mapped = ctx.alloc_device(max_verify * slots * 4)?;
        let h_w2_wts = ctx.alloc_pinned(max_verify * slots * 4)?;
        let d_w2_wts = ctx.alloc_device(max_verify * slots * 4)?;
        let pf = if pf_max_rows > 0 {
            Some(PfState::alloc(ctx, cfg, pf_max_rows.min(max_ctx), max_rows, has_drafter)?)
        } else {
            None
        };

        let vram_bytes = kda_rec[0].len()
            + kda_rec[1].len()
            + kda_conv.len()
            + kda_conv_bak.len()
            + mla_kv.len()
            + mla_kv_sc.as_ref().map_or(0, |s| s.len())
            + dsa.as_ref().map_or(0, |d| d.vram_bytes)
            + kda_k_save.len()
            + kda_v_save.len()
            + kda_g_save.len()
            + kda_b_save.len()
            + kda_p_save.len()
            + taps_buf.len()
            + d_embed_rows.len()
            + x.len()
            + normed.len()
            + branch.len()
            + hc_part.len()
            + post.len()
            + comb.len()
            + kda_p.len()
            + kda_ka.len()
            + kda_kg.len()
            + kda_out.len()
            + mla_qa_raw.len()
            + mla_qr.len()
            + mla_kva_raw.len()
            + mla_lat.len()
            + mla_q.len()
            + mla_qa.len()
            + mla_ol.len()
            + mla_vn.len()
            + mlp_g.len()
            + mlp_u.len()
            + mlp_act.len()
            + sh_out.len()
            + router_logits.len()
            + spec_logits.len()
            + d_picks.len()
            + d_mapped.len()
            + d_wts.len()
            + d_wts_renorm.len()
            + d_spec_picks.len()
            + d_spec_wts.len()
            + d_spec2_picks.len()
            + exl3_sc.uids.len()
            + exl3_sc.ucount.len()
            + exl3_sc.members.len()
            + exl3_sc.xg.len()
            + exl3_sc.xu.len()
            + exl3_sc.xd.len()
            + exl3_sc.z.len()
            + exl3_sc.slot_y.len()
            + final_hidden.len()
            + final_normed.len()
            + d_logits.len()
            + d_argmax_ids.len()
            + d_argmax_vals.len()
            + d_cpu_out.len()
            + pf.as_ref().map_or(0, |p| p.vram_bytes);

        let mut st = Self {
            pos: 0,
            max_ctx,
            max_rows,
            max_verify,
            kv_int8,
            kda_rec,
            kda_cur: 0,
            kda_conv,
            kda_conv_bak,
            mla_kv,
            mla_kv_sc,
            dsa,
            kda_k_save,
            kda_v_save,
            kda_g_save,
            kda_b_save,
            kda_p_save,
            taps_buf,
            h_embed_rows,
            d_embed_rows,
            x,
            normed,
            branch,
            hc_part,
            post,
            comb,
            kda_p,
            kda_ka,
            kda_kg,
            kda_out,
            mla_qa_raw,
            mla_qr,
            mla_kva_raw,
            mla_lat,
            mla_q,
            mla_qa,
            mla_ol,
            mla_vn,
            mlp_g,
            mlp_u,
            mlp_act,
            sh_out,
            router_logits,
            spec_logits,
            d_picks,
            d_mapped,
            d_wts,
            d_wts_renorm,
            d_spec_picks,
            d_spec_wts,
            d_spec2_picks,
            h_picks,
            h_wts,
            h_spec_picks,
            h_spec2_picks,
            d_all_picks,
            d_all_wts,
            h_all_picks,
            h_all_wts,
            exl3_sc,
            final_hidden,
            final_normed,
            d_logits,
            h_logits,
            d_argmax_ids,
            d_argmax_vals,
            h_argmax_ids,
            h_argmax_probs,
            warp_heal_steps: 0,
            h_normed,
            h_cpu_out,
            d_cpu_out,
            h_wts_gpu,
            h_w1_mapped,
            h_w2_mapped,
            d_w2_mapped,
            h_w2_wts,
            d_w2_wts,
            active_keys: Vec::with_capacity(288),
            strict_keys: Vec::with_capacity(64),
            gpu_keys: Vec::with_capacity(288),
            cpu_keys: Vec::with_capacity(32),
            cpu_jobs: Vec::with_capacity(32),
            skipped_cold: Vec::with_capacity(64),
            spec_keys: Vec::with_capacity(64),
            spec_ram_keys: Vec::with_capacity(64),
            warp_missing: Vec::with_capacity(512),
            pf,
            vram_bytes,
        };
        st.reset(stream)?;
        Ok(st)
    }

    pub fn reset(&mut self, stream: &Stream) -> m3tis_cuda::Result<()> {
        self.pos = 0;
        self.kda_cur = 0;
        self.warp_heal_steps = 0;
        stream.memset_d32_async(self.kda_rec[0].ptr(), 0, self.kda_rec[0].len() / 4)?;
        stream.memset_d32_async(self.kda_rec[1].ptr(), 0, self.kda_rec[1].len() / 4)?;
        stream.memset_d32_async(self.kda_conv.ptr(), 0, self.kda_conv.len() / 4)?;
        stream.memset_d32_async(self.mla_kv.ptr(), 0, self.mla_kv.len() / 4)?;
        if let Some(sc) = self.mla_kv_sc.as_ref() {
            stream.memset_d32_async(sc.ptr(), 0, sc.len() / 4)?;
        }
        if let Some(dsa) = self.dsa.as_ref() {
            dsa.reset(stream)?;
        }
        stream.memset_d32_async(self.exl3_sc.slot_y.ptr(), 0, self.exl3_sc.slot_y.len() / 4)?;
        stream.synchronize()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DecodeProf {
    pub calls: u64,
    pub total_rows: u64,
    pub attn_secs: f64,
    pub dense_ffn_secs: f64,
    pub picks_sync_secs: f64,
    pub demand_secs: f64,
    pub exl3_secs: f64,
    pub spec_secs: f64,
    pub head_secs: f64,
    pub row0_cold_rank0: u64,
    pub tail_cold_rank0: u64,
    pub tail_cold_skipped: u64,
}

pub struct Glm53Model {
    pub cfg: Glm53Config,
    pub ctx: Arc<CudaContext>,
    pub compute_stream: Stream,
    picks_ready: Event,
    lin: LinearKernels,
    glm: GlmOpsKernels,
    router_remap: RouterRemapKernels,
    exl3: Exl3MoeKernels,
    embed_host: PinnedHostBuf, // [vocab, 4096] bf16 in pinned Host RAM (0 VRAM!)
    final_norm: DeviceBuf,     // [4096] bf16
    lm_head: LinearWeight,     // [vocab, 4096] Q4 or BF16
    layers: Vec<LayerWeights>,
    pub drafter: Option<DFlash2Drafter>,
    pub prefix_cache: PrefixCache,
    pub pool: TieredExpertPool,
    pub scales: Exl3ScaleBank,
    pub cpu_exl3: CpuExl3Engine,
    exl3_ptr_tables: [DeviceBuf; 3],
    pub load_summary: LoadSummary,
    pub decode_prof: DecodeProf,
}

/// Milestone 7 Self-Calibrating Wall-Clock `tok/s` Adaptive DFlash2 Controller:
/// Measures live wall-clock throughput (`ewma_verify_tps` vs `ewma_serial_tps`) instead of a hardcoded
/// accept-rate cutoff — stays in fast DFlash2 whenever drafting beats serial `tok/s` (even at 2.2-2.8 tok/rd!),
/// switches to serial decode on genuinely un-draftable creative prose, and probes every 24 tokens!
struct AdaptiveDraftCtrl {
    enabled: bool,
    max_drafts: usize,
    rounds_seen: usize,
    ewma_tprd: f64,
    ewma_verify_tps: f64,
    ewma_serial_tps: Option<f64>,
    calibrate_serial_next: bool,
    serial_left: usize,
    probe_next: bool,
}

impl AdaptiveDraftCtrl {
    fn new(enabled: bool, max_drafts: usize) -> Self {
        Self {
            enabled,
            max_drafts,
            rounds_seen: 0,
            ewma_tprd: 3.2,
            ewma_verify_tps: 0.0,
            ewma_serial_tps: None,
            calibrate_serial_next: false,
            serial_left: 0,
            probe_next: false,
        }
    }

    /// Returns `(draft_k, is_probe)` where `draft_k == 0` selects ultra-fast 1-token serial decode.
    fn next_step(&mut self) -> (usize, bool) {
        if !self.enabled {
            return (self.max_drafts, false);
        }
        if self.calibrate_serial_next {
            self.calibrate_serial_next = false;
            return (0, true);
        }
        if self.serial_left > 0 {
            self.serial_left -= 1;
            if self.serial_left == 0 {
                self.probe_next = true;
            }
            return (0, false);
        }
        if self.probe_next {
            self.probe_next = false;
            return (self.max_drafts.min(4), true);
        }
        if self.rounds_seen < 3 {
            return (self.max_drafts, false);
        }
        if self.ewma_tprd >= 3.0 {
            (self.max_drafts, false)
        } else if self.ewma_tprd >= 2.2 {
            (self.max_drafts.min(5), false)
        } else {
            (self.max_drafts.min(3), false)
        }
    }

    fn record_serial(&mut self, elapsed_secs: f64, was_calibration: bool) {
        if !self.enabled {
            return;
        }
        let s_tps = 1.0 / elapsed_secs.max(1e-4);
        self.ewma_serial_tps = Some(match self.ewma_serial_tps {
            Some(prev) => 0.70 * prev + 0.30 * s_tps,
            None => s_tps,
        });
        if was_calibration {
            if let Some(cur_s) = self.ewma_serial_tps {
                if self.ewma_verify_tps > 0.0 && self.ewma_verify_tps < 0.88 * cur_s {
                    self.serial_left = 12;
                }
            }
        }
    }

    fn record_round(&mut self, keep: usize, elapsed_secs: f64, was_probe: bool) {
        if !self.enabled {
            return;
        }
        self.rounds_seen += 1;
        let k = keep as f64;
        let round_tps = k / elapsed_secs.max(1e-4);
        if self.ewma_verify_tps <= 0.0 {
            self.ewma_verify_tps = round_tps;
        } else {
            self.ewma_verify_tps = 0.65 * self.ewma_verify_tps + 0.35 * round_tps;
        }
        if was_probe {
            let beat_serial = match self.ewma_serial_tps {
                Some(s) => round_tps >= 0.92 * s || keep >= 3,
                None => keep >= 3,
            };
            if beat_serial {
                self.ewma_tprd = k.max(3.0);
                self.ewma_verify_tps = round_tps;
                self.serial_left = 0;
            } else {
                self.ewma_tprd = 0.5 * self.ewma_tprd + 0.5 * k;
                self.serial_left = 12;
            }
            return;
        }
        self.ewma_tprd = 0.65 * self.ewma_tprd + 0.35 * k;
        if self.rounds_seen >= 6 && self.ewma_tprd < 2.2 {
            match self.ewma_serial_tps {
                None => {
                    // Take 1 single serial measurement to calibrate true local serial tok/s!
                    self.calibrate_serial_next = true;
                }
                Some(s_tps) => {
                    if self.ewma_verify_tps < 0.88 * s_tps {
                        self.serial_left = 12;
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenerateResult {
    pub prompt_tokens: usize,
    pub prefix_cached_tokens: usize,
    pub generated_tokens: Vec<i32>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub spec_stats: SpecDecodeStats,
    pub finish_reason: &'static str,
}

fn sample_logits(logits: &[f32], temperature: f32, top_p: f32, rng_state: &mut u64) -> i32 {
    if temperature <= 1e-5 {
        let (best_id, _) = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });
        return best_id as i32;
    }
    let inv_t = 1.0f32 / temperature;
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Keep top-256 candidates by scaled logit before nucleus top_p (fast O(V) single pass!)
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(256);
    let mut sum_exp = 0.0f64;
    let thresh = max_l - 20.0 * temperature; // exp(-20) < 2e-9
    for (i, &v) in logits.iter().enumerate() {
        if v >= thresh {
            let p = ((v - max_l) * inv_t).exp();
            sum_exp += p as f64;
            top.push((i, p));
        }
    }
    if top.is_empty() || sum_exp <= 0.0 {
        return 0;
    }
    top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let cutoff = (top_p.clamp(0.01, 1.0) as f64) * sum_exp;
    let mut cum = 0.0f64;
    let mut keep = 0usize;
    for &(_, p) in &top {
        cum += p as f64;
        keep += 1;
        if cum >= cutoff {
            break;
        }
    }
    top.truncate(keep.max(1));
    // xorshift64*
    let mut x = if *rng_state == 0 { 0x9e37_79b9_7f4a_7c15 } else { *rng_state };
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng_state = x;
    let u = ((x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    let target = u * cum;
    let mut acc = 0.0f64;
    for &(id, p) in &top {
        acc += p as f64;
        if acc >= target {
            return id as i32;
        }
    }
    top[0].0 as i32
}

fn read_mem_available_gib() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb as f64 / (1024.0 * 1024.0));
        }
    }
    None
}

impl Glm53Model {
    fn capture_prefix_snapshot(
        &self,
        st: &Glm53State,
        tokens: Vec<i32>,
        last_logits: Option<Vec<f32>>,
    ) -> m3tis_cuda::Result<PrefixSnapshot> {
        let pos = st.pos;
        let mla_n = self.cfg.mla_layers();
        let used_toks = pos.min(st.max_ctx);

        let cur_rec = &st.kda_rec[st.kda_cur];
        let mut kda_rec = vec![0u8; cur_rec.len()];
        let mut kda_conv = vec![0u8; st.kda_conv.len()];
        unsafe {
            self.compute_stream
                .dtoh_async(kda_rec.as_mut_ptr(), cur_rec.ptr(), kda_rec.len())?;
            self.compute_stream
                .dtoh_async(kda_conv.as_mut_ptr(), st.kda_conv.ptr(), kda_conv.len())?;
        }

        let mut mla_kv_layers = Vec::with_capacity(mla_n);
        if st.kv_int8 {
            let q8_stride = st.max_ctx * self.cfg.kv_lora;
            let sc_stride = st.max_ctx * 4;
            let q8_bytes = used_toks * self.cfg.kv_lora;
            let sc_bytes = used_toks * 4;
            let kv_sc = st.mla_kv_sc.as_ref().expect("kv_int8 requires mla_kv_sc");
            for idx in 0..mla_n {
                let mut buf = vec![0u8; q8_bytes + sc_bytes];
                if used_toks > 0 {
                    unsafe {
                        self.compute_stream.dtoh_async(
                            buf.as_mut_ptr(),
                            st.mla_kv.at(idx * q8_stride),
                            q8_bytes,
                        )?;
                        self.compute_stream.dtoh_async(
                            buf.as_mut_ptr().add(q8_bytes),
                            kv_sc.at(idx * sc_stride),
                            sc_bytes,
                        )?;
                    }
                }
                mla_kv_layers.push(buf);
            }
        } else {
            let mla_kv_stride = st.max_ctx * self.cfg.kv_lora * 2;
            let used_kv_bytes = used_toks * self.cfg.kv_lora * 2;
            for idx in 0..mla_n {
                let mut buf = vec![0u8; used_kv_bytes];
                if used_kv_bytes > 0 {
                    unsafe {
                        self.compute_stream.dtoh_async(
                            buf.as_mut_ptr(),
                            st.mla_kv.at(idx * mla_kv_stride),
                            used_kv_bytes,
                        )?;
                    }
                }
                mla_kv_layers.push(buf);
            }
        }

        let mut dsa_ik_layers = Vec::new();
        let mut dsa_ig_layers = Vec::new();
        let mut dsa_pk_layers = Vec::new();
        if let Some(dsa) = st.dsa.as_ref() {
            // Milestone 8: Save ONLY the `pos % 4` (`0..=3`) incomplete tail tokens of `ik`/`ig`
            // (<= 8.4 KiB total vs 704 MiB at 128K!) — completed 4-token pools live in `pk`!
            let row_bytes = DSA_DIM * 2;
            let ik_stride = dsa.ik_cap * row_bytes;
            let pk_stride = dsa.pk_cap * row_bytes;
            let tail = pos % DSA_KPOOL;
            let tail_start = pos - tail;
            let used_ik_bytes = tail * row_bytes;
            let used_pk_bytes = (pos / DSA_KPOOL).min(dsa.pk_cap) * row_bytes;
            for idx in 0..mla_n {
                let mut bik = vec![0u8; used_ik_bytes];
                let mut big = vec![0u8; used_ik_bytes];
                let mut bpk = vec![0u8; used_pk_bytes];
                unsafe {
                    for q in 0..tail {
                        let slot = (tail_start + q) % dsa.ik_cap;
                        self.compute_stream.dtoh_async(
                            bik.as_mut_ptr().add(q * row_bytes),
                            dsa.ik.at(idx * ik_stride + slot * row_bytes),
                            row_bytes,
                        )?;
                        self.compute_stream.dtoh_async(
                            big.as_mut_ptr().add(q * row_bytes),
                            dsa.ig.at(idx * ik_stride + slot * row_bytes),
                            row_bytes,
                        )?;
                    }
                    if used_pk_bytes > 0 {
                        self.compute_stream.dtoh_async(
                            bpk.as_mut_ptr(),
                            dsa.pk.at(idx * pk_stride),
                            used_pk_bytes,
                        )?;
                    }
                }
                dsa_ik_layers.push(bik);
                dsa_ig_layers.push(big);
                dsa_pk_layers.push(bpk);
            }
        }
        self.compute_stream.synchronize()?;

        let drafter_rings = match self.drafter.as_ref() {
            Some(dr) => Some(dr.snapshot_rings(&self.compute_stream)?),
            None => None,
        };

        Ok(PrefixSnapshot {
            tokens,
            pos,
            kda_rec,
            kda_conv,
            mla_kv_layers,
            dsa_ik_layers,
            dsa_ig_layers,
            dsa_pk_layers,
            drafter_rings,
            last_logits,
            last_used: 0,
        })
    }

    fn restore_prefix_snapshot(
        &mut self,
        st: &mut Glm53State,
        snap_idx: usize,
    ) -> m3tis_cuda::Result<(usize, Option<Vec<f32>>)> {
        self.prefix_cache.clock += 1;
        let clock = self.prefix_cache.clock;
        let snap = &mut self.prefix_cache.entries[snap_idx];
        snap.last_used = clock;

        st.pos = snap.pos;
        st.kda_cur = 0;
        let mla_n = self.cfg.mla_layers();
        let used_toks = snap.pos.min(st.max_ctx);
        unsafe {
            self.compute_stream.htod_async(
                st.kda_rec[0].ptr(),
                snap.kda_rec.as_ptr(),
                snap.kda_rec.len(),
            )?;
            self.compute_stream.htod_async(
                st.kda_conv.ptr(),
                snap.kda_conv.as_ptr(),
                snap.kda_conv.len(),
            )?;
            if st.kv_int8 {
                let q8_stride = st.max_ctx * self.cfg.kv_lora;
                let sc_stride = st.max_ctx * 4;
                let q8_bytes = used_toks * self.cfg.kv_lora;
                let sc_bytes = used_toks * 4;
                let kv_sc = st.mla_kv_sc.as_ref().expect("kv_int8 requires mla_kv_sc");
                for idx in 0..mla_n {
                    let buf = &snap.mla_kv_layers[idx];
                    if buf.len() == q8_bytes + sc_bytes && used_toks > 0 {
                        self.compute_stream.htod_async(
                            st.mla_kv.at(idx * q8_stride),
                            buf.as_ptr(),
                            q8_bytes,
                        )?;
                        self.compute_stream.htod_async(
                            kv_sc.at(idx * sc_stride),
                            buf.as_ptr().add(q8_bytes),
                            sc_bytes,
                        )?;
                    }
                }
            } else {
                let mla_kv_stride = st.max_ctx * self.cfg.kv_lora * 2;
                for idx in 0..mla_n {
                    let buf = &snap.mla_kv_layers[idx];
                    if !buf.is_empty() {
                        self.compute_stream.htod_async(
                            st.mla_kv.at(idx * mla_kv_stride),
                            buf.as_ptr(),
                            buf.len(),
                        )?;
                    }
                }
            }
            if let Some(dsa) = st.dsa.as_ref() {
                if snap.dsa_pk_layers.len() == mla_n {
                    let row_bytes = DSA_DIM * 2;
                    let ik_stride = dsa.ik_cap * row_bytes;
                    let pk_stride = dsa.pk_cap * row_bytes;
                    let tail = snap.pos % DSA_KPOOL;
                    let tail_start = snap.pos - tail;
                    for idx in 0..mla_n {
                        if snap.dsa_ik_layers.len() == mla_n
                            && snap.dsa_ik_layers[idx].len() == tail * row_bytes
                        {
                            for q in 0..tail {
                                let slot = (tail_start + q) % dsa.ik_cap;
                                self.compute_stream.htod_async(
                                    dsa.ik.at(idx * ik_stride + slot * row_bytes),
                                    snap.dsa_ik_layers[idx].as_ptr().add(q * row_bytes),
                                    row_bytes,
                                )?;
                                self.compute_stream.htod_async(
                                    dsa.ig.at(idx * ik_stride + slot * row_bytes),
                                    snap.dsa_ig_layers[idx].as_ptr().add(q * row_bytes),
                                    row_bytes,
                                )?;
                            }
                        }
                        if !snap.dsa_pk_layers[idx].is_empty() {
                            self.compute_stream.htod_async(
                                dsa.pk.at(idx * pk_stride),
                                snap.dsa_pk_layers[idx].as_ptr(),
                                snap.dsa_pk_layers[idx].len(),
                            )?;
                        }
                    }
                }
            }
        }
        if let (Some(dr), Some((rk, rv, from_pos))) =
            (self.drafter.as_mut(), snap.drafter_rings.as_ref())
        {
            dr.restore_rings(&self.compute_stream, rk, rv, *from_pos)?;
        }
        self.compute_stream.synchronize()?;

        let matched = snap.tokens.len();
        let logits = snap.last_logits.clone();
        self.prefix_cache.hits += 1;
        self.prefix_cache.tokens_reused += matched as u64;
        Ok((matched, logits))
    }

    /// Prefill `prompt_tokens` while checking BOTH Host-RAM LRU and Persistent NVMe Disk prefix caches!
    /// Returns `(first_logits, cached_tokens, loaded_from_disk)` and persists newly computed prefixes.
    pub fn prefill_with_prefix_cache(
        &mut self,
        st: &mut Glm53State,
        prompt_tokens: &[i32],
        opts: ForwardOpts,
    ) -> Result<(Vec<f32>, usize, bool), Box<dyn Error>> {
        let ram_best = self.prefix_cache.find_best_idx(prompt_tokens, st.max_ctx);
        let ram_len = ram_best.map_or(0, |i| self.prefix_cache.entries[i].tokens.len());
        let disk_best = self
            .prefix_cache
            .disk
            .as_ref()
            .and_then(|d| d.find_best(prompt_tokens, st.max_ctx));
        let disk_len = disk_best.map_or(0, |(_, n)| n);

        let mut loaded_from_disk = false;
        let chosen_idx = if disk_len > ram_len {
            let (d_idx, _) = disk_best.unwrap();
            let loaded = self.prefix_cache.disk.as_mut().unwrap().load(d_idx);
            match loaded {
                Ok(p) => {
                    loaded_from_disk = true;
                    self.prefix_cache.disk_hits += 1;
                    let snap = PrefixSnapshot {
                        tokens: p.tokens,
                        pos: p.pos,
                        kda_rec: p.kda_rec,
                        kda_conv: p.kda_conv,
                        mla_kv_layers: p.mla_kv_layers,
                        dsa_ik_layers: p.dsa_ik_layers,
                        dsa_ig_layers: p.dsa_ig_layers,
                        dsa_pk_layers: p.dsa_pk_layers,
                        drafter_rings: p.drafter_rings,
                        last_logits: p.last_logits,
                        last_used: 0,
                    };
                    Some(self.prefix_cache.insert_ram_only(snap))
                }
                Err(e) => {
                    eprintln!("[prefix-disk] warning: failed to load cached prefix: {e}");
                    ram_best
                }
            }
        } else {
            ram_best
        };

        let mut prefix_cached_tokens = 0usize;
        let first_logits = if let Some(snap_idx) = chosen_idx {
            let (matched, maybe_logits) = self.restore_prefix_snapshot(st, snap_idx)?;
            prefix_cached_tokens = matched;
            if matched == prompt_tokens.len() {
                maybe_logits.expect("exact prefix hit must have cached logits")
            } else {
                self.forward(st, &prompt_tokens[matched..], opts)?
            }
        } else {
            self.reset_sequence(st)?;
            self.forward(st, prompt_tokens, opts)?
        };

        if (self.prefix_cache.max_entries > 0 || self.prefix_cache.disk.is_some())
            && prompt_tokens.len() >= 8
            && prefix_cached_tokens < prompt_tokens.len()
        {
            let snap = self.capture_prefix_snapshot(
                st,
                prompt_tokens.to_vec(),
                Some(first_logits.clone()),
            )?;
            self.prefix_cache.insert(snap);
        }

        Ok((first_logits, prefix_cached_tokens, loaded_from_disk))
    }

    /// End-to-end streaming generation (prefill + DFlash2 greedy OR serial temperature/top-p sampling).
    /// Automatically reuses matching Host-RAM Prefix KV/KDA snapshots to skip redundant prompt prefill
    /// across multi-turn chats while keeping Tier-1 VRAM & Tier-2 Pinned RAM expert caches warm!
    /// `on_token(id)` is invoked immediately for each emitted token; return `false` to stop early.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_stream<F>(
        &mut self,
        st: &mut Glm53State,
        prompt_tokens: &[i32],
        max_tokens: usize,
        max_drafts: usize,
        conf_threshold: f64,
        temperature: f32,
        top_p: f32,
        mut seed: u64,
        opts: ForwardOpts,
        mut on_token: F,
    ) -> Result<GenerateResult, Box<dyn Error>>
    where
        F: FnMut(i32) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err("empty prompt tokens".into());
        }
        if max_tokens == 0 {
            return Ok(GenerateResult {
                prompt_tokens: prompt_tokens.len(),
                finish_reason: "length",
                ..GenerateResult::default()
            });
        }
        let t_pf = Instant::now();
        let (first_logits, prefix_cached_tokens, _from_disk) =
            self.prefill_with_prefix_cache(st, prompt_tokens, opts)?;
        let prefill_secs = t_pf.elapsed().as_secs_f64();

        let first_tok = sample_logits(&first_logits, temperature, top_p, &mut seed);
        let mut generated = Vec::with_capacity(max_tokens.min(4096));
        let mut committed_tokens = prompt_tokens.to_vec();
        let mut spec_stats = SpecDecodeStats::default();
        let mut finish_reason = "length";

        let t_dec = Instant::now();
        if self.cfg.eos_tokens.contains(&first_tok) {
            return Ok(GenerateResult {
                prompt_tokens: prompt_tokens.len(),
                prefix_cached_tokens,
                generated_tokens: generated,
                prefill_secs,
                decode_secs: t_dec.elapsed().as_secs_f64(),
                spec_stats,
                finish_reason: "stop",
            });
        }
        generated.push(first_tok);
        if !on_token(first_tok) {
            return Ok(GenerateResult {
                prompt_tokens: prompt_tokens.len(),
                prefix_cached_tokens,
                generated_tokens: generated,
                prefill_secs,
                decode_secs: t_dec.elapsed().as_secs_f64(),
                spec_stats,
                finish_reason: "stop",
            });
        }

        let use_dflash = self.drafter.is_some() && temperature <= 1e-5;
        let use_self_spec = !use_dflash && opts.self_spec_k > 0 && temperature <= 1e-5;
        let d = self.cfg.hidden;
        let vocab = self.cfg.vocab;
        let mut anchor = first_tok;

        if use_dflash {
            let mut ctrl = AdaptiveDraftCtrl::new(opts.adaptive_draft, max_drafts);
            let min_room = if opts.tree_verify { st.max_verify } else { DF_BLOCK };
            while generated.len() < max_tokens && st.pos + min_room < st.max_ctx {
                let (cur_k, is_probe) = ctrl.next_step();
                if cur_k == 0 {
                    // Ultra-fast serial step (keeps DFlash2 target KV rings 100% synchronized!)
                    let ts0 = Instant::now();
                    let logits = self.forward(st, &[anchor], opts)?;
                    ctrl.record_serial(ts0.elapsed().as_secs_f64(), is_probe);
                    committed_tokens.push(anchor);
                    let next_tok = sample_logits(&logits, 0.0, 1.0, &mut seed);
                    if self.cfg.eos_tokens.contains(&next_tok) {
                        finish_reason = "stop";
                        break;
                    }
                    generated.push(next_tok);
                    if !on_token(next_tok) {
                        finish_reason = "stop";
                        break;
                    }
                    anchor = next_tok;
                    continue;
                }
                let tr_start = Instant::now();
                let td0 = Instant::now();
                let tree: DraftTree = {
                    let dr = self.drafter.as_mut().unwrap();
                    let lin = &self.lin;
                    let glm = &self.glm;
                    let lm_head = &self.lm_head;
                    let embed_slice = self.embed_host.as_slice();
                    if opts.tree_verify && st.max_verify > DF_BLOCK {
                        dr.propose_tree(
                            &self.compute_stream,
                            lin,
                            glm,
                            embed_slice,
                            |stream, x_ptr, out_ptr, r| {
                                lm_head.apply(lin, stream, x_ptr, d, out_ptr, vocab, r, true)
                            },
                            anchor,
                            st.pos,
                            cur_k,
                            conf_threshold,
                            st.max_verify,
                        )?
                    } else {
                        let (primary, row_cum_probs) = dr.propose(
                            &self.compute_stream,
                            lin,
                            glm,
                            embed_slice,
                            |stream, x_ptr, out_ptr, r| {
                                lm_head.apply(lin, stream, x_ptr, d, out_ptr, vocab, r, true)
                            },
                            anchor,
                            st.pos,
                            cur_k,
                            conf_threshold,
                        )?;
                        let fpos = primary.len();
                        DraftTree {
                            primary,
                            row_cum_probs,
                            fork_pos: fpos,
                            branch_b: Vec::new(),
                        }
                    }
                };
                spec_stats.draft_secs += td0.elapsed().as_secs_f64();

                let has_b = !tree.branch_b.is_empty();
                let trunk_len = 1 + tree.fork_pos;
                let branch_b_start = if has_b { 1 + tree.primary.len() } else { 0 };
                let mut verify_toks = Vec::with_capacity(1 + tree.primary.len() + tree.branch_b.len());
                verify_toks.push(anchor);
                verify_toks.extend_from_slice(&tree.primary);
                verify_toks.extend_from_slice(&tree.branch_b);
                let rows = verify_toks.len();

                let tv0 = Instant::now();
                let pred_ids = if has_b {
                    self.verify_tree(st, &verify_toks, trunk_len, branch_b_start, opts)?
                } else {
                    self.verify_block_probs(st, &verify_toks, Some(&tree.row_cum_probs), opts)?
                };
                spec_stats.verify_secs += tv0.elapsed().as_secs_f64();

                let (accepted_toks, next_anchor, branch_b_won) =
                    self.eval_tree_acceptance(&tree, &pred_ids, branch_b_start);
                let accepted = accepted_toks.len();
                let keep = accepted + 1;
                spec_stats.rounds += 1;
                spec_stats.total_drafted += tree.primary.len();
                spec_stats.total_accepted += accepted;
                if branch_b_won {
                    spec_stats.tree_b_wins += 1;
                    spec_stats.tree_b_accepted += accepted - tree.fork_pos;
                }

                let mut stop_now = false;
                for &dtok in &accepted_toks {
                    if self.cfg.eos_tokens.contains(&dtok) {
                        finish_reason = "stop";
                        stop_now = true;
                        break;
                    }
                    if generated.len() >= max_tokens {
                        stop_now = true;
                        break;
                    }
                    generated.push(dtok);
                    if !on_token(dtok) {
                        finish_reason = "stop";
                        stop_now = true;
                        break;
                    }
                }
                if !stop_now {
                    if self.cfg.eos_tokens.contains(&next_anchor) {
                        finish_reason = "stop";
                        stop_now = true;
                    } else if generated.len() < max_tokens {
                        generated.push(next_anchor);
                        if !on_token(next_anchor) {
                            finish_reason = "stop";
                            stop_now = true;
                        }
                    } else {
                        stop_now = true;
                    }
                }

                let tc0 = Instant::now();
                if has_b {
                    self.commit_verify_tree(st, rows, trunk_len, branch_b_start, keep, branch_b_won)?;
                } else {
                    self.commit_verify(st, rows, keep)?;
                }
                committed_tokens.push(anchor);
                committed_tokens.extend_from_slice(&accepted_toks);
                spec_stats.commit_secs += tc0.elapsed().as_secs_f64();
                ctrl.record_round(keep, tr_start.elapsed().as_secs_f64(), is_probe);

                if stop_now || generated.len() >= max_tokens {
                    break;
                }
                anchor = next_anchor;
            }
            self.compute_stream.synchronize()?;
        } else if use_self_spec {
            let k_max = opts.self_spec_k.clamp(1, st.max_verify.saturating_sub(1));
            let rec_bytes = st.kda_rec[0].len();
            let conv_bytes = st.kda_conv.len();
            while generated.len() < max_tokens && st.pos + 1 < st.max_ctx {
                let max_room = (st.max_ctx - 1 - st.pos).min(k_max);
                let td0 = Instant::now();
                let saved_pos = st.pos;
                let saved_cur = st.kda_cur;
                let scratch_cur = 1 - saved_cur;
                self.compute_stream.dtod_async(
                    st.kda_rec[scratch_cur].ptr(),
                    st.kda_rec[saved_cur].ptr(),
                    rec_bytes,
                )?;
                self.compute_stream.dtod_async(
                    st.kda_conv_bak.ptr(),
                    st.kda_conv.ptr(),
                    conv_bytes,
                )?;
                st.kda_cur = scratch_cur;

                let mut drafts = Vec::with_capacity(max_room);
                let mut oracle_keys: Vec<usize> = Vec::new();
                let mut oracle_cursor = 0usize;
                let mut cur_tok = anchor;
                let e_cnt = self.cfg.num_experts;
                let moe_n = self.cfg.moe_layers();
                let slots = self.cfg.top_k + 1;
                for step_i in 0..max_room {
                    self.forward_chunk(st, &[cur_tok], opts, ChunkMode::DraftResident, false, true)?;
                    let pred = st.h_argmax_ids.as_i32_mut()[0];
                    let prob = st.h_argmax_probs.as_f32_mut()[0];
                    drafts.push(pred);
                    // Oracle NVMe Prefetch: draft step 0 ran on `anchor` (row 0 of verify_block!), so its
                    // 42-layer router picks predict the exact cold experts needed ~45 ms later!
                    if step_i == 0 {
                        let hp = &st.h_all_picks.as_i32_mut()[..moe_n * slots];
                        for ml in 0..moe_n {
                            let base_key = ml * e_cnt;
                            for k_rank in 0..self.cfg.top_k {
                                let p = hp[ml * slots + k_rank];
                                if p >= 0 && (p as usize) < e_cnt {
                                    let gk = base_key + (p as usize);
                                    if self.pool.vram_cache.peek(gk).is_none()
                                        && self.pool.ram_cache.peek(gk).is_none()
                                        && self.pool.ram_cache.pending(gk).is_none()
                                        && !oracle_keys.contains(&gk)
                                    {
                                        oracle_keys.push(gk);
                                    }
                                }
                            }
                        }
                    }
                    if oracle_cursor < oracle_keys.len() {
                        let _ = self.pool.poll_spec_only();
                        if self.pool.spec_inflight_count() == 0 {
                            let submitted = self
                                .pool
                                .start_spec_prefetch(&oracle_keys[oracle_cursor..], 8)
                                .unwrap_or(0);
                            oracle_cursor += submitted.max(1);
                        }
                    }
                    if self.cfg.eos_tokens.contains(&pred) {
                        break;
                    }
                    if opts.self_spec_conf > 0.0 && prob < opts.self_spec_conf {
                        break;
                    }
                    cur_tok = pred;
                }
                st.pos = saved_pos;
                st.kda_cur = saved_cur;
                self.compute_stream.dtod_async(
                    st.kda_conv.ptr(),
                    st.kda_conv_bak.ptr(),
                    conv_bytes,
                )?;
                spec_stats.draft_secs += td0.elapsed().as_secs_f64();

                let mut verify_toks = Vec::with_capacity(1 + drafts.len());
                verify_toks.push(anchor);
                verify_toks.extend_from_slice(&drafts);
                let rows = verify_toks.len();

                let tv0 = Instant::now();
                let pred_ids = self.verify_block(st, &verify_toks, opts)?;
                spec_stats.verify_secs += tv0.elapsed().as_secs_f64();

                let mut accepted = 0usize;
                for i in 0..drafts.len() {
                    if pred_ids[i] == drafts[i] {
                        accepted += 1;
                        if self.cfg.eos_tokens.contains(&drafts[i]) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                let keep = accepted + 1;
                spec_stats.rounds += 1;
                spec_stats.total_drafted += drafts.len();
                spec_stats.total_accepted += accepted;

                let mut stop_now = false;
                for &dtok in &drafts[..accepted] {
                    if self.cfg.eos_tokens.contains(&dtok) {
                        finish_reason = "stop";
                        stop_now = true;
                        break;
                    }
                    if generated.len() >= max_tokens {
                        stop_now = true;
                        break;
                    }
                    generated.push(dtok);
                    if !on_token(dtok) {
                        finish_reason = "stop";
                        stop_now = true;
                        break;
                    }
                }
                let next_anchor = pred_ids[accepted];
                if !stop_now {
                    if self.cfg.eos_tokens.contains(&next_anchor) {
                        finish_reason = "stop";
                        stop_now = true;
                    } else if generated.len() < max_tokens {
                        generated.push(next_anchor);
                        if !on_token(next_anchor) {
                            finish_reason = "stop";
                            stop_now = true;
                        }
                    } else {
                        stop_now = true;
                    }
                }

                let tc0 = Instant::now();
                self.commit_verify(st, rows, keep)?;
                committed_tokens.extend_from_slice(&verify_toks[..keep]);
                spec_stats.commit_secs += tc0.elapsed().as_secs_f64();

                if stop_now || generated.len() >= max_tokens {
                    break;
                }
                anchor = next_anchor;
            }
            self.compute_stream.synchronize()?;
        } else {
            while generated.len() < max_tokens && st.pos + 1 < st.max_ctx {
                let logits = self.forward(st, &[anchor], opts)?;
                committed_tokens.push(anchor);
                let next_tok = sample_logits(&logits, temperature, top_p, &mut seed);
                if self.cfg.eos_tokens.contains(&next_tok) {
                    finish_reason = "stop";
                    break;
                }
                generated.push(next_tok);
                if !on_token(next_tok) {
                    finish_reason = "stop";
                    break;
                }
                anchor = next_tok;
            }
        }

        let decode_secs = t_dec.elapsed().as_secs_f64();

        // Save turn-end prefix snapshot so multi-turn follow-ups skip the entire prior conversation!
        if (self.prefix_cache.max_entries > 0 || self.prefix_cache.disk.is_some())
            && committed_tokens.len() > prompt_tokens.len()
            && st.pos == committed_tokens.len()
        {
            let snap = self.capture_prefix_snapshot(st, committed_tokens, None)?;
            self.prefix_cache.insert(snap);
        }

        Ok(GenerateResult {
            prompt_tokens: prompt_tokens.len(),
            prefix_cached_tokens,
            generated_tokens: generated,
            prefill_secs,
            decode_secs,
            spec_stats,
            finish_reason,
        })
    }
}

impl Glm53Model {
    pub fn load(
        ctx: &Arc<CudaContext>,
        ckpt_dir: &Path,
        pack_path: &Path,
        opts: &Glm53LoadOptions,
    ) -> Result<(Self, Glm53State), Box<dyn Error>> {
        let t0 = Instant::now();
        let cfg = Glm53Config::default();
        let compute_stream = ctx.stream()?;
        let picks_ready = ctx.event()?;
        let lin = LinearKernels::load(ctx)?;
        let glm = GlmOpsKernels::load(ctx)?;
        let router_remap = RouterRemapKernels::load(ctx)?;
        let exl3 = Exl3MoeKernels::load(ctx)?;

        let st = SafetensorsIndex::open_dir(ckpt_dir)?;

        // Single reusable 136 MiB staging buffer (host + device) for packing Q4 linears with ZERO VRAM spikes!
        let max_stage_bytes = 16384 * cfg.hidden * 2; // 128 MiB (covers every layer matrix & 16k-row lm_head chunks)
        let mut h_stage = vec![0u8; max_stage_bytes];
        let d_stage = ctx.alloc_device(max_stage_bytes)?;

        let load_raw_dev = |name: &str, h_buf: &mut Vec<u8>| -> Result<DeviceBuf, Box<dyn Error>> {
            let loc = st.get(name)?;
            if h_buf.len() < loc.nbytes {
                h_buf.resize(loc.nbytes, 0);
            }
            st.read_into(name, &mut h_buf[..loc.nbytes])?;
            let d = ctx.alloc_device(loc.nbytes)?;
            unsafe {
                compute_stream.htod_async(d.ptr(), h_buf.as_ptr(), loc.nbytes)?;
            }
            compute_stream.synchronize()?;
            Ok(d)
        };

        #[derive(Clone, Copy)]
        enum DenseMode {
            Bf16,
            Q4,
            Q3,
        }
        let dense_mode = if opts.pack_dense_q3 {
            DenseMode::Q3
        } else if opts.pack_dense_q4 {
            DenseMode::Q4
        } else {
            DenseMode::Bf16
        };
        let moe_dense_mode = if opts.pack_dense_q3 || opts.pack_dense_q3_moe {
            DenseMode::Q3
        } else {
            dense_mode
        };
        let head_mode = if opts.pack_head_q4 {
            DenseMode::Q4
        } else {
            DenseMode::Bf16
        };

        let load_linear = |name: &str,
                           n: usize,
                           k: usize,
                           mode: DenseMode,
                           h_buf: &mut Vec<u8>|
         -> Result<LinearWeight, Box<dyn Error>> {
            let loc = st.get(name)?;
            let expected = n * k * 2;
            if loc.nbytes != expected {
                return Err(format!("{name}: expected {expected} bytes, got {}", loc.nbytes).into());
            }
            if matches!(mode, DenseMode::Bf16) {
                if h_buf.len() < expected {
                    h_buf.resize(expected, 0);
                }
                st.read_into(name, &mut h_buf[..expected])?;
                let d = ctx.alloc_device(expected)?;
                unsafe {
                    compute_stream.htod_async(d.ptr(), h_buf.as_ptr(), expected)?;
                }
                compute_stream.synchronize()?;
                return Ok(LinearWeight::Bf16 { buf: d, n, k });
            }
            if expected <= d_stage.len() {
                st.read_into(name, &mut h_buf[..expected])?;
                unsafe {
                    compute_stream.htod_async(d_stage.ptr(), h_buf.as_ptr(), expected)?;
                }
                let out = match mode {
                    DenseMode::Q3 => LinearWeight::Q3(lin.pack_bf16_to_q3(ctx, &compute_stream, d_stage.ptr(), n, k)?),
                    _ => LinearWeight::Q4(lin.pack_bf16_to_q4(ctx, &compute_stream, d_stage.ptr(), n, k)?),
                };
                compute_stream.synchronize()?;
                return Ok(out);
            }
            // For matrices > 128 MiB (e.g. lm_head), pack in chunks via device-to-device copies!
            let chunk_rows = (d_stage.len() / (k * 2)).max(1).min(n);
            let q_row_bytes = match mode {
                DenseMode::Q3 => (k / 64) * 6 * 4,
                _ => (k / 8) * 4,
            };
            let s_row_bytes = (k / 64) * 4;
            let qdata = ctx.alloc_device(n * q_row_bytes)?;
            let scales = ctx.alloc_device(n * s_row_bytes)?;
            let mut r0 = 0usize;
            while r0 < n {
                let nr = chunk_rows.min(n - r0);
                let byte_off = r0 * k * 2;
                let byte_len = nr * k * 2;
                st.read_range(name, byte_off, &mut h_buf[..byte_len])?;
                unsafe {
                    compute_stream.htod_async(d_stage.ptr(), h_buf.as_ptr(), byte_len)?;
                }
                match mode {
                    DenseMode::Q3 => {
                        let cq = lin.pack_bf16_to_q3(ctx, &compute_stream, d_stage.ptr(), nr, k)?;
                        compute_stream.dtod_async(qdata.at(r0 * q_row_bytes), cq.qdata.ptr(), nr * q_row_bytes)?;
                        compute_stream.dtod_async(scales.at(r0 * s_row_bytes), cq.scales.ptr(), nr * s_row_bytes)?;
                        compute_stream.synchronize()?;
                    }
                    _ => {
                        let cq = lin.pack_bf16_to_q4(ctx, &compute_stream, d_stage.ptr(), nr, k)?;
                        compute_stream.dtod_async(qdata.at(r0 * q_row_bytes), cq.qdata.ptr(), nr * q_row_bytes)?;
                        compute_stream.dtod_async(scales.at(r0 * s_row_bytes), cq.scales.ptr(), nr * s_row_bytes)?;
                        compute_stream.synchronize()?;
                    }
                }
                r0 += nr;
            }
            Ok(match mode {
                DenseMode::Q3 => LinearWeight::Q3(Q3Weight { qdata, scales, n, k }),
                _ => LinearWeight::Q4(Q4Weight { qdata, scales, n, k }),
            })
        };

        // Fuse multiple row-major `[per_n, k]` weight matrices along N into ONE contiguous Q3/Q4/BF16 weight!
        let load_linear_concat = |names: &[String],
                                  per_n: usize,
                                  k: usize,
                                  mode: DenseMode,
                                  h_buf: &mut Vec<u8>|
         -> Result<LinearWeight, Box<dyn Error>> {
            let total_n = per_n * names.len();
            let part_bytes = per_n * k * 2;
            for nm in names {
                let loc = st.get(nm)?;
                if loc.nbytes != part_bytes {
                    return Err(format!("{nm}: expected {part_bytes} bytes, got {}", loc.nbytes).into());
                }
            }
            if matches!(mode, DenseMode::Bf16) {
                let d = ctx.alloc_device(total_n * k * 2)?;
                for (i, nm) in names.iter().enumerate() {
                    if h_buf.len() < part_bytes {
                        h_buf.resize(part_bytes, 0);
                    }
                    st.read_into(nm, &mut h_buf[..part_bytes])?;
                    unsafe {
                        compute_stream.htod_async(d.at(i * part_bytes), h_buf.as_ptr(), part_bytes)?;
                    }
                    compute_stream.synchronize()?;
                }
                return Ok(LinearWeight::Bf16 { buf: d, n: total_n, k });
            }
            let q_row_bytes = match mode {
                DenseMode::Q3 => (k / 64) * 6 * 4,
                _ => (k / 8) * 4,
            };
            let s_row_bytes = (k / 64) * 4;
            let qdata = ctx.alloc_device(total_n * q_row_bytes)?;
            let scales = ctx.alloc_device(total_n * s_row_bytes)?;
            for (i, nm) in names.iter().enumerate() {
                st.read_into(nm, &mut h_buf[..part_bytes])?;
                unsafe {
                    compute_stream.htod_async(d_stage.ptr(), h_buf.as_ptr(), part_bytes)?;
                }
                let r0 = i * per_n;
                match mode {
                    DenseMode::Q3 => {
                        let cq = lin.pack_bf16_to_q3(ctx, &compute_stream, d_stage.ptr(), per_n, k)?;
                        compute_stream.dtod_async(qdata.at(r0 * q_row_bytes), cq.qdata.ptr(), per_n * q_row_bytes)?;
                        compute_stream.dtod_async(scales.at(r0 * s_row_bytes), cq.scales.ptr(), per_n * s_row_bytes)?;
                        compute_stream.synchronize()?;
                    }
                    _ => {
                        let cq = lin.pack_bf16_to_q4(ctx, &compute_stream, d_stage.ptr(), per_n, k)?;
                        compute_stream.dtod_async(qdata.at(r0 * q_row_bytes), cq.qdata.ptr(), per_n * q_row_bytes)?;
                        compute_stream.dtod_async(scales.at(r0 * s_row_bytes), cq.scales.ptr(), per_n * s_row_bytes)?;
                        compute_stream.synchronize()?;
                    }
                }
            }
            Ok(match mode {
                DenseMode::Q3 => LinearWeight::Q3(Q3Weight { qdata, scales, n: total_n, k }),
                _ => LinearWeight::Q4(Q4Weight { qdata, scales, n: total_n, k }),
            })
        };

        // 1. Load embed_tokens into Pinned Host RAM (0 MiB VRAM!)
        let emb_name = format!("{PREFIX}embed_tokens.weight");
        let emb_bytes = cfg.vocab * cfg.hidden * 2;
        let mut embed_host = ctx.alloc_pinned(emb_bytes)?;
        st.read_into(&emb_name, embed_host.as_mut_slice())?;

        // 2. Load final norm & lm_head
        let final_norm = load_raw_dev(&format!("{PREFIX}norm.weight"), &mut h_stage)?;
        let lm_head = load_linear("lm_head.weight", cfg.vocab, cfg.hidden, head_mode, &mut h_stage)?;

        // 3. Load all 45 layers
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let mut kda_idx = 0usize;
        let mut mla_idx = 0usize;
        let mut moe_idx = 0usize;
        let mut dense_vram_bytes = final_norm.len() + lm_head.vram_bytes();

        for l in 0..cfg.num_layers {
            let load_hc = |site: &str, hb: &mut Vec<u8>| -> Result<HcWeights, Box<dyn Error>> {
                Ok(HcWeights {
                    fn_w: load_raw_dev(&format!("{PREFIX}layers.{l}.hc_{site}_fn"), hb)?,
                    base: load_raw_dev(&format!("{PREFIX}layers.{l}.hc_{site}_base"), hb)?,
                    scale: load_raw_dev(&format!("{PREFIX}layers.{l}.hc_{site}_scale"), hb)?,
                })
            };
            let attn_hc = load_hc("attn", &mut h_stage)?;
            let ffn_hc = load_hc("ffn", &mut h_stage)?;
            let in_norm = load_raw_dev(&format!("{PREFIX}layers.{l}.input_layernorm.weight"), &mut h_stage)?;
            let post_norm =
                load_raw_dev(&format!("{PREFIX}layers.{l}.post_attention_layernorm.weight"), &mut h_stage)?;

            let layer_lin_mode = if cfg.is_moe(l) { moe_dense_mode } else { dense_mode };
            let mixer = if !cfg.is_mla(l) {
                let n_kda = cfg.kda_heads * cfg.kda_dim; // 8192
                // Fuse [q_proj; k_proj; v_proj] = [24576, 4096] -> 1 launch instead of 3 across 34 KDA layers (-68 launches/tok!)
                let qkv_proj = load_linear_concat(
                    &[
                        format!("{PREFIX}layers.{l}.self_attn.q_proj.weight"),
                        format!("{PREFIX}layers.{l}.self_attn.k_proj.weight"),
                        format!("{PREFIX}layers.{l}.self_attn.v_proj.weight"),
                    ],
                    n_kda,
                    cfg.hidden,
                    layer_lin_mode,
                    &mut h_stage,
                )?;
                // Fuse [f_a_proj (128); g_a_proj (128); b_proj (64)] -> contiguous [320, 4096] BF16 (1 launch, 98% SM occ!)
                let mut fgb_h = Vec::with_capacity((2 * cfg.kda_dim + cfg.kda_heads) * cfg.hidden * 2);
                fgb_h.extend_from_slice(&st.read_vec(&format!("{PREFIX}layers.{l}.self_attn.f_a_proj.weight"))?);
                fgb_h.extend_from_slice(&st.read_vec(&format!("{PREFIX}layers.{l}.self_attn.g_a_proj.weight"))?);
                fgb_h.extend_from_slice(&st.read_vec(&format!("{PREFIX}layers.{l}.self_attn.b_proj.weight"))?);
                let fgb_proj = ctx.alloc_device(fgb_h.len())?;
                unsafe {
                    compute_stream.htod_async(fgb_proj.ptr(), fgb_h.as_ptr(), fgb_h.len())?;
                }
                compute_stream.synchronize()?;
                let f_b_proj =
                    load_raw_dev(&format!("{PREFIX}layers.{l}.self_attn.f_b_proj.weight"), &mut h_stage)?;
                let g_b_proj =
                    load_raw_dev(&format!("{PREFIX}layers.{l}.self_attn.g_b_proj.weight"), &mut h_stage)?;
                let o_proj = load_linear(
                    &format!("{PREFIX}layers.{l}.self_attn.o_proj.weight"),
                    cfg.hidden,
                    n_kda,
                    layer_lin_mode,
                    &mut h_stage,
                )?;

                // Concat [q|k|v]_conv1d.weight (BF16 [8192, 1, 4]) and widen to F32 [24576, 4]!
                let conv_c = cfg.kda_conv_c();
                let mut conv_f32 = Vec::with_capacity(conv_c * 4);
                for tag in ["q", "k", "v"] {
                    let cname = format!("{PREFIX}layers.{l}.self_attn.{tag}_conv1d.weight");
                    let raw = st.read_vec(&cname)?;
                    for ch in raw.chunks_exact(2) {
                        let b16 = u16::from_le_bytes([ch[0], ch[1]]);
                        conv_f32.push(f32::from_bits((b16 as u32) << 16));
                    }
                }
                let conv_w = ctx.alloc_device(conv_f32.len() * 4)?;
                unsafe {
                    compute_stream.htod_async(
                        conv_w.ptr(),
                        conv_f32.as_ptr() as *const u8,
                        conv_f32.len() * 4,
                    )?;
                }
                compute_stream.synchronize()?;

                let a_log = load_raw_dev(&format!("{PREFIX}layers.{l}.self_attn.A_log"), &mut h_stage)?;
                let dt_bias = load_raw_dev(&format!("{PREFIX}layers.{l}.self_attn.dt_bias"), &mut h_stage)?;
                let o_norm =
                    load_raw_dev(&format!("{PREFIX}layers.{l}.self_attn.o_norm.weight"), &mut h_stage)?;

                let idx = kda_idx;
                kda_idx += 1;
                MixerWeights::Kda {
                    idx,
                    w: KdaWeights {
                        qkv_proj,
                        fgb_proj,
                        f_b_proj,
                        g_b_proj,
                        o_proj,
                        conv_w,
                        a_log,
                        dt_bias,
                        o_norm,
                    },
                }
            } else {
                let q_a_proj = load_linear(
                    &format!("{PREFIX}layers.{l}.self_attn.q_a_proj.weight"),
                    cfg.q_lora,
                    cfg.hidden,
                    dense_mode,
                    &mut h_stage,
                )?;
                let q_a_norm = load_raw_dev(
                    &format!("{PREFIX}layers.{l}.self_attn.q_a_layernorm.weight"),
                    &mut h_stage,
                )?;
                let q_b_proj = load_linear(
                    &format!("{PREFIX}layers.{l}.self_attn.q_b_proj.weight"),
                    cfg.mla_heads * cfg.qk_nope,
                    cfg.q_lora,
                    layer_lin_mode,
                    &mut h_stage,
                )?;
                let kv_a_proj = load_linear(
                    &format!("{PREFIX}layers.{l}.self_attn.kv_a_proj_with_mqa.weight"),
                    cfg.kv_lora,
                    cfg.hidden,
                    dense_mode,
                    &mut h_stage,
                )?;
                let kv_a_norm = load_raw_dev(
                    &format!("{PREFIX}layers.{l}.self_attn.kv_a_layernorm.weight"),
                    &mut h_stage,
                )?;

                // Split kv_b_proj [64 * (256 + 256), 512] -> wk_t [64, 512, 256] (transposed for 128-bit coalesced mla_absorb_q!) and wv [64, 256, 512] BF16!
                let kv_b_raw = st.read_vec(&format!("{PREFIX}layers.{l}.self_attn.kv_b_proj.weight"))?;
                let row_bytes = cfg.kv_lora * 2;
                let k_head_bytes = cfg.qk_nope * row_bytes;
                let v_head_bytes = cfg.dv * row_bytes;
                let per_head_bytes = k_head_bytes + v_head_bytes;
                let mut wk_t_h = vec![0u8; cfg.mla_heads * k_head_bytes];
                let mut wv_h = Vec::with_capacity(cfg.mla_heads * v_head_bytes);
                for h in 0..cfg.mla_heads {
                    let base = h * per_head_bytes;
                    let k_src = &kv_b_raw[base..base + k_head_bytes];
                    let k_dst_base = h * k_head_bytes;
                    for k in 0..cfg.qk_nope {
                        for lw in 0..cfg.kv_lora {
                            let s_off = (k * cfg.kv_lora + lw) * 2;
                            let d_off = k_dst_base + (lw * cfg.qk_nope + k) * 2;
                            wk_t_h[d_off] = k_src[s_off];
                            wk_t_h[d_off + 1] = k_src[s_off + 1];
                        }
                    }
                    wv_h.extend_from_slice(&kv_b_raw[base + k_head_bytes..base + per_head_bytes]);
                }
                let wk = ctx.alloc_device(wk_t_h.len())?;
                let wv = ctx.alloc_device(wv_h.len())?;
                unsafe {
                    compute_stream.htod_async(wk.ptr(), wk_t_h.as_ptr(), wk_t_h.len())?;
                    compute_stream.htod_async(wv.ptr(), wv_h.as_ptr(), wv_h.len())?;
                }
                compute_stream.synchronize()?;

                let o_proj = load_linear(
                    &format!("{PREFIX}layers.{l}.self_attn.o_proj.weight"),
                    cfg.hidden,
                    cfg.mla_heads * cfg.dv,
                    layer_lin_mode,
                    &mut h_stage,
                )?;

                let indexer = if opts.max_ctx > DSA_WIDTH {
                    let pfx = format!("{PREFIX}layers.{l}.self_attn.indexer.");
                    let ix_wk = load_raw_dev(&format!("{pfx}wk.weight"), &mut h_stage)?;
                    let ix_w_proj = load_raw_dev(&format!("{pfx}weights_proj.weight"), &mut h_stage)?;
                    let ix_gate = load_raw_dev(&format!("{pfx}index_kpool_compress_gate"), &mut h_stage)?;
                    let ix_knw = load_raw_dev(&format!("{pfx}k_norm.weight"), &mut h_stage)?;
                    let ix_knb = load_raw_dev(&format!("{pfx}k_norm.bias"), &mut h_stage)?;
                    let ix_ape = load_raw_dev(&format!("{pfx}index_kpool_compress_ape"), &mut h_stage)?;
                    let ix_wqb = load_linear(
                        &format!("{pfx}wq_b.weight"),
                        DSA_HEADS * DSA_DIM,
                        cfg.q_lora,
                        dense_mode,
                        &mut h_stage,
                    )?;
                    Some(DsaIndexerWeights {
                        wk: ix_wk,
                        w_proj: ix_w_proj,
                        gate: ix_gate,
                        k_norm_w: ix_knw,
                        k_norm_b: ix_knb,
                        ape: ix_ape,
                        wq_b: ix_wqb,
                    })
                } else {
                    None
                };

                let idx = mla_idx;
                mla_idx += 1;
                MixerWeights::Mla {
                    idx,
                    w: MlaWeights {
                        q_a_proj,
                        q_a_norm,
                        q_b_proj,
                        kv_a_proj,
                        kv_a_norm,
                        wk,
                        wv,
                        o_proj,
                        indexer,
                    },
                }
            };

            let ffn = if !cfg.is_moe(l) {
                let gate = load_linear(
                    &format!("{PREFIX}layers.{l}.mlp.gate_proj.weight"),
                    cfg.dense_inter,
                    cfg.hidden,
                    dense_mode,
                    &mut h_stage,
                )?;
                let up = load_linear(
                    &format!("{PREFIX}layers.{l}.mlp.up_proj.weight"),
                    cfg.dense_inter,
                    cfg.hidden,
                    dense_mode,
                    &mut h_stage,
                )?;
                let down = load_linear(
                    &format!("{PREFIX}layers.{l}.mlp.down_proj.weight"),
                    cfg.hidden,
                    cfg.dense_inter,
                    dense_mode,
                    &mut h_stage,
                )?;
                FfnWeights::Dense(DenseMlpWeights { gate, up, down })
            } else {
                let router = load_raw_dev(&format!("{PREFIX}layers.{l}.mlp.gate.weight"), &mut h_stage)?;
                let bias = load_raw_dev(
                    &format!("{PREFIX}layers.{l}.mlp.gate.e_score_correction_bias"),
                    &mut h_stage,
                )?;
                // Fuse [shared_gate; shared_up] = [4096, 4096] -> 1 launch instead of 2 across 42 MoE layers (-42 launches/tok!)
                let shared_gu = load_linear_concat(
                    &[
                        format!("{PREFIX}layers.{l}.mlp.shared_experts.gate_proj.weight"),
                        format!("{PREFIX}layers.{l}.mlp.shared_experts.up_proj.weight"),
                    ],
                    cfg.shared_inter,
                    cfg.hidden,
                    layer_lin_mode,
                    &mut h_stage,
                )?;
                let shared_down = load_linear(
                    &format!("{PREFIX}layers.{l}.mlp.shared_experts.down_proj.weight"),
                    cfg.hidden,
                    cfg.shared_inter,
                    layer_lin_mode,
                    &mut h_stage,
                )?;
                let idx = moe_idx;
                moe_idx += 1;
                FfnWeights::Moe(MoeLayerWeights {
                    moe_idx: idx,
                    router,
                    bias,
                    shared_gu,
                    shared_down,
                })
            };

            let lw = LayerWeights {
                attn_hc,
                ffn_hc,
                in_norm,
                post_norm,
                mixer,
                ffn,
            };
            dense_vram_bytes += lw.vram_bytes();
            layers.push(lw);
        }

        // Release staging buffers BEFORE loading drafter & sizing Tier-1 VRAM expert pool!
        drop(d_stage);
        drop(h_stage);

        // 4. Optional DFlash2 speculative drafter (~650 MiB VRAM in Q4, ~505 MiB in Q3!)
        let drafter = match &opts.drafter_dir {
            Some(dir) => Some(DFlash2Drafter::load(
                ctx,
                &compute_stream,
                &lin,
                dir,
                opts.pack_drafter_q4,
                opts.pack_drafter_q3,
                opts.max_chunk_rows,
            )?),
            None => None,
        };
        let drafter_vram_mib = drafter
            .as_ref()
            .map(|dr| dr.vram_bytes as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);

        // 5. Allocate reusable sequence state + KV/KDA caches + GPU scratch (+ Milestone 5B Prefill Workspace)
        let state = Glm53State::alloc(
            ctx,
            &compute_stream,
            &cfg,
            opts.max_ctx,
            opts.max_chunk_rows,
            opts.prefill_batch,
            drafter.is_some(),
            opts.kv_int8,
        )?;

        // 6. Auto-size remaining VRAM for Tier-1 expert slabs & guard host RAM headroom against OOM!
        let (free_vram, _total_vram) = ctx.mem_info()?;
        let free_gib = (free_vram as f64) / ((1u64 << 30) as f64);
        let vram_gib = opts
            .vram_gib
            .unwrap_or_else(|| (free_gib - opts.vram_headroom_gib).max(0.5));
        let eff_ram_gib = if let Some(avail_gib) = read_mem_available_gib() {
            // Keep >= 12 GiB unpinned host headroom (after ~2 GiB embed+scales + ~2.1 GiB LayerAheadStreamer ring)
            let reserve = if opts.ram_gib >= 4.0 { 16.2 } else { 14.0 };
            let safe_cap = (avail_gib - reserve).max(3.0);
            if opts.ram_gib > safe_cap {
                eprintln!(
                    "[load] clamping Tier-2 pinned RAM from {:.1} GiB -> {:.1} GiB (host MemAvailable={:.1} GiB)",
                    opts.ram_gib, safe_cap, avail_gib
                );
                safe_cap
            } else {
                opts.ram_gib
            }
        } else {
            opts.ram_gib
        };

        let mut pool = TieredExpertPool::open(
            ctx,
            pack_path,
            vram_gib,
            eff_ram_gib,
            opts.ram_policy,
            opts.sqpoll,
        )?;
        let (exl3_ptr_tables, _unused_k2) =
            pool.build_exl3_slot_tables(ctx, &compute_stream, opts.exl3_k2)?;
        compute_stream.synchronize()?;

        // 7. Load all 12,096 expert EXL3 Hadamard scales (suh/svh) into resident pinned-RAM ScaleBank (~425 MiB)
        let total_routed = cfg.moe_layers() * cfg.num_experts;
        let mut scales = Exl3ScaleBank::alloc(
            ctx,
            total_routed,
            pool.vram_slots(),
            cfg.hidden,
            cfg.moe_inter,
        )?;
        for ml in 0..cfg.moe_layers() {
            let model_layer = ml + 3;
            for e in 0..cfg.num_experts {
                let gkey = ml * cfg.num_experts + e;
                scales.load_expert_from_safetensors(&st, PREFIX, model_layer, e, gkey)?;
            }
        }

        // 8. Optional Boot Exclusive Tier-1 VRAM + Tier-2 RAM Pre-Warming (~2.5-6s sequential NVMe load of top Zipf experts!)
        let (prewarm_loaded, prewarm_ram_loaded, prewarm_secs) = if opts.prewarm_vram {
            pool.prewarm_vram(&compute_stream, &mut scales, opts.prewarm_cold, opts.prewarm_ram)?
        } else {
            (0, 0, 0.0)
        };

        let expert_vram_mib = pool.expert_vram_bytes() as f64 / (1024.0 * 1024.0);
        let (vram_hot_slots, vram_cold_slots) = pool.vram_hot_cold_slots();
        let (ram_hot_slots, ram_cold_slots) = pool.ram_hot_cold_slots();
        let load_summary = LoadSummary {
            load_secs: t0.elapsed().as_secs_f64(),
            dense_vram_mib: dense_vram_bytes as f64 / (1024.0 * 1024.0),
            drafter_vram_mib,
            state_vram_mib: state.vram_bytes as f64 / (1024.0 * 1024.0),
            expert_vram_mib,
            vram_slots: pool.vram_slots(),
            vram_hot_slots,
            vram_cold_slots,
            ram_slots: pool.ram_cache.capacity(),
            ram_hot_slots,
            ram_cold_slots,
            direct_io: pool.is_direct(),
            fixed_io: pool.is_registered_bufs(),
            prewarm_loaded,
            prewarm_ram_loaded,
            prewarm_secs,
        };

        let disk_store = if let Some(ref pdir) = opts.prefix_cache_dir {
            let pack_len = std::fs::metadata(pack_path).map(|m| m.len()).unwrap_or(0);
            let fp_str = format!(
                "glm53:l={}:d={}:v={}:e={}:pack={}:{}:dr={}:dsa={}:dm={}{}{}{}:kvq8={}:dring=1",
                cfg.num_layers,
                cfg.hidden,
                cfg.vocab,
                cfg.num_experts,
                pack_len,
                pool.catalog.total_experts(),
                drafter.is_some() as u8,
                (opts.max_ctx > DSA_WIDTH) as u8,
                opts.pack_dense_q4 as u8,
                opts.pack_dense_q3 as u8,
                opts.pack_dense_q3_moe as u8,
                opts.pack_head_q4 as u8,
                opts.kv_int8 as u8,
            );
            let fp = fnv1a_u64(fp_str.as_bytes());
            let max_bytes = (opts.prefix_cache_max_gib.max(0.1) * ((1u64 << 30) as f64)) as u64;
            match DiskPrefixStore::open(pdir.clone(), fp, max_bytes) {
                Ok(store) => {
                    eprintln!(
                        "[prefix-disk] opened {} ({} cached prefixes, {:.2} MiB / {:.1} GiB cap)",
                        pdir.display(),
                        store.len(),
                        store.total_bytes() as f64 / (1024.0 * 1024.0),
                        opts.prefix_cache_max_gib
                    );
                    Some(store)
                }
                Err(e) => {
                    eprintln!("[prefix-disk] warning: could not open {}: {e}", pdir.display());
                    None
                }
            }
        } else {
            None
        };

        let model = Self {
            cfg,
            ctx: Arc::clone(ctx),
            compute_stream,
            picks_ready,
            lin,
            glm,
            router_remap,
            exl3,
            embed_host,
            final_norm,
            lm_head,
            layers,
            drafter,
            prefix_cache: PrefixCache::with_disk(opts.prefix_cache_slots, disk_store),
            pool,
            scales,
            cpu_exl3: CpuExl3Engine::new(0),
            exl3_ptr_tables,
            load_summary,
            decode_prof: DecodeProf::default(),
        };
        Ok((model, state))
    }

    pub fn reset_decode_prof(&mut self) {
        self.decode_prof = DecodeProf::default();
    }

    pub fn pool_stats(&self) -> PoolStats {
        self.pool.stats
    }

    pub fn reset_sequence(&mut self, state: &mut Glm53State) -> m3tis_cuda::Result<()> {
        state.reset(&self.compute_stream)?;
        if let Some(dr) = self.drafter.as_mut() {
            dr.reset(&self.compute_stream)?;
        }
        self.compute_stream.synchronize()
    }

    /// Run a complete GLM-5.3-Flash forward pass over `tokens` (chunked to `state.max_rows`),
    /// advancing `state.pos`, optionally ingesting DFlash2 target taps, and returning the last token's FP32 logits.
    pub fn forward(
        &mut self,
        state: &mut Glm53State,
        tokens: &[i32],
        opts: ForwardOpts,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if tokens.is_empty() {
            return Err("empty token slice".into());
        }
        if state.pos + tokens.len() > state.max_ctx {
            return Err(format!(
                "sequence length {} (pos {} + {} tokens) exceeds --max-ctx {}",
                state.pos + tokens.len(),
                state.pos,
                tokens.len(),
                state.max_ctx
            )
            .into());
        }
        let has_dr = self.drafter.is_some();
        if tokens.len() > 1 && state.pf.is_some() {
            let pf_max = state.pf.as_ref().unwrap().max_rows;
            let total_chunks = (tokens.len() + pf_max - 1) / pf_max;
            let t_all = Instant::now();
            let mut chunk_idx = 0usize;
            let mut at = 0usize;
            while at < tokens.len() {
                let end = (at + pf_max).min(tokens.len());
                let is_last = end == tokens.len();
                self.forward_prefill_streaming(state, &tokens[at..end], has_dr, is_last)?;
                chunk_idx += 1;
                if total_chunks >= 2 {
                    eprintln!(
                        "  [prefill-stream] batch {chunk_idx}/{total_chunks} ({end}/{} tokens, {:.2}s elapsed)",
                        tokens.len(),
                        t_all.elapsed().as_secs_f64()
                    );
                }
                at = end;
            }
            return Ok(state.h_logits.as_f32_mut()[..self.cfg.vocab].to_vec());
        }
        let max_r = state.max_rows;
        let multi_chunk = tokens.len() > max_r;
        let total_chunks = (tokens.len() + max_r - 1) / max_r;
        let t_all = Instant::now();
        let mut chunk_idx = 0usize;
        let mut at = 0usize;
        while at < tokens.len() {
            let end = (at + max_r).min(tokens.len());
            let is_last = end == tokens.len();
            let pos0 = state.pos;
            let mode = if tokens.len() == 1 {
                ChunkMode::DecodeSerial
            } else {
                ChunkMode::Prefill
            };
            self.forward_chunk(state, &tokens[at..end], opts, mode, has_dr, is_last)?;
            if let Some(dr) = self.drafter.as_ref() {
                dr.ingest(
                    &self.compute_stream,
                    &self.lin,
                    &self.glm,
                    state.taps_buf.ptr(),
                    pos0,
                    end - at,
                )?;
            }
            chunk_idx += 1;
            if multi_chunk && total_chunks >= 4 {
                eprintln!(
                    "  [prefill] chunk {chunk_idx}/{total_chunks} ({end}/{} tokens, {:.1}s elapsed)",
                    tokens.len(),
                    t_all.elapsed().as_secs_f64()
                );
            }
            at = end;
        }
        if has_dr {
            self.compute_stream.synchronize()?;
        }
        Ok(state.h_logits.as_f32_mut()[..self.cfg.vocab].to_vec())
    }

    /// Run a batched speculative verify pass over `tokens` (`1..=max_verify` rows: `[anchor, d0, ..., d_{k-1}]`),
    /// leaving KDA recurrent & conv state uncommitted until `commit_verify(keep)` is called!
    /// Returns target greedy argmax token IDs for EVERY row in `tokens`.
    pub fn verify_block(
        &mut self,
        state: &mut Glm53State,
        tokens: &[i32],
        opts: ForwardOpts,
    ) -> Result<Vec<i32>, Box<dyn Error>> {
        self.verify_block_probs(state, tokens, None, opts)
    }

    pub fn verify_block_probs(
        &mut self,
        state: &mut Glm53State,
        tokens: &[i32],
        row_cum_probs: Option<&[f32]>,
        opts: ForwardOpts,
    ) -> Result<Vec<i32>, Box<dyn Error>> {
        let rows = tokens.len();
        if rows == 0 || rows > state.max_verify {
            return Err(format!("verify_block rows={rows} out of 1..={}", state.max_verify).into());
        }
        let has_dr = self.drafter.is_some();
        let eff_rows = self.forward_chunk_tree(
            state,
            tokens,
            opts,
            ChunkMode::Verify,
            has_dr,
            true,
            None,
            row_cum_probs,
        )?;
        Ok(state.h_argmax_ids.as_i32_mut()[..eff_rows].to_vec())
    }

    /// Milestone 8: Run ONE batched 2-branch speculative tree verify pass over
    /// `[trunk (len = trunk_len) | Branch A | Branch B (starting at branch_b_start)]` (`rows <= max_verify`)!
    /// All dense weights & MoE expert loads are 100% shared; KDA & MLA mask Branch B cleanly from Branch A.
    pub fn verify_tree(
        &mut self,
        state: &mut Glm53State,
        tokens: &[i32],
        trunk_len: usize,
        branch_b_start: usize,
        opts: ForwardOpts,
    ) -> Result<Vec<i32>, Box<dyn Error>> {
        let rows = tokens.len();
        if rows == 0 || rows > state.max_verify {
            return Err(format!("verify_tree rows={rows} out of 1..={}", state.max_verify).into());
        }
        let has_dr = self.drafter.is_some();
        let eff_rows = self.forward_chunk_tree(
            state,
            tokens,
            opts,
            ChunkMode::Verify,
            has_dr,
            true,
            Some((trunk_len, branch_b_start)),
            None,
        )?;
        Ok(state.h_argmax_ids.as_i32_mut()[..eff_rows].to_vec())
    }

    /// Exact lossless greedy acceptance evaluator for 1- or 2-branch DFlash2 draft trees.
    /// Supports lossless mid-forward speculative prefix pruning (`pred_ids.len() < 1 + tree.primary.len()`).
    /// Returns `(accepted_draft_tokens, next_anchor_token, branch_b_won)`.
    fn eval_tree_acceptance(
        &self,
        tree: &DraftTree,
        pred_ids: &[i32],
        branch_b_start: usize,
    ) -> (Vec<i32>, i32, bool) {
        let max_p = tree.primary.len().min(pred_ids.len().saturating_sub(1));
        let eff_fork = tree.fork_pos.min(max_p);
        let mut accepted = Vec::with_capacity(max_p);
        // 1. Match shared trunk `primary[0..eff_fork]`
        for i in 0..eff_fork {
            if pred_ids[i] == tree.primary[i] {
                accepted.push(tree.primary[i]);
                if self.cfg.eos_tokens.contains(&tree.primary[i]) {
                    return (accepted, pred_ids[i + 1], false);
                }
            } else {
                return (accepted, pred_ids[i], false);
            }
        }
        // 2. Check fork decision at `pred_ids[eff_fork]`
        if eff_fork < max_p && pred_ids[eff_fork] == tree.primary[eff_fork] {
            // Branch A won!
            for i in eff_fork..max_p {
                if pred_ids[i] == tree.primary[i] {
                    accepted.push(tree.primary[i]);
                    if self.cfg.eos_tokens.contains(&tree.primary[i]) {
                        return (accepted, pred_ids[i + 1], false);
                    }
                } else {
                    return (accepted, pred_ids[i], false);
                }
            }
            return (accepted, pred_ids[max_p], false);
        }
        if !tree.branch_b.is_empty()
            && branch_b_start > 0
            && branch_b_start + tree.branch_b.len() <= pred_ids.len()
            && pred_ids[eff_fork] == tree.branch_b[0]
        {
            // Branch B won!!
            accepted.push(tree.branch_b[0]);
            if self.cfg.eos_tokens.contains(&tree.branch_b[0]) {
                return (accepted, pred_ids[branch_b_start], true);
            }
            for j in 1..tree.branch_b.len() {
                if pred_ids[branch_b_start + j - 1] == tree.branch_b[j] {
                    accepted.push(tree.branch_b[j]);
                    if self.cfg.eos_tokens.contains(&tree.branch_b[j]) {
                        return (accepted, pred_ids[branch_b_start + j], true);
                    }
                } else {
                    return (accepted, pred_ids[branch_b_start + j - 1], true);
                }
            }
            let last_b = branch_b_start + tree.branch_b.len() - 1;
            return (accepted, pred_ids[last_b], true);
        }
        // Neither branch matched at eff_fork (or batch was pruned at eff_fork) -> emit trunk + target's true token!
        (accepted, pred_ids[eff_fork], false)
    }

    /// Commit `keep` (`1..=rows`) accepted rows from the preceding `verify_block`:
    /// swaps/replays KDA recurrent state across all 34 KDA layers, shifts all 34 KDA conv windows,
    /// ingests the `keep` accepted target taps into DFlash2's KV rings, and advances `state.pos += keep`.
    pub fn commit_verify(
        &mut self,
        st: &mut Glm53State,
        rows: usize,
        keep: usize,
    ) -> Result<(), Box<dyn Error>> {
        assert!(keep >= 1 && keep <= rows && rows <= st.max_verify);
        let kda_n = self.cfg.kda_layers();
        let cur = st.kda_cur;
        let alt = 1 - cur;
        if keep < rows {
            let state_stride = (self.cfg.kda_heads * self.cfg.kda_dim * self.cfg.kda_dim) as i64;
            let kv_stride = (st.max_verify * self.cfg.kda_heads * self.cfg.kda_dim) as i64;
            let b_stride = (st.max_verify * self.cfg.kda_heads) as i64;
            self.glm.kda_replay_layers(
                &self.compute_stream,
                kda_n,
                self.cfg.kda_heads,
                st.kda_rec[cur].ptr(),
                state_stride,
                st.kda_k_save.ptr(),
                st.kda_v_save.ptr(),
                st.kda_g_save.ptr(),
                st.kda_b_save.ptr(),
                kv_stride,
                b_stride,
                keep,
                st.kda_rec[alt].ptr(),
                0,
                0,
            )?;
        }
        st.kda_cur = alt;
        self.glm.kda_conv_shift_all(
            &self.compute_stream,
            kda_n,
            st.kda_conv.ptr(),
            st.kda_p_save.ptr(),
            keep,
            st.max_verify,
            self.cfg.kda_conv_c(),
            0,
            0,
        )?;
        let pos0 = st.pos;
        if let Some(dr) = self.drafter.as_ref() {
            dr.ingest(
                &self.compute_stream,
                &self.lin,
                &self.glm,
                st.taps_buf.ptr(),
                pos0,
                keep,
            )?;
        }
        st.pos += keep;
        Ok(())
    }

    /// Milestone 8: Commit `keep` accepted rows from a 2-branch `verify_tree` pass!
    /// When `branch_b_won == true`, remaps KDA replay + conv shift, compacts `mla_kv` + `dsa.ik`/`ig`
    /// + refreshes `dsa.pk`, and ingests `[trunk, Branch B]` taps into DFlash2 with 0 host copies!
    pub fn commit_verify_tree(
        &mut self,
        st: &mut Glm53State,
        rows: usize,
        trunk_len: usize,
        branch_b_start: usize,
        keep: usize,
        branch_b_won: bool,
    ) -> Result<(), Box<dyn Error>> {
        assert!(keep >= 1 && keep <= rows && rows <= st.max_verify);
        let kda_n = self.cfg.kda_layers();
        let cur = st.kda_cur;
        let alt = 1 - cur;
        let (tt, tb) = if branch_b_won {
            (trunk_len, branch_b_start)
        } else {
            (0, 0)
        };
        // Note: in tree mode, `state_out` held the fork checkpoint, so always replay exact `keep` steps!
        let state_stride = (self.cfg.kda_heads * self.cfg.kda_dim * self.cfg.kda_dim) as i64;
        let kv_stride = (st.max_verify * self.cfg.kda_heads * self.cfg.kda_dim) as i64;
        let b_stride = (st.max_verify * self.cfg.kda_heads) as i64;
        self.glm.kda_replay_layers(
            &self.compute_stream,
            kda_n,
            self.cfg.kda_heads,
            st.kda_rec[cur].ptr(),
            state_stride,
            st.kda_k_save.ptr(),
            st.kda_v_save.ptr(),
            st.kda_g_save.ptr(),
            st.kda_b_save.ptr(),
            kv_stride,
            b_stride,
            keep,
            st.kda_rec[alt].ptr(),
            tt,
            tb,
        )?;
        st.kda_cur = alt;
        self.glm.kda_conv_shift_all(
            &self.compute_stream,
            kda_n,
            st.kda_conv.ptr(),
            st.kda_p_save.ptr(),
            keep,
            st.max_verify,
            self.cfg.kda_conv_c(),
            tt,
            tb,
        )?;
        let pos0 = st.pos;
        if branch_b_won {
            let acc_b = keep - trunk_len;
            let mla_n = self.cfg.mla_layers();
            if st.kv_int8 {
                let kv_sc = st.mla_kv_sc.as_ref().expect("kv_int8 requires mla_kv_sc");
                self.glm.mla_compact_kv_q8(
                    &self.compute_stream,
                    mla_n,
                    st.mla_kv.ptr(),
                    kv_sc.ptr(),
                    pos0 + trunk_len,
                    pos0 + branch_b_start,
                    st.max_ctx,
                    acc_b,
                    self.cfg.kv_lora,
                )?;
            } else {
                self.glm.mla_compact_kv_bf16(
                    &self.compute_stream,
                    mla_n,
                    st.mla_kv.ptr(),
                    pos0 + trunk_len,
                    pos0 + branch_b_start,
                    st.max_ctx,
                    acc_b,
                    self.cfg.kv_lora,
                )?;
            }
            if let Some(dsa) = st.dsa.as_ref() {
                self.glm.dsa_compact_ik_ig(
                    &self.compute_stream,
                    mla_n,
                    dsa.ik.ptr(),
                    dsa.ig.ptr(),
                    pos0 + trunk_len,
                    pos0 + branch_b_start,
                    dsa.ik_cap,
                    acc_b,
                )?;
                let ik_stride = dsa.ik_cap * DSA_DIM * 2;
                let pk_stride = dsa.pk_cap * DSA_DIM * 2;
                for layer in &self.layers {
                    if let MixerWeights::Mla { idx, w } = &layer.mixer {
                        if let Some(ix) = w.indexer.as_ref() {
                            self.glm.dsa_pool_keys(
                                &self.compute_stream,
                                dsa.ik.at(idx * ik_stride),
                                dsa.ig.at(idx * ik_stride),
                                ix.ape.ptr(),
                                dsa.pk.at(idx * pk_stride),
                                pos0 + trunk_len,
                                acc_b,
                                dsa.ik_cap,
                                dsa.pk_cap,
                            )?;
                        }
                    }
                }
            }
            if let Some(dr) = self.drafter.as_ref() {
                let tap_row_bytes = 5 * self.cfg.hidden * 2;
                dr.ingest(
                    &self.compute_stream,
                    &self.lin,
                    &self.glm,
                    st.taps_buf.ptr(),
                    pos0,
                    trunk_len,
                )?;
                dr.ingest(
                    &self.compute_stream,
                    &self.lin,
                    &self.glm,
                    st.taps_buf.at(branch_b_start * tap_row_bytes),
                    pos0 + trunk_len,
                    acc_b,
                )?;
            }
        } else if let Some(dr) = self.drafter.as_ref() {
            dr.ingest(
                &self.compute_stream,
                &self.lin,
                &self.glm,
                st.taps_buf.ptr(),
                pos0,
                keep,
            )?;
        }
        st.pos += keep;
        Ok(())
    }

    /// Generate up to `max_tokens` greedily starting from `first_token` (after prefill),
    /// using DFlash2 speculative decoding (`block_size = 8`, 2-Branch Speculative Tree when enabled!) when loaded.
    pub fn generate_greedy_dflash2(
        &mut self,
        st: &mut Glm53State,
        first_token: i32,
        max_tokens: usize,
        max_drafts: usize,
        conf_threshold: f64,
        opts: ForwardOpts,
    ) -> Result<(Vec<i32>, SpecDecodeStats), Box<dyn Error>> {
        let mut generated = Vec::with_capacity(max_tokens);
        generated.push(first_token);
        let mut stats = SpecDecodeStats::default();
        let mut anchor = first_token;
        let d = self.cfg.hidden;
        let vocab = self.cfg.vocab;
        let mut ctrl = AdaptiveDraftCtrl::new(opts.adaptive_draft, max_drafts);

        while generated.len() < max_tokens && !self.cfg.eos_tokens.contains(&anchor) {
            let (cur_k, is_probe) = ctrl.next_step();
            if cur_k == 0 {
                // Ultra-fast serial step while keeping DFlash2 target KV rings 100% synchronized!
                let tv0 = Instant::now();
                let logits = self.forward(st, &[anchor], opts)?;
                let dt_s = tv0.elapsed().as_secs_f64();
                stats.verify_secs += dt_s;
                ctrl.record_serial(dt_s, is_probe);
                let (best_id, _) = logits
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv { (i, v) } else { (bi, bv) }
                    });
                let next_tok = best_id as i32;
                generated.push(next_tok);
                anchor = next_tok;
                continue;
            }
            // 1. Propose 1- or 2-branch speculative tree using ONE DFlash2 GPU pass (~2-4 ms!)
            let tr_start = Instant::now();
            let td0 = Instant::now();
            let tree: DraftTree = {
                let dr = self.drafter.as_mut().ok_or("DFlash2 drafter not loaded")?;
                let lin = &self.lin;
                let glm = &self.glm;
                let lm_head = &self.lm_head;
                let embed_slice = self.embed_host.as_slice();
                if opts.tree_verify && st.max_verify > DF_BLOCK {
                    dr.propose_tree(
                        &self.compute_stream,
                        lin,
                        glm,
                        embed_slice,
                        |stream, x_ptr, out_ptr, r| {
                            lm_head.apply(lin, stream, x_ptr, d, out_ptr, vocab, r, true)
                        },
                        anchor,
                        st.pos,
                        cur_k,
                        conf_threshold,
                        st.max_verify,
                    )?
                } else {
                    let (primary, row_cum_probs) = dr.propose(
                        &self.compute_stream,
                        lin,
                        glm,
                        embed_slice,
                        |stream, x_ptr, out_ptr, r| {
                            lm_head.apply(lin, stream, x_ptr, d, out_ptr, vocab, r, true)
                        },
                        anchor,
                        st.pos,
                        cur_k,
                        conf_threshold,
                    )?;
                    let fpos = primary.len();
                    DraftTree {
                        primary,
                        row_cum_probs,
                        fork_pos: fpos,
                        branch_b: Vec::new(),
                    }
                }
            };
            stats.draft_secs += td0.elapsed().as_secs_f64();

            // 2. Verify full 1- or 2-branch tree in ONE batched target pass!
            let has_b = !tree.branch_b.is_empty();
            let trunk_len = 1 + tree.fork_pos;
            let branch_b_start = if has_b { 1 + tree.primary.len() } else { 0 };
            let mut verify_toks = Vec::with_capacity(1 + tree.primary.len() + tree.branch_b.len());
            verify_toks.push(anchor);
            verify_toks.extend_from_slice(&tree.primary);
            verify_toks.extend_from_slice(&tree.branch_b);
            let rows = verify_toks.len();

            let tv0 = Instant::now();
            let pred_ids = if has_b {
                self.verify_tree(st, &verify_toks, trunk_len, branch_b_start, opts)?
            } else {
                self.verify_block_probs(st, &verify_toks, Some(&tree.row_cum_probs), opts)?
            };
            stats.verify_secs += tv0.elapsed().as_secs_f64();

            // 3. Exact lossless greedy acceptance (Trunk -> Branch A or Branch B!)
            let (accepted_toks, next_anchor, branch_b_won) =
                self.eval_tree_acceptance(&tree, &pred_ids, branch_b_start);
            let accepted = accepted_toks.len();
            let keep = accepted + 1;
            stats.rounds += 1;
            stats.total_drafted += tree.primary.len();
            stats.total_accepted += accepted;
            if branch_b_won {
                stats.tree_b_wins += 1;
                stats.tree_b_accepted += accepted - tree.fork_pos;
            }

            let mut hit_eos = false;
            for &dtok in &accepted_toks {
                if generated.len() < max_tokens {
                    generated.push(dtok);
                    if self.cfg.eos_tokens.contains(&dtok) {
                        hit_eos = true;
                        break;
                    }
                }
            }
            if !hit_eos && generated.len() < max_tokens {
                generated.push(next_anchor);
                if self.cfg.eos_tokens.contains(&next_anchor) {
                    hit_eos = true;
                }
            }

            // 4. Commit accepted path (KDA replay + conv shift + Branch B KV/DSA compact + DFlash2 ingest)
            let tc0 = Instant::now();
            if has_b {
                self.commit_verify_tree(st, rows, trunk_len, branch_b_start, keep, branch_b_won)?;
            } else {
                self.commit_verify(st, rows, keep)?;
            }
            stats.commit_secs += tc0.elapsed().as_secs_f64();
            ctrl.record_round(keep, tr_start.elapsed().as_secs_f64(), is_probe);

            if hit_eos {
                break;
            }
            anchor = next_anchor;
        }
        self.compute_stream.synchronize()?;
        Ok((generated, stats))
    }

    /// 100% LOSSLESS Zero-NVMe Resident-Only Self-Speculation (`--self-spec K`):
    /// Drafts up to `K` (`1..=7`) tokens using ONLY already-VRAM-resident experts (`0` NVMe reads,
    /// `0` PCIe HtoD, `0` CPU-GPU syncs across all 45 layers, `0 MiB` extra drafter VRAM, works at ANY `max_ctx`!),
    /// then verifies `[anchor, d_0, ..., d_{k-1}]` in ONE batched exact target pass via `verify_block` + `commit_verify`.
    /// Output is 100% bit-exact to serial decode!
    pub fn generate_greedy_self_spec(
        &mut self,
        st: &mut Glm53State,
        first_token: i32,
        max_tokens: usize,
        draft_k: usize,
        opts: ForwardOpts,
    ) -> Result<(Vec<i32>, SpecDecodeStats), Box<dyn Error>> {
        let k_max = draft_k.clamp(1, st.max_verify.saturating_sub(1));
        let mut generated = Vec::with_capacity(max_tokens);
        generated.push(first_token);
        let mut stats = SpecDecodeStats::default();
        let mut anchor = first_token;
        let rec_bytes = st.kda_rec[0].len();
        let conv_bytes = st.kda_conv.len();

        while generated.len() < max_tokens && !self.cfg.eos_tokens.contains(&anchor) {
            if st.pos + 1 >= st.max_ctx {
                break;
            }
            let max_room = (st.max_ctx - 1 - st.pos).min(k_max);

            // 1. Draft up to `max_room` tokens on GPU using ONLY VRAM-resident experts (ZERO NVMe!)
            let td0 = Instant::now();
            let saved_pos = st.pos;
            let saved_cur = st.kda_cur;
            let scratch_cur = 1 - saved_cur;
            self.compute_stream
                .dtod_async(st.kda_rec[scratch_cur].ptr(), st.kda_rec[saved_cur].ptr(), rec_bytes)?;
            self.compute_stream
                .dtod_async(st.kda_conv_bak.ptr(), st.kda_conv.ptr(), conv_bytes)?;
            st.kda_cur = scratch_cur;

            let mut drafts = Vec::with_capacity(max_room);
            let mut oracle_keys: Vec<usize> = Vec::new();
            let mut oracle_cursor = 0usize;
            let mut cur_tok = anchor;
            let e_cnt = self.cfg.num_experts;
            let moe_n = self.cfg.moe_layers();
            let slots = self.cfg.top_k + 1;
            for step_i in 0..max_room {
                self.forward_chunk(st, &[cur_tok], opts, ChunkMode::DraftResident, false, true)?;
                let pred = st.h_argmax_ids.as_i32_mut()[0];
                let prob = st.h_argmax_probs.as_f32_mut()[0];
                drafts.push(pred);
                if step_i == 0 {
                    let hp = &st.h_all_picks.as_i32_mut()[..moe_n * slots];
                    for ml in 0..moe_n {
                        let base_key = ml * e_cnt;
                        for k_rank in 0..self.cfg.top_k {
                            let p = hp[ml * slots + k_rank];
                            if p >= 0 && (p as usize) < e_cnt {
                                let gk = base_key + (p as usize);
                                if self.pool.vram_cache.peek(gk).is_none()
                                    && self.pool.ram_cache.peek(gk).is_none()
                                    && self.pool.ram_cache.pending(gk).is_none()
                                    && !oracle_keys.contains(&gk)
                                {
                                    oracle_keys.push(gk);
                                }
                            }
                        }
                    }
                }
                if oracle_cursor < oracle_keys.len() {
                    let _ = self.pool.poll_spec_only();
                    if self.pool.spec_inflight_count() == 0 {
                        let submitted = self
                            .pool
                            .start_spec_prefetch(&oracle_keys[oracle_cursor..], 8)
                            .unwrap_or(0);
                        oracle_cursor += submitted.max(1);
                    }
                }
                if self.cfg.eos_tokens.contains(&pred) {
                    break;
                }
                if opts.self_spec_conf > 0.0 && prob < opts.self_spec_conf {
                    break;
                }
                cur_tok = pred;
            }

            // Restore exact pre-draft KDA/conv/pos state (verify_block overwrites mla_kv/dsa slots in place!)
            st.pos = saved_pos;
            st.kda_cur = saved_cur;
            self.compute_stream
                .dtod_async(st.kda_conv.ptr(), st.kda_conv_bak.ptr(), conv_bytes)?;
            stats.draft_secs += td0.elapsed().as_secs_f64();

            // 2. Verify [anchor, d_0, ..., d_{m-1}] in ONE batched exact target pass!
            let mut verify_toks = Vec::with_capacity(1 + drafts.len());
            verify_toks.push(anchor);
            verify_toks.extend_from_slice(&drafts);
            let rows = verify_toks.len();

            let tv0 = Instant::now();
            let pred_ids = self.verify_block(st, &verify_toks, opts)?;
            stats.verify_secs += tv0.elapsed().as_secs_f64();

            // 3. Exact lossless greedy acceptance
            let mut accepted = 0usize;
            for i in 0..drafts.len().min(pred_ids.len().saturating_sub(1)) {
                if pred_ids[i] == drafts[i] {
                    accepted += 1;
                    if self.cfg.eos_tokens.contains(&drafts[i]) {
                        break;
                    }
                } else {
                    break;
                }
            }
            let keep = accepted + 1;
            stats.rounds += 1;
            stats.total_drafted += drafts.len();
            stats.total_accepted += accepted;

            let mut hit_eos = false;
            for &dtok in &drafts[..accepted] {
                if generated.len() < max_tokens {
                    generated.push(dtok);
                    if self.cfg.eos_tokens.contains(&dtok) {
                        hit_eos = true;
                        break;
                    }
                }
            }
            let next_anchor = pred_ids[accepted];
            if !hit_eos && generated.len() < max_tokens {
                generated.push(next_anchor);
                if self.cfg.eos_tokens.contains(&next_anchor) {
                    hit_eos = true;
                }
            }

            // 4. Commit exact accepted prefix
            let tc0 = Instant::now();
            self.commit_verify(st, rows, keep)?;
            stats.commit_secs += tc0.elapsed().as_secs_f64();

            if hit_eos {
                break;
            }
            anchor = next_anchor;
        }
        self.compute_stream.synchronize()?;
        Ok((generated, stats))
    }

    fn forward_chunk(
        &mut self,
        st: &mut Glm53State,
        tokens: &[i32],
        opts: ForwardOpts,
        mode: ChunkMode,
        capture_taps: bool,
        compute_logits: bool,
    ) -> Result<(), Box<dyn Error>> {
        let _ = self.forward_chunk_tree(st, tokens, opts, mode, capture_taps, compute_logits, None, None)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_chunk_tree(
        &mut self,
        st: &mut Glm53State,
        tokens: &[i32],
        opts: ForwardOpts,
        mode: ChunkMode,
        capture_taps: bool,
        compute_logits: bool,
        tree: Option<(usize, usize)>,
        row_cum_probs: Option<&[f32]>,
    ) -> Result<usize, Box<dyn Error>> {
        let (tree_trunk_len, tree_b_start) = tree.unwrap_or((0, 0));
        let mut rows = tokens.len();
        let d = self.cfg.hidden;
        let s = self.cfg.hc_mult;
        let row_bytes = d * 2;
        let is_decode_step = mode != ChunkMode::Prefill;
        let is_verify = mode == ChunkMode::Verify;
        let is_draft_resident = mode == ChunkMode::DraftResident;
        let is_warp_zero_sync = is_decode_step
            && !is_draft_resident
            && opts.zero_stall_mass > 0.0
            && st.warp_heal_steps == 0
            && rows <= st.max_verify;
        let use_zero_sync_moe = is_draft_resident || is_warp_zero_sync;
        if use_zero_sync_moe {
            self.pool.flush_dev_map_if_dirty(&self.compute_stream)?;
        }
        let do_prof = is_decode_step && std::env::var_os("M3TIS_DECODE_PROFILE").is_some();
        if do_prof {
            self.decode_prof.calls += 1;
            self.decode_prof.total_rows += rows as u64;
        }

        // 1. Copy token embeddings from Pinned Host RAM -> GPU (8 KiB/token!) and broadcast to 4 mHC streams
        {
            let emb_src = self.embed_host.as_slice();
            let emb_dst = st.h_embed_rows.as_mut_slice();
            for (r, &tok) in tokens.iter().enumerate() {
                if tok < 0 || (tok as usize) >= self.cfg.vocab {
                    return Err(format!("token id {tok} out of range 0..{}", self.cfg.vocab).into());
                }
                let src_off = (tok as usize) * row_bytes;
                let dst_off = r * row_bytes;
                emb_dst[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&emb_src[src_off..src_off + row_bytes]);
            }
            unsafe {
                self.compute_stream.htod_async(
                    st.d_embed_rows.ptr(),
                    st.h_embed_rows.as_ptr(),
                    rows * row_bytes,
                )?;
            }
            self.glm.embed_broadcast_bf16(
                &self.compute_stream,
                st.d_embed_rows.ptr(),
                st.x.ptr(),
                rows,
                d,
                s,
            )?;
        }

        let kda_rec_stride = self.cfg.kda_heads * self.cfg.kda_dim * self.cfg.kda_dim * 4;
        let kda_conv_stride = 3 * self.cfg.kda_conv_c() * 2;
        let mla_kv_stride = if st.kv_int8 {
            st.max_ctx * self.cfg.kv_lora
        } else {
            st.max_ctx * self.cfg.kv_lora * 2
        };
        let mla_kv_sc_stride = st.max_ctx * 4;
        let p_stride = self.cfg.kda_p_stride();
        let b_off = self.cfg.kda_b_off();
        let n_kda = self.cfg.kda_heads * self.cfg.kda_dim;
        let slots = self.cfg.top_k + 1;
        let vram_slots = self.pool.vram_slots();

        // 2. Run all 45 layers
        for l in 0..self.layers.len() {
            // --- A. Attention / Mixer Sublayer ---
            let tp_attn = if do_prof { self.compute_stream.synchronize()?; Some(Instant::now()) } else { None };
            {
                let layer = &self.layers[l];
                self.glm.hc_pre(
                    &self.compute_stream,
                    st.x.ptr(),
                    layer.attn_hc.fn_w.ptr(),
                    st.hc_part.ptr(),
                    layer.attn_hc.base.ptr(),
                    layer.attn_hc.scale.ptr(),
                    layer.in_norm.ptr(),
                    st.normed.ptr(),
                    st.post.ptr(),
                    st.comb.ptr(),
                    rows,
                    d,
                    self.cfg.rms_eps,
                    self.cfg.hc_eps,
                    self.cfg.hc_sinkhorn_iters,
                )?;

                match &layer.mixer {
                    MixerWeights::Kda { idx, w } => {
                        // Fused [q | k | v] = 24576 contiguous columns of kda_p in ONE launch!
                        w.qkv_proj.apply(
                            &self.lin,
                            &self.compute_stream,
                            st.normed.ptr(),
                            d,
                            st.kda_p.ptr(),
                            p_stride,
                            rows,
                            false,
                        )?;
                        let fa_off = 3 * n_kda;
                        // Fused [f_a (128) | g_a (128) | b (64)] = 320 contiguous columns of kda_p in ONE launch!
                        self.lin.linear_bf16(
                            &self.compute_stream,
                            st.normed.ptr(),
                            d,
                            w.fgb_proj.ptr(),
                            st.kda_p.at(fa_off * 2),
                            p_stride,
                            rows,
                            2 * self.cfg.kda_dim + self.cfg.kda_heads,
                            d,
                            false,
                        )?;

                        // f_b, g_b from strided slices of kda_p
                        self.lin.linear_bf16(
                            &self.compute_stream,
                            st.kda_p.at(fa_off * 2),
                            p_stride,
                            w.f_b_proj.ptr(),
                            st.kda_ka.ptr(),
                            n_kda,
                            rows,
                            n_kda,
                            self.cfg.kda_dim,
                            false,
                        )?;
                        self.lin.linear_bf16(
                            &self.compute_stream,
                            st.kda_p.at((fa_off + self.cfg.kda_dim) * 2),
                            p_stride,
                            w.g_b_proj.ptr(),
                            st.kda_kg.ptr(),
                            n_kda,
                            rows,
                            n_kda,
                            self.cfg.kda_dim,
                            false,
                        )?;

                        let cur = st.kda_cur;
                        let rec_in = st.kda_rec[cur].at(idx * kda_rec_stride);
                        let conv_ptr = st.kda_conv.at(idx * kda_conv_stride);
                        if is_verify {
                            let alt = 1 - cur;
                            let rec_out = st.kda_rec[alt].at(idx * kda_rec_stride);
                            let kv_layer_elems = st.max_verify * n_kda;
                            let b_layer_elems = st.max_verify * self.cfg.kda_heads;
                            let p_layer_elems = st.max_verify * self.cfg.kda_conv_c();
                            self.glm.kda_chain_save(
                                &self.compute_stream,
                                self.cfg.kda_heads,
                                st.kda_p.ptr(),
                                p_stride,
                                b_off,
                                st.kda_ka.ptr(),
                                st.kda_kg.ptr(),
                                conv_ptr,
                                w.conv_w.ptr(),
                                rec_in,
                                w.a_log.ptr(),
                                w.dt_bias.ptr(),
                                w.o_norm.ptr(),
                                self.cfg.rms_eps,
                                self.cfg.kda_lower,
                                rows,
                                st.kda_out.ptr(),
                                rec_out,
                                st.kda_k_save.at(idx * kv_layer_elems * 4),
                                st.kda_v_save.at(idx * kv_layer_elems * 2),
                                st.kda_g_save.at(idx * kv_layer_elems * 4),
                                st.kda_b_save.at(idx * b_layer_elems * 4),
                                st.kda_p_save.at(idx * p_layer_elems * 2),
                                tree_trunk_len,
                                tree_b_start,
                            )?;
                        } else {
                            self.glm.kda_chain(
                                &self.compute_stream,
                                self.cfg.kda_heads,
                                st.kda_p.ptr(),
                                p_stride,
                                b_off,
                                st.kda_ka.ptr(),
                                st.kda_kg.ptr(),
                                conv_ptr,
                                w.conv_w.ptr(),
                                rec_in,
                                w.a_log.ptr(),
                                w.dt_bias.ptr(),
                                w.o_norm.ptr(),
                                self.cfg.rms_eps,
                                self.cfg.kda_lower,
                                rows,
                                st.kda_out.ptr(),
                                rec_in,
                            )?;
                            self.glm.kda_conv_shift(
                                &self.compute_stream,
                                conv_ptr,
                                st.kda_p.ptr(),
                                rows,
                                p_stride,
                                self.cfg.kda_conv_c(),
                            )?;
                        }
                        w.o_proj.apply(
                            &self.lin,
                            &self.compute_stream,
                            st.kda_out.ptr(),
                            n_kda,
                            st.branch.ptr(),
                            d,
                            rows,
                            true,
                        )?;
                    }
                    MixerWeights::Mla { idx, w } => {
                        w.q_a_proj.apply(
                            &self.lin,
                            &self.compute_stream,
                            st.normed.ptr(),
                            d,
                            st.mla_qa_raw.ptr(),
                            self.cfg.q_lora,
                            rows,
                            false,
                        )?;
                        w.kv_a_proj.apply(
                            &self.lin,
                            &self.compute_stream,
                            st.normed.ptr(),
                            d,
                            st.mla_kva_raw.ptr(),
                            self.cfg.kv_lora,
                            rows,
                            false,
                        )?;
                        self.glm.rmsnorm_bf16(
                            &self.compute_stream,
                            st.mla_qa_raw.ptr(),
                            self.cfg.q_lora,
                            w.q_a_norm.ptr(),
                            st.mla_qr.ptr(),
                            self.cfg.q_lora,
                            rows,
                            self.cfg.q_lora,
                            self.cfg.rms_eps,
                        )?;
                        self.glm.rmsnorm_bf16(
                            &self.compute_stream,
                            st.mla_kva_raw.ptr(),
                            self.cfg.kv_lora,
                            w.kv_a_norm.ptr(),
                            st.mla_lat.ptr(),
                            self.cfg.kv_lora,
                            rows,
                            self.cfg.kv_lora,
                            self.cfg.rms_eps,
                        )?;
                        w.q_b_proj.apply(
                            &self.lin,
                            &self.compute_stream,
                            st.mla_qr.ptr(),
                            self.cfg.q_lora,
                            st.mla_q.ptr(),
                            self.cfg.mla_heads * self.cfg.qk_nope,
                            rows,
                            false,
                        )?;
                        self.glm.mla_absorb_q(
                            &self.compute_stream,
                            st.mla_q.ptr(),
                            w.wk.ptr(),
                            st.mla_qa.ptr(),
                            rows,
                            self.cfg.mla_heads,
                            self.cfg.qk_nope,
                            self.cfg.kv_lora,
                        )?;
                        let kv_ptr = st.mla_kv.at(idx * mla_kv_stride);
                        let kv_sc_ptr = st.mla_kv_sc.as_ref().map(|s| s.at(idx * mla_kv_sc_stride));
                        if let Some(sc_ptr) = kv_sc_ptr {
                            self.glm.mla_write_kv_q8(
                                &self.compute_stream,
                                st.mla_lat.ptr(),
                                kv_ptr,
                                sc_ptr,
                                st.pos,
                                st.max_ctx,
                                rows,
                                self.cfg.kv_lora,
                            )?;
                        } else {
                            self.glm.mla_write_kv(
                                &self.compute_stream,
                                st.mla_lat.ptr(),
                                kv_ptr,
                                st.pos,
                                st.max_ctx,
                                rows,
                                self.cfg.kv_lora,
                            )?;
                        }
                        let scale = 1.0f32 / (self.cfg.qk_nope as f32).sqrt();
                        let use_sparse = if let (Some(ix), Some(dsa)) = (w.indexer.as_ref(), st.dsa.as_ref()) {
                            // Always maintain LayerNorm'd index keys + compressed 4-token pools for long-context DSA!
                            self.lin.linear_bf16(
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                ix.wk.ptr(),
                                dsa.k_raw.ptr(),
                                DSA_DIM,
                                rows,
                                DSA_DIM,
                                d,
                                false,
                            )?;
                            self.lin.linear_bf16(
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                ix.w_proj.ptr(),
                                dsa.w_raw.ptr(),
                                DSA_HEADS,
                                rows,
                                DSA_HEADS,
                                d,
                                false,
                            )?;
                            self.lin.linear_bf16(
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                ix.gate.ptr(),
                                dsa.g_raw.ptr(),
                                DSA_DIM,
                                rows,
                                DSA_DIM,
                                d,
                                true,
                            )?;
                            let ik_stride = dsa.ik_cap * DSA_DIM * 2;
                            let pk_stride = dsa.pk_cap * DSA_DIM * 2;
                            let ik_ptr = dsa.ik.at(idx * ik_stride);
                            let ig_ptr = dsa.ig.at(idx * ik_stride);
                            let pk_ptr = dsa.pk.at(idx * pk_stride);
                            self.glm.dsa_index_write(
                                &self.compute_stream,
                                dsa.k_raw.ptr(),
                                dsa.g_raw.ptr(),
                                ix.k_norm_w.ptr(),
                                ix.k_norm_b.ptr(),
                                ik_ptr,
                                ig_ptr,
                                st.pos,
                                dsa.ik_cap,
                                1e-6,
                                rows,
                            )?;
                            self.glm.dsa_pool_keys(
                                &self.compute_stream,
                                ik_ptr,
                                ig_ptr,
                                ix.ape.ptr(),
                                pk_ptr,
                                st.pos,
                                rows,
                                dsa.ik_cap,
                                dsa.pk_cap,
                            )?;
                            if st.pos + rows > DSA_WIDTH {
                                // Context past 2051 tokens: score visible pools -> exact radix top-512 -> sparse MLA!
                                ix.wq_b.apply(
                                    &self.lin,
                                    &self.compute_stream,
                                    st.mla_qr.ptr(),
                                    self.cfg.q_lora,
                                    dsa.qi.ptr(),
                                    DSA_HEADS * DSA_DIM,
                                    rows,
                                    false,
                                )?;
                                let np_vis = ((st.pos + rows) / DSA_KPOOL).min(dsa.pk_cap).max(DSA_TOPK_POOLS + 1);
                                let qscale = (DSA_DIM as f32).powf(-0.5);
                                let wscale = (DSA_HEADS as f32).powf(-0.5);
                                self.glm.dsa_score_pools(
                                    &self.compute_stream,
                                    dsa.qi.ptr(),
                                    dsa.w_raw.ptr(),
                                    pk_ptr,
                                    dsa.scores.ptr(),
                                    st.pos,
                                    np_vis,
                                    dsa.pk_cap,
                                    DSA_HEADS,
                                    qscale,
                                    wscale,
                                    rows,
                                )?;
                                self.glm.dsa_select_tokens(
                                    &self.compute_stream,
                                    dsa.scores.ptr(),
                                    dsa.tokens.ptr(),
                                    dsa.counts.ptr(),
                                    st.pos,
                                    np_vis,
                                    DSA_TOPK_POOLS,
                                    DSA_TOKEN_STRIDE,
                                    rows,
                                    tree_trunk_len,
                                    tree_b_start,
                                )?;
                                if let Some(sc_ptr) = kv_sc_ptr {
                                    self.glm.mla_attend_sparse_q8(
                                        &self.compute_stream,
                                        st.mla_qa.ptr(),
                                        kv_ptr,
                                        sc_ptr,
                                        dsa.tokens.ptr(),
                                        dsa.counts.ptr(),
                                        st.mla_ol.ptr(),
                                        st.pos,
                                        st.max_ctx,
                                        DSA_TOKEN_STRIDE,
                                        rows,
                                        self.cfg.mla_heads,
                                        self.cfg.kv_lora,
                                        scale,
                                        tree_trunk_len,
                                        tree_b_start,
                                    )?;
                                } else {
                                    self.glm.mla_attend_sparse(
                                        &self.compute_stream,
                                        st.mla_qa.ptr(),
                                        kv_ptr,
                                        dsa.tokens.ptr(),
                                        dsa.counts.ptr(),
                                        st.mla_ol.ptr(),
                                        st.pos,
                                        st.max_ctx,
                                        DSA_TOKEN_STRIDE,
                                        rows,
                                        self.cfg.mla_heads,
                                        self.cfg.kv_lora,
                                        scale,
                                        tree_trunk_len,
                                        tree_b_start,
                                    )?;
                                }
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if !use_sparse {
                            if let Some(sc_ptr) = kv_sc_ptr {
                                self.glm.mla_attend_dense_q8(
                                    &self.compute_stream,
                                    st.mla_qa.ptr(),
                                    kv_ptr,
                                    sc_ptr,
                                    st.mla_ol.ptr(),
                                    st.pos,
                                    st.max_ctx,
                                    st.max_ctx,
                                    rows,
                                    self.cfg.mla_heads,
                                    self.cfg.kv_lora,
                                    scale,
                                    tree_trunk_len,
                                    tree_b_start,
                                )?;
                            } else {
                                self.glm.mla_attend_dense(
                                    &self.compute_stream,
                                    st.mla_qa.ptr(),
                                    kv_ptr,
                                    st.mla_ol.ptr(),
                                    st.pos,
                                    st.max_ctx,
                                    st.max_ctx,
                                    rows,
                                    self.cfg.mla_heads,
                                    self.cfg.kv_lora,
                                    scale,
                                    tree_trunk_len,
                                    tree_b_start,
                                )?;
                            }
                        }
                        self.glm.mla_expand_v(
                            &self.compute_stream,
                            st.mla_ol.ptr(),
                            w.wv.ptr(),
                            st.mla_vn.ptr(),
                            rows,
                            self.cfg.mla_heads,
                            self.cfg.dv,
                            self.cfg.kv_lora,
                        )?;
                        w.o_proj.apply(
                            &self.lin,
                            &self.compute_stream,
                            st.mla_vn.ptr(),
                            self.cfg.mla_heads * self.cfg.dv,
                            st.branch.ptr(),
                            d,
                            rows,
                            true,
                        )?;
                    }
                }

                self.glm.hc_post(
                    &self.compute_stream,
                    st.x.ptr(),
                    st.branch.ptr(),
                    st.post.ptr(),
                    st.comb.ptr(),
                    rows,
                    d,
                )?;
            }
            if let Some(t) = tp_attn {
                self.compute_stream.synchronize()?;
                self.decode_prof.attn_secs += t.elapsed().as_secs_f64();
            }

            // --- B. FFN / MoE Sublayer ---
            let tp_ffn = if do_prof { Some(Instant::now()) } else { None };
            {
                let layer = &self.layers[l];
                self.glm.hc_pre(
                    &self.compute_stream,
                    st.x.ptr(),
                    layer.ffn_hc.fn_w.ptr(),
                    st.hc_part.ptr(),
                    layer.ffn_hc.base.ptr(),
                    layer.ffn_hc.scale.ptr(),
                    layer.post_norm.ptr(),
                    st.normed.ptr(),
                    st.post.ptr(),
                    st.comb.ptr(),
                    rows,
                    d,
                    self.cfg.rms_eps,
                    self.cfg.hc_eps,
                    self.cfg.hc_sinkhorn_iters,
                )?;

                match &layer.ffn {
                    FfnWeights::Dense(m) => {
                        m.gate.apply(
                            &self.lin,
                            &self.compute_stream,
                            st.normed.ptr(),
                            d,
                            st.mlp_g.ptr(),
                            self.cfg.dense_inter,
                            rows,
                            false,
                        )?;
                        m.up.apply(
                            &self.lin,
                            &self.compute_stream,
                            st.normed.ptr(),
                            d,
                            st.mlp_u.ptr(),
                            self.cfg.dense_inter,
                            rows,
                            false,
                        )?;
                        self.lin.swiglu_bf16(
                            &self.compute_stream,
                            st.mlp_g.ptr(),
                            st.mlp_u.ptr(),
                            st.mlp_act.ptr(),
                            rows,
                            self.cfg.dense_inter,
                            self.cfg.swiglu_limit,
                        )?;
                        m.down.apply(
                            &self.lin,
                            &self.compute_stream,
                            st.mlp_act.ptr(),
                            self.cfg.dense_inter,
                            st.branch.ptr(),
                            d,
                            rows,
                            true,
                        )?;
                        if let Some(t) = tp_ffn {
                            self.compute_stream.synchronize()?;
                            self.decode_prof.dense_ffn_secs += t.elapsed().as_secs_f64();
                        }
                    }
                    FfnWeights::Moe(m) => {
                        let moe_idx = m.moe_idx;
                        if is_warp_zero_sync && moe_idx == 0 {
                            // Fence any background RAM->VRAM promotion right before Layer 3 (already hidden by DFlash2 + Dense Layers 0..2!)
                            self.pool.wait_bg_promote(&self.compute_stream)?;
                        }
                        let e_cnt = self.cfg.num_experts;
                        let base_key = moe_idx * e_cnt;
                        let layer_map_ptr = self.pool.dev_map_ptr() + (base_key * 4) as u64;
                        let (picks_out_ptr, wts_out_ptr) = if use_zero_sync_moe {
                            let off = moe_idx * rows * slots * 4;
                            (st.d_all_picks.at(off), st.d_all_wts.at(off))
                        } else {
                            (st.d_picks.ptr(), st.d_wts.ptr())
                        };
                        // 1. Current layer router + fused GPU sigmoid/topK/DynTopK (+ optional residency tiebreak!)
                        self.lin.linear_bf16(
                            &self.compute_stream,
                            st.normed.ptr(),
                            d,
                            m.router.ptr(),
                            st.router_logits.ptr(),
                            e_cnt,
                            rows,
                            e_cnt,
                            d,
                            true,
                        )?;
                        let dyn_mass = if is_decode_step { opts.dyn_topk_mass } else { 0.0 };
                        let (rbias, rstrict) = if is_decode_step && !is_draft_resident && opts.residency_bias > 0.0 {
                            (opts.residency_bias, opts.residency_strict_k as i32)
                        } else {
                            (0.0, self.cfg.top_k as i32)
                        };
                        self.router_remap.sigmoid_topk_residency(
                            &self.compute_stream,
                            st.router_logits.ptr(),
                            m.bias.ptr(),
                            picks_out_ptr,
                            wts_out_ptr,
                            rows as i32,
                            e_cnt as i32,
                            self.cfg.top_k as i32,
                            self.cfg.routed_scale,
                            dyn_mass,
                            opts.dyn_topk_min_k as i32,
                            e_cnt as i32,
                            if rbias > 0.0 { layer_map_ptr } else { 0 },
                            rbias,
                            rstrict,
                        )?;

                        if use_zero_sync_moe {
                            // ZERO-SYNC, ZERO-STALL GPU-ONLY PATH (DraftResident OR Self-Healing WarpZeroSync)!
                            m.shared_gu.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                st.mlp_g.ptr(),
                                2 * self.cfg.shared_inter,
                                rows,
                                false,
                            )?;
                            self.lin.swiglu_split_bf16(
                                &self.compute_stream,
                                st.mlp_g.ptr(),
                                st.mlp_act.ptr(),
                                rows,
                                self.cfg.shared_inter,
                                self.cfg.swiglu_limit,
                            )?;
                            m.shared_down.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.mlp_act.ptr(),
                                self.cfg.shared_inter,
                                st.sh_out.ptr(),
                                d,
                                rows,
                                true,
                            )?;
                            self.lin.put_shared_slot_f32(
                                &self.compute_stream,
                                st.sh_out.ptr(),
                                st.exl3_sc.slot_y.ptr(),
                                rows,
                                d,
                                slots,
                            )?;
                            self.router_remap.remap_resident_renorm(
                                &self.compute_stream,
                                picks_out_ptr,
                                wts_out_ptr,
                                st.d_mapped.ptr(),
                                st.d_wts_renorm.ptr(),
                                layer_map_ptr,
                                rows as i32,
                                self.cfg.top_k as i32,
                                e_cnt as i32,
                                e_cnt as i32,
                                vram_slots as i32,
                                self.cfg.routed_scale,
                            )?;
                            let view = Exl3ExpertsView {
                                gt_ptrs: self.exl3_ptr_tables[0].ptr(),
                                ut_ptrs: self.exl3_ptr_tables[1].ptr(),
                                dt_ptrs: self.exl3_ptr_tables[2].ptr(),
                                k2_g: self.pool.vram_k2_ptr(0),
                                k2_u: self.pool.vram_k2_ptr(1),
                                k2_d: self.pool.vram_k2_ptr(2),
                                suh_g: self.scales.vram_ptr(0),
                                suh_u: self.scales.vram_ptr(1),
                                svh_g: self.scales.vram_ptr(2),
                                svh_u: self.scales.vram_ptr(3),
                                suh_d: self.scales.vram_ptr(4),
                                svh_d: self.scales.vram_ptr(5),
                                experts: vram_slots,
                                dims: d,
                                inter: self.cfg.moe_inter,
                                swiglu_limit: self.cfg.swiglu_limit,
                                act_mode: 0,
                            };
                            self.exl3.run_moe(
                                &self.compute_stream,
                                &view,
                                &st.exl3_sc,
                                st.normed.ptr(),
                                d,
                                InputDType::Bf16,
                                st.d_mapped.ptr(),
                                st.d_wts_renorm.ptr(),
                                st.branch.ptr(),
                                rows,
                                slots,
                                true,
                            )?;
                        } else {
                            let use_cpu_hybrid = is_decode_step
                                && opts.cpu_hybrid_max > 0
                                && rows <= st.max_verify
                                && rows <= MAX_CPU_JOB_ROWS;
                            let use_split_wave = is_decode_step
                                && !use_cpu_hybrid
                                && rows <= st.max_verify;
                            let need_wts_host =
                                (is_decode_step && opts.nvme_mass > 0.0) || use_cpu_hybrid || use_split_wave;
                            unsafe {
                                self.compute_stream.dtoh_async(
                                    st.h_picks.as_mut_ptr(),
                                    st.d_picks.ptr(),
                                    rows * slots * 4,
                                )?;
                                if need_wts_host {
                                    self.compute_stream.dtoh_async(
                                        st.h_wts.as_mut_ptr(),
                                        st.d_wts.ptr(),
                                        rows * slots * 4,
                                    )?;
                                }
                                if use_cpu_hybrid {
                                    self.compute_stream.dtoh_async(
                                        st.h_normed.as_mut_ptr(),
                                        st.normed.ptr(),
                                        rows * d * 2,
                                    )?;
                                }
                            }

                            // 2. Milestone 7 Lever 2 speculative lookahead router (L+1 and optional L+2 on the SAME event sync!):
                            //    Enabled for BOTH `spec_prefetch_p > 0` (exact) AND mandatory Rank-0 lookahead on `warp` (`nvme_strict_k > 0`)!
                            let do_spec = is_decode_step
                                && opts.lookahead_depth > 0
                                && (opts.spec_prefetch_p > 0 || opts.nvme_strict_k > 0)
                                && l + 1 < self.layers.len()
                                && matches!(self.layers[l + 1].ffn, FfnWeights::Moe(_));
                            let mut do_spec2 = false;
                            if do_spec {
                                if let FfnWeights::Moe(nm) = &self.layers[l + 1].ffn {
                                    let next_map_ptr = self.pool.dev_map_ptr() + (((moe_idx + 1) * e_cnt) * 4) as u64;
                                    self.lin.linear_bf16(
                                        &self.compute_stream,
                                        st.normed.ptr(),
                                        d,
                                        nm.router.ptr(),
                                        st.spec_logits.ptr(),
                                        e_cnt,
                                        rows,
                                        e_cnt,
                                        d,
                                        true,
                                    )?;
                                    self.router_remap.sigmoid_topk_residency(
                                        &self.compute_stream,
                                        st.spec_logits.ptr(),
                                        nm.bias.ptr(),
                                        st.d_spec_picks.ptr(),
                                        st.d_spec_wts.ptr(),
                                        rows as i32,
                                        e_cnt as i32,
                                        self.cfg.top_k as i32,
                                        self.cfg.routed_scale,
                                        dyn_mass,
                                        opts.dyn_topk_min_k as i32,
                                        e_cnt as i32,
                                        if rbias > 0.0 { next_map_ptr } else { 0 },
                                        rbias,
                                        rstrict,
                                    )?;
                                    unsafe {
                                        self.compute_stream.dtoh_async(
                                            st.h_spec_picks.as_mut_ptr(),
                                            st.d_spec_picks.ptr(),
                                            rows * slots * 4,
                                        )?;
                                    }
                                }
                                if opts.lookahead_depth >= 2
                                    && l + 2 < self.layers.len()
                                    && matches!(self.layers[l + 2].ffn, FfnWeights::Moe(_))
                                {
                                    if let FfnWeights::Moe(nm2) = &self.layers[l + 2].ffn {
                                        let next2_map_ptr =
                                            self.pool.dev_map_ptr() + (((moe_idx + 2) * e_cnt) * 4) as u64;
                                        self.lin.linear_bf16(
                                            &self.compute_stream,
                                            st.normed.ptr(),
                                            d,
                                            nm2.router.ptr(),
                                            st.spec_logits.ptr(),
                                            e_cnt,
                                            rows,
                                            e_cnt,
                                            d,
                                            true,
                                        )?;
                                        self.router_remap.sigmoid_topk_residency(
                                            &self.compute_stream,
                                            st.spec_logits.ptr(),
                                            nm2.bias.ptr(),
                                            st.d_spec2_picks.ptr(),
                                            st.d_spec_wts.ptr(),
                                            rows as i32,
                                            e_cnt as i32,
                                            self.cfg.top_k as i32,
                                            self.cfg.routed_scale,
                                            dyn_mass,
                                            opts.dyn_topk_min_k as i32,
                                            e_cnt as i32,
                                            if rbias > 0.0 { next2_map_ptr } else { 0 },
                                            rbias,
                                            rstrict,
                                        )?;
                                        unsafe {
                                            self.compute_stream.dtoh_async(
                                                st.h_spec2_picks.as_mut_ptr(),
                                                st.d_spec2_picks.ptr(),
                                                rows * slots * 4,
                                            )?;
                                        }
                                        do_spec2 = true;
                                    }
                                }
                            }

                            // Record event right after router DtoH, BEFORE shared expert!
                            self.picks_ready.record(&self.compute_stream)?;

                            // 3. OVERLAP fused shared expert on GPU while CPU + NVMe service `demand_with_scales`!
                            m.shared_gu.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                st.mlp_g.ptr(),
                                2 * self.cfg.shared_inter,
                                rows,
                                false,
                            )?;
                            self.lin.swiglu_split_bf16(
                                &self.compute_stream,
                                st.mlp_g.ptr(),
                                st.mlp_act.ptr(),
                                rows,
                                self.cfg.shared_inter,
                                self.cfg.swiglu_limit,
                            )?;
                            m.shared_down.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.mlp_act.ptr(),
                                self.cfg.shared_inter,
                                st.sh_out.ptr(),
                                d,
                                rows,
                                true,
                            )?;
                            self.lin.put_shared_slot_f32(
                                &self.compute_stream,
                                st.sh_out.ptr(),
                                st.exl3_sc.slot_y.ptr(),
                                rows,
                                d,
                                slots,
                            )?;

                            // 4. Wait ONLY for router picks event (GPU keeps executing shared expert!), then demand experts!
                            let tp_sync = if do_prof { Some(Instant::now()) } else { None };
                            self.picks_ready.synchronize()?;
                            if let Some(t) = tp_sync {
                                self.decode_prof.picks_sync_secs += t.elapsed().as_secs_f64();
                            }
                            let tp_dem = if do_prof { Some(Instant::now()) } else { None };
                            st.active_keys.clear();
                            st.strict_keys.clear();
                            let use_nvme_mass = is_decode_step && opts.nvme_mass > 0.0;
                            let use_nvme_budget = use_nvme_mass
                                || (is_decode_step
                                    && opts.nvme_strict_k > 0
                                    && opts.nvme_strict_k < self.cfg.top_k);
                            let hp = &st.h_picks.as_i32_mut()[..rows * slots];
                            let hw = &st.h_wts.as_f32_mut()[..rows * slots];
                            let strict_row_lim = if is_verify && use_nvme_mass && opts.nvme_tail_row > 0 {
                                opts.nvme_tail_row.min(rows)
                            } else {
                                rows
                            };
                            let can_prune = is_verify
                                && tree_b_start == 0
                                && opts.nvme_prune_row > 0
                                && rows > opts.nvme_prune_row;
                            let can_prune_prob = is_verify
                                && tree_b_start == 0
                                && opts.nvme_prune_prob > 0.0
                                && rows > 1
                                && row_cum_probs.is_some();
                            let mut pruned_to = rows;
                            for r in 0..rows {
                                let is_branch_b_row = tree_b_start > 0 && r >= tree_b_start;
                                let row_is_strict = r < strict_row_lim && !is_branch_b_row;
                                let mut row_total_w = 0.0f32;
                                let mut row_covered_w = 0.0f32;
                                let mut row_keys = [0usize; 16];
                                let mut row_keys_n = 0usize;
                                let mut row_strict = [0usize; 16];
                                let mut row_strict_n = 0usize;
                                let mut row_ram: [(usize, f32); 16] = [(0, 0.0); 16];
                                let mut row_ram_n = 0usize;
                                let mut row_cold: [(usize, f32); 16] = [(0, 0.0); 16];
                                let mut row_cold_n = 0usize;
                                let mut row_needs_nvme = false;

                                for k_rank in 0..self.cfg.top_k {
                                    let p = hp[r * slots + k_rank];
                                    if p >= 0 && (p as usize) < e_cnt {
                                        let k = base_key + (p as usize);
                                        if row_keys_n < 16 {
                                            row_keys[row_keys_n] = k;
                                            row_keys_n += 1;
                                        }
                                        let in_vram = self.pool.vram_cache.peek(k).is_some();
                                        let in_ram = !in_vram
                                            && (self.pool.ram_cache.peek(k).is_some()
                                                || self.pool.ram_cache.pending(k).is_some());
                                        if use_nvme_mass {
                                            let w = hw[r * slots + k_rank].max(0.0);
                                            row_total_w += w;
                                            if k_rank < opts.nvme_strict_k {
                                                if do_prof && !in_vram && !in_ram {
                                                    if r == 0 {
                                                        self.decode_prof.row0_cold_rank0 += 1;
                                                    } else {
                                                        self.decode_prof.tail_cold_rank0 += 1;
                                                    }
                                                }
                                                if row_is_strict {
                                                    row_covered_w += w;
                                                    if !in_vram && !in_ram {
                                                        row_needs_nvme = true;
                                                    }
                                                    if row_strict_n < 16 {
                                                        row_strict[row_strict_n] = k;
                                                        row_strict_n += 1;
                                                    }
                                                } else if in_vram {
                                                    row_covered_w += w;
                                                } else if in_ram {
                                                    row_covered_w += w;
                                                    if row_strict_n < 16 {
                                                        row_strict[row_strict_n] = k;
                                                        row_strict_n += 1;
                                                    }
                                                } else if row_cold_n < 16 {
                                                    row_cold[row_cold_n] = (k, w);
                                                    row_cold_n += 1;
                                                    if do_prof {
                                                        self.decode_prof.tail_cold_skipped += 1;
                                                    }
                                                }
                                            } else if in_vram {
                                                row_covered_w += w;
                                            } else if in_ram {
                                                if row_ram_n < 16 {
                                                    row_ram[row_ram_n] = (k, w);
                                                    row_ram_n += 1;
                                                }
                                            } else if row_cold_n < 16 {
                                                row_cold[row_cold_n] = (k, w);
                                                row_cold_n += 1;
                                            }
                                        } else if use_nvme_budget {
                                            if k_rank < opts.nvme_strict_k {
                                                if !in_vram && !in_ram {
                                                    row_needs_nvme = true;
                                                }
                                                if row_strict_n < 16 {
                                                    row_strict[row_strict_n] = k;
                                                    row_strict_n += 1;
                                                }
                                            }
                                        } else if !in_vram && !in_ram {
                                            row_needs_nvme = true;
                                        }
                                    }
                                }
                                if use_nvme_mass && row_total_w > 0.0 {
                                    let target_w = opts.nvme_mass * row_total_w;
                                    if row_covered_w < target_w && row_ram_n > 0 {
                                        let sl = &mut row_ram[..row_ram_n];
                                        sl.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                                        for &(rk, rw) in sl.iter() {
                                            if row_covered_w >= target_w {
                                                break;
                                            }
                                            row_covered_w += rw;
                                            if row_strict_n < 16 {
                                                row_strict[row_strict_n] = rk;
                                                row_strict_n += 1;
                                            }
                                        }
                                    }
                                    if row_covered_w < target_w && row_cold_n > 0 {
                                        let sl = &mut row_cold[..row_cold_n];
                                        sl.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                                        for &(ck, cw) in sl.iter() {
                                            if row_covered_w >= target_w {
                                                break;
                                            }
                                            row_covered_w += cw;
                                            row_needs_nvme = true;
                                            if row_strict_n < 16 {
                                                row_strict[row_strict_n] = ck;
                                                row_strict_n += 1;
                                            }
                                        }
                                    }
                                }
                                // 100% Lossless Mid-Forward Speculative Prefix Pruning (row-cap OR confidence-gated!):
                                // If speculative row `r` would stall NVMe and has low cumulative probability (or >= nvme_prune_row),
                                // truncate the batch to `0..r` right here -> 0 SSD stall, 0 wasted compute in layers l..44, 100% bit-exact!
                                let prob_prune = can_prune_prob
                                    && r >= 1
                                    && row_needs_nvme
                                    && row_cum_probs.map_or(false, |ps| r < ps.len() && ps[r] < opts.nvme_prune_prob);
                                if (can_prune && r >= opts.nvme_prune_row && row_needs_nvme) || prob_prune {
                                    pruned_to = r;
                                    if do_prof {
                                        self.decode_prof.tail_cold_skipped += 1;
                                    }
                                    break;
                                }
                                for &k in &row_keys[..row_keys_n] {
                                    self.pool.expert_hits[k] += 1;
                                    if !st.active_keys.contains(&k) {
                                        st.active_keys.push(k);
                                    }
                                }
                                for &k in &row_strict[..row_strict_n] {
                                    if !st.strict_keys.contains(&k) {
                                        st.strict_keys.push(k);
                                    }
                                }
                            }
                            if pruned_to < rows {
                                rows = pruned_to;
                            }
                            st.gpu_keys.clear();
                            st.cpu_keys.clear();
                            st.cpu_jobs.clear();
                            if use_cpu_hybrid {
                                // Reap any completed speculative prefetches before classifying RAM hits!
                                let _ = self.pool.poll_spec_only();
                                st.skipped_cold.clear();
                                let mut ram_cands: [(usize, u8, u16); 64] = [(0, 0, 0); 64];
                                let mut ram_cands_n = 0usize;
                                let mut n_cold_nvme = 0usize;
                                for &k in &st.active_keys {
                                    let in_vram = self.pool.vram_cache.peek(k).is_some();
                                    let in_ram_ready = !in_vram && self.pool.ram_cache.peek(k).is_some();
                                    let in_ram_pend = !in_vram && !in_ram_ready && self.pool.ram_cache.pending(k).is_some();
                                    let keep = if use_nvme_budget {
                                        st.strict_keys.contains(&k)
                                            || in_vram
                                            || (!use_nvme_mass && (in_ram_ready || in_ram_pend))
                                    } else {
                                        true
                                    };
                                    if !keep {
                                        self.pool.stats.nvme_tail_skipped += 1;
                                        st.skipped_cold.push(k);
                                    } else if in_vram {
                                        st.gpu_keys.push(k);
                                    } else if in_ram_ready {
                                        if ram_cands_n < ram_cands.len() {
                                            let is_3b = (self.pool.catalog.entries[k].components[0].bits >= 3) as u8;
                                            let hf = self.pool.ram_cache.hist_freq(k);
                                            ram_cands[ram_cands_n] = (k, is_3b, hf);
                                            ram_cands_n += 1;
                                        } else {
                                            st.gpu_keys.push(k);
                                        }
                                    } else {
                                        if !in_ram_pend {
                                            n_cold_nvme += 1;
                                        }
                                        st.gpu_keys.push(k);
                                    }
                                }
                                // Strict-overlap cost-aware hybrid split for PCIe Gen4 (23.5 GB/s) + 6-P-core CPU (~850 us/2b expert):
                                // Only count truly cold (non-pending) NVMe misses, trigger at >=2 cold misses for both decode & verify,
                                // and restrict auto-offload to 2.0bpw experts unless `--cpu-hybrid-always` is set.
                                let need_cold = 2usize;
                                let eligible_n = if opts.cpu_hybrid_always {
                                    ram_cands_n
                                } else {
                                    ram_cands[..ram_cands_n].iter().filter(|&&(_, is_3b, _)| is_3b == 0).count()
                                };
                                let cpu_cap = if eligible_n == 0 {
                                    0
                                } else if opts.cpu_hybrid_always {
                                    opts.cpu_hybrid_max.min(eligible_n)
                                } else if n_cold_nvme >= need_cold {
                                    (n_cold_nvme / need_cold).min(opts.cpu_hybrid_max).min(eligible_n)
                                } else {
                                    0
                                };
                                if cpu_cap > 0 {
                                    // Prefer 2.0bpw (6 MiB) and lowest hist_freq (prevents polluting Tier-1 VRAM!)
                                    ram_cands[..ram_cands_n].sort_unstable_by_key(|&(_, is_3b, hf)| (is_3b, hf));
                                    for i in 0..ram_cands_n {
                                        let k = ram_cands[i].0;
                                        if i < cpu_cap {
                                            st.cpu_keys.push(k);
                                        } else {
                                            st.gpu_keys.push(k);
                                        }
                                    }
                                } else {
                                    for i in 0..ram_cands_n {
                                        st.gpu_keys.push(ram_cands[i].0);
                                    }
                                }

                                if !st.cpu_keys.is_empty() {
                                    // Joint GPU+CPU exact weight renormalization (zero mismatch vs pure GPU!)
                                    let mut row_scale = [1.0f32; MAX_CPU_JOB_ROWS];
                                    let hwg = &mut st.h_wts_gpu.as_f32_mut()[..rows * slots];
                                    for r in 0..rows {
                                        let mut tot_w = 0.0f32;
                                        let mut kept_w = 0.0f32;
                                        for k_rank in 0..self.cfg.top_k {
                                            let p = hp[r * slots + k_rank];
                                            if p >= 0 && (p as usize) < e_cnt {
                                                let k = base_key + (p as usize);
                                                let w = hw[r * slots + k_rank].max(0.0);
                                                tot_w += w;
                                                if st.gpu_keys.contains(&k) || st.cpu_keys.contains(&k) {
                                                    kept_w += w;
                                                }
                                            }
                                        }
                                        let sc = if use_nvme_budget && kept_w > 1e-8 {
                                            tot_w / kept_w
                                        } else {
                                            1.0
                                        };
                                        row_scale[r] = sc;
                                        for k_rank in 0..self.cfg.top_k {
                                            let p = hp[r * slots + k_rank];
                                            let w = if p >= 0 && (p as usize) < e_cnt {
                                                let k = base_key + (p as usize);
                                                if st.gpu_keys.contains(&k) {
                                                    hw[r * slots + k_rank] * sc
                                                } else {
                                                    0.0
                                                }
                                            } else {
                                                0.0
                                            };
                                            hwg[r * slots + k_rank] = w;
                                        }
                                        hwg[r * slots + self.cfg.top_k] = hw[r * slots + self.cfg.top_k];
                                    }

                                    // Build grouped CpuExl3ExpertJob entries and launch ASYNC immediately (<3 us!)
                                    for &ck in &st.cpu_keys {
                                        if let Some((ptrs, k2s)) = self.pool.ram_expert_ptrs(ck) {
                                            let mut job = CpuExl3ExpertJob {
                                                gate_t: ptrs[0],
                                                up_t: ptrs[1],
                                                down_t: ptrs[2],
                                                suh_g: self.scales.ram_scale_ptr(0, ck),
                                                suh_u: self.scales.ram_scale_ptr(1, ck),
                                                svh_g: self.scales.ram_scale_ptr(2, ck),
                                                svh_u: self.scales.ram_scale_ptr(3, ck),
                                                suh_d: self.scales.ram_scale_ptr(4, ck),
                                                svh_d: self.scales.ram_scale_ptr(5, ck),
                                                k2_g: k2s[0],
                                                k2_u: k2s[1],
                                                k2_d: k2s[2],
                                                num_rows: 0,
                                                row_idx: [0; MAX_CPU_JOB_ROWS],
                                                row_wt: [0.0; MAX_CPU_JOB_ROWS],
                                            };
                                            for r in 0..rows {
                                                let mut w_sum = 0.0f32;
                                                for k_rank in 0..self.cfg.top_k {
                                                    let p = hp[r * slots + k_rank];
                                                    if p >= 0 && (p as usize) < e_cnt && base_key + (p as usize) == ck {
                                                        w_sum += hw[r * slots + k_rank] * row_scale[r];
                                                    }
                                                }
                                                if w_sum != 0.0 {
                                                    let idx = job.num_rows as usize;
                                                    job.row_idx[idx] = r as i32;
                                                    job.row_wt[idx] = w_sum;
                                                    job.num_rows += 1;
                                                }
                                            }
                                            if job.num_rows > 0 {
                                                st.cpu_jobs.push(job);
                                            }
                                        }
                                    }
                                    if !st.cpu_jobs.is_empty() {
                                        let x_u16 = unsafe {
                                            std::slice::from_raw_parts(st.h_normed.as_ptr() as *const u16, rows * d)
                                        };
                                        let out_f32 = &mut st.h_cpu_out.as_f32_mut()[..rows * d];
                                        unsafe {
                                            self.cpu_exl3.submit_async(
                                                x_u16,
                                                rows,
                                                d,
                                                self.cfg.moe_inter,
                                                self.cfg.swiglu_limit,
                                                &st.cpu_jobs,
                                                out_f32,
                                            );
                                        }
                                    }
                                }
                            }
                            let view = Exl3ExpertsView {
                                gt_ptrs: self.exl3_ptr_tables[0].ptr(),
                                ut_ptrs: self.exl3_ptr_tables[1].ptr(),
                                dt_ptrs: self.exl3_ptr_tables[2].ptr(),
                                k2_g: self.pool.vram_k2_ptr(0),
                                k2_u: self.pool.vram_k2_ptr(1),
                                k2_d: self.pool.vram_k2_ptr(2),
                                suh_g: self.scales.vram_ptr(0),
                                suh_u: self.scales.vram_ptr(1),
                                svh_g: self.scales.vram_ptr(2),
                                svh_u: self.scales.vram_ptr(3),
                                suh_d: self.scales.vram_ptr(4),
                                svh_d: self.scales.vram_ptr(5),
                                experts: vram_slots,
                                dims: d,
                                inter: self.cfg.moe_inter,
                                swiglu_limit: self.cfg.swiglu_limit,
                                act_mode: 0,
                            };

                            if use_split_wave {
                                // Milestone 8 Split-Wave Resident-First Async EXL3 MoE:
                                // Wave 1 computes shared_expert + ALL already-VRAM-resident experts IMMEDIATELY
                                // while copy_stream (23.5 GB/s PCIe) + io_uring (6.75 GB/s NVMe) stream missing
                                // experts into disjoint unleased VRAM slots concurrently! 100% bit-exact!
                                let (hit_len, miss_len) = self.pool.prepare_split_demand(
                                    &self.compute_stream,
                                    &st.active_keys,
                                    &st.strict_keys,
                                    use_nvme_budget,
                                    !use_nvme_mass,
                                    &mut self.scales,
                                    &mut st.skipped_cold,
                                )?;
                                if miss_len > 0 && hit_len > 0 {
                                    let skip_v = vram_slots as i32;
                                    {
                                        let kept = self.pool.split_kept_keys();
                                        let w1_p = &mut st.h_w1_mapped.as_i32_mut()[..rows * slots];
                                        let w1_w = &mut st.h_wts_gpu.as_f32_mut()[..rows * slots];
                                        let w2_w = &mut st.h_w2_wts.as_f32_mut()[..rows * slots];
                                        for r in 0..rows {
                                            let mut kept_w = 0.0f32;
                                            for k_rank in 0..self.cfg.top_k {
                                                let p = hp[r * slots + k_rank];
                                                if p >= 0 && (p as usize) < e_cnt {
                                                    let k = base_key + (p as usize);
                                                    let w = hw[r * slots + k_rank].max(0.0);
                                                    if w > 0.0 && kept.contains(&k) {
                                                        kept_w += w;
                                                    }
                                                }
                                            }
                                            let sc = if use_nvme_budget && kept_w > 1e-7 && self.cfg.routed_scale > 0.0 {
                                                self.cfg.routed_scale / kept_w
                                            } else {
                                                1.0
                                            };
                                            for k_rank in 0..self.cfg.top_k {
                                                let idx_s = r * slots + k_rank;
                                                let p = hp[idx_s];
                                                let w = hw[idx_s];
                                                if p >= 0 && (p as usize) < e_cnt && w > 0.0 {
                                                    let k = base_key + (p as usize);
                                                    if !kept.contains(&k) {
                                                        w1_p[idx_s] = skip_v;
                                                        w1_w[idx_s] = 0.0;
                                                        w2_w[idx_s] = 0.0;
                                                    } else if let Some(vslot) = self.pool.vram_cache.peek(k) {
                                                        w1_p[idx_s] = vslot as i32;
                                                        w1_w[idx_s] = w * sc;
                                                        w2_w[idx_s] = 0.0;
                                                    } else {
                                                        w1_p[idx_s] = skip_v;
                                                        w1_w[idx_s] = 0.0;
                                                        w2_w[idx_s] = w * sc;
                                                    }
                                                } else {
                                                    w1_p[idx_s] = skip_v;
                                                    w1_w[idx_s] = 0.0;
                                                    w2_w[idx_s] = 0.0;
                                                }
                                            }
                                            let sh_idx = r * slots + self.cfg.top_k;
                                            w1_p[sh_idx] = skip_v;
                                            w1_w[sh_idx] = hw[sh_idx];
                                            w2_w[sh_idx] = 0.0;
                                        }
                                    }
                                    // Launch Wave 1 (VRAM-resident + shared expert) on compute_stream IMMEDIATELY!
                                    unsafe {
                                        self.compute_stream.htod_async(
                                            st.d_mapped.ptr(),
                                            st.h_w1_mapped.as_ptr(),
                                            rows * slots * 4,
                                        )?;
                                        self.compute_stream.htod_async(
                                            st.d_wts_renorm.ptr(),
                                            st.h_wts_gpu.as_ptr(),
                                            rows * slots * 4,
                                        )?;
                                    }
                                    self.exl3.run_moe_ext(
                                        &self.compute_stream,
                                        &view,
                                        &st.exl3_sc,
                                        st.normed.ptr(),
                                        d,
                                        InputDType::Bf16,
                                        st.d_mapped.ptr(),
                                        st.d_wts_renorm.ptr(),
                                        st.branch.ptr(),
                                        rows,
                                        slots,
                                        true,
                                        hit_len,
                                        false,
                                    )?;
                                    // Stream missing Tier-2 RAM PCIe HtoD + cold NVMe concurrently with Wave 1!
                                    self.pool.finish_split_demand(&self.compute_stream, &mut self.scales)?;
                                    if let Some(t) = tp_dem {
                                        self.decode_prof.demand_secs += t.elapsed().as_secs_f64();
                                    }
                                    let tp_exl3 = if do_prof { Some(Instant::now()) } else { None };
                                    // Launch Wave 2 (ONLY the 1-2 newly arrived experts!) & accumulate directly into st.branch!
                                    {
                                        let w2_p = &mut st.h_w2_mapped.as_i32_mut()[..rows * slots];
                                        let w2_w = &st.h_w2_wts.as_f32_mut()[..rows * slots];
                                        for r in 0..rows {
                                            for k_rank in 0..self.cfg.top_k {
                                                let idx_s = r * slots + k_rank;
                                                if w2_w[idx_s] > 0.0 {
                                                    let k = base_key + (hp[idx_s] as usize);
                                                    w2_p[idx_s] = self
                                                        .pool
                                                        .vram_cache
                                                        .peek(k)
                                                        .map_or(skip_v, |s| s as i32);
                                                } else {
                                                    w2_p[idx_s] = skip_v;
                                                }
                                            }
                                            w2_p[r * slots + self.cfg.top_k] = skip_v;
                                        }
                                    }
                                    unsafe {
                                        self.compute_stream.htod_async(
                                            st.d_w2_mapped.ptr(),
                                            st.h_w2_mapped.as_ptr(),
                                            rows * slots * 4,
                                        )?;
                                        self.compute_stream.htod_async(
                                            st.d_w2_wts.ptr(),
                                            st.h_w2_wts.as_ptr(),
                                            rows * slots * 4,
                                        )?;
                                    }
                                    self.exl3.run_moe_ext(
                                        &self.compute_stream,
                                        &view,
                                        &st.exl3_sc,
                                        st.normed.ptr(),
                                        d,
                                        InputDType::Bf16,
                                        st.d_w2_mapped.ptr(),
                                        st.d_w2_wts.ptr(),
                                        st.branch.ptr(),
                                        rows,
                                        slots,
                                        true,
                                        miss_len,
                                        true,
                                    )?;
                                    if let Some(t) = tp_exl3 {
                                        self.compute_stream.synchronize()?;
                                        self.decode_prof.exl3_secs += t.elapsed().as_secs_f64();
                                    }
                                } else {
                                    // 100% VRAM hit (miss_len == 0) OR 100% miss (hit_len == 0): single-wave fast path!
                                    if miss_len > 0 {
                                        self.pool.finish_split_demand(&self.compute_stream, &mut self.scales)?;
                                    }
                                    if let Some(t) = tp_dem {
                                        self.decode_prof.demand_secs += t.elapsed().as_secs_f64();
                                    }
                                    let tp_exl3 = if do_prof { Some(Instant::now()) } else { None };
                                    let wts_ptr = if use_nvme_budget {
                                        self.router_remap.remap_resident_renorm(
                                            &self.compute_stream,
                                            st.d_picks.ptr(),
                                            st.d_wts.ptr(),
                                            st.d_mapped.ptr(),
                                            st.d_wts_renorm.ptr(),
                                            layer_map_ptr,
                                            rows as i32,
                                            self.cfg.top_k as i32,
                                            e_cnt as i32,
                                            e_cnt as i32,
                                            vram_slots as i32,
                                            self.cfg.routed_scale,
                                        )?;
                                        st.d_wts_renorm.ptr()
                                    } else {
                                        self.router_remap.remap_i32(
                                            &self.compute_stream,
                                            st.d_picks.ptr(),
                                            st.d_mapped.ptr(),
                                            layer_map_ptr,
                                            (rows * slots) as i32,
                                            e_cnt as i32,
                                            e_cnt as i32,
                                            vram_slots as i32,
                                        )?;
                                        st.d_wts.ptr()
                                    };
                                    self.exl3.run_moe(
                                        &self.compute_stream,
                                        &view,
                                        &st.exl3_sc,
                                        st.normed.ptr(),
                                        d,
                                        InputDType::Bf16,
                                        st.d_mapped.ptr(),
                                        wts_ptr,
                                        st.branch.ptr(),
                                        rows,
                                        slots,
                                        true,
                                    )?;
                                    if let Some(t) = tp_exl3 {
                                        self.compute_stream.synchronize()?;
                                        self.decode_prof.exl3_secs += t.elapsed().as_secs_f64();
                                    }
                                }
                            } else {
                                if use_cpu_hybrid {
                                    // Execute NVMe + PCIe HtoD for GPU keys while leasing CPU keys in Tier-2 RAM!
                                    self.pool.demand_with_scales_hybrid(
                                        &self.compute_stream,
                                        base_key,
                                        e_cnt,
                                        &st.gpu_keys,
                                        &st.cpu_keys,
                                        &mut self.scales,
                                    )?;
                                } else if use_nvme_budget {
                                    self.pool.demand_with_scales_budget(
                                        &self.compute_stream,
                                        base_key,
                                        e_cnt,
                                        &st.active_keys,
                                        &st.strict_keys,
                                        !use_nvme_mass,
                                        &mut self.scales,
                                        &mut st.skipped_cold,
                                    )?;
                                } else {
                                    self.pool.demand_with_scales(
                                        &self.compute_stream,
                                        base_key,
                                        e_cnt,
                                        &st.active_keys,
                                        &mut self.scales,
                                    )?;
                                }
                                if let Some(t) = tp_dem {
                                    self.decode_prof.demand_secs += t.elapsed().as_secs_f64();
                                }
                                let tp_exl3 = if do_prof { Some(Instant::now()) } else { None };

                                // 5. Remap logical picks -> Tier-1 VRAM slots & launch grouped sm_86 EXL3 MoE immediately!
                                let has_cpu_jobs = !st.cpu_jobs.is_empty();
                                let wts_ptr = if has_cpu_jobs {
                                    unsafe {
                                        self.compute_stream.htod_async(
                                            st.d_wts_renorm.ptr(),
                                            st.h_wts_gpu.as_ptr(),
                                            rows * slots * 4,
                                        )?;
                                    }
                                    self.router_remap.remap_i32(
                                        &self.compute_stream,
                                        st.d_picks.ptr(),
                                        st.d_mapped.ptr(),
                                        layer_map_ptr,
                                        (rows * slots) as i32,
                                        e_cnt as i32,
                                        e_cnt as i32,
                                        vram_slots as i32,
                                    )?;
                                    st.d_wts_renorm.ptr()
                                } else if use_nvme_budget {
                                    self.router_remap.remap_resident_renorm(
                                        &self.compute_stream,
                                        st.d_picks.ptr(),
                                        st.d_wts.ptr(),
                                        st.d_mapped.ptr(),
                                        st.d_wts_renorm.ptr(),
                                        layer_map_ptr,
                                        rows as i32,
                                        self.cfg.top_k as i32,
                                        e_cnt as i32,
                                        e_cnt as i32,
                                        vram_slots as i32,
                                        self.cfg.routed_scale,
                                    )?;
                                    st.d_wts_renorm.ptr()
                                } else {
                                    self.router_remap.remap_i32(
                                        &self.compute_stream,
                                        st.d_picks.ptr(),
                                        st.d_mapped.ptr(),
                                        layer_map_ptr,
                                        (rows * slots) as i32,
                                        e_cnt as i32,
                                        e_cnt as i32,
                                        vram_slots as i32,
                                    )?;
                                    st.d_wts.ptr()
                                };
                                self.exl3.run_moe(
                                    &self.compute_stream,
                                    &view,
                                    &st.exl3_sc,
                                    st.normed.ptr(),
                                    d,
                                    InputDType::Bf16,
                                    st.d_mapped.ptr(),
                                    wts_ptr,
                                    st.branch.ptr(),
                                    rows,
                                    slots,
                                    true,
                                )?;
                                if has_cpu_jobs {
                                    self.cpu_exl3.wait_async();
                                    unsafe {
                                        self.compute_stream.htod_async(
                                            st.d_cpu_out.ptr(),
                                            st.h_cpu_out.as_ptr(),
                                            rows * d * 4,
                                        )?;
                                    }
                                    self.glm.add_f32_inplace(
                                        &self.compute_stream,
                                        st.branch.ptr(),
                                        st.d_cpu_out.ptr(),
                                        rows * d,
                                    )?;
                                }
                                if let Some(t) = tp_exl3 {
                                    self.compute_stream.synchronize()?;
                                    self.decode_prof.exl3_secs += t.elapsed().as_secs_f64();
                                }
                            }

                            // 6. Focused Async L+1 Rank-0 RAM->VRAM Pre-Promotion + Low-Noise NVMe Lookahead!
                            //    Runs AFTER launching Layer L GPU MoE and BEFORE `mark_compute_done()` so
                            //    copy_stream & prefetch_ring overlap 100% with Layer L MoE + Layer L+1 Attention!
                            let tp_spec = if do_prof { Some(Instant::now()) } else { None };
                            if do_spec {
                                st.spec_keys.clear();
                                st.spec_ram_keys.clear();
                                let next_base = (moe_idx + 1) * e_cnt;
                                let sp = &st.h_spec_picks.as_i32_mut()[..rows * slots];
                                // Pre-promote strictly 1 L+1 Rank-0 RAM expert (~0.25 ms PCIe, hidden in L Wave 2 + L+1 Attn!):
                                // Proven +5.4% Tier-1 VRAM hit rate and -4 ms/rd vs 0, without the eviction thrash of >=2!
                                let bg_max: usize = std::env::var("M3TIS_BG_PROMOTE")
                                    .ok()
                                    .and_then(|v| v.parse().ok())
                                    .unwrap_or(1);
                                if bg_max > 0 {
                                    let ram_rows = rows.min(4);
                                    for r in 0..ram_rows {
                                        let p = sp[r * slots];
                                        if p >= 0 && (p as usize) < e_cnt {
                                            let gk = next_base + (p as usize);
                                            if !st.spec_ram_keys.contains(&gk) {
                                                st.spec_ram_keys.push(gk);
                                            }
                                        }
                                    }
                                    let _ = self.pool.promote_next_layer_ram_async(
                                        &st.spec_ram_keys,
                                        bg_max,
                                        &mut self.scales,
                                    )?;
                                }

                                // Separate NVMe prefetch list: strictly L+1 Rank-0 across active verify rows
                                let spec_k_lim = if use_nvme_budget {
                                    opts.nvme_strict_k.min(self.cfg.top_k)
                                } else {
                                    self.cfg.top_k
                                };
                                let spec_row_lim = if use_nvme_budget {
                                    if is_verify && opts.nvme_tail_row > 0 {
                                        opts.nvme_tail_row.min(rows)
                                    } else {
                                        rows.min(2)
                                    }
                                } else {
                                    rows
                                };
                                for k_rank in 0..spec_k_lim {
                                    for r in 0..spec_row_lim {
                                        let p = sp[r * slots + k_rank];
                                        if p >= 0 && (p as usize) < e_cnt {
                                            let gk = next_base + (p as usize);
                                            if !st.spec_keys.contains(&gk) {
                                                st.spec_keys.push(gk);
                                            }
                                        }
                                    }
                                }
                                if do_spec2 {
                                    let next2_base = (moe_idx + 2) * e_cnt;
                                    let sp2 = &st.h_spec2_picks.as_i32_mut()[..rows * slots];
                                    for k_rank in 0..spec_k_lim.min(2) {
                                        for r in 0..spec_row_lim {
                                            let p = sp2[r * slots + k_rank];
                                            if p >= 0 && (p as usize) < e_cnt {
                                                let gk = next2_base + (p as usize);
                                                if !st.spec_keys.contains(&gk) {
                                                    st.spec_keys.push(gk);
                                                }
                                            }
                                        }
                                    }
                                }
                                // Only warm recurring skipped cold experts when explicit `--spec > 0` is requested!
                                if use_nvme_budget && opts.spec_prefetch_p > 0 && !st.skipped_cold.is_empty() {
                                    for &k in &st.skipped_cold {
                                        if self.pool.ram_cache.hist_freq(k) >= 1 && !st.spec_keys.contains(&k) {
                                            st.spec_keys.push(k);
                                        }
                                    }
                                }
                                let nvme_p = if opts.spec_prefetch_p > 0 {
                                    opts.spec_prefetch_p
                                } else {
                                    1
                                };
                                let _ = self.pool.start_spec_prefetch(&st.spec_keys, nvme_p);
                            }
                            if let Some(t) = tp_spec {
                                self.decode_prof.spec_secs += t.elapsed().as_secs_f64();
                            }
                            self.pool.mark_compute_done(&self.compute_stream)?;
                        }
                    }
                }

                self.glm.hc_post(
                    &self.compute_stream,
                    st.x.ptr(),
                    st.branch.ptr(),
                    st.post.ptr(),
                    st.comb.ptr(),
                    rows,
                    d,
                )?;
                if capture_taps {
                    if let Some(slot) = tap_slot_for_layer(l) {
                        self.glm.tap_stream_mean(
                            &self.compute_stream,
                            st.x.ptr(),
                            st.taps_buf.ptr(),
                            rows,
                            d,
                            slot,
                        )?;
                    }
                }
            }
        }

        if is_warp_zero_sync {
            self.pool.mark_compute_done(&self.compute_stream)?;
        } else if is_decode_step && !is_draft_resident && st.warp_heal_steps > 0 {
            st.warp_heal_steps -= 1;
        }

        if !is_verify {
            st.pos += rows;
        }

        // 3. Final stream mean -> RMSNorm -> lm_head (+ batched 42-layer zero-sync pick/wt DtoH!)
        let tp_head = if do_prof { self.compute_stream.synchronize()?; Some(Instant::now()) } else { None };
        let all_moe_bytes = self.cfg.moe_layers() * rows * slots * 4;
        if compute_logits {
            if is_verify || is_draft_resident {
                self.glm.stream_mean_bf16(
                    &self.compute_stream,
                    st.x.ptr(),
                    st.final_hidden.ptr(),
                    rows,
                    d,
                    s,
                )?;
                self.glm.rmsnorm_bf16(
                    &self.compute_stream,
                    st.final_hidden.ptr(),
                    d,
                    self.final_norm.ptr(),
                    st.final_normed.ptr(),
                    d,
                    rows,
                    d,
                    self.cfg.rms_eps,
                )?;
                self.lm_head.apply(
                    &self.lin,
                    &self.compute_stream,
                    st.final_normed.ptr(),
                    d,
                    st.d_logits.ptr(),
                    self.cfg.vocab,
                    rows,
                    true,
                )?;
                if is_draft_resident {
                    self.glm.argmax_prob_f32(
                        &self.compute_stream,
                        st.d_logits.ptr(),
                        st.d_argmax_ids.ptr(),
                        st.d_argmax_vals.ptr(),
                        rows,
                        self.cfg.vocab,
                    )?;
                    unsafe {
                        self.compute_stream.dtoh_async(
                            st.h_argmax_ids.as_mut_ptr(),
                            st.d_argmax_ids.ptr(),
                            rows * 4,
                        )?;
                        self.compute_stream.dtoh_async(
                            st.h_argmax_probs.as_mut_ptr(),
                            st.d_argmax_vals.ptr(),
                            rows * 4,
                        )?;
                        self.compute_stream.dtoh_async(
                            st.h_all_picks.as_mut_ptr(),
                            st.d_all_picks.ptr(),
                            all_moe_bytes,
                        )?;
                    }
                } else {
                    self.glm.argmax_f32(
                        &self.compute_stream,
                        st.d_logits.ptr(),
                        st.d_argmax_ids.ptr(),
                        st.d_argmax_vals.ptr(),
                        rows,
                        self.cfg.vocab,
                    )?;
                    unsafe {
                        self.compute_stream.dtoh_async(
                            st.h_argmax_ids.as_mut_ptr(),
                            st.d_argmax_ids.ptr(),
                            rows * 4,
                        )?;
                        if is_warp_zero_sync {
                            self.compute_stream.dtoh_async(
                                st.h_all_picks.as_mut_ptr(),
                                st.d_all_picks.ptr(),
                                all_moe_bytes,
                            )?;
                            self.compute_stream.dtoh_async(
                                st.h_all_wts.as_mut_ptr(),
                                st.d_all_wts.ptr(),
                                all_moe_bytes,
                            )?;
                        }
                    }
                }
                self.compute_stream.synchronize()?;
            } else {
                // Prefill / serial decode: last row full FP32 logits
                let last_x = st.x.at((rows - 1) * s * d * 2);
                self.glm.stream_mean_bf16(
                    &self.compute_stream,
                    last_x,
                    st.final_hidden.ptr(),
                    1,
                    d,
                    s,
                )?;
                self.glm.rmsnorm_bf16(
                    &self.compute_stream,
                    st.final_hidden.ptr(),
                    d,
                    self.final_norm.ptr(),
                    st.final_normed.ptr(),
                    d,
                    1,
                    d,
                    self.cfg.rms_eps,
                )?;
                self.lm_head.apply(
                    &self.lin,
                    &self.compute_stream,
                    st.final_normed.ptr(),
                    d,
                    st.d_logits.ptr(),
                    self.cfg.vocab,
                    1,
                    true,
                )?;
                unsafe {
                    self.compute_stream.dtoh_async(
                        st.h_logits.as_mut_ptr(),
                        st.d_logits.ptr(),
                        self.cfg.vocab * 4,
                    )?;
                    if is_warp_zero_sync {
                        self.compute_stream.dtoh_async(
                            st.h_all_picks.as_mut_ptr(),
                            st.d_all_picks.ptr(),
                            all_moe_bytes,
                        )?;
                        self.compute_stream.dtoh_async(
                            st.h_all_wts.as_mut_ptr(),
                            st.d_all_wts.ptr(),
                            all_moe_bytes,
                        )?;
                    }
                }
                self.compute_stream.synchronize()?;
            }
            if is_warp_zero_sync {
                self.post_warp_zero_sync(st, rows, is_verify, opts)?;
            }
        }
        if let Some(t) = tp_head {
            self.decode_prof.head_secs += t.elapsed().as_secs_f64();
        }

        Ok(rows)
    }

    /// Milestone 5B: Expert-Major Double-Buffered Streaming Prefill (`B <= pf.max_rows`, default 1024).
    /// Streams each active expert ONCE per layer (Tier-1 VRAM -> Tier-2 RAM -> NVMe sorted by pack offset)
    /// through a dedicated double-buffered 16-slot ring, NEVER evicting warm Tier-1/Tier-2 decode caches!
    fn forward_prefill_streaming(
        &mut self,
        st: &mut Glm53State,
        tokens: &[i32],
        capture_taps: bool,
        compute_logits: bool,
    ) -> Result<(), Box<dyn Error>> {
        let mut pf = st.pf.take().ok_or("PfState not allocated")?;
        let res = self.forward_prefill_streaming_inner(st, &mut pf, tokens, capture_taps, compute_logits);
        self.pool.pf_finish_ahead();
        st.pf = Some(pf);
        res
    }

    fn emit_prefill_waves(
        exl3: &Exl3MoeKernels,
        stream: &Stream,
        view: &Exl3ExpertsView,
        pf: &mut PfState,
        normed_ptr: u64,
        branch_ptr: u64,
        dims: usize,
        experts_with_slot: &[(usize, usize)],
    ) -> Result<(), Box<dyn Error>> {
        let max_pairs = pf.exl3_sc.max_pairs;
        let max_tiles = pf.exl3_sc.max_tiles;
        pf.pair_row.clear();
        pf.pair_slot.clear();
        pf.pair_wt.clear();
        pf.tile_slot.clear();
        pf.tile_off.clear();
        pf.tile_cnt.clear();

        for &(local_e, slot) in experts_with_slot {
            let toks = &pf.exp_tokens[local_e];
            let mut idx = 0usize;
            while idx < toks.len() {
                let c = (toks.len() - idx).min(16);
                if pf.pair_row.len() + c > max_pairs || pf.tile_slot.len() + 1 > max_tiles {
                    exl3.run_prefill_wave(
                        stream,
                        view,
                        &mut pf.exl3_sc,
                        pf.wave_seq,
                        normed_ptr,
                        dims,
                        branch_ptr,
                        &pf.pair_row,
                        &pf.pair_slot,
                        &pf.pair_wt,
                        &pf.tile_slot,
                        &pf.tile_off,
                        &pf.tile_cnt,
                    )?;
                    pf.wave_seq += 1;
                    pf.pair_row.clear();
                    pf.pair_slot.clear();
                    pf.pair_wt.clear();
                    pf.tile_slot.clear();
                    pf.tile_off.clear();
                    pf.tile_cnt.clear();
                }
                let off = pf.pair_row.len() as i32;
                for &(r, w) in &toks[idx..idx + c] {
                    pf.pair_row.push(r);
                    pf.pair_slot.push(slot as i32);
                    pf.pair_wt.push(w);
                }
                pf.tile_slot.push(slot as i32);
                pf.tile_off.push(off);
                pf.tile_cnt.push(c as i32);
                idx += c;
            }
        }
        if !pf.pair_row.is_empty() {
            exl3.run_prefill_wave(
                stream,
                view,
                &mut pf.exl3_sc,
                pf.wave_seq,
                normed_ptr,
                dims,
                branch_ptr,
                &pf.pair_row,
                &pf.pair_slot,
                &pf.pair_wt,
                &pf.tile_slot,
                &pf.tile_off,
                &pf.tile_cnt,
            )?;
            pf.wave_seq += 1;
            pf.pair_row.clear();
            pf.pair_slot.clear();
            pf.pair_wt.clear();
            pf.tile_slot.clear();
            pf.tile_off.clear();
            pf.tile_cnt.clear();
        }
        Ok(())
    }

    fn forward_prefill_streaming_inner(
        &mut self,
        st: &mut Glm53State,
        pf: &mut PfState,
        tokens: &[i32],
        capture_taps: bool,
        compute_logits: bool,
    ) -> Result<(), Box<dyn Error>> {
        let rows = tokens.len();
        assert!(rows > 0 && rows <= pf.max_rows);
        let d = self.cfg.hidden;
        let s = self.cfg.hc_mult;
        let row_bytes = d * 2;
        let micro = st.max_rows;
        let pos0 = st.pos;

        self.pool.pf_begin(&self.compute_stream)?;
        let do_prof = std::env::var_os("M3TIS_PF_PROFILE").is_some();
        let do_detail = std::env::var_os("M3TIS_PF_DETAIL").is_some();
        let (mut t_attn, mut t_b1, mut t_vram, mut t_ram, mut t_nvme_io, mut t_nvme_gpu, mut t_post) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut td_hc, mut td_kda_q4, mut td_kda_bf16, mut td_kda_chain) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut td_mla_q4, mut td_mla_absexp, mut td_mla_dsa, mut td_mla_att) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut n_vram_tot, mut n_ram_tot, mut n_nvme_tot) = (0usize, 0usize, 0usize);

        // Milestone 6: For medium/long prefill super-batches (`rows >= 384`, where 93-99.9% of all experts
        // activate), kick off our dedicated Layer-Ahead NVMe streamer thread at t = 0 BEFORE embedding and
        // Dense Layers 0..2 — overlapping 100% of sequential 6.8 GB/s NVMe streaming with GPU Attention & MoE!
        let use_layer_ahead = rows >= 384 && self.pool.ahead.is_some();
        if use_layer_ahead {
            pf.ahead_plan.clear();
            pf.ahead_key_to_pidx.fill(-1);
            let moe_n = self.cfg.moe_layers();
            let e_per = self.cfg.num_experts;
            let mut layer_tmp: Vec<(u64, usize)> = Vec::with_capacity(e_per);
            for ml in 0..moe_n {
                layer_tmp.clear();
                let base_k = ml * e_per;
                for le in 0..e_per {
                    let gkey = base_k + le;
                    if self.pool.vram_cache.peek(gkey).is_none()
                        && self.pool.ram_cache.peek(gkey).is_none()
                    {
                        let off = self.pool.catalog.entries[gkey].components[0].offset;
                        layer_tmp.push((off, gkey));
                    }
                }
                layer_tmp.sort_unstable_by_key(|&(off, _)| off);
                for &(_, gkey) in &layer_tmp {
                    let pidx = pf.ahead_plan.len();
                    pf.ahead_key_to_pidx[gkey] = pidx as i32;
                    pf.ahead_plan.push(gkey);
                }
                pf.ahead_layer_end[ml] = pf.ahead_plan.len();
            }
            if let Some(ref mut ah) = self.pool.ahead {
                ah.start_plan(&self.pool.catalog, &pf.ahead_plan);
            }
        }

        // 1. Copy all B token embeddings once -> pf.normed -> broadcast to pf.x[B, 4, D]
        {
            let emb_src = self.embed_host.as_slice();
            let emb_dst = pf.h_embed.as_mut_slice();
            for (r, &tok) in tokens.iter().enumerate() {
                if tok < 0 || (tok as usize) >= self.cfg.vocab {
                    return Err(format!("token id {tok} out of range 0..{}", self.cfg.vocab).into());
                }
                let src_off = (tok as usize) * row_bytes;
                let dst_off = r * row_bytes;
                emb_dst[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&emb_src[src_off..src_off + row_bytes]);
            }
            unsafe {
                self.compute_stream.htod_async(
                    pf.normed.ptr(),
                    pf.h_embed.as_ptr(),
                    rows * row_bytes,
                )?;
            }
            self.glm.embed_broadcast_bf16(
                &self.compute_stream,
                pf.normed.ptr(),
                pf.x.ptr(),
                rows,
                d,
                s,
            )?;
        }

        let kda_rec_stride = self.cfg.kda_heads * self.cfg.kda_dim * self.cfg.kda_dim * 4;
        let kda_conv_stride = 3 * self.cfg.kda_conv_c() * 2;
        let mla_kv_stride = if st.kv_int8 {
            st.max_ctx * self.cfg.kv_lora
        } else {
            st.max_ctx * self.cfg.kv_lora * 2
        };
        let mla_kv_sc_stride = st.max_ctx * 4;
        let p_stride = self.cfg.kda_p_stride();
        let b_off = self.cfg.kda_b_off();
        let n_kda = self.cfg.kda_heads * self.cfg.kda_dim;
        let slots = self.cfg.top_k + 1;
        let vram_slots = self.pool.vram_slots();
        let e_cnt = self.cfg.num_experts;

        let vram_view = Exl3ExpertsView {
            gt_ptrs: self.exl3_ptr_tables[0].ptr(),
            ut_ptrs: self.exl3_ptr_tables[1].ptr(),
            dt_ptrs: self.exl3_ptr_tables[2].ptr(),
            k2_g: self.pool.vram_k2_ptr(0),
            k2_u: self.pool.vram_k2_ptr(1),
            k2_d: self.pool.vram_k2_ptr(2),
            suh_g: self.scales.vram_ptr(0),
            suh_u: self.scales.vram_ptr(1),
            svh_g: self.scales.vram_ptr(2),
            svh_u: self.scales.vram_ptr(3),
            suh_d: self.scales.vram_ptr(4),
            svh_d: self.scales.vram_ptr(5),
            experts: vram_slots,
            dims: d,
            inter: self.cfg.moe_inter,
            swiglu_limit: self.cfg.swiglu_limit,
            act_mode: 0,
        };
        let stage_view = Exl3ExpertsView {
            gt_ptrs: self.pool.pf_stage_ptr_table(0),
            ut_ptrs: self.pool.pf_stage_ptr_table(1),
            dt_ptrs: self.pool.pf_stage_ptr_table(2),
            k2_g: self.pool.pf_stage_k2_ptr(0),
            k2_u: self.pool.pf_stage_k2_ptr(1),
            k2_d: self.pool.pf_stage_k2_ptr(2),
            suh_g: self.scales.pf_vram_ptr(0),
            suh_u: self.scales.pf_vram_ptr(1),
            svh_g: self.scales.pf_vram_ptr(2),
            svh_u: self.scales.pf_vram_ptr(3),
            suh_d: self.scales.pf_vram_ptr(4),
            svh_d: self.scales.pf_vram_ptr(5),
            experts: PF_SCALE_SLOTS,
            dims: d,
            inter: self.cfg.moe_inter,
            swiglu_limit: self.cfg.swiglu_limit,
            act_mode: 0,
        };

        // 2. Run all 45 layers
        for l in 0..self.layers.len() {
            // --- A. Attention / Mixer Sublayer (micro-batched to `micro` rows -> 0 extra activation VRAM!) ---
            let tp0 = if do_prof { Some(Instant::now()) } else { None };
            {
                let layer = &self.layers[l];
                let mut r0 = 0usize;
                while r0 < rows {
                    let m = (rows - r0).min(micro);
                    let x_ptr = pf.x.at(r0 * s * d * 2);
                    let cur_pos = pos0 + r0;
                    let dt0 = if do_detail { self.compute_stream.synchronize()?; Some(Instant::now()) } else { None };
                    self.glm.hc_pre(
                        &self.compute_stream,
                        x_ptr,
                        layer.attn_hc.fn_w.ptr(),
                        st.hc_part.ptr(),
                        layer.attn_hc.base.ptr(),
                        layer.attn_hc.scale.ptr(),
                        layer.in_norm.ptr(),
                        st.normed.ptr(),
                        st.post.ptr(),
                        st.comb.ptr(),
                        m,
                        d,
                        self.cfg.rms_eps,
                        self.cfg.hc_eps,
                        self.cfg.hc_sinkhorn_iters,
                    )?;
                    if let Some(t) = dt0 { self.compute_stream.synchronize()?; td_hc += t.elapsed().as_secs_f64(); }
                    match &layer.mixer {
                        MixerWeights::Kda { idx, w } => {
                            let dt1 = if do_detail { Some(Instant::now()) } else { None };
                            w.qkv_proj.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                st.kda_p.ptr(),
                                p_stride,
                                m,
                                false,
                            )?;
                            if let Some(t) = dt1 { self.compute_stream.synchronize()?; td_kda_q4 += t.elapsed().as_secs_f64(); }
                            let dt2 = if do_detail { Some(Instant::now()) } else { None };
                            let fa_off = 3 * n_kda;
                            self.lin.linear_bf16(
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                w.fgb_proj.ptr(),
                                st.kda_p.at(fa_off * 2),
                                p_stride,
                                m,
                                2 * self.cfg.kda_dim + self.cfg.kda_heads,
                                d,
                                false,
                            )?;
                            self.lin.linear_bf16(
                                &self.compute_stream,
                                st.kda_p.at(fa_off * 2),
                                p_stride,
                                w.f_b_proj.ptr(),
                                st.kda_ka.ptr(),
                                n_kda,
                                m,
                                n_kda,
                                self.cfg.kda_dim,
                                false,
                            )?;
                            self.lin.linear_bf16(
                                &self.compute_stream,
                                st.kda_p.at((fa_off + self.cfg.kda_dim) * 2),
                                p_stride,
                                w.g_b_proj.ptr(),
                                st.kda_kg.ptr(),
                                n_kda,
                                m,
                                n_kda,
                                self.cfg.kda_dim,
                                false,
                            )?;
                            if let Some(t) = dt2 { self.compute_stream.synchronize()?; td_kda_bf16 += t.elapsed().as_secs_f64(); }
                            let dt3 = if do_detail { Some(Instant::now()) } else { None };
                            let cur = st.kda_cur;
                            let rec_in = st.kda_rec[cur].at(idx * kda_rec_stride);
                            let conv_ptr = st.kda_conv.at(idx * kda_conv_stride);
                            self.glm.kda_chain(
                                &self.compute_stream,
                                self.cfg.kda_heads,
                                st.kda_p.ptr(),
                                p_stride,
                                b_off,
                                st.kda_ka.ptr(),
                                st.kda_kg.ptr(),
                                conv_ptr,
                                w.conv_w.ptr(),
                                rec_in,
                                w.a_log.ptr(),
                                w.dt_bias.ptr(),
                                w.o_norm.ptr(),
                                self.cfg.rms_eps,
                                self.cfg.kda_lower,
                                m,
                                st.kda_out.ptr(),
                                rec_in,
                            )?;
                            self.glm.kda_conv_shift(
                                &self.compute_stream,
                                conv_ptr,
                                st.kda_p.ptr(),
                                m,
                                p_stride,
                                self.cfg.kda_conv_c(),
                            )?;
                            if let Some(t) = dt3 { self.compute_stream.synchronize()?; td_kda_chain += t.elapsed().as_secs_f64(); }
                            let dt4 = if do_detail { Some(Instant::now()) } else { None };
                            w.o_proj.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.kda_out.ptr(),
                                n_kda,
                                st.branch.ptr(),
                                d,
                                m,
                                true,
                            )?;
                            if let Some(t) = dt4 { self.compute_stream.synchronize()?; td_kda_q4 += t.elapsed().as_secs_f64(); }
                        }
                        MixerWeights::Mla { idx, w } => {
                            let dt1 = if do_detail { Some(Instant::now()) } else { None };
                            w.q_a_proj.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                st.mla_qa_raw.ptr(),
                                self.cfg.q_lora,
                                m,
                                false,
                            )?;
                            w.kv_a_proj.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                st.mla_kva_raw.ptr(),
                                self.cfg.kv_lora,
                                m,
                                false,
                            )?;
                            self.glm.rmsnorm_bf16(
                                &self.compute_stream,
                                st.mla_qa_raw.ptr(),
                                self.cfg.q_lora,
                                w.q_a_norm.ptr(),
                                st.mla_qr.ptr(),
                                self.cfg.q_lora,
                                m,
                                self.cfg.q_lora,
                                self.cfg.rms_eps,
                            )?;
                            self.glm.rmsnorm_bf16(
                                &self.compute_stream,
                                st.mla_kva_raw.ptr(),
                                self.cfg.kv_lora,
                                w.kv_a_norm.ptr(),
                                st.mla_lat.ptr(),
                                self.cfg.kv_lora,
                                m,
                                self.cfg.kv_lora,
                                self.cfg.rms_eps,
                            )?;
                            w.q_b_proj.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.mla_qr.ptr(),
                                self.cfg.q_lora,
                                st.mla_q.ptr(),
                                self.cfg.mla_heads * self.cfg.qk_nope,
                                m,
                                false,
                            )?;
                            if let Some(t) = dt1 { self.compute_stream.synchronize()?; td_mla_q4 += t.elapsed().as_secs_f64(); }
                            let dt2 = if do_detail { Some(Instant::now()) } else { None };
                            self.glm.mla_absorb_q(
                                &self.compute_stream,
                                st.mla_q.ptr(),
                                w.wk.ptr(),
                                st.mla_qa.ptr(),
                                m,
                                self.cfg.mla_heads,
                                self.cfg.qk_nope,
                                self.cfg.kv_lora,
                            )?;
                            let kv_ptr = st.mla_kv.at(idx * mla_kv_stride);
                            let kv_sc_ptr = st.mla_kv_sc.as_ref().map(|s| s.at(idx * mla_kv_sc_stride));
                            if let Some(sc_ptr) = kv_sc_ptr {
                                self.glm.mla_write_kv_q8(
                                    &self.compute_stream,
                                    st.mla_lat.ptr(),
                                    kv_ptr,
                                    sc_ptr,
                                    cur_pos,
                                    st.max_ctx,
                                    m,
                                    self.cfg.kv_lora,
                                )?;
                            } else {
                                self.glm.mla_write_kv(
                                    &self.compute_stream,
                                    st.mla_lat.ptr(),
                                    kv_ptr,
                                    cur_pos,
                                    st.max_ctx,
                                    m,
                                    self.cfg.kv_lora,
                                )?;
                            }
                            if let Some(t) = dt2 { self.compute_stream.synchronize()?; td_mla_absexp += t.elapsed().as_secs_f64(); }
                            let scale = 1.0f32 / (self.cfg.qk_nope as f32).sqrt();
                            let dt3 = if do_detail { Some(Instant::now()) } else { None };
                            let use_sparse = if let (Some(ix), Some(dsa)) = (w.indexer.as_ref(), st.dsa.as_ref()) {
                                self.lin.linear_bf16(
                                    &self.compute_stream,
                                    st.normed.ptr(),
                                    d,
                                    ix.wk.ptr(),
                                    dsa.k_raw.ptr(),
                                    DSA_DIM,
                                    m,
                                    DSA_DIM,
                                    d,
                                    false,
                                )?;
                                self.lin.linear_bf16(
                                    &self.compute_stream,
                                    st.normed.ptr(),
                                    d,
                                    ix.w_proj.ptr(),
                                    dsa.w_raw.ptr(),
                                    DSA_HEADS,
                                    m,
                                    DSA_HEADS,
                                    d,
                                    false,
                                )?;
                                self.lin.linear_bf16(
                                    &self.compute_stream,
                                    st.normed.ptr(),
                                    d,
                                    ix.gate.ptr(),
                                    dsa.g_raw.ptr(),
                                    DSA_DIM,
                                    m,
                                    DSA_DIM,
                                    d,
                                    true,
                                )?;
                                let ik_stride = dsa.ik_cap * DSA_DIM * 2;
                                let pk_stride = dsa.pk_cap * DSA_DIM * 2;
                                let ik_ptr = dsa.ik.at(idx * ik_stride);
                                let ig_ptr = dsa.ig.at(idx * ik_stride);
                                let pk_ptr = dsa.pk.at(idx * pk_stride);
                                self.glm.dsa_index_write(
                                    &self.compute_stream,
                                    dsa.k_raw.ptr(),
                                    dsa.g_raw.ptr(),
                                    ix.k_norm_w.ptr(),
                                    ix.k_norm_b.ptr(),
                                    ik_ptr,
                                    ig_ptr,
                                    cur_pos,
                                    dsa.ik_cap,
                                    1e-6,
                                    m,
                                )?;
                                self.glm.dsa_pool_keys(
                                    &self.compute_stream,
                                    ik_ptr,
                                    ig_ptr,
                                    ix.ape.ptr(),
                                    pk_ptr,
                                    cur_pos,
                                    m,
                                    dsa.ik_cap,
                                    dsa.pk_cap,
                                )?;
                                if cur_pos + m > DSA_WIDTH {
                                    ix.wq_b.apply(
                                        &self.lin,
                                        &self.compute_stream,
                                        st.mla_qr.ptr(),
                                        self.cfg.q_lora,
                                        dsa.qi.ptr(),
                                        DSA_HEADS * DSA_DIM,
                                        m,
                                        false,
                                    )?;
                                    let np_vis =
                                        ((cur_pos + m) / DSA_KPOOL).min(dsa.pk_cap).max(DSA_TOPK_POOLS + 1);
                                    let qscale = (DSA_DIM as f32).powf(-0.5);
                                    let wscale = (DSA_HEADS as f32).powf(-0.5);
                                    self.glm.dsa_score_pools(
                                        &self.compute_stream,
                                        dsa.qi.ptr(),
                                        dsa.w_raw.ptr(),
                                        pk_ptr,
                                        dsa.scores.ptr(),
                                        cur_pos,
                                        np_vis,
                                        dsa.pk_cap,
                                        DSA_HEADS,
                                        qscale,
                                        wscale,
                                        m,
                                    )?;
                                    self.glm.dsa_select_tokens(
                                        &self.compute_stream,
                                        dsa.scores.ptr(),
                                        dsa.tokens.ptr(),
                                        dsa.counts.ptr(),
                                        cur_pos,
                                        np_vis,
                                        DSA_TOPK_POOLS,
                                        DSA_TOKEN_STRIDE,
                                        m,
                                        0,
                                        0,
                                    )?;
                                    if let Some(t) = dt3 { self.compute_stream.synchronize()?; td_mla_dsa += t.elapsed().as_secs_f64(); }
                                    let dt4 = if do_detail { Some(Instant::now()) } else { None };
                                    if let Some(sc_ptr) = kv_sc_ptr {
                                        self.glm.mla_attend_sparse_q8(
                                            &self.compute_stream,
                                            st.mla_qa.ptr(),
                                            kv_ptr,
                                            sc_ptr,
                                            dsa.tokens.ptr(),
                                            dsa.counts.ptr(),
                                            st.mla_ol.ptr(),
                                            cur_pos,
                                            st.max_ctx,
                                            DSA_TOKEN_STRIDE,
                                            m,
                                            self.cfg.mla_heads,
                                            self.cfg.kv_lora,
                                            scale,
                                            0,
                                            0,
                                        )?;
                                    } else {
                                        self.glm.mla_attend_sparse(
                                            &self.compute_stream,
                                            st.mla_qa.ptr(),
                                            kv_ptr,
                                            dsa.tokens.ptr(),
                                            dsa.counts.ptr(),
                                            st.mla_ol.ptr(),
                                            cur_pos,
                                            st.max_ctx,
                                            DSA_TOKEN_STRIDE,
                                            m,
                                            self.cfg.mla_heads,
                                            self.cfg.kv_lora,
                                            scale,
                                            0,
                                            0,
                                        )?;
                                    }
                                    if let Some(t) = dt4 { self.compute_stream.synchronize()?; td_mla_att += t.elapsed().as_secs_f64(); }
                                    true
                                } else {
                                    if let Some(t) = dt3 { self.compute_stream.synchronize()?; td_mla_dsa += t.elapsed().as_secs_f64(); }
                                    false
                                }
                            } else {
                                false
                            };
                            if !use_sparse {
                                let dt4 = if do_detail { Some(Instant::now()) } else { None };
                                if let Some(sc_ptr) = kv_sc_ptr {
                                    self.glm.mla_attend_dense_q8(
                                        &self.compute_stream,
                                        st.mla_qa.ptr(),
                                        kv_ptr,
                                        sc_ptr,
                                        st.mla_ol.ptr(),
                                        cur_pos,
                                        st.max_ctx,
                                        st.max_ctx,
                                        m,
                                        self.cfg.mla_heads,
                                        self.cfg.kv_lora,
                                        scale,
                                        0,
                                        0,
                                    )?;
                                } else {
                                    self.glm.mla_attend_dense(
                                        &self.compute_stream,
                                        st.mla_qa.ptr(),
                                        kv_ptr,
                                        st.mla_ol.ptr(),
                                        cur_pos,
                                        st.max_ctx,
                                        st.max_ctx,
                                        m,
                                        self.cfg.mla_heads,
                                        self.cfg.kv_lora,
                                        scale,
                                        0,
                                        0,
                                    )?;
                                }
                                if let Some(t) = dt4 { self.compute_stream.synchronize()?; td_mla_att += t.elapsed().as_secs_f64(); }
                            }
                            let dt5 = if do_detail { Some(Instant::now()) } else { None };
                            self.glm.mla_expand_v(
                                &self.compute_stream,
                                st.mla_ol.ptr(),
                                w.wv.ptr(),
                                st.mla_vn.ptr(),
                                m,
                                self.cfg.mla_heads,
                                self.cfg.dv,
                                self.cfg.kv_lora,
                            )?;
                            if let Some(t) = dt5 { self.compute_stream.synchronize()?; td_mla_absexp += t.elapsed().as_secs_f64(); }
                            let dt6 = if do_detail { Some(Instant::now()) } else { None };
                            w.o_proj.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.mla_vn.ptr(),
                                self.cfg.mla_heads * self.cfg.dv,
                                st.branch.ptr(),
                                d,
                                m,
                                true,
                            )?;
                            if let Some(t) = dt6 { self.compute_stream.synchronize()?; td_mla_q4 += t.elapsed().as_secs_f64(); }
                        }
                    }
                    let dt7 = if do_detail { Some(Instant::now()) } else { None };
                    self.glm.hc_post(
                        &self.compute_stream,
                        x_ptr,
                        st.branch.ptr(),
                        st.post.ptr(),
                        st.comb.ptr(),
                        m,
                        d,
                    )?;
                    if let Some(t) = dt7 { self.compute_stream.synchronize()?; td_hc += t.elapsed().as_secs_f64(); }
                    r0 += m;
                }
            }
            if let Some(t0) = tp0 {
                self.compute_stream.synchronize()?;
                t_attn += t0.elapsed().as_secs_f64();
            }

            // --- B. FFN / Expert-Major Streaming MoE Sublayer ---
            {
                let layer = &self.layers[l];
                match &layer.ffn {
                    FfnWeights::Dense(m_ffn) => {
                        let mut r0 = 0usize;
                        while r0 < rows {
                            let m = (rows - r0).min(micro);
                            let x_ptr = pf.x.at(r0 * s * d * 2);
                            self.glm.hc_pre(
                                &self.compute_stream,
                                x_ptr,
                                layer.ffn_hc.fn_w.ptr(),
                                st.hc_part.ptr(),
                                layer.ffn_hc.base.ptr(),
                                layer.ffn_hc.scale.ptr(),
                                layer.post_norm.ptr(),
                                st.normed.ptr(),
                                st.post.ptr(),
                                st.comb.ptr(),
                                m,
                                d,
                                self.cfg.rms_eps,
                                self.cfg.hc_eps,
                                self.cfg.hc_sinkhorn_iters,
                            )?;
                            m_ffn.gate.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                st.mlp_g.ptr(),
                                self.cfg.dense_inter,
                                m,
                                false,
                            )?;
                            m_ffn.up.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.normed.ptr(),
                                d,
                                st.mlp_u.ptr(),
                                self.cfg.dense_inter,
                                m,
                                false,
                            )?;
                            self.lin.swiglu_bf16(
                                &self.compute_stream,
                                st.mlp_g.ptr(),
                                st.mlp_u.ptr(),
                                st.mlp_act.ptr(),
                                m,
                                self.cfg.dense_inter,
                                self.cfg.swiglu_limit,
                            )?;
                            m_ffn.down.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.mlp_act.ptr(),
                                self.cfg.dense_inter,
                                st.branch.ptr(),
                                d,
                                m,
                                true,
                            )?;
                            self.glm.hc_post(
                                &self.compute_stream,
                                x_ptr,
                                st.branch.ptr(),
                                st.post.ptr(),
                                st.comb.ptr(),
                                m,
                                d,
                            )?;
                            r0 += m;
                        }
                    }
                    FfnWeights::Moe(m_moe) => {
                        let moe_idx = m_moe.moe_idx;
                        let base_key = moe_idx * e_cnt;
                        let tb1 = if do_prof { Some(Instant::now()) } else { None };

                        // B1. Micro-batched hc_pre -> pf.normed/post/comb + exact router -> pf.d_picks/d_wts
                        //     + shared expert directly initializing pf.branch[0..rows, D]!
                        let mut r0 = 0usize;
                        while r0 < rows {
                            let m = (rows - r0).min(micro);
                            let x_ptr = pf.x.at(r0 * s * d * 2);
                            let norm_ptr = pf.normed.at(r0 * d * 2);
                            let post_ptr = pf.post.at(r0 * s * 4);
                            let comb_ptr = pf.comb.at(r0 * s * s * 4);
                            let branch_ptr = pf.branch.at(r0 * d * 4);
                            self.glm.hc_pre(
                                &self.compute_stream,
                                x_ptr,
                                layer.ffn_hc.fn_w.ptr(),
                                st.hc_part.ptr(),
                                layer.ffn_hc.base.ptr(),
                                layer.ffn_hc.scale.ptr(),
                                layer.post_norm.ptr(),
                                norm_ptr,
                                post_ptr,
                                comb_ptr,
                                m,
                                d,
                                self.cfg.rms_eps,
                                self.cfg.hc_eps,
                                self.cfg.hc_sinkhorn_iters,
                            )?;
                            self.lin.linear_bf16(
                                &self.compute_stream,
                                norm_ptr,
                                d,
                                m_moe.router.ptr(),
                                st.router_logits.ptr(),
                                e_cnt,
                                m,
                                e_cnt,
                                d,
                                true,
                            )?;
                            self.router_remap.sigmoid_topk_residency(
                                &self.compute_stream,
                                st.router_logits.ptr(),
                                m_moe.bias.ptr(),
                                pf.d_picks.at(r0 * slots * 4),
                                pf.d_wts.at(r0 * slots * 4),
                                m as i32,
                                e_cnt as i32,
                                self.cfg.top_k as i32,
                                self.cfg.routed_scale,
                                0.0,
                                self.cfg.top_k as i32,
                                e_cnt as i32,
                                0,
                                0.0,
                                self.cfg.top_k as i32,
                            )?;
                            // Shared expert writes directly into pf.branch (slot weight == 1.0!)
                            m_moe.shared_gu.apply(
                                &self.lin,
                                &self.compute_stream,
                                norm_ptr,
                                d,
                                st.mlp_g.ptr(),
                                2 * self.cfg.shared_inter,
                                m,
                                false,
                            )?;
                            self.lin.swiglu_split_bf16(
                                &self.compute_stream,
                                st.mlp_g.ptr(),
                                st.mlp_act.ptr(),
                                m,
                                self.cfg.shared_inter,
                                self.cfg.swiglu_limit,
                            )?;
                            m_moe.shared_down.apply(
                                &self.lin,
                                &self.compute_stream,
                                st.mlp_act.ptr(),
                                self.cfg.shared_inter,
                                branch_ptr,
                                d,
                                m,
                                true,
                            )?;
                            r0 += m;
                        }

                        // B2. Copy tiny [rows, 9] picks & weights to pinned host RAM and wait ONLY for them
                        unsafe {
                            self.compute_stream.dtoh_async(
                                pf.h_picks.as_mut_ptr(),
                                pf.d_picks.ptr(),
                                rows * slots * 4,
                            )?;
                            self.compute_stream.dtoh_async(
                                pf.h_wts.as_mut_ptr(),
                                pf.d_wts.ptr(),
                                rows * slots * 4,
                            )?;
                        }
                        self.picks_ready.record(&self.compute_stream)?;
                        self.picks_ready.synchronize()?;
                        if let Some(t0) = tb1 {
                            t_b1 += t0.elapsed().as_secs_f64();
                        }

                        // B3. Group all (rows * top_k) token assignments by local expert ID (0..288)
                        for v in &mut pf.exp_tokens {
                            v.clear();
                        }
                        {
                            let hp = &pf.h_picks.as_i32_mut()[..rows * slots];
                            let hw = &pf.h_wts.as_f32_mut()[..rows * slots];
                            for r in 0..rows {
                                for k_rank in 0..self.cfg.top_k {
                                    let p = hp[r * slots + k_rank];
                                    if p >= 0 && (p as usize) < e_cnt {
                                        let w = hw[r * slots + k_rank];
                                        if w != 0.0 {
                                            let le = p as usize;
                                            pf.exp_tokens[le].push((r as i32, w));
                                            self.pool.expert_hits[base_key + le] += 1;
                                        }
                                    }
                                }
                            }
                        }

                        pf.vram_list.clear();
                        pf.vram_scale_pairs.clear();
                        pf.ram_list.clear();
                        pf.nvme_list.clear();
                        for le in 0..e_cnt {
                            if pf.exp_tokens[le].is_empty() {
                                continue;
                            }
                            let gkey = base_key + le;
                            if let Some(vslot) = self.pool.vram_cache.peek(gkey) {
                                pf.vram_list.push((le, vslot));
                                pf.vram_scale_pairs.push((gkey, vslot));
                            } else if self.pool.ram_cache.peek(gkey).is_some() {
                                pf.ram_list.push((le, gkey));
                            } else {
                                let pack_off = self.pool.catalog.entries[gkey].components[0].offset;
                                pf.nvme_list.push((pack_off, le, gkey));
                            }
                        }
                        // Sort cold NVMe experts in STRICT ASCENDING PACK-FILE OFFSET ORDER (~6.8 GB/s sequential!)
                        pf.nvme_list.sort_unstable_by_key(|&(off, _, _)| off);

                        // B4. CONCURRENT 3-TIER DOUBLE-BUFFERED WAVE PIPELINE (ZERO CACHE EVICTION!)
                        let n_nvme = pf.nvme_list.len();
                        let n_nvme_waves = (n_nvme + PF_WAVE_SLOTS - 1) / PF_WAVE_SLOTS;
                        let mut wave_gkeys = [0usize; PF_WAVE_SLOTS];
                        let mut wave_pairs = [(0usize, 0usize); PF_WAVE_SLOTS];

                        // When NOT using Layer-Ahead (short prompts < 384 rows), kick off NVMe Wave 0 immediately!
                        if !use_layer_ahead && n_nvme_waves > 0 {
                            let w0_end = PF_WAVE_SLOTS.min(n_nvme);
                            for i in 0..w0_end {
                                wave_gkeys[i] = pf.nvme_list[i].2;
                            }
                            self.pool.pf_submit_nvme_wave(0, &wave_gkeys[..w0_end])?;
                        }

                        // Tier-1 VRAM hits: ZERO I/O, ZERO PCIe, execute directly from vram_slabs!
                        let tv0 = if do_prof { Some(Instant::now()) } else { None };
                        n_vram_tot += pf.vram_list.len();
                        n_ram_tot += pf.ram_list.len();
                        n_nvme_tot += n_nvme;
                        if !pf.vram_list.is_empty() {
                            self.pool.record_pf_vram_hits(pf.vram_list.len());
                            self.pool.pf_sync_vram_scales(
                                &self.compute_stream,
                                &pf.vram_scale_pairs,
                                &mut self.scales,
                            )?;
                            let vram_items = std::mem::take(&mut pf.vram_list);
                            let res = Self::emit_prefill_waves(
                                &self.exl3,
                                &self.compute_stream,
                                &vram_view,
                                pf,
                                pf.normed.ptr(),
                                pf.branch.ptr(),
                                d,
                                &vram_items,
                            );
                            pf.vram_list = vram_items;
                            res?;
                        }
                        if let Some(t0) = tv0 {
                            self.compute_stream.synchronize()?;
                            t_vram += t0.elapsed().as_secs_f64();
                        }

                        if use_layer_ahead {
                            // Milestone 6 Layer-Ahead Path:
                            // Background `m3tis-nvme-ahead` thread is ALREADY streaming Layers 3..44 at 6.8 GB/s!
                            // 1. Stream all Tier-2 RAM hits via double-buffered 23.5 GB/s PCIe HtoD
                            let n_ram = pf.ram_list.len();
                            let mut r_idx = 0usize;
                            let mut r_wave = 0usize;
                            let tr0 = if do_prof { Some(Instant::now()) } else { None };
                            while r_idx < n_ram {
                                let r_end = (r_idx + PF_WAVE_SLOTS).min(n_ram);
                                let cnt = r_end - r_idx;
                                let buf = r_wave & 1;
                                let slot_base = buf * PF_WAVE_SLOTS;
                                for i in 0..cnt {
                                    let (le, gkey) = pf.ram_list[r_idx + i];
                                    wave_gkeys[i] = gkey;
                                    wave_pairs[i] = (le, slot_base + i);
                                }
                                self.pool.pf_stage_htod_wave(
                                    &self.compute_stream,
                                    buf,
                                    &wave_gkeys[..cnt],
                                    true,
                                    &self.scales,
                                )?;
                                Self::emit_prefill_waves(
                                    &self.exl3,
                                    &self.compute_stream,
                                    &stage_view,
                                    pf,
                                    pf.normed.ptr(),
                                    pf.branch.ptr(),
                                    d,
                                    &wave_pairs[..cnt],
                                )?;
                                self.pool.pf_mark_wave_compute_done(&self.compute_stream, buf)?;
                                r_idx = r_end;
                                r_wave += 1;
                            }
                            if let Some(t0) = tr0 {
                                t_ram += t0.elapsed().as_secs_f64();
                            }

                            // 2. Stream Tier-3 cold experts straight from LayerAheadStreamer pinned RAM ring (0 NVMe wait when ahead!)
                            let mut wave_ahead = [(0usize, 0usize); PF_WAVE_SLOTS];
                            let mut buf_release = [0usize; 2];
                            for w in 0..n_nvme_waves {
                                let cur_buf = w & 1;
                                let w_start = w * PF_WAVE_SLOTS;
                                let w_end = (w_start + PF_WAVE_SLOTS).min(n_nvme);
                                let cnt = w_end - w_start;
                                let slot_base = cur_buf * PF_WAVE_SLOTS;
                                let mut max_pidx = 0usize;
                                for i in 0..cnt {
                                    let (_, le, gkey) = pf.nvme_list[w_start + i];
                                    let pidx = pf.ahead_key_to_pidx[gkey];
                                    debug_assert!(pidx >= 0, "cold expert missing from layer-ahead plan");
                                    let pu = pidx as usize;
                                    if pu > max_pidx {
                                        max_pidx = pu;
                                    }
                                    wave_ahead[i] = (gkey, pu);
                                    wave_pairs[i] = (le, slot_base + i);
                                }
                                let tio0 = if do_prof { Some(Instant::now()) } else { None };
                                self.pool.ahead.as_ref().unwrap().wait_ready(max_pidx)?;
                                if let Some(t0) = tio0 {
                                    t_nvme_io += t0.elapsed().as_secs_f64();
                                }

                                let tg0 = if do_prof { Some(Instant::now()) } else { None };
                                let rel = if buf_release[cur_buf] > 0 {
                                    Some(buf_release[cur_buf])
                                } else {
                                    None
                                };
                                self.pool.pf_stage_htod_from_ahead(
                                    &self.compute_stream,
                                    cur_buf,
                                    &wave_ahead[..cnt],
                                    &self.scales,
                                    rel,
                                )?;
                                buf_release[cur_buf] = max_pidx + 1;
                                Self::emit_prefill_waves(
                                    &self.exl3,
                                    &self.compute_stream,
                                    &stage_view,
                                    pf,
                                    pf.normed.ptr(),
                                    pf.branch.ptr(),
                                    d,
                                    &wave_pairs[..cnt],
                                )?;
                                self.pool.pf_mark_wave_compute_done(&self.compute_stream, cur_buf)?;
                                if let Some(t0) = tg0 {
                                    t_nvme_gpu += t0.elapsed().as_secs_f64();
                                }
                            }
                            // Release any remaining ring slots of this layer immediately for Layer l+2!
                            self.pool.pf_sync_copies_and_release_ahead(pf.ahead_layer_end[moe_idx])?;
                        } else {
                            // On-demand path (short prompts < 384 rows): interleave Tier-2 RAM behind NVMe waves!
                            let n_ram = pf.ram_list.len();
                            let mut r_idx = 0usize;
                            let mut r_wave = 0usize;
                            let mut run_ram_waves = |pool: &mut TieredExpertPool,
                                                     scales: &Exl3ScaleBank,
                                                     pf_ref: &mut PfState,
                                                     max_waves: usize,
                                                     t_ram_acc: &mut f64|
                             -> Result<(), Box<dyn Error>> {
                                let mut rg = [0usize; PF_WAVE_SLOTS];
                                let mut rp = [(0usize, 0usize); PF_WAVE_SLOTS];
                                let mut done = 0usize;
                                while r_idx < n_ram && done < max_waves {
                                    let trw = if do_prof { Some(Instant::now()) } else { None };
                                    let r_end = (r_idx + PF_WAVE_SLOTS).min(n_ram);
                                    let cnt = r_end - r_idx;
                                    let buf = r_wave & 1;
                                    let slot_base = buf * PF_WAVE_SLOTS;
                                    for i in 0..cnt {
                                        let (le, gkey) = pf_ref.ram_list[r_idx + i];
                                        rg[i] = gkey;
                                        rp[i] = (le, slot_base + i);
                                    }
                                    pool.pf_stage_htod_wave(
                                        &self.compute_stream,
                                        buf,
                                        &rg[..cnt],
                                        true,
                                        scales,
                                    )?;
                                    Self::emit_prefill_waves(
                                        &self.exl3,
                                        &self.compute_stream,
                                        &stage_view,
                                        pf_ref,
                                        pf_ref.normed.ptr(),
                                        pf_ref.branch.ptr(),
                                        d,
                                        &rp[..cnt],
                                    )?;
                                    pool.pf_mark_wave_compute_done(&self.compute_stream, buf)?;
                                    if let Some(t0) = trw {
                                        *t_ram_acc += t0.elapsed().as_secs_f64();
                                    }
                                    r_idx = r_end;
                                    r_wave += 1;
                                    done += 1;
                                }
                                Ok(())
                            };

                            if n_nvme_waves > 0 {
                                run_ram_waves(&mut self.pool, &self.scales, pf, 2, &mut t_ram)?;
                            }

                            for w in 0..n_nvme_waves {
                                let cur_buf = w & 1;
                                let next_buf = 1 - cur_buf;
                                let w_start = w * PF_WAVE_SLOTS;
                                let w_end = (w_start + PF_WAVE_SLOTS).min(n_nvme);
                                let cnt = w_end - w_start;

                                let tio0 = if do_prof { Some(Instant::now()) } else { None };
                                self.pool.pf_reap_nvme_wave()?;

                                if w + 1 < n_nvme_waves {
                                    let nw_start = (w + 1) * PF_WAVE_SLOTS;
                                    let nw_end = (nw_start + PF_WAVE_SLOTS).min(n_nvme);
                                    let ncnt = nw_end - nw_start;
                                    for i in 0..ncnt {
                                        wave_gkeys[i] = pf.nvme_list[nw_start + i].2;
                                    }
                                    self.pool.pf_submit_nvme_wave(next_buf, &wave_gkeys[..ncnt])?;
                                }
                                if let Some(t0) = tio0 {
                                    t_nvme_io += t0.elapsed().as_secs_f64();
                                }

                                let tg0 = if do_prof { Some(Instant::now()) } else { None };
                                let slot_base = cur_buf * PF_WAVE_SLOTS;
                                for i in 0..cnt {
                                    let (_, le, gkey) = pf.nvme_list[w_start + i];
                                    wave_gkeys[i] = gkey;
                                    wave_pairs[i] = (le, slot_base + i);
                                }
                                self.pool.pf_stage_htod_wave(
                                    &self.compute_stream,
                                    cur_buf,
                                    &wave_gkeys[..cnt],
                                    false,
                                    &self.scales,
                                )?;
                                Self::emit_prefill_waves(
                                    &self.exl3,
                                    &self.compute_stream,
                                    &stage_view,
                                    pf,
                                    pf.normed.ptr(),
                                    pf.branch.ptr(),
                                    d,
                                    &wave_pairs[..cnt],
                                )?;
                                self.pool.pf_mark_wave_compute_done(&self.compute_stream, cur_buf)?;
                                if let Some(t0) = tg0 {
                                    t_nvme_gpu += t0.elapsed().as_secs_f64();
                                }

                                if w + 1 < n_nvme_waves {
                                    run_ram_waves(&mut self.pool, &self.scales, pf, 2, &mut t_ram)?;
                                }
                            }

                            run_ram_waves(&mut self.pool, &self.scales, pf, usize::MAX, &mut t_ram)?;
                        }

                        // B5. Apply mHC post update over pf.x + accumulated pf.branch
                        let tp5 = if do_prof { Some(Instant::now()) } else { None };
                        self.glm.hc_post(
                            &self.compute_stream,
                            pf.x.ptr(),
                            pf.branch.ptr(),
                            pf.post.ptr(),
                            pf.comb.ptr(),
                            rows,
                            d,
                        )?;
                        if let Some(t0) = tp5 {
                            self.compute_stream.synchronize()?;
                            t_post += t0.elapsed().as_secs_f64();
                        }
                    }
                }
            }

            // Capture DFlash2 target taps (mean of 4 streams -> pinned host RAM, 0 extra GPU VRAM!)
            if capture_taps {
                if let Some(slot) = tap_slot_for_layer(l) {
                    if let Some(ref mut tap_layers) = pf.h_taps_layers {
                        self.glm.stream_mean_bf16(
                            &self.compute_stream,
                            pf.x.ptr(),
                            pf.normed.ptr(),
                            rows,
                            d,
                            s,
                        )?;
                        unsafe {
                            self.compute_stream.dtoh_async(
                                tap_layers[slot as usize].as_mut_ptr(),
                                pf.normed.ptr(),
                                rows * row_bytes,
                            )?;
                        }
                    }
                }
            }
        }

        st.pos += rows;
        self.pool.pf_end()?;

        // 3. Ingest DFlash2 target taps in micro-batches (100% bit-exact KV rings!)
        if capture_taps {
            self.compute_stream.synchronize()?;
            if let (Some(dr), Some(ref tap_layers), Some(ref mut tap_micro)) = (
                self.drafter.as_ref(),
                pf.h_taps_layers.as_ref(),
                pf.h_taps_micro.as_mut(),
            ) {
                let mut r0 = 0usize;
                while r0 < rows {
                    let m = (rows - r0).min(micro);
                    let dst = tap_micro.as_mut_slice();
                    for r in 0..m {
                        for sl in 0..5 {
                            let src = &tap_layers[sl].as_slice()[(r0 + r) * row_bytes..(r0 + r + 1) * row_bytes];
                            let d_off = (r * 5 + sl) * row_bytes;
                            dst[d_off..d_off + row_bytes].copy_from_slice(src);
                        }
                    }
                    unsafe {
                        self.compute_stream.htod_async(
                            st.taps_buf.ptr(),
                            tap_micro.as_ptr(),
                            m * 5 * row_bytes,
                        )?;
                    }
                    dr.ingest(
                        &self.compute_stream,
                        &self.lin,
                        &self.glm,
                        st.taps_buf.ptr(),
                        pos0 + r0,
                        m,
                    )?;
                    self.compute_stream.synchronize()?;
                    r0 += m;
                }
            }
        }

        // 4. Final logits on the LAST token row (if last super-batch)
        if compute_logits {
            let last_x = pf.x.at((rows - 1) * s * d * 2);
            self.glm.stream_mean_bf16(
                &self.compute_stream,
                last_x,
                st.final_hidden.ptr(),
                1,
                d,
                s,
            )?;
            self.glm.rmsnorm_bf16(
                &self.compute_stream,
                st.final_hidden.ptr(),
                d,
                self.final_norm.ptr(),
                st.final_normed.ptr(),
                d,
                1,
                d,
                self.cfg.rms_eps,
            )?;
            self.lm_head.apply(
                &self.lin,
                &self.compute_stream,
                st.final_normed.ptr(),
                d,
                st.d_logits.ptr(),
                self.cfg.vocab,
                1,
                true,
            )?;
            unsafe {
                self.compute_stream.dtoh_async(
                    st.h_logits.as_mut_ptr(),
                    st.d_logits.ptr(),
                    self.cfg.vocab * 4,
                )?;
            }
        }
        self.compute_stream.synchronize()?;
        if do_prof {
            eprintln!(
                "  [pf-profile rows={rows}] attn={t_attn:.2}s b1={t_b1:.2}s vram={t_vram:.2}s({n_vram_tot}) ram={t_ram:.2}s({n_ram_tot}) nvme_io={t_nvme_io:.2}s nvme_gpu={t_nvme_gpu:.2}s({n_nvme_tot}) post={t_post:.2}s"
            );
        }
        if do_detail {
            eprintln!(
                "  [pf-detail rows={rows}] hc={td_hc:.2}s kda_q4={td_kda_q4:.2}s kda_bf16={td_kda_bf16:.2}s kda_chain={td_kda_chain:.2}s mla_q4={td_mla_q4:.2}s mla_absexp={td_mla_absexp:.2}s mla_dsa={td_mla_dsa:.2}s mla_att={td_mla_att:.2}s"
            );
        }
        Ok(())
    }
    /// Tallies 42-layer hit stats, fires non-blocking background RAM->VRAM & NVMe->RAM warming
    /// (hidden behind DFlash2 + Dense Layers 0..2), and triggers 1 sync heal step if cold domain
    /// coverage drops below `opts.zero_stall_mass` on >= 6 layers.
    fn post_warp_zero_sync(
        &mut self,
        st: &mut Glm53State,
        rows: usize,
        is_verify: bool,
        opts: ForwardOpts,
    ) -> Result<(), Box<dyn Error>> {
        let moe_n = self.cfg.moe_layers();
        let e_cnt = self.cfg.num_experts;
        let slots = self.cfg.top_k + 1;
        let hp = &st.h_all_picks.as_i32_mut()[..moe_n * rows * slots];
        let hw = &st.h_all_wts.as_f32_mut()[..moe_n * rows * slots];

        st.warp_missing.clear();
        let mut low_cov_layers = 0usize;
        for ml in 0..moe_n {
            let base_key = ml * e_cnt;
            let l_off = ml * rows * slots;
            let mut layer_low = false;
            st.active_keys.clear();
            for r in 0..rows {
                let mut total_w = 0.0f32;
                let mut cov_w = 0.0f32;
                for k_rank in 0..self.cfg.top_k {
                    let p = hp[l_off + r * slots + k_rank];
                    if p >= 0 && (p as usize) < e_cnt {
                        let gk = base_key + (p as usize);
                        let w = hw[l_off + r * slots + k_rank].max(0.0);
                        self.pool.expert_hits[gk] += 1;
                        total_w += w;
                        let in_vram = self.pool.vram_cache.peek(gk).is_some();
                        if in_vram {
                            cov_w += w;
                        } else {
                            if let Some(entry) = st.warp_missing.iter_mut().find(|x| x.0 == gk) {
                                entry.1 += w;
                            } else {
                                st.warp_missing.push((gk, w));
                            }
                        }
                        if !st.active_keys.contains(&gk) {
                            st.active_keys.push(gk);
                            if in_vram {
                                self.pool.vram_cache.touch_hit(gk);
                                self.pool.stats.vram_hits += 1;
                            } else {
                                self.pool.stats.vram_misses += 1;
                                self.pool.stats.nvme_tail_skipped += 1;
                            }
                        }
                    }
                }
                if total_w > 0.0 && cov_w < opts.zero_stall_mass * total_w {
                    layer_low = true;
                }
            }
            if layer_low {
                low_cov_layers += 1;
            }
        }

        if low_cov_layers >= 6 && !st.warp_missing.is_empty() {
            st.warp_heal_steps = 1;
        }
        if !st.warp_missing.is_empty() {
            st.warp_missing.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            st.spec_keys.clear();
            for &(gk, _) in st.warp_missing.iter().take(48) {
                st.spec_keys.push(gk);
            }
            let max_ram_prom = if is_verify { 14 } else { 6 };
            let _ = self
                .pool
                .promote_async_background(&st.spec_keys, max_ram_prom, 8, &mut self.scales)?;
        }
        Ok(())
    }
}
