//! Clean-room Rust host driver for M3TiS `sm_86` EXL3 MCG Grouped MoE kernels.
//! Supports arbitrary per-expert mixed bitwidth (2b/3b/4b..8b in a single launch),
//! Ampere register-pipelined tile-ahead loads, FWHT128 rotations, and GLM-5.3 clamped SwiGLU.

use crate::{CuDevicePtr, CudaContext, CudaFunction, CudaModule, DeviceBuf, Event, PinnedHostBuf, Result, Stream};
use std::ffi::c_void;
use std::sync::Arc;

const EXL3_MOE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/exl3_moe.cubin"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputDType {
    F16 = 0,
    Bf16 = 1,
    F32 = 2,
}

/// Device-resident view of one layer's (or FSR VRAM slab's) EXL3 experts.
#[derive(Debug, Clone, Copy)]
pub struct Exl3ExpertsView {
    /// `[experts]` u64 device pointers to gate/up/down trellises.
    pub gt_ptrs: CuDevicePtr,
    pub ut_ptrs: CuDevicePtr,
    pub dt_ptrs: CuDevicePtr,
    /// `[experts]` i32 half-bit widths (`k2 = 2 * bits`: 4=2b, 6=3b, 8=4b).
    pub k2_g: CuDevicePtr,
    pub k2_u: CuDevicePtr,
    pub k2_d: CuDevicePtr,
    /// `[experts, K]` / `[experts, N]` fp16 Hadamard input/output scales.
    pub suh_g: CuDevicePtr,
    pub suh_u: CuDevicePtr,
    pub svh_g: CuDevicePtr,
    pub svh_u: CuDevicePtr,
    pub suh_d: CuDevicePtr,
    pub svh_d: CuDevicePtr,
    /// Number of addressable expert slots (e.g. `vram_slots` for FSR pool, or `E` for resident).
    pub experts: usize,
    pub dims: usize,
    pub inter: usize,
    /// GLM-5.3 SwiGLU clamp (`10.0`, or `0.0` for unclamped).
    pub swiglu_limit: f32,
    /// `0` = bf16-rounded GLM SwiGLU, `1` = fp32 SwiGLU.
    pub act_mode: i32,
}

/// Reusable GPU scratch buffers for `Exl3MoeKernels::run_moe`.
pub struct Exl3MoeScratch {
    pub max_rows: usize,
    pub max_slots: usize,
    pub max_sk: usize,
    pub uids: DeviceBuf,    // [max_p] i32
    pub ucount: DeviceBuf,  // [1] i32
    pub members: DeviceBuf, // [max_p * max_rows] i32
    pub xg: DeviceBuf,      // [max_p * dims] f16
    pub xu: DeviceBuf,      // [max_p * dims] f16
    pub xd: DeviceBuf,      // [max_p * inter] f16
    pub z: DeviceBuf,       // [max(2 * max_sk * max_p * inter, max_sk * max_p * dims)] f32
    pub slot_y: DeviceBuf,  // [max_p * dims] f32
}

impl Exl3MoeScratch {
    pub fn alloc(
        ctx: &Arc<CudaContext>,
        max_rows: usize,
        max_slots: usize,
        dims: usize,
        inter: usize,
        max_sk: usize,
    ) -> Result<Self> {
        let max_p = max_rows * max_slots;
        let max_p_split = max_rows.min(8) * max_slots;
        let z_gu = (2 * max_sk * max_p_split * inter).max(2 * max_p * inter);
        let z_d = (max_sk * max_p_split * dims).max(max_p * dims);
        let z_elems = z_gu.max(z_d).max(1);
        Ok(Self {
            max_rows,
            max_slots,
            max_sk,
            uids: ctx.alloc_device(max_p * 4)?,
            ucount: ctx.alloc_device(4)?,
            members: ctx.alloc_device(max_p * max_rows * 4)?,
            xg: ctx.alloc_device(max_p * dims * 2)?,
            xu: ctx.alloc_device(max_p * dims * 2)?,
            xd: ctx.alloc_device(max_p * inter * 2)?,
            z: ctx.alloc_device(z_elems * 4)?,
            slot_y: ctx.alloc_device(max_p * dims * 4)?,
        })
    }
}

pub struct Exl3PrefillScratch {
    pub max_pairs: usize,
    pub max_tiles: usize,
    pub xg: DeviceBuf,           // [max_pairs * dims] f16
    pub xu: DeviceBuf,           // [max_pairs * dims] f16
    pub xd: DeviceBuf,           // [max_pairs * inter] f16
    pub z: DeviceBuf,            // [max(2 * max_pairs * inter, max_pairs * dims)] f32
    pub d_pair_row: DeviceBuf,   // [max_pairs] i32
    pub d_pair_slot: DeviceBuf,  // [max_pairs] i32
    pub d_pair_wt: DeviceBuf,    // [max_pairs] f32
    pub d_tile_slot: DeviceBuf,  // [max_tiles] i32
    pub d_tile_off: DeviceBuf,   // [max_tiles] i32
    pub d_tile_cnt: DeviceBuf,   // [max_tiles] i32
    pub h_pair_row: [PinnedHostBuf; 2],
    pub h_pair_slot: [PinnedHostBuf; 2],
    pub h_pair_wt: [PinnedHostBuf; 2],
    pub h_tile_slot: [PinnedHostBuf; 2],
    pub h_tile_off: [PinnedHostBuf; 2],
    pub h_tile_cnt: [PinnedHostBuf; 2],
    pub meta_done: [Event; 2],
    pub meta_armed: [bool; 2],
    pub vram_bytes: usize,
}

impl Exl3PrefillScratch {
    pub fn alloc(
        ctx: &Arc<CudaContext>,
        max_pairs: usize,
        max_tiles: usize,
        dims: usize,
        inter: usize,
    ) -> Result<Self> {
        let z_elems = (2 * max_pairs * inter).max(max_pairs * dims).max(1);
        let xg = ctx.alloc_device(max_pairs * dims * 2)?;
        let xu = ctx.alloc_device(max_pairs * dims * 2)?;
        let xd = ctx.alloc_device(max_pairs * inter * 2)?;
        let z = ctx.alloc_device(z_elems * 4)?;
        let d_pair_row = ctx.alloc_device(max_pairs * 4)?;
        let d_pair_slot = ctx.alloc_device(max_pairs * 4)?;
        let d_pair_wt = ctx.alloc_device(max_pairs * 4)?;
        let d_tile_slot = ctx.alloc_device(max_tiles * 4)?;
        let d_tile_off = ctx.alloc_device(max_tiles * 4)?;
        let d_tile_cnt = ctx.alloc_device(max_tiles * 4)?;
        let vram_bytes = xg.len()
            + xu.len()
            + xd.len()
            + z.len()
            + d_pair_row.len()
            + d_pair_slot.len()
            + d_pair_wt.len()
            + d_tile_slot.len()
            + d_tile_off.len()
            + d_tile_cnt.len();
        Ok(Self {
            max_pairs,
            max_tiles,
            xg,
            xu,
            xd,
            z,
            d_pair_row,
            d_pair_slot,
            d_pair_wt,
            d_tile_slot,
            d_tile_off,
            d_tile_cnt,
            h_pair_row: [ctx.alloc_pinned(max_pairs * 4)?, ctx.alloc_pinned(max_pairs * 4)?],
            h_pair_slot: [ctx.alloc_pinned(max_pairs * 4)?, ctx.alloc_pinned(max_pairs * 4)?],
            h_pair_wt: [ctx.alloc_pinned(max_pairs * 4)?, ctx.alloc_pinned(max_pairs * 4)?],
            h_tile_slot: [ctx.alloc_pinned(max_tiles * 4)?, ctx.alloc_pinned(max_tiles * 4)?],
            h_tile_off: [ctx.alloc_pinned(max_tiles * 4)?, ctx.alloc_pinned(max_tiles * 4)?],
            h_tile_cnt: [ctx.alloc_pinned(max_tiles * 4)?, ctx.alloc_pinned(max_tiles * 4)?],
            meta_done: [ctx.event()?, ctx.event()?],
            meta_armed: [false, false],
            vram_bytes,
        })
    }
}

pub struct Exl3MoeKernels {
    _module: CudaModule,
    group: CudaFunction,
    rot_in: CudaFunction,
    grouped_ahead_m1: CudaFunction,
    grouped_ahead_m8: CudaFunction,
    grouped_ahead: CudaFunction,
    grouped_plain: CudaFunction,
    gateup_epilogue: CudaFunction,
    down_combine: CudaFunction,
    pf_rot_in: CudaFunction,
    pf_gemm: CudaFunction,
    pf_down_scatter: CudaFunction,
}

impl Exl3MoeKernels {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = ctx.load_module(EXL3_MOE_CUBIN)?;
        Ok(Self {
            group: module.function("m3tis_exl3_group")?,
            rot_in: module.function("m3tis_exl3_rot_in")?,
            grouped_ahead_m1: module.function("m3tis_exl3_grouped_mcg_ahead_m1")?,
            grouped_ahead_m8: module.function("m3tis_exl3_grouped_mcg_ahead_m8")?,
            grouped_ahead: module.function("m3tis_exl3_grouped_mcg_ahead")?,
            grouped_plain: module.function("m3tis_exl3_grouped_mcg_plain")?,
            gateup_epilogue: module.function("m3tis_exl3_gateup_epilogue")?,
            down_combine: module.function("m3tis_exl3_down_combine")?,
            pf_rot_in: module.function("m3tis_exl3_pf_rot_in")?,
            pf_gemm: module.function("m3tis_exl3_pf_gemm")?,
            pf_down_scatter: module.function("m3tis_exl3_pf_down_scatter")?,
            _module: module,
        })
    }

    /// Choose split-K factor so `(k / 16) % (sk * 4) == 0` while keeping strong SM occupancy on RTX 3090 (82 SMs).
    #[inline]
    fn pick_sk(k: usize, rows: usize, max_sk: usize) -> usize {
        let k_groups = (k / 16) / 4; // 4 warps per block
        // Keep identical split-K across decode (rows=1) and DFlash2 verify (rows<=8) for bit-exact parity!
        let target = if rows <= 8 { 4 } else { 1 };
        let mut sk = target.min(max_sk).min(k_groups).max(1);
        while sk > 1 && (k_groups % sk) != 0 {
            sk -= 1;
        }
        sk
    }

    /// Execute full routed EXL3 MoE layer on `stream`:
    /// `group -> rot_in -> grouped_mcg(gate+up) -> gateup_epilogue -> grouped_mcg(down) -> down_combine`.
    ///
    /// - `mapped_picks`: `[rows, slots]` i32 VRAM slot indices (`0..experts`), with skipped/shared slots `>= experts`.
    /// - `wts`: optional `[rows, slots]` f32 weights (if `0`, skips final combine and leaves per-slot outputs in `scratch.slot_y`).
    /// - `out`: `[rows, dims]` f32 weighted sum output (when `wts != 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_moe(
        &self,
        stream: &Stream,
        v: &Exl3ExpertsView,
        sc: &Exl3MoeScratch,
        x_ptr: CuDevicePtr,
        x_stride: usize,
        x_dtype: InputDType,
        mapped_picks: CuDevicePtr,
        wts_ptr: CuDevicePtr,
        out_ptr: CuDevicePtr,
        rows: usize,
        slots: usize,
        use_ahead: bool,
    ) -> Result<()> {
        self.run_moe_ext(
            stream,
            v,
            sc,
            x_ptr,
            x_stride,
            x_dtype,
            mapped_picks,
            wts_ptr,
            out_ptr,
            rows,
            slots,
            use_ahead,
            rows * slots,
            false,
        )
    }

    /// Extended routed EXL3 MoE launcher with tight `max_unique` grid sizing (cuts Wave 2 CTA grid 18x-36x!)
    /// and optional in-place `accum_out` (`out += acc`, eliminating separate `add_f32_inplace` launch!).
    #[allow(clippy::too_many_arguments)]
    pub fn run_moe_ext(
        &self,
        stream: &Stream,
        v: &Exl3ExpertsView,
        sc: &Exl3MoeScratch,
        x_ptr: CuDevicePtr,
        x_stride: usize,
        x_dtype: InputDType,
        mapped_picks: CuDevicePtr,
        wts_ptr: CuDevicePtr,
        out_ptr: CuDevicePtr,
        rows: usize,
        slots: usize,
        use_ahead: bool,
        max_unique: usize,
        accum_out: bool,
    ) -> Result<()> {
        assert!(rows > 0 && rows <= sc.max_rows);
        assert!(slots > 0 && slots <= sc.max_slots);
        assert_eq!(v.dims % 128, 0);
        assert_eq!(v.inter % 128, 0);
        assert!(v.experts <= 4096);

        let p = rows * slots;
        let nexp_max = max_unique.clamp(1, p).min(v.experts);
        let maxm = rows;
        let mt = (maxm + 15) / 16;

        // 1. Group active VRAM slots into unique expert lists (1 block, 1024 threads)
        {
            let mut pick_a = mapped_picks;
            let mut uids_a = sc.uids.ptr();
            let mut ucnt_a = sc.ucount.ptr();
            let mut mem_a = sc.members.ptr();
            let mut rows_i = rows as i32;
            let mut slots_i = slots as i32;
            let mut exp_i = v.experts as i32;
            let mut maxm_i = maxm as i32;
            let mut args: [*mut c_void; 8] = [
                &mut pick_a as *mut _ as *mut c_void,
                &mut uids_a as *mut _ as *mut c_void,
                &mut ucnt_a as *mut _ as *mut c_void,
                &mut mem_a as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut slots_i as *mut _ as *mut c_void,
                &mut exp_i as *mut _ as *mut c_void,
                &mut maxm_i as *mut _ as *mut c_void,
            ];
            let shmem = (p * 4) as u32;
            unsafe { self.group.launch((1, 1, 1), (1024, 1, 1), shmem, stream, &mut args)? };
        }

        // 2. Hadamard input rotation for gate (z=0) and up (z=1)
        {
            let mut xa = x_ptr;
            let mut xs = x_stride as i32;
            let mut xd = x_dtype as i32;
            let mut pa = mapped_picks;
            let mut su0 = v.suh_g;
            let mut su1 = v.suh_u;
            let mut o0 = sc.xg.ptr();
            let mut o1 = sc.xu.ptr();
            let mut k_i = v.dims as i32;
            let mut slots_i = slots as i32;
            let mut exp_i = v.experts as i32;
            let mut args: [*mut c_void; 11] = [
                &mut xa as *mut _ as *mut c_void,
                &mut xs as *mut _ as *mut c_void,
                &mut xd as *mut _ as *mut c_void,
                &mut pa as *mut _ as *mut c_void,
                &mut su0 as *mut _ as *mut c_void,
                &mut su1 as *mut _ as *mut c_void,
                &mut o0 as *mut _ as *mut c_void,
                &mut o1 as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
                &mut slots_i as *mut _ as *mut c_void,
                &mut exp_i as *mut _ as *mut c_void,
            ];
            unsafe {
                self.rot_in.launch(
                    (p as u32, (v.dims / 128) as u32, 2),
                    (32, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }

        let (grouped_fn, cols_per_blk) = if !use_ahead {
            (&self.grouped_plain, 128usize)
        } else if rows == 1 {
            (&self.grouped_ahead_m1, 64usize)
        } else if rows <= 8 {
            (&self.grouped_ahead_m8, 64usize)
        } else {
            (&self.grouped_ahead, 128usize)
        };

        // 3. Grouped EXL3 MCG GEMM for Gate + Up (mats = 2)
        let sk_gu = Self::pick_sk(v.dims, rows, sc.max_sk);
        assert!(2 * sk_gu * p * v.inter <= sc.z.len() / 4, "Exl3MoeScratch.z overflow (gate+up)");
        {
            let mut x0 = sc.xg.ptr();
            let mut x1 = sc.xu.ptr();
            let mut tp0 = v.gt_ptrs;
            let mut tp1 = v.ut_ptrs;
            let mut k20 = v.k2_g;
            let mut k21 = v.k2_u;
            let mut uids = sc.uids.ptr();
            let mut ucnt = sc.ucount.ptr();
            let mut mem = sc.members.ptr();
            let mut z = sc.z.ptr();
            let mut k_i = v.dims as i32;
            let mut n_i = v.inter as i32;
            let mut p_i = p as i32;
            let mut sk_i = sk_gu as i32;
            let mut maxm_i = maxm as i32;
            let mut slots_i = slots as i32;
            let mut args: [*mut c_void; 16] = [
                &mut x0 as *mut _ as *mut c_void,
                &mut x1 as *mut _ as *mut c_void,
                &mut tp0 as *mut _ as *mut c_void,
                &mut tp1 as *mut _ as *mut c_void,
                &mut k20 as *mut _ as *mut c_void,
                &mut k21 as *mut _ as *mut c_void,
                &mut uids as *mut _ as *mut c_void,
                &mut ucnt as *mut _ as *mut c_void,
                &mut mem as *mut _ as *mut c_void,
                &mut z as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut sk_i as *mut _ as *mut c_void,
                &mut maxm_i as *mut _ as *mut c_void,
                &mut slots_i as *mut _ as *mut c_void,
            ];
            let nblocks = (v.inter / cols_per_blk) as u32;
            unsafe {
                grouped_fn.launch(
                    (nexp_max as u32, nblocks, (2 * sk_gu * mt) as u32),
                    (128, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }

        // 4. Fused Gate+Up Epilogue: split-K sum -> FWHT128 -> *svh -> SwiGLU -> *suh_d -> FWHT128 -> Xd
        {
            let mut z = sc.z.ptr();
            let mut pa = mapped_picks;
            let mut svg = v.svh_g;
            let mut svu = v.svh_u;
            let mut sud = v.suh_d;
            let mut xd = sc.xd.ptr();
            let mut p_i = p as i32;
            let mut n_i = v.inter as i32;
            let mut sk_i = sk_gu as i32;
            let mut exp_i = v.experts as i32;
            let mut lim = v.swiglu_limit;
            let mut am = v.act_mode;
            let mut args: [*mut c_void; 12] = [
                &mut z as *mut _ as *mut c_void,
                &mut pa as *mut _ as *mut c_void,
                &mut svg as *mut _ as *mut c_void,
                &mut svu as *mut _ as *mut c_void,
                &mut sud as *mut _ as *mut c_void,
                &mut xd as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut sk_i as *mut _ as *mut c_void,
                &mut exp_i as *mut _ as *mut c_void,
                &mut lim as *mut _ as *mut c_void,
                &mut am as *mut _ as *mut c_void,
            ];
            unsafe {
                self.gateup_epilogue.launch(
                    (p as u32, (v.inter / 128) as u32, 1),
                    (32, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }

        // 5. Grouped EXL3 MCG GEMM for Down (mats = 1)
        let sk_d = Self::pick_sk(v.inter, rows, sc.max_sk);
        assert!(sk_d * p * v.dims <= sc.z.len() / 4, "Exl3MoeScratch.z overflow (down)");
        {
            let mut x0 = sc.xd.ptr();
            let mut x1 = sc.xd.ptr();
            let mut tp0 = v.dt_ptrs;
            let mut tp1 = v.dt_ptrs;
            let mut k20 = v.k2_d;
            let mut k21 = v.k2_d;
            let mut uids = sc.uids.ptr();
            let mut ucnt = sc.ucount.ptr();
            let mut mem = sc.members.ptr();
            let mut z = sc.z.ptr();
            let mut k_i = v.inter as i32;
            let mut n_i = v.dims as i32;
            let mut p_i = p as i32;
            let mut sk_i = sk_d as i32;
            let mut maxm_i = maxm as i32;
            let mut slots_i = slots as i32;
            let mut args: [*mut c_void; 16] = [
                &mut x0 as *mut _ as *mut c_void,
                &mut x1 as *mut _ as *mut c_void,
                &mut tp0 as *mut _ as *mut c_void,
                &mut tp1 as *mut _ as *mut c_void,
                &mut k20 as *mut _ as *mut c_void,
                &mut k21 as *mut _ as *mut c_void,
                &mut uids as *mut _ as *mut c_void,
                &mut ucnt as *mut _ as *mut c_void,
                &mut mem as *mut _ as *mut c_void,
                &mut z as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut sk_i as *mut _ as *mut c_void,
                &mut maxm_i as *mut _ as *mut c_void,
                &mut slots_i as *mut _ as *mut c_void,
            ];
            let nblocks = (v.dims / cols_per_blk) as u32;
            unsafe {
                grouped_fn.launch(
                    (nexp_max as u32, nblocks, (sk_d * mt) as u32),
                    (128, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }

        // 6. Fused Down Epilogue + Weighted Slot Combine into `out[rows, dims]`
        {
            let mut z = sc.z.ptr();
            let mut pa = mapped_picks;
            let mut svd = v.svh_d;
            let mut sy = sc.slot_y.ptr();
            let mut wa = wts_ptr;
            let mut oa = out_ptr;
            let mut p_i = p as i32;
            let mut d_i = v.dims as i32;
            let mut sk_i = sk_d as i32;
            let mut exp_i = v.experts as i32;
            let mut slots_i = slots as i32;
            let mut acc_i: i32 = if accum_out { 1 } else { 0 };
            let mut args: [*mut c_void; 12] = [
                &mut z as *mut _ as *mut c_void,
                &mut pa as *mut _ as *mut c_void,
                &mut svd as *mut _ as *mut c_void,
                &mut sy as *mut _ as *mut c_void,
                &mut wa as *mut _ as *mut c_void,
                &mut oa as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut d_i as *mut _ as *mut c_void,
                &mut sk_i as *mut _ as *mut c_void,
                &mut exp_i as *mut _ as *mut c_void,
                &mut slots_i as *mut _ as *mut c_void,
                &mut acc_i as *mut _ as *mut c_void,
            ];
            unsafe {
                self.down_combine.launch(
                    (rows as u32, (v.dims / 128) as u32, 1),
                    ((slots * 32) as u32, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }
        Ok(())
    }

    /// Execute one Expert-Major flat-tile streaming prefill wave on `stream` and scatter-add
    /// (`atomicAdd`) weighted outputs directly into `out_ptr[B, dims]` (fp32).
    ///
    /// - `pair_row[p]`: token index `0..B-1` into `x_ptr` (bf16) and `out_ptr` (fp32).
    /// - `pair_slot[p]`: VRAM slot `0..v.experts-1` inside `v` (either Tier-1 VRAM slab or double-buffered stage ring).
    /// - `pair_wt[p]`: router weight for `(pair_row[p], pair_slot[p])`.
    /// - `tile_slot[t], tile_off[t], tile_cnt[t]`: flat 16-row Tensor Core tiles (`1 <= tile_cnt[t] <= 16`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_wave(
        &self,
        stream: &Stream,
        v: &Exl3ExpertsView,
        sc: &mut Exl3PrefillScratch,
        buf_idx: usize,
        x_ptr: CuDevicePtr,
        x_stride: usize,
        out_ptr: CuDevicePtr,
        pair_row: &[i32],
        pair_slot: &[i32],
        pair_wt: &[f32],
        tile_slot: &[i32],
        tile_off: &[i32],
        tile_cnt: &[i32],
    ) -> Result<()> {
        let p = pair_row.len();
        let n_tiles = tile_slot.len();
        if p == 0 || n_tiles == 0 {
            return Ok(());
        }
        assert!(p <= sc.max_pairs, "run_prefill_wave pairs={p} > max_pairs={}", sc.max_pairs);
        assert!(n_tiles <= sc.max_tiles, "run_prefill_wave tiles={n_tiles} > max_tiles={}", sc.max_tiles);
        assert_eq!(pair_slot.len(), p);
        assert_eq!(pair_wt.len(), p);
        assert_eq!(tile_off.len(), n_tiles);
        assert_eq!(tile_cnt.len(), n_tiles);
        assert_eq!(v.dims % 128, 0);
        assert_eq!(v.inter % 128, 0);

        let b = buf_idx & 1;
        if sc.meta_armed[b] {
            sc.meta_done[b].synchronize()?;
            sc.meta_armed[b] = false;
        }
        sc.h_pair_row[b].as_i32_mut()[..p].copy_from_slice(pair_row);
        sc.h_pair_slot[b].as_i32_mut()[..p].copy_from_slice(pair_slot);
        sc.h_pair_wt[b].as_f32_mut()[..p].copy_from_slice(pair_wt);
        sc.h_tile_slot[b].as_i32_mut()[..n_tiles].copy_from_slice(tile_slot);
        sc.h_tile_off[b].as_i32_mut()[..n_tiles].copy_from_slice(tile_off);
        sc.h_tile_cnt[b].as_i32_mut()[..n_tiles].copy_from_slice(tile_cnt);
        unsafe {
            stream.htod_async(sc.d_pair_row.ptr(), sc.h_pair_row[b].as_ptr(), p * 4)?;
            stream.htod_async(sc.d_pair_slot.ptr(), sc.h_pair_slot[b].as_ptr(), p * 4)?;
            stream.htod_async(sc.d_pair_wt.ptr(), sc.h_pair_wt[b].as_ptr(), p * 4)?;
            stream.htod_async(sc.d_tile_slot.ptr(), sc.h_tile_slot[b].as_ptr(), n_tiles * 4)?;
            stream.htod_async(sc.d_tile_off.ptr(), sc.h_tile_off[b].as_ptr(), n_tiles * 4)?;
            stream.htod_async(sc.d_tile_cnt.ptr(), sc.h_tile_cnt[b].as_ptr(), n_tiles * 4)?;
        }
        sc.meta_done[b].record(stream)?;
        sc.meta_armed[b] = true;

        // 1. Hadamard input rotation for gate (z=0) and up (z=1)
        {
            let mut xa = x_ptr;
            let mut xs = x_stride as i32;
            let mut pr = sc.d_pair_row.ptr();
            let mut ps = sc.d_pair_slot.ptr();
            let mut su0 = v.suh_g;
            let mut su1 = v.suh_u;
            let mut o0 = sc.xg.ptr();
            let mut o1 = sc.xu.ptr();
            let mut k_i = v.dims as i32;
            let mut args: [*mut c_void; 9] = [
                &mut xa as *mut _ as *mut c_void,
                &mut xs as *mut _ as *mut c_void,
                &mut pr as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut su0 as *mut _ as *mut c_void,
                &mut su1 as *mut _ as *mut c_void,
                &mut o0 as *mut _ as *mut c_void,
                &mut o1 as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
            ];
            unsafe {
                self.pf_rot_in.launch(
                    (p as u32, (v.dims / 128) as u32, 2),
                    (32, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }

        // 2. Flat-tile EXL3 MCG Tensor Core GEMM for Gate + Up (mats = 2, sk = 1)
        {
            let mut x0 = sc.xg.ptr();
            let mut x1 = sc.xu.ptr();
            let mut tp0 = v.gt_ptrs;
            let mut tp1 = v.ut_ptrs;
            let mut k20 = v.k2_g;
            let mut k21 = v.k2_u;
            let mut tslot = sc.d_tile_slot.ptr();
            let mut toff = sc.d_tile_off.ptr();
            let mut tcnt = sc.d_tile_cnt.ptr();
            let mut z = sc.z.ptr();
            let mut k_i = v.dims as i32;
            let mut n_i = v.inter as i32;
            let mut p_i = p as i32;
            let mut args: [*mut c_void; 13] = [
                &mut x0 as *mut _ as *mut c_void,
                &mut x1 as *mut _ as *mut c_void,
                &mut tp0 as *mut _ as *mut c_void,
                &mut tp1 as *mut _ as *mut c_void,
                &mut k20 as *mut _ as *mut c_void,
                &mut k21 as *mut _ as *mut c_void,
                &mut tslot as *mut _ as *mut c_void,
                &mut toff as *mut _ as *mut c_void,
                &mut tcnt as *mut _ as *mut c_void,
                &mut z as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
            ];
            unsafe {
                self.pf_gemm.launch(
                    (n_tiles as u32, (v.inter / 128) as u32, 2),
                    (128, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }

        // 3. Fused Gate+Up Epilogue (reuses exact m3tis_exl3_gateup_epilogue with pick = d_pair_slot, sk = 1!)
        {
            let mut z = sc.z.ptr();
            let mut pa = sc.d_pair_slot.ptr();
            let mut svg = v.svh_g;
            let mut svu = v.svh_u;
            let mut sud = v.suh_d;
            let mut xd = sc.xd.ptr();
            let mut p_i = p as i32;
            let mut n_i = v.inter as i32;
            let mut sk_i = 1i32;
            let mut exp_i = v.experts as i32;
            let mut lim = v.swiglu_limit;
            let mut am = v.act_mode;
            let mut args: [*mut c_void; 12] = [
                &mut z as *mut _ as *mut c_void,
                &mut pa as *mut _ as *mut c_void,
                &mut svg as *mut _ as *mut c_void,
                &mut svu as *mut _ as *mut c_void,
                &mut sud as *mut _ as *mut c_void,
                &mut xd as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut sk_i as *mut _ as *mut c_void,
                &mut exp_i as *mut _ as *mut c_void,
                &mut lim as *mut _ as *mut c_void,
                &mut am as *mut _ as *mut c_void,
            ];
            unsafe {
                self.gateup_epilogue.launch(
                    (p as u32, (v.inter / 128) as u32, 1),
                    (32, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }

        // 4. Flat-tile EXL3 MCG Tensor Core GEMM for Down (mats = 1, sk = 1)
        {
            let mut x0 = sc.xd.ptr();
            let mut x1 = sc.xd.ptr();
            let mut tp0 = v.dt_ptrs;
            let mut tp1 = v.dt_ptrs;
            let mut k20 = v.k2_d;
            let mut k21 = v.k2_d;
            let mut tslot = sc.d_tile_slot.ptr();
            let mut toff = sc.d_tile_off.ptr();
            let mut tcnt = sc.d_tile_cnt.ptr();
            let mut z = sc.z.ptr();
            let mut k_i = v.inter as i32;
            let mut n_i = v.dims as i32;
            let mut p_i = p as i32;
            let mut args: [*mut c_void; 13] = [
                &mut x0 as *mut _ as *mut c_void,
                &mut x1 as *mut _ as *mut c_void,
                &mut tp0 as *mut _ as *mut c_void,
                &mut tp1 as *mut _ as *mut c_void,
                &mut k20 as *mut _ as *mut c_void,
                &mut k21 as *mut _ as *mut c_void,
                &mut tslot as *mut _ as *mut c_void,
                &mut toff as *mut _ as *mut c_void,
                &mut tcnt as *mut _ as *mut c_void,
                &mut z as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut p_i as *mut _ as *mut c_void,
            ];
            unsafe {
                self.pf_gemm.launch(
                    (n_tiles as u32, (v.dims / 128) as u32, 1),
                    (128, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }

        // 5. Fused Down Epilogue + Weighted Scatter-Add directly into `out_ptr[B, dims]`
        {
            let mut z = sc.z.ptr();
            let mut pr = sc.d_pair_row.ptr();
            let mut ps = sc.d_pair_slot.ptr();
            let mut pw = sc.d_pair_wt.ptr();
            let mut svd = v.svh_d;
            let mut oa = out_ptr;
            let mut d_i = v.dims as i32;
            let mut args: [*mut c_void; 7] = [
                &mut z as *mut _ as *mut c_void,
                &mut pr as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut svd as *mut _ as *mut c_void,
                &mut oa as *mut _ as *mut c_void,
                &mut d_i as *mut _ as *mut c_void,
            ];
            unsafe {
                self.pf_down_scatter.launch(
                    (p as u32, (v.dims / 128) as u32, 1),
                    (32, 1, 1),
                    0,
                    stream,
                    &mut args,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use m3tis_format::exl3_ref::{f32_to_bf16, reference_exl3_expert_mlp};

    fn make_f16_one() -> u16 {
        0x3C00 // 1.0 in IEEE fp16
    }

    fn make_f16_scale(i: usize) -> u16 {
        // Alternate +0.25 (0x3400) and -0.25 (0xB400) and +0.125 (0x3000) to keep activations well-scaled
        match i % 3 {
            0 => 0x3400,
            1 => 0xB400,
            _ => 0x3000,
        }
    }

    fn gen_trellis(seed: u64, kt: usize, nt: usize, k2: usize) -> Vec<u8> {
        let mut s = seed;
        let n_u32 = kt * nt * 4 * k2;
        let mut out = Vec::with_capacity(n_u32 * 4);
        for _ in 0..n_u32 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            out.extend_from_slice(&(s as u32).to_le_bytes());
        }
        out
    }

    #[test]
    fn gpu_exl3_mixed_2b_3b_4b_matches_cpu_reference_and_ahead_bit_exact() {
        let ctx = CudaContext::new(0).expect("RTX 3090");
        let stream = ctx.stream().unwrap();
        let kernels = Exl3MoeKernels::load(&ctx).expect("load exl3_moe.cubin");

        let dims = 256usize;
        let inter = 256usize;
        let rows = 2usize;
        let slots = 4usize; // 3 routed experts (mixed 2b, 3b, 4b!) + 1 skipped shared slot
        let n_experts = 3usize;
        let k2s = [4usize, 6usize, 8usize]; // 2-bit, 3-bit, 4-bit simultaneously in ONE launch!

        // Build trellises for all 3 experts (gate, up, down)
        let mut gt_host = Vec::new();
        let mut ut_host = Vec::new();
        let mut dt_host = Vec::new();
        let mut gt_dev = Vec::new();
        let mut ut_dev = Vec::new();
        let mut dt_dev = Vec::new();
        for e in 0..n_experts {
            let g = gen_trellis(100 + e as u64 * 17, dims / 16, inter / 16, k2s[e]);
            let u = gen_trellis(200 + e as u64 * 31, dims / 16, inter / 16, k2s[e]);
            let d = gen_trellis(300 + e as u64 * 43, inter / 16, dims / 16, k2s[e]);
            let dg = ctx.alloc_device(g.len()).unwrap();
            let du = ctx.alloc_device(u.len()).unwrap();
            let dd = ctx.alloc_device(d.len()).unwrap();
            unsafe {
                stream.htod_async(dg.ptr(), g.as_ptr(), g.len()).unwrap();
                stream.htod_async(du.ptr(), u.as_ptr(), u.len()).unwrap();
                stream.htod_async(dd.ptr(), d.as_ptr(), d.len()).unwrap();
            }
            gt_host.push(g);
            ut_host.push(u);
            dt_host.push(d);
            gt_dev.push(dg);
            ut_dev.push(du);
            dt_dev.push(dd);
        }

        let gt_ptrs_h: Vec<u64> = gt_dev.iter().map(|b| b.ptr()).collect();
        let ut_ptrs_h: Vec<u64> = ut_dev.iter().map(|b| b.ptr()).collect();
        let dt_ptrs_h: Vec<u64> = dt_dev.iter().map(|b| b.ptr()).collect();
        let k2_i32: Vec<i32> = k2s.iter().map(|&x| x as i32).collect();

        let d_gt_ptrs = ctx.alloc_device(n_experts * 8).unwrap();
        let d_ut_ptrs = ctx.alloc_device(n_experts * 8).unwrap();
        let d_dt_ptrs = ctx.alloc_device(n_experts * 8).unwrap();
        let d_k2 = ctx.alloc_device(n_experts * 4).unwrap();
        unsafe {
            stream.htod_async(d_gt_ptrs.ptr(), gt_ptrs_h.as_ptr() as *const u8, n_experts * 8).unwrap();
            stream.htod_async(d_ut_ptrs.ptr(), ut_ptrs_h.as_ptr() as *const u8, n_experts * 8).unwrap();
            stream.htod_async(d_dt_ptrs.ptr(), dt_ptrs_h.as_ptr() as *const u8, n_experts * 8).unwrap();
            stream.htod_async(d_k2.ptr(), k2_i32.as_ptr() as *const u8, n_experts * 4).unwrap();
        }

        // Scales [n_experts, dims/inter] fp16
        let suh_g_h: Vec<u16> = (0..n_experts * dims).map(make_f16_scale).collect();
        let suh_u_h: Vec<u16> = (0..n_experts * dims).map(|i| make_f16_scale(i + 1)).collect();
        let svh_g_h: Vec<u16> = (0..n_experts * inter).map(|_| make_f16_one()).collect();
        let svh_u_h: Vec<u16> = (0..n_experts * inter).map(|_| make_f16_one()).collect();
        let suh_d_h: Vec<u16> = (0..n_experts * inter).map(make_f16_scale).collect();
        let svh_d_h: Vec<u16> = (0..n_experts * dims).map(|_| make_f16_one()).collect();

        let upload_u16 = |v: &[u16]| -> DeviceBuf {
            let b = ctx.alloc_device(v.len() * 2).unwrap();
            unsafe {
                stream.htod_async(b.ptr(), v.as_ptr() as *const u8, v.len() * 2).unwrap();
            }
            b
        };
        let d_suh_g = upload_u16(&suh_g_h);
        let d_suh_u = upload_u16(&suh_u_h);
        let d_svh_g = upload_u16(&svh_g_h);
        let d_svh_u = upload_u16(&svh_u_h);
        let d_suh_d = upload_u16(&suh_d_h);
        let d_svh_d = upload_u16(&svh_d_h);

        // Input BF16 activations [rows, dims]
        let mut x_f32 = vec![0.0f32; rows * dims];
        let mut x_bf16 = vec![0u16; rows * dims];
        for i in 0..rows * dims {
            let val = (((i * 37 + 11) % 200) as f32 - 100.0) * 0.01;
            let b16 = f32_to_bf16(val);
            x_bf16[i] = b16;
            x_f32[i] = f32::from_bits((b16 as u32) << 16);
        }
        let d_x = upload_u16(&x_bf16);

        // Row 0 picks experts [0 (2b), 1 (3b), 2 (4b), skip(3)]; Row 1 picks [2 (4b), 0 (2b), 1 (3b), skip(3)]
        let picks_h: [i32; 8] = [0, 1, 2, 3, 2, 0, 1, 3];
        let wts_h: [f32; 8] = [0.5, 1.2, 0.8, 0.0, 0.9, 0.6, 1.0, 0.0];
        let d_picks = ctx.alloc_device(8 * 4).unwrap();
        let d_wts = ctx.alloc_device(8 * 4).unwrap();
        let d_out_ahead = ctx.alloc_device(rows * dims * 4).unwrap();
        let d_out_plain = ctx.alloc_device(rows * dims * 4).unwrap();
        unsafe {
            stream.htod_async(d_picks.ptr(), picks_h.as_ptr() as *const u8, 8 * 4).unwrap();
            stream.htod_async(d_wts.ptr(), wts_h.as_ptr() as *const u8, 8 * 4).unwrap();
        }

        let sc = Exl3MoeScratch::alloc(&ctx, rows, slots, dims, inter, 4).unwrap();
        stream.memset_d32_async(sc.slot_y.ptr(), 0, rows * slots * dims).unwrap();

        let view = Exl3ExpertsView {
            gt_ptrs: d_gt_ptrs.ptr(),
            ut_ptrs: d_ut_ptrs.ptr(),
            dt_ptrs: d_dt_ptrs.ptr(),
            k2_g: d_k2.ptr(),
            k2_u: d_k2.ptr(),
            k2_d: d_k2.ptr(),
            suh_g: d_suh_g.ptr(),
            suh_u: d_suh_u.ptr(),
            svh_g: d_svh_g.ptr(),
            svh_u: d_svh_u.ptr(),
            suh_d: d_suh_d.ptr(),
            svh_d: d_svh_d.ptr(),
            experts: n_experts,
            dims,
            inter,
            swiglu_limit: 10.0,
            act_mode: 0,
        };

        kernels
            .run_moe(&stream, &view, &sc, d_x.ptr(), dims, InputDType::Bf16, d_picks.ptr(), d_wts.ptr(), d_out_ahead.ptr(), rows, slots, true)
            .unwrap();
        kernels
            .run_moe(&stream, &view, &sc, d_x.ptr(), dims, InputDType::Bf16, d_picks.ptr(), d_wts.ptr(), d_out_plain.ptr(), rows, slots, false)
            .unwrap();

        let mut out_ahead = vec![0.0f32; rows * dims];
        let mut out_plain = vec![0.0f32; rows * dims];
        unsafe {
            stream.dtoh_async(out_ahead.as_mut_ptr() as *mut u8, d_out_ahead.ptr(), rows * dims * 4).unwrap();
            stream.dtoh_async(out_plain.as_mut_ptr() as *mut u8, d_out_plain.ptr(), rows * dims * 4).unwrap();
        }
        stream.synchronize().unwrap();

        // 1. AHEAD vs PLAIN must be 100% BIT-EXACT
        for i in 0..rows * dims {
            assert_eq!(
                out_ahead[i].to_bits(),
                out_plain[i].to_bits(),
                "AHEAD != PLAIN at index {i}: {} vs {}",
                out_ahead[i],
                out_plain[i]
            );
        }

        // 2. Compare against clean-room CPU FP64 EXL3 oracle
        let mut max_abs = 0.0f32;
        let mut err_sq = 0.0f64;
        let mut dot = 0.0f64;
        let mut n_gpu = 0.0f64;
        let mut n_ref = 0.0f64;
        for r in 0..rows {
            let mut ref_row = vec![0.0f32; dims];
            for s in 0..slots {
                let e = picks_h[r * slots + s] as usize;
                let w = wts_h[r * slots + s];
                if e >= n_experts || w == 0.0 {
                    continue;
                }
                let y = reference_exl3_expert_mlp(
                    &x_f32[r * dims..(r + 1) * dims],
                    &gt_host[e],
                    &ut_host[e],
                    &dt_host[e],
                    k2s[e],
                    k2s[e],
                    k2s[e],
                    &suh_g_h[e * dims..(e + 1) * dims],
                    &suh_u_h[e * dims..(e + 1) * dims],
                    &svh_g_h[e * inter..(e + 1) * inter],
                    &svh_u_h[e * inter..(e + 1) * inter],
                    &suh_d_h[e * inter..(e + 1) * inter],
                    &svh_d_h[e * dims..(e + 1) * dims],
                    dims,
                    inter,
                    10.0,
                );
                for j in 0..dims {
                    ref_row[j] += w * y[j];
                }
            }
            for j in 0..dims {
                let g = out_ahead[r * dims + j];
                let rf = ref_row[j];
                let diff = (g - rf) as f64;
                max_abs = max_abs.max(diff.abs() as f32);
                err_sq += diff * diff;
                dot += (g as f64) * (rf as f64);
                n_gpu += (g as f64) * (g as f64);
                n_ref += (rf as f64) * (rf as f64);
            }
        }
        let rel_l2 = (err_sq / n_ref.max(1e-30)).sqrt();
        let cos = dot / (n_gpu.sqrt() * n_ref.sqrt());
        println!(
            "[RTX 3090 EXL3 2b+3b+4b mixed parity] rel_L2={rel_l2:.6} max_abs={max_abs:.6} cos_sim={cos:.8} (AHEAD==PLAIN BIT-EXACT)"
        );
        assert!(cos > 0.99995, "cosine similarity too low: {cos}");
        assert!(rel_l2 < 0.005, "rel_L2 too high: {rel_l2}");

        // 3. Verify Milestone 5B flat-tile Expert-Major Prefill kernel matches AHEAD across 2 waves (cos_sim > 0.999999!)
        let mut pf_sc = Exl3PrefillScratch::alloc(&ctx, 64, 32, dims, inter).unwrap();
        let d_out_pf = ctx.alloc_device(rows * dims * 4).unwrap();
        stream.memset_d32_async(d_out_pf.ptr(), 0, rows * dims).unwrap();
        // Split experts across 2 independent waves ([0, 1] then [2]) to exercise atomicAdd scatter across waves!
        for (wave_idx, exp_group) in [&[0usize, 1usize][..], &[2usize][..]].iter().enumerate() {
            let mut p_row = Vec::new();
            let mut p_slot = Vec::new();
            let mut p_wt = Vec::new();
            let mut t_slot = Vec::new();
            let mut t_off = Vec::new();
            let mut t_cnt = Vec::new();
            for &e in *exp_group {
                let start = p_row.len();
                for r in 0..rows {
                    for s in 0..slots {
                        if picks_h[r * slots + s] as usize == e && wts_h[r * slots + s] != 0.0 {
                            p_row.push(r as i32);
                            p_slot.push(e as i32);
                            p_wt.push(wts_h[r * slots + s]);
                        }
                    }
                }
                let tot = p_row.len() - start;
                let mut off = 0usize;
                while off < tot {
                    let c = (tot - off).min(16);
                    t_slot.push(e as i32);
                    t_off.push((start + off) as i32);
                    t_cnt.push(c as i32);
                    off += c;
                }
            }
            kernels
                .run_prefill_wave(
                    &stream,
                    &view,
                    &mut pf_sc,
                    wave_idx,
                    d_x.ptr(),
                    dims,
                    d_out_pf.ptr(),
                    &p_row,
                    &p_slot,
                    &p_wt,
                    &t_slot,
                    &t_off,
                    &t_cnt,
                )
                .unwrap();
        }
        let mut out_pf = vec![0.0f32; rows * dims];
        unsafe {
            stream
                .dtoh_async(out_pf.as_mut_ptr() as *mut u8, d_out_pf.ptr(), rows * dims * 4)
                .unwrap();
        }
        stream.synchronize().unwrap();
        let mut pf_max_abs = 0.0f32;
        let mut pf_dot = 0.0f64;
        let mut pf_na = 0.0f64;
        let mut pf_nb = 0.0f64;
        for i in 0..rows * dims {
            let a = out_ahead[i];
            let b = out_pf[i];
            pf_max_abs = pf_max_abs.max((a - b).abs());
            pf_dot += (a as f64) * (b as f64);
            pf_na += (a as f64) * (a as f64);
            pf_nb += (b as f64) * (b as f64);
        }
        let pf_cos = pf_dot / (pf_na.sqrt() * pf_nb.sqrt());
        println!("[RTX 3090 EXL3 5B Prefill Wave vs AHEAD] max_abs={pf_max_abs:.6} cos_sim={pf_cos:.8}");
        assert!(pf_cos > 0.999999, "prefill wave cos_sim too low: {pf_cos}");
        assert!(pf_max_abs < 1e-3, "prefill wave max_abs too high: {pf_max_abs}");
    }
}
