//! Resident pinned-Host-RAM EXL3 Hadamard scale bank (`suh`/`svh`) + per-VRAM-slot mirror.
//!
//! For GLM-5.3-Flash (42 MoE layers x 288 experts = 12,096 experts, dims=4096, inter=2048):
//! - All 6 fp16 scale tables across the entire 320B model occupy only ~425 MiB of pinned Host RAM!
//! - Mirroring active VRAM slots (`vram_slots <= 600`) takes only ~21 MiB of GPU VRAM, eliminating
//!   67% of NVMe IOPS (3 cold trellis reads instead of 9 per expert!).

use m3tis_cuda::{CuDevicePtr, CudaContext, DeviceBuf, PinnedHostBuf, Stream};
use m3tis_format::safetensors::SafetensorsIndex;
use std::sync::Arc;

/// `[suh_g, suh_u, svh_g, svh_u, suh_d, svh_d]`
pub const NUM_SCALES: usize = 6;
/// Double-buffered streaming prefill scale ring slots (`2 x 8` experts = 576 KiB VRAM).
pub const PF_SCALE_SLOTS: usize = 16;

pub struct Exl3ScaleBank {
    pub dims: usize,
    pub inter: usize,
    pub vram_slots: usize,
    pub strides: [usize; NUM_SCALES], // byte length per expert for each scale
    ram: [PinnedHostBuf; NUM_SCALES],
    vram: [DeviceBuf; NUM_SCALES],
    pf_vram: [DeviceBuf; NUM_SCALES],
    slot_owner: Vec<i32>, // [vram_slots] global expert key currently mirrored in VRAM slot
}

impl Exl3ScaleBank {
    /// Allocate scale bank for `total_experts` (pinned RAM) and `vram_slots` (VRAM).
    pub fn alloc(
        ctx: &Arc<CudaContext>,
        total_experts: usize,
        vram_slots: usize,
        dims: usize,
        inter: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let strides = [
            dims * 2,  // suh_g
            dims * 2,  // suh_u
            inter * 2, // svh_g
            inter * 2, // svh_u
            inter * 2, // suh_d
            dims * 2,  // svh_d
        ];
        let ram = [
            ctx.alloc_pinned(total_experts * strides[0])?,
            ctx.alloc_pinned(total_experts * strides[1])?,
            ctx.alloc_pinned(total_experts * strides[2])?,
            ctx.alloc_pinned(total_experts * strides[3])?,
            ctx.alloc_pinned(total_experts * strides[4])?,
            ctx.alloc_pinned(total_experts * strides[5])?,
        ];
        let vram = [
            ctx.alloc_device(vram_slots * strides[0])?,
            ctx.alloc_device(vram_slots * strides[1])?,
            ctx.alloc_device(vram_slots * strides[2])?,
            ctx.alloc_device(vram_slots * strides[3])?,
            ctx.alloc_device(vram_slots * strides[4])?,
            ctx.alloc_device(vram_slots * strides[5])?,
        ];
        let pf_vram = [
            ctx.alloc_device(PF_SCALE_SLOTS * strides[0])?,
            ctx.alloc_device(PF_SCALE_SLOTS * strides[1])?,
            ctx.alloc_device(PF_SCALE_SLOTS * strides[2])?,
            ctx.alloc_device(PF_SCALE_SLOTS * strides[3])?,
            ctx.alloc_device(PF_SCALE_SLOTS * strides[4])?,
            ctx.alloc_device(PF_SCALE_SLOTS * strides[5])?,
        ];
        Ok(Self {
            dims,
            inter,
            vram_slots,
            strides,
            ram,
            vram,
            pf_vram,
            slot_owner: vec![-1; vram_slots],
        })
    }

    /// Load one expert's 6 scale tensors (`suh`/`svh` for gate/up/down) from safetensors into pinned RAM.
    pub fn load_expert_from_safetensors(
        &mut self,
        st: &SafetensorsIndex,
        prefix: &str,
        model_layer: usize,
        local_expert: usize,
        global_key: usize,
    ) -> std::io::Result<()> {
        let names = [
            format!("{prefix}layers.{model_layer}.mlp.experts.{local_expert}.gate_proj.suh"),
            format!("{prefix}layers.{model_layer}.mlp.experts.{local_expert}.up_proj.suh"),
            format!("{prefix}layers.{model_layer}.mlp.experts.{local_expert}.gate_proj.svh"),
            format!("{prefix}layers.{model_layer}.mlp.experts.{local_expert}.up_proj.svh"),
            format!("{prefix}layers.{model_layer}.mlp.experts.{local_expert}.down_proj.suh"),
            format!("{prefix}layers.{model_layer}.mlp.experts.{local_expert}.down_proj.svh"),
        ];
        for s in 0..NUM_SCALES {
            let stride = self.strides[s];
            let off = global_key * stride;
            st.read_into(&names[s], &mut self.ram[s].as_mut_slice()[off..off + stride])?;
        }
        Ok(())
    }

    /// Asynchronously mirror scales into `vram_slot` for `global_key` if not already mirrored there.
    pub fn sync_slot_async(
        &mut self,
        copy_stream: &Stream,
        global_key: usize,
        vram_slot: usize,
    ) -> Result<bool, m3tis_cuda::CudaError> {
        if self.slot_owner[vram_slot] == global_key as i32 {
            return Ok(false);
        }
        for s in 0..NUM_SCALES {
            let stride = self.strides[s];
            let src = unsafe { self.ram[s].as_ptr().add(global_key * stride) };
            let dst = self.vram[s].at(vram_slot * stride);
            unsafe {
                copy_stream.htod_async(dst, src, stride)?;
            }
        }
        self.slot_owner[vram_slot] = global_key as i32;
        Ok(true)
    }

    #[inline]
    pub fn vram_ptr(&self, idx: usize) -> CuDevicePtr {
        self.vram[idx].ptr()
    }

    #[inline]
    pub fn pf_vram_ptr(&self, idx: usize) -> CuDevicePtr {
        self.pf_vram[idx].ptr()
    }

    /// Asynchronously copy all 6 scales of `global_key` into dedicated prefill streaming ring `stage_slot` (`0..PF_SCALE_SLOTS`).
    pub fn pf_copy_slot_async(
        &self,
        copy_stream: &Stream,
        global_key: usize,
        stage_slot: usize,
    ) -> Result<(), m3tis_cuda::CudaError> {
        debug_assert!(stage_slot < PF_SCALE_SLOTS);
        for s in 0..NUM_SCALES {
            let stride = self.strides[s];
            let src = unsafe { self.ram[s].as_ptr().add(global_key * stride) };
            let dst = self.pf_vram[s].at(stage_slot * stride);
            unsafe {
                copy_stream.htod_async(dst, src, stride)?;
            }
        }
        Ok(())
    }

    #[inline]
    pub fn ram_scale_ptr(&self, idx: usize, global_key: usize) -> *const u16 {
        let stride = self.strides[idx];
        let off = global_key * stride;
        unsafe { self.ram[idx].as_ptr().add(off) as *const u16 }
    }

    pub fn ram_scale_u16(&self, idx: usize, global_key: usize) -> Vec<u16> {
        let stride = self.strides[idx];
        let off = global_key * stride;
        let bytes = &self.ram[idx].as_slice()[off..off + stride];
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}
