// Clean-room M3TiS sm_86 CUDA kernels: FSR slot remap, fused sigmoid+DynTopK selection, slab checksum.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" {

// Map logical expert IDs [0..logical_experts) to resident VRAM slab slots [0..vram_slots).
// Entries equal to `skip_in` (shared expert or DynTopK-pruned slots) map to `skip_out`.
__global__ void m3tis_fsr_remap_i32(
    const int32_t* __restrict__ in_picks,
    int32_t* __restrict__ out_picks,
    const int32_t* __restrict__ layer_map,
    int32_t count,
    int32_t logical_experts,
    int32_t skip_in,
    int32_t skip_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int32_t e = in_picks[idx];
    if (e == skip_in || e < 0 || e >= logical_experts) {
        out_picks[idx] = skip_out;
        return;
    }
    int32_t slot = layer_map[e];
    out_picks[idx] = (slot >= 0) ? slot : skip_out;
}

// Map logical expert IDs [0..logical_experts) -> VRAM slots while masking out non-resident (cold NVMe)
// slots and renormalizing kept routed weights back to `target_scale` in ONE zero-sync GPU launch!
// Powers both 100% lossless zero-NVMe self-speculation and Never-Stall tail routing.
// Launch: grid(rows, 1, 1), block(32, 1, 1)
__global__ void m3tis_fsr_remap_resident_renorm(
    const int32_t* __restrict__ in_picks,  // [rows, slots]
    const float* __restrict__ in_wts,      // [rows, slots]
    int32_t* __restrict__ out_picks,       // [rows, slots]
    float* __restrict__ out_wts,           // [rows, slots] (may alias in_wts)
    const int32_t* __restrict__ layer_map, // [logical_experts] -> VRAM slot or -1
    int32_t rows,
    int32_t top_k,
    int32_t logical_experts,
    int32_t skip_in,
    int32_t skip_out,
    float target_scale
) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= rows || lane != 0) return;

    int slots = top_k + 1;
    const int32_t* rp_in = in_picks + (size_t)row * (size_t)slots;
    const float* rw_in = in_wts + (size_t)row * (size_t)slots;
    int32_t* rp_out = out_picks + (size_t)row * (size_t)slots;
    float* rw_out = out_wts + (size_t)row * (size_t)slots;

    float w_tmp[16];
    int32_t p_tmp[16];
    float sum_w = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int32_t e = rp_in[k];
        float w = rw_in[k];
        if (e == skip_in || e < 0 || e >= logical_experts || w <= 0.0f) {
            p_tmp[k] = skip_out;
            w_tmp[k] = 0.0f;
        } else {
            int32_t slot = layer_map[e];
            if (slot >= 0) {
                p_tmp[k] = slot;
                w_tmp[k] = w;
                sum_w += w;
            } else {
                p_tmp[k] = skip_out;
                w_tmp[k] = 0.0f;
            }
        }
    }
    float renorm = (sum_w > 1e-7f && target_scale > 0.0f) ? (target_scale / sum_w) : 1.0f;
    for (int k = 0; k < top_k; ++k) {
        rp_out[k] = p_tmp[k];
        rw_out[k] = w_tmp[k] * renorm;
    }
    rp_out[top_k] = skip_out;
    rw_out[top_k] = rw_in[top_k];
}

// Fused GLM-5.3 router selection + entropy-adaptive Dynamic Top-K + optional VRAM residency bias on GPU.
// 4 warps (128 threads) per token row -> up to 8 register-unrolled chunks/thread (E <= 1024, ZERO local-mem spill!).
// Launch: grid(rows, 1, 1), block(128, 1, 1)
__global__ void __launch_bounds__(128) m3tis_sigmoid_topk_dyntopk_f32(
    const float* __restrict__ logits,  // [rows, E]
    const float* __restrict__ bias,    // [E] (nullable)
    int32_t* __restrict__ out_picks,   // [rows, slots] where slots = top_k + 1
    float* __restrict__ out_wts,       // [rows, slots]
    int32_t rows,
    int32_t E,
    int32_t top_k,
    float scale,
    float dyn_mass,                    // 0.0f = exact top_k; e.g. 0.80f = keep min_k..top_k covering 80% mass
    int32_t dyn_min_k,
    int32_t shared_id,
    const int32_t* __restrict__ layer_map, // nullable [E] -> VRAM slot >= 0 or -1
    float residency_bias,              // 0.0f = exact; >0 gives resident VRAM experts a tail tiebreak bonus
    int32_t strict_k                   // ranks 0..strict_k-1 never apply residency_bias
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (row >= rows) return;

    const float* row_logits = logits + (size_t)row * (size_t)E;
    const int slots = top_k + 1;
    int32_t* row_p = out_picks + (size_t)row * (size_t)slots;
    float* row_w = out_wts + (size_t)row * (size_t)slots;

    __shared__ int32_t chosen_id[16];
    __shared__ float chosen_prob[16];
    __shared__ float w_score[4];
    __shared__ float w_prob[4];
    __shared__ int32_t w_id[4];
    __shared__ int32_t s_win_id;

    // 8 compile-time unrolled chunks x 128 threads = up to 1024 experts kept 100% in registers (ZERO local spill!).
    float p_reg[8];
    float s_base[8];
    uint32_t res_mask = 0u;
    uint32_t ram_mask = 0u;
    uint32_t used_mask = 0u;
    const bool has_rb = (layer_map != nullptr && residency_bias > 0.0f);

#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const int e = (c << 7) + tid;
        if (e < E) {
            float p = 1.0f / (1.0f + expf(-row_logits[e]));
            p_reg[c] = p;
            s_base[c] = bias ? (p + bias[e]) : p;
            if (has_rb) {
                const int32_t lv = layer_map[e];
                if (lv >= 0) res_mask |= (1u << c);
                else if (lv == -2) ram_mask |= (1u << c);
            }
        } else {
            p_reg[c] = 0.0f;
            s_base[c] = -1e30f;
            used_mask |= (1u << c);
        }
    }

    for (int k = 0; k < top_k; ++k) {
        float best_score = -1e30f;
        float best_prob = 0.0f;
        int32_t best_id = -1;
        const bool apply_rb = has_rb && (k >= strict_k);

#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if ((used_mask & (1u << c)) == 0u) {
                const int e = (c << 7) + tid;
                float bonus = 0.0f;
                if (apply_rb) {
                    if (res_mask & (1u << c)) bonus = residency_bias;
                    else if (ram_mask & (1u << c)) bonus = 0.45f * residency_bias;
                }
                float s = s_base[c] + bonus;
                if (s > best_score || (s == best_score && (best_id < 0 || e < best_id))) {
                    best_score = s;
                    best_prob = p_reg[c];
                    best_id = e;
                }
            }
        }

        // 1. Warp-wide argmax
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o_score = __shfl_down_sync(0xffffffffu, best_score, offset);
            float o_prob  = __shfl_down_sync(0xffffffffu, best_prob, offset);
            int32_t o_id  = __shfl_down_sync(0xffffffffu, best_id, offset);
            if (o_score > best_score || (o_score == best_score && o_id >= 0 && (best_id < 0 || o_id < best_id))) {
                best_score = o_score;
                best_prob = o_prob;
                best_id = o_id;
            }
        }
        if (lane == 0) {
            w_score[warp] = best_score;
            w_prob[warp] = best_prob;
            w_id[warp] = best_id;
        }
        __syncthreads();

        // 2. Cross-warp winner in thread 0
        if (tid == 0) {
            float bs = w_score[0];
            float bp = w_prob[0];
            int32_t bi = w_id[0];
#pragma unroll
            for (int w = 1; w < 4; ++w) {
                if (w_score[w] > bs || (w_score[w] == bs && w_id[w] >= 0 && (bi < 0 || w_id[w] < bi))) {
                    bs = w_score[w];
                    bp = w_prob[w];
                    bi = w_id[w];
                }
            }
            chosen_id[k] = bi;
            chosen_prob[k] = bp;
            s_win_id = bi;
        }
        __syncthreads();

        const int32_t win_id = s_win_id;
        if (win_id >= 0 && (win_id & 127) == tid) {
            used_mask |= (1u << (win_id >> 7));
        }
    }

    if (tid == 0) {
        float sum_p = 0.0f;
        for (int k = 0; k < top_k; ++k) sum_p += chosen_prob[k];
        float inv = (sum_p > 0.0f) ? (scale / sum_p) : 0.0f;
        for (int k = 0; k < top_k; ++k) chosen_prob[k] *= inv;

        int keep = top_k;
        if (dyn_mass > 0.0f && dyn_min_k < top_k && sum_p > 0.0f) {
            for (int i = 1; i < top_k; ++i) {
                float w = chosen_prob[i];
                int32_t id = chosen_id[i];
                int j = i - 1;
                while (j >= 0 && chosen_prob[j] < w) {
                    chosen_prob[j + 1] = chosen_prob[j];
                    chosen_id[j + 1] = chosen_id[j];
                    --j;
                }
                chosen_prob[j + 1] = w;
                chosen_id[j + 1] = id;
            }
            float total_w = 0.0f;
            for (int k = 0; k < top_k; ++k) total_w += chosen_prob[k];
            float target = total_w * dyn_mass;
            float cum = 0.0f;
            keep = 0;
            while (keep < top_k) {
                cum += chosen_prob[keep];
                ++keep;
                if (keep >= dyn_min_k && cum >= target) break;
            }
            if (keep < top_k && cum > 0.0f) {
                float renorm = total_w / cum;
                for (int k = 0; k < keep; ++k) chosen_prob[k] *= renorm;
                for (int k = keep; k < top_k; ++k) {
                    chosen_id[k] = shared_id;
                    chosen_prob[k] = 0.0f;
                }
            }
        }
        for (int k = 0; k < top_k; ++k) {
            row_p[k] = chosen_id[k];
            row_w[k] = chosen_prob[k];
        }
        row_p[top_k] = shared_id;
        row_w[top_k] = 1.0f;
    }
}

// Fast parallel XOR-reduction over a VRAM slab [u32; n_words] for end-to-end DMA bit-exactness verification.
__global__ void m3tis_slab_xor_checksum_u32(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ out_xor,
    uint64_t n_words
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (uint64_t i = idx; i < n_words; i += stride) {
        acc ^= data[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc ^= __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if ((threadIdx.x & 31) == 0 && acc != 0) {
        atomicXor(out_xor, acc);
    }
}

} // extern "C"
