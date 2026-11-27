// Clean-room M3TiS sm_86 On-GPU EXL3 3-bit -> 2-bit MCG Trellis Beam-Viterbi Requantizer.
//
// Operates directly in the Hadamard-rotated 16x16 EXL3 tile domain, so requantized 2-bit cold
// experts reuse the exact same suh/svh scales as the 3-bit checkpoint with ZERO scale changes.
// Each warp (32 lanes) encodes one 16x16 tile using 32-beam tail-biting Viterbi search
// with pure warp-shuffle register bitonic pruning (ZERO smem sort traffic!) and exact 16-bit closure.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace m3tis_req {

constexpr int WARPS_PER_CTA = 2; // 64 threads = 2 warps = ~18.5 KiB smem -> 5 CTAs/SM = 10 warps/SM!

__device__ __forceinline__ float decode_mcg_f32(uint32_t state) {
    uint32_t x = ((state * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    __half2 h2 = *reinterpret_cast<const __half2*>(&x);
    float2 f2 = __half22float2(h2);
    return f2.x + f2.y;
}

__device__ __forceinline__ uint32_t read_16bits_ring(const uint32_t* words, int total_bits, int start) {
    uint32_t state = 0u;
#pragma unroll
    for (int b = 0; b < 16; ++b) {
        int bit_pos = start + b;
        if (bit_pos >= total_bits) bit_pos -= total_bits;
        uint32_t w = words[bit_pos >> 5];
        uint32_t bit = (w >> (31 - (bit_pos & 31))) & 1u;
        state = (state << 1) | bit;
    }
    return state;
}

struct WarpBuf {
    float target[256];
    uint8_t parent[256][32];
};

// Sort 32 elements across a full warp purely in registers via __shfl_xor_sync!
__device__ __forceinline__ void bitonic_sort_32_reg(
    int lane, float& c, uint32_t& s, uint32_t& m, bool dir_asc
) {
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const float oc = __shfl_xor_sync(0xffffffffu, c, j);
            const uint32_t os = __shfl_xor_sync(0xffffffffu, s, j);
            const uint32_t om = __shfl_xor_sync(0xffffffffu, m, j);
            const bool hi = (lane & j) != 0;
            const bool asc = ((lane & k) == 0) == dir_asc;
            const bool take = hi ? (asc ? (oc > c) : (oc < c)) : (asc ? (oc < c) : (oc > c));
            if (take) { c = oc; s = os; m = om; }
        }
    }
}

// Merge two 32-element sequences (A in (c0,s0,m0), B in (c1,s1,m1)) and keep the exact 32 smallest
// in (c0,s0,m0) sorted ascending (lane 0 holds global min)!
__device__ __forceinline__ void merge_top32_reg(
    int lane,
    float& c0, uint32_t& s0, uint32_t& m0,
    float c1, uint32_t s1, uint32_t m1
) {
    bitonic_sort_32_reg(lane, c0, s0, m0, true);
    bitonic_sort_32_reg(lane, c1, s1, m1, false);
    if (c1 < c0) { c0 = c1; s0 = s1; m0 = m1; }
#pragma unroll
    for (int j = 16; j > 0; j >>= 1) {
        const float oc = __shfl_xor_sync(0xffffffffu, c0, j);
        const uint32_t os = __shfl_xor_sync(0xffffffffu, s0, j);
        const uint32_t om = __shfl_xor_sync(0xffffffffu, m0, j);
        const bool hi = (lane & j) != 0;
        if (hi ? (oc > c0) : (oc < c0)) { c0 = oc; s0 = os; m0 = om; }
    }
}

} // namespace m3tis_req

extern "C" {

// Requantize `num_tiles` 3-bit (`k2 = 6`, 24 u32 words/tile) EXL3 MCG tiles -> 2-bit (`k2 = 4`, 16 u32 words/tile).
// Launch: grid((num_tiles + 1) / 2, 1, 1), block(64, 1, 1) -> 2 warps = 2 tiles per CTA.
__global__ void __launch_bounds__(64) m3tis_exl3_requant_3b_to_2b(
    const uint32_t* __restrict__ in_trellis, // [num_tiles, 24]
    uint32_t* __restrict__ out_trellis,      // [num_tiles, 16]
    int32_t num_tiles
) {
    __shared__ m3tis_req::WarpBuf smem[m3tis_req::WARPS_PER_CTA];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile = blockIdx.x * m3tis_req::WARPS_PER_CTA + warp;
    if (tile >= num_tiles) return;

    m3tis_req::WarpBuf& wb = smem[warp];
    const uint32_t* src = in_trellis + (size_t)tile * 24;

    // 1. Decode incoming 3-bit (k2=6, 768 bits) MCG tile -> target[0..255] (8 values per lane).
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int p = lane * 8 + j;
        const int end = (p + 1) * 3;
        const int start = (end + 768 - 16) % 768;
        const uint32_t st3 = m3tis_req::read_16bits_ring(src, 768, start);
        wb.target[p] = m3tis_req::decode_mcg_f32(st3);
    }
    __syncwarp();

    // 2. Short 24-step tail warmup (p = 232..255) to find optimal cyclic closing 16-bit state!
    uint32_t b_state = 0u;
    float b_cost = (lane == 0) ? 0.0f : 1e30f;
    for (int p = 232; p < 256; ++p) {
        const float tgt = wb.target[p];
        float c0 = 1e30f, c1 = 1e30f;
        uint32_t s0 = 0u, s1 = 0u, m0 = 0u, m1 = 0u;
        if (b_cost < 1e29f) {
            const uint32_t shifted = (b_state << 2) & 0xFFFFu;
#pragma unroll
            for (uint32_t sym = 0u; sym < 4u; ++sym) {
                const uint32_t ns = shifted | sym;
                const float diff = m3tis_req::decode_mcg_f32(ns) - tgt;
                const float nc = b_cost + diff * diff;
                if (nc < c0) { c1 = c0; s1 = s0; c0 = nc; s0 = ns; }
                else if (nc < c1) { c1 = nc; s1 = ns; }
            }
        }
        m3tis_req::merge_top32_reg(lane, c0, s0, m0, c1, s1, m1);
        const float base = __shfl_sync(0xffffffffu, c0, 0);
        b_cost = (c0 < 1e29f) ? (c0 - base) : 1e30f;
        b_state = s0;
    }
    const uint32_t seed_state = __shfl_sync(0xffffffffu, b_state, 0);

    // 3. Main 32-beam Viterbi pass (p = 0..247 free, p = 248..255 steered cleanly back to seed_state!).
    if (lane < 4) {
        const uint32_t ns = ((seed_state << 2) | (uint32_t)lane) & 0xFFFFu;
        const float diff = m3tis_req::decode_mcg_f32(ns) - wb.target[0];
        b_cost = diff * diff;
        b_state = ns;
        wb.parent[0][lane] = (uint8_t)(lane << 5);
    } else {
        b_cost = 1e30f;
        b_state = 0u;
        wb.parent[0][lane] = 0u;
    }
    __syncwarp();

    for (int p = 1; p < 248; ++p) {
        const float tgt = wb.target[p];
        float c0 = 1e30f, c1 = 1e30f;
        uint32_t s0 = 0u, s1 = 0u, m0 = 0u, m1 = 0u;
        if (b_cost < 1e29f) {
            const uint32_t shifted = (b_state << 2) & 0xFFFFu;
#pragma unroll
            for (uint32_t sym = 0u; sym < 4u; ++sym) {
                const uint32_t ns = shifted | sym;
                const float diff = m3tis_req::decode_mcg_f32(ns) - tgt;
                const float nc = b_cost + diff * diff;
                const uint32_t meta = (sym << 5) | (uint32_t)lane;
                if (nc < c0) { c1 = c0; s1 = s0; m1 = m0; c0 = nc; s0 = ns; m0 = meta; }
                else if (nc < c1) { c1 = nc; s1 = ns; m1 = meta; }
            }
        }
        m3tis_req::merge_top32_reg(lane, c0, s0, m0, c1, s1, m1);
        const float base = __shfl_sync(0xffffffffu, c0, 0);
        b_cost = (c0 < 1e29f) ? (c0 - base) : 1e30f;
        b_state = s0;
        wb.parent[p][lane] = (uint8_t)m0;
    }

    // Closing 8 steps p = 248..255: steer all 32 beams back to exact cyclic `seed_state`!
    for (int p = 248; p < 256; ++p) {
        const uint32_t sym = (seed_state >> (14 - 2 * (p - 248))) & 3u;
        const uint32_t ns = ((b_state << 2) | sym) & 0xFFFFu;
        if (b_cost < 1e29f) {
            const float diff = m3tis_req::decode_mcg_f32(ns) - wb.target[p];
            b_cost += diff * diff;
        }
        b_state = ns;
        wb.parent[p][lane] = (uint8_t)((sym << 5) | (uint32_t)lane);
    }

    // Pick best surviving beam across 32 lanes in registers
    uint32_t best_lane = (uint32_t)lane;
    float best_c = b_cost;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float oc = __shfl_xor_sync(0xffffffffu, best_c, off);
        const uint32_t ol = __shfl_xor_sync(0xffffffffu, best_lane, off);
        if (oc < best_c) { best_c = oc; best_lane = ol; }
    }
    __syncwarp();

    // 4. Traceback on lane 0 & write 16 MSB-first u32 words!
    if (lane == 0) {
        int cur = (int)best_lane;
        uint32_t words[16];
#pragma unroll
        for (int w = 0; w < 16; ++w) words[w] = 0u;
        for (int p = 255; p >= 0; --p) {
            const uint8_t meta = wb.parent[p][cur];
            const uint32_t sym = (uint32_t)(meta >> 5) & 3u;
            cur = (int)(meta & 31u);
            words[p >> 4] |= sym << (30 - 2 * (p & 15));
        }
        uint32_t* dst = out_trellis + (size_t)tile * 16;
#pragma unroll
        for (int w = 0; w < 16; ++w) dst[w] = words[w];
    }
}

} // extern "C"
