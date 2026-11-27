// Clean-room M3TiS sm_86 DFlash2 Drafter Kernels (after z-lab/dflash, MIT).
// Supports GLM-5.3-Flash-DFlash2 (5-layer Qwen3, D=4096, QH=32, KVH=8, HD=128, WIN=2048, BLOCK=8):
// - Per-head RMSNorm + rotate-half RoPE (context ring ingest + block QKV)
// - 2-tap dynamic grouped causal conv (group_size = 16)
// - Chunked GQA sliding-window + bidirectional-block attention (CH = 16)
// - Top-16 FP32 logit extraction per draft row for codebook path selection

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace m3tis_df {

constexpr int D = 4096;
constexpr int HD = 128;
constexpr int KVH = 8;
constexpr int QH = 32;
constexpr int KVW = KVH * HD; // 1024
constexpr int WIN = 2048;
constexpr int BLOCK = 8;
constexpr int GROUP = 16;
constexpr int GROUPS = D / GROUP; // 256
constexpr int CH = 16;
constexpr int CHUNK_COLS = 4096;
constexpr int TOP = 16;

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

// Qwen3 per-head RMSNorm + rotate-half RoPE (64 threads handle dims d and d + 64).
__device__ __forceinline__ void norm_rope(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w,
    float eps,
    int64_t pos,
    float theta,
    __nv_bfloat16* out,
    float* red
) {
    const int d = threadIdx.x; // 0..63
    const float a = bf(x[d]);
    const float b = bf(x[d + 64]);
    float s = a * a + b * b;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((d & 31) == 0) red[d >> 5] = s;
    __syncthreads();
    const float rinv = rsqrtf((red[0] + red[1]) / (float)HD + eps);
    __syncthreads();

    const float na = bf(tobf(bf(w[d]) * bf(tobf(a * rinv))));
    const float nb = bf(tobf(bf(w[d + 64]) * bf(tobf(b * rinv))));
    const float inv = powf(theta, -(float)(2 * d) / (float)HD);
    const float ang = (float)pos * inv;
    const float c = cosf(ang);
    const float sn = sinf(ang);
    out[d] = tobf(na * c - nb * sn);
    out[d + 64] = tobf(nb * c + na * sn);
}

__device__ __forceinline__ bool better(float a, int32_t ia, float b, int32_t ib) {
    return a > b || (a == b && ia < ib);
}

__device__ void take_best(
    float* sv,
    int32_t* si,
    int n,
    float* rv,
    int32_t* ri,
    float* outv,
    int32_t* outi
) {
    float bv = -INFINITY;
    int32_t bi = 0x7fffffff;
    int32_t bs = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (better(sv[j], si[j], bv, bi)) {
            bv = sv[j];
            bi = si[j];
            bs = j;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        const int32_t os = __shfl_xor_sync(0xffffffffu, bs, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
            bs = os;
        }
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        rv[w] = bv;
        ri[w] = bi;
        ri[8 + w] = bs;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int best = 0;
        for (int k = 1; k < (int)(blockDim.x >> 5); ++k) {
            if (better(rv[k], ri[k], rv[best], ri[best])) best = k;
        }
        *outv = rv[best];
        *outi = ri[best];
        if (ri[8 + best] >= 0) sv[ri[8 + best]] = -INFINITY;
    }
    __syncthreads();
}

} // namespace m3tis_df

extern "C" {

// Ingest N context rows' K and V into layer's ring buffer at slots (pos0 + r) % WIN.
// Launch: grid(N, KVH=8, 1), block(64, 1, 1)
__global__ void m3tis_df_ctx_kv(
    const __nv_bfloat16* __restrict__ k_in, // [N, KVW]
    const __nv_bfloat16* __restrict__ v_in, // [N, KVW]
    const __nv_bfloat16* __restrict__ kn,   // [HD]
    __nv_bfloat16* __restrict__ ring_k,     // [WIN, KVW] for this layer
    __nv_bfloat16* __restrict__ ring_v,     // [WIN, KVW] for this layer
    int64_t pos0,
    float eps,
    float theta
) {
    __shared__ float red[2];
    const int r = blockIdx.x;
    const int kh = blockIdx.y;
    const int64_t p = pos0 + r;
    const int slot = (int)(p % m3tis_df::WIN);

    m3tis_df::norm_rope(
        k_in + (size_t)r * m3tis_df::KVW + kh * m3tis_df::HD,
        kn,
        eps,
        p,
        theta,
        ring_k + (size_t)slot * m3tis_df::KVW + kh * m3tis_df::HD,
        red
    );
    const __nv_bfloat16* vi = v_in + (size_t)r * m3tis_df::KVW + kh * m3tis_df::HD;
    __nv_bfloat16* vo = ring_v + (size_t)slot * m3tis_df::KVW + kh * m3tis_df::HD;
    vo[threadIdx.x] = vi[threadIdx.x];
    vo[threadIdx.x + 64] = vi[threadIdx.x + 64];
}

// Norm + RoPE block Q[M, 32*128] and K[M, 8*128] in-place (or to out) at positions pos0 + r.
// Launch: grid(M, QH + KVH = 40, 1), block(64, 1, 1)
__global__ void m3tis_df_block_qk_rope(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ qn,
    const __nv_bfloat16* __restrict__ kn,
    int64_t pos0,
    float eps,
    float theta,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out
) {
    __shared__ float red[2];
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int64_t p = pos0 + r;
    if (h < m3tis_df::QH) {
        m3tis_df::norm_rope(
            q_in + (size_t)r * m3tis_df::D + h * m3tis_df::HD,
            qn,
            eps,
            p,
            theta,
            q_out + (size_t)r * m3tis_df::D + h * m3tis_df::HD,
            red
        );
    } else {
        const int kh = h - m3tis_df::QH;
        m3tis_df::norm_rope(
            k_in + (size_t)r * m3tis_df::KVW + kh * m3tis_df::HD,
            kn,
            eps,
            p,
            theta,
            k_out + (size_t)r * m3tis_df::KVW + kh * m3tis_df::HD,
            red
        );
    }
}

// 2-tap dynamic causal grouped conv over block of 8 rows:
//   y[t, c] = (base[0, c] + dyn[t, which, 0, c/16]) * z[t, c]
//           + (t > 0 ? (base[1, c] + dyn[t, which, 1, c/16]) * z[t-1, c] : 0)
// Launch: grid(M, D / 1024 = 4, 1), block(256, 1, 1)
__global__ void m3tis_df_conv(
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ dyn,  // [M, 1024] = [M, 4, 256]
    int32_t which,                          // 0 = pre, 1 = post
    const __nv_bfloat16* __restrict__ base, // [2, D] slice of base_kernel
    __nv_bfloat16* __restrict__ out,
    int32_t add
) {
    const int r = blockIdx.x;
    const int t = r % m3tis_df::BLOCK;
    const __nv_bfloat16* dr = dyn + (size_t)r * (4 * m3tis_df::GROUPS) + (size_t)which * (2 * m3tis_df::GROUPS);
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x) {
        const int g = c / m3tis_df::GROUP;
        float y = (m3tis_df::bf(base[c]) + m3tis_df::bf(dr[g])) * m3tis_df::bf(z[(size_t)r * m3tis_df::D + c]);
        if (t > 0) {
            y += (m3tis_df::bf(base[m3tis_df::D + c]) + m3tis_df::bf(dr[m3tis_df::GROUPS + g]))
               * m3tis_df::bf(z[(size_t)(r - 1) * m3tis_df::D + c]);
        }
        __nv_bfloat16* o = out + (size_t)r * m3tis_df::D + c;
        *o = add ? m3tis_df::tobf(m3tis_df::bf(*o) + m3tis_df::bf(m3tis_df::tobf(y))) : m3tis_df::tobf(y);
    }
}

// Chunked GQA sliding-window + bidirectional 8-token block attention (Part 1).
// Launch: grid(KVH=8, CH=16, 1), block(256, 1, 1) (8 warps = 8 block rows j=0..7)
__global__ void __launch_bounds__(256) m3tis_df_attn_part(
    const __nv_bfloat16* __restrict__ q,      // [BLOCK, 32, 128]
    const __nv_bfloat16* __restrict__ kb,     // [BLOCK, 8, 128]
    const __nv_bfloat16* __restrict__ vb,     // [BLOCK, 8, 128]
    const __nv_bfloat16* __restrict__ ring_k, // [WIN, 8, 128]
    const __nv_bfloat16* __restrict__ ring_v, // [WIN, 8, 128]
    int64_t start_pos,
    int64_t from_pos,
    float scale,
    float* __restrict__ pm,                   // [BLOCK, 32, CH]
    float* __restrict__ pl,                   // [BLOCK, 32, CH]
    float* __restrict__ pacc                  // [BLOCK, 32, CH, 128]
) {
    const int kh = blockIdx.x;
    const int c = blockIdx.y;
    const int j = threadIdx.x >> 5;  // 0..7 block row
    const int lane = threadIdx.x & 31;

    const int64_t p = start_pos + j;
    int64_t lo = p - (m3tis_df::WIN - 1);
    if (lo < from_pos) lo = from_pos;
    if (lo < 0) lo = 0;
    const int64_t n_ctx = (start_pos > lo) ? (start_pos - lo) : 0;
    const int64_t T = n_ctx + m3tis_df::BLOCK;
    const int64_t L = (T + m3tis_df::CH - 1) / m3tis_df::CH;
    const int64_t n0 = (int64_t)c * L;
    const int64_t n1 = (n0 + L < T) ? (n0 + L) : T;

    float qv[4][4], m[4], l[4], acc[4][4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (size_t)j * m3tis_df::D + (kh * 4 + h) * m3tis_df::HD + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) qv[h][i] = m3tis_df::bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.0f;
    }

    for (int64_t n = n0; n < n1; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % m3tis_df::WIN);
            kr = ring_k + (size_t)slot * m3tis_df::KVW + kh * m3tis_df::HD + lane * 4;
            vr = ring_v + (size_t)slot * m3tis_df::KVW + kh * m3tis_df::HD + lane * 4;
        } else {
            const int64_t rb = n - n_ctx;
            kr = kb + (size_t)rb * m3tis_df::KVW + kh * m3tis_df::HD + lane * 4;
            vr = vb + (size_t)rb * m3tis_df::KVW + kh * m3tis_df::HD + lane * 4;
        }
        float kf[4], vf[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kf[i] = m3tis_df::bf(kr[i]);
            vf[i] = m3tis_df::bf(vr[i]);
        }
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn);
            const float pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }

#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const size_t at = ((size_t)j * m3tis_df::QH + kh * 4 + h) * m3tis_df::CH + c;
        float* pa = pacc + at * m3tis_df::HD + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) pa[i] = acc[h][i];
        if (lane == 0) {
            pm[at] = m[h];
            pl[at] = l[h];
        }
    }
}

// Merge CH=16 chunks into BF16 attention output [BLOCK, 32, 128].
// Launch: grid(BLOCK=8, QH=32, 1), block(HD=128, 1, 1)
__global__ void m3tis_df_attn_merge(
    const float* __restrict__ pm,
    const float* __restrict__ pl,
    const float* __restrict__ pacc,
    __nv_bfloat16* __restrict__ out
) {
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const size_t base = ((size_t)r * m3tis_df::QH + h) * m3tis_df::CH;
    float m = -INFINITY;
    for (int c = 0; c < m3tis_df::CH; ++c) m = fmaxf(m, pm[base + c]);
    float l = 0.0f;
    float a = 0.0f;
    for (int c = 0; c < m3tis_df::CH; ++c) {
        const float w = __expf(pm[base + c] - m);
        l += pl[base + c] * w;
        a += pacc[(base + c) * m3tis_df::HD + d] * w;
    }
    out[(size_t)r * m3tis_df::D + h * m3tis_df::HD + d] = m3tis_df::tobf(a / l);
}

// Top-16 FP32 logits per chunk of 4096 vocab columns.
// Launch: grid(rows, ceil(cols / 4096), 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_df_top16_part_f32(
    const float* __restrict__ logits,
    int32_t cols,
    float* __restrict__ pv,
    int32_t* __restrict__ pi
) {
    __shared__ float sv[m3tis_df::CHUNK_COLS];
    __shared__ int32_t si[m3tis_df::CHUNK_COLS];
    __shared__ float rv[8];
    __shared__ int32_t ri[16];
    const int r = blockIdx.x;
    const int c0 = blockIdx.y * m3tis_df::CHUNK_COLS;
    const int n = min(m3tis_df::CHUNK_COLS, cols - c0);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = logits[(size_t)r * cols + c0 + j];
        si[j] = c0 + j;
    }
    __syncthreads();
    const size_t base = ((size_t)r * gridDim.y + blockIdx.y) * m3tis_df::TOP;
    for (int t = 0; t < m3tis_df::TOP; ++t) {
        m3tis_df::take_best(sv, si, n, rv, ri, pv + base + t, pi + base + t);
    }
}

// Merge per-chunk top-16 candidates into final top-16 (ids[rows, 16], vals[rows, 16]).
// Launch: grid(rows, 1, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_df_top16_merge_f32(
    const float* __restrict__ pv,
    const int32_t* __restrict__ pi,
    int32_t chunks,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_vals
) {
    __shared__ float sv[64 * m3tis_df::TOP];
    __shared__ int32_t si[64 * m3tis_df::TOP];
    __shared__ float rv[8];
    __shared__ int32_t ri[16];
    __shared__ float ov[m3tis_df::TOP];
    __shared__ int32_t oi[m3tis_df::TOP];
    const int r = blockIdx.x;
    const int n = chunks * m3tis_df::TOP;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = pv[(size_t)r * n + j];
        si[j] = pi[(size_t)r * n + j];
    }
    __syncthreads();
    for (int t = 0; t < m3tis_df::TOP; ++t) {
        m3tis_df::take_best(sv, si, n, rv, ri, ov + t, oi + t);
    }
    if (threadIdx.x < m3tis_df::TOP) {
        out_ids[(size_t)r * m3tis_df::TOP + threadIdx.x] = oi[threadIdx.x];
        out_vals[(size_t)r * m3tis_df::TOP + threadIdx.x] = ov[threadIdx.x];
    }
}

} // extern "C"
