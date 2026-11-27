// Clean-room M3TiS sm_86 GLM-5.3-Flash Core Operations:
// - Embedding broadcast to S=4 mHC streams
// - mHC Hyper-Connections (hc_pre with 20-iter Sinkhorn doubly-stochastic 4x4 + fused RMSNorm, hc_post, stream_mean)
// - RMSNorm (any D multiple of 64: 4096, 1536, 512)
// - KDA Linear Attention (64 heads x 128 dim, 4-tap conv1d + SiLU, L2 q/k norm, delta-rule recurrence, gated o_norm)
// - DSA/MLA Latent Attention (absorb_q, sliding-window/linear KV write, online-softmax dense causal attention, expand_v)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace m3tis_glm {

__device__ __forceinline__ float bf( __nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float sigmoidf_safe(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float row4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v + __shfl_xor_sync(0xffffffffu, v, 1);
}

__device__ __forceinline__ float col4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    return v + __shfl_xor_sync(0xffffffffu, v, 4);
}

} // namespace m3tis_glm

extern "C" {

// Broadcast BF16 token embeddings [rows, D] -> S streams x[rows, S, D].
// Launch: grid(rows, (D + 255) / 256, 1), block(256, 1, 1)
__global__ void m3tis_embed_broadcast_bf16(
    const __nv_bfloat16* __restrict__ rows_in,
    __nv_bfloat16* __restrict__ x_out,
    int32_t rows,
    int32_t D,
    int32_t S
) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || d >= D) return;
    const __nv_bfloat16 v = rows_in[(size_t)r * D + d];
    for (int s = 0; s < S; ++s) {
        x_out[((size_t)r * S + s) * D + d] = v;
    }
}

// Generic RMSNorm: out[r, d] = bf16(w[d] * bf16(x[r, d] / sqrt(mean(x^2) + eps))).
// Launch: grid(rows, 1, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int32_t o_stride,
    int32_t rows,
    int32_t D,
    float eps
) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    __shared__ float smem[8];

    const __nv_bfloat16* xr = x + (size_t)r * x_stride;
    __nv_bfloat16* orow = out + (size_t)r * o_stride;

    float ss = 0.0f;
    for (int d = t; d < D; d += 256) {
        float v = m3tis_glm::bf(xr[d]);
        ss = fmaf(v, v, ss);
    }
    ss = m3tis_glm::warp_sum(ss);
    if (lane == 0) smem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? smem[lane] : 0.0f;
        tot = m3tis_glm::warp_sum(tot);
        if (lane == 0) smem[0] = rsqrtf(tot / (float)D + eps);
    }
    __syncthreads();
    const float rinv = smem[0];

    for (int d = t; d < D; d += 256) {
        float v = m3tis_glm::bf(xr[d]);
        float wf = m3tis_glm::bf(w[d]);
        orow[d] = __float2bfloat16_rn(wf * m3tis_glm::rbf(v * rinv));
    }
}

// mHC Pre Step Part 1: compute 24 dot products (hc_fn[24, 16384] . x[r, 16384]) + sum-of-squares ss[r].
// Milestone 6: 4-row tiled weight reuse -> loads `fn` ONCE per 4 rows (4x fewer HBM reads during prefill!).
// Launch: grid((rows + 3) / 4, nb, 1), block(256, 1, 1) -> writes part[rows, nb, 32] (24 dots + ss at [24]).
__global__ void __launch_bounds__(256) m3tis_hc_partial(
    const __nv_bfloat16* __restrict__ x,  // [rows, S*D] (16384)
    const __nv_bfloat16* __restrict__ fn, // [24, S*D]
    float* __restrict__ part,             // [rows, nb, 32]
    int32_t wide,
    int32_t nb,
    int32_t rows
) {
    const int r0 = blockIdx.x * 4;
    const int b = blockIdx.y;
    if (r0 >= rows) return;
    const int nr = min(4, rows - r0);
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;

    const int kb = wide / nb;
    const int base = b * kb;

    // 8 warps x 3 fn-rows each = 24 dot products across up to 4 input rows in parallel!
    float acc[4][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
    };
    float ss[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = lane * 8; k < kb; k += 256) {
        float wv[3][8];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int m = warp * 3 + i;
            const uint4 wu = __ldg(reinterpret_cast<const uint4*>(fn + (size_t)m * (size_t)wide + base + k));
            const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu);
#pragma unroll
            for (int j = 0; j < 8; ++j) wv[i][j] = m3tis_glm::bf(wh[j]);
        }
        for (int dr = 0; dr < nr; ++dr) {
            const uint4 xu = *reinterpret_cast<const uint4*>(x + (size_t)(r0 + dr) * (size_t)wide + base + k);
            const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
            float xv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                xv[j] = m3tis_glm::bf(xh[j]);
                if (warp == 0) ss[dr] = fmaf(xv[j], xv[j], ss[dr]);
            }
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                float d = acc[dr][i];
#pragma unroll
                for (int j = 0; j < 8; ++j) d = fmaf(xv[j], wv[i][j], d);
                acc[dr][i] = d;
            }
        }
    }

    for (int dr = 0; dr < nr; ++dr) {
        float* out = part + ((size_t)(r0 + dr) * nb + b) * 32;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            float d = m3tis_glm::warp_sum(acc[dr][i]);
            if (lane == 0) out[warp * 3 + i] = d;
        }
        if (warp == 0) {
            float s = m3tis_glm::warp_sum(ss[dr]);
            if (lane == 0) out[24] = s;
        }
    }
}

// mHC Pre Step Part 2: sum partials, 20-iter Sinkhorn doubly-stochastic 4x4, collapse 4 streams, fused RMSNorm!
// Matches `glm_glue_hc.cuh::finish` exactly (D=4096, S=4).
// Launch: grid(rows, 1, 1), block(256, 1, 1).
__global__ void __launch_bounds__(256) m3tis_hc_finish(
    const __nv_bfloat16* __restrict__ x,      // [rows, 4, D]
    const float* __restrict__ part,           // [rows, nb, 32]
    const float* __restrict__ base,           // [24]
    const float* __restrict__ scale,          // [3]
    const __nv_bfloat16* __restrict__ nw,     // [D]
    __nv_bfloat16* __restrict__ normed_out,   // [rows, D]
    float* __restrict__ post_out,             // [rows, 4]
    float* __restrict__ comb_out,             // [rows, 16]
    int32_t D,
    int32_t nb,
    float eps,
    float hc_eps,
    int32_t sinkhorn_iters
) {
    constexpr int S = 4;
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    __shared__ float pre[4];
    __shared__ float smem[8];

    if (warp == 0) {
        const float* p = part + (size_t)r * nb * 32;
        float mix = 0.0f;
        float ss = 0.0f;
        for (int b = 0; b < nb; ++b) {
            if (lane < 24) mix += p[b * 32 + lane];
            ss += p[b * 32 + 24];
        }
        const float rinv = 1.0f / sqrtf(eps + ss / (float)(S * D));
        mix *= rinv;

        const int k = lane & 3;
        const int idx = lane & 15;
        const float mpre  = __shfl_sync(0xffffffffu, mix, k);
        const float mpost = __shfl_sync(0xffffffffu, mix, 4 + k);
        const float mcomb = __shfl_sync(0xffffffffu, mix, 8 + idx);

        const float pr = hc_eps + m3tis_glm::sigmoidf_safe(fmaf(mpre, scale[0], base[k]));
        const float sp = m3tis_glm::sigmoidf_safe(fmaf(mpost, scale[1], base[4 + k]));
        const float cl = fmaf(mcomb, scale[2], base[8 + idx]);

        // Row-wise max across 4 columns (xor 2, xor 1)
        float cmax = fmaxf(cl, __shfl_xor_sync(0xffffffffu, cl, 2));
        cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, 1));
        const float ce = expf(cl - cmax);
        float cm = hc_eps + ce / m3tis_glm::row4(ce);
        cm = cm / (hc_eps + m3tis_glm::col4(cm));
        for (int it = 1; it < sinkhorn_iters; ++it) {
            cm = cm / (hc_eps + m3tis_glm::row4(cm));
            cm = cm / (hc_eps + m3tis_glm::col4(cm));
        }
        if (lane < 4) {
            post_out[r * S + lane] = sp + sp;
            pre[lane] = pr;
        }
        if (lane < 16) {
            comb_out[r * 16 + lane] = cm;
        }
    }
    __syncthreads();

    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    const __nv_bfloat16* xrow = x + (size_t)r * S * D;
    __nv_bfloat16* orow = normed_out + (size_t)r * D;

    // Fast single-pass 128-bit uint4 vectorized path for GLM-5.3 D == 4096 (256 threads x 2 chunks x 8 bf16 = 4096)!
    // Caches the 16 collapsed floats in registers so xrow[0..3] is read ONCE instead of twice (~2.5x faster!).
    if (D == 4096) {
        float c_reg[16];
        float acc = 0.0f;
#pragma unroll
        for (int ch = 0; ch < 2; ++ch) {
            const int d0 = ch * 2048 + t * 8;
            const uint4 u0 = *reinterpret_cast<const uint4*>(xrow + 0 * 4096 + d0);
            const uint4 u1 = *reinterpret_cast<const uint4*>(xrow + 1 * 4096 + d0);
            const uint4 u2 = *reinterpret_cast<const uint4*>(xrow + 2 * 4096 + d0);
            const uint4 u3 = *reinterpret_cast<const uint4*>(xrow + 3 * 4096 + d0);
            const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&u0);
            const __nv_bfloat16* h1 = reinterpret_cast<const __nv_bfloat16*>(&u1);
            const __nv_bfloat16* h2 = reinterpret_cast<const __nv_bfloat16*>(&u2);
            const __nv_bfloat16* h3 = reinterpret_cast<const __nv_bfloat16*>(&u3);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float c = fmaf(p0, m3tis_glm::bf(h0[j]),
                          fmaf(p1, m3tis_glm::bf(h1[j]),
                          fmaf(p2, m3tis_glm::bf(h2[j]),
                               p3 * m3tis_glm::bf(h3[j]))));
                c = m3tis_glm::rbf(c);
                c_reg[ch * 8 + j] = c;
                acc = fmaf(c, c, acc);
            }
        }
        acc = m3tis_glm::warp_sum(acc);
        if (lane == 0) smem[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 8) ? smem[lane] : 0.0f;
            tot = m3tis_glm::warp_sum(tot);
            if (lane == 0) smem[0] = 1.0f / sqrtf(eps + tot / 4096.0f);
        }
        __syncthreads();
        const float rinv = smem[0];

#pragma unroll
        for (int ch = 0; ch < 2; ++ch) {
            const int d0 = ch * 2048 + t * 8;
            const uint4 wu = __ldg(reinterpret_cast<const uint4*>(nw + d0));
            const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu);
            uint4 ou;
            __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ou);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                oh[j] = __float2bfloat16_rn(m3tis_glm::bf(wh[j]) * m3tis_glm::rbf(c_reg[ch * 8 + j] * rinv));
            }
            *reinterpret_cast<uint4*>(orow + d0) = ou;
        }
        return;
    }

    // Generic fallback for any non-4096 dimension (unit tests)
    float acc = 0.0f;
    for (int d = t; d < D; d += 256) {
        float c = fmaf(p0, m3tis_glm::bf(xrow[0 * D + d]),
                  fmaf(p1, m3tis_glm::bf(xrow[1 * D + d]),
                  fmaf(p2, m3tis_glm::bf(xrow[2 * D + d]),
                       p3 * m3tis_glm::bf(xrow[3 * D + d]))));
        c = m3tis_glm::rbf(c);
        acc = fmaf(c, c, acc);
    }
    acc = m3tis_glm::warp_sum(acc);
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? smem[lane] : 0.0f;
        tot = m3tis_glm::warp_sum(tot);
        if (lane == 0) smem[0] = 1.0f / sqrtf(eps + tot / (float)D);
    }
    __syncthreads();
    const float rinv = smem[0];

    for (int d = t; d < D; d += 256) {
        float c = fmaf(p0, m3tis_glm::bf(xrow[0 * D + d]),
                  fmaf(p1, m3tis_glm::bf(xrow[1 * D + d]),
                  fmaf(p2, m3tis_glm::bf(xrow[2 * D + d]),
                       p3 * m3tis_glm::bf(xrow[3 * D + d]))));
        c = m3tis_glm::rbf(c);
        float wf = m3tis_glm::bf(nw[d]);
        orow[d] = __float2bfloat16_rn(wf * m3tis_glm::rbf(c * rinv));
    }
}

// mHC Post Step: x[r, s, d] = bf16(post[r, s] * bf16(branch[r, d]) + sum_t comb[r, t*4 + s] * x[r, t, d]).
// Safe in-place (x_in == x_out).
// Launch: grid(rows, (D + 255) / 256, 1), block(256, 1, 1)
__global__ void m3tis_hc_post(
    __nv_bfloat16* __restrict__ x,          // [rows, 4, D] in-place
    const float* __restrict__ branch,       // [rows, D] fp32
    const float* __restrict__ post,         // [rows, 4]
    const float* __restrict__ comb,         // [rows, 16]
    int32_t rows,
    int32_t D
) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || d >= D) return;

    __nv_bfloat16* xrow = x + (size_t)r * 4 * D;
    float xv[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) xv[t] = m3tis_glm::bf(xrow[t * D + d]);
    const float br = m3tis_glm::rbf(branch[(size_t)r * D + d]);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float* cm = comb + r * 16 + s;
        float m = fmaf(xv[0], cm[0],
                  fmaf(xv[1], cm[4],
                  fmaf(xv[2], cm[8],
                       xv[3] * cm[12])));
        float o = fmaf(br, post[r * 4 + s], m);
        xrow[s * D + d] = __float2bfloat16_rn(o);
    }
}

// Stream mean: out[r, d] = bf16((x[r, 0, d] + x[r, 1, d] + x[r, 2, d] + x[r, 3, d]) / 4).
// Launch: grid(rows, (D + 255) / 256, 1), block(256, 1, 1)
__global__ void m3tis_stream_mean_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int32_t rows,
    int32_t D,
    int32_t S
) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || d >= D) return;
    const __nv_bfloat16* xrow = x + (size_t)r * S * D;
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) acc += m3tis_glm::bf(xrow[s * D + d]);
    out[(size_t)r * D + d] = __float2bfloat16_rn(acc / (float)S);
}

// KDA Linear Attention Chain (exact match to common/kda.cuh: 1024 threads/head, H=64, DK=DV=128, TAPS=4).
// P = [q(H*128) | k(H*128) | v(H*128) | f_a(128) | g_a(128) | b(H)], p_stride >= 3*H*128 + 256 + H.
// Launch: grid(H, 1, 1), block(1024, 1, 1)
__global__ void __launch_bounds__(1024) m3tis_kda_chain(
    int32_t H,
    const __nv_bfloat16* __restrict__ P,
    int32_t p_stride,
    int32_t b_off,
    const __nv_bfloat16* __restrict__ A,   // [rows, H*128] (f_b output)
    const __nv_bfloat16* __restrict__ G,   // [rows, H*128] (g_b output)
    const __nv_bfloat16* __restrict__ cs,  // [3, 3*H*128] conv state
    const float* __restrict__ cw,          // [3*H*128, 4] conv weights f32
    const float* __restrict__ state_in,    // [H, 128, 128] f32
    const float* __restrict__ a_log,       // [H] f32
    const float* __restrict__ dt_bias,     // [H*128] f32
    const __nv_bfloat16* __restrict__ norm_w, // [128] bf16
    float eps,
    float lower,
    int32_t rows,
    __nv_bfloat16* __restrict__ out,       // [rows, H*128] bf16
    float* __restrict__ state_out          // [H, 128, 128] f32 (may alias state_in)
) {
    constexpr int DK = 128, DV = 128, TAPS = 4;
    const int C = 3 * H * DK;
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;

    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV], gs[DK];
    __shared__ float beta_s, rinv;

    int c = -1;
    if (t < 3 * DK) c = (t / DK) * H * DK + h * DK + (t % DK);

    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 sv = *reinterpret_cast<const float4*>(&state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4]);
        s[j][0] = sv.x; s[j][1] = sv.y; s[j][2] = sv.z; s[j][3] = sv.w;
    }

    const float decay_rate = expf(a_log[h]);

    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float xv = (at < TAPS - 1)
                    ? m3tis_glm::bf(cs[(size_t)at * C + c])
                    : m3tis_glm::bf(P[(size_t)(at - (TAPS - 1)) * p_stride + c]);
                acc += cw[(size_t)c * TAPS + tap] * xv;
            }
            const float act = m3tis_glm::rbf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        } else if (t >= 512 && t < 512 + DK) {
            const int i = t - 512;
            const float a = m3tis_glm::bf(A[(size_t)r * H * DK + h * DK + i]) + dt_bias[h * DK + i];
            gs[i] = expf(lower * m3tis_glm::sigmoidf_safe(decay_rate * a));
        } else if (t == 1023) {
            beta_s = m3tis_glm::rbf(m3tis_glm::sigmoidf_safe(m3tis_glm::bf(P[(size_t)r * p_stride + b_off + h])));
        }
        __syncthreads();

        if (warp < 2) {
            float* xp = (warp == 0) ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v4[i] = xp[lane * 4 + i];
                ss += v4[i] * v4[i];
            }
            ss = m3tis_glm::warp_sum(ss);
            const float inv = 1.0f / sqrtf(ss + 1e-6f);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = v4[i] * inv;
                if (warp == 0) y *= (1.0f / sqrtf((float)DK));
                xp[lane * 4 + i] = y;
            }
        }
        __syncthreads();

        const float beta = beta_s;
        float kk[4], qq[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kk[i] = ks[lane * 4 + i];
            qq[i] = qs[lane * 4 + i];
            gg[i] = gs[lane * 4 + i];
        }
        // Delta-rule state update
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] *= gg[i];
                kv += s[j][i] * kk[i];
            }
            kv = m3tis_glm::warp_sum(kv);
            const float delta = (vs[warp * 4 + j] - kv) * beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] += kk[i] * delta;
        }
        // Readout y = s . q
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o += s[j][i] * qq[i];
            o = m3tis_glm::warp_sum(o);
            if (lane == 0) ys[warp * 4 + j] = m3tis_glm::rbf(o);
        }
        __syncthreads();

        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = ys[lane * 4 + i];
                ss += y * y;
            }
            ss = m3tis_glm::warp_sum(ss);
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        __syncthreads();

        if (t < DV) {
            const float yn = ys[t] * rinv;
            const float yw = m3tis_glm::bf(norm_w[t]) * yn;
            const float gate = m3tis_glm::bf(G[(size_t)r * H * DV + h * DV + t]);
            out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * m3tis_glm::sigmoidf_safe(gate));
        }
        __syncthreads();
    }

    if (state_out) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            *reinterpret_cast<float4*>(&state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4]) =
                make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
        }
    }
}

// KDA Linear Attention Chain WITH per-row (k, v, g, b, p_save) saves for exact speculative verify rollback/replay!
// Milestone 8: Supports 2-Branch Speculative Tree Verification (`tree_trunk_len > 0 && tree_b_start > 0`):
// - Saves fork KDA state into `state_out` (hot in L2, 0 extra regs!) at `r == tree_trunk_len - 1`,
// - Reloads fork state at `r == tree_b_start`, and maps Branch B conv taps back onto trunk rows!
// - Fused `p_save` write eliminates 34 `copy_bf16_strided` launches per verify round!
// Launch: grid(H, 1, 1), block(1024, 1, 1)
__global__ void __launch_bounds__(1024) m3tis_kda_chain_save(
    int32_t H,
    const __nv_bfloat16* __restrict__ P,
    int32_t p_stride,
    int32_t b_off,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ G,
    const __nv_bfloat16* __restrict__ cs,
    const float* __restrict__ cw,
    const float* __restrict__ state_in,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w,
    float eps,
    float lower,
    int32_t rows,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ state_out,
    float* __restrict__ k_save,         // [rows, H, 128] f32
    __nv_bfloat16* __restrict__ v_save, // [rows, H, 128] bf16
    float* __restrict__ g_save,         // [rows, H, 128] f32
    float* __restrict__ b_save,         // [rows, H] f32
    __nv_bfloat16* __restrict__ p_save, // nullable [rows, 3*H*128] bf16 (fused conv save!)
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    constexpr int DK = 128, DV = 128, TAPS = 4;
    const int C = 3 * H * DK;
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;

    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV], gs[DK];
    __shared__ float beta_s, rinv;

    int c = -1;
    if (t < 3 * DK) c = (t / DK) * H * DK + h * DK + (t % DK);

    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 sv = *reinterpret_cast<const float4*>(&state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4]);
        s[j][0] = sv.x; s[j][1] = sv.y; s[j][2] = sv.z; s[j][3] = sv.w;
    }

    const float decay_rate = expf(a_log[h]);

    for (int r = 0; r < rows; ++r) {
        if (tree_b_start > 0 && r == tree_b_start && state_out) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 sv = *reinterpret_cast<const float4*>(&state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4]);
                s[j][0] = sv.x; s[j][1] = sv.y; s[j][2] = sv.z; s[j][3] = sv.w;
            }
        }
        if (c >= 0) {
            if (p_save) {
                p_save[(size_t)r * C + c] = P[(size_t)r * p_stride + c];
            }
            float acc = 0.0f;
            if (tree_b_start > 0 && r >= tree_b_start) {
                const int r_eff = tree_trunk_len + (r - tree_b_start);
#pragma unroll
                for (int tap = 0; tap < TAPS; ++tap) {
                    const int at = r_eff + tap - (TAPS - 1);
                    float xv;
                    if (at < 0) {
                        xv = m3tis_glm::bf(cs[(size_t)(at + (TAPS - 1)) * C + c]);
                    } else if (at < tree_trunk_len) {
                        xv = m3tis_glm::bf(P[(size_t)at * p_stride + c]);
                    } else {
                        xv = m3tis_glm::bf(P[(size_t)(tree_b_start + (at - tree_trunk_len)) * p_stride + c]);
                    }
                    acc += cw[(size_t)c * TAPS + tap] * xv;
                }
            } else {
#pragma unroll
                for (int tap = 0; tap < TAPS; ++tap) {
                    const int at = r + tap;
                    const float xv = (at < TAPS - 1)
                        ? m3tis_glm::bf(cs[(size_t)at * C + c])
                        : m3tis_glm::bf(P[(size_t)(at - (TAPS - 1)) * p_stride + c]);
                    acc += cw[(size_t)c * TAPS + tap] * xv;
                }
            }
            const float act = m3tis_glm::rbf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        } else if (t >= 512 && t < 512 + DK) {
            const int i = t - 512;
            const float a = m3tis_glm::bf(A[(size_t)r * H * DK + h * DK + i]) + dt_bias[h * DK + i];
            gs[i] = expf(lower * m3tis_glm::sigmoidf_safe(decay_rate * a));
        } else if (t == 1023) {
            beta_s = m3tis_glm::rbf(m3tis_glm::sigmoidf_safe(m3tis_glm::bf(P[(size_t)r * p_stride + b_off + h])));
        }
        __syncthreads();

        if (warp < 2) {
            float* xp = (warp == 0) ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v4[i] = xp[lane * 4 + i];
                ss += v4[i] * v4[i];
            }
            ss = m3tis_glm::warp_sum(ss);
            const float inv = 1.0f / sqrtf(ss + 1e-6f);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = v4[i] * inv;
                if (warp == 0) y *= (1.0f / sqrtf((float)DK));
                xp[lane * 4 + i] = y;
            }
        }
        __syncthreads();

        const float beta = beta_s;
        if (k_save) {
            const size_t base = ((size_t)r * H + h) * DK;
            if (t < DK) { k_save[base + t] = ks[t]; g_save[base + t] = gs[t]; }
            if (t >= DK && t < DK + DV) v_save[base + t - DK] = __float2bfloat16_rn(vs[t - DK]);
            if (t == 0) b_save[(size_t)r * H + h] = beta;
        }

        float kk[4], qq[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kk[i] = ks[lane * 4 + i];
            qq[i] = qs[lane * 4 + i];
            gg[i] = gs[lane * 4 + i];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] *= gg[i];
                kv += s[j][i] * kk[i];
            }
            kv = m3tis_glm::warp_sum(kv);
            const float delta = (vs[warp * 4 + j] - kv) * beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] += kk[i] * delta;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o += s[j][i] * qq[i];
            o = m3tis_glm::warp_sum(o);
            if (lane == 0) ys[warp * 4 + j] = m3tis_glm::rbf(o);
        }
        __syncthreads();

        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = ys[lane * 4 + i];
                ss += y * y;
            }
            ss = m3tis_glm::warp_sum(ss);
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        __syncthreads();

        if (t < DV) {
            const float yn = ys[t] * rinv;
            const float yw = m3tis_glm::bf(norm_w[t]) * yn;
            const float gate = m3tis_glm::bf(G[(size_t)r * H * DV + h * DV + t]);
            out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * m3tis_glm::sigmoidf_safe(gate));
        }
        if (tree_b_start > 0 && r == tree_trunk_len - 1 && state_out) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                *reinterpret_cast<float4*>(&state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4]) =
                    make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
            }
        }
        __syncthreads();
    }

    if (state_out && tree_b_start == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            *reinterpret_cast<float4*>(&state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4]) =
                make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
        }
    }
}

// Replay `keep` accepted KDA steps across ALL KDA layers in ONE launch!
// Milestone 8: When `tree_b_start > 0`, replays `[0..tree_trunk_len)` then `[tree_b_start..tree_b_start+(keep-tree_trunk_len))`!
// Launch: grid(num_layers * H, 1, 1), block(1024, 1, 1)
__global__ void __launch_bounds__(1024) m3tis_kda_replay_layers(
    int32_t H,
    const float* __restrict__ state_in,
    int64_t state_stride,
    const float* __restrict__ k_save,
    const __nv_bfloat16* __restrict__ v_save,
    const float* __restrict__ g_save,
    const float* __restrict__ b_save,
    int64_t kv_stride,
    int64_t b_stride,
    int32_t keep,
    float* __restrict__ state_out,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    constexpr int DK = 128, DV = 128;
    const int layer = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;

    __shared__ float vs[DV];
    const float* sin = state_in + (size_t)layer * (size_t)state_stride;
    float* sout = state_out + (size_t)layer * (size_t)state_stride;
    const float* ks = k_save + (size_t)layer * (size_t)kv_stride;
    const __nv_bfloat16* vsv = v_save + (size_t)layer * (size_t)kv_stride;
    const float* gsv = g_save + (size_t)layer * (size_t)kv_stride;
    const float* bsv = b_save + (size_t)layer * (size_t)b_stride;

    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 sv = *reinterpret_cast<const float4*>(&sin[sbase + (size_t)(warp * 4 + j) * DK + lane * 4]);
        s[j][0] = sv.x; s[j][1] = sv.y; s[j][2] = sv.z; s[j][3] = sv.w;
    }

    for (int step = 0; step < keep; ++step) {
        const int r = (tree_b_start > 0 && step >= tree_trunk_len)
            ? (tree_b_start + (step - tree_trunk_len))
            : step;
        const size_t base = ((size_t)r * H + h) * DK;
        if (t < DV) vs[t] = m3tis_glm::bf(vsv[base + t]);
        __syncthreads();
        const float4 kv4 = *reinterpret_cast<const float4*>(&ks[base + lane * 4]);
        const float4 gv4 = *reinterpret_cast<const float4*>(&gsv[base + lane * 4]);
        const float kk[4] = {kv4.x, kv4.y, kv4.z, kv4.w};
        const float gg[4] = {gv4.x, gv4.y, gv4.z, gv4.w};
        const float beta = bsv[(size_t)r * H + h];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] *= gg[i];
                kv += s[j][i] * kk[i];
            }
            kv = m3tis_glm::warp_sum(kv);
            const float delta = (vs[warp * 4 + j] - kv) * beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] += kk[i] * delta;
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        *reinterpret_cast<float4*>(&sout[sbase + (size_t)(warp * 4 + j) * DK + lane * 4]) =
            make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
    }
}

// Copy the conv-relevant columns [0..C] of P[rows, p_stride] into p_save[rows, C] for deferred conv shift.
// Launch: grid(rows, (C + 255) / 256, 1), block(256, 1, 1)
__global__ void m3tis_copy_bf16_strided(
    const __nv_bfloat16* __restrict__ src,
    int32_t src_stride,
    __nv_bfloat16* __restrict__ dst,
    int32_t dst_stride,
    int32_t rows,
    int32_t cols
) {
    const int r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    dst[(size_t)r * dst_stride + c] = src[(size_t)r * src_stride + c];
}

// Shift ALL KDA layers' 3-row conv states in one launch using saved p_save[num_layers, max_rows, C].
// Milestone 8: Supports Branch B row remapping when `tree_b_start > 0`!
// Launch: grid(num_layers, (C + 255) / 256, 1), block(256, 1, 1)
__global__ void m3tis_kda_conv_shift_all(
    __nv_bfloat16* __restrict__ cs_all,       // [num_layers, 3, C]
    const __nv_bfloat16* __restrict__ p_save, // [num_layers, max_rows, C]
    int32_t keep,
    int32_t max_rows,
    int32_t C,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    const int l = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= C) return;
    __nv_bfloat16* cs = cs_all + (size_t)l * 3 * C;
    const __nv_bfloat16* P = p_save + (size_t)l * max_rows * C;
    __nv_bfloat16 v[3];
    for (int j = 0; j < 3; ++j) {
        const int src = keep + j;
        if (src < 3) {
            v[j] = cs[(size_t)src * C + c];
        } else {
            const int lr = src - 3;
            const int pr = (tree_b_start > 0 && lr >= tree_trunk_len)
                ? (tree_b_start + (lr - tree_trunk_len))
                : lr;
            v[j] = P[(size_t)pr * C + c];
        }
    }
    for (int j = 0; j < 3; ++j) cs[(size_t)j * C + c] = v[j];
}

// DFlash2 target tap: 4-stream mean of x[rows, 4, D] -> taps_out[rows, 5, D] at slot (0..4).
// Launch: grid(rows, (D + 255) / 256, 1), block(256, 1, 1)
__global__ void m3tis_tap_stream_mean(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ taps_out,
    int32_t rows,
    int32_t D,
    int32_t slot
) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || d >= D) return;
    const __nv_bfloat16* xr = x + (size_t)r * 4 * D;
    const float sum = m3tis_glm::bf(xr[d]) + m3tis_glm::bf(xr[D + d])
                    + m3tis_glm::bf(xr[2 * D + d]) + m3tis_glm::bf(xr[3 * D + d]);
    taps_out[((size_t)r * 5 + slot) * D + d] = __float2bfloat16_rn(sum * 0.25f);
}

// Fast per-row FP32 argmax over logits[rows, V] -> out_ids[rows], out_vals[rows].
// Launch: grid(rows, 1, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_argmax_f32(
    const float* __restrict__ logits,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_vals,
    int32_t rows,
    int32_t V
) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    __shared__ float rv[8];
    __shared__ int32_t ri[8];

    const float* row = logits + (size_t)r * V;
    float best_v = -1e38f;
    int32_t best_i = 0;
    for (int i = t; i < V; i += 256) {
        float v = row[i];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, o);
        int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, o);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0) {
        rv[warp] = best_v;
        ri[warp] = best_i;
    }
    __syncthreads();
    if (t == 0) {
        float bv = rv[0];
        int32_t bi = ri[0];
        for (int w = 1; w < 8; ++w) {
            if (rv[w] > bv || (rv[w] == bv && ri[w] < bi)) {
                bv = rv[w];
                bi = ri[w];
            }
        }
        out_ids[r] = bi;
        if (out_vals) out_vals[r] = bv;
    }
}

// Per-row FP32 argmax + exact top-1 softmax probability over logits[rows, V] (powers confidence-gated self-spec!).
// Launch: grid(rows, 1, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_argmax_prob_f32(
    const float* __restrict__ logits,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_probs,
    int32_t rows,
    int32_t V
) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    __shared__ float rv[8];
    __shared__ int32_t ri[8];
    __shared__ float s_max;

    const float* row = logits + (size_t)r * V;
    float best_v = -1e38f;
    int32_t best_i = 0;
    for (int i = t; i < V; i += 256) {
        float v = row[i];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, o);
        int32_t oi = __shfl_xor_sync(0xffffffffu, best_i, o);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0) {
        rv[warp] = best_v;
        ri[warp] = best_i;
    }
    __syncthreads();
    if (t == 0) {
        float bv = rv[0];
        int32_t bi = ri[0];
        for (int w = 1; w < 8; ++w) {
            if (rv[w] > bv || (rv[w] == bv && ri[w] < bi)) {
                bv = rv[w];
                bi = ri[w];
            }
        }
        out_ids[r] = bi;
        s_max = bv;
    }
    __syncthreads();
    const float mx = s_max;
    float sum_exp = 0.0f;
    for (int i = t; i < V; i += 256) {
        sum_exp += expf(row[i] - mx);
    }
    sum_exp = m3tis_glm::warp_sum(sum_exp);
    if (lane == 0) rv[warp] = sum_exp;
    __syncthreads();
    if (warp == 0) {
        float tot = (lane < 8) ? rv[lane] : 0.0f;
        tot = m3tis_glm::warp_sum(tot);
        if (lane == 0 && out_probs) {
            out_probs[r] = (tot > 0.0f) ? (1.0f / tot) : 0.0f;
        }
    }
}

// Shift KDA 3-row causal conv window by `rows` committed tokens: new_cs[0..3] = tail of [old_cs(3); P(rows)].
// Launch: grid((C + 255) / 256, 1, 1), block(256, 1, 1)
__global__ void m3tis_kda_conv_shift(
    __nv_bfloat16* __restrict__ cs,       // [3, C] in-place
    const __nv_bfloat16* __restrict__ P,  // [rows, p_stride]
    int32_t rows,
    int32_t p_stride,
    int32_t C
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    __nv_bfloat16 v[3];
    for (int j = 0; j < 3; ++j) {
        const int src = rows + j;
        v[j] = (src < 3) ? cs[(size_t)src * C + c] : P[(size_t)(src - 3) * p_stride + c];
    }
    for (int j = 0; j < 3; ++j) cs[(size_t)j * C + c] = v[j];
}

// MLA Absorb Q (128-bit uint4 coalesced on transposed wk_t[heads, lw, qk], 16-row tiled weight reuse!):
// qa[r, h, lw] = sum_{k=0..qk-1} q[r, h, k] * wk_t[h, lw, k]  (qk=256, lw=512).
// Launch: grid(((rows + 15) / 16) * heads, lw / 4, 1), block(128, 1, 1) -> 4 warps = 4 lw outputs x 16 rows/block!
__global__ void __launch_bounds__(128) m3tis_mla_absorb_q(
    const __nv_bfloat16* __restrict__ q,    // [rows, heads, qk]
    const __nv_bfloat16* __restrict__ wk_t, // [heads, lw, qk] (transposed at load for coalesced 128-bit reads!)
    __nv_bfloat16* __restrict__ qa,         // [rows, heads, lw]
    int32_t rows,
    int32_t heads,
    int32_t qk,
    int32_t lw
) {
    const int bh = blockIdx.x;
    const int r0 = (bh / heads) * 16;
    const int h = bh % heads;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.y * 4 + warp;
    if (r0 >= rows || col >= lw) return;
    const int nr = min(16, rows - r0);
    const __nv_bfloat16* wr = wk_t + ((size_t)h * lw + col) * qk;

    if (qk == 256) {
        const int k = lane * 8;
        const uint4 wu = __ldg(reinterpret_cast<const uint4*>(wr + k));
        const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu);
        float wv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) wv[j] = m3tis_glm::bf(wh[j]);
        for (int dr = 0; dr < nr; ++dr) {
            const int r = r0 + dr;
            const uint4 qu = *reinterpret_cast<const uint4*>(q + ((size_t)r * heads + h) * 256 + k);
            const __nv_bfloat16* qh = reinterpret_cast<const __nv_bfloat16*>(&qu);
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc = fmaf(m3tis_glm::bf(qh[j]), wv[j], acc);
            acc = m3tis_glm::warp_sum(acc);
            if (lane == 0) qa[((size_t)r * heads + h) * lw + col] = __float2bfloat16_rn(acc);
        }
        return;
    }

    for (int dr = 0; dr < nr; ++dr) {
        const int r = r0 + dr;
        const __nv_bfloat16* qr = q + ((size_t)r * heads + h) * qk;
        float acc = 0.0f;
        for (int k = lane * 8; k < qk; k += 256) {
            const uint4 qu = *reinterpret_cast<const uint4*>(qr + k);
            const uint4 wu = __ldg(reinterpret_cast<const uint4*>(wr + k));
            const __nv_bfloat16* qh = reinterpret_cast<const __nv_bfloat16*>(&qu);
            const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc = fmaf(m3tis_glm::bf(qh[j]), m3tis_glm::bf(wh[j]), acc);
        }
        acc = m3tis_glm::warp_sum(acc);
        if (lane == 0) qa[((size_t)r * heads + h) * lw + col] = __float2bfloat16_rn(acc);
    }
}

// MLA Write Latent KV Cache with built-in ring-buffer sliding window support (capacity = max_ctx).
// Launch: grid(rows, (lw + 255) / 256, 1), block(256, 1, 1)
__global__ void m3tis_mla_write_kv(
    const __nv_bfloat16* __restrict__ lat,  // [rows, lw]
    __nv_bfloat16* __restrict__ kv_cache,   // [capacity, lw]
    int32_t pos,
    int32_t capacity,
    int32_t rows,
    int32_t lw
) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || d >= lw) return;
    const int slot = (pos + r) % capacity;
    kv_cache[(size_t)slot * lw + d] = lat[(size_t)r * lw + d];
}

// MLA Dense Causal Latent Attention (online softmax over latent KV cache, supports sliding window + 2-Branch Tree!).
// Fast path (lw == 512, heads % 8 == 0): 8 warps = 8 heads of row `r` share 8 cached KV vectors in __shared__
// loaded via 128-bit uint4 (8x fewer KV reads, 16x fewer __syncthreads(), pure warp_sum dot products!).
// Launch: grid(rows, (lw == 512 && (heads & 7) == 0) ? (heads / 8) : heads, 1), block(256, 1, 1).
__global__ void __launch_bounds__(256) m3tis_mla_attend_dense(
    const __nv_bfloat16* __restrict__ qa,       // [rows, heads, lw]
    const __nv_bfloat16* __restrict__ kv_cache, // [capacity, lw]
    __nv_bfloat16* __restrict__ ol,             // [rows, heads, lw]
    int32_t pos,
    int32_t capacity,
    int32_t window,                             // sliding window size (<= capacity, e.g. 2048 or capacity)
    int32_t rows,
    int32_t heads,
    int32_t lw,
    float scale,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const bool is_branch_b = (tree_b_start > 0 && r >= tree_b_start);
    const int skip_lo = pos + tree_trunk_len;
    const int skip_hi = pos + tree_b_start;

    if (lw == 512 && (heads & 7) == 0) {
        const int h = blockIdx.y * 8 + warp;
        if (r >= rows || h >= heads) return;
        __shared__ __align__(16) __nv_bfloat16 k_sh[8][512];
        __shared__ int k_valid[8];

        const __nv_bfloat16* qrow = qa + ((size_t)r * heads + h) * 512;
        const int d0 = lane * 8;
        const uint4 qu0 = *reinterpret_cast<const uint4*>(qrow + d0);
        const uint4 qu1 = *reinterpret_cast<const uint4*>(qrow + 256 + d0);
        const __nv_bfloat16* qh0 = reinterpret_cast<const __nv_bfloat16*>(&qu0);
        const __nv_bfloat16* qh1 = reinterpret_cast<const __nv_bfloat16*>(&qu1);
        float q0[8], q1[8], o0[8], o1[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q0[j] = m3tis_glm::bf(qh0[j]);
            q1[j] = m3tis_glm::bf(qh1[j]);
            o0[j] = 0.0f;
            o1[j] = 0.0f;
        }
        float m_max = -1e30f;
        float l_sum = 0.0f;

        const int q_pos = pos + r;
        const int k_start = max(0, q_pos - window + 1);
        const int total_k = q_pos - k_start + 1;

        for (int kb = 0; kb < total_k; kb += 8) {
            const int n_k = min(8, total_k - kb);
            if (warp < n_k) {
                const int kp = k_start + kb + warp;
                const bool skip = is_branch_b && (kp >= skip_lo && kp < skip_hi);
                if (!skip) {
                    const int slot = kp % capacity;
                    const __nv_bfloat16* ksrc = kv_cache + (size_t)slot * 512;
                    *reinterpret_cast<uint4*>(&k_sh[warp][d0]) =
                        __ldg(reinterpret_cast<const uint4*>(ksrc + d0));
                    *reinterpret_cast<uint4*>(&k_sh[warp][256 + d0]) =
                        __ldg(reinterpret_cast<const uint4*>(ksrc + 256 + d0));
                    if (lane == 0) k_valid[warp] = 1;
                } else {
                    if (lane == 0) k_valid[warp] = 0;
                }
            }
            __syncthreads();

            for (int kk = 0; kk < n_k; ++kk) {
                if (!k_valid[kk]) continue;
                const uint4 ku0 = *reinterpret_cast<const uint4*>(&k_sh[kk][d0]);
                const uint4 ku1 = *reinterpret_cast<const uint4*>(&k_sh[kk][256 + d0]);
                const __nv_bfloat16* kh0 = reinterpret_cast<const __nv_bfloat16*>(&ku0);
                const __nv_bfloat16* kh1 = reinterpret_cast<const __nv_bfloat16*>(&ku1);
                float k0[8], k1[8];
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    k0[j] = m3tis_glm::bf(kh0[j]);
                    k1[j] = m3tis_glm::bf(kh1[j]);
                    dot = fmaf(q0[j], k0[j], fmaf(q1[j], k1[j], dot));
                }
                const float score = m3tis_glm::warp_sum(dot) * scale;
                const float m_new = fmaxf(m_max, score);
                const float alpha = expf(m_max - m_new);
                const float p = expf(score - m_new);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    o0[j] = fmaf(p, k0[j], o0[j] * alpha);
                    o1[j] = fmaf(p, k1[j], o1[j] * alpha);
                }
                l_sum = fmaf(l_sum, alpha, p);
                m_max = m_new;
            }
            __syncthreads();
        }

        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        uint4 ou0, ou1;
        __nv_bfloat16* oh0 = reinterpret_cast<__nv_bfloat16*>(&ou0);
        __nv_bfloat16* oh1 = reinterpret_cast<__nv_bfloat16*>(&ou1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            oh0[j] = __float2bfloat16_rn(o0[j] * inv_l);
            oh1[j] = __float2bfloat16_rn(o1[j] * inv_l);
        }
        __nv_bfloat16* orow = ol + ((size_t)r * heads + h) * 512;
        *reinterpret_cast<uint4*>(orow + d0) = ou0;
        *reinterpret_cast<uint4*>(orow + 256 + d0) = ou1;
        return;
    }

    const int h = blockIdx.y;
    if (r >= rows || h >= heads) return;
    __shared__ float smem[8];
    __shared__ float s_score;

    const __nv_bfloat16* qrow = qa + ((size_t)r * heads + h) * lw;
    float q0 = (t < lw) ? m3tis_glm::bf(qrow[t]) : 0.0f;
    float q1 = (t + 256 < lw) ? m3tis_glm::bf(qrow[t + 256]) : 0.0f;
    float o0 = 0.0f, o1 = 0.0f;
    float m_max = -1e30f;
    float l_sum = 0.0f;

    const int q_pos = pos + r;
    const int k_start = max(0, q_pos - window + 1);

    for (int kp = k_start; kp <= q_pos; ++kp) {
        if (is_branch_b && kp >= skip_lo && kp < skip_hi) continue;
        const int slot = kp % capacity;
        const __nv_bfloat16* krow = kv_cache + (size_t)slot * lw;
        float k0 = (t < lw) ? m3tis_glm::bf(krow[t]) : 0.0f;
        float k1 = (t + 256 < lw) ? m3tis_glm::bf(krow[t + 256]) : 0.0f;

        float dot = fmaf(q0, k0, q1 * k1);
        dot = m3tis_glm::warp_sum(dot);
        if (lane == 0) smem[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 8) ? smem[lane] : 0.0f;
            tot = m3tis_glm::warp_sum(tot);
            if (lane == 0) s_score = tot * scale;
        }
        __syncthreads();
        const float score = s_score;
        __syncthreads();

        const float m_new = fmaxf(m_max, score);
        const float alpha = expf(m_max - m_new);
        const float p = expf(score - m_new);
        o0 = fmaf(p, k0, o0 * alpha);
        o1 = fmaf(p, k1, o1 * alpha);
        l_sum = fmaf(l_sum, alpha, p);
        m_max = m_new;
    }

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    __nv_bfloat16* orow = ol + ((size_t)r * heads + h) * lw;
    if (t < lw) orow[t] = __float2bfloat16_rn(o0 * inv_l);
    if (t + 256 < lw) orow[t + 256] = __float2bfloat16_rn(o1 * inv_l);
}

// MLA Expand V (16-row tiled register-cached weight reuse -> 16x fewer HBM weight reads during prefill!):
// vn[r, h, dv] = sum_{l=0..lw-1} ol[r, h, l] * wv[h, dv, l]  (dv=256, lw=512).
// Launch: grid(((rows + 15) / 16) * heads, dv / 4, 1), block(128, 1, 1) -> 4 warps = 4 dv outputs x 16 rows/block!
__global__ void __launch_bounds__(128) m3tis_mla_expand_v(
    const __nv_bfloat16* __restrict__ ol,  // [rows, heads, lw]
    const __nv_bfloat16* __restrict__ wv,  // [heads, dv, lw]
    __nv_bfloat16* __restrict__ vn,        // [rows, heads, dv]
    int32_t rows,
    int32_t heads,
    int32_t dv,
    int32_t lw
) {
    const int bh = blockIdx.x;
    const int r0 = (bh / heads) * 16;
    const int h = bh % heads;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.y * 4 + warp;
    if (r0 >= rows || col >= dv) return;
    const int nr = min(16, rows - r0);
    const __nv_bfloat16* wrow = wv + ((size_t)h * dv + col) * lw;

    if (lw == 512) {
        const int l0 = lane * 8;
        const uint4 wu0 = __ldg(reinterpret_cast<const uint4*>(wrow + l0));
        const uint4 wu1 = __ldg(reinterpret_cast<const uint4*>(wrow + 256 + l0));
        const __nv_bfloat16* wh0 = reinterpret_cast<const __nv_bfloat16*>(&wu0);
        const __nv_bfloat16* wh1 = reinterpret_cast<const __nv_bfloat16*>(&wu1);
        float w0[8], w1[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w0[j] = m3tis_glm::bf(wh0[j]);
            w1[j] = m3tis_glm::bf(wh1[j]);
        }
        for (int dr = 0; dr < nr; ++dr) {
            const int r = r0 + dr;
            const __nv_bfloat16* orow = ol + ((size_t)r * heads + h) * 512;
            const uint4 ou0 = *reinterpret_cast<const uint4*>(orow + l0);
            const uint4 ou1 = *reinterpret_cast<const uint4*>(orow + 256 + l0);
            const __nv_bfloat16* oh0 = reinterpret_cast<const __nv_bfloat16*>(&ou0);
            const __nv_bfloat16* oh1 = reinterpret_cast<const __nv_bfloat16*>(&ou1);
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc = fmaf(m3tis_glm::bf(oh0[j]), w0[j], acc);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc = fmaf(m3tis_glm::bf(oh1[j]), w1[j], acc);
            acc = m3tis_glm::warp_sum(acc);
            if (lane == 0) vn[((size_t)r * heads + h) * dv + col] = __float2bfloat16_rn(acc);
        }
        return;
    }

    for (int dr = 0; dr < nr; ++dr) {
        const int r = r0 + dr;
        const __nv_bfloat16* orow = ol + ((size_t)r * heads + h) * lw;
        float acc = 0.0f;
        for (int l = lane * 8; l < lw; l += 256) {
            const uint4 ou = *reinterpret_cast<const uint4*>(orow + l);
            const uint4 wu = __ldg(reinterpret_cast<const uint4*>(wrow + l));
            const __nv_bfloat16* oh = reinterpret_cast<const __nv_bfloat16*>(&ou);
            const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc = fmaf(m3tis_glm::bf(oh[j]), m3tis_glm::bf(wh[j]), acc);
        }
        acc = m3tis_glm::warp_sum(acc);
        if (lane == 0) vn[((size_t)r * heads + h) * dv + col] = __float2bfloat16_rn(acc);
    }
}

// =================================================================================================
// Milestone 3C: Sparse Top-2048 DSA Indexer (GLM-5.3-Flash: 32 heads, D=128, kpool=4, topk=512 pools)
// =================================================================================================

// 1. LayerNorm indexer keys k_raw[rows, 128] -> ik[pos + r, 128] (BF16) and store gate g_raw -> ig[pos + r, 128].
// Launch: grid(rows, 1, 1), block(32, 1, 1)
__global__ void __launch_bounds__(32) m3tis_dsa_index_write(
    const __nv_bfloat16* __restrict__ k_raw, // [rows, 128]
    const float* __restrict__ g_raw,         // [rows, 128]
    const __nv_bfloat16* __restrict__ ln_w,  // [128]
    const __nv_bfloat16* __restrict__ ln_b,  // [128]
    __nv_bfloat16* __restrict__ ik_cache,    // [max_ctx, 128]
    __nv_bfloat16* __restrict__ ig_cache,    // [max_ctx, 128]
    int32_t pos,
    int32_t capacity,
    float eps
) {
    constexpr int D = 128;
    const int r = blockIdx.x;
    const int slot = (pos + r) % capacity;
    const int d0 = threadIdx.x * 4;

    float x[4];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        x[j] = m3tis_glm::bf(k_raw[(size_t)r * D + d0 + j]);
        s += x[j];
    }
    s = m3tis_glm::warp_sum(s);
    const float mean = s / (float)D;

    float xc[4];
    float v = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        xc[j] = x[j] - mean;
        v = fmaf(xc[j], xc[j], v);
    }
    v = m3tis_glm::warp_sum(v);
    const float inv_sd = rsqrtf(v / (float)D + eps);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = d0 + j;
        const float y = fmaf(xc[j] * inv_sd, m3tis_glm::bf(ln_w[d]), m3tis_glm::bf(ln_b[d]));
        ik_cache[(size_t)slot * D + d] = __float2bfloat16_rn(y);
        ig_cache[(size_t)slot * D + d] = __float2bfloat16_rn(g_raw[(size_t)r * D + d]);
    }
}

// 2. Compress every completed 4-token pool into pk_cache[p % pk_cap, 128] via softmax(ig + ape) weighted sum of ik.
//    Milestone 8: Uses compact ring capacity `ik_cap` (e.g. 264 slots = 1.4 MiB vs 704 MiB at 128K, 100% bit-exact!).
// Launch: grid(rows / 4 + 2, 1, 1), block(32, 1, 1)
__global__ void __launch_bounds__(32) m3tis_dsa_pool_keys(
    const __nv_bfloat16* __restrict__ ik_cache, // [ik_cap, 128]
    const __nv_bfloat16* __restrict__ ig_cache, // [ik_cap, 128]
    const __nv_bfloat16* __restrict__ ape,      // [4, 128]
    __nv_bfloat16* __restrict__ pk_cache,       // [pk_cap, 128]
    int32_t pos,
    int32_t rows,
    int32_t ik_cap,
    int32_t pk_cap
) {
    constexpr int D = 128;
    const int p = (pos / 4) + blockIdx.x;
    if (4 * p + 3 > pos + rows - 1) return;
    const int p_slot = p % pk_cap;
    const int d0 = threadIdx.x * 4;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = d0 + j;
        float l[4], e[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int s_tok = (4 * p + q) % ik_cap;
            l[q] = m3tis_glm::bf(ig_cache[(size_t)s_tok * D + d]) + m3tis_glm::bf(ape[(size_t)q * D + d]);
        }
        const float m = fmaxf(fmaxf(l[0], l[1]), fmaxf(l[2], l[3]));
        float sum_e = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            e[q] = expf(l[q] - m);
            sum_e += e[q];
        }
        float acc = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int s_tok = (4 * p + q) % ik_cap;
            const float w = m3tis_glm::rbf(e[q] / sum_e);
            acc += m3tis_glm::rbf(w * m3tis_glm::bf(ik_cache[(size_t)s_tok * D + d]));
        }
        pk_cache[(size_t)p_slot * D + d] = __float2bfloat16_rn(acc);
    }
}

// 3. Score all visible compressed pools for each query row (4-row tiled pk_cache reuse -> 4x faster at 4K-16K!):
//    score[r, p] = sum_{h=0..H-1} (w_raw[r, h] * wscale) * max(0, dot(qi[r, h], pk[p]) * qscale).
// Launch: grid((rows + 3) / 4, (np_stride + 7) / 8, 1), block(256, 1, 1) -> 8 warps = 8 pools x 4 rows/CTA!
__global__ void __launch_bounds__(256) m3tis_dsa_score_pools(
    const __nv_bfloat16* __restrict__ qi,       // [rows, H, 128]
    const __nv_bfloat16* __restrict__ w_raw,    // [rows, H]
    const __nv_bfloat16* __restrict__ pk_cache, // [pk_cap, 128]
    float* __restrict__ scores,                 // [rows, np_stride]
    int32_t pos,
    int32_t np_stride,
    int32_t pk_cap,
    int32_t heads,
    float qscale,
    float wscale,
    int32_t rows
) {
    constexpr int D = 128;
    const int r0 = blockIdx.x * 4;
    if (r0 >= rows) return;
    const int nr = min(4, rows - r0);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int p = blockIdx.y * 8 + warp;
    if (p >= np_stride) return;

    const int max_npool = (pos + r0 + nr) / 4;
    if (p >= max_npool) {
        if (lane == 0) {
            for (int dr = 0; dr < nr; ++dr) scores[(size_t)(r0 + dr) * np_stride + p] = -INFINITY;
        }
        return;
    }

    const int p_slot = p % pk_cap;
    const __nv_bfloat16* pk = pk_cache + (size_t)p_slot * D + lane * 4;
    const uint2 ku = __ldg(reinterpret_cast<const uint2*>(pk));
    const __nv_bfloat16* kh = reinterpret_cast<const __nv_bfloat16*>(&ku);
    const float k0 = m3tis_glm::bf(kh[0]);
    const float k1 = m3tis_glm::bf(kh[1]);
    const float k2 = m3tis_glm::bf(kh[2]);
    const float k3 = m3tis_glm::bf(kh[3]);

    for (int dr = 0; dr < nr; ++dr) {
        const int r = r0 + dr;
        const int npool = (pos + r + 1) / 4;
        if (p >= npool) {
            if (lane == 0) scores[(size_t)r * np_stride + p] = -INFINITY;
            continue;
        }
        const __nv_bfloat16* qbase = qi + (size_t)r * heads * D + lane * 4;
        const __nv_bfloat16* wrow = w_raw + (size_t)r * heads;
        float total = 0.0f;
        for (int h = 0; h < heads; ++h) {
            const uint2 qu = *reinterpret_cast<const uint2*>(qbase + (size_t)h * D);
            const __nv_bfloat16* qh = reinterpret_cast<const __nv_bfloat16*>(&qu);
            float dot = fmaf(m3tis_glm::bf(qh[0]), k0,
                        fmaf(m3tis_glm::bf(qh[1]), k1,
                        fmaf(m3tis_glm::bf(qh[2]), k2,
                             m3tis_glm::bf(qh[3]) * k3)));
            dot = m3tis_glm::warp_sum(dot);
            if (lane == 0) {
                const float relu = fmaxf(dot * qscale, 0.0f);
                const float wh = m3tis_glm::bf(wrow[h]) * wscale;
                total = fmaf(wh, relu, total);
            }
        }
        if (lane == 0) scores[(size_t)r * np_stride + p] = total;
    }
}

// 3b. Milestone 7: Ampere sm_86 BF16 Tensor Core DSA Pool Scoring (`heads == 32, D == 128`)!
//     Computes all 32 heads x 32 pools per CTA using mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32 +
//     register-only 32-head reduction (0 smem reduction, ~8-12x faster at 32K-64K context!).
// Launch: grid(rows, (np_stride + 31) / 32, 1), block(128, 1, 1) -> 4 warps = 32 pools/CTA
__global__ void __launch_bounds__(128) m3tis_dsa_score_pools_tc(
    const __nv_bfloat16* __restrict__ qi,       // [rows, 32, 128]
    const __nv_bfloat16* __restrict__ w_raw,    // [rows, 32]
    const __nv_bfloat16* __restrict__ pk_cache, // [pk_cap, 128]
    float* __restrict__ scores,                 // [rows, np_stride]
    int32_t pos,
    int32_t np_stride,
    int32_t pk_cap,
    int32_t heads,
    float qscale,
    float wscale,
    int32_t rows
) {
    (void)heads;
    const int r = blockIdx.x;
    const int p_blk = blockIdx.y * 32;
    if (r >= rows || p_blk >= np_stride) return;

    const int tid = threadIdx.x;
    const int npool = (pos + r + 1) / 4;
    if (p_blk >= npool) {
        if (tid < 32 && p_blk + tid < np_stride) {
            scores[(size_t)r * np_stride + p_blk + tid] = -INFINITY;
        }
        return;
    }

    __shared__ __align__(16) __nv_bfloat16 sh_q[32][136]; // 8.5 KiB, stride 272B -> 0 bank conflicts!
    __shared__ float sh_w[32];

    const int lq_c = (tid & 15) * 8; // 0, 8, ..., 120
    const int lq_h0 = tid >> 4;      // 0..7
    const __nv_bfloat16* qrow = qi + (size_t)r * 32 * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int hh = i * 8 + lq_h0;
        *reinterpret_cast<uint4*>(&sh_q[hh][lq_c]) =
            *reinterpret_cast<const uint4*>(qrow + (size_t)hh * 128 + lq_c);
    }
    if (tid < 32) {
        sh_w[tid] = m3tis_glm::bf(w_raw[(size_t)r * 32 + tid]) * wscale;
    }
    __syncthreads();

    const int warp = tid >> 5; // 0..3
    const int lane = tid & 31;
    const int g = lane >> 2;   // 0..7
    const int t = lane & 3;    // 0..3
    const int p0 = p_blk + warp * 8;
    if (p0 >= np_stride) return;
    if (p0 >= npool) {
        if (lane < 8 && p0 + lane < np_stride) {
            scores[(size_t)r * np_stride + p0 + lane] = -INFINITY;
        }
        return;
    }

    const int p_mine = p0 + g;
    const __nv_bfloat16* pk_row =
        (p_mine < npool && p_mine < np_stride)
            ? (pk_cache + (size_t)(p_mine % pk_cap) * 128)
            : nullptr;

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // heads 0..15
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // heads 16..31

#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int k0 = ks * 16 + 2 * t;
        const uint32_t b0 = pk_row ? __ldg(reinterpret_cast<const uint32_t*>(pk_row + k0)) : 0u;
        const uint32_t b1 = pk_row ? __ldg(reinterpret_cast<const uint32_t*>(pk_row + k0 + 8)) : 0u;

        const uint32_t a00 = *reinterpret_cast<const uint32_t*>(&sh_q[g][k0]);
        const uint32_t a01 = *reinterpret_cast<const uint32_t*>(&sh_q[g + 8][k0]);
        const uint32_t a02 = *reinterpret_cast<const uint32_t*>(&sh_q[g][k0 + 8]);
        const uint32_t a03 = *reinterpret_cast<const uint32_t*>(&sh_q[g + 8][k0 + 8]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "r"(a00), "r"(a01), "r"(a02), "r"(a03), "r"(b0), "r"(b1)
        );

        const uint32_t a10 = *reinterpret_cast<const uint32_t*>(&sh_q[16 + g][k0]);
        const uint32_t a11 = *reinterpret_cast<const uint32_t*>(&sh_q[24 + g][k0]);
        const uint32_t a12 = *reinterpret_cast<const uint32_t*>(&sh_q[16 + g][k0 + 8]);
        const uint32_t a13 = *reinterpret_cast<const uint32_t*>(&sh_q[24 + g][k0 + 8]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
            : "r"(a10), "r"(a11), "r"(a12), "r"(a13), "r"(b0), "r"(b1)
        );
    }

    const float w0 = sh_w[g];
    const float w1 = sh_w[g + 8];
    const float w2 = sh_w[16 + g];
    const float w3 = sh_w[24 + g];
    float s_even = fmaf(w0, fmaxf(acc0[0] * qscale, 0.0f),
                   fmaf(w1, fmaxf(acc0[2] * qscale, 0.0f),
                   fmaf(w2, fmaxf(acc1[0] * qscale, 0.0f),
                        w3 * fmaxf(acc1[2] * qscale, 0.0f))));
    float s_odd  = fmaf(w0, fmaxf(acc0[1] * qscale, 0.0f),
                   fmaf(w1, fmaxf(acc0[3] * qscale, 0.0f),
                   fmaf(w2, fmaxf(acc1[1] * qscale, 0.0f),
                        w3 * fmaxf(acc1[3] * qscale, 0.0f))));

    s_even += __shfl_xor_sync(0xffffffffu, s_even, 16);
    s_even += __shfl_xor_sync(0xffffffffu, s_even, 8);
    s_even += __shfl_xor_sync(0xffffffffu, s_even, 4);
    s_odd  += __shfl_xor_sync(0xffffffffu, s_odd, 16);
    s_odd  += __shfl_xor_sync(0xffffffffu, s_odd, 8);
    s_odd  += __shfl_xor_sync(0xffffffffu, s_odd, 4);

    if (g == 0) {
        const int pe = p0 + 2 * t;
        const int po = pe + 1;
        if (pe < np_stride) scores[(size_t)r * np_stride + pe] = (pe < npool) ? s_even : -INFINITY;
        if (po < np_stride) scores[(size_t)r * np_stride + po] = (po < npool) ? s_odd  : -INFINITY;
    }
}

namespace m3tis_dsa {
__device__ __forceinline__ uint32_t order_key(float s) {
    const int32_t b = __float_as_int(s + 0.0f);
    return (uint32_t)(b ^ ((b >> 31) | (int32_t)0x80000000));
}

__device__ __forceinline__ int block_scan_256(int v, int (*wsum)[8], int slot, int* total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += u;
    }
    if (lane == 31) wsum[slot][warp] = inc;
    __syncthreads();
    int before = inc - v;
    int all = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const int s = wsum[slot][w];
        if (w < warp) before += s;
        all += s;
    }
    *total = all;
    return before;
}
} // namespace m3tis_dsa

// 4. Exact 4-pass 8-bit radix top-K pool selection (ties -> lower pool index, ascending order)
//    fused with 4-token pool + incomplete-tail expansion into `tokens[rows, width]` & `counts[rows]`.
//    Milestone 8: Supports 2-Branch Speculative Tree Verification (`tree_trunk_len`, `tree_b_start`):
//    for Branch B rows (`r >= tree_b_start`), selects from pure history pools `0..p_fork` and appends
//    both trunk tail `[p_fork*4..pos+tree_trunk_len)` and Branch B tokens `[pos+tree_b_start..pos+r]`!
//    When `npool <= K`, sets `counts[r] = 0` (signaling dense causal attention for row `r`).
// Launch: grid(rows, 1, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_dsa_select_tokens(
    const float* __restrict__ scores, // [rows, np_stride]
    int32_t* __restrict__ tokens,     // [rows, width]
    int32_t* __restrict__ counts,     // [rows]
    int32_t pos,
    int32_t np_stride,
    int32_t topk_pools,               // e.g. 512 (<= 1024)
    int32_t width,                    // e.g. 4*K + 16
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    __shared__ int hist[256];
    __shared__ int wsum[2][8];
    __shared__ uint32_t sh_prefix;
    __shared__ int sh_need;
    __shared__ int sel_pools[1024];

    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int q_pos = pos + r;
    const bool is_branch_b = (tree_b_start > 0 && r >= tree_b_start);
    const int fork_pos_abs = pos + tree_trunk_len;
    const int npool = is_branch_b ? (fork_pos_abs / 4) : ((q_pos + 1) / 4);

    if (npool <= topk_pools) {
        if (t == 0) counts[r] = 0;
        return;
    }

    const float* row = scores + (size_t)r * np_stride;
    const int limit = min(npool, np_stride);
    uint32_t prefix = 0u, fixed = 0u;
    int need = topk_pools;

    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[t] = 0;
        __syncthreads();
        for (int i = t; i < limit; i += 256) {
            const uint32_t u = m3tis_dsa::order_key(row[i]);
            if ((u & fixed) == prefix) {
                atomicAdd(&hist[(u >> shift) & 0xFFu], 1);
            }
        }
        __syncthreads();
        if (t == 0) {
            int above = 0;
            int digit = 0;
            for (int b = 255; b >= 0; --b) {
                const int h = hist[b];
                if (above + h >= need) {
                    digit = b;
                    break;
                }
                above += h;
            }
            sh_need = need - above;
            sh_prefix = prefix | ((uint32_t)digit << shift);
        }
        __syncthreads();
        need = sh_need;
        prefix = sh_prefix;
        fixed |= 0xFFu << shift;
        __syncthreads();
    }

    // Collect top-K pools in strictly ascending pool-index order (ties go to lower index!)
    int run_take = 0;
    int run_eq = 0;
    for (int c = 0; c < limit; c += 256) {
        const int idx = c + t;
        const bool in = idx < limit;
        const uint32_t u = in ? m3tis_dsa::order_key(row[idx]) : 0u;
        const int is_gt = (in && u > prefix) ? 1 : 0;
        const int is_eq = (in && u == prefix) ? 1 : 0;
        int all_eq = 0, all_take = 0;
        const int eq_before = run_eq + m3tis_dsa::block_scan_256(is_eq, wsum, 0, &all_eq);
        const int take = (is_gt || (is_eq && eq_before < need)) ? 1 : 0;
        const int out_pos = run_take + m3tis_dsa::block_scan_256(take, wsum, 1, &all_take);
        if (take && out_pos < topk_pools) {
            sel_pools[out_pos] = idx;
        }
        run_eq += all_eq;
        run_take += all_take;
        __syncthreads();
    }

    // Expand K selected pools -> 4*K token indices + tail tokens
    int32_t* tok_row = tokens + (size_t)r * width;
    for (int i = t; i < 4 * topk_pools; i += 256) {
        tok_row[i] = sel_pools[i >> 2] * 4 + (i & 3);
    }
    if (is_branch_b) {
        if (t == 0) {
            int out_c = 4 * topk_pools;
            for (int kp = npool * 4; kp < fork_pos_abs && out_c < width; ++kp) {
                tok_row[out_c++] = kp;
            }
            for (int kp = pos + tree_b_start; kp <= q_pos && out_c < width; ++kp) {
                tok_row[out_c++] = kp;
            }
            counts[r] = out_c;
        }
    } else {
        if (t < 3) {
            const int tail_tok = npool * 4 + t;
            tok_row[4 * topk_pools + t] = (tail_tok <= q_pos) ? tail_tok : -1;
        }
        if (t == 0) {
            int tail = (q_pos + 1) - npool * 4;
            if (tail < 0) tail = 0;
            if (tail > 3) tail = 3;
            counts[r] = 4 * topk_pools + tail;
        }
    }
}

// 5. MLA Sparse Causal Latent Attention over selected token list `tokens[rows, width]` (falls back to dense
//    causal `0..pos+r` for any row with `counts[r] == 0` so chunks crossing the dense/sparse boundary work cleanly!).
//    Fast path (lw == 512, heads % 8 == 0): 8 heads share 8 cached KV vectors in __shared__ via 128-bit uint4!
// Launch: grid(rows, (lw == 512 && (heads & 7) == 0) ? (heads / 8) : heads, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_mla_attend_sparse(
    const __nv_bfloat16* __restrict__ qa,       // [rows, heads, lw]
    const __nv_bfloat16* __restrict__ kv_cache, // [capacity, lw]
    const int32_t* __restrict__ tokens,         // [rows, width]
    const int32_t* __restrict__ counts,         // [rows]
    __nv_bfloat16* __restrict__ ol,             // [rows, heads, lw]
    int32_t pos,
    int32_t capacity,
    int32_t width,
    int32_t rows,
    int32_t heads,
    int32_t lw,
    float scale,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const bool is_branch_b = (tree_b_start > 0 && r >= tree_b_start);
    const int skip_lo = pos + tree_trunk_len;
    const int skip_hi = pos + tree_b_start;

    if (lw == 512 && (heads & 7) == 0) {
        const int h = blockIdx.y * 8 + warp;
        if (r >= rows || h >= heads) return;
        __shared__ __align__(16) __nv_bfloat16 k_sh[8][512];
        __shared__ int k_valid[8];

        const __nv_bfloat16* qrow = qa + ((size_t)r * heads + h) * 512;
        const int d0 = lane * 8;
        const uint4 qu0 = *reinterpret_cast<const uint4*>(qrow + d0);
        const uint4 qu1 = *reinterpret_cast<const uint4*>(qrow + 256 + d0);
        const __nv_bfloat16* qh0 = reinterpret_cast<const __nv_bfloat16*>(&qu0);
        const __nv_bfloat16* qh1 = reinterpret_cast<const __nv_bfloat16*>(&qu1);
        float q0[8], q1[8], o0[8], o1[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q0[j] = m3tis_glm::bf(qh0[j]);
            q1[j] = m3tis_glm::bf(qh1[j]);
            o0[j] = 0.0f;
            o1[j] = 0.0f;
        }
        float m_max = -1e30f;
        float l_sum = 0.0f;

        const int cnt = counts[r];
        const bool is_sparse = cnt > 0;
        const int n_iter = is_sparse ? cnt : (pos + r + 1);
        const int32_t* tok_row = tokens + (size_t)r * width;

        for (int jb = 0; jb < n_iter; jb += 8) {
            const int n_k = min(8, n_iter - jb);
            if (warp < n_k) {
                const int kp = is_sparse ? tok_row[jb + warp] : (jb + warp);
                const bool skip = (kp < 0) || (is_branch_b && kp >= skip_lo && kp < skip_hi);
                if (!skip) {
                    const int slot = kp % capacity;
                    const __nv_bfloat16* ksrc = kv_cache + (size_t)slot * 512;
                    *reinterpret_cast<uint4*>(&k_sh[warp][d0]) =
                        __ldg(reinterpret_cast<const uint4*>(ksrc + d0));
                    *reinterpret_cast<uint4*>(&k_sh[warp][256 + d0]) =
                        __ldg(reinterpret_cast<const uint4*>(ksrc + 256 + d0));
                    if (lane == 0) k_valid[warp] = 1;
                } else {
                    if (lane == 0) k_valid[warp] = 0;
                }
            }
            __syncthreads();

            for (int kk = 0; kk < n_k; ++kk) {
                if (!k_valid[kk]) continue;
                const uint4 ku0 = *reinterpret_cast<const uint4*>(&k_sh[kk][d0]);
                const uint4 ku1 = *reinterpret_cast<const uint4*>(&k_sh[kk][256 + d0]);
                const __nv_bfloat16* kh0 = reinterpret_cast<const __nv_bfloat16*>(&ku0);
                const __nv_bfloat16* kh1 = reinterpret_cast<const __nv_bfloat16*>(&ku1);
                float k0[8], k1[8];
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    k0[j] = m3tis_glm::bf(kh0[j]);
                    k1[j] = m3tis_glm::bf(kh1[j]);
                    dot = fmaf(q0[j], k0[j], fmaf(q1[j], k1[j], dot));
                }
                const float score = m3tis_glm::warp_sum(dot) * scale;
                const float m_new = fmaxf(m_max, score);
                const float alpha = expf(m_max - m_new);
                const float p = expf(score - m_new);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    o0[j] = fmaf(p, k0[j], o0[j] * alpha);
                    o1[j] = fmaf(p, k1[j], o1[j] * alpha);
                }
                l_sum = fmaf(l_sum, alpha, p);
                m_max = m_new;
            }
            __syncthreads();
        }

        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        uint4 ou0, ou1;
        __nv_bfloat16* oh0 = reinterpret_cast<__nv_bfloat16*>(&ou0);
        __nv_bfloat16* oh1 = reinterpret_cast<__nv_bfloat16*>(&ou1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            oh0[j] = __float2bfloat16_rn(o0[j] * inv_l);
            oh1[j] = __float2bfloat16_rn(o1[j] * inv_l);
        }
        __nv_bfloat16* orow = ol + ((size_t)r * heads + h) * 512;
        *reinterpret_cast<uint4*>(orow + d0) = ou0;
        *reinterpret_cast<uint4*>(orow + 256 + d0) = ou1;
        return;
    }

    const int h = blockIdx.y;
    if (r >= rows || h >= heads) return;
    __shared__ float smem[8];
    __shared__ float s_score;

    const __nv_bfloat16* qrow = qa + ((size_t)r * heads + h) * lw;
    float q0 = (t < lw) ? m3tis_glm::bf(qrow[t]) : 0.0f;
    float q1 = (t + 256 < lw) ? m3tis_glm::bf(qrow[t + 256]) : 0.0f;
    float o0 = 0.0f, o1 = 0.0f;
    float m_max = -1e30f;
    float l_sum = 0.0f;

    const int cnt = counts[r];
    const bool is_sparse = cnt > 0;
    const int n_iter = is_sparse ? cnt : (pos + r + 1);
    const int32_t* tok_row = tokens + (size_t)r * width;

    for (int j = 0; j < n_iter; ++j) {
        const int kp = is_sparse ? tok_row[j] : j;
        if (kp < 0 || (is_branch_b && kp >= skip_lo && kp < skip_hi)) continue;
        const int slot = kp % capacity;
        const __nv_bfloat16* krow = kv_cache + (size_t)slot * lw;
        float k0 = (t < lw) ? m3tis_glm::bf(krow[t]) : 0.0f;
        float k1 = (t + 256 < lw) ? m3tis_glm::bf(krow[t + 256]) : 0.0f;

        float dot = fmaf(q0, k0, q1 * k1);
        dot = m3tis_glm::warp_sum(dot);
        if (lane == 0) smem[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float tot = (lane < 8) ? smem[lane] : 0.0f;
            tot = m3tis_glm::warp_sum(tot);
            if (lane == 0) s_score = tot * scale;
        }
        __syncthreads();
        const float score = s_score;
        __syncthreads();

        const float m_new = fmaxf(m_max, score);
        const float alpha = expf(m_max - m_new);
        const float p = expf(score - m_new);
        o0 = fmaf(p, k0, o0 * alpha);
        o1 = fmaf(p, k1, o1 * alpha);
        l_sum = fmaf(l_sum, alpha, p);
        m_max = m_new;
    }

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    __nv_bfloat16* orow = ol + ((size_t)r * heads + h) * lw;
    if (t < lw) orow[t] = __float2bfloat16_rn(o0 * inv_l);
    if (t + 256 < lw) orow[t + 256] = __float2bfloat16_rn(o1 * inv_l);
}

// Milestone 6: 16-Head Shared-KV Ampere sm_86 BF16 Tensor Core FlashMLA (Dense & Sparse, lw == 512, heads % 16 == 0)!
// Launch: grid(rows, heads / 16, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_mla_attend_dense_tc(
    const __nv_bfloat16* __restrict__ qa,       // [rows, heads, 512]
    const __nv_bfloat16* __restrict__ kv_cache, // [capacity, 512]
    __nv_bfloat16* __restrict__ ol,             // [rows, heads, 512]
    int32_t pos,
    int32_t capacity,
    int32_t window,
    int32_t rows,
    int32_t heads,
    int32_t lw,
    float scale,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    (void)lw;
    const int r = blockIdx.x;
    const int h0 = blockIdx.y * 16;
    if (r >= rows || h0 >= heads) return;
    const bool is_branch_b = (tree_b_start > 0 && r >= tree_b_start);
    const int skip_lo = pos + tree_trunk_len;
    const int skip_hi = pos + tree_b_start;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int c0 = warp * 64;

    __shared__ __align__(16) __nv_bfloat16 sh_kv[16][520];
    __shared__ float sh_s_part[8][16][16];
    __shared__ __align__(16) __nv_bfloat16 sh_p[16][24];
    __shared__ float sh_alpha[16];
    __shared__ float sh_inv_l[16];
    __shared__ int sh_k_valid[16];

    const int lc = (tid & 63) * 8;
    const int lr0 = tid >> 6;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int hl = i * 4 + lr0;
        const __nv_bfloat16* qsrc = qa + ((size_t)r * heads + h0 + hl) * 512 + lc;
        *reinterpret_cast<uint4*>(&sh_kv[hl][lc]) = *reinterpret_cast<const uint4*>(qsrc);
    }
    __syncthreads();

    uint32_t q_frag[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int k0 = c0 + ks * 16 + 2 * t;
        q_frag[ks][0] = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0]);
        q_frag[ks][1] = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0]);
        q_frag[ks][2] = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0 + 8]);
        q_frag[ks][3] = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0 + 8]);
    }
    __syncthreads();

    float o_acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) o_acc[nt][c] = 0.0f;

    const int h_sub = lane >> 4;
    const int kk_lane = lane & 15;
    const int hl_sm = warp * 2 + h_sub;
    float m_max = -1e30f;
    float l_sum = 0.0f;

    const int q_pos = pos + r;
    const int k_start = max(0, q_pos - window + 1);
    const int total_k = q_pos - k_start + 1;

    for (int kb = 0; kb < total_k; kb += 16) {
        const int n_k = min(16, total_k - kb);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int kk = i * 4 + lr0;
            const int kp = k_start + kb + kk;
            const bool ok = (kk < n_k) && !(is_branch_b && kp >= skip_lo && kp < skip_hi);
            if (ok) {
                const int slot = kp % capacity;
                const __nv_bfloat16* ksrc = kv_cache + (size_t)slot * 512 + lc;
                *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = __ldg(reinterpret_cast<const uint4*>(ksrc));
                if (lc == 0) sh_k_valid[kk] = 1;
            } else {
                *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = make_uint4(0u, 0u, 0u, 0u);
                if (lc == 0) sh_k_valid[kk] = 0;
            }
        }
        __syncthreads();

        float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k0 = c0 + ks * 16 + 2 * t;
            const uint32_t b00 = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0]);
            const uint32_t b01 = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0 + 8]);
            const uint32_t b10 = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0]);
            const uint32_t b11 = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0 + 8]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(s0[0]), "+f"(s0[1]), "+f"(s0[2]), "+f"(s0[3])
                : "r"(q_frag[ks][0]), "r"(q_frag[ks][1]), "r"(q_frag[ks][2]), "r"(q_frag[ks][3]), "r"(b00), "r"(b01)
            );
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(s1[0]), "+f"(s1[1]), "+f"(s1[2]), "+f"(s1[3])
                : "r"(q_frag[ks][0]), "r"(q_frag[ks][1]), "r"(q_frag[ks][2]), "r"(q_frag[ks][3]), "r"(b10), "r"(b11)
            );
        }
        sh_s_part[warp][g][2 * t]             = s0[0];
        sh_s_part[warp][g][2 * t + 1]         = s0[1];
        sh_s_part[warp][g + 8][2 * t]         = s0[2];
        sh_s_part[warp][g + 8][2 * t + 1]     = s0[3];
        sh_s_part[warp][g][8 + 2 * t]         = s1[0];
        sh_s_part[warp][g][8 + 2 * t + 1]     = s1[1];
        sh_s_part[warp][g + 8][8 + 2 * t]     = s1[2];
        sh_s_part[warp][g + 8][8 + 2 * t + 1] = s1[3];
        __syncthreads();

        float sc = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) sc += sh_s_part[w][hl_sm][kk_lane];
        const bool valid = (kk_lane < n_k) && (sh_k_valid[kk_lane] != 0);
        sc = valid ? (sc * scale) : -1e30f;

        float m_blk = sc;
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 8));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 4));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 2));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 1));

        if (m_blk > -1e29f) {
            const float m_new = fmaxf(m_max, m_blk);
            const float alpha = (m_max > -1e29f) ? expf(m_max - m_new) : 0.0f;
            const float p = valid ? expf(sc - m_new) : 0.0f;
            float l_blk = p;
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 8);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 4);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 2);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 1);
            l_sum = fmaf(l_sum, alpha, l_blk);
            m_max = m_new;
            sh_p[hl_sm][kk_lane] = __float2bfloat16_rn(p);
            if (kk_lane == 0) sh_alpha[hl_sm] = alpha;
        } else {
            sh_p[hl_sm][kk_lane] = __float2bfloat16_rn(0.0f);
            if (kk_lane == 0) sh_alpha[hl_sm] = 1.0f;
        }
        __syncthreads();

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(&sh_p[g][2 * t]);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(&sh_p[g + 8][2 * t]);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(&sh_p[g][8 + 2 * t]);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(&sh_p[g + 8][8 + 2 * t]);
        const float al0 = sh_alpha[g];
        const float al1 = sh_alpha[g + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= al0;
            o_acc[nt][1] *= al0;
            o_acc[nt][2] *= al1;
            o_acc[nt][3] *= al1;
            const int vc = c0 + nt * 8 + g;
            const uint16_t v0a = *reinterpret_cast<const uint16_t*>(&sh_kv[2 * t][vc]);
            const uint16_t v0b = *reinterpret_cast<const uint16_t*>(&sh_kv[2 * t + 1][vc]);
            const uint16_t v1a = *reinterpret_cast<const uint16_t*>(&sh_kv[8 + 2 * t][vc]);
            const uint16_t v1b = *reinterpret_cast<const uint16_t*>(&sh_kv[8 + 2 * t + 1][vc]);
            const uint32_t vb0 = (uint32_t)v0a | ((uint32_t)v0b << 16);
            const uint32_t vb1 = (uint32_t)v1a | ((uint32_t)v1b << 16);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(vb0), "r"(vb1)
            );
        }
        __syncthreads();
    }

    if (kk_lane == 0) sh_inv_l[hl_sm] = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    __syncthreads();

    const float il0 = sh_inv_l[g];
    const float il1 = sh_inv_l[g + 8];
    __nv_bfloat16* orow0 = ol + ((size_t)r * heads + h0 + g) * 512 + c0;
    __nv_bfloat16* orow1 = ol + ((size_t)r * heads + h0 + g + 8) * 512 + c0;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int oc = nt * 8 + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(orow0 + oc) =
            __floats2bfloat162_rn(o_acc[nt][0] * il0, o_acc[nt][1] * il0);
        *reinterpret_cast<__nv_bfloat162*>(orow1 + oc) =
            __floats2bfloat162_rn(o_acc[nt][2] * il1, o_acc[nt][3] * il1);
    }
}

// Sparse Top-2048 DSA 16-Head Shared-KV Ampere sm_86 BF16 Tensor Core FlashMLA (`lw == 512, heads % 16 == 0`)!
// Launch: grid(rows, heads / 16, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_mla_attend_sparse_tc(
    const __nv_bfloat16* __restrict__ qa,       // [rows, heads, 512]
    const __nv_bfloat16* __restrict__ kv_cache, // [capacity, 512]
    const int32_t* __restrict__ tokens,         // [rows, width]
    const int32_t* __restrict__ counts,         // [rows]
    __nv_bfloat16* __restrict__ ol,             // [rows, heads, 512]
    int32_t pos,
    int32_t capacity,
    int32_t width,
    int32_t rows,
    int32_t heads,
    int32_t lw,
    float scale,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    (void)lw;
    const int r = blockIdx.x;
    const int h0 = blockIdx.y * 16;
    if (r >= rows || h0 >= heads) return;
    const bool is_branch_b = (tree_b_start > 0 && r >= tree_b_start);
    const int skip_lo = pos + tree_trunk_len;
    const int skip_hi = pos + tree_b_start;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int c0 = warp * 64;

    __shared__ __align__(16) __nv_bfloat16 sh_kv[16][520];
    __shared__ float sh_s_part[8][16][16];
    __shared__ __align__(16) __nv_bfloat16 sh_p[16][24];
    __shared__ float sh_alpha[16];
    __shared__ float sh_inv_l[16];
    __shared__ int sh_k_valid[16];

    const int lc = (tid & 63) * 8;
    const int lr0 = tid >> 6;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int hl = i * 4 + lr0;
        const __nv_bfloat16* qsrc = qa + ((size_t)r * heads + h0 + hl) * 512 + lc;
        *reinterpret_cast<uint4*>(&sh_kv[hl][lc]) = *reinterpret_cast<const uint4*>(qsrc);
    }
    __syncthreads();

    uint32_t q_frag[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int k0 = c0 + ks * 16 + 2 * t;
        q_frag[ks][0] = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0]);
        q_frag[ks][1] = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0]);
        q_frag[ks][2] = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0 + 8]);
        q_frag[ks][3] = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0 + 8]);
    }
    __syncthreads();

    float o_acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) o_acc[nt][c] = 0.0f;

    const int h_sub = lane >> 4;
    const int kk_lane = lane & 15;
    const int hl_sm = warp * 2 + h_sub;
    float m_max = -1e30f;
    float l_sum = 0.0f;

    const int cnt = counts[r];
    const bool is_sparse = cnt > 0;
    const int n_iter = is_sparse ? cnt : (pos + r + 1);
    const int32_t* tok_row = tokens + (size_t)r * width;

    for (int jb = 0; jb < n_iter; jb += 16) {
        const int n_k = min(16, n_iter - jb);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int kk = i * 4 + lr0;
            if (kk < n_k) {
                const int kp = is_sparse ? tok_row[jb + kk] : (jb + kk);
                const bool ok = (kp >= 0) && !(is_branch_b && kp >= skip_lo && kp < skip_hi);
                if (ok) {
                    const int slot = kp % capacity;
                    const __nv_bfloat16* ksrc = kv_cache + (size_t)slot * 512 + lc;
                    *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = __ldg(reinterpret_cast<const uint4*>(ksrc));
                    if (lc == 0) sh_k_valid[kk] = 1;
                } else {
                    *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = make_uint4(0u, 0u, 0u, 0u);
                    if (lc == 0) sh_k_valid[kk] = 0;
                }
            } else {
                *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = make_uint4(0u, 0u, 0u, 0u);
                if (lc == 0) sh_k_valid[kk] = 0;
            }
        }
        __syncthreads();

        float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k0 = c0 + ks * 16 + 2 * t;
            const uint32_t b00 = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0]);
            const uint32_t b01 = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0 + 8]);
            const uint32_t b10 = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0]);
            const uint32_t b11 = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0 + 8]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(s0[0]), "+f"(s0[1]), "+f"(s0[2]), "+f"(s0[3])
                : "r"(q_frag[ks][0]), "r"(q_frag[ks][1]), "r"(q_frag[ks][2]), "r"(q_frag[ks][3]), "r"(b00), "r"(b01)
            );
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(s1[0]), "+f"(s1[1]), "+f"(s1[2]), "+f"(s1[3])
                : "r"(q_frag[ks][0]), "r"(q_frag[ks][1]), "r"(q_frag[ks][2]), "r"(q_frag[ks][3]), "r"(b10), "r"(b11)
            );
        }
        sh_s_part[warp][g][2 * t]             = s0[0];
        sh_s_part[warp][g][2 * t + 1]         = s0[1];
        sh_s_part[warp][g + 8][2 * t]         = s0[2];
        sh_s_part[warp][g + 8][2 * t + 1]     = s0[3];
        sh_s_part[warp][g][8 + 2 * t]         = s1[0];
        sh_s_part[warp][g][8 + 2 * t + 1]     = s1[1];
        sh_s_part[warp][g + 8][8 + 2 * t]     = s1[2];
        sh_s_part[warp][g + 8][8 + 2 * t + 1] = s1[3];
        __syncthreads();

        float sc = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) sc += sh_s_part[w][hl_sm][kk_lane];
        const bool valid = (kk_lane < n_k) && (sh_k_valid[kk_lane] != 0);
        sc = valid ? (sc * scale) : -1e30f;

        float m_blk = sc;
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 8));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 4));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 2));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 1));

        if (m_blk > -1e29f) {
            const float m_new = fmaxf(m_max, m_blk);
            const float alpha = (m_max > -1e29f) ? expf(m_max - m_new) : 0.0f;
            const float p = valid ? expf(sc - m_new) : 0.0f;
            float l_blk = p;
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 8);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 4);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 2);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 1);
            l_sum = fmaf(l_sum, alpha, l_blk);
            m_max = m_new;
            sh_p[hl_sm][kk_lane] = __float2bfloat16_rn(p);
            if (kk_lane == 0) sh_alpha[hl_sm] = alpha;
        } else {
            sh_p[hl_sm][kk_lane] = __float2bfloat16_rn(0.0f);
            if (kk_lane == 0) sh_alpha[hl_sm] = 1.0f;
        }
        __syncthreads();

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(&sh_p[g][2 * t]);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(&sh_p[g + 8][2 * t]);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(&sh_p[g][8 + 2 * t]);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(&sh_p[g + 8][8 + 2 * t]);
        const float al0 = sh_alpha[g];
        const float al1 = sh_alpha[g + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= al0;
            o_acc[nt][1] *= al0;
            o_acc[nt][2] *= al1;
            o_acc[nt][3] *= al1;
            const int vc = c0 + nt * 8 + g;
            const uint16_t v0a = *reinterpret_cast<const uint16_t*>(&sh_kv[2 * t][vc]);
            const uint16_t v0b = *reinterpret_cast<const uint16_t*>(&sh_kv[2 * t + 1][vc]);
            const uint16_t v1a = *reinterpret_cast<const uint16_t*>(&sh_kv[8 + 2 * t][vc]);
            const uint16_t v1b = *reinterpret_cast<const uint16_t*>(&sh_kv[8 + 2 * t + 1][vc]);
            const uint32_t vb0 = (uint32_t)v0a | ((uint32_t)v0b << 16);
            const uint32_t vb1 = (uint32_t)v1a | ((uint32_t)v1b << 16);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(vb0), "r"(vb1)
            );
        }
        __syncthreads();
    }

    if (kk_lane == 0) sh_inv_l[hl_sm] = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    __syncthreads();

    const float il0 = sh_inv_l[g];
    const float il1 = sh_inv_l[g + 8];
    __nv_bfloat16* orow0 = ol + ((size_t)r * heads + h0 + g) * 512 + c0;
    __nv_bfloat16* orow1 = ol + ((size_t)r * heads + h0 + g + 8) * 512 + c0;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int oc = nt * 8 + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(orow0 + oc) =
            __floats2bfloat162_rn(o_acc[nt][0] * il0, o_acc[nt][1] * il0);
        *reinterpret_cast<__nv_bfloat162*>(orow1 + oc) =
            __floats2bfloat162_rn(o_acc[nt][2] * il1, o_acc[nt][3] * il1);
    }
}

// =================================================================================================
// Milestone 8: Symmetric Per-Token INT8 Latent KV Cache (516 B/tok vs 1024 B/tok -> 2x VRAM & BW!)
//              + Sequential Race-Free Branch-B Compaction Helpers for 2-Branch Speculative Trees!
// =================================================================================================

// Write `lat[rows, lw]` BF16 -> symmetric per-token `kv_q8[capacity, lw]` (i8) + `kv_sc[capacity]` (f32).
// Launch: grid(rows, 1, 1), block(64, 1, 1)
__global__ void __launch_bounds__(64) m3tis_mla_write_kv_q8(
    const __nv_bfloat16* __restrict__ lat,
    int8_t* __restrict__ kv_q8,
    float* __restrict__ kv_sc,
    int32_t pos,
    int32_t capacity,
    int32_t rows,
    int32_t lw
) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int slot = (pos + r) % capacity;
    __shared__ float wmax[2];

    if (lw == 512) {
        const int d0 = tid * 8;
        const uint4 xu = *reinterpret_cast<const uint4*>(lat + (size_t)r * 512 + d0);
        const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
        float v[8];
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = m3tis_glm::bf(xh[j]);
            amax = fmaxf(amax, fabsf(v[j]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        if (lane == 0) wmax[warp] = amax;
        __syncthreads();
        const float tot_max = fmaxf(wmax[0], wmax[1]);
        const float sc = (tot_max > 1e-12f) ? (tot_max / 127.0f) : 0.0f;
        const float inv = (tot_max > 1e-12f) ? (127.0f / tot_max) : 0.0f;
        if (tid == 0) kv_sc[slot] = sc;

        int8_t q[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int qi = __float2int_rn(v[j] * inv);
            qi = max(-127, min(127, qi));
            q[j] = (int8_t)qi;
        }
        *reinterpret_cast<uint2*>(kv_q8 + (size_t)slot * 512 + d0) = *reinterpret_cast<const uint2*>(q);
        return;
    }

    float amax = 0.0f;
    for (int d = tid; d < lw; d += 64) {
        amax = fmaxf(amax, fabsf(m3tis_glm::bf(lat[(size_t)r * lw + d])));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) wmax[warp] = amax;
    __syncthreads();
    const float tot_max = fmaxf(wmax[0], wmax[1]);
    const float sc = (tot_max > 1e-12f) ? (tot_max / 127.0f) : 0.0f;
    const float inv = (tot_max > 1e-12f) ? (127.0f / tot_max) : 0.0f;
    if (tid == 0) kv_sc[slot] = sc;
    for (int d = tid; d < lw; d += 64) {
        int qi = __float2int_rn(m3tis_glm::bf(lat[(size_t)r * lw + d]) * inv);
        kv_q8[(size_t)slot * lw + d] = (int8_t)max(-127, min(127, qi));
    }
}

// MLA Dense Causal Latent Attention over INT8 KV Cache (`lw == 512, heads % 8 == 0`, 2x less KV HBM BW!).
// Launch: grid(rows, heads / 8, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_mla_attend_dense_q8(
    const __nv_bfloat16* __restrict__ qa,   // [rows, heads, 512]
    const int8_t* __restrict__ kv_q8,       // [capacity, 512]
    const float* __restrict__ kv_sc,        // [capacity]
    __nv_bfloat16* __restrict__ ol,         // [rows, heads, 512]
    int32_t pos,
    int32_t capacity,
    int32_t window,
    int32_t rows,
    int32_t heads,
    int32_t lw,
    float scale,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    (void)lw;
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int h = blockIdx.y * 8 + warp;
    if (r >= rows || h >= heads) return;
    const bool is_branch_b = (tree_b_start > 0 && r >= tree_b_start);
    const int skip_lo = pos + tree_trunk_len;
    const int skip_hi = pos + tree_b_start;

    __shared__ __align__(16) int8_t k_sh[8][512];
    __shared__ float k_sc_sh[8];
    __shared__ int k_valid[8];

    const __nv_bfloat16* qrow = qa + ((size_t)r * heads + h) * 512;
    const int d0 = lane * 8;
    const uint4 qu0 = *reinterpret_cast<const uint4*>(qrow + d0);
    const uint4 qu1 = *reinterpret_cast<const uint4*>(qrow + 256 + d0);
    const __nv_bfloat16* qh0 = reinterpret_cast<const __nv_bfloat16*>(&qu0);
    const __nv_bfloat16* qh1 = reinterpret_cast<const __nv_bfloat16*>(&qu1);
    float q0[8], q1[8], o0[8], o1[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        q0[j] = m3tis_glm::bf(qh0[j]);
        q1[j] = m3tis_glm::bf(qh1[j]);
        o0[j] = 0.0f;
        o1[j] = 0.0f;
    }
    float m_max = -1e30f;
    float l_sum = 0.0f;

    const int q_pos = pos + r;
    const int k_start = max(0, q_pos - window + 1);
    const int total_k = q_pos - k_start + 1;

    for (int kb = 0; kb < total_k; kb += 8) {
        const int n_k = min(8, total_k - kb);
        if (warp < n_k) {
            const int kp = k_start + kb + warp;
            const bool skip = is_branch_b && (kp >= skip_lo && kp < skip_hi);
            if (!skip) {
                const int slot = kp % capacity;
                const int8_t* ksrc = kv_q8 + (size_t)slot * 512;
                *reinterpret_cast<uint2*>(&k_sh[warp][d0]) =
                    __ldg(reinterpret_cast<const uint2*>(ksrc + d0));
                *reinterpret_cast<uint2*>(&k_sh[warp][256 + d0]) =
                    __ldg(reinterpret_cast<const uint2*>(ksrc + 256 + d0));
                if (lane == 0) {
                    k_sc_sh[warp] = __ldg(kv_sc + slot);
                    k_valid[warp] = 1;
                }
            } else {
                if (lane == 0) k_valid[warp] = 0;
            }
        }
        __syncthreads();

        for (int kk = 0; kk < n_k; ++kk) {
            if (!k_valid[kk]) continue;
            const uint2 ku0 = *reinterpret_cast<const uint2*>(&k_sh[kk][d0]);
            const uint2 ku1 = *reinterpret_cast<const uint2*>(&k_sh[kk][256 + d0]);
            const int8_t* kh0 = reinterpret_cast<const int8_t*>(&ku0);
            const int8_t* kh1 = reinterpret_cast<const int8_t*>(&ku1);
            float k0[8], k1[8];
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                k0[j] = (float)kh0[j];
                k1[j] = (float)kh1[j];
                dot = fmaf(q0[j], k0[j], fmaf(q1[j], k1[j], dot));
            }
            const float sc_k = k_sc_sh[kk];
            const float score = m3tis_glm::warp_sum(dot) * (sc_k * scale);
            const float m_new = fmaxf(m_max, score);
            const float alpha = expf(m_max - m_new);
            const float p = expf(score - m_new);
            const float ps = p * sc_k;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                o0[j] = fmaf(ps, k0[j], o0[j] * alpha);
                o1[j] = fmaf(ps, k1[j], o1[j] * alpha);
            }
            l_sum = fmaf(l_sum, alpha, p);
            m_max = m_new;
        }
        __syncthreads();
    }

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    uint4 ou0, ou1;
    __nv_bfloat16* oh0 = reinterpret_cast<__nv_bfloat16*>(&ou0);
    __nv_bfloat16* oh1 = reinterpret_cast<__nv_bfloat16*>(&ou1);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        oh0[j] = __float2bfloat16_rn(o0[j] * inv_l);
        oh1[j] = __float2bfloat16_rn(o1[j] * inv_l);
    }
    __nv_bfloat16* orow = ol + ((size_t)r * heads + h) * 512;
    *reinterpret_cast<uint4*>(orow + d0) = ou0;
    *reinterpret_cast<uint4*>(orow + 256 + d0) = ou1;
}

// MLA Sparse Causal Latent Attention over INT8 KV Cache (`lw == 512, heads % 8 == 0`).
// Launch: grid(rows, heads / 8, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_mla_attend_sparse_q8(
    const __nv_bfloat16* __restrict__ qa,   // [rows, heads, 512]
    const int8_t* __restrict__ kv_q8,       // [capacity, 512]
    const float* __restrict__ kv_sc,        // [capacity]
    const int32_t* __restrict__ tokens,     // [rows, width]
    const int32_t* __restrict__ counts,     // [rows]
    __nv_bfloat16* __restrict__ ol,         // [rows, heads, 512]
    int32_t pos,
    int32_t capacity,
    int32_t width,
    int32_t rows,
    int32_t heads,
    int32_t lw,
    float scale,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    (void)lw;
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int h = blockIdx.y * 8 + warp;
    if (r >= rows || h >= heads) return;
    const bool is_branch_b = (tree_b_start > 0 && r >= tree_b_start);
    const int skip_lo = pos + tree_trunk_len;
    const int skip_hi = pos + tree_b_start;

    __shared__ __align__(16) int8_t k_sh[8][512];
    __shared__ float k_sc_sh[8];
    __shared__ int k_valid[8];

    const __nv_bfloat16* qrow = qa + ((size_t)r * heads + h) * 512;
    const int d0 = lane * 8;
    const uint4 qu0 = *reinterpret_cast<const uint4*>(qrow + d0);
    const uint4 qu1 = *reinterpret_cast<const uint4*>(qrow + 256 + d0);
    const __nv_bfloat16* qh0 = reinterpret_cast<const __nv_bfloat16*>(&qu0);
    const __nv_bfloat16* qh1 = reinterpret_cast<const __nv_bfloat16*>(&qu1);
    float q0[8], q1[8], o0[8], o1[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        q0[j] = m3tis_glm::bf(qh0[j]);
        q1[j] = m3tis_glm::bf(qh1[j]);
        o0[j] = 0.0f;
        o1[j] = 0.0f;
    }
    float m_max = -1e30f;
    float l_sum = 0.0f;

    const int cnt = counts[r];
    const bool is_sparse = cnt > 0;
    const int n_iter = is_sparse ? cnt : (pos + r + 1);
    const int32_t* tok_row = tokens + (size_t)r * width;

    for (int jb = 0; jb < n_iter; jb += 8) {
        const int n_k = min(8, n_iter - jb);
        if (warp < n_k) {
            const int kp = is_sparse ? tok_row[jb + warp] : (jb + warp);
            const bool skip = (kp < 0) || (is_branch_b && kp >= skip_lo && kp < skip_hi);
            if (!skip) {
                const int slot = kp % capacity;
                const int8_t* ksrc = kv_q8 + (size_t)slot * 512;
                *reinterpret_cast<uint2*>(&k_sh[warp][d0]) =
                    __ldg(reinterpret_cast<const uint2*>(ksrc + d0));
                *reinterpret_cast<uint2*>(&k_sh[warp][256 + d0]) =
                    __ldg(reinterpret_cast<const uint2*>(ksrc + 256 + d0));
                if (lane == 0) {
                    k_sc_sh[warp] = __ldg(kv_sc + slot);
                    k_valid[warp] = 1;
                }
            } else {
                if (lane == 0) k_valid[warp] = 0;
            }
        }
        __syncthreads();

        for (int kk = 0; kk < n_k; ++kk) {
            if (!k_valid[kk]) continue;
            const uint2 ku0 = *reinterpret_cast<const uint2*>(&k_sh[kk][d0]);
            const uint2 ku1 = *reinterpret_cast<const uint2*>(&k_sh[kk][256 + d0]);
            const int8_t* kh0 = reinterpret_cast<const int8_t*>(&ku0);
            const int8_t* kh1 = reinterpret_cast<const int8_t*>(&ku1);
            float k0[8], k1[8];
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                k0[j] = (float)kh0[j];
                k1[j] = (float)kh1[j];
                dot = fmaf(q0[j], k0[j], fmaf(q1[j], k1[j], dot));
            }
            const float sc_k = k_sc_sh[kk];
            const float score = m3tis_glm::warp_sum(dot) * (sc_k * scale);
            const float m_new = fmaxf(m_max, score);
            const float alpha = expf(m_max - m_new);
            const float p = expf(score - m_new);
            const float ps = p * sc_k;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                o0[j] = fmaf(ps, k0[j], o0[j] * alpha);
                o1[j] = fmaf(ps, k1[j], o1[j] * alpha);
            }
            l_sum = fmaf(l_sum, alpha, p);
            m_max = m_new;
        }
        __syncthreads();
    }

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    uint4 ou0, ou1;
    __nv_bfloat16* oh0 = reinterpret_cast<__nv_bfloat16*>(&ou0);
    __nv_bfloat16* oh1 = reinterpret_cast<__nv_bfloat16*>(&ou1);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        oh0[j] = __float2bfloat16_rn(o0[j] * inv_l);
        oh1[j] = __float2bfloat16_rn(o1[j] * inv_l);
    }
    __nv_bfloat16* orow = ol + ((size_t)r * heads + h) * 512;
    *reinterpret_cast<uint4*>(orow + d0) = ou0;
    *reinterpret_cast<uint4*>(orow + 256 + d0) = ou1;
}

// 16-Head Shared-KV BF16 Tensor Core FlashMLA (Dense, INT8 KV -> on-the-fly BF16 dequant in smem, 2x less HBM!).
// Launch: grid(rows, heads / 16, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_mla_attend_dense_tc_q8(
    const __nv_bfloat16* __restrict__ qa,
    const int8_t* __restrict__ kv_q8,
    const float* __restrict__ kv_sc,
    __nv_bfloat16* __restrict__ ol,
    int32_t pos,
    int32_t capacity,
    int32_t window,
    int32_t rows,
    int32_t heads,
    int32_t lw,
    float scale,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    (void)lw;
    const int r = blockIdx.x;
    const int h0 = blockIdx.y * 16;
    if (r >= rows || h0 >= heads) return;
    const bool is_branch_b = (tree_b_start > 0 && r >= tree_b_start);
    const int skip_lo = pos + tree_trunk_len;
    const int skip_hi = pos + tree_b_start;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int c0 = warp * 64;

    __shared__ __align__(16) __nv_bfloat16 sh_kv[16][520];
    __shared__ float sh_s_part[8][16][16];
    __shared__ __align__(16) __nv_bfloat16 sh_p[16][24];
    __shared__ float sh_alpha[16];
    __shared__ float sh_inv_l[16];
    __shared__ int sh_k_valid[16];

    const int lc = (tid & 63) * 8;
    const int lr0 = tid >> 6;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int hl = i * 4 + lr0;
        const __nv_bfloat16* qsrc = qa + ((size_t)r * heads + h0 + hl) * 512 + lc;
        *reinterpret_cast<uint4*>(&sh_kv[hl][lc]) = *reinterpret_cast<const uint4*>(qsrc);
    }
    __syncthreads();

    uint32_t q_frag[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int k0 = c0 + ks * 16 + 2 * t;
        q_frag[ks][0] = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0]);
        q_frag[ks][1] = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0]);
        q_frag[ks][2] = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0 + 8]);
        q_frag[ks][3] = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0 + 8]);
    }
    __syncthreads();

    float o_acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) o_acc[nt][c] = 0.0f;

    const int h_sub = lane >> 4;
    const int kk_lane = lane & 15;
    const int hl_sm = warp * 2 + h_sub;
    float m_max = -1e30f;
    float l_sum = 0.0f;

    const int q_pos = pos + r;
    const int k_start = max(0, q_pos - window + 1);
    const int total_k = q_pos - k_start + 1;

    for (int kb = 0; kb < total_k; kb += 16) {
        const int n_k = min(16, total_k - kb);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int kk = i * 4 + lr0;
            const int kp = k_start + kb + kk;
            const bool ok = (kk < n_k) && !(is_branch_b && kp >= skip_lo && kp < skip_hi);
            if (ok) {
                const int slot = kp % capacity;
                const float sc_k = __ldg(kv_sc + slot);
                const uint2 qu = __ldg(reinterpret_cast<const uint2*>(kv_q8 + (size_t)slot * 512 + lc));
                const int8_t* qh = reinterpret_cast<const int8_t*>(&qu);
                __nv_bfloat162 bh[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    bh[j] = __floats2bfloat162_rn((float)qh[2 * j] * sc_k, (float)qh[2 * j + 1] * sc_k);
                }
                *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = *reinterpret_cast<const uint4*>(bh);
                if (lc == 0) sh_k_valid[kk] = 1;
            } else {
                *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = make_uint4(0u, 0u, 0u, 0u);
                if (lc == 0) sh_k_valid[kk] = 0;
            }
        }
        __syncthreads();

        float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k0 = c0 + ks * 16 + 2 * t;
            const uint32_t b00 = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0]);
            const uint32_t b01 = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0 + 8]);
            const uint32_t b10 = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0]);
            const uint32_t b11 = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0 + 8]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(s0[0]), "+f"(s0[1]), "+f"(s0[2]), "+f"(s0[3])
                : "r"(q_frag[ks][0]), "r"(q_frag[ks][1]), "r"(q_frag[ks][2]), "r"(q_frag[ks][3]), "r"(b00), "r"(b01)
            );
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(s1[0]), "+f"(s1[1]), "+f"(s1[2]), "+f"(s1[3])
                : "r"(q_frag[ks][0]), "r"(q_frag[ks][1]), "r"(q_frag[ks][2]), "r"(q_frag[ks][3]), "r"(b10), "r"(b11)
            );
        }
        sh_s_part[warp][g][2 * t]             = s0[0];
        sh_s_part[warp][g][2 * t + 1]         = s0[1];
        sh_s_part[warp][g + 8][2 * t]         = s0[2];
        sh_s_part[warp][g + 8][2 * t + 1]     = s0[3];
        sh_s_part[warp][g][8 + 2 * t]         = s1[0];
        sh_s_part[warp][g][8 + 2 * t + 1]     = s1[1];
        sh_s_part[warp][g + 8][8 + 2 * t]     = s1[2];
        sh_s_part[warp][g + 8][8 + 2 * t + 1] = s1[3];
        __syncthreads();

        float sc = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) sc += sh_s_part[w][hl_sm][kk_lane];
        const bool valid = (kk_lane < n_k) && (sh_k_valid[kk_lane] != 0);
        sc = valid ? (sc * scale) : -1e30f;

        float m_blk = sc;
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 8));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 4));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 2));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 1));

        if (m_blk > -1e29f) {
            const float m_new = fmaxf(m_max, m_blk);
            const float alpha = (m_max > -1e29f) ? expf(m_max - m_new) : 0.0f;
            const float p = valid ? expf(sc - m_new) : 0.0f;
            float l_blk = p;
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 8);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 4);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 2);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 1);
            l_sum = fmaf(l_sum, alpha, l_blk);
            m_max = m_new;
            sh_p[hl_sm][kk_lane] = __float2bfloat16_rn(p);
            if (kk_lane == 0) sh_alpha[hl_sm] = alpha;
        } else {
            sh_p[hl_sm][kk_lane] = __float2bfloat16_rn(0.0f);
            if (kk_lane == 0) sh_alpha[hl_sm] = 1.0f;
        }
        __syncthreads();

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(&sh_p[g][2 * t]);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(&sh_p[g + 8][2 * t]);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(&sh_p[g][8 + 2 * t]);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(&sh_p[g + 8][8 + 2 * t]);
        const float al0 = sh_alpha[g];
        const float al1 = sh_alpha[g + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= al0;
            o_acc[nt][1] *= al0;
            o_acc[nt][2] *= al1;
            o_acc[nt][3] *= al1;
            const int vc = c0 + nt * 8 + g;
            const uint16_t v0a = *reinterpret_cast<const uint16_t*>(&sh_kv[2 * t][vc]);
            const uint16_t v0b = *reinterpret_cast<const uint16_t*>(&sh_kv[2 * t + 1][vc]);
            const uint16_t v1a = *reinterpret_cast<const uint16_t*>(&sh_kv[8 + 2 * t][vc]);
            const uint16_t v1b = *reinterpret_cast<const uint16_t*>(&sh_kv[8 + 2 * t + 1][vc]);
            const uint32_t vb0 = (uint32_t)v0a | ((uint32_t)v0b << 16);
            const uint32_t vb1 = (uint32_t)v1a | ((uint32_t)v1b << 16);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(vb0), "r"(vb1)
            );
        }
        __syncthreads();
    }

    if (kk_lane == 0) sh_inv_l[hl_sm] = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    __syncthreads();

    const float il0 = sh_inv_l[g];
    const float il1 = sh_inv_l[g + 8];
    __nv_bfloat16* orow0 = ol + ((size_t)r * heads + h0 + g) * 512 + c0;
    __nv_bfloat16* orow1 = ol + ((size_t)r * heads + h0 + g + 8) * 512 + c0;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int oc = nt * 8 + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(orow0 + oc) =
            __floats2bfloat162_rn(o_acc[nt][0] * il0, o_acc[nt][1] * il0);
        *reinterpret_cast<__nv_bfloat162*>(orow1 + oc) =
            __floats2bfloat162_rn(o_acc[nt][2] * il1, o_acc[nt][3] * il1);
    }
}

// 16-Head Shared-KV BF16 Tensor Core FlashMLA (Sparse DSA, INT8 KV -> 2x less HBM!).
// Launch: grid(rows, heads / 16, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_mla_attend_sparse_tc_q8(
    const __nv_bfloat16* __restrict__ qa,
    const int8_t* __restrict__ kv_q8,
    const float* __restrict__ kv_sc,
    const int32_t* __restrict__ tokens,
    const int32_t* __restrict__ counts,
    __nv_bfloat16* __restrict__ ol,
    int32_t pos,
    int32_t capacity,
    int32_t width,
    int32_t rows,
    int32_t heads,
    int32_t lw,
    float scale,
    int32_t tree_trunk_len,
    int32_t tree_b_start
) {
    (void)lw;
    const int r = blockIdx.x;
    const int h0 = blockIdx.y * 16;
    if (r >= rows || h0 >= heads) return;
    const bool is_branch_b = (tree_b_start > 0 && r >= tree_b_start);
    const int skip_lo = pos + tree_trunk_len;
    const int skip_hi = pos + tree_b_start;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int c0 = warp * 64;

    __shared__ __align__(16) __nv_bfloat16 sh_kv[16][520];
    __shared__ float sh_s_part[8][16][16];
    __shared__ __align__(16) __nv_bfloat16 sh_p[16][24];
    __shared__ float sh_alpha[16];
    __shared__ float sh_inv_l[16];
    __shared__ int sh_k_valid[16];

    const int lc = (tid & 63) * 8;
    const int lr0 = tid >> 6;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int hl = i * 4 + lr0;
        const __nv_bfloat16* qsrc = qa + ((size_t)r * heads + h0 + hl) * 512 + lc;
        *reinterpret_cast<uint4*>(&sh_kv[hl][lc]) = *reinterpret_cast<const uint4*>(qsrc);
    }
    __syncthreads();

    uint32_t q_frag[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int k0 = c0 + ks * 16 + 2 * t;
        q_frag[ks][0] = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0]);
        q_frag[ks][1] = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0]);
        q_frag[ks][2] = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0 + 8]);
        q_frag[ks][3] = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0 + 8]);
    }
    __syncthreads();

    float o_acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) o_acc[nt][c] = 0.0f;

    const int h_sub = lane >> 4;
    const int kk_lane = lane & 15;
    const int hl_sm = warp * 2 + h_sub;
    float m_max = -1e30f;
    float l_sum = 0.0f;

    const int cnt = counts[r];
    const bool is_sparse = cnt > 0;
    const int n_iter = is_sparse ? cnt : (pos + r + 1);
    const int32_t* tok_row = tokens + (size_t)r * width;

    for (int jb = 0; jb < n_iter; jb += 16) {
        const int n_k = min(16, n_iter - jb);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int kk = i * 4 + lr0;
            if (kk < n_k) {
                const int kp = is_sparse ? tok_row[jb + kk] : (jb + kk);
                const bool ok = (kp >= 0) && !(is_branch_b && kp >= skip_lo && kp < skip_hi);
                if (ok) {
                    const int slot = kp % capacity;
                    const float sc_k = __ldg(kv_sc + slot);
                    const uint2 qu = __ldg(reinterpret_cast<const uint2*>(kv_q8 + (size_t)slot * 512 + lc));
                    const int8_t* qh = reinterpret_cast<const int8_t*>(&qu);
                    __nv_bfloat162 bh[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        bh[j] = __floats2bfloat162_rn((float)qh[2 * j] * sc_k, (float)qh[2 * j + 1] * sc_k);
                    }
                    *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = *reinterpret_cast<const uint4*>(bh);
                    if (lc == 0) sh_k_valid[kk] = 1;
                } else {
                    *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = make_uint4(0u, 0u, 0u, 0u);
                    if (lc == 0) sh_k_valid[kk] = 0;
                }
            } else {
                *reinterpret_cast<uint4*>(&sh_kv[kk][lc]) = make_uint4(0u, 0u, 0u, 0u);
                if (lc == 0) sh_k_valid[kk] = 0;
            }
        }
        __syncthreads();

        float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k0 = c0 + ks * 16 + 2 * t;
            const uint32_t b00 = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0]);
            const uint32_t b01 = *reinterpret_cast<const uint32_t*>(&sh_kv[g][k0 + 8]);
            const uint32_t b10 = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0]);
            const uint32_t b11 = *reinterpret_cast<const uint32_t*>(&sh_kv[g + 8][k0 + 8]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(s0[0]), "+f"(s0[1]), "+f"(s0[2]), "+f"(s0[3])
                : "r"(q_frag[ks][0]), "r"(q_frag[ks][1]), "r"(q_frag[ks][2]), "r"(q_frag[ks][3]), "r"(b00), "r"(b01)
            );
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(s1[0]), "+f"(s1[1]), "+f"(s1[2]), "+f"(s1[3])
                : "r"(q_frag[ks][0]), "r"(q_frag[ks][1]), "r"(q_frag[ks][2]), "r"(q_frag[ks][3]), "r"(b10), "r"(b11)
            );
        }
        sh_s_part[warp][g][2 * t]             = s0[0];
        sh_s_part[warp][g][2 * t + 1]         = s0[1];
        sh_s_part[warp][g + 8][2 * t]         = s0[2];
        sh_s_part[warp][g + 8][2 * t + 1]     = s0[3];
        sh_s_part[warp][g][8 + 2 * t]         = s1[0];
        sh_s_part[warp][g][8 + 2 * t + 1]     = s1[1];
        sh_s_part[warp][g + 8][8 + 2 * t]     = s1[2];
        sh_s_part[warp][g + 8][8 + 2 * t + 1] = s1[3];
        __syncthreads();

        float sc = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) sc += sh_s_part[w][hl_sm][kk_lane];
        const bool valid = (kk_lane < n_k) && (sh_k_valid[kk_lane] != 0);
        sc = valid ? (sc * scale) : -1e30f;

        float m_blk = sc;
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 8));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 4));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 2));
        m_blk = fmaxf(m_blk, __shfl_xor_sync(0xffffffffu, m_blk, 1));

        if (m_blk > -1e29f) {
            const float m_new = fmaxf(m_max, m_blk);
            const float alpha = (m_max > -1e29f) ? expf(m_max - m_new) : 0.0f;
            const float p = valid ? expf(sc - m_new) : 0.0f;
            float l_blk = p;
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 8);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 4);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 2);
            l_blk += __shfl_xor_sync(0xffffffffu, l_blk, 1);
            l_sum = fmaf(l_sum, alpha, l_blk);
            m_max = m_new;
            sh_p[hl_sm][kk_lane] = __float2bfloat16_rn(p);
            if (kk_lane == 0) sh_alpha[hl_sm] = alpha;
        } else {
            sh_p[hl_sm][kk_lane] = __float2bfloat16_rn(0.0f);
            if (kk_lane == 0) sh_alpha[hl_sm] = 1.0f;
        }
        __syncthreads();

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(&sh_p[g][2 * t]);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(&sh_p[g + 8][2 * t]);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(&sh_p[g][8 + 2 * t]);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(&sh_p[g + 8][8 + 2 * t]);
        const float al0 = sh_alpha[g];
        const float al1 = sh_alpha[g + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= al0;
            o_acc[nt][1] *= al0;
            o_acc[nt][2] *= al1;
            o_acc[nt][3] *= al1;
            const int vc = c0 + nt * 8 + g;
            const uint16_t v0a = *reinterpret_cast<const uint16_t*>(&sh_kv[2 * t][vc]);
            const uint16_t v0b = *reinterpret_cast<const uint16_t*>(&sh_kv[2 * t + 1][vc]);
            const uint16_t v1a = *reinterpret_cast<const uint16_t*>(&sh_kv[8 + 2 * t][vc]);
            const uint16_t v1b = *reinterpret_cast<const uint16_t*>(&sh_kv[8 + 2 * t + 1][vc]);
            const uint32_t vb0 = (uint32_t)v0a | ((uint32_t)v0b << 16);
            const uint32_t vb1 = (uint32_t)v1a | ((uint32_t)v1b << 16);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(o_acc[nt][0]), "+f"(o_acc[nt][1]), "+f"(o_acc[nt][2]), "+f"(o_acc[nt][3])
                : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(vb0), "r"(vb1)
            );
        }
        __syncthreads();
    }

    if (kk_lane == 0) sh_inv_l[hl_sm] = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    __syncthreads();

    const float il0 = sh_inv_l[g];
    const float il1 = sh_inv_l[g + 8];
    __nv_bfloat16* orow0 = ol + ((size_t)r * heads + h0 + g) * 512 + c0;
    __nv_bfloat16* orow1 = ol + ((size_t)r * heads + h0 + g + 8) * 512 + c0;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int oc = nt * 8 + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(orow0 + oc) =
            __floats2bfloat162_rn(o_acc[nt][0] * il0, o_acc[nt][1] * il0);
        *reinterpret_cast<__nv_bfloat162*>(orow1 + oc) =
            __floats2bfloat162_rn(o_acc[nt][2] * il1, o_acc[nt][3] * il1);
    }
}

// Race-free sequential compaction of Branch B rows `[pos_src .. pos_src+n_rows)` -> `[pos_dst .. pos_dst+n_rows)`
// across ALL MLA layers in ONE launch (`pos_dst < pos_src`, sequential `j` loop guarantees 0 overlap races!).
// Launch: grid(num_layers, 1, 1), block(256, 1, 1)
__global__ void m3tis_mla_compact_kv_bf16(
    __nv_bfloat16* __restrict__ kv_all, // [num_layers, capacity, lw]
    int32_t pos_dst,
    int32_t pos_src,
    int32_t capacity,
    int32_t n_rows,
    int32_t lw
) {
    const int l = blockIdx.x;
    const int t = threadIdx.x;
    __nv_bfloat16* kv = kv_all + (size_t)l * (size_t)capacity * (size_t)lw;
    for (int j = 0; j < n_rows; ++j) {
        const int s_dst = (pos_dst + j) % capacity;
        const int s_src = (pos_src + j) % capacity;
        for (int d = t; d < lw; d += 256) {
            kv[(size_t)s_dst * lw + d] = kv[(size_t)s_src * lw + d];
        }
        __syncthreads();
    }
}

// Race-free sequential compaction of Branch B INT8 KV + scales across ALL MLA layers in ONE launch!
// Launch: grid(num_layers, 1, 1), block(256, 1, 1)
__global__ void m3tis_mla_compact_kv_q8(
    int8_t* __restrict__ kv_q8_all, // [num_layers, capacity, lw]
    float* __restrict__ kv_sc_all,  // [num_layers, capacity]
    int32_t pos_dst,
    int32_t pos_src,
    int32_t capacity,
    int32_t n_rows,
    int32_t lw
) {
    const int l = blockIdx.x;
    const int t = threadIdx.x;
    int8_t* q8 = kv_q8_all + (size_t)l * (size_t)capacity * (size_t)lw;
    float* sc = kv_sc_all + (size_t)l * (size_t)capacity;
    for (int j = 0; j < n_rows; ++j) {
        const int s_dst = (pos_dst + j) % capacity;
        const int s_src = (pos_src + j) % capacity;
        for (int d = t; d < lw; d += 256) {
            q8[(size_t)s_dst * lw + d] = q8[(size_t)s_src * lw + d];
        }
        if (t == 0) sc[s_dst] = sc[s_src];
        __syncthreads();
    }
}

// Race-free sequential compaction of Branch B DSA `ik` & `ig` ring slots across ALL MLA layers in ONE launch!
// Launch: grid(num_layers, 1, 1), block(128, 1, 1)
__global__ void m3tis_dsa_compact_ik_ig(
    __nv_bfloat16* __restrict__ ik_all, // [num_layers, ik_cap, 128]
    __nv_bfloat16* __restrict__ ig_all, // [num_layers, ik_cap, 128]
    int32_t pos_dst,
    int32_t pos_src,
    int32_t ik_cap,
    int32_t n_rows
) {
    constexpr int D = 128;
    const int l = blockIdx.x;
    const int d = threadIdx.x;
    __nv_bfloat16* ik = ik_all + (size_t)l * (size_t)ik_cap * D;
    __nv_bfloat16* ig = ig_all + (size_t)l * (size_t)ik_cap * D;
    for (int j = 0; j < n_rows; ++j) {
        const int s_dst = (pos_dst + j) % ik_cap;
        const int s_src = (pos_src + j) % ik_cap;
        ik[(size_t)s_dst * D + d] = ik[(size_t)s_src * D + d];
        ig[(size_t)s_dst * D + d] = ig[(size_t)s_src * D + d];
        __syncthreads();
    }
}

// Vectorized `dst[i] += src[i]` for accumulating CPU-offloaded hybrid MoE partial vectors into `st.branch`!
// Launch: grid(((n / 4) + 255) / 256, 1, 1), block(256, 1, 1)
__global__ void m3tis_add_f32_inplace(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int32_t n
) {
    const int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx4 + 3 < n) {
        float4 d = *reinterpret_cast<const float4*>(dst + idx4);
        float4 s = *reinterpret_cast<const float4*>(src + idx4);
        d.x += s.x; d.y += s.y; d.z += s.z; d.w += s.w;
        *reinterpret_cast<float4*>(dst + idx4) = d;
    } else {
        for (int i = idx4; i < n && i < idx4 + 4; ++i) dst[i] += src[i];
    }
}

} // extern "C"
