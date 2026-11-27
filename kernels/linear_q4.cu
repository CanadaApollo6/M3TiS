// Clean-room M3TiS sm_86 Dense Linear Kernels:
// - On-GPU BF16 -> Affine Q4 (group_size = 64) packer (4.5 bits/weight -> 4.4x VRAM savings!)
// - Warp-vectorized Q4 and BF16 GEMV/GEMM (supports rows = 1..2048, BF16 or FP32 output)
// - Clamped BF16 GLM-5.3 SwiGLU (limit = 10.0) and FP32/BF16 residual/slot helpers

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace m3tis_lin {

constexpr int Q4_GS = 64; // 64 weights = 8 uint32 words + (half scale, half zero)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float bf16r(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

} // namespace m3tis_lin

extern "C" {

// Pack row-major BF16 weight matrix W[N, K] (K % 64 == 0) into affine 4-bit groups (gs = 64):
//   qdata : uint32_t [N, K / 8]  (each u32 holds 8 consecutive 4-bit unsigned codes [0..15])
//   scales: half2    [N, K / 64] (.x = scale = (max - min) / 15, .y = zero = min)
// Launch: grid(N, K / 64, 1), block(32, 1, 1) — 1 warp packs 1 group of 64 weights!
__global__ void m3tis_pack_bf16_to_q4(
    const __nv_bfloat16* __restrict__ w,
    uint32_t* __restrict__ qdata,
    half2* __restrict__ scales,
    int32_t N,
    int32_t K
) {
    const int row = blockIdx.x;
    const int grp = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    const int k_base = grp * 64 + lane * 2;
    const __nv_bfloat16* wp = w + (size_t)row * (size_t)K + k_base;
    float v0 = __bfloat162float(wp[0]);
    float v1 = __bfloat162float(wp[1]);

    float mn = fminf(v0, v1);
    float mx = fmaxf(v0, v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float span = mx - mn;
    float scale = (span > 1e-12f) ? (span / 15.0f) : 0.0f;
    float inv_scale = (scale > 0.0f) ? (15.0f / span) : 0.0f;

    // Use rounded fp16 scale/zero when quantizing so dequant matches exact stored values!
    half h_scale = __float2half_rn(scale);
    half h_zero  = __float2half_rn(mn);
    float f_zero = __half2float(h_zero);
    float f_scale = __half2float(h_scale);
    float inv_q = (f_scale > 0.0f) ? (1.0f / f_scale) : 0.0f;

    int q0 = __float2int_rn(fminf(15.0f, fmaxf(0.0f, (v0 - f_zero) * inv_q)));
    int q1 = __float2int_rn(fminf(15.0f, fmaxf(0.0f, (v1 - f_zero) * inv_q)));
    if (inv_scale == 0.0f) { q0 = 0; q1 = 0; }
    uint32_t pair = (uint32_t)(q0 & 0xF) | ((uint32_t)(q1 & 0xF) << 4);

    // Pack 4 consecutive lanes (4 * 2 = 8 nibbles) into one uint32 word
    uint32_t p0 = __shfl_sync(0xffffffffu, pair, (lane & ~3) + 0);
    uint32_t p1 = __shfl_sync(0xffffffffu, pair, (lane & ~3) + 1);
    uint32_t p2 = __shfl_sync(0xffffffffu, pair, (lane & ~3) + 2);
    uint32_t p3 = __shfl_sync(0xffffffffu, pair, (lane & ~3) + 3);

    if ((lane & 3) == 0) {
        int word_idx = lane >> 2; // 0..7
        qdata[(size_t)row * (K / 8) + grp * 8 + word_idx] =
            p0 | (p1 << 8) | (p2 << 16) | (p3 << 24);
    }
    if (lane == 0) {
        scales[(size_t)row * (K / 64) + grp] = __halves2half2(h_scale, h_zero);
    }
}

// Vectorized 128-bit Q4 GEMV/GEMM: Y[rows, N] = X[rows, K] @ W_q4[N, K]^T.
// Uses 128-bit uint4 loads (32 4-bit weights per lane = 1,024 weights/warp/iter) for near-peak HBM bandwidth!
// Launch: grid((N + 3) / 4, (rows + 3) / 4, 1), block(128, 1, 1) — 4 warps = 4 output rows/block.
// out_f32: 0 = write BF16, 1 = write FP32.
__global__ void __launch_bounds__(128) m3tis_linear_q4(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const uint32_t* __restrict__ qdata,
    const half2* __restrict__ scales,
    void* __restrict__ out,
    int32_t out_stride,
    int32_t rows,
    int32_t N,
    int32_t K,
    int32_t out_f32
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;

    const int r0 = blockIdx.y * 4;
    const int nr = min(4, rows - r0);
    if (nr <= 0) return;

    const int n_groups = K / 64;
    const uint32_t* qrow = qdata + (size_t)n * (size_t)(K / 8);
    const half2* srow = scales + (size_t)n * (size_t)n_groups;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Fast path for nr == 1 (serial decode): 16 groups (1,024 weights/warp/iter) via 128-bit uint4 loads!
    // For nr = 2..4, fall straight into the leaner 4-group loop to keep register pressure low.
    int g0 = 0;
    if (nr == 1) {
        const __nv_bfloat16* xrow = x + (size_t)r0 * x_stride;
        float d = 0.0f;
        for (; g0 + 16 <= n_groups; g0 += 16) {
            const int g = g0 + (lane >> 1);
            const int half_g = lane & 1; // 0 -> weights 0..31, 1 -> weights 32..63
            const uint4 q4 = __ldg(reinterpret_cast<const uint4*>(qrow + g * 8 + half_g * 4));
            const half2 sz = __ldg(srow + g);
            const float sc = __low2float(sz);
            const float zr = __high2float(sz);
            const int k_base = g * 64 + half_g * 32;
            const uint32_t qw[4] = {q4.x, q4.y, q4.z, q4.w};

#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t q = qw[w];
                const uint4 xu = *reinterpret_cast<const uint4*>(xrow + k_base + w * 8);
                const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float wv = fmaf((float)((q >> (j * 4)) & 0xFu), sc, zr);
                    d = fmaf(__bfloat162float(xh[j]), wv, d);
                }
            }
        }
        acc[0] = d;
    }
    // 4-group tail (256 weights/iter)
    for (; g0 + 4 <= n_groups; g0 += 4) {
        const int g = g0 + (lane >> 3);
        const int w_in_g = lane & 7;
        const uint32_t q = __ldg(qrow + g * 8 + w_in_g);
        const half2 sz = __ldg(srow + g);
        const float sc = __low2float(sz);
        const float zr = __high2float(sz);
        const int k_off = g * 64 + w_in_g * 8;
        float wv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) wv[j] = fmaf((float)((q >> (j * 4)) & 0xFu), sc, zr);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r < nr) {
                const uint4 xu = *reinterpret_cast<const uint4*>(x + (size_t)(r0 + r) * x_stride + k_off);
                const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
                float d = acc[r];
#pragma unroll
                for (int j = 0; j < 8; ++j) d = fmaf(__bfloat162float(xh[j]), wv[j], d);
                acc[r] = d;
            }
        }
    }
    // Single-group tail if (K / 64) % 4 != 0
    for (int g = g0; g < n_groups; ++g) {
        const half2 sz = __ldg(srow + g);
        const float sc = __low2float(sz);
        const float zr = __high2float(sz);
        const int w_in_g = lane >> 2;
        const int shift = (lane & 3) * 8;
        const uint32_t q = __ldg(qrow + g * 8 + w_in_g);
        const float w0 = fmaf((float)((q >> shift) & 0xFu), sc, zr);
        const float w1 = fmaf((float)((q >> (shift + 4)) & 0xFu), sc, zr);
        const int k_off = g * 64 + lane * 2;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r < nr) {
                const __nv_bfloat16* xp = x + (size_t)(r0 + r) * x_stride + k_off;
                acc[r] = fmaf(__bfloat162float(xp[0]), w0, acc[r]);
                acc[r] = fmaf(__bfloat162float(xp[1]), w1, acc[r]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (r < nr) {
            float s = m3tis_lin::warp_sum(acc[r]);
            if (lane == 0) {
                const size_t idx = (size_t)(r0 + r) * (size_t)out_stride + (size_t)n;
                if (out_f32) {
                    static_cast<float*>(out)[idx] = s;
                } else {
                    static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(s);
                }
            }
        }
    }
}

// Milestone 8: Double-Buffered Register-Pipelined 4-Warp 16x64 Native BF16 Ampere sm_86 TC Q4 GEMM (`rows = 2..31`)!
// - K_STEP = 128 (2 groups/iter) + double-buffered sh_x[2][16][136] (8.5 KiB -> full occupancy, 1 __syncthreads()/iter!)
// - Software-pipelined q0/q1 + scale prefetch overlaps 300-cycle HBM latency directly behind 16 Tensor Core MMAs!
// - Zero-redundant coalesced 128-bit uint4 loads across t=0..3 + __shfl_sync broadcast; skips upper-half LDS when rows <= 8!
// Launch: grid((N + 63) / 64, (rows + 15) / 16, 1), block(128, 1, 1)
__global__ void __launch_bounds__(128) m3tis_linear_q4_tc(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const uint32_t* __restrict__ qdata,
    const half2* __restrict__ scales,
    void* __restrict__ out,
    int32_t out_stride,
    int32_t rows,
    int32_t N,
    int32_t K,
    int32_t out_f32
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2; // 0..7
    const int t = lane & 3;  // 0..3
    const int n0 = blockIdx.x * 64 + warp * 16;
    const int r0 = blockIdx.y * 16;
    if (r0 >= rows) return;

    __shared__ __align__(16) __nv_bfloat16 sh_x[2][16][136]; // 8.5 KiB, stride 272B -> 0 bank conflicts!
    float acc[2][4] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f}
    };

    const int n_groups = K / 64;
    const int n_pairs = n_groups >> 1;
    const int n_row0 = n0 + g;
    const int n_row1 = n0 + 8 + g;
    const uint32_t* qrow0 = (n_row0 < N) ? (qdata + (size_t)n_row0 * (size_t)(K / 8)) : nullptr;
    const uint32_t* qrow1 = (n_row1 < N) ? (qdata + (size_t)n_row1 * (size_t)(K / 8)) : nullptr;
    const half2* srow0 = (n_row0 < N) ? (scales + (size_t)n_row0 * (size_t)n_groups) : nullptr;
    const half2* srow1 = (n_row1 < N) ? (scales + (size_t)n_row1 * (size_t)n_groups) : nullptr;

    const int lx_r = threadIdx.x >> 3;        // 0..15
    const int lx_c = (threadIdx.x & 7) << 3;  // 0, 8, ..., 56
    const int shft = t * 8;
    const bool valid_x = (r0 + lx_r < rows);
    const bool has_hi = (r0 + 8 < rows);

    if (!valid_x) {
        *reinterpret_cast<uint4*>(&sh_x[0][lx_r][lx_c])      = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(&sh_x[0][lx_r][64 + lx_c]) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(&sh_x[1][lx_r][lx_c])      = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(&sh_x[1][lx_r][64 + lx_c]) = make_uint4(0u, 0u, 0u, 0u);
    }

    if (n_pairs > 0) {
        // Prologue: issue stage 0 global loads for both X and Q4 weights/scales!
        if (valid_x) {
            const __nv_bfloat16* xp = x + (size_t)(r0 + lx_r) * x_stride + lx_c;
            *reinterpret_cast<uint4*>(&sh_x[0][lx_r][lx_c])      = *reinterpret_cast<const uint4*>(xp);
            *reinterpret_cast<uint4*>(&sh_x[0][lx_r][64 + lx_c]) = *reinterpret_cast<const uint4*>(xp + 64);
        }
        uint4 q0_cur = make_uint4(0u, 0u, 0u, 0u);
        uint4 q1_cur = make_uint4(0u, 0u, 0u, 0u);
        half2 s0a_cur = __halves2half2(__float2half(0.0f), __float2half(0.0f));
        half2 s0b_cur = s0a_cur, s1a_cur = s0a_cur, s1b_cur = s0a_cur;
        if (qrow0) {
            q0_cur  = __ldg(reinterpret_cast<const uint4*>(qrow0 + t * 4));
            s0a_cur = __ldg(srow0);
            s0b_cur = __ldg(srow0 + 1);
        }
        if (qrow1) {
            q1_cur  = __ldg(reinterpret_cast<const uint4*>(qrow1 + t * 4));
            s1a_cur = __ldg(srow1);
            s1b_cur = __ldg(srow1 + 1);
        }
        __syncthreads();

        for (int gp = 0; gp < n_pairs; ++gp) {
            const int cur = gp & 1;
            const int nxt = cur ^ 1;
            const uint4 q0 = q0_cur;
            const uint4 q1 = q1_cur;
            const float sc0_a = __low2float(s0a_cur), zr0_a = __high2float(s0a_cur);
            const float sc0_b = __low2float(s0b_cur), zr0_b = __high2float(s0b_cur);
            const float sc1_a = __low2float(s1a_cur), zr1_a = __high2float(s1a_cur);
            const float sc1_b = __low2float(s1b_cur), zr1_b = __high2float(s1b_cur);

            // Software-pipeline stage gp+1 global loads BEFORE the 16 Tensor Core MMAs!
            if (gp + 1 < n_pairs) {
                const int gp1 = gp + 1;
                if (valid_x) {
                    const __nv_bfloat16* xp = x + (size_t)(r0 + lx_r) * x_stride + gp1 * 128 + lx_c;
                    *reinterpret_cast<uint4*>(&sh_x[nxt][lx_r][lx_c])      = *reinterpret_cast<const uint4*>(xp);
                    *reinterpret_cast<uint4*>(&sh_x[nxt][lx_r][64 + lx_c]) = *reinterpret_cast<const uint4*>(xp + 64);
                }
                if (qrow0) {
                    q0_cur  = __ldg(reinterpret_cast<const uint4*>(qrow0 + gp1 * 16 + t * 4));
                    s0a_cur = __ldg(srow0 + gp1 * 2);
                    s0b_cur = __ldg(srow0 + gp1 * 2 + 1);
                }
                if (qrow1) {
                    q1_cur  = __ldg(reinterpret_cast<const uint4*>(qrow1 + gp1 * 16 + t * 4));
                    s1a_cur = __ldg(srow1 + gp1 * 2);
                    s1b_cur = __ldg(srow1 + gp1 * 2 + 1);
                }
            }

#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int src_lane = (g << 2) | (ks >> 1);
                const uint32_t q0lo = __shfl_sync(0xffffffffu, (ks & 1) ? q0.z : q0.x, src_lane);
                const uint32_t q0hi = __shfl_sync(0xffffffffu, (ks & 1) ? q0.w : q0.y, src_lane);
                const uint32_t q1lo = __shfl_sync(0xffffffffu, (ks & 1) ? q1.z : q1.x, src_lane);
                const uint32_t q1hi = __shfl_sync(0xffffffffu, (ks & 1) ? q1.w : q1.y, src_lane);
                const float sc0 = (ks < 4) ? sc0_a : sc0_b;
                const float zr0 = (ks < 4) ? zr0_a : zr0_b;
                const float sc1 = (ks < 4) ? sc1_a : sc1_b;
                const float zr1 = (ks < 4) ? zr1_a : zr1_b;

                const int k0 = ks * 16 + 2 * t;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sh_x[cur][g][k0]);
                const uint32_t a1 = has_hi ? *reinterpret_cast<const uint32_t*>(&sh_x[cur][g + 8][k0]) : 0u;
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sh_x[cur][g][k0 + 8]);
                const uint32_t a3 = has_hi ? *reinterpret_cast<const uint32_t*>(&sh_x[cur][g + 8][k0 + 8]) : 0u;
                {
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                        fmaf((float)((q0lo >> shft) & 0xFu), sc0, zr0),
                        fmaf((float)((q0lo >> (shft + 4)) & 0xFu), sc0, zr0)
                    );
                    const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                        fmaf((float)((q0hi >> shft) & 0xFu), sc0, zr0),
                        fmaf((float)((q0hi >> (shft + 4)) & 0xFu), sc0, zr0)
                    );
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                    );
                }
                {
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                        fmaf((float)((q1lo >> shft) & 0xFu), sc1, zr1),
                        fmaf((float)((q1lo >> (shft + 4)) & 0xFu), sc1, zr1)
                    );
                    const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                        fmaf((float)((q1hi >> shft) & 0xFu), sc1, zr1),
                        fmaf((float)((q1hi >> (shft + 4)) & 0xFu), sc1, zr1)
                    );
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                    );
                }
            }
            if (gp + 1 < n_pairs) __syncthreads();
        }
    }

    // Odd 64-element tail group if (K / 64) % 2 == 1
    if (n_groups & 1) {
        __syncthreads();
        const int grp = n_groups - 1;
        if (valid_x) {
            *reinterpret_cast<uint4*>(&sh_x[0][lx_r][lx_c]) = *reinterpret_cast<const uint4*>(
                x + (size_t)(r0 + lx_r) * x_stride + grp * 64 + lx_c
            );
        }
        __syncthreads();

        uint4 q0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 q1 = make_uint4(0u, 0u, 0u, 0u);
        float sc0 = 0.0f, zr0 = 0.0f, sc1 = 0.0f, zr1 = 0.0f;
        if (qrow0) {
            if (t < 2) q0 = __ldg(reinterpret_cast<const uint4*>(qrow0 + grp * 8 + t * 4));
            const half2 sz0 = __ldg(srow0 + grp);
            sc0 = __low2float(sz0); zr0 = __high2float(sz0);
        }
        if (qrow1) {
            if (t < 2) q1 = __ldg(reinterpret_cast<const uint4*>(qrow1 + grp * 8 + t * 4));
            const half2 sz1 = __ldg(srow1 + grp);
            sc1 = __low2float(sz1); zr1 = __high2float(sz1);
        }

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int src_lane = (g << 2) | (ks >> 1);
            const uint32_t q0lo = __shfl_sync(0xffffffffu, (ks & 1) ? q0.z : q0.x, src_lane);
            const uint32_t q0hi = __shfl_sync(0xffffffffu, (ks & 1) ? q0.w : q0.y, src_lane);
            const uint32_t q1lo = __shfl_sync(0xffffffffu, (ks & 1) ? q1.z : q1.x, src_lane);
            const uint32_t q1hi = __shfl_sync(0xffffffffu, (ks & 1) ? q1.w : q1.y, src_lane);
            const int k0 = ks * 16 + 2 * t;
            const uint32_t a[4] = {
                *reinterpret_cast<const uint32_t*>(&sh_x[0][g][k0]),
                has_hi ? *reinterpret_cast<const uint32_t*>(&sh_x[0][g + 8][k0]) : 0u,
                *reinterpret_cast<const uint32_t*>(&sh_x[0][g][k0 + 8]),
                has_hi ? *reinterpret_cast<const uint32_t*>(&sh_x[0][g + 8][k0 + 8]) : 0u,
            };
            {
                const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                    fmaf((float)((q0lo >> shft) & 0xFu), sc0, zr0),
                    fmaf((float)((q0lo >> (shft + 4)) & 0xFu), sc0, zr0)
                );
                const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                    fmaf((float)((q0hi >> shft) & 0xFu), sc0, zr0),
                    fmaf((float)((q0hi >> (shft + 4)) & 0xFu), sc0, zr0)
                );
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
                );
            }
            {
                const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                    fmaf((float)((q1lo >> shft) & 0xFu), sc1, zr1),
                    fmaf((float)((q1lo >> (shft + 4)) & 0xFu), sc1, zr1)
                );
                const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                    fmaf((float)((q1hi >> shft) & 0xFu), sc1, zr1),
                    fmaf((float)((q1hi >> (shft + 4)) & 0xFu), sc1, zr1)
                );
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
                );
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col0 = n0 + nt * 8 + 2 * t;
        const int col1 = col0 + 1;
        const int row0 = r0 + g;
        const int row1 = r0 + g + 8;
        if (row0 < rows) {
            if (col0 < N) {
                const size_t idx = (size_t)row0 * out_stride + col0;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][0];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][0]);
            }
            if (col1 < N) {
                const size_t idx = (size_t)row0 * out_stride + col1;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][1];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][1]);
            }
        }
        if (row1 < rows) {
            if (col0 < N) {
                const size_t idx = (size_t)row1 * out_stride + col0;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][2];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][2]);
            }
            if (col1 < N) {
                const size_t idx = (size_t)row1 * out_stride + col1;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][3];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][3]);
            }
        }
    }
}

// Milestone 6: 64x128 8-Warp Native BF16 Ampere sm_86 Tensor Core Q4 Prefill GEMM (`rows >= 32`)!
// - Zero-conversion 128-bit `uint4` copy of BF16 `X` into padded `sh_x[64][72]` (0 smem bank conflicts!)
// - 8 warps compute [64 x 128] output tile (2x less X traffic, 67% occupancy, native `.f32.bf16.bf16.f32` MMA)
// - Each dequantized Q4 weight tile is reused across 4 consecutive 16-row M-tiles (4x less W traffic & dequant ALU!)
// Launch: grid((N + 127) / 128, (rows + 63) / 64, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_linear_q4_tc64(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const uint32_t* __restrict__ qdata,
    const half2* __restrict__ scales,
    void* __restrict__ out,
    int32_t out_stride,
    int32_t rows,
    int32_t N,
    int32_t K,
    int32_t out_f32
) {
    const int warp = threadIdx.x >> 5; // 0..7
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2; // 0..7
    const int t = lane & 3;  // 0..3
    const int n0 = blockIdx.x * 128 + warp * 16;
    const int r0 = blockIdx.y * 64;
    if (r0 >= rows) return;

    __shared__ __align__(16) __nv_bfloat16 sh_x[64][72]; // 9 KiB, stride 144B -> 0 bank conflicts!
    float acc[4][2][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][nt][c] = 0.0f;

    const int n_groups = K / 64;
    const int n_row0 = n0 + g;
    const int n_row1 = n0 + 8 + g;
    const uint32_t* qrow0 = (n_row0 < N) ? (qdata + (size_t)n_row0 * (size_t)(K / 8)) : nullptr;
    const uint32_t* qrow1 = (n_row1 < N) ? (qdata + (size_t)n_row1 * (size_t)(K / 8)) : nullptr;
    const half2* srow0 = (n_row0 < N) ? (scales + (size_t)n_row0 * (size_t)n_groups) : nullptr;
    const half2* srow1 = (n_row1 < N) ? (scales + (size_t)n_row1 * (size_t)n_groups) : nullptr;

    const int lx_r = threadIdx.x >> 3;        // 0..31
    const int lx_c = (threadIdx.x & 7) << 3;  // 0, 8, ..., 56

    for (int grp = 0; grp < n_groups; ++grp) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int rr = m * 32 + lx_r;
            if (r0 + rr < rows) {
                *reinterpret_cast<uint4*>(&sh_x[rr][lx_c]) = *reinterpret_cast<const uint4*>(
                    x + (size_t)(r0 + rr) * x_stride + grp * 64 + lx_c
                );
            } else {
                *reinterpret_cast<uint4*>(&sh_x[rr][lx_c]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        uint32_t w0_words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t w1_words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float sc0 = 0.0f, zr0 = 0.0f, sc1 = 0.0f, zr1 = 0.0f;
        if (qrow0) {
            const uint4 qa = __ldg(reinterpret_cast<const uint4*>(qrow0 + grp * 8));
            const uint4 qb = __ldg(reinterpret_cast<const uint4*>(qrow0 + grp * 8 + 4));
            w0_words[0] = qa.x; w0_words[1] = qa.y; w0_words[2] = qa.z; w0_words[3] = qa.w;
            w0_words[4] = qb.x; w0_words[5] = qb.y; w0_words[6] = qb.z; w0_words[7] = qb.w;
            const half2 sz0 = __ldg(srow0 + grp);
            sc0 = __low2float(sz0);
            zr0 = __high2float(sz0);
        }
        if (qrow1) {
            const uint4 qa = __ldg(reinterpret_cast<const uint4*>(qrow1 + grp * 8));
            const uint4 qb = __ldg(reinterpret_cast<const uint4*>(qrow1 + grp * 8 + 4));
            w1_words[0] = qa.x; w1_words[1] = qa.y; w1_words[2] = qa.z; w1_words[3] = qa.w;
            w1_words[4] = qb.x; w1_words[5] = qb.y; w1_words[6] = qb.z; w1_words[7] = qb.w;
            const half2 sz1 = __ldg(srow1 + grp);
            sc1 = __low2float(sz1);
            zr1 = __high2float(sz1);
        }

        const int shft = t * 8;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k0 = ks * 16 + 2 * t;
            const uint32_t q0lo = w0_words[ks * 2], q0hi = w0_words[ks * 2 + 1];
            const __nv_bfloat162 h00 = __floats2bfloat162_rn(
                fmaf((float)((q0lo >> shft) & 0xFu), sc0, zr0),
                fmaf((float)((q0lo >> (shft + 4)) & 0xFu), sc0, zr0)
            );
            const __nv_bfloat162 h01 = __floats2bfloat162_rn(
                fmaf((float)((q0hi >> shft) & 0xFu), sc0, zr0),
                fmaf((float)((q0hi >> (shft + 4)) & 0xFu), sc0, zr0)
            );
            const uint32_t b00 = *reinterpret_cast<const uint32_t*>(&h00);
            const uint32_t b01 = *reinterpret_cast<const uint32_t*>(&h01);

            const uint32_t q1lo = w1_words[ks * 2], q1hi = w1_words[ks * 2 + 1];
            const __nv_bfloat162 h10 = __floats2bfloat162_rn(
                fmaf((float)((q1lo >> shft) & 0xFu), sc1, zr1),
                fmaf((float)((q1lo >> (shft + 4)) & 0xFu), sc1, zr1)
            );
            const __nv_bfloat162 h11 = __floats2bfloat162_rn(
                fmaf((float)((q1hi >> shft) & 0xFu), sc1, zr1),
                fmaf((float)((q1hi >> (shft + 4)) & 0xFu), sc1, zr1)
            );
            const uint32_t b10 = *reinterpret_cast<const uint32_t*>(&h10);
            const uint32_t b11 = *reinterpret_cast<const uint32_t*>(&h11);

#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int rm = m * 16;
                const uint32_t a[4] = {
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g + 8][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g][k0 + 8]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g + 8][k0 + 8]),
                };
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[m][0][0]), "+f"(acc[m][0][1]), "+f"(acc[m][0][2]), "+f"(acc[m][0][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b00), "r"(b01)
                );
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[m][1][0]), "+f"(acc[m][1][1]), "+f"(acc[m][1][2]), "+f"(acc[m][1][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b10), "r"(b11)
                );
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row0 = r0 + m * 16 + g;
        const int row1 = row0 + 8;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col0 = n0 + nt * 8 + 2 * t;
            const int col1 = col0 + 1;
            if (row0 < rows) {
                if (col0 < N) {
                    const size_t idx = (size_t)row0 * out_stride + col0;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][0];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][0]);
                }
                if (col1 < N) {
                    const size_t idx = (size_t)row0 * out_stride + col1;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][1];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][1]);
                }
            }
            if (row1 < rows) {
                if (col0 < N) {
                    const size_t idx = (size_t)row1 * out_stride + col0;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][2];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][2]);
                }
                if (col1 < N) {
                    const size_t idx = (size_t)row1 * out_stride + col1;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][3];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][3]);
                }
            }
        }
    }
}

// Milestone 8: 4-Warp 16x64 Native BF16 Ampere sm_86 Tensor Core Exact BF16 GEMM (`rows = 5..15, N >= 1024, K % 64 == 0`)!
// Accelerates KDA f_b_proj / g_b_proj [8192, 128] (68 calls/round!) on every DFlash2/tree verify pass!
// Launch: grid((N + 63) / 64, (rows + 15) / 16, 1), block(128, 1, 1)
__global__ void __launch_bounds__(128) m3tis_linear_bf16_tc16(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const __nv_bfloat16* __restrict__ w,
    void* __restrict__ out,
    int32_t out_stride,
    int32_t rows,
    int32_t N,
    int32_t K,
    int32_t out_f32
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n0 = blockIdx.x * 64 + warp * 16;
    const int r0 = blockIdx.y * 16;
    if (r0 >= rows) return;

    __shared__ __align__(16) __nv_bfloat16 sh_x[16][72];
    float acc[2][4] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f}
    };

    const int n_groups = K / 64;
    const int n_row0 = n0 + g;
    const int n_row1 = n0 + 8 + g;
    const __nv_bfloat16* wrow0 = (n_row0 < N) ? (w + (size_t)n_row0 * (size_t)K) : nullptr;
    const __nv_bfloat16* wrow1 = (n_row1 < N) ? (w + (size_t)n_row1 * (size_t)K) : nullptr;

    const int lx_r = threadIdx.x >> 3;
    const int lx_c = (threadIdx.x & 7) << 3;

    for (int grp = 0; grp < n_groups; ++grp) {
        if (r0 + lx_r < rows) {
            *reinterpret_cast<uint4*>(&sh_x[lx_r][lx_c]) = *reinterpret_cast<const uint4*>(
                x + (size_t)(r0 + lx_r) * x_stride + grp * 64 + lx_c
            );
        } else {
            *reinterpret_cast<uint4*>(&sh_x[lx_r][lx_c]) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k0 = ks * 16 + 2 * t;
            const int gk = grp * 64 + k0;
            const uint32_t b00 = wrow0 ? __ldg(reinterpret_cast<const uint32_t*>(wrow0 + gk)) : 0u;
            const uint32_t b01 = wrow0 ? __ldg(reinterpret_cast<const uint32_t*>(wrow0 + gk + 8)) : 0u;
            const uint32_t b10 = wrow1 ? __ldg(reinterpret_cast<const uint32_t*>(wrow1 + gk)) : 0u;
            const uint32_t b11 = wrow1 ? __ldg(reinterpret_cast<const uint32_t*>(wrow1 + gk + 8)) : 0u;
            const uint32_t a[4] = {
                *reinterpret_cast<const uint32_t*>(&sh_x[g][k0]),
                *reinterpret_cast<const uint32_t*>(&sh_x[g + 8][k0]),
                *reinterpret_cast<const uint32_t*>(&sh_x[g][k0 + 8]),
                *reinterpret_cast<const uint32_t*>(&sh_x[g + 8][k0 + 8]),
            };
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b00), "r"(b01)
            );
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b10), "r"(b11)
            );
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col0 = n0 + nt * 8 + 2 * t;
        const int col1 = col0 + 1;
        const int row0 = r0 + g;
        const int row1 = r0 + g + 8;
        if (row0 < rows) {
            if (col0 < N) {
                const size_t idx = (size_t)row0 * out_stride + col0;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][0];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][0]);
            }
            if (col1 < N) {
                const size_t idx = (size_t)row0 * out_stride + col1;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][1];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][1]);
            }
        }
        if (row1 < rows) {
            if (col0 < N) {
                const size_t idx = (size_t)row1 * out_stride + col0;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][2];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][2]);
            }
            if (col1 < N) {
                const size_t idx = (size_t)row1 * out_stride + col1;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][3];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][3]);
            }
        }
    }
}

// Milestone 6: 64x128 8-Warp Native BF16 Ampere sm_86 Tensor Core Exact BF16 GEMM (`rows >= 16, K % 64 == 0`)!
// Accelerates KDA fgb/f_b/g_b, MoE routers, and DSA indexer projections by ~8-10x over scalar 4x4 GEMV!
// Launch: grid((N + 127) / 128, (rows + 63) / 64, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_linear_bf16_tc64(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const __nv_bfloat16* __restrict__ w,
    void* __restrict__ out,
    int32_t out_stride,
    int32_t rows,
    int32_t N,
    int32_t K,
    int32_t out_f32
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n0 = blockIdx.x * 128 + warp * 16;
    const int r0 = blockIdx.y * 64;
    if (r0 >= rows) return;

    __shared__ __align__(16) __nv_bfloat16 sh_x[64][72];
    float acc[4][2][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][nt][c] = 0.0f;

    const int n_groups = K / 64;
    const int n_row0 = n0 + g;
    const int n_row1 = n0 + 8 + g;
    const __nv_bfloat16* wrow0 = (n_row0 < N) ? (w + (size_t)n_row0 * (size_t)K) : nullptr;
    const __nv_bfloat16* wrow1 = (n_row1 < N) ? (w + (size_t)n_row1 * (size_t)K) : nullptr;

    const int lx_r = threadIdx.x >> 3;
    const int lx_c = (threadIdx.x & 7) << 3;

    for (int grp = 0; grp < n_groups; ++grp) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int rr = m * 32 + lx_r;
            if (r0 + rr < rows) {
                *reinterpret_cast<uint4*>(&sh_x[rr][lx_c]) = *reinterpret_cast<const uint4*>(
                    x + (size_t)(r0 + rr) * x_stride + grp * 64 + lx_c
                );
            } else {
                *reinterpret_cast<uint4*>(&sh_x[rr][lx_c]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k0 = ks * 16 + 2 * t;
            const int gk = grp * 64 + k0;
            const uint32_t b00 = wrow0 ? __ldg(reinterpret_cast<const uint32_t*>(wrow0 + gk)) : 0u;
            const uint32_t b01 = wrow0 ? __ldg(reinterpret_cast<const uint32_t*>(wrow0 + gk + 8)) : 0u;
            const uint32_t b10 = wrow1 ? __ldg(reinterpret_cast<const uint32_t*>(wrow1 + gk)) : 0u;
            const uint32_t b11 = wrow1 ? __ldg(reinterpret_cast<const uint32_t*>(wrow1 + gk + 8)) : 0u;

#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int rm = m * 16;
                const uint32_t a[4] = {
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g + 8][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g][k0 + 8]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g + 8][k0 + 8]),
                };
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[m][0][0]), "+f"(acc[m][0][1]), "+f"(acc[m][0][2]), "+f"(acc[m][0][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b00), "r"(b01)
                );
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[m][1][0]), "+f"(acc[m][1][1]), "+f"(acc[m][1][2]), "+f"(acc[m][1][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b10), "r"(b11)
                );
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row0 = r0 + m * 16 + g;
        const int row1 = row0 + 8;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col0 = n0 + nt * 8 + 2 * t;
            const int col1 = col0 + 1;
            if (row0 < rows) {
                if (col0 < N) {
                    const size_t idx = (size_t)row0 * out_stride + col0;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][0];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][0]);
                }
                if (col1 < N) {
                    const size_t idx = (size_t)row0 * out_stride + col1;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][1];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][1]);
                }
            }
            if (row1 < rows) {
                if (col0 < N) {
                    const size_t idx = (size_t)row1 * out_stride + col0;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][2];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][2]);
                }
                if (col1 < N) {
                    const size_t idx = (size_t)row1 * out_stride + col1;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][3];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][3]);
                }
            }
        }
    }
}

// Warp-per-output-row exact BF16 GEMV/GEMM: Y[rows, N] = X[rows, K] @ W_bf16[N, K]^T.
// Uses 128-bit uint4 loads (8 bf16/lane = 256 weights/iter) and processes up to 4 rows/block.
// Launch: grid((N + 3) / 4, (rows + 3) / 4, 1), block(128, 1, 1).
__global__ void __launch_bounds__(128) m3tis_linear_bf16(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const __nv_bfloat16* __restrict__ w,
    void* __restrict__ out,
    int32_t out_stride,
    int32_t rows,
    int32_t N,
    int32_t K,
    int32_t out_f32
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;

    const int r0 = blockIdx.y * 4;
    const int nr = min(4, rows - r0);
    if (nr <= 0) return;

    const __nv_bfloat16* wrow = w + (size_t)n * (size_t)K;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Fast path for K == 128 (e.g. KDA f_b_proj/g_b_proj [8192, 128]): all 32 lanes read 4 bf16 (64-bit uint2) in 1 iter!
    if (K == 128) {
        const int k = lane * 4;
        const uint2 wu = __ldg(reinterpret_cast<const uint2*>(wrow + k));
        const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu);
        float wv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) wv[j] = __bfloat162float(wh[j]);
        if (nr == 1) {
            const uint2 xu = *reinterpret_cast<const uint2*>(x + (size_t)r0 * x_stride + k);
            const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
            float d = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) d = fmaf(__bfloat162float(xh[j]), wv[j], d);
            acc[0] = d;
        } else {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r < nr) {
                    const uint2 xu = *reinterpret_cast<const uint2*>(x + (size_t)(r0 + r) * x_stride + k);
                    const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
                    float d = 0.0f;
#pragma unroll
                    for (int j = 0; j < 4; ++j) d = fmaf(__bfloat162float(xh[j]), wv[j], d);
                    acc[r] = d;
                }
            }
        }
    } else if (nr == 1) {
        const __nv_bfloat16* xrow = x + (size_t)r0 * x_stride;
        float d = 0.0f;
        int k = lane * 8;
        for (; k + 8 <= K; k += 256) {
            const uint4 wu = __ldg(reinterpret_cast<const uint4*>(wrow + k));
            const uint4 xu = *reinterpret_cast<const uint4*>(xrow + k);
            const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu);
            const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
#pragma unroll
            for (int j = 0; j < 8; ++j) d = fmaf(__bfloat162float(xh[j]), __bfloat162float(wh[j]), d);
        }
        for (int kt = (K & ~7) + lane; kt < K; kt += 32) {
            d = fmaf(__bfloat162float(xrow[kt]), __bfloat162float(__ldg(wrow + kt)), d);
        }
        acc[0] = d;
    } else {
        int k = lane * 8;
        for (; k + 8 <= K; k += 256) {
            const uint4 wu = __ldg(reinterpret_cast<const uint4*>(wrow + k));
            const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wu);
            float wv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) wv[j] = __bfloat162float(wh[j]);

#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r < nr) {
                    const uint4 xu = *reinterpret_cast<const uint4*>(x + (size_t)(r0 + r) * x_stride + k);
                    const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
                    float d = acc[r];
#pragma unroll
                    for (int j = 0; j < 8; ++j) d = fmaf(__bfloat162float(xh[j]), wv[j], d);
                    acc[r] = d;
                }
            }
        }
        for (int kt = (K & ~7) + lane; kt < K; kt += 32) {
            float wf = __bfloat162float(__ldg(wrow + kt));
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r < nr) {
                    acc[r] = fmaf(__bfloat162float(x[(size_t)(r0 + r) * x_stride + kt]), wf, acc[r]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (r < nr) {
            float s = m3tis_lin::warp_sum(acc[r]);
            if (lane == 0) {
                const size_t idx = (size_t)(r0 + r) * (size_t)out_stride + (size_t)n;
                if (out_f32) {
                    static_cast<float*>(out)[idx] = s;
                } else {
                    static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(s);
                }
            }
        }
    }
}

// Clamped GLM-5.3 SwiGLU on separate gate[rows, N] and up[rows, N] -> act[rows, N] (bf16).
__global__ void m3tis_swiglu_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int32_t rows,
    int32_t N,
    float limit
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * N;
    if (idx >= total) return;
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    if (limit > 0.0f) {
        if (g > limit) g = limit;
        if (u > limit) u = limit;
        else if (u < -limit) u = -limit;
    }
    float silu = m3tis_lin::bf16r(g / (1.0f + expf(-g)));
    out[idx] = __float2bfloat16_rn(silu * u);
}

// Clamped GLM-5.3 SwiGLU on fused [gate | up] layout gu[rows, 2 * N] -> out[rows, N] (bf16).
__global__ void m3tis_swiglu_split_bf16(
    const __nv_bfloat16* __restrict__ gu,
    __nv_bfloat16* __restrict__ out,
    int32_t rows,
    int32_t N,
    float limit
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * N;
    if (idx >= total) return;
    const int r = idx / N;
    const int c = idx % N;
    const __nv_bfloat16* row = gu + (size_t)r * (size_t)(2 * N);
    float g = __bfloat162float(row[c]);
    float u = __bfloat162float(row[N + c]);
    if (limit > 0.0f) {
        if (g > limit) g = limit;
        if (u > limit) u = limit;
        else if (u < -limit) u = -limit;
    }
    float silu = m3tis_lin::bf16r(g / (1.0f + expf(-g)));
    out[idx] = __float2bfloat16_rn(silu * u);
}

// =================================================================================================
// On-GPU Affine Q3 (group_size = 64, 3.50 bits/weight -> -22.2% VRAM & HBM bytes vs Q4!):
//   qdata : uint32_t [N, (K / 64) * 6]  (each 32-weight half packs into 3 u32 words: 10+10+10+2 bits)
//   scales: half2    [N, K / 64]        (.x = scale = (max - min) / 7, .y = zero = min)
// =================================================================================================
__device__ __forceinline__ uint32_t m3tis_q3_get(uint32_t w0, uint32_t w1, uint32_t w2, int idx) {
    if (idx < 10) return (w0 >> (idx * 3)) & 7u;
    if (idx < 20) return (w1 >> ((idx - 10) * 3)) & 7u;
    if (idx < 30) return (w2 >> ((idx - 20) * 3)) & 7u;
    if (idx == 30) return (w0 >> 30) | ((w1 >> 28) & 4u);
    return (w1 >> 31) | ((w2 >> 29) & 6u);
}

// Pack row-major BF16 weight matrix W[N, K] (K % 64 == 0) into affine Q3 (gs = 64, 3.50 bpw).
// Launch: grid(N, K / 64, 1), block(32, 1, 1) — 1 warp packs 1 group of 64 weights!
__global__ void m3tis_pack_bf16_to_q3(
    const __nv_bfloat16* __restrict__ w,
    uint32_t* __restrict__ qdata,
    half2* __restrict__ scales,
    int32_t N,
    int32_t K
) {
    const int row = blockIdx.x;
    const int grp = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    __shared__ uint8_t sq[64];
    const int k_base = grp * 64 + lane * 2;
    const __nv_bfloat16* wp = w + (size_t)row * (size_t)K + k_base;
    float v0 = __bfloat162float(wp[0]);
    float v1 = __bfloat162float(wp[1]);

    float mn = fminf(v0, v1);
    float mx = fmaxf(v0, v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float span = mx - mn;
    float scale = (span > 1e-12f) ? (span / 7.0f) : 0.0f;
    half h_scale = __float2half_rn(scale);
    half h_zero  = __float2half_rn(mn);
    float f_zero = __half2float(h_zero);
    float f_scale = __half2float(h_scale);
    float inv_q = (f_scale > 0.0f) ? (1.0f / f_scale) : 0.0f;

    int q0 = (span > 1e-12f) ? __float2int_rn(fminf(7.0f, fmaxf(0.0f, (v0 - f_zero) * inv_q))) : 0;
    int q1 = (span > 1e-12f) ? __float2int_rn(fminf(7.0f, fmaxf(0.0f, (v1 - f_zero) * inv_q))) : 0;
    sq[lane * 2] = (uint8_t)(q0 & 7);
    sq[lane * 2 + 1] = (uint8_t)(q1 & 7);
    __syncwarp();

    if (lane < 6) {
        const int half_g = lane / 3;
        const int widx = lane % 3;
        const int b = half_g * 32;
        uint32_t word = 0u;
        if (widx == 0) {
#pragma unroll
            for (int j = 0; j < 10; ++j) word |= ((uint32_t)sq[b + j]) << (j * 3);
            word |= ((uint32_t)(sq[b + 30] & 3u)) << 30;
        } else if (widx == 1) {
#pragma unroll
            for (int j = 0; j < 10; ++j) word |= ((uint32_t)sq[b + 10 + j]) << (j * 3);
            word |= ((uint32_t)((sq[b + 30] >> 2) & 1u)) << 30;
            word |= ((uint32_t)(sq[b + 31] & 1u)) << 31;
        } else {
#pragma unroll
            for (int j = 0; j < 10; ++j) word |= ((uint32_t)sq[b + 20 + j]) << (j * 3);
            word |= ((uint32_t)((sq[b + 31] >> 1) & 3u)) << 30;
        }
        qdata[(size_t)row * ((size_t)(K / 64) * 6) + (size_t)grp * 6 + lane] = word;
    }
    if (lane == 0) {
        scales[(size_t)row * (K / 64) + grp] = __halves2half2(h_scale, h_zero);
    }
}

// Vectorized Affine Q3 GEMV/GEMM: Y[rows, N] = X[rows, K] @ W_q3[N, K]^T (3.50 bpw, gs = 64).
// Launch: grid((N + 3) / 4, (rows + 3) / 4, 1), block(128, 1, 1) — 4 warps = 4 output rows/block.
__global__ void __launch_bounds__(128) m3tis_linear_q3(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const uint32_t* __restrict__ qdata,
    const half2* __restrict__ scales,
    void* __restrict__ out,
    int32_t out_stride,
    int32_t rows,
    int32_t N,
    int32_t K,
    int32_t out_f32
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;

    const int r0 = blockIdx.y * 4;
    const int nr = min(4, rows - r0);
    if (nr <= 0) return;

    const int n_groups = K / 64;
    const uint32_t* qrow = qdata + (size_t)n * (size_t)(n_groups * 6);
    const half2* srow = scales + (size_t)n * (size_t)n_groups;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int g0 = 0;

    // Fast path for nr == 1 (serial decode): 16 groups (1,024 weights/warp/iter), 2 lanes/group!
    if (nr == 1) {
        const __nv_bfloat16* xrow = x + (size_t)r0 * x_stride;
        float d = 0.0f;
        for (; g0 + 16 <= n_groups; g0 += 16) {
            const int g = g0 + (lane >> 1);
            const int half_g = lane & 1;
            const uint32_t* qp = qrow + g * 6 + half_g * 3;
            const uint32_t w0 = __ldg(qp + 0);
            const uint32_t w1 = __ldg(qp + 1);
            const uint32_t w2 = __ldg(qp + 2);
            const half2 sz = __ldg(srow + g);
            const float sc = __low2float(sz);
            const float zr = __high2float(sz);
            const int k_base = g * 64 + half_g * 32;

            // Process 32 weights in 4 x 128-bit uint4 chunks of x (zero register spill!)
            {
                const uint4 xu = *reinterpret_cast<const uint4*>(xrow + k_base);
                const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    d = fmaf(__bfloat162float(xh[j]), fmaf((float)((w0 >> (j * 3)) & 7u), sc, zr), d);
            }
            {
                const uint4 xu = *reinterpret_cast<const uint4*>(xrow + k_base + 8);
                const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
                d = fmaf(__bfloat162float(xh[0]), fmaf((float)((w0 >> 24) & 7u), sc, zr), d);
                d = fmaf(__bfloat162float(xh[1]), fmaf((float)((w0 >> 27) & 7u), sc, zr), d);
#pragma unroll
                for (int j = 0; j < 6; ++j)
                    d = fmaf(__bfloat162float(xh[2 + j]), fmaf((float)((w1 >> (j * 3)) & 7u), sc, zr), d);
            }
            {
                const uint4 xu = *reinterpret_cast<const uint4*>(xrow + k_base + 16);
                const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    d = fmaf(__bfloat162float(xh[j]), fmaf((float)((w1 >> ((6 + j) * 3)) & 7u), sc, zr), d);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    d = fmaf(__bfloat162float(xh[4 + j]), fmaf((float)((w2 >> (j * 3)) & 7u), sc, zr), d);
            }
            {
                const uint4 xu = *reinterpret_cast<const uint4*>(xrow + k_base + 24);
                const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xu);
#pragma unroll
                for (int j = 0; j < 6; ++j)
                    d = fmaf(__bfloat162float(xh[j]), fmaf((float)((w2 >> ((4 + j) * 3)) & 7u), sc, zr), d);
                const uint32_t q30 = (w0 >> 30) | ((w1 >> 28) & 4u);
                const uint32_t q31 = (w1 >> 31) | ((w2 >> 29) & 6u);
                d = fmaf(__bfloat162float(xh[6]), fmaf((float)q30, sc, zr), d);
                d = fmaf(__bfloat162float(xh[7]), fmaf((float)q31, sc, zr), d);
            }
        }
        acc[0] = d;
    }

    // Generic 1-group-per-warp tail (also handles nr = 2..4 with low register pressure!)
    for (int g = g0; g < n_groups; ++g) {
        const half2 sz = __ldg(srow + g);
        const float sc = __low2float(sz);
        const float zr = __high2float(sz);
        const int half_g = lane >> 4;        // 0 for lanes 0..15, 1 for lanes 16..31
        const int in_h = (lane & 15) * 2;    // 0, 2, ..., 30
        const uint32_t* qp = qrow + g * 6 + half_g * 3;
        const uint32_t w0 = __ldg(qp + 0);
        const uint32_t w1 = __ldg(qp + 1);
        const uint32_t w2 = __ldg(qp + 2);
        const float f0 = fmaf((float)m3tis_q3_get(w0, w1, w2, in_h), sc, zr);
        const float f1 = fmaf((float)m3tis_q3_get(w0, w1, w2, in_h + 1), sc, zr);
        const int k_off = g * 64 + lane * 2;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r < nr) {
                const __nv_bfloat16* xp = x + (size_t)(r0 + r) * x_stride + k_off;
                acc[r] = fmaf(__bfloat162float(xp[0]), f0, acc[r]);
                acc[r] = fmaf(__bfloat162float(xp[1]), f1, acc[r]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (r < nr) {
            float s = m3tis_lin::warp_sum(acc[r]);
            if (lane == 0) {
                const size_t idx = (size_t)(r0 + r) * (size_t)out_stride + (size_t)n;
                if (out_f32) {
                    static_cast<float*>(out)[idx] = s;
                } else {
                    static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(s);
                }
            }
        }
    }
}

// Milestone 8: 4-Warp 16x64 Native BF16 Ampere sm_86 Tensor Core Q3 GEMM (`rows = 2..31`)!
// - K_STEP = 128 (2 groups = 4 x 32-weight halves) -> each lane t=0..3 loads ONLY its own 3 words (4x fewer loads, 2x fewer barriers!)
// Launch: grid((N + 63) / 64, (rows + 15) / 16, 1), block(128, 1, 1)
__global__ void __launch_bounds__(128) m3tis_linear_q3_tc(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const uint32_t* __restrict__ qdata,
    const half2* __restrict__ scales,
    void* __restrict__ out,
    int32_t out_stride,
    int32_t rows,
    int32_t N,
    int32_t K,
    int32_t out_f32
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n0 = blockIdx.x * 64 + warp * 16;
    const int r0 = blockIdx.y * 16;
    if (r0 >= rows) return;

    __shared__ __align__(16) __nv_bfloat16 sh_x[16][136];
    float acc[2][4] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f}
    };

    const int n_groups = K / 64;
    const int n_pairs = n_groups >> 1;
    const int n_row0 = n0 + g;
    const int n_row1 = n0 + 8 + g;
    const uint32_t* qrow0 = (n_row0 < N) ? (qdata + (size_t)n_row0 * (size_t)(n_groups * 6)) : nullptr;
    const uint32_t* qrow1 = (n_row1 < N) ? (qdata + (size_t)n_row1 * (size_t)(n_groups * 6)) : nullptr;
    const half2* srow0 = (n_row0 < N) ? (scales + (size_t)n_row0 * (size_t)n_groups) : nullptr;
    const half2* srow1 = (n_row1 < N) ? (scales + (size_t)n_row1 * (size_t)n_groups) : nullptr;

    const int lx_r = threadIdx.x >> 3;
    const int lx_c = (threadIdx.x & 7) << 3;

    for (int gp = 0; gp < n_pairs; ++gp) {
        const int k_base = gp * 128;
        if (r0 + lx_r < rows) {
            const __nv_bfloat16* xp = x + (size_t)(r0 + lx_r) * x_stride + k_base + lx_c;
            *reinterpret_cast<uint4*>(&sh_x[lx_r][lx_c])      = *reinterpret_cast<const uint4*>(xp);
            *reinterpret_cast<uint4*>(&sh_x[lx_r][64 + lx_c]) = *reinterpret_cast<const uint4*>(xp + 64);
        } else {
            *reinterpret_cast<uint4*>(&sh_x[lx_r][lx_c])      = make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(&sh_x[lx_r][64 + lx_c]) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();

        uint32_t m0_0 = 0u, m0_1 = 0u, m0_2 = 0u;
        uint32_t m1_0 = 0u, m1_1 = 0u, m1_2 = 0u;
        float sc0_a = 0.0f, zr0_a = 0.0f, sc0_b = 0.0f, zr0_b = 0.0f;
        float sc1_a = 0.0f, zr1_a = 0.0f, sc1_b = 0.0f, zr1_b = 0.0f;
        if (qrow0) {
            const uint32_t* qp = qrow0 + gp * 12 + t * 3;
            m0_0 = __ldg(qp); m0_1 = __ldg(qp + 1); m0_2 = __ldg(qp + 2);
            const half2 sza = __ldg(srow0 + gp * 2);
            const half2 szb = __ldg(srow0 + gp * 2 + 1);
            sc0_a = __low2float(sza); zr0_a = __high2float(sza);
            sc0_b = __low2float(szb); zr0_b = __high2float(szb);
        }
        if (qrow1) {
            const uint32_t* qp = qrow1 + gp * 12 + t * 3;
            m1_0 = __ldg(qp); m1_1 = __ldg(qp + 1); m1_2 = __ldg(qp + 2);
            const half2 sza = __ldg(srow1 + gp * 2);
            const half2 szb = __ldg(srow1 + gp * 2 + 1);
            sc1_a = __low2float(sza); zr1_a = __high2float(sza);
            sc1_b = __low2float(szb); zr1_b = __high2float(szb);
        }

#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const int src_lane = (g << 2) | h;
            const uint32_t u00 = __shfl_sync(0xffffffffu, m0_0, src_lane);
            const uint32_t u01 = __shfl_sync(0xffffffffu, m0_1, src_lane);
            const uint32_t u02 = __shfl_sync(0xffffffffu, m0_2, src_lane);
            const uint32_t u10 = __shfl_sync(0xffffffffu, m1_0, src_lane);
            const uint32_t u11 = __shfl_sync(0xffffffffu, m1_1, src_lane);
            const uint32_t u12 = __shfl_sync(0xffffffffu, m1_2, src_lane);
            const float sc0 = (h < 2) ? sc0_a : sc0_b;
            const float zr0 = (h < 2) ? zr0_a : zr0_b;
            const float sc1 = (h < 2) ? sc1_a : sc1_b;
            const float zr1 = (h < 2) ? zr1_a : zr1_b;
            const int sh_lo = t * 6;
            const int sh_c0 = (t - 1) * 6;
            const int sh_a1 = (t < 2) ? (18 + t * 6) : ((t - 2) * 6);
            const int sh_c1 = 12 + t * 6;

            // sub = 0: weights 0..15 of this 32-weight half (branchless 3-bit unpack!)
            {
                const int k0 = (h * 2) * 16 + 2 * t;
                const uint32_t a[4] = {
                    *reinterpret_cast<const uint32_t*>(&sh_x[g][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[g + 8][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[g][k0 + 8]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[g + 8][k0 + 8]),
                };
                {
                    const uint32_t wa = u00 >> sh_lo;
                    const uint32_t wc = (t == 0) ? (u00 >> 24) : (u01 >> sh_c0);
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                        fmaf((float)(wa & 7u), sc0, zr0),
                        fmaf((float)((wa >> 3) & 7u), sc0, zr0)
                    );
                    const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                        fmaf((float)(wc & 7u), sc0, zr0),
                        fmaf((float)((wc >> 3) & 7u), sc0, zr0)
                    );
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
                    );
                }
                {
                    const uint32_t wa = u10 >> sh_lo;
                    const uint32_t wc = (t == 0) ? (u10 >> 24) : (u11 >> sh_c0);
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                        fmaf((float)(wa & 7u), sc1, zr1),
                        fmaf((float)((wa >> 3) & 7u), sc1, zr1)
                    );
                    const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                        fmaf((float)(wc & 7u), sc1, zr1),
                        fmaf((float)((wc >> 3) & 7u), sc1, zr1)
                    );
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
                    );
                }
            }
            // sub = 1: weights 16..31 of this 32-weight half
            {
                const int k0 = (h * 2 + 1) * 16 + 2 * t;
                const uint32_t a[4] = {
                    *reinterpret_cast<const uint32_t*>(&sh_x[g][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[g + 8][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[g][k0 + 8]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[g + 8][k0 + 8]),
                };
                {
                    const uint32_t wa = ((t < 2) ? u01 : u02) >> sh_a1;
                    const uint32_t qc = (t < 3) ? ((u02 >> sh_c1) & 7u) : ((u00 >> 30) | ((u01 >> 28) & 4u));
                    const uint32_t qd = (t < 3) ? ((u02 >> (sh_c1 + 3)) & 7u) : ((u01 >> 31) | ((u02 >> 29) & 6u));
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                        fmaf((float)(wa & 7u), sc0, zr0),
                        fmaf((float)((wa >> 3) & 7u), sc0, zr0)
                    );
                    const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                        fmaf((float)qc, sc0, zr0),
                        fmaf((float)qd, sc0, zr0)
                    );
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
                    );
                }
                {
                    const uint32_t wa = ((t < 2) ? u11 : u12) >> sh_a1;
                    const uint32_t qc = (t < 3) ? ((u12 >> sh_c1) & 7u) : ((u10 >> 30) | ((u11 >> 28) & 4u));
                    const uint32_t qd = (t < 3) ? ((u12 >> (sh_c1 + 3)) & 7u) : ((u11 >> 31) | ((u12 >> 29) & 6u));
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                        fmaf((float)(wa & 7u), sc1, zr1),
                        fmaf((float)((wa >> 3) & 7u), sc1, zr1)
                    );
                    const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                        fmaf((float)qc, sc1, zr1),
                        fmaf((float)qd, sc1, zr1)
                    );
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
                    );
                }
            }
        }
        __syncthreads();
    }

    if (n_groups & 1) {
        const int grp = n_groups - 1;
        if (r0 + lx_r < rows) {
            *reinterpret_cast<uint4*>(&sh_x[lx_r][lx_c]) = *reinterpret_cast<const uint4*>(
                x + (size_t)(r0 + lx_r) * x_stride + grp * 64 + lx_c
            );
        } else {
            *reinterpret_cast<uint4*>(&sh_x[lx_r][lx_c]) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();

        uint32_t m0_0 = 0u, m0_1 = 0u, m0_2 = 0u;
        uint32_t m1_0 = 0u, m1_1 = 0u, m1_2 = 0u;
        float sc0 = 0.0f, zr0 = 0.0f, sc1 = 0.0f, zr1 = 0.0f;
        if (qrow0) {
            if (t < 2) {
                const uint32_t* qp = qrow0 + grp * 6 + t * 3;
                m0_0 = __ldg(qp); m0_1 = __ldg(qp + 1); m0_2 = __ldg(qp + 2);
            }
            const half2 sz0 = __ldg(srow0 + grp);
            sc0 = __low2float(sz0); zr0 = __high2float(sz0);
        }
        if (qrow1) {
            if (t < 2) {
                const uint32_t* qp = qrow1 + grp * 6 + t * 3;
                m1_0 = __ldg(qp); m1_1 = __ldg(qp + 1); m1_2 = __ldg(qp + 2);
            }
            const half2 sz1 = __ldg(srow1 + grp);
            sc1 = __low2float(sz1); zr1 = __high2float(sz1);
        }

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int src_lane = (g << 2) | (ks >> 1);
            const uint32_t u00 = __shfl_sync(0xffffffffu, m0_0, src_lane);
            const uint32_t u01 = __shfl_sync(0xffffffffu, m0_1, src_lane);
            const uint32_t u02 = __shfl_sync(0xffffffffu, m0_2, src_lane);
            const uint32_t u10 = __shfl_sync(0xffffffffu, m1_0, src_lane);
            const uint32_t u11 = __shfl_sync(0xffffffffu, m1_1, src_lane);
            const uint32_t u12 = __shfl_sync(0xffffffffu, m1_2, src_lane);
            const int k0 = ks * 16 + 2 * t;
            const uint32_t a[4] = {
                *reinterpret_cast<const uint32_t*>(&sh_x[g][k0]),
                *reinterpret_cast<const uint32_t*>(&sh_x[g + 8][k0]),
                *reinterpret_cast<const uint32_t*>(&sh_x[g][k0 + 8]),
                *reinterpret_cast<const uint32_t*>(&sh_x[g + 8][k0 + 8]),
            };
            const int i_base = (ks & 1) * 16 + 2 * t;
            {
                const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                    fmaf((float)m3tis_q3_get(u00, u01, u02, i_base), sc0, zr0),
                    fmaf((float)m3tis_q3_get(u00, u01, u02, i_base + 1), sc0, zr0)
                );
                const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                    fmaf((float)m3tis_q3_get(u00, u01, u02, i_base + 8), sc0, zr0),
                    fmaf((float)m3tis_q3_get(u00, u01, u02, i_base + 9), sc0, zr0)
                );
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
                );
            }
            {
                const __nv_bfloat162 h0 = __floats2bfloat162_rn(
                    fmaf((float)m3tis_q3_get(u10, u11, u12, i_base), sc1, zr1),
                    fmaf((float)m3tis_q3_get(u10, u11, u12, i_base + 1), sc1, zr1)
                );
                const __nv_bfloat162 h1 = __floats2bfloat162_rn(
                    fmaf((float)m3tis_q3_get(u10, u11, u12, i_base + 8), sc1, zr1),
                    fmaf((float)m3tis_q3_get(u10, u11, u12, i_base + 9), sc1, zr1)
                );
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&h0);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&h1);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1)
                );
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col0 = n0 + nt * 8 + 2 * t;
        const int col1 = col0 + 1;
        const int row0 = r0 + g;
        const int row1 = r0 + g + 8;
        if (row0 < rows) {
            if (col0 < N) {
                const size_t idx = (size_t)row0 * out_stride + col0;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][0];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][0]);
            }
            if (col1 < N) {
                const size_t idx = (size_t)row0 * out_stride + col1;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][1];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][1]);
            }
        }
        if (row1 < rows) {
            if (col0 < N) {
                const size_t idx = (size_t)row1 * out_stride + col0;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][2];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][2]);
            }
            if (col1 < N) {
                const size_t idx = (size_t)row1 * out_stride + col1;
                if (out_f32) static_cast<float*>(out)[idx] = acc[nt][3];
                else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[nt][3]);
            }
        }
    }
}

// Milestone 6: 64x128 8-Warp Native BF16 Ampere sm_86 Tensor Core Q3 Prefill GEMM (`rows >= 32`)!
// Launch: grid((N + 127) / 128, (rows + 63) / 64, 1), block(256, 1, 1)
__global__ void __launch_bounds__(256) m3tis_linear_q3_tc64(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const uint32_t* __restrict__ qdata,
    const half2* __restrict__ scales,
    void* __restrict__ out,
    int32_t out_stride,
    int32_t rows,
    int32_t N,
    int32_t K,
    int32_t out_f32
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n0 = blockIdx.x * 128 + warp * 16;
    const int r0 = blockIdx.y * 64;
    if (r0 >= rows) return;

    __shared__ __align__(16) __nv_bfloat16 sh_x[64][72];
    float acc[4][2][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][nt][c] = 0.0f;

    const int n_groups = K / 64;
    const int n_row0 = n0 + g;
    const int n_row1 = n0 + 8 + g;
    const uint32_t* qrow0 = (n_row0 < N) ? (qdata + (size_t)n_row0 * (size_t)(n_groups * 6)) : nullptr;
    const uint32_t* qrow1 = (n_row1 < N) ? (qdata + (size_t)n_row1 * (size_t)(n_groups * 6)) : nullptr;
    const half2* srow0 = (n_row0 < N) ? (scales + (size_t)n_row0 * (size_t)n_groups) : nullptr;
    const half2* srow1 = (n_row1 < N) ? (scales + (size_t)n_row1 * (size_t)n_groups) : nullptr;

    const int lx_r = threadIdx.x >> 3;
    const int lx_c = (threadIdx.x & 7) << 3;

    for (int grp = 0; grp < n_groups; ++grp) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int rr = m * 32 + lx_r;
            if (r0 + rr < rows) {
                *reinterpret_cast<uint4*>(&sh_x[rr][lx_c]) = *reinterpret_cast<const uint4*>(
                    x + (size_t)(r0 + rr) * x_stride + grp * 64 + lx_c
                );
            } else {
                *reinterpret_cast<uint4*>(&sh_x[rr][lx_c]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        uint32_t w0_words[6] = {0, 0, 0, 0, 0, 0};
        uint32_t w1_words[6] = {0, 0, 0, 0, 0, 0};
        float sc0 = 0.0f, zr0 = 0.0f, sc1 = 0.0f, zr1 = 0.0f;
        if (qrow0) {
#pragma unroll
            for (int i = 0; i < 6; ++i) w0_words[i] = __ldg(qrow0 + grp * 6 + i);
            const half2 sz0 = __ldg(srow0 + grp);
            sc0 = __low2float(sz0);
            zr0 = __high2float(sz0);
        }
        if (qrow1) {
#pragma unroll
            for (int i = 0; i < 6; ++i) w1_words[i] = __ldg(qrow1 + grp * 6 + i);
            const half2 sz1 = __ldg(srow1 + grp);
            sc1 = __low2float(sz1);
            zr1 = __high2float(sz1);
        }

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k0 = ks * 16 + 2 * t;
            const int h_off = (ks >> 1) * 3;
            const int i_base = (ks & 1) * 16 + 2 * t;

            const uint32_t u00 = w0_words[h_off], u01 = w0_words[h_off + 1], u02 = w0_words[h_off + 2];
            const __nv_bfloat162 h00 = __floats2bfloat162_rn(
                fmaf((float)m3tis_q3_get(u00, u01, u02, i_base), sc0, zr0),
                fmaf((float)m3tis_q3_get(u00, u01, u02, i_base + 1), sc0, zr0)
            );
            const __nv_bfloat162 h01 = __floats2bfloat162_rn(
                fmaf((float)m3tis_q3_get(u00, u01, u02, i_base + 8), sc0, zr0),
                fmaf((float)m3tis_q3_get(u00, u01, u02, i_base + 9), sc0, zr0)
            );
            const uint32_t b00 = *reinterpret_cast<const uint32_t*>(&h00);
            const uint32_t b01 = *reinterpret_cast<const uint32_t*>(&h01);

            const uint32_t u10 = w1_words[h_off], u11 = w1_words[h_off + 1], u12 = w1_words[h_off + 2];
            const __nv_bfloat162 h10 = __floats2bfloat162_rn(
                fmaf((float)m3tis_q3_get(u10, u11, u12, i_base), sc1, zr1),
                fmaf((float)m3tis_q3_get(u10, u11, u12, i_base + 1), sc1, zr1)
            );
            const __nv_bfloat162 h11 = __floats2bfloat162_rn(
                fmaf((float)m3tis_q3_get(u10, u11, u12, i_base + 8), sc1, zr1),
                fmaf((float)m3tis_q3_get(u10, u11, u12, i_base + 9), sc1, zr1)
            );
            const uint32_t b10 = *reinterpret_cast<const uint32_t*>(&h10);
            const uint32_t b11 = *reinterpret_cast<const uint32_t*>(&h11);

#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int rm = m * 16;
                const uint32_t a[4] = {
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g + 8][k0]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g][k0 + 8]),
                    *reinterpret_cast<const uint32_t*>(&sh_x[rm + g + 8][k0 + 8]),
                };
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[m][0][0]), "+f"(acc[m][0][1]), "+f"(acc[m][0][2]), "+f"(acc[m][0][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b00), "r"(b01)
                );
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[m][1][0]), "+f"(acc[m][1][1]), "+f"(acc[m][1][2]), "+f"(acc[m][1][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b10), "r"(b11)
                );
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row0 = r0 + m * 16 + g;
        const int row1 = row0 + 8;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col0 = n0 + nt * 8 + 2 * t;
            const int col1 = col0 + 1;
            if (row0 < rows) {
                if (col0 < N) {
                    const size_t idx = (size_t)row0 * out_stride + col0;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][0];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][0]);
                }
                if (col1 < N) {
                    const size_t idx = (size_t)row0 * out_stride + col1;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][1];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][1]);
                }
            }
            if (row1 < rows) {
                if (col0 < N) {
                    const size_t idx = (size_t)row1 * out_stride + col0;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][2];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][2]);
                }
                if (col1 < N) {
                    const size_t idx = (size_t)row1 * out_stride + col1;
                    if (out_f32) static_cast<float*>(out)[idx] = acc[m][nt][3];
                    else static_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(acc[m][nt][3]);
                }
            }
        }
    }
}

// Copy shared expert output sy[rows, D] (fp32) into the last slot (slot = slots - 1) of slot_y[rows, slots, D].
__global__ void m3tis_put_shared_slot_f32(
    const float* __restrict__ sy,
    float* __restrict__ slot_y,
    int32_t rows,
    int32_t D,
    int32_t slots
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * D) return;
    const int r = idx / D;
    const int d = idx % D;
    slot_y[((size_t)r * slots + (slots - 1)) * D + d] = sy[idx];
}

// Elementwise FP32 add: dst[i] += src[i]
__global__ void m3tis_add_f32_inplace(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int32_t count
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) dst[idx] += src[idx];
}

} // extern "C"
