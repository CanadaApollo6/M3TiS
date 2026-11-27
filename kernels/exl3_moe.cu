// Clean-room M3TiS sm_86 EXL3 (MCG codebook) Grouped MoE Kernels.
// EXL3 trellis bit-packing & MCG codebook math follows ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp).
// Native Ampere sm_86 Tensor Core (mma.sync.aligned.m16n8k16 f16->f32) with per-expert mixed bitwidth
// (2b/3b/4b..8b in a single launch), register-pipelined tile-ahead streaming loads, and fused FWHT128.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace m3tis_exl3 {

constexpr float HAD_SCALE = 0.08838834764831845f; // 1 / sqrt(128)

__device__ __forceinline__ uint32_t ld_nc32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void mma_m16n8k16_f16(
    float (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
    );
}

__device__ __forceinline__ float bf16r(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float clamp_upper(float x, float lim) {
    return x > lim ? lim : x;
}

__device__ __forceinline__ float clamp_sym(float x, float lim) {
    return x > lim ? lim : (x < -lim ? -lim : x);
}

// Unscaled 128-point Walsh-Hadamard transform (32 lanes x 4 floats/lane).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1];
    const float c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? (o - v[j]) : (v[j] + o);
        }
    }
}

template <int K2>
__host__ __device__ constexpr int tile_words() {
    return 4 * K2; // 16x16 = 256 vals * (K2/2) bits / 32 bits
}

template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? (((p + 1) * K2 - ((p + 1) & 1)) / 2) : ((p + 1) * (K2 / 2));
}

template <int K2>
__host__ __device__ constexpr int lane_words() {
    int most = 0;
    for (int l = 0; l < 32; ++l) {
        const int first = (stream_end<K2>(8 * l) - 16 + 128 * K2) % 32;
        const int need = first + stream_end<K2>(8 * l + 7) - stream_end<K2>(8 * l) + 16;
        if (need > most) most = need;
    }
    return (most + 31) / 32;
}

template <int K2>
__device__ __forceinline__ void lane_start(int lane, int& word, int& offset) {
    const int first = 4 * lane * K2 + K2 / 2 - 16 + 128 * K2;
    word = (first >> 5) % tile_words<K2>();
    offset = first & 31;
}

template <int K2>
__device__ __forceinline__ void ldg_lane_words(
    const uint32_t* tile,
    int lane,
    uint32_t (&w)[lane_words<K2>()]
) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) {
        w[i] = ld_nc32(tile + (word + i) % tile_words<K2>());
    }
}

__device__ __forceinline__ uint32_t window16(uint32_t hi, uint32_t mid, uint32_t lo, int d) {
    if (d <= 16) return (hi >> (16 - d)) & 0xffffu;
    if (d < 32) return __funnelshift_l(mid, hi, d) >> 16;
    if (d <= 48) return (mid >> (48 - d)) & 0xffffu;
    return __funnelshift_l(lo, mid, d - 32) >> 16;
}

// MCG codebook: 2 16-bit states -> packed half2 in 4 integer/fp16 instructions.
__device__ __forceinline__ uint32_t decode2_mcg(uint32_t s0, uint32_t s1) {
    uint32_t x0 = (s0 * 0xCBAC1FEDu & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t x1 = (s1 * 0xCBAC1FEDu & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const uint32_t lo = __byte_perm(x0, x1, 0x5410);
    const uint32_t hi = __byte_perm(x0, x1, 0x7632);
    const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
    return *reinterpret_cast<const uint32_t*>(&r);
}

template <int K2>
__device__ __forceinline__ void decode_lane_mcg(
    const uint32_t (&w)[lane_words<K2>()],
    int lane,
    uint32_t (&b0)[2],
    uint32_t (&b1)[2]
) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
    const uint32_t c = lane_words<K2>() > 2 ? w[lane_words<K2>() > 2 ? 2 : 0] : 0u;
    const uint32_t hi = __funnelshift_l(w[1], w[0], offset);
    const uint32_t mid = __funnelshift_l(c, w[1], offset);
    const uint32_t lo = c << offset;
    uint32_t s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s[j] = window16(hi, mid, lo, stream_end<K2>(j) - stream_end<K2>(0));
    }
    b0[0] = decode2_mcg(s[0], s[1]);
    b0[1] = decode2_mcg(s[2], s[3]);
    b1[0] = decode2_mcg(s[4], s[5]);
    b1[1] = decode2_mcg(s[6], s[7]);
}

template <int K2, int NT, bool AHEAD>
__device__ __forceinline__ void warp_tiles_mcg(
    const uint32_t* __restrict__ T,
    int ntiles,
    int kt0,
    int nkt,
    int nt0,
    const half* __restrict__ x0,
    const half* __restrict__ x1,
    bool ok0,
    bool ok1,
    int lane,
    float (&acc)[NT][2][4]
) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    const size_t kstride = (size_t)ntiles * (size_t)TW;
    const uint32_t* tb = T + (size_t)kt0 * kstride + (size_t)nt0 * TW;

    uint32_t nx[NT][LW];
    if constexpr (AHEAD) {
        if (nkt > 0) {
#pragma unroll
            for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(tb + j * TW, lane, nx[j]);
        }
    }

    for (int i = 0; i < nkt; ++i) {
        uint32_t w[NT][LW];
        if constexpr (AHEAD) {
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int l = 0; l < LW; ++l) w[j][l] = nx[j][l];
            if (i + 1 < nkt) {
                const uint32_t* tn = tb + (size_t)(i + 1) * kstride;
#pragma unroll
                for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(tn + j * TW, lane, nx[j]);
            }
        } else {
            const uint32_t* tc = tb + (size_t)i * kstride;
#pragma unroll
            for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(tc + j * TW, lane, w[j]);
        }

        const int k = (kt0 + i) * 16;
        const uint32_t a[4] = {
            ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u,
            ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u,
            ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u,
            ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u,
        };
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t b0[2], b1[2];
            decode_lane_mcg<K2>(w[j], lane, b0, b1);
            mma_m16n8k16_f16(acc[j][0], a, b0);
            mma_m16n8k16_f16(acc[j][1], a, b1);
        }
    }
}

template <int NT, int W, bool AHEAD, int MAX_M>
__device__ __forceinline__ void grouped_mcg_impl(
    const half* __restrict__ x0_all,
    const half* __restrict__ x1_all,
    const uint64_t* __restrict__ tp0,
    const uint64_t* __restrict__ tp1,
    const int32_t* __restrict__ k2_0,
    const int32_t* __restrict__ k2_1,
    const int32_t* __restrict__ uids,
    const int32_t* __restrict__ ucount,
    const int32_t* __restrict__ members,
    float* __restrict__ z,
    int32_t K,
    int32_t N,
    int32_t P,
    int32_t sk,
    int32_t maxm,
    int32_t slots
) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int nblock = blockIdx.y;
    const int zz = blockIdx.z;

    const int MT = (maxm + 15) / 16;
    const int mtile = zz % MT;
    const int split = (zz / MT) % sk;
    const int mat = zz / MT / sk;

    const half* X = mat ? x1_all : x0_all;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? tp1[e] : tp0[e]);
    const int k2 = mat ? k2_1[e] : k2_0[e];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int ntiles = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = (m < maxm) ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = (code >= 0) ? ((code >> 5) * slots + (code & 31)) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;

    const int r0 = (g < MAX_M) ? rows_sh[g] : -1;
    const int r1 = (MAX_M > 8) ? rows_sh[g + 8] : -1;
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * (size_t)K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * (size_t)K + 2 * t;

    const int per_warp = (K >> 4) / sk / W;
    const int kt0 = split * (per_warp * W) + warp * per_warp;
    const int nt0 = nblock * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.0f;

    switch (k2) {
#define M3TIS_EXL3_K2_CASE(K2_) \
    case K2_: warp_tiles_mcg<K2_, NT, AHEAD>(T, ntiles, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc); break;
        M3TIS_EXL3_K2_CASE(2)
        M3TIS_EXL3_K2_CASE(4)
        M3TIS_EXL3_K2_CASE(6)
        M3TIS_EXL3_K2_CASE(8)
        M3TIS_EXL3_K2_CASE(10)
        M3TIS_EXL3_K2_CASE(12)
        M3TIS_EXL3_K2_CASE(14)
        M3TIS_EXL3_K2_CASE(16)
#undef M3TIS_EXL3_K2_CASE
        default:
            __trap();
    }

    __shared__ __align__(16) float red[W][MAX_M][NT * 16];
    if (g < MAX_M) {
#pragma unroll
        for (int i = 0; i < NT; ++i) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col]     = acc[i][h][0];
                red[warp][g][col + 1] = acc[i][h][1];
                if constexpr (MAX_M > 8) {
                    red[warp][g + 8][col]     = acc[i][h][2];
                    red[warp][g + 8][col + 1] = acc[i][h][3];
                }
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < MAX_M * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16);
        const int col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        z[(((size_t)mat * (size_t)sk + (size_t)split) * (size_t)P + (size_t)r) * (size_t)N + (size_t)nt0 * 16 + col] = s;
    }
}

} // namespace m3tis_exl3

extern "C" {

// 1-block grouper for up to 4096 VRAM slots (1024 threads x 4 slots/thread).
__global__ void __launch_bounds__(1024) m3tis_exl3_group(
    const int32_t* __restrict__ pick,
    int32_t* __restrict__ uids,
    int32_t* __restrict__ ucount,
    int32_t* __restrict__ members,
    int32_t rows,
    int32_t slots,
    int32_t experts,
    int32_t maxm
) {
    extern __shared__ int32_t sh_pick[];
    __shared__ int32_t warp_tot[32];
    const int n = rows * slots;
    for (int i = threadIdx.x; i < n; i += 1024) sh_pick[i] = pick[i];
    __syncthreads();

    int cnt[4];
    int used = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int e = threadIdx.x * 4 + q;
        int c = 0;
        if (e < experts) {
            for (int i = 0; i < n; ++i) c += (sh_pick[i] == e);
        }
        cnt[q] = c;
        used += (c > 0);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * 4 + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i) {
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        }
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
}

// Hadamard input rotation: Xh = fp16(fwht128(x * suh) * HAD_SCALE) for gate (z=0) and up (z=1).
// x_dtype: 0 = fp16, 1 = bf16, 2 = fp32.
__global__ void m3tis_exl3_rot_in(
    const void* __restrict__ x,
    int32_t x_stride,
    int32_t x_dtype,
    const int32_t* __restrict__ pick,
    const half* __restrict__ suh0,
    const half* __restrict__ suh1,
    half* __restrict__ out0,
    half* __restrict__ out1,
    int32_t K,
    int32_t slots,
    int32_t experts
) {
    const int p = blockIdx.x;
    const int blk = blockIdx.y;
    const int mat = blockIdx.z;
    const int e = pick[p];
    if (e < 0 || e >= experts) return;
    const int lane = threadIdx.x;
    const int row = p / slots;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * (size_t)K + blk * 128 + 4 * lane;

    float v[4];
    if (x_dtype == 1) {
        const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(x) + (size_t)row * x_stride + blk * 128 + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xv[j]) * __half2float(suh[j]);
    } else if (x_dtype == 0) {
        const half* xv = reinterpret_cast<const half*>(x) + (size_t)row * x_stride + blk * 128 + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __half2float(xv[j]) * __half2float(suh[j]);
    } else {
        const float* xv = reinterpret_cast<const float*>(x) + (size_t)row * x_stride + blk * 128 + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = xv[j] * __half2float(suh[j]);
    }
    m3tis_exl3::fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * (size_t)K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * m3tis_exl3::HAD_SCALE);
}

// Ampere sm_86 grouped EXL3 MCG GEMM with register-pipelined tile-ahead loads:
// - m1: NT=4 (64 cols/block), MAX_M=1 -> 1 KiB shmem, half regs, 2x grid blocks for rows == 1 serial decode!
// - m8: NT=4 (64 cols/block), MAX_M=8 -> 8 KiB shmem, half regs, 2x grid blocks for DFlash2/SelfSpec (rows <= 8)!
// - default: NT=8 (128 cols/block), MAX_M=16 -> prefill (rows > 8).
__global__ void __launch_bounds__(128) m3tis_exl3_grouped_mcg_ahead_m1(
    const half* __restrict__ x0,
    const half* __restrict__ x1,
    const uint64_t* __restrict__ tp0,
    const uint64_t* __restrict__ tp1,
    const int32_t* __restrict__ k2_0,
    const int32_t* __restrict__ k2_1,
    const int32_t* __restrict__ uids,
    const int32_t* __restrict__ ucount,
    const int32_t* __restrict__ members,
    float* __restrict__ z,
    int32_t K,
    int32_t N,
    int32_t P,
    int32_t sk,
    int32_t maxm,
    int32_t slots
) {
    m3tis_exl3::grouped_mcg_impl<4, 4, true, 1>(
        x0, x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z, K, N, P, sk, maxm, slots
    );
}

__global__ void __launch_bounds__(128) m3tis_exl3_grouped_mcg_ahead_m8(
    const half* __restrict__ x0,
    const half* __restrict__ x1,
    const uint64_t* __restrict__ tp0,
    const uint64_t* __restrict__ tp1,
    const int32_t* __restrict__ k2_0,
    const int32_t* __restrict__ k2_1,
    const int32_t* __restrict__ uids,
    const int32_t* __restrict__ ucount,
    const int32_t* __restrict__ members,
    float* __restrict__ z,
    int32_t K,
    int32_t N,
    int32_t P,
    int32_t sk,
    int32_t maxm,
    int32_t slots
) {
    m3tis_exl3::grouped_mcg_impl<4, 4, true, 8>(
        x0, x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z, K, N, P, sk, maxm, slots
    );
}

__global__ void __launch_bounds__(128) m3tis_exl3_grouped_mcg_ahead(
    const half* __restrict__ x0,
    const half* __restrict__ x1,
    const uint64_t* __restrict__ tp0,
    const uint64_t* __restrict__ tp1,
    const int32_t* __restrict__ k2_0,
    const int32_t* __restrict__ k2_1,
    const int32_t* __restrict__ uids,
    const int32_t* __restrict__ ucount,
    const int32_t* __restrict__ members,
    float* __restrict__ z,
    int32_t K,
    int32_t N,
    int32_t P,
    int32_t sk,
    int32_t maxm,
    int32_t slots
) {
    m3tis_exl3::grouped_mcg_impl<8, 4, true, 16>(
        x0, x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z, K, N, P, sk, maxm, slots
    );
}

// Same kernel without tile-ahead pipelining (for bit-exact A/B benchmarking on sm_86).
__global__ void __launch_bounds__(128) m3tis_exl3_grouped_mcg_plain(
    const half* __restrict__ x0,
    const half* __restrict__ x1,
    const uint64_t* __restrict__ tp0,
    const uint64_t* __restrict__ tp1,
    const int32_t* __restrict__ k2_0,
    const int32_t* __restrict__ k2_1,
    const int32_t* __restrict__ uids,
    const int32_t* __restrict__ ucount,
    const int32_t* __restrict__ members,
    float* __restrict__ z,
    int32_t K,
    int32_t N,
    int32_t P,
    int32_t sk,
    int32_t maxm,
    int32_t slots
) {
    m3tis_exl3::grouped_mcg_impl<8, 4, false, 16>(
        x0, x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z, K, N, P, sk, maxm, slots
    );
}

// Fused gate+up split-K sum -> FWHT128 -> * svh -> clamped bf16 SwiGLU -> * suh_d -> FWHT128 -> Xd (fp16).
__global__ void m3tis_exl3_gateup_epilogue(
    const float* __restrict__ z,
    const int32_t* __restrict__ pick,
    const half* __restrict__ svh_g,
    const half* __restrict__ svh_u,
    const half* __restrict__ suh_d,
    half* __restrict__ xd,
    int32_t P,
    int32_t N,
    int32_t sk,
    int32_t experts,
    float limit,
    int32_t act_mode // 0 = bf16-rounded GLM SwiGLU, 1 = fp32 SwiGLU
) {
    const int p = blockIdx.x;
    const int blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= experts) return;
    const int lane = threadIdx.x;
    const int col = blk * 128 + 4 * lane;

    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.0f, su = 0.0f;
        for (int s = 0; s < sk; ++s) {
            sg += z[((size_t)s * P + p) * N + col + j];
            su += z[((size_t)(sk + s) * P + p) * N + col + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    m3tis_exl3::fwht128(gv, lane);
    m3tis_exl3::fwht128(uv, lane);

    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = m3tis_exl3::bf16r(gv[j] * m3tis_exl3::HAD_SCALE * __half2float(svh_g[(size_t)e * N + col + j]));
            float uu = m3tis_exl3::bf16r(uv[j] * m3tis_exl3::HAD_SCALE * __half2float(svh_u[(size_t)e * N + col + j]));
            if (limit > 0.0f) {
                gg = m3tis_exl3::clamp_upper(gg, limit);
                uu = m3tis_exl3::clamp_sym(uu, limit);
            }
            act = m3tis_exl3::bf16r(m3tis_exl3::bf16r(gg / (1.0f + expf(-gg))) * uu);
        } else {
            float gg = gv[j] * m3tis_exl3::HAD_SCALE * __half2float(svh_g[(size_t)e * N + col + j]);
            float uu = uv[j] * m3tis_exl3::HAD_SCALE * __half2float(svh_u[(size_t)e * N + col + j]);
            if (limit > 0.0f) {
                gg = m3tis_exl3::clamp_upper(gg, limit);
                uu = m3tis_exl3::clamp_sym(uu, limit);
            }
            act = (gg / (1.0f + expf(-gg))) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + col + j]);
    }
    m3tis_exl3::fwht128(v, lane);
    half* o = xd + (size_t)p * (size_t)N + col;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * m3tis_exl3::HAD_SCALE);
}

// Fused down split-K sum -> FWHT128 -> * svh_d -> optional weighted combine across `slots` into `out[rows, D]`.
__global__ void m3tis_exl3_down_combine(
    const float* __restrict__ z,
    const int32_t* __restrict__ pick,
    const half* __restrict__ svh_d,
    float* __restrict__ slot_y,       // fp32 [P, D] (pre-populated with shared expert at slot = slots-1 if used)
    const float* __restrict__ wts,    // nullable fp32 [rows, slots]
    float* __restrict__ out,          // fp32 [rows, D]
    int32_t P,
    int32_t D,
    int32_t sk,
    int32_t experts,
    int32_t slots,
    int32_t accum_out
) {
    __shared__ float4 part[32][32];
    const int r = blockIdx.x;
    const int blk = blockIdx.y;
    const int k = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int col = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];

    float o[4];
    if (e >= 0 && e < experts) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int q = 0; q < sk; ++q) s += z[((size_t)q * P + p) * D + col + j];
            v[j] = s;
        }
        m3tis_exl3::fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * m3tis_exl3::HAD_SCALE * __half2float(svh_d[(size_t)e * D + col + j]);
            if (!wts) slot_y[(size_t)p * D + col + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = slot_y[(size_t)p * D + col + j];
    }

    if (!wts) return;
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        if (w != 0.0f) {
            const float4 u = part[q][lane];
            acc[0] = fmaf(w, u.x, acc[0]);
            acc[1] = fmaf(w, u.y, acc[1]);
            acc[2] = fmaf(w, u.z, acc[2]);
            acc[3] = fmaf(w, u.w, acc[3]);
        }
    }
    if (accum_out) {
#pragma unroll
        for (int j = 0; j < 4; ++j) out[(size_t)r * D + col + j] += acc[j];
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) out[(size_t)r * D + col + j] = acc[j];
    }
}

// =================================================================================================
// Milestone 5B: Expert-Major Flat-Tile Streaming Prefill Kernels (zero maxm limit, zero wasted CTAs)
// =================================================================================================

// Hadamard input rotation for flat expert-major pairs `p in 0..P`:
// reads `x[pair_row[p]]` (bf16), rotates with `suh0/suh1[pair_slot[p]]`, writes `out0/out1[p]`.
// Launch: grid(P, K / 128, 2), block(32, 1, 1)
__global__ void m3tis_exl3_pf_rot_in(
    const __nv_bfloat16* __restrict__ x,
    int32_t x_stride,
    const int32_t* __restrict__ pair_row,
    const int32_t* __restrict__ pair_slot,
    const half* __restrict__ suh0,
    const half* __restrict__ suh1,
    half* __restrict__ out0,
    half* __restrict__ out1,
    int32_t K
) {
    const int p = blockIdx.x;
    const int blk = blockIdx.y;
    const int mat = blockIdx.z;
    const int row = pair_row[p];
    const int e = pair_slot[p];
    const int lane = threadIdx.x;

    const __nv_bfloat16* xv = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * (size_t)K + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = __bfloat162float(xv[j]) * __half2float(suh[j]);
    }
    m3tis_exl3::fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * (size_t)K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * m3tis_exl3::HAD_SCALE);
}

// Flat-tile Ampere sm_86 EXL3 MCG Tensor Core GEMM (NT=8, W=4, AHEAD=true, MAX_M=16, sk=1).
// Each CTA processes one valid <=16-row contiguous slice `[tile_off[ti] .. tile_off[ti] + tile_cnt[ti])`
// of expert `tile_slot[ti]`. Zero wasted blocks and no maxm cap!
// Launch: grid(num_tiles, N / 128, mats), block(128, 1, 1)
__global__ void __launch_bounds__(128) m3tis_exl3_pf_gemm(
    const half* __restrict__ x0_all,
    const half* __restrict__ x1_all,
    const uint64_t* __restrict__ tp0,
    const uint64_t* __restrict__ tp1,
    const int32_t* __restrict__ k2_0,
    const int32_t* __restrict__ k2_1,
    const int32_t* __restrict__ tile_slot,
    const int32_t* __restrict__ tile_off,
    const int32_t* __restrict__ tile_cnt,
    float* __restrict__ z,
    int32_t K,
    int32_t N,
    int32_t P
) {
    constexpr int NT = 8;
    constexpr int W = 4;
    constexpr int MAX_M = 16;

    const int ti = blockIdx.x;
    const int nblock = blockIdx.y;
    const int mat = blockIdx.z;

    const int e = tile_slot[ti];
    const int base_p = tile_off[ti];
    const int cnt = tile_cnt[ti];
    if (cnt <= 0) return;

    const half* X = mat ? x1_all : x0_all;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? tp1[e] : tp0[e]);
    const int k2 = mat ? k2_1[e] : k2_0[e];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int ntiles = N >> 4;

    const bool ok0 = g < cnt;
    const bool ok1 = (g + 8) < cnt;
    const half* x0 = X + (size_t)(ok0 ? (base_p + g) : 0) * (size_t)K + 2 * t;
    const half* x1 = X + (size_t)(ok1 ? (base_p + g + 8) : 0) * (size_t)K + 2 * t;

    const int per_warp = (K >> 4) / W;
    const int kt0 = warp * per_warp;
    const int nt0 = nblock * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.0f;

    switch (k2) {
#define M3TIS_EXL3_PF_K2_CASE(K2_) \
    case K2_: m3tis_exl3::warp_tiles_mcg<K2_, NT, true>(T, ntiles, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc); break;
        M3TIS_EXL3_PF_K2_CASE(2)
        M3TIS_EXL3_PF_K2_CASE(4)
        M3TIS_EXL3_PF_K2_CASE(6)
        M3TIS_EXL3_PF_K2_CASE(8)
        M3TIS_EXL3_PF_K2_CASE(10)
        M3TIS_EXL3_PF_K2_CASE(12)
        M3TIS_EXL3_PF_K2_CASE(14)
        M3TIS_EXL3_PF_K2_CASE(16)
#undef M3TIS_EXL3_PF_K2_CASE
        default:
            __trap();
    }

    __shared__ __align__(16) float red[W][MAX_M][NT * 16];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col]         = acc[i][h][0];
            red[warp][g][col + 1]     = acc[i][h][1];
            red[warp][g + 8][col]     = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    }
    __syncthreads();

    const int total_out = cnt * NT * 16;
    for (int idx = threadIdx.x; idx < total_out; idx += W * 32) {
        const int row = idx / (NT * 16);
        const int col = idx % (NT * 16);
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        z[((size_t)mat * (size_t)P + (size_t)(base_p + row)) * (size_t)N + (size_t)nt0 * 16 + col] = s;
    }
}

// Fused Down Epilogue + Weighted Scatter-Add directly into `out[pair_row[p], D]` (fp32 atomicAdd).
// Eliminates `slot_y[B*9, D]` VRAM allocation completely!
// Launch: grid(P, D / 128, 1), block(32, 1, 1)
__global__ void m3tis_exl3_pf_down_scatter(
    const float* __restrict__ z,          // [P, D] fp32
    const int32_t* __restrict__ pair_row, // [P]
    const int32_t* __restrict__ pair_slot,// [P]
    const float* __restrict__ pair_wt,    // [P]
    const half* __restrict__ svh_d,       // [experts, D]
    float* __restrict__ out,              // [B, D] fp32 (accumulated in-place!)
    int32_t D
) {
    const int p = blockIdx.x;
    const int blk = blockIdx.y;
    const int lane = threadIdx.x;
    const int col = blk * 128 + 4 * lane;
    const int row = pair_row[p];
    const int e = pair_slot[p];
    const float w = pair_wt[p];

    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = z[(size_t)p * (size_t)D + col + j];
    m3tis_exl3::fwht128(v, lane);

    const half* sv = svh_d + (size_t)e * (size_t)D + col;
    float* dst = out + (size_t)row * (size_t)D + col;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float val = v[j] * m3tis_exl3::HAD_SCALE * __half2float(sv[j]) * w;
        atomicAdd(&dst[j], val);
    }
}

} // extern "C"
