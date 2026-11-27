//! Clean-room CPU FP64/FP32 reference implementation of EXL3 MCG decoding, FWHT128, and SwiGLU.
//! Used exclusively to verify our sm_86 Tensor Core CUDA kernels to strict numerical tolerances
//! on both synthetic mixed-bitwidth (2b/3b/4b) experts and real GLM-5.3-Flash checkpoint tensors.

pub const HAD_SCALE: f64 = 0.08838834764831845; // 1 / sqrt(128)

/// IEEE-754 binary16 -> f32 exact conversion.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let mut f = frac;
        let mut e = -14i32;
        while (f & 0x400) == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3FF;
        let f_exp = ((e + 127) as u32) << 23;
        f32::from_bits(sign | f_exp | (f << 13))
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (frac << 13))
    } else {
        let f_exp = ((exp - 15 + 127) as u32) << 23;
        f32::from_bits(sign | f_exp | (frac << 13))
    }
}

/// Round-to-nearest-even f32 -> bf16 -> f32.
#[inline]
pub fn bf16_round(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let u = x.to_bits();
    let lsb = (u >> 16) & 1;
    let rounded = u.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    (bf16_round(x).to_bits() >> 16) as u16
}

/// Exact MCG codebook decode for one 16-bit trellis state (matches `decode2_mcg` in exl3_moe.cu).
#[inline]
pub fn decode_mcg_state(state: u32) -> f32 {
    let x = (state.wrapping_mul(0xCBAC1FEDu32) & 0x8FFF8FFFu32) ^ 0x3B603B60u32;
    let lo = f16_to_f32((x & 0xFFFF) as u16);
    let hi = f16_to_f32((x >> 16) as u16);
    lo + hi
}

#[inline]
fn stream_end(k2: usize, p: usize) -> usize {
    if (k2 & 1) != 0 {
        ((p + 1) * k2 - ((p + 1) & 1)) / 2
    } else {
        (p + 1) * (k2 / 2)
    }
}

/// Unpack one 16x16 EXL3 MCG tile (`4 * k2` u32 words) into row-major `[16][16]` f32 (`W[k, n]`).
pub fn decode_mcg_tile_16x16(tile_words: &[u32], k2: usize) -> [f32; 256] {
    let tw = 4 * k2;
    assert_eq!(tile_words.len(), tw);
    let total_bits = tw * 32;
    let mut out = [0.0f32; 256];
    for lane in 0..32usize {
        for j in 0..8usize {
            let p = 8 * lane + j;
            let end = stream_end(k2, p);
            let start = (end + total_bits - 16) % total_bits;
            let mut state = 0u32;
            for b in 0..16usize {
                let bit_pos = (start + b) % total_bits;
                let w = tile_words[bit_pos >> 5];
                let bit = (w >> (31 - (bit_pos & 31))) & 1;
                state = (state << 1) | bit;
            }
            let row = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1);
            let col = (lane >> 2) + 8 * (j >> 2);
            out[row * 16 + col] = decode_mcg_state(state);
        }
    }
    out
}

/// Exact 128-point Walsh-Hadamard transform scaled by `1/sqrt(128)` matching GPU `fwht128`.
pub fn fwht128_scaled(data: &mut [f64]) {
    assert_eq!(data.len() % 128, 0);
    for blk in data.chunks_exact_mut(128) {
        // Local 4-element butterfly per lane (matches GPU thread-local butterfly)
        for lane in 0..32 {
            let base = lane * 4;
            let a = blk[base] + blk[base + 1];
            let b = blk[base] - blk[base + 1];
            let c = blk[base + 2] + blk[base + 3];
            let d = blk[base + 2] - blk[base + 3];
            blk[base] = a + c;
            blk[base + 1] = b + d;
            blk[base + 2] = a - c;
            blk[base + 3] = b - d;
        }
        // Warp shuffle butterfly across 32 lanes (masks 1, 2, 4, 8, 16)
        for m in [1usize, 2, 4, 8, 16] {
            for lane in 0..32 {
                if (lane & m) == 0 {
                    let other = lane | m;
                    for j in 0..4 {
                        let u = blk[lane * 4 + j];
                        let v = blk[other * 4 + j];
                        blk[lane * 4 + j] = u + v;
                        blk[other * 4 + j] = u - v;
                    }
                }
            }
        }
        for v in blk.iter_mut() {
            *v *= HAD_SCALE;
        }
    }
}

/// Reference EXL3 MCG matvec (`y[N] = fwht128((fwht128(x * suh)) @ W_q) * svh`) in FP64.
pub fn reference_exl3_matvec(
    x: &[f32],         // [K]
    trellis: &[u8],    // [K/16, N/16, 4*k2] u32 little-endian
    k2: usize,         // 2 * bits
    suh_f16: &[u16],   // [K]
    svh_f16: &[u16],   // [N]
    k_dim: usize,
    n_dim: usize,
) -> Vec<f32> {
    let kt = k_dim / 16;
    let nt = n_dim / 16;
    let tw = 4 * k2;
    assert_eq!(trellis.len(), kt * nt * tw * 4);

    let mut xh: Vec<f64> = (0..k_dim)
        .map(|i| (x[i] as f64) * (f16_to_f32(suh_f16[i]) as f64))
        .collect();
    fwht128_scaled(&mut xh);

    let mut z = vec![0.0f64; n_dim];
    let mut tile_u32 = vec![0u32; tw];
    for ki in 0..kt {
        for ni in 0..nt {
            let off = (ki * nt + ni) * tw * 4;
            for w in 0..tw {
                tile_u32[w] = u32::from_le_bytes(trellis[off + w * 4..off + w * 4 + 4].try_into().unwrap());
            }
            let mat = decode_mcg_tile_16x16(&tile_u32, k2);
            for r in 0..16 {
                let xv = xh[ki * 16 + r];
                for c in 0..16 {
                    z[ni * 16 + c] += xv * (mat[r * 16 + c] as f64);
                }
            }
        }
    }
    fwht128_scaled(&mut z);
    (0..n_dim)
        .map(|j| (z[j] * (f16_to_f32(svh_f16[j]) as f64)) as f32)
        .collect()
}

/// Reference complete EXL3 SwiGLU expert MLP (`down(swiglu(gate(x), up(x)))`) matching GLM-5.3.
#[allow(clippy::too_many_arguments)]
pub fn reference_exl3_expert_mlp(
    x: &[f32],
    gate_t: &[u8],
    up_t: &[u8],
    down_t: &[u8],
    k2_g: usize,
    k2_u: usize,
    k2_d: usize,
    suh_g: &[u16],
    suh_u: &[u16],
    svh_g: &[u16],
    svh_u: &[u16],
    suh_d: &[u16],
    svh_d: &[u16],
    dims: usize,
    inter: usize,
    limit: f32,
) -> Vec<f32> {
    let g = reference_exl3_matvec(x, gate_t, k2_g, suh_g, svh_g, dims, inter);
    let u = reference_exl3_matvec(x, up_t, k2_u, suh_u, svh_u, dims, inter);
    let mut act = vec![0.0f32; inter];
    for i in 0..inter {
        let mut gg = bf16_round(g[i]);
        let mut uu = bf16_round(u[i]);
        if limit > 0.0 {
            if gg > limit {
                gg = limit;
            }
            if uu > limit {
                uu = limit;
            } else if uu < -limit {
                uu = -limit;
            }
        }
        let silu = bf16_round(gg / (1.0 + (-gg).exp()));
        act[i] = bf16_round(silu * uu);
    }
    reference_exl3_matvec(&act, down_t, k2_d, suh_d, svh_d, inter, dims)
}
