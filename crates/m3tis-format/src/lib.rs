//! `.m3tis` 4K-aligned contiguous expert container format (v1 `M3TISPK1` & v2 `M3TISPK2`).
//!
//! Design invariants:
//! - Every component offset and aligned stride is a multiple of `SECTOR_BYTES` (4096) so Linux
//!   `O_DIRECT` `io_uring` reads DMA directly into `cuMemHostAlloc` pinned RAM with zero bounce copy.
//! - v1 (`M3TISPK1`) stores uniform-bitwidth 3-component EXL3 trellises (compatible with `tools/m3tis_pack.py`).
//! - v2 (`M3TISPK2`) adds per-expert mixed bitwidth (e.g. 4b/3b hot super-experts + 2b cold NVMe tail),
//!   an optional Zipf frequency rank table for hot-set seeding, and resident scale table offsets.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub mod exl3_ref;
pub mod safetensors;
pub mod tokenizer;

pub const SECTOR_BYTES: usize = 4096;
pub const MAGIC_V1: &[u8; 8] = b"M3TISPK1";
pub const MAGIC_V2: &[u8; 8] = b"M3TISPK2";

#[inline]
pub const fn align_4k(n: usize) -> usize {
    (n + SECTOR_BYTES - 1) & !(SECTOR_BYTES - 1)
}

#[inline]
pub const fn is_4k_aligned(n: u64) -> bool {
    (n & ((SECTOR_BYTES as u64) - 1)) == 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackVersion {
    V1Uniform,
    V2MixedZipf,
}

/// Physical location of one 4K-aligned component (e.g. gate, up, or down trellis) in a `.m3tis` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentSpan {
    /// 4K-aligned byte offset from start of `.m3tis` file.
    pub offset: u64,
    /// Logical payload length in bytes (must be <= `aligned_len`).
    pub len: usize,
    /// 4K-aligned read length in bytes (always multiple of 4096).
    pub aligned_len: usize,
    /// Quantization bitwidth (e.g. 2, 3, 4, 6, 8) for EXL3 trellis decoding (`k2 = 2 * bits`).
    pub bits: u8,
}

/// Complete disk layout for one routed expert (`[gate, up, down]` trellises).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertEntry {
    pub key: usize,
    pub layer: u16,
    pub expert: u16,
    pub zipf_rank: u32,
    pub components: [ComponentSpan; 3],
}

impl ExpertEntry {
    /// Total 4K-aligned disk bytes for this expert across all 3 projections.
    #[inline]
    pub fn total_disk_bytes(&self) -> usize {
        self.components[0].aligned_len
            + self.components[1].aligned_len
            + self.components[2].aligned_len
    }

    /// True when all 3 projections are stored back-to-back contiguously in the `.m3tis` file.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        let c = &self.components;
        c[0].offset + c[0].aligned_len as u64 == c[1].offset
            && c[1].offset + c[1].aligned_len as u64 == c[2].offset
    }
}

/// Parsed `.m3tis` container index kept in host memory (tiny: ~576 KiB for 12,096 experts).
#[derive(Debug, Clone)]
pub struct PackCatalog {
    pub version: PackVersion,
    pub layers: usize,
    pub experts_per_layer: usize,
    /// Maximum 4K-aligned component stride across all experts (slab slot stride in RAM & VRAM).
    pub max_strides: [usize; 3],
    pub file_bytes: u64,
    pub entries: Vec<ExpertEntry>,
}

impl PackCatalog {
    #[inline]
    pub fn total_experts(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn max_expert_bytes(&self) -> usize {
        self.max_strides[0] + self.max_strides[1] + self.max_strides[2]
    }

    /// Parse header and table of contents of a `.m3tis` v1 or v2 file and validate 4K alignment & bounds.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let file_bytes = f.metadata()?.len();
        if file_bytes < SECTOR_BYTES as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "m3tis file smaller than 4096-byte header",
            ));
        }
        let mut hdr = [0u8; SECTOR_BYTES];
        f.read_exact(&mut hdr)?;
        if &hdr[0..8] == MAGIC_V1 {
            Self::parse_v1(&hdr, file_bytes)
        } else if &hdr[0..8] == MAGIC_V2 {
            Self::parse_v2(&mut f, &hdr, file_bytes)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unrecognized .m3tis magic (expected M3TISPK1 or M3TISPK2)",
            ))
        }
    }

    fn parse_v1(hdr: &[u8; SECTOR_BYTES], file_bytes: u64) -> io::Result<Self> {
        let layers = read_u32_le(hdr, 8) as usize;
        let experts_per_layer = read_u32_le(hdr, 12) as usize;
        let s0 = read_u64_le(hdr, 16) as usize;
        let s1 = read_u64_le(hdr, 24) as usize;
        let s2 = read_u64_le(hdr, 32) as usize;
        let l0 = read_u64_le(hdr, 40) as usize;
        let l1 = read_u64_le(hdr, 48) as usize;
        let l2 = read_u64_le(hdr, 56) as usize;

        if layers == 0 || experts_per_layer == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zero layers/experts"));
        }
        for (s, l) in [(s0, l0), (s1, l1), (s2, l2)] {
            if s == 0 || !is_4k_aligned(s as u64) || l == 0 || l > s {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid or non-4K-aligned v1 strides",
                ));
            }
        }
        let total = layers
            .checked_mul(experts_per_layer)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "expert count overflow"))?;
        let per_exp = (s0 + s1 + s2) as u64;
        let needed = (SECTOR_BYTES as u64)
            .checked_add(per_exp.saturating_mul(total as u64))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "file size overflow"))?;
        if file_bytes < needed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("m3tis v1 file truncated: {file_bytes} < {needed}"),
            ));
        }

        let mut entries = Vec::with_capacity(total);
        for li in 0..layers {
            for e in 0..experts_per_layer {
                let key = li * experts_per_layer + e;
                let base = (SECTOR_BYTES as u64) + (key as u64) * per_exp;
                entries.push(ExpertEntry {
                    key,
                    layer: li as u16,
                    expert: e as u16,
                    zipf_rank: key as u32,
                    components: [
                        ComponentSpan {
                            offset: base,
                            len: l0,
                            aligned_len: s0,
                            bits: 3,
                        },
                        ComponentSpan {
                            offset: base + s0 as u64,
                            len: l1,
                            aligned_len: s1,
                            bits: 3,
                        },
                        ComponentSpan {
                            offset: base + (s0 + s1) as u64,
                            len: l2,
                            aligned_len: s2,
                            bits: 3,
                        },
                    ],
                });
            }
        }
        Ok(Self {
            version: PackVersion::V1Uniform,
            layers,
            experts_per_layer,
            max_strides: [s0, s1, s2],
            file_bytes,
            entries,
        })
    }

    /// V2 binary layout (`M3TISPK2`):
    /// - Bytes 0..4096: Header
    ///   - `[0..8]` magic `"M3TISPK2"`
    ///   - `[8..12]` layers (u32), `[12..16]` experts_per_layer (u32)
    ///   - `[16..24]` toc_offset (u64, 4K-aligned), `[24..32]` toc_bytes (u64)
    ///   - `[32..56]` max_strides `[u64; 3]` (each 4K-aligned)
    /// - TOC: `total` fixed 64-byte records (`offset[3]: u64`, `len[3]: u32`, `aligned[3]: u32`,
    ///   `zipf_rank: u32`, `bits: [u8; 3]`, `pad: [u8; 9]`).
    fn parse_v2(f: &mut File, hdr: &[u8; SECTOR_BYTES], file_bytes: u64) -> io::Result<Self> {
        let layers = read_u32_le(hdr, 8) as usize;
        let experts_per_layer = read_u32_le(hdr, 12) as usize;
        let toc_offset = read_u64_le(hdr, 16);
        let toc_bytes = read_u64_le(hdr, 24);
        let max_strides = [
            read_u64_le(hdr, 32) as usize,
            read_u64_le(hdr, 40) as usize,
            read_u64_le(hdr, 48) as usize,
        ];
        let total = layers
            .checked_mul(experts_per_layer)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "expert count overflow"))?;
        const REC_BYTES: usize = 64;
        if total == 0 || toc_bytes != (total * REC_BYTES) as u64 || toc_offset + toc_bytes > file_bytes {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid v2 TOC bounds"));
        }
        f.seek(SeekFrom::Start(toc_offset))?;
        let mut toc = vec![0u8; toc_bytes as usize];
        f.read_exact(&mut toc)?;
        let mut entries = Vec::with_capacity(total);
        for key in 0..total {
            let rec = &toc[key * REC_BYTES..(key + 1) * REC_BYTES];
            let mut comps = [ComponentSpan {
                offset: 0,
                len: 0,
                aligned_len: 0,
                bits: 0,
            }; 3];
            for p in 0..3 {
                let offset = read_u64_le(rec, p * 8);
                let len = read_u32_le(rec, 24 + p * 4) as usize;
                let aligned_len = read_u32_le(rec, 36 + p * 4) as usize;
                let bits = rec[52 + p];
                if !is_4k_aligned(offset)
                    || !is_4k_aligned(aligned_len as u64)
                    || len == 0
                    || len > aligned_len
                    || aligned_len > max_strides[p]
                    || offset + aligned_len as u64 > file_bytes
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid v2 component alignment/bounds for expert {key} proj {p}"),
                    ));
                }
                comps[p] = ComponentSpan {
                    offset,
                    len,
                    aligned_len,
                    bits,
                };
            }
            let zipf_rank = read_u32_le(rec, 48);
            entries.push(ExpertEntry {
                key,
                layer: (key / experts_per_layer) as u16,
                expert: (key % experts_per_layer) as u16,
                zipf_rank,
                components: comps,
            });
        }
        Ok(Self {
            version: PackVersion::V2MixedZipf,
            layers,
            experts_per_layer,
            max_strides,
            file_bytes,
            entries,
        })
    }

    /// Write a valid `.m3tis` v2 mixed-bitwidth container (used by tests & the future Zipf 3b/2b packer).
    pub fn write_v2(
        path: &Path,
        layers: usize,
        experts_per_layer: usize,
        payloads: &[(u32, [(&[u8], u8); 3])],
    ) -> io::Result<Self> {
        let total = layers * experts_per_layer;
        if payloads.len() != total {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "payload count mismatch"));
        }
        const REC_BYTES: usize = 64;
        let toc_offset = SECTOR_BYTES as u64;
        let toc_bytes = (total * REC_BYTES) as u64;
        let data_start = align_4k((toc_offset + toc_bytes) as usize) as u64;

        let mut max_strides = [0usize; 3];
        let mut cursor = data_start;
        let mut toc = vec![0u8; toc_bytes as usize];

        for (key, (zipf_rank, comps)) in payloads.iter().enumerate() {
            let rec = &mut toc[key * REC_BYTES..(key + 1) * REC_BYTES];
            for p in 0..3 {
                let (bytes, bits) = comps[p];
                let al = align_4k(bytes.len());
                max_strides[p] = max_strides[p].max(al);
                rec[p * 8..(p + 1) * 8].copy_from_slice(&cursor.to_le_bytes());
                rec[24 + p * 4..28 + p * 4].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
                rec[36 + p * 4..40 + p * 4].copy_from_slice(&(al as u32).to_le_bytes());
                rec[52 + p] = bits;
                cursor += al as u64;
            }
            rec[48..52].copy_from_slice(&zipf_rank.to_le_bytes());
        }

        let mut hdr = [0u8; SECTOR_BYTES];
        hdr[0..8].copy_from_slice(MAGIC_V2);
        hdr[8..12].copy_from_slice(&(layers as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(experts_per_layer as u32).to_le_bytes());
        hdr[16..24].copy_from_slice(&toc_offset.to_le_bytes());
        hdr[24..32].copy_from_slice(&toc_bytes.to_le_bytes());
        for p in 0..3 {
            hdr[32 + p * 8..40 + p * 8].copy_from_slice(&(max_strides[p] as u64).to_le_bytes());
        }

        let mut f = File::create(path)?;
        f.write_all(&hdr)?;
        f.write_all(&toc)?;
        let pad_after_toc = (data_start - (toc_offset + toc_bytes)) as usize;
        if pad_after_toc > 0 {
            f.write_all(&vec![0u8; pad_after_toc])?;
        }
        let zero_page = [0u8; SECTOR_BYTES];
        for (_, comps) in payloads {
            for &(bytes, _) in comps {
                f.write_all(bytes)?;
                let rem = align_4k(bytes.len()) - bytes.len();
                if rem > 0 {
                    f.write_all(&zero_page[..rem])?;
                }
            }
        }
        f.flush()?;
        Self::open(path)
    }
}

#[inline]
fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

#[inline]
fn read_u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_v2_mixed_bitwidth_4k_aligned() {
        let tmp = std::env::temp_dir().join(format!("m3tis_fmt_test_{}.m3tis", std::process::id()));
        let hot = vec![0xA5u8; 8192]; // 4-bit hot super-expert (8 KiB)
        let cold = vec![0x3Cu8; 4096]; // 2-bit cold tail expert (4 KiB)
        let payloads = vec![
            (0u32, [(&hot[..], 4u8), (&hot[..], 4), (&hot[..], 4)]),
            (1u32, [(&cold[..], 2u8), (&cold[..], 2), (&cold[..], 2)]),
        ];
        let cat = PackCatalog::write_v2(&tmp, 1, 2, &payloads).expect("write v2");
        assert_eq!(cat.version, PackVersion::V2MixedZipf);
        assert_eq!(cat.max_strides, [8192, 8192, 8192]);
        assert_eq!(cat.entries[0].components[0].bits, 4);
        assert_eq!(cat.entries[0].total_disk_bytes(), 24576);
        assert_eq!(cat.entries[1].components[0].bits, 2);
        assert_eq!(cat.entries[1].total_disk_bytes(), 12288);
        assert!(cat.entries[0].is_contiguous());
        assert!(cat.entries[1].is_contiguous());
        let _ = std::fs::remove_file(&tmp);
    }
}
