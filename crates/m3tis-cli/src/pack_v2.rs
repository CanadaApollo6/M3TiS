//! `.m3tis` v2 Mixed-Bitwidth (`3b` hot / `2b` cold) Zipf Packer (`m3tis pack-v2`).
//!
//! Supports two modes:
//! 1. Dual-pack stitch (`--cold-pack <2b.m3tis>`): combines official 3b hot + official 2b cold packs.
//! 2. On-GPU 3b -> 2b MCG beam-Viterbi requantizer (`--requant-cold-2b`): keeps the hottest
//!    `--hot-frac` (default 0.30) experts at untouched 3.0 bpw and requantizes the cold 70% tail
//!    directly to 2.0 bpw on the RTX 3090 in 4K-aligned `M3TISPK2` layout (33% smaller cold NVMe reads!).

use m3tis_cuda::exl3_requant::Exl3RequantKernels;
use m3tis_cuda::CudaContext;
use m3tis_format::safetensors::drop_page_cache;
use m3tis_format::{align_4k, PackCatalog, MAGIC_V2, SECTOR_BYTES};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

pub struct PackV2Opts {
    pub hot_pack: std::path::PathBuf,
    pub cold_pack: Option<std::path::PathBuf>,
    pub out_path: std::path::PathBuf,
    pub zipf_path: Option<std::path::PathBuf>,
    pub hot_frac: f64,
    pub requant_cold_2b: bool,
}

/// Load raw `[u64; total]` routing hit counts saved via `--save-zipf` and return `(ranks, is_hot)`
/// where `ranks[key]` is global popularity rank (`0` = hottest) and `is_hot[key]` marks the top `hot_frac`.
fn build_zipf_plan(total: usize, zipf_path: Option<&Path>, hot_frac: f64) -> Result<(Vec<u32>, Vec<bool>), Box<dyn Error>> {
    let hot_n = ((total as f64) * hot_frac.clamp(0.0, 1.0)).round() as usize;
    let mut counts = vec![0u64; total];
    if let Some(zp) = zipf_path {
        let raw = std::fs::read(zp)?;
        if raw.len() != total * 8 {
            return Err(format!("zipf profile {} has {} bytes, expected {}", zp.display(), raw.len(), total * 8).into());
        }
        for (i, ch) in raw.chunks_exact(8).enumerate() {
            counts[i] = u64::from_le_bytes(ch.try_into().unwrap());
        }
    } else {
        // Default deterministic fallback when no profile given: lowest expert indices per layer treated as hot
        for i in 0..total {
            counts[i] = (total - i) as u64;
        }
    }

    let mut order: Vec<usize> = (0..total).collect();
    order.sort_unstable_by(|&a, &b| counts[b].cmp(&counts[a]).then_with(|| a.cmp(&b)));

    let mut ranks = vec![0u32; total];
    let mut is_hot = vec![false; total];
    for (rank, &key) in order.iter().enumerate() {
        ranks[key] = rank as u32;
        if rank < hot_n {
            is_hot[key] = true;
        }
    }
    Ok((ranks, is_hot))
}

pub fn run_pack_v2(opts: &PackV2Opts) -> Result<(), Box<dyn Error>> {
    let t0 = Instant::now();
    let hot_cat = PackCatalog::open(&opts.hot_pack)?;
    let total = hot_cat.total_experts();
    let layers = hot_cat.layers;
    let e_per = hot_cat.experts_per_layer;

    if opts.cold_pack.is_none() && !opts.requant_cold_2b {
        return Err("pack-v2 requires either --cold-pack <2b.m3tis> or --requant-cold-2b".into());
    }

    let cold_cat = match &opts.cold_pack {
        Some(p) => {
            let c = PackCatalog::open(p)?;
            if c.total_experts() != total {
                return Err(format!("cold pack expert count {} != hot {}", c.total_experts(), total).into());
            }
            Some(c)
        }
        None => None,
    };

    let (ranks, is_hot) = build_zipf_plan(total, opts.zipf_path.as_deref(), opts.hot_frac)?;
    let hot_count = is_hot.iter().filter(|&&h| h).count();
    let cold_count = total - hot_count;
    if let Some(zp) = opts.zipf_path.as_deref() {
        if let Ok(raw) = std::fs::read(zp) {
            if raw.len() == total * 8 {
                let mut tot_hits = 0u64;
                let mut hot_hits = 0u64;
                for (i, ch) in raw.chunks_exact(8).enumerate() {
                    let c = u64::from_le_bytes(ch.try_into().unwrap());
                    tot_hits += c;
                    if is_hot[i] {
                        hot_hits += c;
                    }
                }
                if tot_hits > 0 {
                    println!(
                        "[pack-v2] Zipf profile {}: top {:.0}% ({hot_count}/{total}) covers {:.1}% of routed token mass ({hot_hits}/{tot_hits} picks)",
                        zp.display(),
                        100.0 * hot_count as f64 / total as f64,
                        100.0 * hot_hits as f64 / tot_hits as f64
                    );
                }
            }
        }
    }

    // Verify 3-bit source tile counts if GPU requantizing
    let mut cold_lens = [0usize; 3];
    let mut cold_bits = [2u8; 3];
    if let Some(cc) = &cold_cat {
        for p in 0..3 {
            cold_lens[p] = cc.entries[0].components[p].len;
            cold_bits[p] = cc.entries[0].components[p].bits;
        }
    } else {
        for p in 0..3 {
            let len3 = hot_cat.entries[0].components[p].len;
            if len3 % 96 != 0 {
                return Err(format!("hot trellis proj {p} len {len3} not divisible by 96 (3-bit tile)").into());
            }
            let n_tiles = len3 / 96;
            cold_lens[p] = n_tiles * 64; // exact 2-bit EXL3 MCG trellis bytes!
            cold_bits[p] = 2;
        }
    }

    // Build v2 TOC layout in memory upfront (4K-aligned offsets & strides)
    const REC_BYTES: usize = 64;
    let toc_offset = SECTOR_BYTES as u64;
    let toc_bytes = (total * REC_BYTES) as u64;
    let data_start = align_4k((toc_offset + toc_bytes) as usize) as u64;

    let mut max_strides = [0usize; 3];
    let mut toc = vec![0u8; toc_bytes as usize];
    let mut cursor = data_start;

    for key in 0..total {
        let rec = &mut toc[key * REC_BYTES..(key + 1) * REC_BYTES];
        let he = &hot_cat.entries[key];
        for p in 0..3 {
            let (len, bits) = if is_hot[key] {
                (he.components[p].len, he.components[p].bits)
            } else if let Some(cc) = &cold_cat {
                let c = cc.entries[key].components[p];
                let inferred_bits = if cc.version == m3tis_format::PackVersion::V1Uniform
                    && he.components[p].len > 0
                    && (c.len * he.components[p].bits as usize) % he.components[p].len == 0
                {
                    ((c.len * he.components[p].bits as usize) / he.components[p].len) as u8
                } else {
                    c.bits
                };
                (c.len, inferred_bits)
            } else {
                (cold_lens[p], cold_bits[p])
            };
            let al = align_4k(len);
            max_strides[p] = max_strides[p].max(al);
            rec[p * 8..(p + 1) * 8].copy_from_slice(&cursor.to_le_bytes());
            rec[24 + p * 4..28 + p * 4].copy_from_slice(&(len as u32).to_le_bytes());
            rec[36 + p * 4..40 + p * 4].copy_from_slice(&(al as u32).to_le_bytes());
            rec[52 + p] = bits;
            cursor += al as u64;
        }
        rec[48..52].copy_from_slice(&ranks[key].to_le_bytes());
    }

    let mut hdr = [0u8; SECTOR_BYTES];
    hdr[0..8].copy_from_slice(MAGIC_V2);
    hdr[8..12].copy_from_slice(&(layers as u32).to_le_bytes());
    hdr[12..16].copy_from_slice(&(e_per as u32).to_le_bytes());
    hdr[16..24].copy_from_slice(&toc_offset.to_le_bytes());
    hdr[24..32].copy_from_slice(&toc_bytes.to_le_bytes());
    for p in 0..3 {
        hdr[32 + p * 8..40 + p * 8].copy_from_slice(&(max_strides[p] as u64).to_le_bytes());
    }

    println!(
        "[pack-v2] {} layers x {} experts ({} total) | hot={} (3b) cold={} (2b) | mode={} | output={:.2} GiB (vs {:.2} GiB v1)",
        layers,
        e_per,
        total,
        hot_count,
        cold_count,
        if cold_cat.is_some() { "dual-pack" } else { "gpu-requant-3b-to-2b" },
        cursor as f64 / (1u64 << 30) as f64,
        hot_cat.file_bytes as f64 / (1u64 << 30) as f64
    );

    // Pre-create file and apply btrfs chattr +C (nodatacow) before writing!
    if let Some(parent) = opts.out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if opts.out_path.exists() {
        std::fs::remove_file(&opts.out_path)?;
    }
    File::create(&opts.out_path)?;
    let _ = Command::new("chattr").arg("+C").arg(&opts.out_path).status();

    let out_file = OpenOptions::new().write(true).open(&opts.out_path)?;
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, out_file);
    out.write_all(&hdr)?;
    out.write_all(&toc)?;
    let pad_toc = (data_start - (toc_offset + toc_bytes)) as usize;
    if pad_toc > 0 {
        out.write_all(&vec![0u8; pad_toc])?;
    }

    let mut hot_f = File::open(&opts.hot_pack)?;
    let mut cold_f = match &opts.cold_pack {
        Some(p) => Some(File::open(p)?),
        None => None,
    };

    // Optional GPU requantizer context & reusable staging buffers (holds 1 full projection at a time)
    let gpu = if cold_cat.is_none() {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.stream()?;
        let kern = Exl3RequantKernels::load(&ctx)?;
        let max_in = hot_cat.max_strides.iter().copied().max().unwrap();
        let max_out = cold_lens.iter().copied().max().unwrap();
        let d_in = ctx.alloc_device(max_in)?;
        let d_out = ctx.alloc_device(max_out)?;
        Some((ctx, stream, kern, d_in, d_out))
    } else {
        None
    };

    let max_comp = hot_cat.max_strides.iter().copied().max().unwrap();
    let mut read_buf = vec![0u8; max_comp];
    let mut out_buf = vec![0u8; max_comp];
    let zero_page = [0u8; SECTOR_BYTES];
    let mut written_bytes = data_start;

    for key in 0..total {
        let he = &hot_cat.entries[key];
        for p in 0..3 {
            if is_hot[key] {
                let c = he.components[p];
                hot_f.seek(SeekFrom::Start(c.offset))?;
                hot_f.read_exact(&mut read_buf[..c.len])?;
                out.write_all(&read_buf[..c.len])?;
                let pad = align_4k(c.len) - c.len;
                if pad > 0 {
                    out.write_all(&zero_page[..pad])?;
                }
                written_bytes += align_4k(c.len) as u64;
            } else if let (Some(cc), Some(cf)) = (&cold_cat, cold_f.as_mut()) {
                let c = cc.entries[key].components[p];
                cf.seek(SeekFrom::Start(c.offset))?;
                cf.read_exact(&mut read_buf[..c.len])?;
                out.write_all(&read_buf[..c.len])?;
                let pad = align_4k(c.len) - c.len;
                if pad > 0 {
                    out.write_all(&zero_page[..pad])?;
                }
                written_bytes += align_4k(c.len) as u64;
            } else {
                let (_ctx, stream, kern, d_in, d_out) = gpu.as_ref().unwrap();
                let c = he.components[p];
                hot_f.seek(SeekFrom::Start(c.offset))?;
                hot_f.read_exact(&mut read_buf[..c.len])?;
                let n_tiles = c.len / 96;
                let out_len = n_tiles * 64;
                unsafe {
                    stream.htod_async(d_in.ptr(), read_buf.as_ptr(), c.len)?;
                }
                kern.requant_3b_to_2b(stream, d_in.ptr(), d_out.ptr(), n_tiles)?;
                unsafe {
                    stream.dtoh_async(out_buf.as_mut_ptr(), d_out.ptr(), out_len)?;
                }
                stream.synchronize()?;
                out.write_all(&out_buf[..out_len])?;
                let pad = align_4k(out_len) - out_len;
                if pad > 0 {
                    out.write_all(&zero_page[..pad])?;
                }
                written_bytes += align_4k(out_len) as u64;
            }
        }
        if (key + 1) % e_per == 0 || key + 1 == total {
            out.flush()?;
            let _ = out.get_ref().sync_data();
            drop_page_cache(&hot_f);
            if let Some(cf) = cold_f.as_ref() {
                drop_page_cache(cf);
            }
            drop_page_cache(out.get_ref());
            let done_layers = (key + 1) / e_per;
            if done_layers % 3 == 0 || key + 1 == total {
                let dt = t0.elapsed().as_secs_f64();
                let gb = written_bytes as f64 / 1e9;
                println!(
                    "  [pack-v2] layer {done_layers}/{layers} ({}/{total} experts) | {gb:.2} GB in {dt:.1}s ({:.2} GB/s)",
                    key + 1,
                    gb / dt.max(1e-6)
                );
            }
        }
    }
    out.flush()?;
    drop(out);

    // Self-verify written M3TISPK2 container header & TOC immediately!
    let verify_cat = PackCatalog::open(&opts.out_path)?;
    println!(
        "[pack-v2 DONE] {} | {:?} | {:.2} GiB | max_strides={:?} | verified {} entries in {:.1}s",
        opts.out_path.display(),
        verify_cat.version,
        verify_cat.file_bytes as f64 / (1u64 << 30) as f64,
        verify_cat.max_strides,
        verify_cat.total_experts(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use m3tis_format::{PackVersion, MAGIC_V1};

    fn write_tiny_v1_pack(path: &Path, layers: usize, e_per: usize, comp_len: usize, fill: u8) {
        let total = layers * e_per;
        let al = align_4k(comp_len);
        let mut hdr = [0u8; SECTOR_BYTES];
        hdr[0..8].copy_from_slice(MAGIC_V1);
        hdr[8..12].copy_from_slice(&(layers as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(e_per as u32).to_le_bytes());
        for p in 0..3 {
            hdr[16 + p * 8..24 + p * 8].copy_from_slice(&(al as u64).to_le_bytes());
            hdr[40 + p * 8..48 + p * 8].copy_from_slice(&(comp_len as u64).to_le_bytes());
        }
        let mut f = File::create(path).unwrap();
        f.write_all(&hdr).unwrap();
        let mut blk = vec![0u8; al];
        blk[..comp_len].fill(fill);
        for _ in 0..(total * 3) {
            f.write_all(&blk).unwrap();
        }
    }

    #[test]
    fn dual_pack_v2_stitch_respects_zipf_hot_cold_layout() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let hot_p = dir.join(format!("m3tis_hot_{pid}.m3tis"));
        let cold_p = dir.join(format!("m3tis_cold_{pid}.m3tis"));
        let zipf_p = dir.join(format!("m3tis_zipf_{pid}.bin"));
        let out_p = dir.join(format!("m3tis_out_{pid}.m3tis"));

        // 2 layers x 4 experts = 8 experts; hot len=3072 (3b), cold len=2048 (2b)
        write_tiny_v1_pack(&hot_p, 2, 4, 3072, 0x3B);
        write_tiny_v1_pack(&cold_p, 2, 4, 2048, 0x2B);

        // Experts 5 and 2 are hottest (top 25% of 8 = 2 hot experts)
        let hits: [u64; 8] = [1, 2, 90, 3, 4, 100, 5, 6];
        let mut zbytes = Vec::new();
        for h in hits {
            zbytes.extend_from_slice(&h.to_le_bytes());
        }
        std::fs::write(&zipf_p, &zbytes).unwrap();

        run_pack_v2(&PackV2Opts {
            hot_pack: hot_p.clone(),
            cold_pack: Some(cold_p.clone()),
            out_path: out_p.clone(),
            zipf_path: Some(zipf_p.clone()),
            hot_frac: 0.25,
            requant_cold_2b: false,
        })
        .unwrap();

        let cat = PackCatalog::open(&out_p).unwrap();
        assert_eq!(cat.version, PackVersion::V2MixedZipf);
        assert_eq!(cat.total_experts(), 8);
        assert_eq!(cat.entries[5].zipf_rank, 0);
        assert_eq!(cat.entries[2].zipf_rank, 1);
        assert_eq!(cat.entries[5].components[0].len, 3072);
        assert_eq!(cat.entries[5].components[0].bits, 3);
        assert_eq!(cat.entries[2].components[0].len, 3072);
        assert_eq!(cat.entries[2].components[0].bits, 3);
        assert_eq!(cat.entries[0].components[0].len, 2048);
        assert_eq!(cat.entries[0].components[0].bits, 2);

        let _ = std::fs::remove_file(&hot_p);
        let _ = std::fs::remove_file(&cold_p);
        let _ = std::fs::remove_file(&zipf_p);
        let _ = std::fs::remove_file(&out_p);
    }
}
