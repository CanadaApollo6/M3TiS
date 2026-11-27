//! M3TiS CLI — clean-room Rust + CUDA/PTX 3-tier MoE inference engine.
//!
//! Subcommands:
//! - `info <pack.m3tis>`        : Inspect `.m3tis` pack catalog, GPU VRAM, and Linux `io_uring`.
//! - `bench-io <pack.m3tis>`    : Raw zero-copy `O_DIRECT` `io_uring` benchmark directly into CUDA pinned RAM.
//! - `bench-pool <pack.m3tis>`  : Full 3-tier `VRAM <- Pinned RAM <- Dual-Ring NVMe` replay benchmark.
//! - `bench-moe <dir> [pack]`   : Hardware parity + latency gate for EXL3 MoE kernels.
//! - `forward-check <dir> [pack]`: End-to-end forward/decode benchmark and parity check.
//! - `serve <dir> [pack]`       : OpenAI-compatible HTTP/1.1 + SSE inference server.
//! - `pack-v2 <in> <out>`       : Mixed-bitwidth (`3b` hot / `2b` cold) Zipf `.m3tis` v2 packer.

mod pack_v2;
mod server;

use m3tis_cache::scales::Exl3ScaleBank;
use m3tis_cache::{EvictionPolicy, TieredExpertPool};
use m3tis_cuda::exl3::{Exl3ExpertsView, Exl3MoeKernels, Exl3MoeScratch, InputDType};
use m3tis_cuda::{CudaContext, RouterRemapKernels};
use m3tis_format::exl3_ref::{f32_to_bf16, reference_exl3_expert_mlp};
use m3tis_format::safetensors::SafetensorsIndex;
use m3tis_format::tokenizer::{BpeTokenizer, ChatMessage};
use m3tis_format::PackCatalog;
use m3tis_io::{DirectFile, IoRing, IoStats, ReadOp};
use m3tis_models::glm53::{ForwardOpts, Glm53LoadOptions, Glm53Model};
use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::{Duration, Instant};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(2);
    }
    let res = match args[1].as_str() {
        "info" => cmd_info(&args[2..]),
        "bench-io" => cmd_bench_io(&args[2..]),
        "bench-pool" => cmd_bench_pool(&args[2..]),
        "bench-moe" => cmd_bench_moe(&args[2..]),
        "forward-check" => cmd_forward_check(&args[2..]),
        "serve" => cmd_serve(&args[2..]),
        "pack-v2" => cmd_pack_v2(&args[2..]),
        _ => {
            print_usage();
            std::process::exit(2);
        }
    };
    if let Err(e) = res {
        eprintln!("m3tis error: {e}");
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!(
        "M3TiS v{} (Rust + CUDA sm_86 + Linux dual-io_uring)",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("Usage:");
    eprintln!("  m3tis info <pack.m3tis>");
    eprintln!("  m3tis bench-io <pack.m3tis> [--qd 24] [--gb 12] [--fixed] [--sqpoll]");
    eprintln!("  m3tis bench-pool <pack.m3tis> [--ram-gib 20] [--vram-gib 4.5] [--rounds 25] [--spec 3] [--policy 2q|clock]");
    eprintln!("  m3tis bench-moe <checkpoint_dir> [pack.m3tis] [--iters 50]");
    eprintln!("  m3tis forward-check <checkpoint_dir> [pack.m3tis] (--tokens <csv> | --prompt <text> | --prompt-file <path>)");
    eprintln!("      [--preset exact|balanced|turbo|ludicrous|warp] [--drafter <dir>] [--max-tokens 64] [--max-ctx 2048] [--prefill-batch 2048]");
    eprintln!("      [--kv-int8 | --no-kv-int8] [--prefix-cache-dir <dir>] [--prefix-cache-max-gib 8.0]");
    eprintln!("      [--ram-gib 10] [--vram-gib <f>] [--prewarm | --no-prewarm] [--prewarm-ram | --no-prewarm-ram]");
    eprintln!("      [--spec 4] [--lookahead 1] [--residency-bias 0.0] [--nvme-mass 0.0] [--nvme-prune-prob 0.0]");
    eprintln!("      [--max-drafts 7] [--draft-conf 0.7] [--warm-runs 1] [--save-zipf <path>]");
    eprintln!("  m3tis serve <checkpoint_dir> [pack.m3tis] [--preset exact|balanced|turbo|ludicrous|warp] [--drafter <dir>]");
    eprintln!("      [--host 127.0.0.1] [--port 8080] [--model-name glm-5.3-flash] [--max-ctx 2048] [--prefill-batch 2048]");
    eprintln!("      [--kv-int8 | --no-kv-int8] [--prefix-cache-slots 4] [--prefix-cache-dir <dir>] [--ram-gib 10] [--thinking]");
    eprintln!("  m3tis pack-v2 <hot_3b.m3tis> <out_v2.m3tis> (--cold-pack <2b.m3tis> | --requant-cold-2b) [--zipf <hits.bin>] [--hot-frac 0.30]");
}

/// Resolve `<checkpoint_dir> [pack.m3tis]`.
/// When `[pack.m3tis]` is omitted (or next arg is a `--flag`), auto-discovers a single `*.m3tis` inside `<checkpoint_dir>`.
fn resolve_ckpt_and_pack(args: &[String]) -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let ckpt_dir = PathBuf::from(args.first().ok_or("missing <checkpoint_dir>")?);
    if let Some(arg1) = args.get(1) {
        if !arg1.starts_with('-') {
            return Ok((ckpt_dir, PathBuf::from(arg1)));
        }
    }
    let mut packs: Vec<PathBuf> = std::fs::read_dir(&ckpt_dir)?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("m3tis"))
        .collect();
    packs.sort();
    match packs.len() {
        1 => Ok((ckpt_dir, packs.remove(0))),
        0 => Err(format!(
            "missing <pack.m3tis> and no *.m3tis file found in {}",
            ckpt_dir.display()
        )
        .into()),
        _ => Err(format!(
            "multiple *.m3tis files in {}; pass <pack.m3tis> explicitly",
            ckpt_dir.display()
        )
        .into()),
    }
}

fn parse_flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == name {
            if let Ok(v) = args[i + 1].parse::<T>() {
                return v;
            }
        }
    }
    default
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

struct DecodeTuning {
    fwd: ForwardOpts,
    max_drafts: usize,
    draft_conf: f64,
    prewarm_vram: bool,
    prewarm_ram: bool,
    dense_q3: bool,
    dense_q3_moe: bool,
    drafter_q3: bool,
}

/// Resolve Milestone 4A/4B `--preset exact|balanced|turbo|ludicrous|warp` defaults, then apply explicit CLI overrides.
fn parse_decode_tuning(args: &[String]) -> Result<DecodeTuning, Box<dyn std::error::Error>> {
    let preset: String = parse_flag(args, "--preset", String::new());
    let mut t = match preset.as_str() {
        "" | "exact" => DecodeTuning {
            fwd: ForwardOpts {
                spec_prefetch_p: 4,
                lookahead_depth: 1,
                dyn_topk_mass: 0.0,
                dyn_topk_min_k: 4,
                residency_bias: 0.0,
                residency_strict_k: 3,
                nvme_strict_k: 0,
                nvme_mass: 0.0,
                nvme_tail_row: 0,
                nvme_prune_row: 0,
                nvme_prune_prob: 0.0,
                zero_stall_mass: 0.0,
                self_spec_k: 0,
                self_spec_conf: 0.70,
                cpu_hybrid_max: 0,
                cpu_hybrid_always: false,
                adaptive_draft: false,
                tree_verify: true,
            },
            max_drafts: 7,
            draft_conf: 0.7,
            prewarm_vram: false,
            prewarm_ram: false,
            dense_q3: false,
            dense_q3_moe: false,
            drafter_q3: false,
        },
        "balanced" => DecodeTuning {
            fwd: ForwardOpts {
                spec_prefetch_p: 0,
                lookahead_depth: 1,
                dyn_topk_mass: 0.0,
                dyn_topk_min_k: 4,
                residency_bias: 0.04,
                residency_strict_k: 3,
                nvme_strict_k: 3,
                nvme_mass: 0.0,
                nvme_tail_row: 0,
                nvme_prune_row: 0,
                nvme_prune_prob: 0.0,
                zero_stall_mass: 0.0,
                self_spec_k: 0,
                self_spec_conf: 0.70,
                cpu_hybrid_max: 0,
                cpu_hybrid_always: false,
                adaptive_draft: false,
                tree_verify: true,
            },
            max_drafts: 4,
            draft_conf: 0.75,
            prewarm_vram: false,
            prewarm_ram: false,
            dense_q3: false,
            dense_q3_moe: false,
            drafter_q3: false,
        },
        "turbo" => DecodeTuning {
            fwd: ForwardOpts {
                spec_prefetch_p: 0,
                lookahead_depth: 1,
                dyn_topk_mass: 0.0,
                dyn_topk_min_k: 4,
                residency_bias: 0.05,
                residency_strict_k: 2,
                nvme_strict_k: 1,
                nvme_mass: 0.65,
                nvme_tail_row: 0,
                nvme_prune_row: 0,
                nvme_prune_prob: 0.0,
                zero_stall_mass: 0.0,
                self_spec_k: 0,
                self_spec_conf: 0.70,
                cpu_hybrid_max: 0,
                cpu_hybrid_always: false,
                adaptive_draft: false,
                tree_verify: true,
            },
            max_drafts: 4,
            draft_conf: 0.75,
            prewarm_vram: false,
            prewarm_ram: false,
            dense_q3: false,
            dense_q3_moe: false,
            drafter_q3: false,
        },
        "ludicrous" => DecodeTuning {
            fwd: ForwardOpts {
                spec_prefetch_p: 0,
                lookahead_depth: 1,
                dyn_topk_mass: 0.0,
                dyn_topk_min_k: 4,
                residency_bias: 0.06,
                residency_strict_k: 2,
                nvme_strict_k: 1,
                nvme_mass: 0.50,
                nvme_tail_row: 0,
                nvme_prune_row: 0,
                nvme_prune_prob: 0.0,
                zero_stall_mass: 0.0,
                self_spec_k: 0,
                self_spec_conf: 0.70,
                cpu_hybrid_max: 0,
                cpu_hybrid_always: false,
                adaptive_draft: false,
                tree_verify: true,
            },
            max_drafts: 4,
            draft_conf: 0.75,
            prewarm_vram: false,
            prewarm_ram: false,
            dense_q3: false,
            dense_q3_moe: false,
            drafter_q3: false,
        },
        "warp" => DecodeTuning {
            fwd: ForwardOpts {
                spec_prefetch_p: 0,
                lookahead_depth: 1,
                dyn_topk_mass: 0.0,
                dyn_topk_min_k: 4,
                residency_bias: 0.05,
                residency_strict_k: 2,
                nvme_strict_k: 1,
                nvme_mass: 0.45,
                nvme_tail_row: 0,
                nvme_prune_row: 0,
                nvme_prune_prob: 0.65,
                zero_stall_mass: 0.0,
                self_spec_k: 0,
                self_spec_conf: 0.70,
                cpu_hybrid_max: 0,
                cpu_hybrid_always: false,
                adaptive_draft: false,
                tree_verify: false,
            },
            max_drafts: 7,
            draft_conf: 0.45,
            prewarm_vram: true,
            prewarm_ram: true,
            dense_q3: false,
            dense_q3_moe: false,
            drafter_q3: false,
        },
        other => {
            return Err(format!(
                "unknown --preset '{other}' (expected exact|balanced|turbo|ludicrous|warp)"
            )
            .into());
        }
    };
    t.fwd.spec_prefetch_p = parse_flag(args, "--spec", t.fwd.spec_prefetch_p);
    t.fwd.lookahead_depth = parse_flag(args, "--lookahead", t.fwd.lookahead_depth);
    t.fwd.dyn_topk_mass = parse_flag(args, "--dyn-mass", t.fwd.dyn_topk_mass);
    t.fwd.dyn_topk_min_k = parse_flag(args, "--dyn-min-k", t.fwd.dyn_topk_min_k);
    t.fwd.residency_bias = parse_flag(args, "--residency-bias", t.fwd.residency_bias);
    t.fwd.residency_strict_k = parse_flag(args, "--residency-strict-k", t.fwd.residency_strict_k);
    t.fwd.nvme_strict_k = parse_flag(args, "--nvme-strict-k", t.fwd.nvme_strict_k);
    t.fwd.nvme_mass = parse_flag(args, "--nvme-mass", t.fwd.nvme_mass);
    t.fwd.nvme_tail_row = parse_flag(args, "--nvme-tail-row", t.fwd.nvme_tail_row);
    t.fwd.nvme_prune_row = parse_flag(args, "--nvme-prune-row", t.fwd.nvme_prune_row);
    t.fwd.nvme_prune_prob = parse_flag(args, "--nvme-prune-prob", t.fwd.nvme_prune_prob);
    t.fwd.zero_stall_mass = parse_flag(args, "--zero-stall", t.fwd.zero_stall_mass);
    t.fwd.self_spec_k = parse_flag(args, "--self-spec", t.fwd.self_spec_k);
    t.fwd.self_spec_conf = parse_flag(args, "--self-spec-conf", t.fwd.self_spec_conf);
    t.fwd.cpu_hybrid_max = parse_flag(args, "--cpu-hybrid", t.fwd.cpu_hybrid_max);
    if has_flag(args, "--cpu-hybrid-always") {
        t.fwd.cpu_hybrid_always = true;
    }
    if has_flag(args, "--adaptive-draft") {
        t.fwd.adaptive_draft = true;
    }
    if has_flag(args, "--no-adaptive-draft") {
        t.fwd.adaptive_draft = false;
    }
    if has_flag(args, "--tree-verify") {
        t.fwd.tree_verify = true;
    }
    if has_flag(args, "--no-tree-verify") {
        t.fwd.tree_verify = false;
    }
    t.max_drafts = parse_flag(args, "--max-drafts", t.max_drafts);
    t.draft_conf = parse_flag(args, "--draft-conf", t.draft_conf);
    if has_flag(args, "--prewarm") {
        t.prewarm_vram = true;
    }
    if has_flag(args, "--no-prewarm") {
        t.prewarm_vram = false;
        t.prewarm_ram = false;
    }
    if has_flag(args, "--prewarm-ram") {
        t.prewarm_vram = true;
        t.prewarm_ram = true;
    }
    if has_flag(args, "--no-prewarm-ram") {
        t.prewarm_ram = false;
    }
    if has_flag(args, "--dense-q3") {
        t.dense_q3 = true;
    }
    if has_flag(args, "--no-dense-q3") {
        t.dense_q3 = false;
        t.dense_q3_moe = false;
    }
    if has_flag(args, "--dense-q3-moe") {
        t.dense_q3_moe = true;
    }
    if has_flag(args, "--drafter-q3") {
        t.drafter_q3 = true;
    }
    if has_flag(args, "--no-drafter-q3") {
        t.drafter_q3 = false;
    }
    Ok(t)
}

fn cmd_info(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(args.first().ok_or("missing <pack.m3tis>")?);
    let cat = PackCatalog::open(&path)?;
    let ctx = CudaContext::new(0)?;
    let (free, total) = ctx.mem_info()?;
    println!("M3TiS v{} Hardware & Pack Info", env!("CARGO_PKG_VERSION"));
    println!(
        "  GPU : {} (sm_{}{}) | VRAM {:.2} GiB free / {:.2} GiB total",
        ctx.name,
        ctx.sm_major,
        ctx.sm_minor,
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    println!(
        "  Pack: {:?} | {} layers x {} experts ({} total) | strides {:?} | {:.2} GiB",
        cat.version,
        cat.layers,
        cat.experts_per_layer,
        cat.total_experts(),
        cat.max_strides,
        cat.file_bytes as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

fn cmd_bench_io(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(args.first().ok_or("missing <pack.m3tis>")?);
    let qd: usize = parse_flag(args, "--qd", 24);
    let target_gb: f64 = parse_flag(args, "--gb", 12.0);
    let use_fixed = has_flag(args, "--fixed");
    let use_sqpoll = has_flag(args, "--sqpoll");

    let cat = PackCatalog::open(&path)?;
    let df = DirectFile::open(&path)?;
    let ctx = CudaContext::new(0)?;

    let comp_bytes = cat.max_strides[0]; // 3 MiB per trellis
    let buf_bytes = qd * comp_bytes;
    let mut pinned = ctx.alloc_pinned(buf_bytes)?;
    let mut ring = IoRing::new(64, use_sqpoll)?;
    if use_fixed {
        if let Err(e) = ring.register_buffers(&[(pinned.as_mut_ptr(), buf_bytes)]) {
            eprintln!("[bench-io] note: register_buffers unavailable ({e}); falling back to direct O_DIRECT read");
        }
    }

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let total_experts = cat.total_experts();
    let target_bytes = (target_gb * 1e9) as u64;
    let mut stats = IoStats::default();
    let mut ops = Vec::with_capacity(qd);
    let mut batches = 0u64;

    let t0 = Instant::now();
    while stats.bytes < target_bytes {
        ops.clear();
        for i in 0..qd {
            let key = (rng.next_u64() as usize) % total_experts;
            let comp = &cat.entries[key].components[i % 3];
            let dst = unsafe { pinned.as_mut_ptr().add(i * comp_bytes) };
            ops.push(ReadOp {
                fd: df.fd(),
                offset: comp.offset,
                dst,
                len: comp.aligned_len,
                user_data: i as u64,
                fixed_buf_idx: if use_fixed { 0 } else { -1 },
            });
        }
        ring.read_batch_sync(&ops, &mut stats)?;
        batches += 1;
    }
    let dt = t0.elapsed().as_secs_f64();
    let gbps = (stats.bytes as f64) / 1e9 / dt;
    println!(
        "[bench-io] O_DIRECT={} fixed_buf={} sqpoll={} qd={} | {:.2} GB in {:.2}s ({} batches) => {:.2} GB/s",
        df.direct,
        ring.registered_bufs,
        ring.sqpoll,
        qd,
        stats.bytes as f64 / 1e9,
        dt,
        batches,
        gbps
    );
    Ok(())
}

/// Build a Zipf CDF matching GLM-5.3's empirically measured distribution (top 30% = ~73% mass).
fn build_zipf_cdf(n: usize, alpha: f64) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(n);
    let mut sum = 0.0;
    for r in 1..=n {
        sum += 1.0 / (r as f64).powf(alpha);
        cdf.push(sum);
    }
    for v in &mut cdf {
        *v /= sum;
    }
    cdf
}

fn sample_zipf(cdf: &[f64], u: f64) -> usize {
    match cdf.binary_search_by(|x| x.partial_cmp(&u).unwrap()) {
        Ok(i) => i,
        Err(i) => i.min(cdf.len() - 1),
    }
}

fn cpu_file_xor_checksum(path: &PathBuf, entry: &m3tis_format::ExpertEntry) -> Result<u32, Box<dyn std::error::Error>> {
    let mut f = File::open(path)?;
    let mut acc = 0u32;
    let mut buf = vec![0u8; entry.components[0].len];
    for p in 0..3 {
        let c = entry.components[p];
        if buf.len() != c.len {
            buf.resize(c.len, 0);
        }
        f.seek(SeekFrom::Start(c.offset))?;
        f.read_exact(&mut buf)?;
        for chunk in buf.chunks_exact(4) {
            acc ^= u32::from_le_bytes(chunk.try_into().unwrap());
        }
    }
    Ok(acc)
}

fn cmd_bench_pool(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(args.first().ok_or("missing <pack.m3tis>")?);
    let ram_gib: f64 = parse_flag(args, "--ram-gib", 20.0);
    let vram_gib: f64 = parse_flag(args, "--vram-gib", 4.5);
    let rounds: usize = parse_flag(args, "--rounds", 20);
    let spec_p: usize = parse_flag(args, "--spec", 3);
    let gpu_ms: f64 = parse_flag(args, "--gpu-ms", 4.5);
    let policy_str: String = parse_flag(args, "--policy", "2q".to_string());
    let policy = match policy_str.as_str() {
        "clock" => EvictionPolicy::PlainClock,
        _ => EvictionPolicy::TwoQLfu,
    };

    let ctx = CudaContext::new(0)?;
    let compute_stream = ctx.stream()?;
    let kernels = RouterRemapKernels::load(&ctx)?;
    let mut pool = TieredExpertPool::open(&ctx, &path, vram_gib, ram_gib, policy, false)?;

    let layers = pool.catalog.layers;
    let e_per = pool.catalog.experts_per_layer;
    // alpha=0.92 matches GLM-5.3 measured top10%=43%, top30%=73%, top50%=88%
    let cdf = build_zipf_cdf(e_per, 0.92);
    let mut rng = Rng(0x517c_c1b7_2722_0a95);

    // Pre-generate deterministic Zipf DFlash2 verify batches (~20 unique experts/layer)
    // and 82%-accurate speculative next-layer lookahead lists.
    let mut round_keys: Vec<Vec<Vec<usize>>> = Vec::with_capacity(rounds);
    let mut round_specs: Vec<Vec<Vec<usize>>> = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let mut l_keys = Vec::with_capacity(layers);
        for li in 0..layers {
            let base = li * e_per;
            let mut ks = Vec::with_capacity(20);
            while ks.len() < 20 {
                let k = base + sample_zipf(&cdf, rng.next_f64());
                if !ks.contains(&k) {
                    ks.push(k);
                }
            }
            l_keys.push(ks);
        }
        let mut l_specs = Vec::with_capacity(layers);
        for li in 0..layers {
            if li + 1 >= layers {
                l_specs.push(Vec::new());
                continue;
            }
            let next_base = (li + 1) * e_per;
            let next_real = &l_keys[li + 1];
            let mut pred = Vec::with_capacity(20);
            for &rk in next_real {
                if rng.next_f64() < 0.82 {
                    pred.push(rk);
                } else {
                    pred.push(next_base + sample_zipf(&cdf, rng.next_f64()));
                }
            }
            l_specs.push(pred);
        }
        round_keys.push(l_keys);
        round_specs.push(l_specs);
    }

    // Phase 1: Warm-up rounds (first half)
    let half = rounds / 2;
    let mut demand_wall = Duration::ZERO;
    let mut total_wall = Duration::ZERO;

    let run_round = |pool: &mut TieredExpertPool,
                     r_idx: usize,
                     demand_wall: &mut Duration,
                     total_wall: &mut Duration|
     -> Result<(), Box<dyn std::error::Error>> {
        let rt0 = Instant::now();
        for li in 0..layers {
            let dt0 = Instant::now();
            pool.demand(&compute_stream, li * e_per, e_per, &round_keys[r_idx][li])?;
            *demand_wall += dt0.elapsed();
            pool.mark_compute_done(&compute_stream)?;
            if spec_p > 0 && li + 1 < layers {
                pool.start_spec_prefetch(&round_specs[r_idx][li], spec_p)?;
            }
            // Simulate GPU routed EXL3 + next-layer KDA/DSA mixer window overlapping prefetch_ring!
            if gpu_ms > 0.0 {
                let spin_until = Instant::now() + Duration::from_secs_f64(gpu_ms / 1000.0);
                while Instant::now() < spin_until {
                    std::hint::spin_loop();
                }
            }
        }
        *total_wall += rt0.elapsed();
        Ok(())
    };

    for r in 0..half {
        run_round(&mut pool, r, &mut demand_wall, &mut total_wall)?;
    }

    // Phase 2: Inject a disruptive 1-shot prefill scan (120 cold experts/layer = 5,040 experts)
    // right in the middle to stress-test 2Q/LFU scan shielding vs Plain CLOCK!
    for li in 0..layers {
        let base = li * e_per;
        let scan_keys: Vec<usize> = (160..280).map(|e| base + e).collect();
        pool.demand(&compute_stream, base, e_per, &scan_keys)?;
        pool.mark_compute_done(&compute_stream)?;
    }

    // Reset post-prefill decode stats so we measure the EXACT post-prefill recovery speed!
    let pre_hits = pool.stats.vram_hits + pool.stats.ram_hits;
    let pre_misses = pool.stats.disk_misses;
    let pre_disk = pool.stats.disk_bytes;
    let pre_spec_sub = pool.stats.spec_submitted;
    let pre_spec_hit = pool.stats.spec_hits;
    let pre_spec_early = pool.stats.spec_ready_early;
    demand_wall = Duration::ZERO;
    total_wall = Duration::ZERO;

    for r in half..rounds {
        run_round(&mut pool, r, &mut demand_wall, &mut total_wall)?;
    }

    // End-to-end RTX 3090 bit-exact GPU XOR checksum verification on the last layer's resident expert!
    let check_key = round_keys[rounds - 1][layers - 1][0];
    let gpu_xor = pool.vram_xor_checksum(&ctx, &compute_stream, &kernels, check_key)?;
    let cpu_xor = cpu_file_xor_checksum(&path, &pool.catalog.entries[check_key])?;
    assert_eq!(
        gpu_xor, cpu_xor,
        "GPU VRAM checksum mismatch for expert {check_key}: gpu={gpu_xor:#x} vs cpu={cpu_xor:#x}"
    );

    let post_hits = (pool.stats.vram_hits + pool.stats.ram_hits) - pre_hits;
    let post_misses = pool.stats.disk_misses - pre_misses;
    let post_tot = (post_hits + post_misses).max(1);
    let post_disk = pool.stats.disk_bytes - pre_disk;
    let eff_bw = (post_disk as f64) / 1e9 / demand_wall.as_secs_f64().max(1e-6);
    let sub = pool.stats.spec_submitted - pre_spec_sub;
    let sh = pool.stats.spec_hits - pre_spec_hit;
    let se = pool.stats.spec_ready_early - pre_spec_early;
    // Assume 3.0 tokens/verify-round (our measured DFlash2 average)
    let sim_tps = ((rounds - half) as f64 * 3.0) / total_wall.as_secs_f64();

    println!(
        "[bench-pool policy={:?} spec={} ram={:.1}G vram={:.1}G] post-prefill hit={:.1}% (am_pinned={}) | disk={:.2}GB in demand={:.2}s => eff_BW={:.2} GB/s | spec={}/{} ({:.1}% hit, {:.1}% early) | sim_decode={:.2} tok/s | GPU_XOR_CHECK=BIT-EXACT ({:#010x})",
        policy,
        spec_p,
        ram_gib,
        vram_gib,
        100.0 * post_hits as f64 / post_tot as f64,
        pool.ram_cache.protected_count(),
        post_disk as f64 / 1e9,
        demand_wall.as_secs_f64(),
        eff_bw,
        sh,
        sub,
        100.0 * sh as f64 / sub.max(1) as f64,
        100.0 * se as f64 / sub.max(1) as f64,
        sim_tps,
        gpu_xor
    );
    Ok(())
}

/// Milestone 2 hardware parity + latency gate on REAL GLM-5.3-Flash 3.0bpw routed experts:
/// Streams real Layer-3 experts through `TieredExpertPool` (`O_DIRECT` -> Pinned RAM -> VRAM),
/// loads their real fp16 `suh`/`svh` scales from safetensors, validates GPU Tensor Core outputs
/// against our clean-room CPU FP64 EXL3 oracle, and benchmarks `AHEAD` vs `PLAIN` latency across
/// decode (`rows=1`) and DFlash2 verify (`rows=4, 8, 16`) batch sizes!
fn cmd_bench_moe(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (ckpt_dir, pack_path) = resolve_ckpt_and_pack(args)?;
    let iters: usize = parse_flag(args, "--iters", 40);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.stream()?;
    let remap_k = RouterRemapKernels::load(&ctx)?;
    let moe_k = Exl3MoeKernels::load(&ctx)?;

    let t0 = Instant::now();
    let st = SafetensorsIndex::open_dir(&ckpt_dir)?;
    println!(
        "[bench-moe] indexed {} safetensors shards ({} tensors) in {:.2}s",
        st.shards.len(),
        st.tensors.len(),
        t0.elapsed().as_secs_f64()
    );

    // Small 1 GiB VRAM / 2 GiB RAM pool for fast kernel gate
    let mut pool = TieredExpertPool::open(&ctx, &pack_path, 1.0, 2.0, EvictionPolicy::TwoQLfu, false)?;
    let dims = 4096usize;
    let inter = 2048usize;
    let e_per = pool.catalog.experts_per_layer; // 288
    let vram_slots = pool.vram_slots();
    let ([d_gt_ptrs, d_ut_ptrs, d_dt_ptrs], _d_k2) = pool.build_exl3_slot_tables(&ctx, &stream, 6)?;

    // Allocate compact scale bank just for layer 0 (model layer 3, 288 experts)
    let mut scales = Exl3ScaleBank::alloc(&ctx, e_per, vram_slots, dims, inter)?;

    // Pick 8 real GLM-5.3 Layer-3 experts to stream from glm53_3bpw.m3tis and verify bit-for-bit!
    let test_experts: [usize; 8] = [0, 7, 19, 42, 88, 127, 201, 287];
    for &e in &test_experts {
        scales.load_expert_from_safetensors(&st, "model.language_model.", 3, e, e)?;
    }

    // Stream those 8 real experts from NVMe (.m3tis) -> Pinned RAM -> RTX 3090 VRAM
    pool.demand(&stream, 0, e_per, &test_experts)?;
    for &e in &test_experts {
        let vslot = pool.vram_cache.peek(e).ok_or("expert not in VRAM")?;
        scales.sync_slot_async(&pool.copy_stream, e, vslot)?;
    }
    pool.copy_stream.synchronize()?;
    stream.synchronize()?;

    // Build realistic BF16 input row (x ~ N(0, 0.25))
    let max_rows = 16usize;
    let slots = 9usize; // top-8 + 1 skipped shared slot
    let mut x_bf16 = vec![0u16; max_rows * dims];
    let mut x_f32 = vec![0.0f32; max_rows * dims];
    let mut rng = Rng(0x243f_6a88_85a3_08d3);
    for i in 0..max_rows * dims {
        let u = (rng.next_f64() * 2.0 - 1.0) as f32 * 0.35;
        let b16 = f32_to_bf16(u);
        x_bf16[i] = b16;
        x_f32[i] = f32::from_bits((b16 as u32) << 16);
    }
    let d_x = ctx.alloc_device(max_rows * dims * 2)?;
    unsafe {
        stream.htod_async(d_x.ptr(), x_bf16.as_ptr() as *const u8, max_rows * dims * 2)?;
    }

    // Logical picks [max_rows, 9] -> remap via our sm_86 FSR remap kernel -> mapped VRAM slots
    let mut log_picks = vec![e_per as i32; max_rows * slots];
    let mut wts_h = vec![0.0f32; max_rows * slots];
    let base_wts = [0.60f32, 0.45, 0.38, 0.31, 0.25, 0.20, 0.17, 0.14];
    for r in 0..max_rows {
        for k in 0..8 {
            log_picks[r * slots + k] = test_experts[(k + r) % 8] as i32;
            wts_h[r * slots + k] = base_wts[k];
        }
        log_picks[r * slots + 8] = e_per as i32; // shared skipped
        wts_h[r * slots + 8] = 0.0;
    }
    let d_log_picks = ctx.alloc_device(max_rows * slots * 4)?;
    let d_mapped = ctx.alloc_device(max_rows * slots * 4)?;
    let d_wts = ctx.alloc_device(max_rows * slots * 4)?;
    let d_out_ahead = ctx.alloc_device(max_rows * dims * 4)?;
    let d_out_plain = ctx.alloc_device(max_rows * dims * 4)?;
    unsafe {
        stream.htod_async(d_log_picks.ptr(), log_picks.as_ptr() as *const u8, max_rows * slots * 4)?;
        stream.htod_async(d_wts.ptr(), wts_h.as_ptr() as *const u8, max_rows * slots * 4)?;
    }
    remap_k.remap_i32(
        &stream,
        d_log_picks.ptr(),
        d_mapped.ptr(),
        pool.dev_map_ptr(),
        (max_rows * slots) as i32,
        e_per as i32,
        e_per as i32,
        vram_slots as i32,
    )?;

    let scratch = Exl3MoeScratch::alloc(&ctx, max_rows, slots, dims, inter, 4)?;
    stream.memset_d32_async(scratch.slot_y.ptr(), 0, max_rows * slots * dims)?;

    let view = Exl3ExpertsView {
        gt_ptrs: d_gt_ptrs.ptr(),
        ut_ptrs: d_ut_ptrs.ptr(),
        dt_ptrs: d_dt_ptrs.ptr(),
        k2_g: pool.vram_k2_ptr(0),
        k2_u: pool.vram_k2_ptr(1),
        k2_d: pool.vram_k2_ptr(2),
        suh_g: scales.vram_ptr(0),
        suh_u: scales.vram_ptr(1),
        svh_g: scales.vram_ptr(2),
        svh_u: scales.vram_ptr(3),
        suh_d: scales.vram_ptr(4),
        svh_d: scales.vram_ptr(5),
        experts: vram_slots,
        dims,
        inter,
        swiglu_limit: 10.0,
        act_mode: 0,
    };

    // 1. REAL GLM-5.3 PARITY CHECK (rows = 1, all 8 real 3.0bpw Layer-3 experts combined!)
    moe_k.run_moe(&stream, &view, &scratch, d_x.ptr(), dims, InputDType::Bf16, d_mapped.ptr(), d_wts.ptr(), d_out_ahead.ptr(), 1, slots, true)?;
    moe_k.run_moe(&stream, &view, &scratch, d_x.ptr(), dims, InputDType::Bf16, d_mapped.ptr(), d_wts.ptr(), d_out_plain.ptr(), 1, slots, false)?;
    let mut gpu_ahead = vec![0.0f32; dims];
    let mut gpu_plain = vec![0.0f32; dims];
    unsafe {
        stream.dtoh_async(gpu_ahead.as_mut_ptr() as *mut u8, d_out_ahead.ptr(), dims * 4)?;
        stream.dtoh_async(gpu_plain.as_mut_ptr() as *mut u8, d_out_plain.ptr(), dims * 4)?;
    }
    stream.synchronize()?;

    let mut bit_diffs = 0usize;
    for j in 0..dims {
        if gpu_ahead[j].to_bits() != gpu_plain[j].to_bits() {
            bit_diffs += 1;
        }
    }
    assert_eq!(bit_diffs, 0, "AHEAD vs PLAIN had {bit_diffs} bit differences!");

    // Compute clean-room CPU FP64 reference directly from glm53_3bpw.m3tis + safetensors scales
    let mut pf = File::open(&pack_path)?;
    let mut ref_row = vec![0.0f32; dims];
    for (k_idx, &e) in test_experts.iter().enumerate() {
        let entry = &pool.catalog.entries[e];
        let mut comp_bytes = [Vec::new(), Vec::new(), Vec::new()];
        for p in 0..3 {
            let c = entry.components[p];
            comp_bytes[p].resize(c.len, 0);
            pf.seek(SeekFrom::Start(c.offset))?;
            pf.read_exact(&mut comp_bytes[p])?;
        }
        let y = reference_exl3_expert_mlp(
            &x_f32[0..dims],
            &comp_bytes[0],
            &comp_bytes[1],
            &comp_bytes[2],
            (entry.components[0].bits as usize) * 2,
            (entry.components[1].bits as usize) * 2,
            (entry.components[2].bits as usize) * 2,
            &scales.ram_scale_u16(0, e),
            &scales.ram_scale_u16(1, e),
            &scales.ram_scale_u16(2, e),
            &scales.ram_scale_u16(3, e),
            &scales.ram_scale_u16(4, e),
            &scales.ram_scale_u16(5, e),
            dims,
            inter,
            10.0,
        );
        let w = base_wts[k_idx];
        for j in 0..dims {
            ref_row[j] += w * y[j];
        }
    }

    let mut max_abs = 0.0f32;
    let mut err_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut gpu_sq = 0.0f64;
    for j in 0..dims {
        let g = gpu_ahead[j];
        let rf = ref_row[j];
        let d = (g - rf) as f64;
        max_abs = max_abs.max(d.abs() as f32);
        err_sq += d * d;
        ref_sq += (rf as f64) * (rf as f64);
        gpu_sq += (g as f64) * (g as f64);
        dot += (g as f64) * (rf as f64);
    }
    let rel_l2 = (err_sq / ref_sq.max(1e-30)).sqrt();
    let cos = dot / (gpu_sq.sqrt() * ref_sq.sqrt()).max(1e-30);
    println!(
        "[REAL GLM-5.3 Layer-3 8-Expert Parity] AHEAD==PLAIN: BIT-EXACT (0/{dims} diffs) | vs CPU FP64: max_abs={max_abs:.6} rel_L2={rel_l2:.6} cos_sim={cos:.8}"
    );
    assert!(cos > 0.9995, "Real GLM-5.3 parity failed: cos={cos}");

    // 2. RTX 3090 KERNEL LATENCY BENCHMARK (AHEAD vs PLAIN across decode/verify batch sizes)
    println!("\n=== RTX 3090 sm_86 EXL3 MoE Kernel Latency (dims=4096, inter=2048, 8 experts, 3.0bpw) ===");
    for &rows in &[1usize, 4, 8, 16] {
        // Warmup
        for _ in 0..5 {
            moe_k.run_moe(&stream, &view, &scratch, d_x.ptr(), dims, InputDType::Bf16, d_mapped.ptr(), d_wts.ptr(), d_out_ahead.ptr(), rows, slots, true)?;
        }
        stream.synchronize()?;

        let t_a = Instant::now();
        for _ in 0..iters {
            moe_k.run_moe(&stream, &view, &scratch, d_x.ptr(), dims, InputDType::Bf16, d_mapped.ptr(), d_wts.ptr(), d_out_ahead.ptr(), rows, slots, true)?;
        }
        stream.synchronize()?;
        let us_ahead = t_a.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t_p = Instant::now();
        for _ in 0..iters {
            moe_k.run_moe(&stream, &view, &scratch, d_x.ptr(), dims, InputDType::Bf16, d_mapped.ptr(), d_wts.ptr(), d_out_plain.ptr(), rows, slots, false)?;
        }
        stream.synchronize()?;
        let us_plain = t_p.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // 8 experts * 3 mats * 2 * rows * 4096 * 2048 FLOPs
        let flops = 8.0 * 3.0 * 2.0 * (rows as f64) * (dims as f64) * (inter as f64);
        let tflops_a = flops / (us_ahead * 1e-6) / 1e12;
        let bytes_8exp = 8.0 * 9.0 * 1024.0 * 1024.0; // 72 MiB of 3.0bpw trellises
        let eff_bw_a = bytes_8exp / (us_ahead * 1e-6) / 1e9;
        println!(
            "  rows={rows:2} | AHEAD={us_ahead:7.1} us ({eff_bw_a:6.1} GB/s, {tflops_a:5.2} TFLOP/s) | PLAIN={us_plain:7.1} us | speedup={:.2}x",
            us_plain / us_ahead
        );
    }
    Ok(())
}

/// Milestone 3A/3B end-to-end GLM-5.3-Flash 320B-A18B forward & decode verification on RTX 3090.
fn cmd_forward_check(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (ckpt_dir, pack_path) = resolve_ckpt_and_pack(args)?;
    let tokens_csv: String = parse_flag(args, "--tokens", String::new());
    let prompt_file: String = parse_flag(args, "--prompt-file", String::new());
    let prompt_text: String = if !prompt_file.is_empty() {
        std::fs::read_to_string(&prompt_file)?
    } else {
        parse_flag(args, "--prompt", String::new())
    };
    let thinking = has_flag(args, "--thinking");
    let tokenizer = if !prompt_text.is_empty() || has_flag(args, "--decode-text") {
        Some(BpeTokenizer::from_path(&ckpt_dir)?)
    } else {
        None
    };
    let prompt_tokens: Vec<i32> = if !tokens_csv.is_empty() {
        tokens_csv
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<i32>())
            .collect::<Result<_, _>>()?
    } else if !prompt_text.is_empty() {
        let tok = tokenizer.as_ref().unwrap();
        let formatted = tok.format_glm53_chat(
            &[ChatMessage {
                role: "user".to_string(),
                content: prompt_text.clone(),
            }],
            thinking,
            None,
        );
        tok.encode(&formatted)
    } else {
        return Err("missing --tokens <csv>, --prompt <text>, or --prompt-file <path>".into());
    };
    let max_tokens: usize = parse_flag(args, "--max-tokens", 32);
    let max_ctx: usize = parse_flag(args, "--max-ctx", 2048);
    let prefill_batch: usize = parse_flag(args, "--prefill-batch", 2048);
    let chunk_rows: usize = parse_flag(
        args,
        "--chunk-rows",
        if prefill_batch > 0 { 256.min(max_ctx) } else if max_ctx > 2051 { 256 } else { 16 },
    );
    let save_zipf: String = parse_flag(args, "--save-zipf", String::new());
    let warm_runs: usize = parse_flag(args, "--warm-runs", 1).max(1);
    let ram_gib: f64 = parse_flag(args, "--ram-gib", 10.0);
    let vram_gib_opt: f64 = parse_flag(args, "--vram-gib", -1.0);
    let tuning = parse_decode_tuning(args)?;
    let (fwd_opts, max_drafts, draft_conf) = (tuning.fwd, tuning.max_drafts, tuning.draft_conf);
    let policy_str: String = parse_flag(args, "--policy", "2q".to_string());
    let ram_policy = match policy_str.as_str() {
        "clock" => EvictionPolicy::PlainClock,
        _ => EvictionPolicy::TwoQLfu,
    };
    let drafter_str: String = parse_flag(args, "--drafter", String::new());
    let drafter_dir = if drafter_str.is_empty() {
        None
    } else {
        Some(PathBuf::from(drafter_str))
    };
    let pfx_dir_str: String = parse_flag(args, "--prefix-cache-dir", String::new());
    let prefix_cache_dir = if pfx_dir_str.is_empty() {
        None
    } else {
        Some(PathBuf::from(pfx_dir_str))
    };
    let prefix_cache_max_gib: f64 = parse_flag(args, "--prefix-cache-max-gib", 8.0);

    let bf16_dense = has_flag(args, "--bf16-dense");
    let kv_int8 = if has_flag(args, "--no-kv-int8") {
        false
    } else if has_flag(args, "--kv-int8") {
        true
    } else {
        max_ctx >= 65536
    };
    let load_opts = Glm53LoadOptions {
        pack_dense_q4: !bf16_dense,
        pack_dense_q3: tuning.dense_q3 && !bf16_dense,
        pack_dense_q3_moe: tuning.dense_q3_moe && !bf16_dense,
        pack_head_q4: !has_flag(args, "--bf16-head"),
        pack_drafter_q4: !has_flag(args, "--bf16-drafter"),
        pack_drafter_q3: tuning.drafter_q3 && !has_flag(args, "--bf16-drafter"),
        vram_gib: if vram_gib_opt > 0.0 { Some(vram_gib_opt) } else { None },
        ram_gib,
        max_ctx,
        kv_int8,
        max_chunk_rows: chunk_rows,
        prefill_batch,
        ram_policy,
        sqpoll: has_flag(args, "--sqpoll"),
        drafter_dir,
        prefix_cache_dir: prefix_cache_dir.clone(),
        prefix_cache_max_gib,
        prewarm_vram: tuning.prewarm_vram,
        prewarm_ram: tuning.prewarm_ram,
        ..Glm53LoadOptions::default()
    };

    let ctx = CudaContext::new(0)?;
    let (free0, tot0) = ctx.mem_info()?;
    println!(
        "[m3tis forward-check] GPU={} | start VRAM {:.2}/{:.2} GiB free | dense={} q4_head={} drafter={}{} adapt={} cpu_hyb={}/{} prewarm={}/{} ram={:.1}G spec={} la={} dyn={:.2}/{} rbias={:.3}/{} nvme_strict={} nvme_mass={:.2} tail_row={}",
        ctx.name,
        free0 as f64 / (1u64 << 30) as f64,
        tot0 as f64 / (1u64 << 30) as f64,
        if load_opts.pack_dense_q3 { "q3" } else if load_opts.pack_dense_q3_moe { "q3-moe" } else if load_opts.pack_dense_q4 { "q4" } else { "bf16" },
        load_opts.pack_head_q4,
        load_opts.drafter_dir.is_some(),
        if load_opts.pack_drafter_q3 { "(q3)" } else { "" },
        fwd_opts.adaptive_draft,
        fwd_opts.cpu_hybrid_max,
        fwd_opts.cpu_hybrid_always,
        load_opts.prewarm_vram,
        load_opts.prewarm_ram,
        load_opts.ram_gib,
        fwd_opts.spec_prefetch_p,
        fwd_opts.lookahead_depth,
        fwd_opts.dyn_topk_mass,
        fwd_opts.dyn_topk_min_k,
        fwd_opts.residency_bias,
        fwd_opts.residency_strict_k,
        fwd_opts.nvme_strict_k,
        fwd_opts.nvme_mass,
        fwd_opts.nvme_tail_row
    );

    let (mut model, mut state) = Glm53Model::load(&ctx, &ckpt_dir, &pack_path, &load_opts)?;
    let ls = &model.load_summary;
    let (free1, _) = ctx.mem_info()?;
    println!(
        "[load] {:.2}s (prewarm={}+{} in {:.2}s) | dense={:.1} MiB | drafter={:.1} MiB | state+kv={:.1} MiB | experts={:.1} MiB ({} VRAM [{}+{}], {} RAM [{}+{}], O_DIRECT={}, fixed_io={}) | VRAM left={:.2} GiB",
        ls.load_secs,
        ls.prewarm_loaded,
        ls.prewarm_ram_loaded,
        ls.prewarm_secs,
        ls.dense_vram_mib,
        ls.drafter_vram_mib,
        ls.state_vram_mib,
        ls.expert_vram_mib,
        ls.vram_slots,
        ls.vram_hot_slots,
        ls.vram_cold_slots,
        ls.ram_slots,
        ls.ram_hot_slots,
        ls.ram_cold_slots,
        ls.direct_io,
        ls.fixed_io,
        free1 as f64 / (1u64 << 30) as f64
    );

    for run_idx in 0..warm_runs {
        if run_idx > 0 {
            model.reset_sequence(&mut state)?;
        }
        let tag = if run_idx == 0 { "cold" } else { "warm" };
        // 1. Prefill (using Persistent Disk Prefix Cache when `--prefix-cache-dir` is enabled!)
        let t_pf = Instant::now();
        let (first_logits, cached_toks, from_disk) = if prefix_cache_dir.is_some() {
            model.prefill_with_prefix_cache(&mut state, &prompt_tokens, fwd_opts)?
        } else {
            (model.forward(&mut state, &prompt_tokens, fwd_opts)?, 0, false)
        };
        let pf_dt = t_pf.elapsed().as_secs_f64();

        let mut top5: Vec<(usize, f32)> = first_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        top5.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        top5.truncate(5);
        let cache_note = if cached_toks > 0 {
            format!(" [cached={cached_toks} disk={from_disk}]")
        } else {
            String::new()
        };
        println!(
            "[prefill #{run_idx} {tag}] {} tokens in {:.2}s ({:.2} tok/s){cache_note} | first top5: {:?}",
            prompt_tokens.len(),
            pf_dt,
            prompt_tokens.len() as f64 / pf_dt.max(1e-6),
            top5
        );

        // 2. Greedy decode (DFlash2 speculative, OR Zero-NVMe Resident Self-Spec, OR serial)
        let first_tok = top5[0].0 as i32;
        model.reset_decode_prof();
        let pre_stats = model.pool_stats();
        let t_dec = Instant::now();
        let (generated, spec_stats) = if model.drafter.is_some() {
            model.generate_greedy_dflash2(&mut state, first_tok, max_tokens, max_drafts, draft_conf, fwd_opts)?
        } else if fwd_opts.self_spec_k > 0 {
            model.generate_greedy_self_spec(&mut state, first_tok, max_tokens, fwd_opts.self_spec_k, fwd_opts)?
        } else {
            let mut out_toks = Vec::with_capacity(max_tokens);
            let mut next_tok = first_tok;
            out_toks.push(next_tok);
            while out_toks.len() < max_tokens && !model.cfg.eos_tokens.contains(&next_tok) {
                let logits = model.forward(&mut state, &[next_tok], fwd_opts)?;
                let (best_id, _) = logits
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv { (i, v) } else { (bi, bv) }
                    });
                next_tok = best_id as i32;
                out_toks.push(next_tok);
            }
            (out_toks, Default::default())
        };
        let dec_dt = t_dec.elapsed().as_secs_f64();
        let dec_steps = generated.len().saturating_sub(1);
        let post_stats = model.pool_stats();

        let v_hits = post_stats.vram_hits - pre_stats.vram_hits;
        let v_miss = post_stats.vram_misses - pre_stats.vram_misses;
        let r_hits = post_stats.ram_hits - pre_stats.ram_hits;
        let c_hits = post_stats.cpu_hits - pre_stats.cpu_hits;
        let d_miss = post_stats.disk_misses - pre_stats.disk_misses;
        let tot_exp = (v_hits + v_miss + c_hits).max(1);
        let s_sub = post_stats.spec_submitted - pre_stats.spec_submitted;
        let s_hit = post_stats.spec_hits - pre_stats.spec_hits;
        let s_early = post_stats.spec_ready_early - pre_stats.spec_ready_early;
        let n_skip = post_stats.nvme_tail_skipped - pre_stats.nvme_tail_skipped;
        let d_gb = (post_stats.disk_bytes - pre_stats.disk_bytes) as f64 / 1e9;

        if spec_stats.rounds > 0 {
            let avg_tok_round = dec_steps as f64 / spec_stats.rounds as f64;
            let acc_pct = 100.0 * spec_stats.total_accepted as f64 / spec_stats.total_drafted.max(1) as f64;
            let tag_spec = if model.drafter.is_some() { "DFlash2" } else { "SelfSpec" };
            let tree_note = if spec_stats.tree_b_wins > 0 {
                format!(" treeB={}wins/+{}tok", spec_stats.tree_b_wins, spec_stats.tree_b_accepted)
            } else {
                String::new()
            };
            println!(
                "[decode #{run_idx} {tag}] {} tokens ({} steps, {} rounds, {:.2} tok/rd, accept={}/{}={:.1}%{tree_note}) in {:.2}s => {:.2} tok/s ({tag_spec} draft={:.2}s verify={:.2}s commit={:.2}s)",
                generated.len(),
                dec_steps,
                spec_stats.rounds,
                avg_tok_round,
                spec_stats.total_accepted,
                spec_stats.total_drafted,
                acc_pct,
                dec_dt,
                dec_steps as f64 / dec_dt.max(1e-6),
                spec_stats.draft_secs,
                spec_stats.verify_secs,
                spec_stats.commit_secs
            );
        } else {
            println!(
                "[decode #{run_idx} {tag}] {} tokens ({} steps) in {:.2}s => {:.2} tok/s (serial, no drafter)",
                generated.len(),
                dec_steps,
                dec_dt,
                dec_steps as f64 / dec_dt.max(1e-6)
            );
        }
        println!(
            "[3-tier decode stats] vram_hit={:.1}% ({}/{}) | cpu_hit={} | ram_hit={} | disk_miss={} ({:.2} GB) | tail_skip={} | spec={}/{} ({:.1}% hit, {:.1}% early) | vram_am={} ram_am={}",
            100.0 * v_hits as f64 / tot_exp as f64,
            v_hits,
            tot_exp,
            c_hits,
            r_hits,
            d_miss,
            d_gb,
            n_skip,
            s_hit,
            s_sub,
            100.0 * s_hit as f64 / s_sub.max(1) as f64,
            100.0 * s_early as f64 / s_sub.max(1) as f64,
            model.pool.vram_cache.protected_count(),
            model.pool.ram_cache.protected_count()
        );
        if model.decode_prof.calls > 0 {
            let p = model.decode_prof;
            let c = p.calls as f64;
            println!(
                "[decode profile] calls={} avg_rows={:.2} | per-call ms: attn={:.2} dense_ffn={:.2} picks_sync={:.2} demand={:.2} exl3={:.2} spec={:.2} head={:.2} | cold_r0: row0={} tail={} (gated={})",
                p.calls,
                p.total_rows as f64 / c,
                p.attn_secs * 1e3 / c,
                p.dense_ffn_secs * 1e3 / c,
                p.picks_sync_secs * 1e3 / c,
                p.demand_secs * 1e3 / c,
                p.exl3_secs * 1e3 / c,
                p.spec_secs * 1e3 / c,
                p.head_secs * 1e3 / c,
                p.row0_cold_rank0,
                p.tail_cold_rank0,
                p.tail_cold_skipped,
            );
        }
        if let Some(tok) = tokenizer.as_ref() {
            println!("TEXT:\n{}", tok.decode(&generated, true));
        }
        let ids_csv: Vec<String> = generated.iter().map(|t| t.to_string()).collect();
        println!("TOKENS_CSV={}", ids_csv.join(","));
    }
    if !save_zipf.is_empty() {
        let mut bytes = Vec::with_capacity(model.pool.expert_hits.len() * 8);
        for &c in &model.pool.expert_hits {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        std::fs::write(&save_zipf, &bytes)?;
        let active = model.pool.expert_hits.iter().filter(|&&c| c > 0).count();
        println!(
            "[zipf] saved {} expert routing counts ({active} active) to {save_zipf}",
            model.pool.expert_hits.len()
        );
    }
    Ok(())
}

/// Milestone 3C `.m3tis` v2 Mixed-Bitwidth (`3b` hot / `2b` cold) Zipf Packer.
fn cmd_pack_v2(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let hot_pack = PathBuf::from(args.first().ok_or("missing <hot_3b.m3tis>")?);
    let out_path = PathBuf::from(args.get(1).ok_or("missing <out_v2.m3tis>")?);
    let cold_str: String = parse_flag(args, "--cold-pack", String::new());
    let zipf_str: String = parse_flag(args, "--zipf", String::new());
    let hot_frac: f64 = parse_flag(args, "--hot-frac", 0.30);
    let requant_cold_2b = has_flag(args, "--requant-cold-2b");
    pack_v2::run_pack_v2(&pack_v2::PackV2Opts {
        hot_pack,
        cold_pack: if cold_str.is_empty() { None } else { Some(PathBuf::from(cold_str)) },
        out_path,
        zipf_path: if zipf_str.is_empty() { None } else { Some(PathBuf::from(zipf_str)) },
        hot_frac,
        requant_cold_2b,
    })
}

/// OpenAI-compatible HTTP/1.1 + SSE server (`m3tis serve`).
fn cmd_serve(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (ckpt_dir, pack_path) = resolve_ckpt_and_pack(args)?;
    let host: String = parse_flag(args, "--host", "127.0.0.1".to_string());
    let port: u16 = parse_flag(args, "--port", 8080);
    let model_name: String = parse_flag(args, "--model-name", "glm-5.3-flash".to_string());
    let ram_gib: f64 = parse_flag(args, "--ram-gib", 10.0);
    let vram_gib_opt: f64 = parse_flag(args, "--vram-gib", -1.0);
    let max_ctx: usize = parse_flag(args, "--max-ctx", 2048);
    let prefill_batch: usize = parse_flag(args, "--prefill-batch", 2048);
    let chunk_rows: usize = parse_flag(
        args,
        "--chunk-rows",
        if prefill_batch > 0 { 256.min(max_ctx) } else if max_ctx > 2051 { 256 } else { 16 },
    );
    let prefix_cache_slots: usize = parse_flag(args, "--prefix-cache-slots", 4);
    let pfx_dir_str: String = parse_flag(args, "--prefix-cache-dir", String::new());
    let prefix_cache_dir = if pfx_dir_str.is_empty() {
        None
    } else {
        Some(PathBuf::from(pfx_dir_str))
    };
    let prefix_cache_max_gib: f64 = parse_flag(args, "--prefix-cache-max-gib", 8.0);
    let tuning = parse_decode_tuning(args)?;
    let (fwd_opts, max_drafts, draft_conf) = (tuning.fwd, tuning.max_drafts, tuning.draft_conf);
    let default_thinking = has_flag(args, "--thinking");
    let drafter_str: String = parse_flag(args, "--drafter", String::new());
    let drafter_dir = if drafter_str.is_empty() {
        None
    } else {
        Some(PathBuf::from(drafter_str))
    };

    let t_tok = Instant::now();
    let tokenizer = BpeTokenizer::from_path(&ckpt_dir)?;
    println!(
        "[m3tis serve] loaded zero-dep BPE tokenizer ({} tokens) in {:.2}s",
        tokenizer.vocab_size(),
        t_tok.elapsed().as_secs_f64()
    );

    let bf16_dense = has_flag(args, "--bf16-dense");
    let kv_int8 = if has_flag(args, "--no-kv-int8") {
        false
    } else if has_flag(args, "--kv-int8") {
        true
    } else {
        max_ctx >= 65536
    };
    let load_opts = Glm53LoadOptions {
        pack_dense_q4: !bf16_dense,
        pack_dense_q3: tuning.dense_q3 && !bf16_dense,
        pack_dense_q3_moe: tuning.dense_q3_moe && !bf16_dense,
        pack_head_q4: !has_flag(args, "--bf16-head"),
        pack_drafter_q4: !has_flag(args, "--bf16-drafter"),
        pack_drafter_q3: tuning.drafter_q3 && !has_flag(args, "--bf16-drafter"),
        vram_gib: if vram_gib_opt > 0.0 { Some(vram_gib_opt) } else { None },
        ram_gib,
        max_ctx,
        kv_int8,
        max_chunk_rows: chunk_rows,
        prefill_batch,
        prefix_cache_slots,
        prefix_cache_dir,
        prefix_cache_max_gib,
        ram_policy: EvictionPolicy::TwoQLfu,
        sqpoll: has_flag(args, "--sqpoll"),
        drafter_dir,
        prewarm_vram: tuning.prewarm_vram,
        prewarm_ram: tuning.prewarm_ram,
        ..Glm53LoadOptions::default()
    };
    let ctx = CudaContext::new(0)?;
    let (mut model, mut state) = Glm53Model::load(&ctx, &ckpt_dir, &pack_path, &load_opts)?;
    let ls = &model.load_summary;
    println!(
        "[m3tis serve] loaded GLM-5.3-Flash in {:.2}s (prewarm={}+{} in {:.2}s) | dense={:.1} MiB ({}) | drafter={:.1} MiB | experts={:.1} MiB ({} VRAM [{}+{}] / {} RAM [{}+{}] slots, O_DIRECT={}) | spec={} dyn={:.2}/{} rbias={:.3}/{} nvme_strict={} nvme_mass={:.2} zero_stall={:.2} self_spec={}/{:.2}",
        ls.load_secs,
        ls.prewarm_loaded,
        ls.prewarm_ram_loaded,
        ls.prewarm_secs,
        ls.dense_vram_mib,
        if load_opts.pack_dense_q3 { "q3" } else if load_opts.pack_dense_q4 { "q4" } else { "bf16" },
        ls.drafter_vram_mib,
        ls.expert_vram_mib,
        ls.vram_slots,
        ls.vram_hot_slots,
        ls.vram_cold_slots,
        ls.ram_slots,
        ls.ram_hot_slots,
        ls.ram_cold_slots,
        ls.direct_io,
        fwd_opts.spec_prefetch_p,
        fwd_opts.dyn_topk_mass,
        fwd_opts.dyn_topk_min_k,
        fwd_opts.residency_bias,
        fwd_opts.residency_strict_k,
        fwd_opts.nvme_strict_k,
        fwd_opts.nvme_mass,
        fwd_opts.zero_stall_mass,
        fwd_opts.self_spec_k,
        fwd_opts.self_spec_conf
    );

    let srv_cfg = server::ServerConfig {
        host,
        port,
        model_name,
        fwd_opts,
        max_drafts,
        draft_conf,
        default_thinking,
    };
    server::run_server(&mut model, &mut state, &tokenizer, &srv_cfg)
}
