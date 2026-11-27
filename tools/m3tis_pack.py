#!/usr/bin/env python3
"""
Pack GLM-5.3-Flash EXL3 routed-expert trellises into a single 4K-aligned .m3tis file
for zero-copy O_DIRECT io_uring DMA directly into CUDA pinned Host RAM.

Format (M3TISPK1):
  Bytes 0..4096: Header (little-endian)
    [0..8]   magic = b"M3TISPK1"
    [8..12]  layers (u32, e.g. 42)
    [12..16] experts_per_layer (u32, e.g. 288)
    [16..24] stride0 (u64, 4K-aligned gate trellis bytes)
    [24..32] stride1 (u64, 4K-aligned up trellis bytes)
    [32..40] stride2 (u64, 4K-aligned down trellis bytes)
    [40..48] len0 (u64, logical gate trellis bytes)
    [48..56] len1 (u64, logical up trellis bytes)
    [56..64] len2 (u64, logical down trellis bytes)
  Bytes 4096..: Contiguous per-expert slabs [layer * experts_per_layer + e]:
    gate_proj.trellis (padded to stride0)
    up_proj.trellis   (padded to stride1)
    down_proj.trellis (padded to stride2)
"""
import json, os, struct, sys, time
from pathlib import Path

def align4k(n: int) -> int:
    return (n + 4095) & ~4095

def read_st_header(path: Path):
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hdr_len))
    return meta, 8 + hdr_len

def main():
    if len(sys.argv) < 3:
        print("usage: m3tis_pack.py <checkpoint_dir> <output.m3tis>", file=sys.stderr)
        sys.exit(2)
    ckpt = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    cfg = json.loads((ckpt / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    n_layers = int(text_cfg["num_hidden_layers"])
    first_dense = int(text_cfg.get("first_k_dense_replace", 3))
    n_experts = int(text_cfg.get("n_routed_experts", text_cfg.get("num_experts", 288)))
    moe_layers = [i for i in range(first_dense, n_layers)]
    print(f"Checkpoint: {len(moe_layers)} MoE layers ({first_dense}..{n_layers-1}) x {n_experts} experts = {len(moe_layers)*n_experts} total")

    shards = sorted(ckpt.glob("*.safetensors"))
    tensor_loc = {}
    for sp in shards:
        meta, data_off = read_st_header(sp)
        for name, info in meta.items():
            if name == "__metadata__":
                continue
            if name.endswith(".trellis") and ".mlp.experts." in name:
                b, e = info["data_offsets"]
                tensor_loc[name] = (sp, data_off + b, e - b)

    prefix = "model.language_model."
    projs = ("gate_proj", "up_proj", "down_proj")
    lens = [0, 0, 0]
    for li in moe_layers:
        for e in range(n_experts):
            for pi, proj in enumerate(projs):
                name = f"{prefix}layers.{li}.mlp.experts.{e}.{proj}.trellis"
                if name not in tensor_loc:
                    raise SystemExit(f"Missing tensor: {name}")
                _, _, nbytes = tensor_loc[name]
                if lens[pi] == 0:
                    lens[pi] = nbytes
                elif lens[pi] != nbytes:
                    raise SystemExit(f"Non-uniform trellis length for {name}: {nbytes} vs {lens[pi]}")

    strides = [align4k(x) for x in lens]
    per_exp = sum(strides)
    total_exp = len(moe_layers) * n_experts
    total_bytes = 4096 + total_exp * per_exp
    print(f"Trellis lengths: {lens} | 4K strides: {strides} | total pack size: {total_bytes / (1<<30):.2f} GiB")

    # Pre-create empty file and attempt chattr +C (nodatacow) on btrfs before writing.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    out_path.touch()
    os.system(f"chattr +C '{out_path}' 2>/dev/null || true")

    t0 = time.monotonic()
    fds = {}
    try:
        hdr = bytearray(4096)
        struct.pack_into(
            "<8sIIQQQQQQ",
            hdr,
            0,
            b"M3TISPK1",
            len(moe_layers),
            n_experts,
            strides[0],
            strides[1],
            strides[2],
            lens[0],
            lens[1],
            lens[2],
        )
        pads = [bytes(strides[i] - lens[i]) for i in range(3)]
        written = 0
        with open(out_path, "r+b", buffering=0) as out_f:
            out_f.write(hdr)
            written += 4096
            for li_idx, li in enumerate(moe_layers):
                for e in range(n_experts):
                    for pi, proj in enumerate(projs):
                        name = f"{prefix}layers.{li}.mlp.experts.{e}.{proj}.trellis"
                        sp, off, nbytes = tensor_loc[name]
                        sf = fds.get(sp)
                        if sf is None:
                            sf = open(sp, "rb")
                            fds[sp] = sf
                        sf.seek(off)
                        buf = sf.read(nbytes)
                        if len(buf) != nbytes:
                            raise SystemExit(f"Short read on {name}")
                        out_f.write(buf)
                        if pads[pi]:
                            out_f.write(pads[pi])
                        written += strides[pi]
                if (li_idx + 1) % 6 == 0 or li_idx + 1 == len(moe_layers):
                    dt = time.monotonic() - t0
                    gb = written / 1e9
                    print(f"  layer {li_idx+1}/{len(moe_layers)} | {gb:.1f} GB written in {dt:.1f}s ({gb/max(dt,1e-6):.2f} GB/s)", flush=True)
    finally:
        for sf in fds.values():
            sf.close()
    dt = time.monotonic() - t0
    print(f"DONE: {out_path} ({written/(1<<30):.2f} GiB in {dt:.1f}s, {written/1e9/max(dt,1e-6):.2f} GB/s)")

if __name__ == "__main__":
    main()
