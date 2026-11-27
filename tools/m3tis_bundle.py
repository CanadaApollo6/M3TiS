#!/usr/bin/env python3
"""
Build a self-contained M3TiS checkpoint bundle from an EXL3 safetensors checkpoint
and a packed `.m3tis` routed-expert container.

Strips redundant routed expert `.trellis` weights (already stored in the `.m3tis` file),
unused MTP layer (`layers.45.*`), and vision weights (`model.visual.*`), keeping only:
  - Non-routed language model weights for layers 0..44 (`embed_tokens`, `norm`, `lm_head`,
    hyper-connections, KDA/MLA attention + DSA indexers, dense FFNs, routers, shared experts)
  - EXL3 Hadamard scale vectors (`suh` / `svh`) for all 12,096 routed experts
  - Tokenizer, chat template, generation config, updated `config.json`, `LICENSE`, and `.m3tis` pack

Result for GLM-5.3-Flash + v2 mixed 3b/2b pack: ~98.5 GiB total (~17.0 GiB shards + ~81.5 GiB pack)
instead of ~223 GiB when keeping both full safetensors and the pack separately.
"""
import argparse
import json
import os
import shutil
import struct
import time
from pathlib import Path


def read_st_header(path: Path):
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hdr_len))
    return meta, 8 + hdr_len


def should_keep_tensor(name: str, num_layers: int) -> bool:
    if name in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ):
        return True
    prefix = "model.language_model.layers."
    if not name.startswith(prefix):
        return False
    rest = name[len(prefix):]
    dot = rest.find(".")
    if dot <= 0:
        return False
    try:
        layer_idx = int(rest[:dot])
    except ValueError:
        return False
    if layer_idx >= num_layers:
        return False
    if ".mlp.experts." in name:
        return name.endswith(".suh") or name.endswith(".svh")
    return True


def write_shard(out_path: Path, items: list, src_fds: dict):
    header = {"__metadata__": {"format": "pt", "bundled_by": "m3tis_bundle.py"}}
    offset = 0
    for name, dtype, shape, _, _, nbytes in items:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_bytes += b" " * pad

    buf_size = 16 * 1024 * 1024
    with open(out_path, "wb") as out_f:
        out_f.write(struct.pack("<Q", len(hdr_bytes)))
        out_f.write(hdr_bytes)
        for _, _, _, sp, src_off, nbytes in items:
            sf = src_fds.get(sp)
            if sf is None:
                sf = open(sp, "rb")
                src_fds[sp] = sf
            sf.seek(src_off)
            rem = nbytes
            while rem > 0:
                chunk = sf.read(min(rem, buf_size))
                if not chunk:
                    raise SystemExit(f"Unexpected EOF reading {sp}")
                out_f.write(chunk)
                rem -= len(chunk)


def main():
    ap = argparse.ArgumentParser(description="Build a self-contained M3TiS checkpoint bundle.")
    ap.add_argument("checkpoint_dir", type=Path, help="Source EXL3 safetensors checkpoint directory")
    ap.add_argument("pack_file", type=Path, help="Packed .m3tis routed-expert container")
    ap.add_argument("output_dir", type=Path, help="Destination bundle directory")
    ap.add_argument("--max-shard-gib", type=float, default=4.5, help="Max safetensors shard size in GiB")
    ap.add_argument("--copy-pack", action="store_true", help="Copy .m3tis instead of hardlinking when on same filesystem")
    args = ap.parse_args()

    ckpt = args.checkpoint_dir.resolve()
    pack = args.pack_file.resolve()
    out_dir = args.output_dir.resolve()

    if not ckpt.is_dir():
        raise SystemExit(f"Missing checkpoint_dir: {ckpt}")
    if not pack.is_file():
        raise SystemExit(f"Missing pack_file: {pack}")

    cfg = json.loads((ckpt / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    num_layers = int(text_cfg.get("num_hidden_layers", 45))

    kept = []
    skipped_bytes = 0
    skipped_count = 0
    for sp in sorted(ckpt.glob("*.safetensors")):
        meta, data_off = read_st_header(sp)
        for name, info in meta.items():
            if name == "__metadata__":
                continue
            b, e = info["data_offsets"]
            nbytes = e - b
            if should_keep_tensor(name, num_layers):
                kept.append((name, info["dtype"], info["shape"], sp, data_off + b, nbytes))
            else:
                skipped_bytes += nbytes
                skipped_count += 1

    kept.sort(key=lambda x: x[0])
    total_kept_bytes = sum(x[5] for x in kept)
    print(
        f"Keeping {len(kept)} tensors ({total_kept_bytes / (1 << 30):.2f} GiB) | "
        f"Skipping {skipped_count} tensors ({skipped_bytes / (1 << 30):.2f} GiB)"
    )

    max_shard_bytes = int(args.max_shard_gib * (1 << 30))
    shards = []
    cur = []
    cur_bytes = 0
    for item in kept:
        if cur and cur_bytes + item[5] > max_shard_bytes:
            shards.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(item)
        cur_bytes += item[5]
    if cur:
        shards.append(cur)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    src_fds = {}
    weight_map = {}
    try:
        n_shards = len(shards)
        for idx, items in enumerate(shards, 1):
            fname = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
            out_path = out_dir / fname
            s_bytes = sum(x[5] for x in items)
            print(f"Writing {fname} ({len(items)} tensors, {s_bytes / (1 << 30):.2f} GiB)...", flush=True)
            write_shard(out_path, items, src_fds)
            for item in items:
                weight_map[item[0]] = fname
    finally:
        for sf in src_fds.values():
            sf.close()

    index_obj = {
        "metadata": {
            "total_size": total_kept_bytes,
            "m3tis_pack_file": pack.name,
            "m3tis_pack_bytes": pack.stat().st_size,
        },
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_obj, indent=2) + "\n"
    )

    # Write clean config.json with updated M3TiS quantization metadata
    qcfg = cfg.get("quantization_config", {})
    qcfg.pop("grafted_by", None)
    qcfg.update(
        {
            "quant_method": "exl3",
            "format": "m3tis-v2",
            "pack_file": pack.name,
            "hot_bits": 3,
            "cold_bits": 2,
            "codebook": "mcg",
            "scope": "glm53_routed_experts_only",
            "non_routed_dtype_policy": "source_native_bf16",
        }
    )
    cfg["quantization_config"] = qcfg
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "LICENSE",
    ):
        src_f = ckpt / fname
        if src_f.exists():
            shutil.copy2(src_f, out_dir / fname)

    dst_pack = out_dir / pack.name
    if dst_pack.exists():
        dst_pack.unlink()
    if args.copy_pack:
        print(f"Copying {pack.name}...", flush=True)
        shutil.copy2(pack, dst_pack)
    else:
        try:
            os.link(pack, dst_pack)
            print(f"Hardlinked {pack} -> {dst_pack}")
        except OSError:
            print(f"Cross-device link unavailable; copying {pack.name}...", flush=True)
            shutil.copy2(pack, dst_pack)

    dt = time.monotonic() - t0
    total_bundle = total_kept_bytes + pack.stat().st_size
    print(
        f"DONE in {dt:.1f}s | Bundle: {out_dir} | "
        f"Safetensors: {total_kept_bytes / (1 << 30):.2f} GiB ({len(shards)} shards) + "
        f"Pack: {pack.stat().st_size / (1 << 30):.2f} GiB = {total_bundle / (1 << 30):.2f} GiB total"
    )


if __name__ == "__main__":
    main()
