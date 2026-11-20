"""Build a tiny random-weight Kimi K3.1 checkpoint with the real architecture (step 1.2).

Keeps d_model=7168, the real vocab/tokenizer, KDA + DSA attention, latent MoE, AttnRes,
shared experts, and (variant "mxfp4") the production MXFP4 compressed-tensors expert format.
Shrinks: 8 layers, attn_res_block_size=4, 16 routed experts (top-4). No real weights are
downloaded; tensor shapes/dtypes come from the real checkpoint's safetensors headers
(tests/fixtures/k31_shapes_sample.json, layers 0/1/3 + globals).

Tiny layer layout (0-indexed) repeats the real 3xKDA + 1xDSA pattern twice:
0 = KDA + dense MLP, 1,2,4,5,6 = KDA + MoE, 3,7 = DSA + MoE. Block-write layers: 0, 4.
Why 8 layers, not 4: vLLM sizes the packed hybrid KV block from the KDA-states-per-DSA-layer
ratio (models/kimi_k3/common/dsa_cache.py::packed_dsa_block_size). With one DSA layer the
block is ~3x the real model's and the DSA indexer cannot tile it; 6 KDA : 2 DSA matches
the real 69:24 row layout.

Usage (on the pod):
  python tools/make_tiny_k3.py --variant mxfp4 --out /dev/shm/tiny-k3/mxfp4
  python tools/make_tiny_k3.py --variant bf16  --out /dev/shm/tiny-k3/bf16
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = "moonshotai/Kimi-K3.1-Pro"
ROOT = Path(__file__).resolve().parent.parent
SHAPES = ROOT / "tests/fixtures/k31_shapes_sample.json"

N_LAYERS = 8
# tiny layer -> real layer used as shape template (0: KDA+dense, 1: KDA+MoE, 3: DSA+MoE)
SOURCE_LAYER = {0: 0, 1: 1, 2: 1, 3: 3, 4: 1, 5: 1, 6: 1, 7: 3}
KDA_LAYERS_1IDX = [1, 2, 3, 5, 6, 7]
DSA_LAYERS_1IDX = [4, 8]
N_EXPERTS, TOP_K = 16, 4
BLOCK = 4
MXFP4_SCALE_BYTE = 127 - 8  # e8m0 exponent -> 2^-8; |w| <= 6/256 ~ 0.023
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}
SHARD_BYTES = 2 << 30


def init_tensor(name: str, dtype: str, shape: list[int], g: torch.Generator) -> torch.Tensor:
    dt = DTYPES[dtype]
    n = name.rsplit(".", 1)[-1] if not name.endswith(".weight") else name
    if dtype == "U8":
        if name.endswith("weight_scale"):
            return torch.full(shape, MXFP4_SCALE_BYTE, dtype=torch.uint8)
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    if name.endswith("A_log"):
        return torch.empty(shape).uniform_(1, 16, generator=g).log().to(dt)
    if name.endswith("dt_bias"):
        dtv = torch.empty(shape).uniform_(math.log(1e-3), math.log(1e-1), generator=g).exp()
        return (dtv + torch.log(-torch.expm1(-dtv))).to(dt)  # inverse softplus
    if "conv1d" in name:
        return (torch.randn(shape, generator=g) * 0.3).to(dt)
    if name.endswith(("e_score_correction_bias", ".bias")):
        return torch.zeros(shape, dtype=dt)
    if len(shape) == 1:  # all remaining 1-D tensors are norm gains
        return torch.ones(shape, dtype=dt)
    return (torch.randn(shape, generator=g) * 0.02).to(dt)


def build_tensors(variant: str, seed: int) -> dict[str, torch.Tensor]:
    src = json.loads(SHAPES.read_text())
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    pre = "language_model.model."
    for name, (dtype, shape) in sorted(src.items()):
        if name.startswith(("vision_tower", "mm_projector")):
            continue  # vision tower is meta (limit_mm_per_prompt image=0)
        m = re.match(re.escape(pre) + r"layers\.(\d+)\.(.*)", name)
        if not m:
            out[name] = init_tensor(name, dtype, shape, g)
            continue
        src_layer, rest = int(m.group(1)), m.group(2)
        for tiny, s in SOURCE_LAYER.items():
            if s != src_layer:
                continue
            tname = f"{pre}layers.{tiny}.{rest}"
            if rest.startswith("block_sparse_moe.gate."):
                shape = [N_EXPERTS, *shape[1:]]
            if ".experts.0." in rest:
                for e in range(N_EXPERTS):
                    ename = tname.replace(".experts.0.", f".experts.{e}.")
                    if variant == "mxfp4":
                        out[ename] = init_tensor(ename, dtype, shape, g)
                    elif rest.endswith("weight_packed"):
                        # BF16 expert: unpacked [out, in] = packed [out, in/2] * 2
                        bname = ename.replace("weight_packed", "weight")
                        out[bname] = init_tensor(bname, "BF16", [shape[0], shape[1] * 2], g)
                continue
            out[tname] = init_tensor(tname, dtype, shape, g)
    return out


def tiny_config(variant: str, meta_dir: Path) -> dict:
    cfg = json.loads((meta_dir / "config.json").read_text())
    t = cfg["text_config"]
    t["num_hidden_layers"] = N_LAYERS
    t["num_experts"] = N_EXPERTS
    t["num_experts_per_token"] = TOP_K
    t["attn_res_block_size"] = BLOCK
    t["linear_attn_config"]["kda_layers"] = KDA_LAYERS_1IDX
    t["linear_attn_config"]["full_attn_layers"] = DSA_LAYERS_1IDX
    t["dsa_config"]["dsa_layers"] = DSA_LAYERS_1IDX
    if variant == "bf16":
        t.pop("quantization_config", None)
        cfg.pop("quantization_config", None)
    cfg["_tiny_k3"] = {"variant": variant, "source": REPO, "layers": SOURCE_LAYER, "experts": N_EXPERTS}
    return cfg


def fetch_meta(meta_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO,
        local_dir=meta_dir,
        allow_patterns=["*.json", "*.py", "*.model", "*.tiktoken", "*.jinja", "*.txt"],
        ignore_patterns=["model.safetensors.index.json"],
    )


def write_checkpoint(tensors: dict[str, torch.Tensor], out: Path) -> None:
    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for name, t in tensors.items():
        nbytes = t.numel() * t.element_size()
        if size and size + nbytes > SHARD_BYTES:
            shards.append({})
            size = 0
        shards[-1][name] = t.contiguous()
        size += nbytes
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, out / fname, metadata={"format": "pt"})
        for name, t in shard.items():
            weight_map[name] = fname
            total += t.numel() * t.element_size()
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["mxfp4", "bf16"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=Path("/dev/shm/k31-meta"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not (args.meta / "config.json").exists():
        fetch_meta(args.meta)
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for f in args.meta.iterdir():
        if f.is_file() and f.name != "config.json":
            shutil.copy2(f, args.out / f.name)
    (args.out / "config.json").write_text(json.dumps(tiny_config(args.variant, args.meta), indent=2))

    tensors = build_tensors(args.variant, args.seed)
    write_checkpoint(tensors, args.out)
    gb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9
    print(f"wrote {len(tensors)} tensors, {gb:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
