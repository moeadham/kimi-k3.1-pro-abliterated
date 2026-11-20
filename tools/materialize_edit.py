"""Materialize a writer edit into a full standalone checkpoint (CPU only).

Copies every file of the base checkpoint to OUT; in the shards that hold edited writers, overwrites just those
tensors' bytes in place (same dtype and shape, so the safetensors header and every other tensor stay byte-identical,
including the MXFP4 routed experts). The edit is the one quick_probe / tp.edit_writers applies at load time:
W <- W - lam_slot * Q_slot^T (Q_slot W) for the attention (slot 2L+1) and latent-MoE + shared-expert (slot 2L+2)
writers of the layers in the spec, Q from quick_probe.step_basis, lam from quick_probe.lam_for_layers.

Resumable (progress.jsonl, one line per finished shard); DONE is written last. Verification afterwards: every file
size matches; each edited tensor re-read from OUT equals the recomputed edit bit for bit; one untouched tensor per
edited shard and sampled byte blocks of every copied shard match the base.

  python tools/materialize_edit.py /vol/Kimi-K3.1-Pro /vol/Kimi-K3.1-Pro-H3 /vol/banks/run3_final/edit_l1.5.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from harness import ablation, steering  # noqa: E402

LAYER_RE = re.compile(r"layers\.(\d+)\.")
KINDS = {"self_attn.o_proj.weight": "attn", "block_sparse_moe.routed_expert_up_proj.weight": "mlp",
         "block_sparse_moe.shared_experts.down_proj.weight": "mlp", "mlp.down_proj.weight": "mlp"}


def writer_slot(name: str) -> int | None:
    """Residual-writer slot of a checkpoint tensor name, or None (routed experts and non-writers)."""
    if ".experts." in name:
        return None
    m = LAYER_RE.search(name)
    if not m:
        return None
    layer = int(m.group(1))
    for suffix, kind in KINDS.items():
        if name.endswith(suffix):
            if suffix == "mlp.down_proj.weight" and "shared_experts" in name:
                continue
            return steering.slot_of(kind, layer)
    return None


def load_edit(spec_path: str):
    import quick_probe as qp
    spec = json.loads(Path(spec_path).read_text())
    comps = spec.get("components") or [{k: spec[k] for k in ("bank", "family", "variant")}]
    banks = {c["bank"]: torch.load(c["bank"], map_location="cpu") for c in comps}
    Q = qp.step_basis(banks, {"components": comps})  # [S, k, d]
    lam = qp.lam_for_layers(Q.shape[0], tuple(spec["layers"]), float(spec["lam"]), spec.get("writers", "both"),
                            spec.get("exclude"), spec.get("lam_by_layer"))
    return spec, Q, lam


def header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def edited_bytes(src: Path, name: str, Q, lam) -> tuple[torch.Tensor, bytes]:
    slot = writer_slot(name)
    with safe_open(str(src), "pt") as f:
        W = f.get_tensor(name)
    new = ablation.ablate_output_weight(W, Q[slot], float(lam[slot]))
    return new, new.contiguous().view(torch.uint8).numpy().tobytes()


def do_shard(src: Path, dst: Path, targets: list[str], Q, lam) -> dict:
    t0 = time.time()
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    out = {"shard": src.name, "edited": {}}
    if targets:
        base, hdr = header(src)
        fd = os.open(tmp, os.O_RDWR)
        try:
            for name in targets:
                meta = hdr[name]
                if meta["dtype"] != "BF16":
                    raise RuntimeError(f"{name}: dtype {meta['dtype']}, expected BF16")
                _, b = edited_bytes(src, name, Q, lam)
                a, e = meta["data_offsets"]
                if len(b) != e - a:
                    raise RuntimeError(f"{name}: {len(b)} bytes, slot holds {e - a}")
                os.pwrite(fd, b, base + a)
                out["edited"][name] = hashlib.sha256(b).hexdigest()
            os.fsync(fd)
        finally:
            os.close(fd)
    os.replace(tmp, dst)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def verify(base: Path, out: Path, plan: dict, Q, lam, blocks: int = 8, block: int = 4 << 20) -> dict:
    rng = random.Random(0)
    bad, n_ok = [], 0
    for shard, targets in plan.items():
        s, d = base / shard, out / shard
        if s.stat().st_size != d.stat().st_size:
            bad.append(f"{shard}: size differs")
            continue
        if targets:
            with safe_open(str(d), "pt") as fo, safe_open(str(s), "pt") as fb:
                for name in targets:
                    want, _ = edited_bytes(s, name, Q, lam)
                    if not torch.equal(fo.get_tensor(name), want):
                        bad.append(f"{shard}: {name} differs from the recomputed edit")
                    else:
                        n_ok += 1
                other = next((k for k in fb.keys() if k not in targets), None)
                if other is not None and not torch.equal(fo.get_tensor(other), fb.get_tensor(other)):
                    bad.append(f"{shard}: untouched tensor {other} changed")
        else:
            size = s.stat().st_size
            with open(s, "rb") as fs, open(d, "rb") as fd:
                for off in [0, max(size - block, 0)] + [rng.randrange(0, max(size - block, 1)) for _ in range(blocks)]:
                    fs.seek(off), fd.seek(off)
                    if fs.read(block) != fd.read(block):
                        bad.append(f"{shard}: bytes differ at {off}")
                        break
    return {"edited_tensors_verified": n_ok, "problems": bad}


def main(base: Path, out: Path, spec_path: str, workers: int) -> None:
    spec, Q, lam = load_edit(spec_path)
    out.mkdir(parents=True, exist_ok=True)
    wmap = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    plan: dict[str, list[str]] = {sh: [] for sh in sorted(set(wmap.values()))}
    for name, sh in wmap.items():
        slot = writer_slot(name)
        if slot is not None and float(lam[slot]) != 0.0:
            plan[sh].append(name)
    n_edit = sum(map(len, plan.values()))
    print(f"[mat] {len(plan)} shards, {n_edit} tensors to edit in {sum(1 for v in plan.values() if v)} shards",
          flush=True)
    prog = out / "progress.jsonl"
    done = {json.loads(l)["shard"]: json.loads(l) for l in open(prog)} if prog.exists() else {}
    todo = [sh for sh in plan if sh not in done or not (out / sh).exists()]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool, open(prog, "a") as pf:
        for i, r in enumerate(pool.map(lambda sh: do_shard(base / sh, out / sh, plan[sh], Q, lam), todo), 1):
            pf.write(json.dumps(r) + "\n")
            pf.flush()
            done[r["shard"]] = r
            print(f"[mat] {i}/{len(todo)} {r['shard']} ({len(r['edited'])} edited, {r['seconds']}s; "
                  f"{time.time() - t0:.0f}s total)", flush=True)
    for p in base.iterdir():  # config, tokenizer, index, code, assets (not the HF download cache)
        if p.name in plan or p.name in (".cache", "DONE"):
            continue
        dst = out / p.name
        if p.is_dir():
            shutil.copytree(p, dst, dirs_exist_ok=True)
        else:
            shutil.copyfile(p, dst)
    print(f"[mat] copy done in {time.time() - t0:.0f}s; verifying", flush=True)
    v = verify(base, out, plan, Q, lam)
    edit = {"base": str(base), "spec": spec, "spec_file": spec_path, "edited_tensors": n_edit,
            "tensor_sha256": {k: h for r in done.values() for k, h in r["edited"].items()}, "verify": v,
            "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "EDIT.json").write_text(json.dumps(edit, indent=1))
    print(f"[mat] verify: {v['edited_tensors_verified']}/{n_edit} edited tensors exact, {len(v['problems'])} problems",
          flush=True)
    for p in v["problems"][:20]:
        print(f"[mat] PROBLEM {p}", flush=True)
    if v["problems"] or v["edited_tensors_verified"] != n_edit:
        sys.exit(1)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print(f"[mat] DONE {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(4)
    main(Path(a.base), Path(a.out), a.spec, a.workers)
