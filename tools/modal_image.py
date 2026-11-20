"""Modal image + persistent Volume for the K3.1 Step-2 session (port of tools/setup_pod.sh).

The image is built on Modal's CPU builders (no GPU billing) and cached; only the last layer
(this repo) rebuilds when code changes.

  uv run --no-project --with modal --env-file .env python tools/modal_image.py prep

`prep` runs a CPU-only sandbox on the image that:
  1. checks installed torch/transformers/flashinfer against vLLM's own pins (same guard as setup_pod.sh)
  2. lists which GPU architectures the compiled kernels in vLLM / torch / FlashInfer ship for
     (B300 is sm_103; sm_100a-only cubins do NOT run there and would need PTX JIT or fail)
  3. builds the tiny K3 checkpoints onto the Volume (CPU-only; tools/make_tiny_k3.py)

Volume `kimi-k31-pro-ab` (v2), mounted at /vol:
  /vol/Kimi-K3.1-Pro        weights (tools/modal_stage_weights.py)
  /vol/Kimi-K3.1-Pro-abl    baked copy (written once)
  /vol/session              session $OUT (resumable stages)
  /vol/cache/...            FlashInfer / vLLM / Triton / inductor caches
  /vol/tiny-k3/{mxfp4,bf16} tiny checkpoints;  /vol/k31-meta  config + tokenizer
  /vol/logs                 logs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
VOLUME = "kimi-k31-pro-ab"
MOUNT = "/vol"
APP = "k3-session"

VLLM_SPEC = "vllm==0.34.0rc1.dev673+g2440ffc38"  # pinned nightly; see docs/runtime_notes.md
# Per-commit wheel index for 2440ffc38: /nightly only keeps the latest build, so the pin vanishes from it.
VLLM_INDEX = "https://wheels.vllm.ai/2440ffc3870cdd6329bdf7ec0f780e05a2f01eb0"
CUDA_BASE = "nvidia/cuda:13.2.1-devel-ubuntu24.04"  # devel: nvcc/cuobjdump for FlashInfer JIT + arch checks

CACHE_ENV = {
    "FLASHINFER_WORKSPACE_BASE": f"{MOUNT}/cache/flashinfer",
    "VLLM_CACHE_ROOT": f"{MOUNT}/cache/vllm",
    "TRITON_CACHE_DIR": f"{MOUNT}/cache/triton",
    "TORCHINDUCTOR_CACHE_DIR": f"{MOUNT}/cache/inductor",
    "HF_HOME": "/tmp/hf",
    "HF_XET_HIGH_PERFORMANCE": "1",
}

PIN_CHECK = r'''
from importlib.metadata import requires, version
from packaging.requirements import Requirement
bad = []
for r in map(Requirement, requires("vllm") or []):
    if r.marker and not r.marker.evaluate():
        continue
    if r.name in {"torch", "transformers", "flashinfer-python", "torchvision", "triton"}:
        try:
            v = version(r.name)
        except Exception:
            continue
        if r.specifier and v.split("+")[0] not in r.specifier:
            bad.append(f"{r.name} {v} !~ {r.specifier}")
if bad:
    raise SystemExit("version drift vs vllm pins: " + "; ".join(bad))
print("vllm pins OK")
'''


def volume() -> modal.Volume:
    return modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)


def image() -> modal.Image:
    img = (
        modal.Image.from_registry(CUDA_BASE, add_python="3.12")
        .apt_install("tmux", "rsync", "jq", "patch", "git", "curl", "procps", "binutils")
        .pip_install("uv")
        .run_commands(
            # One resolver call so vLLM's exact pins win (never -U extras separately).
            f"uv pip install --system --pre '{VLLM_SPEC}' compressed-tensors safetensors pytest typesafe-sdk "
            "huggingface_hub[hf_xet] pyarrow httpx tiktoken blobfile optuna math-verify lm-eval langdetect "
            f"immutabledict nltk --extra-index-url {VLLM_INDEX} --torch-backend=cu132 --index-strategy unsafe-best-match",
            f"python - <<'EOF'\n{PIN_CHECK}\nEOF",
        )
        # Prebuilt FlashInfer kernels (cubins + AOT JIT cache) at the exact installed version, so the first
        # GPU launch doesn't spend ~10 min downloading cubins and JIT-compiling. --no-deps: never move pins.
        .run_commands(
            "FI=$(python -c 'import flashinfer; print(flashinfer.__version__)') && echo flashinfer $FI && "
            # flashinfer-jit-cache itself is a 4 KB meta package; the kernels live in the per-arch wheel
            # (B300 = sm_103a), which --no-deps would otherwise skip.
            "uv pip install --system --no-deps \"flashinfer-cubin==$FI\" \"flashinfer-jit-cache==$FI+cu130\" "
            "\"flashinfer-jit-cache-sm103a==$FI+cu130\" "
            "--find-links https://flashinfer.ai/whl/flashinfer-cubin/ "
            "--find-links https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/ "
            "--find-links https://flashinfer.ai/whl/cu130/flashinfer-jit-cache-sm103a/",
            f"python - <<'EOF'\n{PIN_CHECK}\nEOF",
        )
        .env(CACHE_ENV)
        .add_local_dir(
            ROOT, "/root/harness", copy=True,
            ignore=[".git", ".venv", "**/__pycache__", ".env", ".env.*", "out", ".claude"],
        )
        .run_commands(
            "uv pip install --system --no-deps -e /root/harness",
            # Vendored patches, all-or-nothing (same rule as setup_pod.sh). None are committed today.
            # Builder runs /bin/sh, so run the loop under bash (nullglob).
            "bash <<'EOF'\nset -e\n"
            "SITE=$(python -c 'import site; print(site.getsitepackages()[0])'); shopt -s nullglob\n"
            "for p in /root/harness/tools/patches/*.diff; do\n"
            "  if patch -p1 -d $SITE --dry-run -R -s -f < $p >/dev/null 2>&1; then echo already: $p\n"
            "  elif patch -p1 -d $SITE --dry-run -s -f < $p >/dev/null 2>&1; then patch -p1 -d $SITE -s -f < $p && echo applied: $p\n"
            "  else echo DOES NOT APPLY: $p >&2; exit 1; fi\ndone\necho patches: done\nEOF",
        )
    )
    return img


PREP = r'''
set -uo pipefail
mkdir -p /vol/logs /vol/cache/flashinfer /vol/cache/vllm /vol/cache/triton /vol/cache/inductor /vol/session
cd /root/harness
echo "=== versions"
python - <<'EOF'
import importlib.metadata as md, pathlib, json, vllm, torch
root = pathlib.Path(vllm.__file__).parent
k3 = root / "models" / "kimi_k3" / "nvidia" / "model.py"
v = {p: md.version(p) for p in ["vllm", "torch", "transformers", "flashinfer-python", "compressed-tensors", "triton"]}
for p in ["flashinfer-cubin", "flashinfer-jit-cache"]:
    try: v[p] = md.version(p)
    except Exception: v[p] = None
v["torch.cuda"] = torch.version.cuda
v["torch_arch_list"] = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else None
v["k3_model_code"] = k3.exists()
print(json.dumps(v, indent=2))
EOF
echo "=== GPU archs in compiled kernels (need sm_103 / sm_103a / sm_100f, or compute_10x PTX)"
python - <<'EOF'
import pathlib, subprocess, collections, re, site
sp = pathlib.Path(site.getsitepackages()[0])
groups = {"vllm": sp/"vllm", "torch": sp/"torch/lib", "flashinfer": sp/"flashinfer",
          "flashinfer_jit_cache": sp/"flashinfer_jit_cache", "flashinfer_jit_cache_sm103a": sp/"flashinfer_jit_cache_sm103a",
          "flashinfer_cubin": sp/"flashinfer_cubin"}
for g, d in groups.items():
    if not d.exists(): print(f"{g}: (not installed)"); continue
    files = [p for p in d.rglob("*") if p.suffix in (".so", ".cubin") and p.is_file()]
    elf, ptx = collections.Counter(), collections.Counter()
    for f in files:
        if f.suffix == ".cubin":
            m = re.search(r"sm_?(\d+[af]?)", f.name); elf[f"sm_{m.group(1)}" if m else "cubin?"] += 1; continue
        for flag, ctr in (("--list-elf", elf), ("--list-ptx", ptx)):
            out = subprocess.run(["cuobjdump", flag, str(f)], capture_output=True, text=True).stdout
            for a in set(re.findall(r"sm_(\d+[af]?)", out)): ctr[f"sm_{a}"] += 1
    big = lambda c: {k: v for k, v in sorted(c.items()) if k.startswith(("sm_10", "sm_11", "sm_12"))}
    print(f"{g}: {len(files)} files; Blackwell SASS {big(elf)}; Blackwell PTX {big(ptx)}")
EOF
echo "=== tiny K3 checkpoints -> /vol/tiny-k3"
for v in mxfp4 bf16; do
  if [ -f /vol/tiny-k3/$v/model.safetensors.index.json ]; then echo "tiny $v exists"; continue; fi
  python tools/make_tiny_k3.py --variant $v --out /vol/tiny-k3/$v --meta /vol/k31-meta || echo "tiny $v FAILED"
done
du -sh /vol/tiny-k3/* /vol/k31-meta 2>/dev/null
sync /vol && echo "volume synced"
'''


def prep() -> None:
    app = modal.App.lookup(APP, create_if_missing=True)
    sb = modal.Sandbox.create(
        "bash", "-c", f"( {PREP} ) 2>&1 | tee {MOUNT}/logs/prep.log",
        app=app, image=image(), volumes={MOUNT: volume()},
        secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})],
        cpu=16, memory=64 * 1024, timeout=60 * 60,
    )
    print(f"prep sandbox {sb.object_id}", flush=True)
    for line in sb.stdout:
        print(line, end="", flush=True)
    sb.wait()
    print(f"prep exit {sb.returncode}")


if __name__ == "__main__":
    {"prep": prep}[sys.argv[1] if len(sys.argv) > 1 else "prep"]()
