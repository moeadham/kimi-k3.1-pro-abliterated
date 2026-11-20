#!/usr/bin/env bash
# Idempotent dev-pod setup. Run on the pod: bash /root/harness/tools/setup_pod.sh
#
# Disk layout (RunPod):
#   /            container disk (overlay, ~50 GB, erased on pod stop) -> venv, code
#   /dev/shm     RAM-backed (~117 GB, erased on stop)                 -> uv/HF caches, tiny checkpoints
#   /workspace   global volume (object-backed FUSE, no chmod/rename)  -> durable outputs only
set -euo pipefail

VENV=/opt/venv
export UV_CACHE_DIR=/dev/shm/uv-cache
export HF_HOME=/dev/shm/hf
# Pinned nightly from main @ 2440ffc38 (2026-11-17). The 0.34.0 *release* has the K3 model
# code but its branch predates the K3.1 packed hybrid KV-cache plumbing on main, and K3.1
# (KDA + DSA) fails KV-cache init there ("DEEPSEEK_V32_INDEXER needs 64-token kernel blocks").
VLLM_SPEC="${VLLM_SPEC:-vllm==0.34.0rc1.dev673+g2440ffc38}"
# Per-commit index: /nightly only keeps the latest build, so the pinned wheel disappears from it.
VLLM_INDEX="${VLLM_INDEX:-https://wheels.vllm.ai/2440ffc3870cdd6329bdf7ec0f780e05a2f01eb0}"

apt-get -qq update >/dev/null && apt-get -qq install -y tmux rsync jq >/dev/null || true

if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" --python 3.12
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# One resolver call so vLLM's exact pins (torch, transformers, flashinfer) win; never -U extras
# separately, which silently upgraded torch underneath vLLM's compiled ops.
uv pip install --pre "$VLLM_SPEC" compressed-tensors safetensors pytest typesafe-sdk huggingface_hub pyarrow httpx \
  tiktoken blobfile optuna math-verify lm-eval langdetect immutabledict nltk \
  --extra-index-url "$VLLM_INDEX" --torch-backend=auto --index-strategy unsafe-best-match
python - <<'EOF'
# Fail loudly if installed versions drift from vLLM's own requirements.
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
        if v.split("+")[0] not in r.specifier and r.specifier:
            bad.append(f"{r.name} {v} !~ {r.specifier}")
if bad:
    raise SystemExit("version drift vs vllm pins: " + "; ".join(bad))
print("vllm pins OK")
EOF

# Our package (editable, no deps) so vLLM discovers the `vllm.general_plugins` entry point
# (harness.vllm_plugin:register; inert unless K3H_STEERING=1).
uv pip install --no-deps -e /root/harness

# Vendored upstream fixes (Python-only). All-or-nothing: a patch is applied only if every
# hunk applies cleanly; a partially applied patch is a silent-failure risk, so it aborts.
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
shopt -s nullglob
for p in /root/harness/tools/patches/*.diff; do
  name=$(basename "$p")
  if patch -p1 -d "$SITE" --dry-run -R -s -f < "$p" >/dev/null 2>&1; then
    echo "patch already applied: $name"
  elif patch -p1 -d "$SITE" --dry-run -s -f < "$p" >/dev/null 2>&1; then
    patch -p1 -d "$SITE" -s -f --no-backup-if-mismatch < "$p" && echo "patch applied: $name"
  else
    echo "patch does NOT apply cleanly: $name" >&2; exit 1
  fi
done

python - <<'EOF'
import importlib.util, pathlib, json, subprocess, torch, vllm
root = pathlib.Path(vllm.__file__).parent
k3_dir = root / "models" / "kimi_k3"
assert (k3_dir / "nvidia" / "model.py").exists(), f"vLLM has no Kimi K3 model code at {k3_dir}"
k3 = sorted(str(p.relative_to(root)) for p in k3_dir.rglob("*.py") if p.parent == k3_dir / "nvidia")
info = {
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "cc": ".".join(map(str, torch.cuda.get_device_capability(0))),
    "flashinfer": (importlib.util.find_spec("flashinfer") and __import__("flashinfer").__version__) or None,
    "compressed_tensors": __import__("compressed_tensors").__version__,
    "kimi_model_files": k3,
}
print(json.dumps(info, indent=2))
pathlib.Path("/root/harness/env.lock.json").write_text(json.dumps(info, indent=2))
EOF

grep -q 'opt/venv' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
export UV_CACHE_DIR=/dev/shm/uv-cache
export HF_HOME=/dev/shm/hf
source /opt/venv/bin/activate
EOF
df -h / /dev/shm | sed 1d
echo "setup_pod: done"
