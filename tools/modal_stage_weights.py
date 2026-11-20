"""Stage Kimi K3.1 Pro (1.56 TB) from Hugging Face onto a dedicated Modal Volume (v2).

Runs a CPU-only Modal Sandbox (no GPU billing) that downloads straight into the Volume, verifies every
file size against the hub metadata at a pinned revision, writes DONE, and `sync`s the Volume. The
sandbox runs on Modal, so the download continues after this script exits. Resumable: rerunning skips
files that are already complete.

  uv run --no-project --with modal --env-file .env python tools/modal_stage_weights.py           # launch
  uv run --no-project --with modal --env-file .env python tools/modal_stage_weights.py status    # progress

Layout on the Volume `kimi-k31-pro-ab`:
  /Kimi-K3.1-Pro/            weights (+ DONE containing the revision sha)
  /logs/stage_weights.log    download log
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

APP = "k3-stage-weights"
VOLUME = "kimi-k31-pro-ab"
MOUNT = "/vol"
DST = f"{MOUNT}/Kimi-K3.1-Pro"
LOG = f"{MOUNT}/logs/stage_weights.log"
TOOLS = Path(__file__).resolve().parent

INNER = r'''
import os, sys, time
sys.path.insert(0, "/tools")
from pathlib import Path
from download_weights import REPO, expected_sizes, verify
from huggingface_hub import snapshot_download

dst = Path(os.environ["DST"])
dst.mkdir(parents=True, exist_ok=True)
if (dst / "DONE").exists():
    print("already DONE at", (dst / "DONE").read_text()); sys.exit(0)
sizes, rev = expected_sizes()
total = sum(v or 0 for v in sizes.values())
print(f"{REPO}@{rev}: {total/1e12:.3f} TB in {len(sizes)} files", flush=True)
for attempt in range(1, 6):
    t = time.time()
    try:
        snapshot_download(REPO, revision=rev, local_dir=dst, max_workers=int(os.environ.get("WORKERS", "32")))
    except Exception as e:
        print(f"attempt {attempt} failed after {(time.time()-t)/60:.1f} min: {type(e).__name__}: {e}", flush=True)
        time.sleep(30); continue
    print(f"attempt {attempt} finished in {(time.time()-t)/60:.1f} min", flush=True)
    bad = verify(dst, sizes)
    if not bad:
        (dst / "DONE").write_text(rev)
        print("verified all sizes; DONE", flush=True)
        break
    print(f"{len(bad)} files missing/wrong size, e.g. {bad[:3]}; retrying", flush=True)
else:
    sys.exit("giving up after 5 attempts")
'''


def volume() -> modal.Volume:
    return modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)


def launch() -> None:
    app = modal.App.lookup(APP, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("huggingface_hub[hf_xet]>=0.34")
        .add_local_file(TOOLS / "download_weights.py", "/tools/download_weights.py")
    )
    secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
    cmd = (
        f"mkdir -p {MOUNT}/logs && "
        f"python -u - <<'EOF' 2>&1 | tee -a {LOG}\n{INNER}\nEOF\n"
        f"rc=${{PIPESTATUS[0]}}; sync {MOUNT}; echo \"exit $rc; volume synced\" | tee -a {LOG}; sync {MOUNT}; exit $rc"
    )
    sb = modal.Sandbox.create(
        "bash", "-c", cmd,
        app=app, image=image, secrets=[secret],
        volumes={MOUNT: volume()},
        cpu=16, memory=32 * 1024,
        env={"DST": DST, "HF_HOME": "/tmp/hf", "HF_XET_HIGH_PERFORMANCE": "1", "WORKERS": "32"},
        timeout=24 * 60 * 60,
    )
    print(f"launched sandbox {sb.object_id} -> volume {VOLUME}:{DST}")


def status() -> None:
    app = modal.App.lookup(APP, create_if_missing=True)
    sbs = list(modal.Sandbox.list(app_id=app.app_id))
    for sb in sbs:
        rc = sb.poll()
        print(f"sandbox {sb.object_id}: {'running' if rc is None else f'exited {rc}'}")
        if rc is None:
            p = sb.exec("bash", "-c", f"du -sh --apparent-size {DST} 2>/dev/null; tail -5 {LOG}")
            p.wait()
            print(p.stdout.read(), p.stderr.read())
    if not sbs:
        print("no sandboxes for app; tail of log on volume:")
        try:
            print(b"".join(volume().read_file("logs/stage_weights.log")).decode()[-2000:])
        except Exception as e:
            print(f"(no log: {e})")


if __name__ == "__main__":
    {"launch": launch, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "launch"]()
