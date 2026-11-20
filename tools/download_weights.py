"""Download Kimi K3.1 Pro (1.56 TB) to local NVMe with a speed test first.

- Speed test: time one shard; abort with a clear message if the projected total exceeds
  --max-minutes (then use a pre-staged same-DC network volume instead).
- Full download via snapshot_download(local_dir=...), many workers, xet high-performance mode.
  HF_HOME / xet cache are pointed at the target disk (never /dev/shm: it is RAM).
- Verifies every file's size against the hub metadata; writes DONE.

Usage: python tools/download_weights.py /local/Kimi-K3.1-Pro [--max-minutes 60] [--from-dir /workspace/Kimi-K3.1-Pro]
  --from-dir: copy from a pre-staged volume instead of downloading (parallel cp, then verify).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "moonshotai/Kimi-K3.1-Pro"


def expected_sizes():
    from huggingface_hub import HfApi

    info = HfApi().model_info(REPO, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings}, info.sha


def verify(dst: Path, sizes: dict) -> list[str]:
    bad = []
    for f, n in sizes.items():
        p = dst / f
        if not p.exists() or (n is not None and p.stat().st_size != n):
            bad.append(f)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max-minutes", type=float, default=60)
    ap.add_argument("--from-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    if (a.dst / "DONE").exists():
        print("already downloaded")
        return
    os.environ.setdefault("HF_HOME", str(a.dst.parent / ".hf"))
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    sizes, rev = expected_sizes()
    total = sum(v or 0 for v in sizes.values())
    free = shutil.disk_usage(a.dst).free
    print(f"{REPO}@{rev}: {total / 1e12:.2f} TB in {len(sizes)} files; free on target {free / 1e12:.2f} TB")
    if free < total * 1.02:
        sys.exit("not enough free disk on target")

    if a.from_dir:
        t = time.time()
        files = [f for f in sizes if not (a.dst / f).exists() or (a.dst / f).stat().st_size != sizes[f]]
        procs = []
        for f in files:
            (a.dst / f).parent.mkdir(parents=True, exist_ok=True)
            procs.append(subprocess.Popen(["cp", str(a.from_dir / f), str(a.dst / f)]))
            while len([p for p in procs if p.poll() is None]) >= a.workers:
                time.sleep(0.5)
        for p in procs:
            p.wait()
        print(f"copied {len(files)} files in {(time.time() - t) / 60:.1f} min")
    else:
        from huggingface_hub import hf_hub_download, snapshot_download

        shard = max((f for f in sizes if f.endswith(".safetensors")), key=lambda f: sizes[f])
        t = time.time()
        hf_hub_download(REPO, shard, revision=rev, local_dir=a.dst)
        dt = time.time() - t
        rate = sizes[shard] / dt
        proj = total / rate / 60 / max(1, a.workers / 4)  # parallel downloads scale sublinearly; rough
        print(f"speed test: {sizes[shard] / 1e9:.1f} GB in {dt:.0f}s = {rate / 1e9:.2f} GB/s single-stream; "
              f"projected ~{proj:.0f} min with {a.workers} workers")
        if proj > a.max_minutes:
            sys.exit(f"projected download {proj:.0f} min > --max-minutes {a.max_minutes}; use --from-dir with a "
                     "pre-staged volume, or raise --max-minutes")
        t = time.time()
        snapshot_download(REPO, revision=rev, local_dir=a.dst, max_workers=a.workers)
        print(f"downloaded in {(time.time() - t) / 60:.1f} min")
    bad = verify(a.dst, sizes)
    if bad:
        sys.exit(f"{len(bad)} files missing or wrong size, e.g. {bad[:3]}")
    (a.dst / "DONE").write_text(rev)
    print("verified; DONE")


if __name__ == "__main__":
    main()
