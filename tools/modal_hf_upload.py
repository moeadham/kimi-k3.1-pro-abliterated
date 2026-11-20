"""Upload the edited checkpoint on the Modal Volume to the private Hugging Face repo, then verify it there.

Runs a CPU-only Modal Sandbox with the Volume mounted, so the upload continues after this script exits.
Commits the small files first, then the 96 shards in groups of GROUP (default 4, ~65 GB) with one
`upload_folder` commit per group, retrying a failed group and skipping groups already on the hub, so a rerun
resumes. (The first two attempts ran one `hf upload` of the whole folder, which put all 100 large files in a
single batch; both died with 0 commits, on an Xet 429 and then an Xet "xorb not found": `/vol/logs/hfup*.log`.)

After the upload the sandbox checks the repo independently of the uploader:
  1. every local file is on the hub with the same size (and nothing local-only like DONE is);
  2. every unedited shard's sha256 on the hub equals the original's in moonshotai/Kimi-K3.1-Pro;
  3. every edited tensor (111, listed in EDIT.json) is range-read back from the hub and its sha256 matches
     EDIT.json (which `materialize_edit.py` verified against a recomputation of the edit).

  uv run --no-project --with 'modal[api-proxy-support]' --env-file .env python tools/modal_hf_upload.py          # launch
  uv run --no-project --with 'modal[api-proxy-support]' --env-file .env python tools/modal_hf_upload.py status   # progress
  uv run --no-project --with 'modal[api-proxy-support]' --env-file .env python tools/modal_hf_upload.py verify   # checks only

Log: /vol/logs/hf_upload.log; verification report: /vol/logs/hf_verify.json.
"""

from __future__ import annotations

import os
import sys

import modal

APP = "k3-hf-upload"
VOLUME = "kimi-k31-pro-ab"
MOUNT = "/vol"
SRC = f"{MOUNT}/Kimi-K3.1-Pro-H3"
REPO = "moeadham/Kimi-K3.1-Pro-abliterated"
BASE = "moonshotai/Kimi-K3.1-Pro"
LOG = f"{MOUNT}/logs/hf_upload.log"
REPORT = f"{MOUNT}/logs/hf_verify.json"
# Volume bookkeeping from materialize_edit.py; not part of the model.
IGNORE = ["DONE", "progress.jsonl", ".cache/**"]

UPLOAD = r'''
import os, shutil, sys, time
from huggingface_hub import HfApi
api = HfApi()
src, repo = os.environ["SRC"], os.environ["REPO"]
ignore = os.environ["IGNORE"].split(",")
group = int(os.environ["GROUP"])
if not os.path.exists(os.path.join(src, "DONE")):
    sys.exit(f"{src} has no DONE marker; refusing to upload a partial checkpoint")
if not api.model_info(repo).private:
    sys.exit(f"{repo} is not private; refusing to upload")

def remote():
    i = api.model_info(repo, files_metadata=True)
    return {s.rfilename: s.size for s in i.siblings}

shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
# Small files first (one commit), then shards a few at a time so each group is its own commit: an Xet
# failure costs at most one group, and a rerun skips every group already on the hub with the right size.
batches = [("model card, config, tokenizer, code, license, EDIT.json", None)]
batches += [(f"weights {s[0]} .. {s[-1]}", s) for s in (shards[i:i + group] for i in range(0, len(shards), group))]
t0 = time.time()
for n, (msg, files) in enumerate(batches, 1):
    if files is not None:
        have = remote()
        if all(have.get(f) == os.path.getsize(os.path.join(src, f)) for f in files):
            print(f"[{n}/{len(batches)}] already on hub: {msg}", flush=True); continue
    for attempt in range(1, 7):
        t = time.time()
        try:
            api.upload_folder(repo_id=repo, repo_type="model", folder_path=src, commit_message=f"Upload {msg}",
                              allow_patterns=files, ignore_patterns=ignore + ([] if files else ["*.safetensors"]))
            gb = sum(os.path.getsize(os.path.join(src, f)) for f in files or []) / 1e9
            dt = time.time() - t
            print(f"[{n}/{len(batches)}] committed {msg}: {gb:.0f} GB in {dt/60:.1f} min ({gb/max(dt,1):.2f} GB/s); "
                  f"elapsed {(time.time()-t0)/3600:.2f} h", flush=True)
            break
        except Exception as e:
            print(f"[{n}/{len(batches)}] attempt {attempt} failed after {(time.time()-t)/60:.1f} min: "
                  f"{type(e).__name__}: {str(e)[:500]}", flush=True)
            # A stale local Xet shard cache can reference xorbs the server never committed ("xorb not found").
            shutil.rmtree(os.path.join(os.environ["HF_HOME"], "xet"), ignore_errors=True)
            time.sleep(min(60 * attempt, 300))
    else:
        sys.exit(f"giving up on {msg} after 6 attempts")
print(f"upload complete in {(time.time()-t0)/3600:.2f} h", flush=True)
'''

VERIFY = r'''
import fnmatch, hashlib, json, os, struct, sys
from concurrent.futures import ThreadPoolExecutor
import httpx
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers
api = HfApi()
src, repo, base = os.environ["SRC"], os.environ["REPO"], os.environ["BASE"]
ignore = os.environ["IGNORE"].split(",")
problems = []

local = {}
for root, dirs, files in os.walk(src):
    dirs[:] = [d for d in dirs if d != ".cache"]
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), src)
        if not any(fnmatch.fnmatch(rel, p) for p in ignore):
            local[rel] = os.path.getsize(os.path.join(root, f))

def meta(r):
    i = api.model_info(r, files_metadata=True)
    return i, {s.rfilename: s for s in i.siblings}
info, hub = meta(repo)
_, up = meta(base)
print(f"{repo}@{info.sha} private={info.private}: {len(hub)} files; local {len(local)}", flush=True)
if not info.private:
    problems.append("repo is not private")

# 1. presence and size
for k, n in local.items():
    if k not in hub: problems.append(f"missing on hub: {k}")
    elif hub[k].size != n: problems.append(f"size {k}: hub {hub[k].size} local {n}")
for k in hub:
    if k not in local: problems.append(f"on hub but not local: {k}")

# 2. unedited shards are byte-identical to the original (hub-side sha256, no download)
edit = json.load(open(os.path.join(src, "EDIT.json")))
wmap = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
edited_shards = sorted({wmap[t] for t in edit["tensor_sha256"]})
shards = sorted(k for k in local if k.endswith(".safetensors"))
sha = lambda s: s.lfs.sha256 if s.lfs else None
same = 0
for k in shards:
    if k in edited_shards or k not in hub: continue
    if sha(hub[k]) and sha(hub[k]) == sha(up.get(k)): same += 1
    else: problems.append(f"unedited shard differs from {base}: {k}")
for k in edited_shards:
    if k in hub and sha(hub[k]) == sha(up.get(k)):
        problems.append(f"edited shard identical to the original: {k}")
print(f"unedited shards identical to the original: {same}/{len(shards) - len(edited_shards)}", flush=True)

# 3. range-read each edited tensor back from the hub and check its sha256
headers = build_hf_headers()
client = httpx.Client(headers=headers, follow_redirects=True, timeout=600)
def rng(url, a, b):
    for attempt in range(5):
        try:
            r = client.get(url, headers={"Range": f"bytes={a}-{b-1}"})
            r.raise_for_status()
            if len(r.content) != b - a: raise IOError(f"short read {len(r.content)} != {b-a}")
            return r.content
        except Exception as e:
            err = e
    raise err
def check_shard(shard):
    url = hf_hub_url(repo, shard, revision=info.sha)
    n = struct.unpack("<Q", rng(url, 0, 8))[0]
    hdr = json.loads(rng(url, 8, 8 + n))
    out = []
    for t, want in edit["tensor_sha256"].items():
        if wmap[t] != shard: continue
        a, b = hdr[t]["data_offsets"]
        h, step = hashlib.sha256(), 256 << 20
        for o in range(a, b, step):
            h.update(rng(url, 8 + n + o, 8 + n + min(o + step, b)))
        out.append((t, h.hexdigest() == want))
    return out
ok = 0
with ThreadPoolExecutor(8) as ex:
    for res in ex.map(check_shard, edited_shards):
        for t, good in res:
            ok += good
            if not good: problems.append(f"edited tensor sha256 mismatch: {t}")
print(f"edited tensors matching EDIT.json, read back from the hub: {ok}/{len(edit['tensor_sha256'])}", flush=True)

report = {"repo": repo, "revision": info.sha, "files_hub": len(hub), "files_local": len(local),
          "unedited_shards_identical": same, "edited_shards": len(edited_shards),
          "edited_tensors_verified": ok, "problems": problems}
json.dump(report, open(os.environ["REPORT"], "w"), indent=1)
print(json.dumps(report, indent=1), flush=True)
sys.exit(1 if problems else 0)
'''


def volume() -> modal.Volume:
    return modal.Volume.from_name(VOLUME, version=2)


def launch(steps: str) -> None:
    app = modal.App.lookup(APP, create_if_missing=True)
    for sb in modal.Sandbox.list(app_id=app.app_id):
        if sb.poll() is None:
            sys.exit(f"sandbox {sb.object_id} is still running; check `status` or stop it first")
    image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet]==2.4.0", "httpx")
    secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
    parts = [f"mkdir -p {MOUNT}/logs", f"echo \"=== {steps} $(date -u +%FT%TZ)\" >> {LOG}"]
    if "upload" in steps:
        parts.append(f"python -u - <<'EOF' 2>&1 | tee -a {LOG}\n{UPLOAD}\nEOF\n"
                     f"rc=${{PIPESTATUS[0]}}; sync {MOUNT}; [ $rc = 0 ] || {{ echo \"upload exit $rc\" | tee -a {LOG}; exit $rc; }}")
    parts.append(f"python -u - <<'EOF' 2>&1 | tee -a {LOG}\n{VERIFY}\nEOF\n"
                 f"rc=${{PIPESTATUS[0]}}; sync {MOUNT}; echo \"verify exit $rc\" | tee -a {LOG}; sync {MOUNT}; exit $rc")
    sb = modal.Sandbox.create(
        "bash", "-c", "\n".join(parts),
        app=app, image=image, secrets=[secret],
        volumes={MOUNT: volume()},
        cpu=16, memory=32 * 1024,
        env={"SRC": SRC, "REPO": REPO, "BASE": BASE, "IGNORE": ",".join(IGNORE), "REPORT": REPORT,
             "HF_HOME": "/tmp/hf", "GROUP": os.environ.get("GROUP", "4")},
        timeout=24 * 60 * 60,
    )
    print(f"launched sandbox {sb.object_id} ({steps}): {SRC} -> {REPO}")


def status() -> None:
    app = modal.App.lookup(APP, create_if_missing=True)
    sbs = list(modal.Sandbox.list(app_id=app.app_id))
    for sb in sbs:
        rc = sb.poll()
        print(f"sandbox {sb.object_id}: {'running' if rc is None else f'exited {rc}'}")
    try:
        print(b"".join(volume().read_file("logs/hf_upload.log")).decode()[-3000:])
    except Exception as e:
        print(f"(no log yet: {e})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "launch"
    {"launch": lambda: launch("upload+verify"), "verify": lambda: launch("verify"), "status": status}[cmd]()
