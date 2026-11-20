"""Recovery capture: activations of branch continuations from the `forks` branch test (live driver, after capture).

For answer positions where substituting a token either let the model RECOVER (same outcome as the original) or
FLIPPED the outcome, capture matched sequences that share every token up to the swap:
  original   the greedy answer itself
  recovered  a substitution whose continuation kept the original outcome (e.g. "I can help you ... but I can't")
  flipped    a substitution whose continuation changed it (e.g. "Subject: Classified briefing ...")
Positions: the state just before the swap (identical across the three: a sanity check) through `after` states after
it, every `stride`-th stream index (+ final). Differences after the swap are the model's own push back toward (or
away from) refusal: the cleanest contrast for checking a candidate refusal vector without editing.

Branch continuations were saved as text (`branches.jsonl`, the whole answer after the substitution, <= 600 chars):
the tail after the substituted token is re-tokenized, which can differ from the generated ids at a token boundary.

  modal_gpu.py put tools/recovery_capture.py /root/harness/tools/recovery_capture.py
  modal_gpu.py live "py /root/harness/tools/recovery_capture.py"     (env K3_RECOVERY_DIR, default /vol/session_nt/forks)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

FORKS = Path(os.environ.get("K3_RECOVERY_DIR", "/vol/session_nt/forks"))
MAX_POSITIONS = int(os.environ.get("K3_RECOVERY_MAX", "600"))
AFTER, STRIDE, N_LAYERS = 32, 4, 93


def plan(forks_dir: Path, max_positions: int) -> list[dict]:
    """Matched sequences: [{key, i, kind, alt, flagged, original_flagged}] grouped by (key, i)."""
    judged = {x["key"]: x for x in map(json.loads, open(forks_dir / "judged.jsonl"))}
    by_pos: dict = {}
    for b in map(json.loads, open(forks_dir / "branches.jsonl")):
        if b.get("flagged") is None or b["key"] not in judged:
            continue
        by_pos.setdefault((b["key"], b["i"]), []).append(b)
    out, n = [], 0
    # positions with both a recovered and a flipped substitution first (full triplets), then flip-only ones
    groups = []
    for (key, i), bs in by_pos.items():
        orig = bool(judged[key]["flagged"])
        rec = [b for b in bs if bool(b["flagged"]) == orig]
        flp = [b for b in bs if bool(b["flagged"]) != orig]
        if flp:
            groups.append((0 if rec else 1, key, i, orig, rec, flp))
    groups.sort(key=lambda g: (g[0], g[1], g[2]))
    for _, key, i, orig, rec, flp in groups[:max_positions]:
        out.append({"key": key, "i": i, "kind": "original", "alt": None, "flagged": orig, "original_flagged": orig})
        for b in rec[:1]:
            out.append({"key": key, "i": i, "kind": "recovered", "alt": b["alt"], "flagged": orig,
                        "original_flagged": orig, "text": b["text"]})
        for b in flp[:1]:
            out.append({"key": key, "i": i, "kind": "flipped", "alt": b["alt"], "flagged": not orig,
                        "original_flagged": orig, "text": b["text"]})
        n += 1
    return out


def sequence(tok, m, gen: dict, row: dict) -> list[int]:
    """Prompt + answer up to the swap + the substituted token + the re-tokenized continuation."""
    from harness.k3chat import parse

    P, out = list(gen["prompt_ids"]), list(gen["out_ids"])
    i = row["i"]
    if row["kind"] == "original":
        return P + parse(out, m, thinking=False).answer_ids
    head = tok.decode(out[:i] + [row["alt"]])
    text = row["text"]
    tail = text[len(head):] if text.startswith(head) else text[len(head):]  # best effort at odd boundaries
    return P + out[:i] + [row["alt"]] + tok.encode(tail, allow_special_tokens=False)


def run(llm, tok, log, capture_one=None, forks_dir: Path = FORKS, max_positions: int = MAX_POSITIONS) -> dict:
    from harness import tp
    from harness.k3chat import Markers
    from harness.session import _file_positions

    capture_one = capture_one or tp.capture_one
    m = Markers(tok)
    gens = {x["item_id"] + "#" + str(x["sample"]): x for x in map(json.loads, open(forks_dir / "gens_raw.jsonl"))}
    rows = plan(forks_dir, max_positions)
    d_out = forks_dir.parent / "recovery"
    (d_out / "states").mkdir(parents=True, exist_ok=True)
    S = 2 * N_LAYERS + 1
    streams = sorted(set(range(0, S, STRIDE)) | {S - 1})
    index, streak, t0 = [], 0, time.time()
    log(f"[recovery] {len(rows)} sequences over {len({(r['key'], r['i']) for r in rows})} positions")
    try:
        for n, r in enumerate(rows):
            g = gens[r["key"]]
            seq = sequence(tok, m, g, r)
            swap_state = len(g["prompt_ids"]) + r["i"] - 1
            positions = list(range(max(swap_state, 0), min(swap_state + AFTER + 1, len(seq) - 1)))
            if not positions:
                continue
            path = d_out / "states" / f"{n:05d}.safetensors"
            try:
                have = _file_positions(path, None) if path.exists() else None
                if have is None or list(have) != positions:
                    capture_one(llm, seq[: positions[-1] + 1], positions, str(path), streams=streams)
                streak = 0
            except Exception as e:  # noqa: BLE001
                log(f"[recovery] capture FAILED {r['key']} i={r['i']} {r['kind']}: {e!r}")
                streak += 1
                if streak >= 3:
                    log("[recovery] 3 capture failures in a row: stopping (files so far are kept)")
                    break
                continue
            index.append({**{k: v for k, v in r.items() if k != "text"}, "file": path.name, "positions": positions,
                          "streams": streams, "swap_state": swap_state, "prompt_len": len(g["prompt_ids"])})
            if n % 200 == 0:
                log(f"[recovery] captured {n}/{len(rows)} ({time.time() - t0:.0f}s)")
    finally:
        with open(d_out / "index.jsonl", "w") as f:
            for x in index:
                f.write(json.dumps(x) + "\n")
    res = {"sequences": len(index), "planned": len(rows), "seconds": round(time.time() - t0)}
    log(f"[recovery] done {json.dumps(res)}")
    return res


if __name__ == "__live__":
    run(llm, tok, log)  # noqa: F821
