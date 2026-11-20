"""Build the step-2 replay set: hosted transcripts -> exact K3.1 token IDs + capture positions
+ behaviour-based contrast labels + train/val split.

Contrast labels come from Jev behaviour (`censored`), not from the dataset split: deccp's
"uncensored" questions are censored 28-41% of the time by K3.1 (docs/hosted_deccp_baseline.md).
`mixed` marks prompts whose samples disagree (same prompt, different behaviour), the
cleanest contrast. Split is by prompt (all samples of a prompt go to the same side).

Usage: uv run --extra tok python tools/build_replay_set.py [judged.jsonl ... out.jsonl]
(default: hosted deccp + hosted controls -> data/step2/replay_set.jsonl)
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.replay import build, load_tokenizer  # noqa: E402

CJK = re.compile(r"[一-鿿]")
VAL_FRACTION = 0.2


def _h(prompt_id: str) -> int:
    return int(hashlib.sha256(prompt_id.encode()).hexdigest()[:8], 16)


def folds(rows) -> dict:
    """Prompt-level split, stratified by (dataset split, language): within each stratum the
    VAL_FRACTION of prompts with the smallest hash go to val (deterministic)."""
    strata = collections.defaultdict(set)
    for r in rows:
        strata[(r["split"], "zh" if CJK.search(r["question"]) else "en")].add(r["prompt_id"])
    out = {}
    for ids in strata.values():
        ids = sorted(ids, key=_h)
        n_val = max(1, round(len(ids) * VAL_FRACTION))
        for i, pid in enumerate(ids):
            out[pid] = "val" if i < n_val else "train"
    return out


def main():
    srcs = sys.argv[1:-1] if len(sys.argv) > 2 else ["results/hosted_deccp/judged.jsonl", "results/hosted_controls/judged.jsonl"]
    out = Path(sys.argv[-1] if len(sys.argv) > 2 else "data/step2/replay_set.jsonl")
    rows = [json.loads(l) for src in srcs for l in open(src)]
    fold_of = folds(rows)
    tok = load_tokenizer()
    by_prompt = collections.defaultdict(set)
    for r in rows:
        by_prompt[r["prompt_id"]].add(r["censored"])
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    with open(out, "w") as f:
        for r in rows:
            effort = r["effort"] if r["effort"] in ("low", "high", "max") else "max"
            rep = build(tok, r["question"], r["reasoning"], r["content"], effort=effort)
            onset = r.get("onset") if r["censored"] else None
            onset_chars = None
            if onset and onset.get("onset_sentence") is not None:
                onset_chars = tuple(onset["spans"][onset["onset_sentence"]])
            pos = rep.positions(onset_chars=onset_chars)
            row = {
                "key": r["key"], "prompt_id": r["prompt_id"], "dataset_split": r["split"],
                "lang": "zh" if CJK.search(r["question"]) else "en",
                "provider": r["provider"], "effort": effort, "temperature": r["temperature"],
                "censored": r["censored"], "mixed_prompt": len(by_prompt[r["prompt_id"]]) == 2,
                "fold": fold_of[r["prompt_id"]],
                "verdict": {k: v for k, v in r["verdict"].items() if not k.endswith("_probs")},
                "ids": rep.ids, "prompt_len": rep.prompt_len, "reasoning_span": rep.reasoning,
                "answer_span": rep.answer, "positions": pos,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats[(row["fold"], row["lang"], "censored" if row["censored"] else "factual")] += 1
            stats["mixed_rows"] += row["mixed_prompt"]
            stats["with_onset"] += "onset" in pos
            stats["tokens"] += len(rep.ids)
    for k, v in sorted(stats.items(), key=str):
        print(k, v)


if __name__ == "__main__":
    main()
