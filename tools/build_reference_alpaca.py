"""Run 3 projection reference: 2,000 ordinary instructions from mlabonne/harmless_alpaca (train split; the KL
prompts come from its test split and are excluded by text too). Replaces the WildJailbreak-benign stopgap
(data/refusal_v3/reference_harmless.jsonl): plain requests, so projecting their mean out of H removes what any chat
prompt shares, not something specific to jailbreak-style benign prompts.

  uv run --env-file ../../../.env python tools/build_reference_alpaca.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_evals import dl, parquet_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N, SEED = 2000, 0

if __name__ == "__main__":
    path, rev = dl("mlabonne/harmless_alpaca", "data/train-00000-of-00001.parquet")
    kl = {json.loads(l)["prompt"].strip() for l in open(ROOT / "data/evals/kl_prompts.jsonl")}
    texts = sorted({r["text"].strip() for r in parquet_rows(path)} - kl)
    random.Random(SEED).shuffle(texts)
    out = ROOT / "data/refusal_v3/reference_alpaca.jsonl"
    with open(out, "w") as f:
        for i, t in enumerate(texts[:N]):
            f.write(json.dumps({"id": f"alp-{i}", "prompt": t, "split": "harmless", "kind": "reference_alpaca",
                                "category": "general", "source": f"mlabonne/harmless_alpaca@{rev}:train"},
                               ensure_ascii=False) + "\n")
    print(f"{min(N, len(texts))} of {len(texts)} -> {out}")
