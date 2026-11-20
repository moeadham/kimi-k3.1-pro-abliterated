"""Compare two sanity-panel results and print PASS/FAIL (exit code 0 = pass, 1 = fail).

Accepts either format:
  - engine results (session stage `sanity`: OUT/sanity/base.json, edited.json)
  - hosted summaries (tools/sanity_panel_hosted.py: results/.../summary.json; the "sets" block is used)

Usage: python tools/compare_sanity.py BASE.json CANDIDATE.json [--max-accuracy-drop 0.05]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.sanity import THRESHOLDS, compare  # noqa: E402


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    return d["sets"] if "sets" in d else d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("candidate")
    for k, v in THRESHOLDS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    a = ap.parse_args(argv)
    t = {k: getattr(a, k) for k in THRESHOLDS}
    r = compare(load(a.base), load(a.candidate), t)
    for c in r["checks"]:
        print(f"{'ok  ' if c['pass'] else 'FAIL'}  {c['check']:<34} {c['value']:>8}  (limit {c['limit']})")
    print("PASS" if r["pass"] else "FAIL")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
