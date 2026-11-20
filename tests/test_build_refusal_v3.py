"""CPU checks on the frozen run-3 corpus (data/refusal_v3/, built by tools/build_refusal_v3.py). No network.

Held-out overlap is re-checked independently of the builder (same rule: normalized exact match or word-5-gram
containment >= 0.5) against every held-out file in the repo.
"""
import ast
import collections
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data/refusal_v3"
FILES = ("build_harmful", "build_harmless", "reference_harmless")
HELDOUT = ["results/harmbench_standard/prompts.jsonl", "results/xstest_safe/prompts.jsonl",
           "data/refusal_v1/heldout_wjb_harmful.jsonl", "data/refusal_v1/heldout_wjb_benign.jsonl",
           *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data/evals").glob("*.jsonl"))]
ALLOWED_NON_ASCII = set("‘’“”–—…")

pytestmark = pytest.mark.skipif(not (V3 / "manifest.json").exists(), reason="data/refusal_v3 not built")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def shingles(s, n=5):
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def rows(name):
    return [json.loads(ln) for ln in open(V3 / f"{name}.jsonl")]


def heldout_texts():
    out = []
    for f in HELDOUT:
        p = ROOT / f
        if not p.exists():
            continue
        for ln in open(p):
            x = json.loads(ln)
            if "messages" in x:
                msgs = x["messages"] if isinstance(x["messages"], list) else ast.literal_eval(x["messages"])
                out += [m["content"] for m in msgs if m.get("role") == "user"]
            else:
                out.append(x.get("question") or x.get("prompt") or x.get("problem") or x.get("instruction") or "")
    return [t for t in out if t]


@pytest.fixture(scope="module")
def data():
    return {n: rows(n) for n in FILES}


def test_manifest_matches_files(data):
    m = json.loads((V3 / "manifest.json").read_text())
    assert m["seed"] == 0
    for n in FILES:
        f = m["files"][n]
        assert hashlib.sha256((V3 / f["file"]).read_bytes()).hexdigest() == f["sha256"]
        assert f["n"] == len(data[n])
        assert f["by_kind"] == dict(sorted(collections.Counter(r["kind"] for r in data[n]).items()))


def test_fields_ids_and_english(data):
    for n, rs in data.items():
        assert len({r["id"] for r in rs}) == len(rs), n
        for r in rs:
            assert {"id", "prompt", "split", "kind", "source"} <= r.keys(), (n, r["id"])
            assert r["split"] == ("harmful" if n == "build_harmful" else "harmless")
            assert r["prompt"].strip() and all(ord(c) < 128 or c in ALLOWED_NON_ASCII for c in r["prompt"]), r["id"]


def test_build_files_are_aligned_and_sized(data):
    h, l = data["build_harmful"], data["build_harmless"]
    assert len(h) == len(l) >= 4000
    for a, b in zip(h, l):
        assert a["kind"].replace("_harmful", "") == b["kind"].replace("_benign", "")
        assert a.get("pair") == b.get("pair")
        if a["kind"].startswith(("residual_", "enriched_", "v1_pairs_")):
            assert a["pair"]
    k = collections.Counter(r["kind"] for r in h)
    assert k["wjb_adversarial_harmful"] == 1000 and k["wjb_vanilla_harmful"] == 500
    assert sum(v for kk, v in k.items() if kk.startswith("residual_")) == 500
    assert sum(v for kk, v in k.items() if kk.startswith("enriched_")) == 1000
    assert sum(v for kk, v in k.items() if kk.startswith("v1_pairs_")) >= 1050
    assert len(data["reference_harmless"]) == 2000


def test_v1_pairs_kept_as_is(data):
    v1 = {r["id"]: r for r in map(json.loads, open(ROOT / "data/refusal_v1/pairs_harmful.jsonl"))}
    for r in data["build_harmful"]:
        if r["kind"].startswith("v1_pairs_"):
            assert v1[r["id"]]["prompt"] == r["prompt"] and v1[r["id"]]["pair"] == r["pair"]


def test_no_duplicates_across_outputs_or_v1_borderline(data):
    seen = collections.Counter(norm(r["prompt"]) for rs in data.values() for r in rs)
    assert max(seen.values()) == 1
    border = {norm(json.loads(ln)["prompt"]) for f in ("borderline_harmful", "borderline_harmless")
              for ln in open(ROOT / f"data/refusal_v1/{f}.jsonl")}
    assert not border & set(seen)


def test_no_heldout_overlap(data):
    held = heldout_texts()
    assert len(held) > 4000
    exact = {norm(t) for t in held}
    sh = [shingles(t) for t in held]
    index = collections.defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            index[g].append(i)
    bad = []
    for rs in data.values():
        for r in rs:
            if norm(r["prompt"]) in exact:
                bad.append(r["id"])
                continue
            s = shingles(r["prompt"])
            cand = collections.Counter(i for g in s for i in index.get(g, ()))
            if any(c / min(len(s), len(sh[i])) >= 0.5 for i, c in cand.items()):
                bad.append(r["id"])
    assert not bad, bad[:10]
