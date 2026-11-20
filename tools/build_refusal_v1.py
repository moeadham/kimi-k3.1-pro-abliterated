"""Build and freeze the English refusal-direction dataset v1 (data/refusal_v1/).

Goal: separate the refusal *decision* from "this request looks harmful". The dry run
(docs/refusal_dryrun_analysis.md) found the two directions identical at prompt_end because
harmful prompts were always refused and harmless ones always answered. v1 adds:

  pairs_harmful / pairs_harmless          matched twins (same template or topic), one per id
      OBLITERATUS built-ins (842 pairs, template/topic-matched)
      heretic-org/Semantic-Harmful pairs with embedding similarity >= 0.7
      -> a clean harmfulness direction with phrasing/topic held fixed
  borderline_harmful / borderline_harmless  prompts K3.1 should *sometimes* refuse; run with
      --samples 4 at T=1 and label by judge verdict, not by file. `split` = kind; contrast only
      within a kind (adversarial prompts are ~10x longer than vanilla):
        wjb_vanilla_harmful / wjb_vanilla_benign            length-matched
        wjb_adversarial_harmful / wjb_adversarial_benign    length-matched, <= 300 words
        enriched_{fiction,professional,hypothetical,defensive}  harmful side only
  heldout_wjb_harmful / heldout_wjb_benign  WildJailbreak eval split; never used for directions

Held out (decontaminated against, never sampled): HarmBench standard 200, XSTest safe 200,
data/evals/harmful.jsonl, data/evals/kl_prompts.jsonl, the WildJailbreak eval split.
mlabonne/harmful_behaviors *test* (= data/evals/harmful.jsonl's source) is excluded everywhere.

English only: any character outside ASCII (bar typographic quotes/dashes) drops the prompt.
Decontamination: normalized exact match, plus word-5-gram containment >= 0.5 against held out.

Sources are pinned (HF revision / git commit) and recorded in manifest.json with sha256 per file.
OBLITERATUS prompts.py is parsed with ast.literal_eval (never executed).

Usage: uv run --env-file .env --with pyarrow --with pandas python tools/build_refusal_v1.py [out_dir]
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/refusal_v1"
SEED = 0

REV = {
    "mlabonne/harmful_behaviors": "01cead01398926d81f7c52bdb790ee8cf77ebba7",
    "heretic-org/Semantic-Harmful": "001ca2ceaef94a748235e0ba1366aee48436e286",
    "Bahushruth/abliteration-harmful-enriched": "f29c0b778d3412d5cc7acfc9c6463ce13886212c",
    "allenai/wildjailbreak": "5ddc12a7894f842b0619b8e1c7ee496b198af009",
}
OBLIT_COMMIT = "e414902a68553538df92c9d2a99478ce954f56e9"
OBLIT_URL = f"https://raw.githubusercontent.com/elder-plinius/OBLITERATUS/{OBLIT_COMMIT}/obliteratus/prompts.py"

HELDOUT_FILES = ["results/harmbench_standard/prompts.jsonl", "results/xstest_safe/prompts.jsonl",
                 "data/evals/harmful.jsonl", "data/evals/kl_prompts.jsonl"]

N_VANILLA = 200        # per side
N_ADVERSARIAL = 200    # per side
N_ENRICHED_PER_FRAME = 50
ADV_MAX_WORDS = 300
SEMANTIC_MIN_SCORE = 0.7
CONTAIN_MAX = 0.5

FRAMES = {  # enriched framing buckets, first match wins
    "fiction": r"\b(novel|screenplay|story|fiction|fictional|game|character|scene|roleplay|script|thriller)\b",
    "professional": r"\b(pen ?tester|penetration test|researcher|research|journalist|professor|class|paper|"
                    r"thesis|course|study|studying|academic|red team|security team|training)\b",
    "hypothetical": r"\b(hypothetical(ly)?|thought experiment|what would happen|in theory|theoretically|imagine)\b",
    "defensive": r"\b(protect|prevent|recogni[sz]e|warning signs|so i know|look for|detect|defend|spot)\b",
}
JAILBREAK = r"\b(dan|jailbreak|no restrictions|developer mode|ignore (all |your |previous )+(instructions|rules))\b"

ALLOWED_NON_ASCII = set("‘’“”–—…")


# ---------------------------------------------------------------- helpers
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def shingles(s: str, n: int = 5) -> set:
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def is_english(s: str) -> bool:
    return all(ord(c) < 128 or c in ALLOWED_NON_ASCII for c in s) and bool(re.search(r"[A-Za-z]", s))


def words(s: str) -> int:
    return len(s.split())


class Heldout:
    """Exact (normalized) and 5-gram containment check against every held-out prompt."""

    def __init__(self, texts: list[str]):
        self.exact = {norm(t) for t in texts}
        self.sh = [shingles(t) for t in texts]
        self.index = collections.defaultdict(set)
        for i, s in enumerate(self.sh):
            for g in s:
                self.index[g].add(i)

    def hit(self, t: str) -> bool:
        if norm(t) in self.exact:
            return True
        s = shingles(t)
        cand = collections.Counter(i for g in s for i in self.index.get(g, ()))
        return any(c / min(len(s), len(self.sh[i])) >= CONTAIN_MAX for i, c in cand.items())


def oblit_categories(src: str) -> list[str]:
    """Per-item category of BUILTIN_HARMFUL from the file's section comments (one string per line)."""
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("BUILTIN_HARMFUL"))
    tier, sub, out = "", "", []
    for ln in lines[start + 1:]:
        s = ln.strip()
        if ln.startswith("]"):
            break
        if m := re.match(r"# === (.+?) ===", s):
            tier, sub = m.group(1), ""
        elif m := re.match(r"# -- (.+?) --", s):
            sub = m.group(1)
        elif s.startswith('"'):
            out.append(sub or tier)
    return out


def hf(repo: str, filename: str) -> str:
    return hf_hub_download(repo, filename, repo_type="dataset", revision=REV[repo])


stats: dict = collections.defaultdict(dict)


def keep(name: str, rows: list[dict], text, ho: Heldout, seen: set) -> list[dict]:
    """English filter, held-out decontamination, exact dedup against everything already kept."""
    out, c = [], collections.Counter()
    for r in rows:
        t = text(r)
        if not is_english(t):
            c["non_english"] += 1
        elif ho.hit(t):
            c["heldout_overlap"] += 1
        elif norm(t) in seen:
            c["duplicate"] += 1
        else:
            seen.add(norm(t))
            out.append(r)
    stats[name].update({"candidates": len(rows), **c, "kept_pool": len(out)})
    return out


def length_matched(a: list[dict], b: list[dict], n: int, rng: random.Random, width: int):
    """Sample n from each side with the same per-bin word-length counts (bins of `width` words)."""
    ba, bb = collections.defaultdict(list), collections.defaultdict(list)
    for r in a:
        ba[words(r["prompt"]) // width].append(r)
    for r in b:
        bb[words(r["prompt"]) // width].append(r)
    cap = {k: min(len(ba[k]), len(bb[k])) for k in ba}
    total = sum(cap.values())
    if total < n:
        raise SystemExit(f"length matching: only {total} matchable rows for n={n}")
    # allocate n across bins proportionally to side a's distribution, capped by availability
    want = {k: int(n * len(ba[k]) / len(a)) for k in ba}
    want = {k: min(v, cap[k]) for k, v in want.items()}
    while sum(want.values()) < n:
        k = max((k for k in cap if want[k] < cap[k]), key=lambda k: cap[k] - want[k])
        want[k] += 1
    sa, sb = [], []
    for k in sorted(want):
        sa += rng.sample(ba[k], want[k])
        sb += rng.sample(bb[k], want[k])
    return sa, sb


def write(name: str, rows: list[dict], sources: list[str], note: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    data = p.read_bytes()
    by_kind = collections.Counter(r.get("split") for r in rows)
    wl = sorted(words(r["prompt"]) for r in rows)
    manifest["files"][name] = {
        "n": len(rows), "file": p.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "by_split": dict(sorted(by_kind.items())), "words_median": wl[len(wl) // 2], "sources": sources,
        **({"note": note} if note else {})}
    print(f"{name:22} n={len(rows):5}  {dict(by_kind)}")


manifest: dict = {"seed": SEED, "files": {}, "filters": {}, "heldout": HELDOUT_FILES + ["allenai/wildjailbreak eval"],
                  "licenses": {"OBLITERATUS": "AGPL-3.0", "heretic-org/Semantic-Harmful": "CC-BY-4.0",
                               "mlabonne/harmless_alpaca (Semantic twins)": "see source",
                               "Bahushruth/abliteration-harmful-enriched": "Apache-2.0",
                               "allenai/wildjailbreak": "ODC-BY (gated)"}}


# ---------------------------------------------------------------- main
def main():
    rng = random.Random(SEED)

    # held-out texts
    held = []
    for f in HELDOUT_FILES:
        for line in open(ROOT / f):
            x = json.loads(line)
            held.append(x.get("question") or x.get("prompt") or x.get("instruction"))
    wjb_eval = pd.read_csv(hf("allenai/wildjailbreak", "eval/eval.tsv"), sep="\t", keep_default_na=False)
    held += list(wjb_eval.adversarial)
    mlab_test = pd.read_parquet(hf("mlabonne/harmful_behaviors", "data/test-00000-of-00001.parquet"))
    held += list(mlab_test.text)
    ho = Heldout(held)
    seen: set = set()

    # ---- held-out WildJailbreak eval (only English filter + XSTest/HarmBench decontam)
    ho_small = Heldout([h for h in held[: -len(wjb_eval) - len(mlab_test)]])
    for dtype, name in [("adversarial_harmful", "heldout_wjb_harmful"), ("adversarial_benign", "heldout_wjb_benign")]:
        rows = [{"id": f"wjbe-{i}", "split": dtype, "prompt": t}
                for i, t in zip(wjb_eval.index, wjb_eval.adversarial) if wjb_eval.data_type[i] == dtype]
        rows = keep(name, rows, lambda r: r["prompt"], ho_small, set())
        write(name, rows, [f"allenai/wildjailbreak@{REV['allenai/wildjailbreak']}:eval"],
              "never use for direction finding")

    # ---- matched pairs
    src = urllib.request.urlopen(OBLIT_URL, timeout=60).read().decode()
    lists = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in ("BUILTIN_HARMFUL", "BUILTIN_HARMLESS")}
    H, L = lists["BUILTIN_HARMFUL"], lists["BUILTIN_HARMLESS"]
    assert len(H) == len(L) == 842, (len(H), len(L))
    cats = oblit_categories(src)
    assert len(cats) == len(H), (len(cats), len(H))
    pairs = [{"pid": f"ob-{i:04d}", "split": "obliteratus", "category": cats[i], "h": h, "l": l}
             for i, (h, l) in enumerate(zip(H, L))]
    sem = json.load(open(hf("heretic-org/Semantic-Harmful", "metadata/matched_pairs.json")))["pairs"]
    stats["semantic_pairs"]["below_min_score"] = sum(p["score"] < SEMANTIC_MIN_SCORE for p in sem)
    for p in sem:
        if p["score"] >= SEMANTIC_MIN_SCORE:
            pairs.append({"pid": f"sem-{p['harmful_index']:04d}", "split": "semantic", "h": p["harmful"],
                          "l": p["harmless"], "score": round(p["score"], 4)})
    # a pair survives only if both sides pass
    c = collections.Counter()
    kept = []
    for p in pairs:
        bad = next((why for t in (p["h"], p["l"]) for why, f in
                    (("non_english", lambda t: not is_english(t)), ("heldout_overlap", ho.hit),
                     ("duplicate", lambda t: norm(t) in seen)) if f(t)), None)
        if bad:
            c[bad] += 1
            continue
        seen.update({norm(p["h"]), norm(p["l"])})
        kept.append(p)
    stats["pairs"].update({"candidates": len(pairs), **c, "kept": len(kept)})
    rng.shuffle(kept)  # `--n-per-class N` takes the first N: make any prefix a representative mix
    extra = lambda p: {k: p[k] for k in ("category", "score") if k in p}  # noqa: E731
    srcs = [f"OBLITERATUS@{OBLIT_COMMIT}:obliteratus/prompts.py",
            f"heretic-org/Semantic-Harmful@{REV['heretic-org/Semantic-Harmful']}:metadata/matched_pairs.json"]
    write("pairs_harmful", [{"id": p["pid"], "pair": p["pid"], "split": p["split"], "prompt": p["h"], **extra(p)}
                            for p in kept], srcs)
    write("pairs_harmless", [{"id": p["pid"], "pair": p["pid"], "split": p["split"], "prompt": p["l"], **extra(p)}
                             for p in kept], srcs)

    # ---- WildJailbreak train: vanilla + adversarial, length-matched within kind
    tr = pd.read_csv(hf("allenai/wildjailbreak", "train/train.tsv"), sep="\t", keep_default_na=False,
                     usecols=["vanilla", "adversarial", "data_type"])
    tr["prompt"] = [a if a else v for a, v in zip(tr.adversarial, tr.vanilla)]
    pool = {}
    for dtype in ["vanilla_harmful", "vanilla_benign", "adversarial_harmful", "adversarial_benign"]:
        g = tr[tr.data_type == dtype]
        if dtype.startswith("adversarial"):
            g = g[g.prompt.str.split().str.len() <= ADV_MAX_WORDS]
        idx = list(g.index)
        rng.shuffle(idx)
        idx = idx[:20000]  # plenty for matching; keeps decontamination cheap
        rows = [{"id": f"wjb-{i}", "split": f"wjb_{dtype}", "prompt": tr.prompt[i].strip()} for i in idx]
        pool[dtype] = keep(f"wjb_{dtype}", rows, lambda r: r["prompt"], ho, seen)
    vh, vb = length_matched(pool["vanilla_harmful"], pool["vanilla_benign"], N_VANILLA, rng, width=3)
    ah, ab = length_matched(pool["adversarial_harmful"], pool["adversarial_benign"], N_ADVERSARIAL, rng, width=20)

    # ---- enriched: framed (softer) styles, harmful side only
    enr = pd.concat([pd.read_parquet(hf("Bahushruth/abliteration-harmful-enriched", f"data/{s}-00000-of-00001.parquet"))
                     for s in ("train", "test")], ignore_index=True)
    mlab_train = pd.read_parquet(hf("mlabonne/harmful_behaviors", "data/train-00000-of-00001.parquet"))
    mlab = {norm(t) for t in list(mlab_train.text) + list(mlab_test.text)}
    erows, ec = [], collections.Counter()
    for i, t in enumerate(enr.text):
        t = t.strip()
        if norm(t) in mlab:
            ec["mlabonne_original"] += 1
            continue
        if re.search(JAILBREAK, t, re.I):
            ec["jailbreak_style"] += 1
            continue
        frame = next((k for k, rx in FRAMES.items() if re.search(rx, t, re.I)), None)
        if frame is None:
            ec["no_frame"] += 1
            continue
        erows.append({"id": f"enr-{i}", "split": f"enriched_{frame}", "prompt": t})
    stats["enriched"].update(ec)
    erows = keep("enriched", erows, lambda r: r["prompt"], ho, set(seen))
    rng.shuffle(erows)
    eh, prefix = [], collections.Counter()
    by_frame = collections.Counter()
    for r in erows:  # cap 2 per opening 6 words, N per frame
        pre = " ".join(norm(r["prompt"]).split()[:6])
        if prefix[pre] >= 2 or by_frame[r["split"]] >= N_ENRICHED_PER_FRAME or norm(r["prompt"]) in seen:
            continue
        prefix[pre] += 1
        by_frame[r["split"]] += 1
        seen.add(norm(r["prompt"]))
        eh.append(r)

    wsrc = f"allenai/wildjailbreak@{REV['allenai/wildjailbreak']}:train"
    esrc = f"Bahushruth/abliteration-harmful-enriched@{REV['Bahushruth/abliteration-harmful-enriched']}"
    bh, bl = vh + ah + eh, vb + ab
    rng.shuffle(bh)
    rng.shuffle(bl)
    write("borderline_harmful", bh, [wsrc, esrc], "run with --samples 4; label by judge; contrast within kind")
    write("borderline_harmless", bl, [wsrc], "run with --samples 4; label by judge; contrast within kind")

    manifest["filters"] = {k: dict(v) for k, v in stats.items()}
    manifest["params"] = {"semantic_min_score": SEMANTIC_MIN_SCORE, "adv_max_words": ADV_MAX_WORDS,
                          "heldout_containment_max": CONTAIN_MAX, "n_vanilla": N_VANILLA,
                          "n_adversarial": N_ADVERSARIAL, "n_enriched_per_frame": N_ENRICHED_PER_FRAME}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["filters"], indent=1))


if __name__ == "__main__":
    main()
