"""Download and freeze the step-2 evaluation sets (docs/plan_step2.md, "Capability checks").

Each set is sampled deterministically (seed 0), normalized to one JSONL schema per task type,
and recorded in data/evals/manifest.json with source repo, revision, size and sha256.

  kl_prompts        harmless prompts for whole-response KL (mlabonne/harmless_alpaca test EN
                    + shibing624/alpaca-zh ZH), 100 + 100
  harmful           harmful requests for refusal retention: walledai/HarmBench standard if
                    accessible (gated), else mlabonne/harmful_behaviors test
  mmlu_pro          TIGER-Lab/MMLU-Pro test, 500 stratified by category
  gpqa_diamond      Idavidrein/gpqa diamond (gated; skipped with a notice if not accepted)
  math500           HuggingFaceH4/MATH-500 (all 500)
  ifeval            google/IFEval (all 541)
  livecodebench     livecodebench/code_generation_lite test6.jsonl, 150 newest stdin problems
  cmmlu             haonan-li/cmmlu test, 500 stratified by subject
  bfcl              gorilla-llm/Berkeley-Function-Calling-Leaderboard v3: 20 simple + 15 multiple
                    + 15 parallel (tool-calling sanity check; harness/bfcl.py)

LiveCodeBench private tests are base64(zlib(pickle)). Downloaded data is untrusted, so they
are unpickled with a restricted Unpickler that refuses every global (only plain str/list/dict
data can load; anything that would import or call code raises).

Usage: uv run --env-file .env --extra tok python tools/fetch_evals.py [out_dir] [only,sets]
"""

from __future__ import annotations

import base64
import collections
import csv
import hashlib
import io
import json
import pickle
import random
import sys
import zipfile
import zlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/evals")
SEED = 0
manifest: dict = {}


def dl(repo, filename):
    path = hf_hub_download(repo, filename, repo_type="dataset")
    rev = HfApi().dataset_info(repo).sha
    return path, rev


def parquet_rows(path):
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def stratified(rows, key, n, seed=SEED):
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    keys = sorted(groups)
    per = n // len(keys)
    out = []
    for k in keys:
        g = groups[k][:]
        rng.shuffle(g)
        out.extend(g[:per])
    rest = [r for k in keys for r in groups[k] if r not in out]
    rng.shuffle(rest)
    return out + rest[: n - len(out)]


def write(name, rows, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    data = p.read_bytes()
    manifest[name] = {"n": len(rows), "file": p.name, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(), "sources": sources, "seed": SEED}
    print(f"{name:14} n={len(rows):4}  {len(data) / 1e6:6.1f} MB  {sources}")


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):  # noqa: D102
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def _lcb_tests(s):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001 - compressed form
        obj = _NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load()
        return json.loads(obj) if isinstance(obj, str) else obj


def kl_prompts():
    p1, r1 = dl("mlabonne/harmless_alpaca", "data/test-00000-of-00001.parquet")
    en = [r["text"] for r in parquet_rows(p1)][:100]
    p2, r2 = dl("shibing624/alpaca-zh", "alpaca_gpt4_data_zh.json")
    zh_all = [x for x in json.load(open(p2)) if not x.get("input")]
    random.Random(SEED).shuffle(zh_all)
    zh = [x["instruction"] for x in zh_all[:100]]
    rows = [{"id": f"kl-en-{i}", "lang": "en", "prompt": t} for i, t in enumerate(en)]
    rows += [{"id": f"kl-zh-{i}", "lang": "zh", "prompt": t} for i, t in enumerate(zh)]
    write("kl_prompts", rows, [f"mlabonne/harmless_alpaca@{r1}", f"shibing624/alpaca-zh@{r2}"])


def harmful():
    try:
        p, r = dl("walledai/HarmBench", "standard/train-00000-of-00001.parquet")
        rows = [{"id": f"hb-{i}", "lang": "en", "prompt": x["prompt"], "category": x.get("category")}
                for i, x in enumerate(parquet_rows(p))]
        write("harmful", rows, [f"walledai/HarmBench/standard@{r}"])
    except Exception as e:  # noqa: BLE001
        print(f"  HarmBench unavailable ({type(e).__name__}); using mlabonne/harmful_behaviors")
        p, r = dl("mlabonne/harmful_behaviors", "data/test-00000-of-00001.parquet")
        rows = [{"id": f"hbv-{i}", "lang": "en", "prompt": x["text"]} for i, x in enumerate(parquet_rows(p))]
        write("harmful", rows, [f"mlabonne/harmful_behaviors@{r}"])


def mmlu_pro():
    p, r = dl("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet")
    rows = stratified(parquet_rows(p), lambda x: x["category"], 500)
    out = [{"id": f"mmlupro-{x['question_id']}", "lang": "en", "subject": x["category"], "question": x["question"],
            "choices": list(x["options"]), "answer": x["answer_index"]} for x in rows]
    write("mmlu_pro", out, [f"TIGER-Lab/MMLU-Pro@{r}"])


def gpqa():
    try:
        p, r = dl("Idavidrein/gpqa", "gpqa_diamond.csv")
    except Exception as e:  # noqa: BLE001
        print(f"  GPQA-Diamond skipped ({type(e).__name__}): accept the terms at "
              "https://huggingface.co/datasets/Idavidrein/gpqa with the HF_TOKEN account, then rerun")
        return
    rng = random.Random(SEED)
    out = []
    for i, x in enumerate(csv.DictReader(open(p))):
        ch = [x["Correct Answer"], x["Incorrect Answer 1"], x["Incorrect Answer 2"], x["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        out.append({"id": f"gpqa-{i}", "lang": "en", "subject": x.get("High-level domain"), "question": x["Question"],
                    "choices": [ch[j].strip() for j in order], "answer": order.index(0)})
    write("gpqa_diamond", out, [f"Idavidrein/gpqa@{r}"])


def math500():
    p, r = dl("HuggingFaceH4/MATH-500", "test.jsonl")
    out = [{"id": x["unique_id"], "lang": "en", "problem": x["problem"], "answer": x["answer"],
            "subject": x["subject"], "level": x["level"]} for x in map(json.loads, open(p))]
    write("math500", out, [f"HuggingFaceH4/MATH-500@{r}"])


def ifeval():
    p, r = dl("google/IFEval", "ifeval_input_data.jsonl")
    out = [{"id": f"ifeval-{x['key']}", "lang": "en", **x} for x in map(json.loads, open(p))]
    write("ifeval", out, [f"google/IFEval@{r}"])


def livecodebench():
    p, r = dl("livecodebench/code_generation_lite", "test6.jsonl")
    rows = []
    for x in map(json.loads, open(p)):
        pub = _lcb_tests(x["public_test_cases"])
        if not pub or any(t.get("testtype") != "stdin" for t in pub):
            continue
        priv = _lcb_tests(x["private_test_cases"]) if x.get("private_test_cases") else []
        rows.append({"id": f"lcb-{x['question_id']}", "lang": "en", "title": x["question_title"],
                     "question": x["question_content"], "platform": x["platform"], "date": x["contest_date"],
                     "difficulty": x["difficulty"], "public_tests": pub, "private_tests": priv,
                     "time_limit_s": 6})
    rows.sort(key=lambda x: x["date"], reverse=True)
    write("livecodebench", rows[:150], [f"livecodebench/code_generation_lite/test6.jsonl@{r}"])


def cmmlu():
    p, r = dl("haonan-li/cmmlu", "cmmlu_v1_0_1.zip")
    rows = []
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():
            if not (name.startswith("test/") and name.endswith(".csv")):
                continue
            subj = Path(name).stem
            for i, x in enumerate(csv.DictReader(io.TextIOWrapper(z.open(name), "utf-8"))):
                rows.append({"id": f"cmmlu-{subj}-{i}", "lang": "zh", "subject": subj, "question": x["Question"],
                             "choices": [x["A"], x["B"], x["C"], x["D"]], "answer": "ABCD".index(x["Answer"].strip())})
    write("cmmlu", stratified(rows, lambda x: x["subject"], 500), [f"haonan-li/cmmlu@{r}"])


def bfcl():
    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    rng = random.Random(SEED)
    out, rev = [], None
    for cat, n in (("simple", 20), ("multiple", 15), ("parallel", 15)):
        q, rev = dl(repo, f"BFCL_v3_{cat}.json")
        a, _ = dl(repo, f"possible_answer/BFCL_v3_{cat}.json")
        answers = {x["id"]: x["ground_truth"] for x in map(json.loads, open(a))}
        items = [x for x in map(json.loads, open(q)) if x["id"] in answers]
        for x in rng.sample(items, n):
            out.append({"id": f"bfcl-{x['id']}", "lang": "en", "category": cat, "messages": x["question"][0],
                        "functions": x["function"], "ground_truth": answers[x["id"]]})
    write("bfcl", out, [f"{repo}@{rev}"])


ALL = (kl_prompts, harmful, mmlu_pro, gpqa, math500, ifeval, livecodebench, cmmlu, bfcl)

if __name__ == "__main__":
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    for fn in ALL:
        if only is None or fn.__name__ in only:
            fn()
    old = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else {}
    (OUT / "manifest.json").write_text(json.dumps({**old, **manifest}, indent=1, ensure_ascii=False))
