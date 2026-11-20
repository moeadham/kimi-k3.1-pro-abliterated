"""Benchmark prompt builders and scorers (step 2 capability checks). Pure Python; the code
runner executes model-written programs in a subprocess with limits."""

from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile

LETTERS = "ABCDEFGHIJ"

# ---- multiple choice (MMLU-Pro, GPQA, CMMLU) ----

MC_EN = ("Answer the following multiple-choice question. Think it through, then give the final answer "
         "on the last line in the form \"Answer: X\" where X is the letter.\n\n{q}\n\n{opts}")
MC_ZH = ("回答下面的选择题。请先思考，然后在最后一行以“答案：X”的格式给出最终答案，X 为选项字母。\n\n{q}\n\n{opts}")


def mc_prompt(item: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return (MC_ZH if item.get("lang") == "zh" else MC_EN).format(q=item["question"].strip(), opts=opts)


_MC_PATTERNS = [
    re.compile(r"(?:Answer|ANSWER|answer)\s*[:：]\s*\**\s*\(?([A-J])\b"),
    re.compile(r"答案\s*[:：是为]?\s*\**\s*\(?([A-J])\b"),
    re.compile(r"\\boxed\{\s*([A-J])\s*\}"),
]


def mc_extract(answer: str, n_choices: int) -> int | None:
    for pat in _MC_PATTERNS:
        hits = pat.findall(answer)
        if hits:
            i = LETTERS.index(hits[-1])
            return i if i < n_choices else None
    m = re.findall(r"^\s*\(?([A-J])[\).]?\s*$", answer.strip().splitlines()[-1] if answer.strip() else "")
    return LETTERS.index(m[-1]) if m and LETTERS.index(m[-1]) < n_choices else None


def mc_score(item: dict, answer: str) -> dict:
    pred = mc_extract(answer, len(item["choices"]))
    return {"pred": pred, "correct": pred == item["answer"]}


# ---- math (MATH-500) ----

MATH_PROMPT = "Solve the following problem. Put the final answer in \\boxed{{}}.\n\n{p}"


def math_prompt(item: dict) -> str:
    return MATH_PROMPT.format(p=item["problem"])


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def _norm_math(s: str) -> str:
    s = s.strip().strip("$").replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = re.sub(r"\\(left|right|displaystyle)", "", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.rstrip(".").replace(" ", "")
    s = re.sub(r"^(x|y|a|b|n|k)=", "", s)
    return s


def math_score(item: dict, answer: str) -> dict:
    pred = last_boxed(answer)
    if pred is None:
        return {"pred": None, "correct": False}
    try:  # HF math-verify when installed (pod); string normalization otherwise
        from math_verify import parse, verify

        ok = bool(verify(parse("$" + item["answer"] + "$"), parse("$" + pred + "$")))
    except Exception:  # noqa: BLE001
        ok = _norm_math(pred) == _norm_math(item["answer"])
    return {"pred": pred, "correct": ok}


# ---- IFEval (strict prompt-level, via lm-eval's instruction registry) ----

def ifeval_score(item: dict, answer: str) -> dict:
    from lm_eval.tasks.ifeval import instructions_registry as reg

    follows = []
    for iid, kw in zip(item["instruction_id_list"], item["kwargs"]):
        inst = reg.INSTRUCTION_DICT[iid](iid)
        kw = {k: v for k, v in (kw or {}).items() if v}  # same filter as lm-eval
        inst.build_description(**kw)
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=item["prompt"])
        follows.append(bool(answer.strip()) and inst.check_following(answer))
    return {"correct": all(follows), "instructions_followed": sum(follows), "instructions": len(follows)}


# ---- LiveCodeBench (stdin/stdout) ----

CODE_PROMPT = ("Solve the following programming problem in Python 3. Read from standard input and write to "
               "standard output. Return the complete program in a single ```python code block.\n\n{q}")


def code_prompt(item: dict) -> str:
    return CODE_PROMPT.format(q=item["question"])


def extract_code(answer: str) -> str | None:
    blocks = re.findall(r"```(?:python|py|python3)?\s*\n(.*?)```", answer, re.S)
    return blocks[-1] if blocks else None


def _limits(mem_bytes: int, cpu_s: int):
    def f():
        os.setsid()
        for lim, val in ((resource.RLIMIT_AS, mem_bytes), (resource.RLIMIT_CPU, cpu_s)):
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):  # macOS rejects RLIMIT_AS; the pod (Linux) enforces it
                pass
    return f


def _same_output(got: str, want: str) -> bool:
    g = [l.rstrip() for l in got.strip().splitlines()]
    w = [l.rstrip() for l in want.strip().splitlines()]
    if g == w:
        return True
    try:  # tolerate float formatting
        gt = " ".join(g).split()
        wt = " ".join(w).split()
        return len(gt) == len(wt) and all(a == b or abs(float(a) - float(b)) <= 1e-6 * max(1, abs(float(b)))
                                          for a, b in zip(gt, wt))
    except ValueError:
        return False


def code_score(item: dict, answer: str, max_tests: int | None = None, mem_gb: int = 2) -> dict:
    """Run the program on public + private tests. Network is cut with `unshare -n` when
    available (root on the pod); runs in an empty temp dir with CPU/memory limits."""
    code = extract_code(answer)
    if code is None:
        return {"correct": False, "passed": 0, "tests": 0, "error": "no_code"}
    tests = item["public_tests"] + item["private_tests"]
    if max_tests:
        tests = tests[:max_tests]
    tl = item.get("time_limit_s", 6)
    base = ([shutil.which("unshare"), "-n"] if shutil.which("unshare") and os.geteuid() == 0 else [])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        open(path, "w").write(code)
        passed = 0
        for t in tests:
            try:
                r = subprocess.run(base + [sys.executable, "-I", path], input=t["input"], capture_output=True,
                                   text=True, timeout=tl, cwd=d, preexec_fn=_limits(mem_gb << 30, tl))
                ok = r.returncode == 0 and _same_output(r.stdout, t["output"])
            except subprocess.TimeoutExpired:
                ok = False
            if not ok:
                break  # stop at first failure: pass@1 needs all tests
            passed += 1
    return {"correct": passed == len(tests), "passed": passed, "tests": len(tests)}
