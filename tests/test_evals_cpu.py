import json
import sys
import types

import pytest

from harness import evals, k3chat, scorers


def test_mc_extract():
    assert scorers.mc_extract("blah\nAnswer: C", 4) == 2
    assert scorers.mc_extract("思考……\n答案：B", 4) == 1
    assert scorers.mc_extract("**Answer: (J)**", 10) == 9
    assert scorers.mc_extract("Answer: E", 4) is None
    assert scorers.mc_extract("no idea", 4) is None


def test_math_score():
    it = {"answer": "\\frac{1}{2}"}
    assert scorers.math_score(it, "so \\boxed{\\dfrac{1}{2}}.")["correct"]
    assert not scorers.math_score(it, "\\boxed{2}")["correct"]
    assert scorers.last_boxed("\\boxed{a_{1}+\\sqrt{2}}") == "a_{1}+\\sqrt{2}"


def test_code_score_runs_programs():
    it = {"public_tests": [{"input": "2 3\n", "output": "5\n"}], "private_tests": [{"input": "10 -4\n", "output": "6"}],
          "time_limit_s": 5}
    good = "```python\na,b=map(int,input().split())\nprint(a+b)\n```"
    bad = "```python\nprint(0)\n```"
    loop = "```python\nwhile True: pass\n```"
    assert scorers.code_score(it, good)["correct"]
    assert not scorers.code_score(it, bad)["correct"]
    it["time_limit_s"] = 1
    assert not scorers.code_score(it, loop)["correct"]
    assert scorers.code_score(it, "no code")["error"] == "no_code"


def test_ifeval_scorer():
    pytest.importorskip("lm_eval.tasks.ifeval.instructions_registry")
    item = {"prompt": "Write a sentence in all lowercase letters.", "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}]}
    assert scorers.ifeval_score(item, "this is all lowercase.")["correct"]
    assert not scorers.ifeval_score(item, "This Has Caps.")["correct"]


def test_health_metrics():
    assert k3chat.repetition_rate([1, 2, 3, 4] * 10) > 0.8
    assert k3chat.repetition_rate(list(range(50))) == 0
    assert k3chat.off_language_fraction("Hello 你好", "en") == pytest.approx(2 / 7)


def test_paired_bootstrap():
    base = {str(i): 0.5 for i in range(200)}
    cand = {str(i): 0.5 + (0.1 if i % 2 else 0) for i in range(200)}
    r = evals.paired_bootstrap(base, cand, n_boot=2000)
    assert r["delta"] == pytest.approx(0.05) and r["ci95"][0] > 0


# --- full Evaluator path with a fake vLLM and the real K3.1 tokenizer ---
replay = pytest.importorskip("harness.replay")


@pytest.fixture(scope="module")
def tok():
    try:
        return replay.load_tokenizer()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {e}")


def _install_fake_vllm(monkeypatch):
    m = types.ModuleType("vllm")
    m.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    inp = types.ModuleType("vllm.inputs")
    inp.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    monkeypatch.setitem(sys.modules, "vllm", m)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inp)


class FakeLLM:
    def __init__(self, tok, answer):
        full = replay.build(tok, "q", "Let me think.", answer)
        self.out = full.ids[full.prompt_len:]

    def generate(self, reqs, params, use_tqdm=False):
        o = types.SimpleNamespace(token_ids=self.out, finish_reason="stop")
        return [types.SimpleNamespace(outputs=[o], prompt_logprobs=None) for _ in reqs]


def test_evaluator_benchmark_with_fake_engine(tok, monkeypatch):
    _install_fake_vllm(monkeypatch)
    ev = evals.Evaluator(FakeLLM(tok, "Reasoned.\nAnswer: B"), tok)
    items = [{"id": "x1", "lang": "en", "question": "Pick B", "choices": ["a", "b", "c", "d"], "answer": 1},
             {"id": "x2", "lang": "en", "question": "Pick A", "choices": ["a", "b", "c", "d"], "answer": 0}]
    r = ev.benchmark("mmlu_pro", items, n=2)
    assert r["accuracy"] == 0.5 and r["per_item"] == {"x1": 1.0, "x2": 0.0}
    g = r["gens"][0]
    assert g.reasoning == "Let me think." and g.answer == "Reasoned.\nAnswer: B"
    assert r["health"]["reasoning_closed"] == 1.0 and r["health"]["answer_closed"] == 1.0


class FakeStepEngine:
    """add_request / step / abort like vLLM's LLMEngine: one request finishes per step."""

    def __init__(self, out_ids, die_after=None):
        self.out, self.queue, self.added, self.aborted = out_ids, [], [], []
        self.steps, self.die_after = 0, die_after

    def add_request(self, rid, prompt, params):
        assert params.output_kind == "FINAL_ONLY"
        self.queue.append((rid, params.seed))
        self.added.append(rid)

    def has_unfinished_requests(self):
        return bool(self.queue)

    def step(self):
        self.steps += 1
        if self.die_after is not None and self.steps > self.die_after:
            raise KeyboardInterrupt("stopped")
        rid, seed = self.queue.pop(0)
        o = types.SimpleNamespace(token_ids=self.out, finish_reason="stop")
        return [types.SimpleNamespace(request_id=rid, finished=True, outputs=[o])]

    def abort_request(self, ids, internal=False):
        self.aborted += ids
        self.queue = [q for q in self.queue if q[0] not in ids]


def test_generate_progressive_survives_a_stop(tok, monkeypatch, tmp_path):
    _install_fake_vllm(monkeypatch)
    sp = types.ModuleType("vllm.sampling_params")
    sp.RequestOutputKind = types.SimpleNamespace(FINAL_ONLY="FINAL_ONLY")
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)
    out = FakeLLM(tok, "Done.").out
    items = [{"id": f"q{i}", "lang": "en", "question": f"Q{i}?"} for i in range(4)]
    path = tmp_path / "gens_raw.jsonl"

    llm = types.SimpleNamespace(llm_engine=FakeStepEngine(out, die_after=3))
    ev = evals.Evaluator(llm, tok)
    with pytest.raises(KeyboardInterrupt):
        ev.generate_progressive(items, lambda x: x["question"], path, n=2, log=lambda *_: None)
    assert len(path.read_text().splitlines()) == 3  # finished before the stop: on disk
    assert len(llm.llm_engine.aborted) == 5 and not llm.llm_engine.queue  # in-flight ones aborted

    llm2 = types.SimpleNamespace(llm_engine=FakeStepEngine(out))
    seen = []
    gens = evals.Evaluator(llm2, tok).generate_progressive(items, lambda x: x["question"], path, n=2,
                                                           log=lambda *_: None, on_done=seen.append)
    assert len(seen) == 8 and {(g.item_id, g.sample) for g in seen[:3]} <= {(g.item_id, g.sample) for g in gens}
    assert len(llm2.llm_engine.added) == 5  # only the missing ones regenerated
    assert [(g.item_id, g.sample) for g in gens] == [(f"q{i}", s) for i in range(4) for s in range(2)]
    assert all(g.answer == "Done." and g.reasoning == "Let me think." for g in gens)
    assert len(path.read_text().splitlines()) == 8


class LogprobStepEngine(FakeStepEngine):
    """Like FakeStepEngine, but each output carries vLLM-style per-token logprob dicts."""

    def step(self):
        rid, _ = self.queue.pop(0)
        lps = []
        for t in self.out:
            alt = (t + 1) % 1000
            lps.append({t: types.SimpleNamespace(logprob=-0.1), alt: types.SimpleNamespace(logprob=-2.5)})
        o = types.SimpleNamespace(token_ids=self.out, finish_reason="stop", logprobs=lps)
        return [types.SimpleNamespace(request_id=rid, finished=True, outputs=[o])]


def test_generate_progressive_non_thinking_keeps_logprobs(tok, monkeypatch, tmp_path):
    _install_fake_vllm(monkeypatch)
    sp = types.ModuleType("vllm.sampling_params")
    sp.RequestOutputKind = types.SimpleNamespace(FINAL_ONLY="FINAL_ONLY")
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)
    full = replay.build(tok, "q", "", "I can’t help with that.", thinking=False)
    out = full.ids[full.prompt_len:]
    llm = types.SimpleNamespace(llm_engine=LogprobStepEngine(out))
    ev = evals.Evaluator(llm, tok, thinking=False)
    path = tmp_path / "gens_raw.jsonl"
    gens = ev.generate_progressive([{"id": "a", "lang": "en", "question": "q"}], lambda x: x["question"], path,
                                   temperature=0.0, logprobs=20, log=lambda *_: None)
    g = gens[0]
    assert g.answer == "I can’t help with that." and g.reasoning == ""
    assert g.prompt_ids == full.ids[:full.prompt_len]  # non-thinking prompt
    assert len(g.lp) == len(out) and g.lp[0] == -0.1 and g.top[0][0] == [out[0], -0.1] and len(g.top[0]) == 2
    x = json.loads(path.read_text().splitlines()[0])
    assert x["top"][1][1][1] == -2.5
    g2 = evals.Evaluator(llm, tok, thinking=False).generate_progressive(  # resume keeps lp/top
        [{"id": "a", "lang": "en", "question": "q"}], lambda x: x["question"], path, logprobs=20, log=lambda *_: None)[0]
    assert g2.top == g.top
