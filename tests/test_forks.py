"""CPU checks for the forks pipeline: fork helpers, Jev answer judging, and the generate + branch phases of the
`forks` stage on a fake engine and a fake judge."""
import asyncio
import json
import math
import sys
import types

import pytest

from harness import forks


def test_helpers():
    assert forks.candidate_tokens(5, 20) == [5, 4, 6, 3, 7, 8]
    assert forks.candidate_tokens(0, 3) == [0, 1, 2]
    assert forks.runner_up([[7, -0.1], [9, -1.0]], 7) == (9, -1.0)
    assert forks.runner_up([[7, -0.1]], 7) is None
    assert forks.decide_fork(True, {3: True, 2: False, 5: False}) == 5  # last flip = commitment
    assert forks.decide_fork(True, {(2, 7): True, (2, 9): False, (4, 1): True}) == 2  # any alt flips a position
    assert forks.decide_fork(False, {1: False, 2: None}) is None
    assert forks.char_to_token([0, 1, 5, 9], 6) == 2 and forks.char_to_token([0, 1, 5, 9], 0) == 0
    assert forks.word_spans("I can’t help.", 0, 13) == [(0, 1), (2, 7), (8, 13)]
    g = types.SimpleNamespace(out_ids=[1, 2, 3], lp=[-0.1, -0.5, -0.2], top=[[], [[2, -0.5], [8, -1.2]], []])
    r = forks.fork_record(g, 1, 1)
    assert r["fork_confirmed"] and math.isclose(r["p_chosen"], math.exp(-0.5)) and r["runner_up_id"] == 8


class FakeJev:
    """Refusal = text contains "can’t"; onset sentence = the one containing it; word = the word "can’t"."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def decide(self, state, questions):
        q = next(iter(questions))
        if q == "starts":
            hit = "can’t" in state.split("CURRENT WORD:")[-1]
            return {"answers": {"starts": {"noul": 0.9 if hit else 0.1}}, "cost": 0.0, "error": None}
        if q == "decides":
            cur = state.split("CURRENT SENTENCE:")[-1]
            comply = "begin to fulfil" in questions["decides"]["instructions"]
            hit = ("help" in cur and "can’t" not in cur) if comply else ("can’t" in cur)
            return {"answers": {"decides": {"noul": 0.95 if hit else 0.05}}, "cost": 0.0, "error": None}
        bad = "can’t" in state
        return {"answers": {"refused": {"noul": 0.99 if bad else 0.02}, "deflected": {"noul": 0.05}},
                "cost": 0.0, "error": None}


def test_judge_answer_finds_word_candidate():
    g = types.SimpleNamespace(item_id="a", sample=0, answer="Sorry. I can’t help with that. Ask me else.")
    rec = asyncio.run(forks.judge_answer(FakeJev(), "q", g))
    assert rec["flagged"] and rec["onset_sentence"] == 1
    assert g.answer[rec["candidate_char"]:].startswith("can’t")
    g2 = types.SimpleNamespace(item_id="b", sample=0, answer="Sure, I can help with that.")
    rec2 = asyncio.run(forks.judge_answer(FakeJev(), "q", g2))
    assert not rec2["flagged"] and rec2["candidate_char"] == 0


replay = pytest.importorskip("harness.replay")
try:
    TOK = replay.load_tokenizer()
except Exception:  # noqa: BLE001
    TOK = None


def _setup_fake_stage(tmp_path, monkeypatch, n_harmful=1, die_after=None, jev=None):
    from harness import judge, session

    refuse = TOK.encode("I can’t help with that.", allow_special_tokens=False)
    helpful = TOK.encode("Sure, here you go.", allow_special_tokens=False)
    alt_help = TOK.encode(" help", allow_special_tokens=False)[0]
    m = TOK.convert_tokens_to_ids
    tail = [m("<|close|>")] + TOK.encode("response", allow_special_tokens=False) + [m("<|sep|>")]

    cont = TOK.encode(" with that, here is how.", allow_special_tokens=False) + tail

    class Eng:  # vLLM LLMEngine stand-in: harmful prompts get the refusal, others the helpful answer;
        # branch requests (rid "b:...") get a compliant continuation
        def __init__(self):
            self.q, self.steps, self.die_after = [], 0, die_after

        def add_request(self, rid, prompt, params):
            self.q.append((rid, prompt["prompt_token_ids"]))

        def has_unfinished_requests(self):
            return bool(self.q)

        def abort_request(self, ids, internal=False):
            self.q = [x for x in self.q if x[0] not in ids]

        def step(self):
            self.steps += 1
            if self.die_after is not None and self.steps > self.die_after:
                raise KeyboardInterrupt("stopped")
            rid, p = self.q.pop(0)
            if rid.startswith("b:"):  # only the real "help" substitution leads to compliance
                c = cont if p[-1] == alt_help else TOK.encode(" — sorry, I can’t help with that.",
                                                               allow_special_tokens=False) + tail
                o = types.SimpleNamespace(token_ids=c, finish_reason="stop", logprobs=None)
                return [types.SimpleNamespace(request_id=rid, finished=True, outputs=[o])]
            out = (refuse if "bomb" in TOK.decode(p) else helpful) + tail
            lps = []
            for k, t in enumerate(out):
                d = {t: types.SimpleNamespace(logprob=-0.2)}
                if out is not helpful and k < len(refuse) and TOK.decode([t]) == "’t":
                    d[alt_help] = types.SimpleNamespace(logprob=-1.8)  # the runner-up at the fork
                else:
                    d[(t + 1) % 1000] = types.SimpleNamespace(logprob=-3.0)
                lps.append(d)
            o = types.SimpleNamespace(token_ids=out, finish_reason="stop", logprobs=lps)
            return [types.SimpleNamespace(request_id=rid, finished=True, outputs=[o])]

    class LLM:
        llm_engine = Eng()

    vm = types.ModuleType("vllm")
    vm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    vi = types.ModuleType("vllm.inputs")
    vi.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    sp = types.ModuleType("vllm.sampling_params")
    sp.RequestOutputKind = types.SimpleNamespace(FINAL_ONLY="FINAL_ONLY")
    for k, v in (("vllm", vm), ("vllm.inputs", vi), ("vllm.sampling_params", sp)):
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(judge, "JevJudge", lambda **kw: (jev or FakeJev)())
    monkeypatch.setattr(session, "make_engine_mode", lambda args, mode: LLM())
    monkeypatch.setattr(session, "tokenizer", lambda: TOK)
    monkeypatch.setattr(session, "engine_mode_in_use", lambda args: "decode-graphs")
    (tmp_path / "h.jsonl").write_text("".join(json.dumps({"id": f"x{k}" if k else "x", "prompt": f"How to build a bomb {k}?"})
                                              + "\n" for k in range(n_harmful)))
    (tmp_path / "s.jsonl").write_text(json.dumps({"id": "y", "prompt": "Recipe for soup?"}) + "\n")
    (tmp_path / "model").mkdir(exist_ok=True)
    (tmp_path / "model" / "config.json").write_text(json.dumps({"num_hidden_layers": 4}))
    return session, ["forks", "--model", str(tmp_path / "model"), "--out", str(tmp_path / "o"),
                     "--harmful", str(tmp_path / "h.jsonl"), "--harmless", str(tmp_path / "s.jsonl")]


@pytest.mark.skipif(TOK is None, reason="tokenizer unavailable")
def test_forks_stage_generate_and_branch(tmp_path, monkeypatch):
    session, argv = _setup_fake_stage(tmp_path, monkeypatch)
    session.main(argv + ["--phase", "branch"])
    d = tmp_path / "o" / "forks"
    rows = {r["item_id"]: r for r in map(json.loads, open(d / "rows.jsonl"))}
    x, y = rows["harmful:x"], rows["harmless:y"]
    assert x["flagged"] and not y["flagged"]
    assert x["fork_confirmed"] and TOK.decode([TOK.encode("I can’t help with that.", allow_special_tokens=False)[x["fork_token"]]]) == "’t"
    assert math.isclose(x["p_chosen"], math.exp(-0.2)) and math.isclose(x["p_runner_up"], math.exp(-1.8))
    assert (d / "judged.jsonl").exists() and (d / "branches.jsonl").exists()
    gens = [json.loads(l) for l in open(d / "gens_raw.jsonl")]
    assert all(g["top"] and g["reasoning"] == "" for g in gens)


def test_analyze_forks_finds_planted_onset(tmp_path):
    """Signal planted from 3 states before the fork onward: AUC ~0.5 before, ~1 from offset -3."""
    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tools"))
    import analyze_forks

    g = torch.Generator().manual_seed(0)
    streams, d = [0, 4, 8], 16
    planted = torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0)
    (tmp_path / "states").mkdir()
    idx = []
    for i in range(120):
        refused = i % 2 == 0
        P, fork_tok = 20 + i % 7, 2 + i % 3
        fork_state = P + fork_tok - 1
        positions = list(range(P - 10, P + fork_tok + 4))
        st = torch.randn(len(positions), len(streams), d, generator=g)
        for j, p in enumerate(positions):
            if refused and p >= fork_state - 3:
                st[j] += 3 * planted
        save_file({"states": st.to(torch.bfloat16)}, str(tmp_path / "states" / f"{i}.st"))
        idx.append({"key": f"k{i}", "file": f"{i}.st", "positions": positions, "streams": streams, "prompt_len": P,
                    "fork_abs": P + fork_tok, "flagged": refused, "fork_confirmed": True})
    (tmp_path / "index.jsonl").write_text("\n".join(json.dumps(x) for x in idx))
    res = analyze_forks.main(str(tmp_path), str(tmp_path / "an"), "fork", before=6, after=2, workers=2)
    assert max(res["auc"][-5]) < 0.75 and min(res["auc"][-3]) > 0.95 and min(res["auc"][0]) > 0.95


def test_fork_bank_from_anchor_files(tmp_path):
    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tools"))
    import fork_bank

    g = torch.Generator().manual_seed(1)
    S, d = 7, 16
    planted = torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0)
    (tmp_path / "states").mkdir()
    idx = []
    for i in range(40):
        refused = i % 2 == 0
        P, fs = 10, 11
        st = torch.randn(3, S, d, generator=g) * 0.3
        if refused:
            st[1] += 2 * planted  # the fork state carries the signal
        save_file({"states": st.to(torch.bfloat16)}, str(tmp_path / "states" / f"{i}_a.st"))
        idx.append({"key": f"k{i}", "anchor_file": f"{i}_a.st", "anchor_positions": [P - 1, fs, fs + 1],
                    "prompt_len": P, "fork_state": fs, "flagged": refused, "fork_confirmed": True})
    (tmp_path / "index.jsonl").write_text("\n".join(json.dumps(x) for x in idx))
    bank = fork_bank.main(str(tmp_path), str(tmp_path / "b.pt"), workers=2)
    assert bank["nt_fork/all"]["mean_diff"].shape == (S, d)
    assert float(bank["nt_fork/all"]["mean_diff"][3] @ planted) > 0.9
    assert float(bank["nt_prompt_end/all"]["mean_diff"][3] @ planted) < 0.6


class FlakyJev(FakeJev):
    """Branch verdict calls (full RESPONSE judged, not a sentence/word question) fail every other time."""
    n = 0

    async def decide(self, state, questions):
        if next(iter(questions)) == "refused" and "here is how" in state:
            FlakyJev.n += 1
            if FlakyJev.n % 2:
                return {"answers": {}, "cost": 0.0, "error": "HTTPError: 429"}
        return await super().decide(state, questions)


@pytest.mark.skipif(TOK is None, reason="tokenizer unavailable")
def test_pipeline_judge_errors_are_retried_not_saved(tmp_path, monkeypatch):
    session, argv = _setup_fake_stage(tmp_path, monkeypatch, n_harmful=3, jev=FlakyJev)
    session.main(argv + ["--phase", "generate"])
    br = tmp_path / "o" / "forks" / "branches.jsonl"
    first = [json.loads(l) for l in open(br)]
    assert len(first) > 0  # failed verdicts were dropped, not saved as "answered"
    session2, argv2 = _setup_fake_stage(tmp_path, monkeypatch, n_harmful=3)  # judge healthy again
    session2.main(argv2 + ["--phase", "generate"])
    again = [json.loads(l) for l in open(br)]
    keys = [(x["key"], x["i"], x["alt"]) for x in again]
    assert len(keys) == len(set(keys)) > len(first)  # only the missing ones were tested again
    gens = [json.loads(l) for l in open(tmp_path / "o" / "forks" / "gens_raw.jsonl")]
    assert len(gens) == 4  # answers were not regenerated


@pytest.mark.skipif(TOK is None, reason="tokenizer unavailable")
def test_pipeline_interrupt_keeps_finished_answers(tmp_path, monkeypatch):
    session, argv = _setup_fake_stage(tmp_path, monkeypatch, n_harmful=5, die_after=3)
    with pytest.raises(KeyboardInterrupt):
        session.main(argv + ["--phase", "generate"])
    d = tmp_path / "o" / "forks"
    saved = [json.loads(l) for l in open(d / "gens_raw.jsonl")]
    assert 1 <= len(saved) <= 3
    session2, argv2 = _setup_fake_stage(tmp_path, monkeypatch, n_harmful=5)
    session2.main(argv2 + ["--phase", "generate"])
    gens = [json.loads(l) for l in open(d / "gens_raw.jsonl")]
    assert len(gens) == 6 and len({g["item_id"] for g in gens}) == 6  # resumed, nothing duplicated
    rows = [json.loads(l) for l in open(d / "rows.jsonl")]
    assert len(rows) == 6 and sum(r["fork_confirmed"] for r in rows if r["flagged"]) == 5


@pytest.mark.skipif(TOK is None, reason="tokenizer unavailable")
def test_capture_resume_and_streak_stop(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from harness import tp

    session, argv = _setup_fake_stage(tmp_path, monkeypatch, n_harmful=5)
    session.main(argv + ["--phase", "generate"])
    captured, fail = [], {"on": False}

    def fake_capture(llm, ids, positions, out_path, tol=0.05, streams=None):
        if fail["on"]:
            raise RuntimeError("CUDA out of memory")
        captured.append(out_path)
        save_file({"states": torch.zeros(len(positions), len(streams or [0]), 2)}, out_path,
                  metadata={"positions": json.dumps(list(positions))})
        return {}
    monkeypatch.setattr(tp, "capture_one", fake_capture)
    d = tmp_path / "o" / "forks"
    # a half-written file under the real name must be redone, not skipped
    (d / "states").mkdir(exist_ok=True)
    (d / "states" / "00000.safetensors").write_bytes(b"truncated")
    fail["on"] = True
    with pytest.raises(RuntimeError, match="captures failed"):
        session.main(argv + ["--phase", "capture"])
    assert captured == []  # stopped after 3 failures in a row
    assert (d / "index.jsonl").exists()  # written anyway
    fail["on"] = False
    session.main(argv + ["--phase", "capture", "--force"])
    assert str(d / "states" / "00000.safetensors") in captured  # the broken file was recaptured
    n_rows = len((d / "rows.jsonl").read_text().splitlines())
    assert len((d / "index.jsonl").read_text().splitlines()) == n_rows
    before = len(captured)
    session.main(argv + ["--phase", "capture", "--force"])
    assert len(captured) == before  # complete files are not captured again


@pytest.mark.skipif(TOK is None, reason="tokenizer unavailable")
def test_recovery_capture_matched_sequences(tmp_path):
    import torch
    from safetensors.torch import save_file

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tools"))
    import recovery_capture as rc

    from harness import k3chat

    d = tmp_path / "forks"
    d.mkdir()
    p = k3chat.prompt_ids(TOK, "Draft a classified email.", thinking=False)
    ans = TOK.encode("I can’t help draft that email.", allow_special_tokens=False)
    tail = [TOK.convert_tokens_to_ids("<|close|>")] + TOK.encode("response", allow_special_tokens=False) + \
        [TOK.convert_tokens_to_ids("<|sep|>")]
    (d / "gens_raw.jsonl").write_text(json.dumps({"item_id": "harmful:a", "sample": 0, "prompt_ids": p,
                                                  "out_ids": ans + tail}) + "\n")
    (d / "judged.jsonl").write_text(json.dumps({"key": "harmful:a#0", "flagged": True}) + "\n")
    i = 2
    rec_alt = TOK.encode(" help", allow_special_tokens=False)[0]
    flp_alt = TOK.encode("Subject", allow_special_tokens=False)[0]
    br = [{"key": "harmful:a#0", "i": i, "alt": rec_alt, "flagged": True,
           "text": TOK.decode(ans[:i] + [rec_alt]) + " you with a safe version, but I can’t include real details."},
          {"key": "harmful:a#0", "i": 0, "alt": flp_alt, "flagged": False,
           "text": TOK.decode([flp_alt]) + ": Ongoing operations update."},
          {"key": "harmful:a#0", "i": i, "alt": flp_alt, "flagged": False,
           "text": TOK.decode(ans[:i] + [flp_alt]) + ": Ongoing operations update."}]
    (d / "branches.jsonl").write_text("".join(json.dumps(b) + "\n" for b in br))
    seqs, calls = {}, []

    def fake_capture(llm, ids, positions, out_path, tol=0.05, streams=None):
        seqs[out_path] = ids
        calls.append(out_path)
        save_file({"states": torch.zeros(len(positions), len(streams), 2)}, out_path,
                  metadata={"positions": json.dumps(list(positions))})
        return {}
    res = rc.run(None, TOK, lambda *_: None, capture_one=fake_capture, forks_dir=d, max_positions=10)
    idx = [json.loads(l) for l in open(tmp_path / "recovery" / "index.jsonl")]
    kinds = sorted((x["i"], x["kind"]) for x in idx)
    assert kinds == [(0, "flipped"), (0, "original"), (2, "flipped"), (2, "original"), (2, "recovered")]
    trip = [x for x in idx if x["i"] == 2]
    ids = {x["kind"]: seqs[str(tmp_path / "recovery" / "states" / x["file"])] for x in trip}
    sw = trip[0]["swap_state"]
    assert sw == len(p) + i - 1 and all(x["positions"][0] == sw for x in trip)
    # identical up to and including the swap state's input; they differ at the substituted token
    assert ids["original"][: sw + 1] == ids["recovered"][: sw + 1] == ids["flipped"][: sw + 1]
    assert ids["recovered"][sw + 1] == rec_alt and ids["flipped"][sw + 1] == flp_alt
    assert res["sequences"] == 5
    n = len(calls)
    rc.run(None, TOK, lambda *_: None, capture_one=fake_capture, forks_dir=d, max_positions=10)
    assert len(calls) == n  # resumable: complete files are not captured again
