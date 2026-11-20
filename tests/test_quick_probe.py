"""CPU checks for tools/quick_probe.py with a fake engine: step order, restore after every edit, resume, check."""
import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import quick_probe as qp  # noqa: E402


def test_lam_for_layers_marks_attn_and_mlp_slots():
    w = qp.lam_for_layers(187, (54, 55), 1.0)
    assert w.nonzero().flatten().tolist() == [109, 110, 111, 112]
    assert qp.lam_for_layers(187, (92, 92), 0.5).nonzero().flatten().tolist() == [185, 186]


class Fake:
    """Engine stand-in: an 'edited' flag that changes greedy output; gen refuses unless edited."""

    def __init__(self):
        self.edited, self.calls = False, []

    def deps(self, bank):
        def gen(items, mt, t, thinking=True, effort="low"):
            self.calls.append(("effort", effort))
            self.calls.append(("gen", self.edited))
            return [types.SimpleNamespace(item_id=x["id"], reasoning="r", answer="ok" if self.edited else "no",
                                          finish="stop", health={"answer_tokens": 3}) for x in items]

        def judge(items, gens):
            return [{"refused": 0.0 if g.answer == "ok" else 0.99, "deflected": 0.0} for g in gens]

        def edit(Q, lam):
            self.edited = True
            self.calls.append(("edit", tuple(Q.shape)))
            return [{"edited": 5}] * 8

        def restore():
            self.edited = False

        return {"load_items": lambda cfg: [{"id": "harmful:a", "split": "harmful", "question": "q"},
                                           {"id": "harmless:b", "split": "harmless", "question": "q"}],
                "gen": gen, "judge": judge, "edit": edit, "restore": restore,
                "greedy": lambda ps: [[1, 2] if self.edited else [3] for _ in ps], "load_bank": lambda p: bank}


def test_run_check_baseline_edit_and_resume(tmp_path):
    S, d = 187, 8
    bank = {"prompt_end/all": {"mean_diff": torch.randn(S, d)}}
    cfg = {"out": str(tmp_path), "harmful": ["x", 1], "harmless": ["y", 1],
           "steps": [{"kind": "check", "d": d}, {"kind": "baseline", "name": "base"},
                     {"kind": "edit", "name": "e1", "bank": "b", "family": "prompt_end/all", "layers": [54, 60]}]}
    f = Fake()
    res = qp.run(cfg, f.deps(bank), log=lambda *_: None)
    assert [r["name"] for r in res] == ["check", "base", "e1"] and res[0]["ok"]
    assert res[1]["harmful_refusal"] == 1.0 and res[2]["harmful_refusal"] == 0.0  # the edit took effect
    assert not f.edited  # restored after the edit
    assert ("edit", (S, 1, d)) in f.calls and res[2]["edited_per_rank"] == [5] * 8
    assert len((tmp_path / "gens" / "e1.jsonl").read_text().splitlines()) == 2
    # rerun: everything already recorded is skipped
    f2 = Fake()
    assert qp.run(cfg, f2.deps(bank), log=lambda *_: None) == [] and f2.calls == []


def test_failure_mid_edit_restores_and_check_failure_stops(tmp_path):
    S, d = 187, 8
    bank = {"f": {"mean_diff": torch.randn(S, d)}}
    f = Fake()
    deps = f.deps(bank)

    def boom(*a):
        raise RuntimeError("engine died")
    deps["gen"] = boom
    cfg = {"out": str(tmp_path), "harmful": ["x", 1], "harmless": ["y", 1],
           "steps": [{"kind": "edit", "name": "e", "bank": "b", "family": "f", "layers": [1, 2]}]}
    with pytest.raises(RuntimeError, match="engine died"):
        qp.run(cfg, deps, log=lambda *_: None)
    assert not f.edited  # never left edited
    with pytest.raises(KeyError, match="not in bank"):
        qp.run({**cfg, "steps": [{"kind": "edit", "name": "e2", "bank": "b", "family": "nope", "layers": [1, 2]}]},
               f.deps(bank), log=lambda *_: None)
    f3 = Fake()
    d3 = f3.deps(bank)
    d3["greedy"] = lambda ps: [[3] for _ in ps]  # edit does not change output -> check fails
    with pytest.raises(RuntimeError, match="edit check failed"):
        qp.run({**cfg, "out": str(tmp_path / "c"), "steps": [{"kind": "check", "d": d}]}, d3, log=lambda *_: None)
    assert json.loads((tmp_path / "c" / "results.jsonl").read_text())["ok"] is False


def test_failed_check_is_rerun(tmp_path):
    bank = {"f": {"mean_diff": torch.randn(187, 8)}}
    (tmp_path / "results.jsonl").write_text(json.dumps({"name": "check", "ok": False}) + "\n")
    f = Fake()
    res = qp.run({"out": str(tmp_path), "harmful": ["x", 1], "harmless": ["y", 1],
                  "steps": [{"kind": "check", "d": 8}]}, f.deps(bank), log=lambda *_: None)
    assert res and res[0]["ok"]


def test_per_step_thinking_and_paired_flips(tmp_path):
    bank = {"f": {"mean_diff": torch.randn(187, 8)}}
    f = Fake()
    deps = f.deps(bank)
    seen = []
    g0 = deps["gen"]
    deps["gen"] = lambda items, mt, t, thinking=True, effort="low": (seen.append((mt, t, thinking)), g0(items, mt, t))[1]
    cfg = {"out": str(tmp_path), "harmful": ["x", 1], "harmless": ["y", 1], "temperature": 0.0, "thinking": False,
           "steps": [{"kind": "baseline", "name": "base_nt"},
                     {"kind": "edit", "name": "e_nt", "bank": "b", "family": "f", "layers": [1, 2],
                      "compare_to": "base_nt"},
                     {"kind": "baseline", "name": "base_think", "thinking": True, "max_tokens": 99}]}
    res = qp.run(cfg, deps, log=lambda *_: None)
    assert seen == [(4096, 0.0, False), (4096, 0.0, False), (99, 0.0, True)]
    e = res[1]
    assert e["thinking"] is False and e["paired_vs"] == "base_nt"
    assert e["harmful_refused_to_answered"] == 1 and e["harmless_answered_to_refused"] == 0


def test_step_basis_components_and_pca():
    S, d = 5, 8
    g = torch.Generator().manual_seed(0)
    b1 = {"a": {"mean_diff": torch.randn(S, d, generator=g), "pca": torch.randn(S, 8, d, generator=g)}}
    b2 = {"b": {"projected": torch.randn(S, d, generator=g)}}
    banks = {"p1": b1, "p2": b2}
    q = qp.step_basis(banks, {"bank": "p1", "family": "a", "variant": "pca", "rank": 3})
    assert q.shape == (S, 3, d)
    q = qp.step_basis(banks, {"components": [{"bank": "p1", "family": "a"}, {"bank": "p2", "family": "b",
                                                                              "variant": "projected"}]})
    assert q.shape == (S, 2, d) and torch.allclose(q.norm(dim=-1), torch.ones(S, 2))


def test_ablation_basis_drops_zero_and_dependent_rows():
    from harness import ablation

    d = 6
    v = torch.nn.functional.normalize(torch.randn(d), dim=0)
    assert ablation._basis(torch.zeros(d)).shape == (0, d)  # no edit, not a random direction
    assert ablation._basis(torch.stack([v, 2 * v])).shape == (1, d)  # parallel pair -> one direction
    W = torch.randn(d, 3)
    assert torch.equal(ablation.ablate_output_weight(W, torch.zeros(d)), W)
    out = ablation.ablate_output_weight(W, v)
    assert torch.allclose(v @ out, torch.zeros(3), atol=1e-5)


def test_effort_defaults_low_and_cutoff_is_reported_apart(tmp_path):
    """Reasoning steps run at effort "low" unless set; cut-off traces are counted apart from finished ones."""
    S, d = 187, 8
    f = Fake()
    deps = f.deps({"f": {"mean_diff": torch.randn(S, d)}})

    def gen(items, mt, t, thinking=True, effort="low"):
        f.calls.append(("effort", effort))
        return [types.SimpleNamespace(item_id=x["id"], reasoning="r", answer="", finish="length",
                                      health={"reasoning_closed": x["id"] != "harmful:a"}) for x in items]
    deps["gen"] = gen
    deps["judge"] = lambda items, gens: [{"cutoff": True, "heading_to_refuse": 0.1} if not g.health["reasoning_closed"]
                                         else {"refused": 0.9, "deflected": 0.0} for g in gens]
    cfg = {"out": str(tmp_path), "harmful": ["x", 1], "harmless": ["y", 1],
           "steps": [{"kind": "baseline", "name": "low"}, {"kind": "baseline", "name": "max", "effort": "max"}]}
    res = qp.run(cfg, deps, log=lambda *_: None)
    assert [c for c in f.calls if c[0] == "effort"] == [("effort", "low"), ("effort", "max")]
    r = res[0]
    assert r["effort"] == "low" and r["harmful_cutoff"] == 1.0
    assert r["harmful_refusal"] == 0.0 and r["harmful_cutoff_heading_to_refuse"] == 0.0  # heading, not empty answer
    assert r["harmful_refusal_finished"] is None and r["harmless_refusal"] == 1.0


def test_lam_writers_exclude_and_profile():
    w = qp.lam_for_layers(187, (54, 56), 1.5, writers="attn")
    assert w.nonzero().flatten().tolist() == [109, 111, 113] and float(w[109]) == 1.5
    assert qp.lam_for_layers(187, (54, 56), 1.0, writers="mlp").nonzero().flatten().tolist() == [110, 112, 114]
    w = qp.lam_for_layers(187, (50, 60), 1.0, exclude=[[52, 58]])
    assert sorted({(int(i) - 1) // 2 for i in w.nonzero().flatten()}) == [50, 51, 59, 60]
    w = qp.lam_for_layers(187, (54, 55), 1.0, lam_by_layer={"55": 0.5, "70": 2.0})
    assert float(w[109]) == 1.0 and float(w[111]) == 0.5 and float(w[141]) == 0.0  # profile outside the band ignored
    with pytest.raises(ValueError):
        qp.lam_for_layers(187, (1, 2), 1.0, writers="moe")


def test_text_health_flags_run2_damage_modes():
    assert qp.text_health("a lack of resources. expertise, and time.", False)["period_per_1k"] > 0
    assert qp.text_health("Use e.g. this one, etc. and more", False)["period_per_1k"] == 0
    assert qp.text_health("theyweregluedtogethereverywhere here", False)["glued"]
    assert qp.text_health("中文中文中文 text", False)["cjk"]
    leak = "I need respond to user. The user asks for a poem. We need to be careful."
    assert qp.text_health(leak, False)["leak"] and not qp.text_health(leak, True)["leak"]


def test_eval_sets_tiers_kl_and_panel_under_edit(tmp_path):
    rows = {"h.jsonl": [{"id": f"harmful:{i}"} for i in range(6)], "s.jsonl": [{"id": f"harmless:{i}"} for i in range(3)]}

    def load(path, split):
        return [{**x, "question": "q", "split": split} for x in rows[path]]
    cfg = {"out": str(tmp_path), "sets": [{"tag": "dev", "file": "h.jsonl", "split": "harmful", "n": 2},
                                          {"tag": "test", "file": "h.jsonl", "split": "harmful", "skip": 2, "n": 3},
                                          {"tag": "xs", "file": "s.jsonl", "split": "harmless"}],
           "tiers": {"screen": ["dev", "xs"]}}
    items = qp.load_eval_items(cfg, load)
    assert [x["id"] for x in items][:5] == ["harmful:dev:0", "harmful:dev:1", "harmful:test:2", "harmful:test:3",
                                           "harmful:test:4"] and len(items) == 8
    S, d = 187, 8
    f = Fake()
    deps = f.deps({"f": {"mean_diff": torch.randn(S, d)}})
    deps["load_items"] = lambda c: qp.load_eval_items(c, load)
    seen = []
    deps["kl_reference"] = lambda c: [{"ref": 1}]
    deps["kl"] = lambda ref: (seen.append(("kl", f.edited)), {"kl_per_token": 0.01})[1]
    deps["panel"] = lambda: (seen.append(("panel", f.edited)), {"mmlu_pro": {"accuracy": 0.8, "health": {}}})[1]
    cfg["steps"] = [{"kind": "kl_reference", "name": "ref"},
                    {"kind": "baseline", "name": "base", "tier": "screen", "panel": True},
                    {"kind": "edit", "name": "e", "tier": "screen", "bank": "b", "family": "f", "layers": [54, 60],
                     "writers": "attn", "kl": True, "panel": True, "compare_to": "base"}]
    res = qp.run(cfg, deps, log=lambda *_: None)
    assert (tmp_path / "kl_ref.json").exists() and res[1]["n"] == 5  # screen tier: dev 2 + xs 3
    assert ("kl", True) in seen and ("panel", True) in seen and ("panel", False) in seen  # measured under the edit
    e = res[2]
    assert e["kl"]["kl_per_token"] == 0.01 and e["writers"] == "attn" and e["by_set"]["dev"]["n"] == 2
    assert e["panel_vs_ref"]["checks"][0]["pass"] and not f.edited
    assert e["harmful_hard_refusal"] == 0.0 and res[1]["harmful_hard_refusal"] == 1.0


def test_reasoning_length_ratio_vs_reference(tmp_path):
    S, d = 187, 8
    f = Fake()
    deps = f.deps({"f": {"mean_diff": torch.randn(S, d)}})

    def gen(items, mt, t, thinking=True, effort="low"):
        n = 400 if f.edited else 100
        return [types.SimpleNamespace(item_id=x["id"], reasoning="r", answer="ok" if f.edited else "no", finish="stop",
                                      health={"reasoning_tokens": n, "reasoning_closed": not f.edited}) for x in items]
    deps["gen"] = gen
    cfg = {"out": str(tmp_path), "harmful": ["x", 1], "harmless": ["y", 1],
           "steps": [{"kind": "baseline", "name": "base"},
                     {"kind": "edit", "name": "e", "bank": "b", "family": "f", "layers": [1, 2], "compare_to": "base"}]}
    res = qp.run(cfg, deps, log=lambda *_: None)
    assert res[1]["reasoning_length_ratio"] == 4.0 and res[1]["reasoning_closed_drop"] == 1.0


def test_final_rate_clears_delivered_deflections():
    items = [{"id": f"harmful:{i}", "split": "harmful"} for i in range(4)]
    gens = [types.SimpleNamespace(answer="x", health={}) for _ in items]
    verdicts = [{"refused": 0.9, "deflected": 0.9}, {"refused": 0.1, "deflected": 0.7, "delivered": 0.85},
                {"refused": 0.1, "deflected": 0.7, "delivered": 0.3}, {"refused": 0.0, "deflected": 0.1}]
    r = qp.summarize("x", items, gens, verdicts, [True, True, True, False], thinking=False)
    assert r["harmful_refusal"] == 0.75 and r["harmful_refusal_final"] == 0.5 and r["harmful_hard_refusal"] == 0.25


def test_eval_sets_kind_filter():
    rows = [{"id": "harmful:a", "source_split": "v1_pairs_obliteratus"}, {"id": "harmful:b", "source_split": "residual_exploit"}]
    cfg = {"sets": [{"tag": "t", "file": "f", "split": "harmful", "exclude_kinds": ["v1_pairs_obliteratus"]}]}
    assert [x["id"] for x in qp.load_eval_items(cfg, lambda p, s: [dict(x) for x in rows])] == ["harmful:t:b"]


def test_step_saves_each_response_and_resumes_mid_step(tmp_path):
    """A step cut off mid-generation keeps what finished: the rerun generates only the missing prompts, reuses the
    saved verdicts, and discards the partial files if the step changed."""
    bank = {"f": {"mean_diff": torch.randn(187, 8)}}
    items = [{"id": f"harmful:x{i}", "split": "harmful", "question": "q"} for i in range(4)]
    generated, judged = [], []

    def make_deps(crash_after=None):
        f = Fake()
        deps = f.deps(bank)
        deps["load_items"] = lambda cfg: items

        def gen(its, mt, t, thinking=True, effort="low", path=None, on_done=None):
            have = {json.loads(l)["item_id"] for l in open(path)} if path.exists() else set()
            out = []
            for it in its:
                g = types.SimpleNamespace(item_id=it["id"], sample=0, reasoning="r", answer="no", finish="stop",
                                          health={"answer_tokens": 3})
                if it["id"] not in have:
                    if crash_after is not None and len(generated) == crash_after:
                        raise RuntimeError("sandbox lost")
                    generated.append(it["id"])
                    with open(path, "a") as fh:
                        fh.write(json.dumps({"item_id": it["id"]}) + "\n")
                if on_done:
                    on_done(g)
                out.append(g)
            return out

        class SJ:
            def __init__(self, its, path):
                self.path = path
                self.done = {json.loads(l)["key"]: json.loads(l) for l in open(path)} if path.exists() else {}

            def submit(self, g):
                k = f"{g.item_id}#0"
                if k not in self.done:
                    judged.append(g.item_id)
                    self.done[k] = {"key": k, "verdict": {"refused": 0.99, "deflected": 0.0}}
                    with open(self.path, "a") as fh:
                        fh.write(json.dumps(self.done[k]) + "\n")

            def close(self, cancel=False):
                return dict(self.done)

        deps["gen"] = gen
        deps["stream_judge"] = SJ
        deps["judge"] = lambda its, gens, done=None: [done[f"{g.item_id}#0"]["verdict"] for g in gens]
        return deps

    cfg = {"out": str(tmp_path), "steps": [{"kind": "baseline", "name": "base"}]}
    with pytest.raises(RuntimeError):
        qp.run(cfg, make_deps(crash_after=2), log=lambda *_: None)
    assert generated == ["harmful:x0", "harmful:x1"] and judged == generated
    res = qp.run(cfg, make_deps(), log=lambda *_: None)
    assert generated == [f"harmful:x{i}" for i in range(4)] and judged == generated  # nothing redone
    assert res[0]["name"] == "base" and len(open(tmp_path / "gens" / "base.jsonl").readlines()) == 4
    # same name, changed step: the partial responses are discarded rather than mixed in
    (tmp_path / "results.jsonl").unlink()
    generated.clear()
    qp.run({**cfg, "steps": [{"kind": "baseline", "name": "base", "max_tokens": 512}]}, make_deps(),
           log=lambda *_: None)
    assert len(generated) == 4


def test_call_with_drops_unaccepted_keywords():
    assert qp.call_with(lambda a, b=1: (a, b), 1, b=2, path="x") == (1, 2)
    assert qp.call_with(lambda a, **kw: kw, 1, path="x") == {"path": "x"}


def test_live_stream_judge_saves_and_judge_reuses(tmp_path, monkeypatch):
    """live_deps' stream_judge (refusal.StreamJudge + the cutoff/delivery judge) writes each verdict to disk, and
    judge() reuses those and judges only the rest. Jev is faked."""
    import harness.judge as hj
    import harness.response_judge as rj

    calls = []

    class FakeJev:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

    async def jr(j, q, reasoning, answer):
        calls.append(("response", answer))
        return {"refused": 0.9, "deflected": 0.0, "errors": []}

    async def jc(j, q, reasoning):
        calls.append(("cutoff", reasoning))
        return {"cutoff": True, "heading_to_refuse": 0.1, "errors": []}

    monkeypatch.setattr(hj, "JevJudge", FakeJev)
    monkeypatch.setattr(rj, "judge_response", jr)
    monkeypatch.setattr(rj, "judge_cutoff", jc)

    class FakeEv:
        def __init__(self, *a, **k):
            pass

        def _jev(self, fn):
            import asyncio
            return asyncio.run(fn(FakeJev()))
    monkeypatch.setattr("harness.evals.Evaluator", FakeEv)
    deps = qp.live_deps(None, None)
    items = [{"id": "a", "question": "q1"}, {"id": "b", "question": "q2"}]
    g = [types.SimpleNamespace(item_id="a", sample=0, reasoning="r", answer="no", health={"reasoning_closed": True}),
         types.SimpleNamespace(item_id="b", sample=0, reasoning="long", answer="", health={"reasoning_closed": False})]
    path = tmp_path / "j.jsonl"
    sj = deps["stream_judge"](items, path)
    sj.submit(g[0])
    done = sj.close()
    assert [json.loads(l)["key"] for l in open(path)] == ["a#0"]
    v = deps["judge"](items, g, done=done)
    assert v[0]["refused"] == 0.9 and v[1]["cutoff"] is True
    assert calls == [("response", "no"), ("cutoff", "long")]  # "a" judged once, during generation
