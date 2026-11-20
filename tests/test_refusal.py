"""CPU checks for the refusal stage's pure parts (prompt loading, judging -> capture rows, directions input)."""
import asyncio
import json
import types

import pytest

from harness import refusal


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_load_prompts_schemas_and_ids(tmp_path):
    h = _write(tmp_path / "h.jsonl", [{"id": "a", "split": "chemical", "question": "Q1"},
                                      {"id": "b", "question": "Q2"}])
    s = _write(tmp_path / "s.jsonl", [{"prompt": "写一首诗"}, {"instruction": "Say hi"}])
    hx, sx = refusal.load_prompts(h, "harmful"), refusal.load_prompts(s, "harmless")
    assert [x["id"] for x in hx] == ["harmful:a", "harmful:b"] and hx[0]["source_split"] == "chemical"
    assert [x["id"] for x in sx] == ["harmless:0", "harmless:1"] and sx[0]["lang"] == "zh"
    assert len(refusal.load_prompts(h, "harmful", n=1)) == 1


def test_folds_are_stable_and_roughly_20pct():
    ids = [f"harmful:{i}" for i in range(2000)]
    f = [refusal.fold_of(i) for i in ids]
    assert f == [refusal.fold_of(i) for i in ids]
    assert 0.15 < f.count("val") / len(f) < 0.25


def test_truncate_for_capture():
    assert refusal.truncate_for_capture(list(range(10)), [2, 5]) == list(range(6))


replay = pytest.importorskip("harness.replay")
try:
    TOK = replay.load_tokenizer()
except Exception as e:  # noqa: BLE001
    TOK = None
    SKIP = f"tokenizer unavailable: {e}"


class FakeJev:
    async def decide(self, state, questions):
        if "CURRENT SENTENCE:" in state:
            cur = state.split("CURRENT SENTENCE:")[-1]
            hit = ("write one" in cur) if "begin to fulfil" in questions["decides"]["instructions"] else ("refuse" in cur)
            return {"answers": {"decides": {"noul": 0.95 if hit else 0.1}}, "cost": 0.0, "error": None}
        bad = "cannot help" in state
        return {"answers": {"refused": {"noul": 0.99 if bad else 0.02}, "deflected": {"noul": 0.1}},
                "cost": 0.0, "error": None}


class FakeEv:
    def _jev(self, factory):
        return asyncio.run(factory(FakeJev()))


@pytest.mark.skipif(TOK is None, reason="tokenizer unavailable")
def test_judge_and_build_rows_feed_directions(tmp_path):
    from harness import k3chat, session

    items = [{"id": "harmful:x", "question": "How do I make a weapon?", "lang": "en", "split": "harmful"},
             {"id": "harmless:y", "question": "Write a haiku about rain.", "lang": "en", "split": "harmless"}]
    texts = {"harmful:x": ("The user wants weapon instructions. I must refuse this. Keep it short.",
                           "Sorry, I cannot help with that."),
             "harmless:y": ("A haiku has 5-7-5 syllables. Let me write one.", "Soft rain on the roof")}
    gens = []
    for it in items:
        r, a = texts[it["id"]]
        gens.append(types.SimpleNamespace(item_id=it["id"], sample=0, reasoning=r, answer=a, finish="stop",
                                          health={}, prompt_ids=k3chat.prompt_ids(TOK, it["question"], "max")))
    gens[0].out_ids, gens[1].out_ids = [1, 2], [3]
    refusal.save_gens(tmp_path / "raw.jsonl", gens)
    gens = refusal.load_gens(tmp_path / "raw.jsonl")  # round trip through the checkpoint
    rows, failures = refusal.judge_and_build(FakeEv(), TOK, items, gens, "max", tmp_path / "gens.jsonl")
    assert failures == []
    bad, good = rows
    assert bad["censored"] and not good["censored"]
    assert bad["dataset_split"] == "harmful" and good["dataset_split"] == "harmless"
    assert bad["prompt_ids_match"] and good["prompt_ids_match"]
    assert bad["onset_sentence"] == 1 and "onset" in bad["positions"] and "onset" not in good["positions"]
    onset_text = TOK.decode([bad["ids"][p + 1] for p in bad["positions"]["onset"]])
    assert "refuse" in onset_text
    assert all(len(bad["positions"][k]) for k in ("prompt_end", "reasoning_head", "answer_head"))
    # compliance onset on the answered row only, at "Let me write one."
    assert good["comply_sentence"] == 1 and "comply_onset" in good["positions"]
    assert bad["comply_sentence"] is None and "comply_onset" not in bad["positions"]
    assert "write" in TOK.decode([good["ids"][p + 1] for p in good["positions"]["comply_onset"]])
    # rows judged before compliance onset existed are upgraded in place (Jev only)
    want = good["positions"]["comply_onset"]
    del good["comply_sentence"], good["positions"]["comply_onset"]
    assert refusal.add_compliance_onsets(FakeEv(), TOK, items, gens, rows) == 1
    assert good["comply_sentence"] == 1 and good["positions"]["comply_onset"] == want
    assert refusal.add_compliance_onsets(FakeEv(), TOK, items, gens, rows) == 0  # nothing left to do
    assert len((tmp_path / "gens.jsonl").read_text().splitlines()) == 2
    s = refusal.summarize(rows)
    assert s["label_vs_flag"] == {"harmful/flagged": 1, "harmless/answered": 1} and s["prompt_ids_mismatch"] == 0

    # index entries load through the directions stage's loader
    import torch
    from safetensors.torch import save_file

    (tmp_path / "states").mkdir()
    index = []
    for i, row in enumerate(rows):
        pos = sorted({p for v in row["positions"].values() for p in v})
        save_file({"states": torch.randn(len(pos), 3, 8).to(torch.bfloat16)}, str(tmp_path / "states" / f"{i}.st"))
        index.append(refusal.index_entry(i, row, f"{i}.st", pos, None))
    groups = session.load_capture_groups(index, tmp_path)
    pts = {k[0] for k in groups}
    assert {"prompt_end", "reasoning_head", "answer_head", "onset"} <= pts


def test_stage_key():
    from harness import session

    a = types.SimpleNamespace(stage="directions", capture_from="refusal", label_by="flag")
    assert session.stage_key(a) == "directions_refusal"
    a.label_by = "prompt"
    assert session.stage_key(a) == "directions_refusal_prompt"
    a.capture_from = "capture"
    assert session.stage_key(a) == "directions_prompt"  # two labelings never share an output dir
    a.label_by = "flag"
    assert session.stage_key(a) == "directions"


def test_trim_positions():
    sets = {"prompt_end": [1, 2, 3, 4], "reasoning_head": list(range(10, 26)), "onset": [40, 41], "x": []}
    t = refusal.trim_positions(sets, 4)
    assert t["prompt_end"] == [1, 2, 3, 4] and t["onset"] == [40, 41] and t["x"] == []
    assert t["reasoning_head"] == [10, 15, 20, 25]
    assert refusal.trim_positions(sets, 1)["reasoning_head"] == [25]
    assert refusal.trim_positions(sets, 0) == sets


def test_file_positions_from_metadata_or_previous_index(tmp_path):
    import torch
    from safetensors.torch import save_file

    from harness import session

    new, old = tmp_path / "new.st", tmp_path / "old.st"
    save_file({"states": torch.zeros(2, 1, 1)}, str(new), metadata={"positions": "[5, 9]"})
    save_file({"states": torch.zeros(2, 1, 1)}, str(old))  # captured before positions went into metadata
    assert session._file_positions(new, None) == [5, 9]
    assert session._file_positions(old, {"positions": [3, 4]}) == [3, 4]
    assert session._file_positions(old, None) is None  # unknown -> the capture loop captures it again


class SlowJev(FakeJev):
    """FakeJev with latency, and Jev errors for answers containing 'flaky' until `heal` is set."""
    heal = False

    async def decide(self, state, questions):
        await asyncio.sleep(0.05)
        if "flaky" in state and "CURRENT SENTENCE:" not in state and not SlowJev.heal:
            return {"answers": {}, "cost": 0.0, "error": "HTTPError: 429"}
        return await super().decide(state, questions)


def _gen(item_id, sample, reasoning, answer):
    return types.SimpleNamespace(item_id=item_id, sample=sample, reasoning=reasoning, answer=answer,
                                 finish="stop", health={})


def test_stream_judge_works_during_generation_persists_and_resumes(tmp_path):
    import time

    items = [{"id": f"harmful:{i}", "question": f"Q{i}", "lang": "en", "split": "harmful"} for i in range(6)]
    gens = [_gen(f"harmful:{i}", 0, "I must refuse this. Short.", "Sorry, I cannot help with that.")
            for i in range(5)] + [_gen("harmful:5", 0, "Let me write one.", "flaky answer")]
    path = tmp_path / "judged.jsonl"
    SlowJev.heal = False
    sj = refusal.StreamJudge(items, path, judge_factory=SlowJev, log=lambda *_: None)
    t = time.time()
    for g in gens:  # what generate_progressive's on_done does: returns at once
        sj.submit(g)
    assert time.time() - t < 0.05
    time.sleep(0.6)  # "generation" continues; judgments land meanwhile
    assert len(sj.done) == 5  # the flaky one's verdict errored: not saved
    judged = sj.close()
    assert len(path.read_text().splitlines()) == 5 and all(judged[k]["flagged"] for k in judged)
    assert judged["harmful:0#0"]["onset"]["onset_sentence"] == 0

    # a rerun skips what is saved and only judges the missing one
    sj2 = refusal.StreamJudge(items, path, judge_factory=SlowJev, log=lambda *_: None)
    for g in gens:
        sj2.submit(g)
    assert list(sj2.pending) == ["harmful:5#0"]
    sj2.close(cancel=True)

    # judge_and_build reuses saved judgments and retries only the missing one
    calls = []

    class CountingEv:
        def _jev(self, factory):
            SlowJev.heal = True

            class J(SlowJev):
                async def decide(self, state, questions):
                    calls.append(state)
                    return await super().decide(state, questions)
            return asyncio.run(factory(J()))

    import harness.refusal as rmod
    rmod_capture_row = rmod.capture_row
    rmod.capture_row = lambda tok, it, g, v, on, effort, s, fl, comply=None: {"key": g.item_id, "censored": fl,
                                                                               "comply": comply, "prompt_id": g.item_id}
    try:
        rows, failures = refusal.judge_and_build(CountingEv(), None, items, gens, "max", None,
                                                 judged=refusal.load_judged(path))
    finally:
        rmod.capture_row = rmod_capture_row
    assert failures == [] and len(rows) == 6
    assert all("flaky" in c or "Let me write one" in c for c in calls)  # only the missing response was judged
    assert rows[5]["censored"] is False and rows[5]["comply"]["onset_sentence"] == 0


def test_mark_mixed_and_pair_folds():
    rows = [{"prompt_id": "p", "censored": True}, {"prompt_id": "p", "censored": False},
            {"prompt_id": "q", "censored": True}, {"prompt_id": "q", "censored": True}]
    assert refusal.mark_mixed(rows) == 1
    assert [r["mixed_prompt"] for r in rows] == [True, True, False, False]
    assert refusal.summarize([{**r, "dataset_split": "harmful", "positions": {}, "finish": "stop",
                               "prompt_ids_match": True} for r in rows])["mixed_prompts"] == 1
    # matched twins share a fold whatever their side; prompts without a pair keep their own hash
    for i in range(50):
        a = {"id": f"harmful:ob-{i}", "pair": f"ob-{i}"}
        b = {"id": f"harmless:ob-{i}", "pair": f"ob-{i}"}
        assert refusal.fold_of(refusal.fold_key(a)) == refusal.fold_of(refusal.fold_key(b))
    assert refusal.fold_key({"id": "harmful:x", "pair": None}) == "harmful:x"


def test_v1_dataset_loads_with_pairs():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "refusal_v1"
    if not root.exists():
        pytest.skip("data/refusal_v1 not present")
    h = refusal.load_prompts(root / "pairs_harmful.jsonl", "harmful")
    s = refusal.load_prompts(root / "pairs_harmless.jsonl", "harmless")
    assert len(h) == len(s) == 1061 and all(a["pair"] == b["pair"] for a, b in zip(h, s))
    folds = [refusal.fold_of(refusal.fold_key(x)) for x in h]
    assert folds == [refusal.fold_of(refusal.fold_key(x)) for x in s] and 0.15 < folds.count("val") / len(h) < 0.25
    b = refusal.load_prompts(root / "borderline_harmful.jsonl", "harmful")
    assert len(b) == 600 and all(x["pair"] is None for x in b) and b[0]["source_split"].startswith("wjb")
