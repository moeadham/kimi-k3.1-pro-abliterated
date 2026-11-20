"""Needs the K3.1 tokenizer files from HF (skipped offline)."""
import json
from pathlib import Path

import pytest

replay = pytest.importorskip("harness.replay")
try:
    TOK = replay.load_tokenizer()
except Exception as e:  # noqa: BLE001
    pytest.skip(f"tokenizer unavailable: {e}", allow_module_level=True)


def test_structure_and_spans():
    r = replay.build(TOK, "Hi?", "Think a bit. 然后说中文。", "Hello.", effort="max")
    d = TOK.decode
    assert d(r.ids[r.prompt_len - 3:r.prompt_len]) == "<|open|>think<|sep|>"
    assert d(r.ids[slice(*r.reasoning)]) == "Think a bit. 然后说中文。"
    assert d(r.ids[slice(*r.answer)]) == "Hello."
    assert d(r.ids[r.answer[0] - 3:r.answer[0]]) == "<|open|>response<|sep|>"
    assert d(r.ids[-1:]) == "<|end_of_msg|>"


def test_onset_char_span_maps_to_tokens():
    text = "User asks X. 需要与官方表述保持一致。 Respond."
    c0 = text.index("需"); c1 = text.index(" Respond")
    r = replay.build(TOK, "Q", text, "A")
    s, e = r.reasoning_tokens_for_chars(c0, c1)
    assert TOK.decode(r.ids[s:e]).strip() == "需要与官方表述保持一致。"
    pos = r.positions(onset_chars=(c0, c1))
    assert pos["onset"] == list(range(s - 1, e - 1))
    assert pos["prompt_end"][-1] == r.prompt_len - 1 and pos["reasoning_head"][0] == r.prompt_len - 1


def test_hosted_transcripts_round_trip():
    p = Path("results/hosted_deccp/judged.jsonl")
    if not p.exists():
        pytest.skip("no hosted results")
    rows = [json.loads(l) for l in open(p)][:60]
    for x in rows:
        r = replay.build(TOK, x["question"], x["reasoning"], x["content"], effort=x["effort"] if x["effort"] in ("low", "high", "max") else "max")
        assert TOK.decode(r.ids[slice(*r.reasoning)]) == x["reasoning"]
        assert TOK.decode(r.ids[slice(*r.answer)]) == x["content"]
