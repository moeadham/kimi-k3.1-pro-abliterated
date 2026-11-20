import json

import pytest

from harness import bfcl

ITEM = {"category": "parallel",
        "functions": [{"name": "spotify.play", "description": "d", "parameters": {"type": "dict", "properties": {
            "artist": {"type": "string"}, "duration": {"type": "integer"}, "shuffle": {"type": "boolean"}},
            "required": ["artist", "duration"]}}],
        "ground_truth": [{"spotify.play": {"artist": ["Taylor Swift"], "duration": [20], "shuffle": [False, ""]}},
                         {"spotify.play": {"artist": ["Maroon 5"], "duration": [15], "shuffle": [False, ""]}}]}


def test_schema_conversion_and_names():
    t = bfcl.to_openai_tools(ITEM["functions"])[0]["function"]
    assert t["name"] == "spotify_play" and t["parameters"]["type"] == "object"


def test_score_parallel_any_order_and_normalization():
    ok = [("spotify_play", {"artist": "maroon 5", "duration": 15}), ("spotify_play", {"artist": "Taylor Swift", "duration": 20.0})]
    assert bfcl.score(ITEM, ok)["correct"]
    assert not bfcl.score(ITEM, ok[:1])["correct"]
    bad = [("spotify_play", {"artist": "Maroon 5", "duration": 16}), ok[1]]
    assert not bfcl.score(ITEM, bad)["correct"]
    extra = [("spotify_play", {"artist": "Maroon 5", "duration": 15, "volume": 3}), ok[1]]
    assert not bfcl.score(ITEM, extra)["correct"]


def test_calls_from_openai():
    msg = {"tool_calls": [{"function": {"name": "f", "arguments": json.dumps({"a": 1})}}]}
    assert bfcl.calls_from_openai(msg) == [("f", {"a": 1})]


def test_xtml_round_trip_through_k3_template():
    replay = pytest.importorskip("harness.replay")
    try:
        tok = replay.load_tokenizer()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {e}")
    calls = [{"id": "c1", "type": "function", "function": {"name": "spotify_play", "arguments": json.dumps(
        {"artist": "Taylor Swift", "duration": 20, "tags": ["a", "b"], "opts": {"x": True}})}}]
    ids = tok.apply_chat_template([{"role": "user", "content": "go"},
                                   {"role": "assistant", "reasoning_content": "r", "content": "", "tool_calls": calls}],
                                  tools=bfcl.to_openai_tools(ITEM["functions"]), tokenize=True, add_generation_prompt=False)
    text = tok.decode(list(ids["input_ids"] if isinstance(ids, dict) else ids))
    assert bfcl.calls_from_xtml(text) == [("spotify_play", {"artist": "Taylor Swift", "duration": 20, "tags": ["a", "b"],
                                                            "opts": {"x": True}})]
