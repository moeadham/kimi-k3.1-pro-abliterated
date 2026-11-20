"""CPU checks for harness/judge.py on allmodels' Decisions endpoint: OpenAI-style question and answer translation,
the request body, the API key check, and retries on rate limits. HTTP is faked with httpx.MockTransport."""
import asyncio
import json

import httpx
import pytest

from harness import judge as jm
from harness.response_judge import RESPONSE_Q

Q = {"refused": {"type": "noul", "instructions": "Refuses?"},
     "label": {"type": "choice", "instructions": "Which?", "criteria": {"factual": "f", "refusal": "r"}},
     "factuality": {"type": "score", "instructions": "How much?", "criteria": ["None", "Some", "Full"]}}


def test_to_oai_questions():
    q = jm.to_oai_questions(Q)
    assert q[0] == {"type": "predicate", "name": "refused", "instructions": "Refuses?"}
    assert q[1]["choices"] == [{"value": "factual", "description": "f"}, {"value": "refusal", "description": "r"}]
    assert q[2]["levels"] == [{"label": "None"}, {"label": "Some"}, {"label": "Full"}]
    assert [x["name"] for x in jm.to_oai_questions(RESPONSE_Q)] == list(RESPONSE_Q)


def test_from_oai_answers():
    a = jm.from_oai_answers([
        {"type": "predicate", "name": "refused", "probability": 0.98},
        {"type": "choice", "name": "label", "choice": "refusal", "confidence": 1,
         "probabilities": [{"value": "factual", "probability": 0.1}, {"value": "refusal", "probability": 0.9}]},
        {"type": "score", "name": "factuality", "score": 1, "confidence": 0.8,
         "probabilities": [{"label": "None", "value": 0, "probability": 0.1}, {"label": "Some", "value": 1, "probability": 0.8}]},
        {"type": "refusal", "name": "other"}])
    assert a["refused"]["noul"] == 0.98
    assert a["label"]["choice"] == "refusal" and a["label"]["probabilities"] == {"factual": 0.1, "refusal": 0.9}
    assert a["factuality"]["score"] == 1 and a["factuality"]["probabilities"]["Some"] == 0.8
    assert a["other"] == {"type": "refusal"}


def judge_with(handler, monkeypatch):
    monkeypatch.setenv("ALLMODELS_API_KEY", "k")
    j = jm.JevJudge(retries=3)
    j.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return j


def test_allmodels_request_and_verdict(monkeypatch):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append((str(req.url), body))
        return httpx.Response(200, json={"model": "typesafe/jev", "usage": {"total_tokens": 42}, "answers": [
            {"type": "predicate", "name": "deflects", "probability": 0.2},
            {"type": "choice", "name": "label", "choice": "factual",
             "probabilities": [{"value": "factual", "probability": 0.9}]},
            {"type": "score", "name": "factuality", "score": 3, "probabilities": []}]})
    j = judge_with(handler, monkeypatch)
    v = asyncio.run(j.judge("q", "chunk"))
    url, body = seen[0]
    assert url == "https://api.allmodels.io/oai/decisions" and "input" in body and "state" not in body
    assert isinstance(body["questions"], list) and body["model"] == "typesafe/jev-1.14"
    assert (v.deflects, v.label, v.factuality, v.error) == (0.2, "factual", 3.0, None)


def test_rate_limit_is_retried(monkeypatch):
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return httpx.Response(200, json={"answers": [{"type": "predicate", "name": "refused", "probability": 0.5}]})
    j = judge_with(handler, monkeypatch)
    d = asyncio.run(j.decide("S", {"refused": Q["refused"]}))
    assert len(calls) == 3 and d["error"] is None and d["answers"]["refused"]["noul"] == 0.5


def test_bad_request_stops_early(monkeypatch):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, json={"error": "bad"})
    j = judge_with(handler, monkeypatch)
    d = asyncio.run(j.decide("S", {"refused": Q["refused"]}))
    assert d["error"].startswith("HTTP 400") and len(calls) == 2


def test_api_key_required(monkeypatch):
    monkeypatch.delenv("ALLMODELS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALLMODELS_API_KEY"):
        jm.JevJudge()
    monkeypatch.setenv("ALLMODELS_API_KEY", "k")
    assert jm.JevJudge().url == "https://api.allmodels.io/oai/decisions"
