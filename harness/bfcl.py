"""BFCL (Berkeley Function Calling Leaderboard, v3, Apache-2.0) subset: tool-calling sanity check.

Categories used: simple (1 function, 1 call), multiple (several functions, pick 1),
parallel (1 function, several calls). Scoring follows BFCL's AST checker in spirit:
- the right function(s) are called, the right number of times (parallel: any order),
- every required parameter is present, no unknown parameters are passed,
- each value is one of the allowed ground-truth values ("" in the allowed list = may be
  omitted); strings compared case-insensitively ignoring spaces and simple punctuation
  (BFCL's standardization), ints accepted for floats.

Two transports:
- hosted (OpenAI-style `tools` + `message.tool_calls`): `to_openai_tools`, `calls_from_openai`
- our vLLM K3.1 engine: the chat template renders `tools` itself (tool-declare system
  message), and the model emits XTML after the response channel:
    <|open|>tools<|sep|><|open|>call tool="NAME" index="1"<|sep|>
      <|open|>argument key="K" type="T"<|sep|>VALUE<|close|>argument<|sep|> ...
    <|close|>call<|sep|><|close|>tools<|sep|>
  `calls_from_xtml` parses that (also the <json type="object"> fallback form).
"""

from __future__ import annotations

import html
import json
import re

TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "integer": "integer", "string": "string",
            "boolean": "boolean", "array": "array", "number": "number", "object": "object"}


def safe_name(name: str) -> str:
    """OpenAI tool names must match ^[a-zA-Z0-9_-]+$ (BFCL does the same '.' -> '_' mapping)."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _fix_schema(s):
    if isinstance(s, dict):
        out = {}
        for k, v in s.items():
            if k == "type" and isinstance(v, str):
                if v == "any":
                    continue
                out[k] = TYPE_MAP.get(v, v)
            elif k in ("optional",):
                continue
            else:
                out[k] = _fix_schema(v)
        return out
    if isinstance(s, list):
        return [_fix_schema(x) for x in s]
    return s


def to_openai_tools(functions: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": safe_name(f["name"]), "description": f.get("description", ""),
                                              "parameters": _fix_schema(f["parameters"])}} for f in functions]


def calls_from_openai(message: dict) -> list[tuple[str, dict]]:
    out = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"__unparseable__": fn.get("arguments")}
        out.append((fn.get("name", ""), args if isinstance(args, dict) else {"__non_object__": args}))
    return out


_CALL = re.compile(r'<\|open\|>call tool="(?P<name>[^"]*)"[^<]*?<\|sep\|>(?P<body>.*?)<\|close\|>call<\|sep\|>', re.S)
_ARG = re.compile(r'<\|open\|>argument key="(?P<key>[^"]*)" type="(?P<type>[^"]*)"<\|sep\|>(?P<val>.*?)<\|close\|>argument<\|sep\|>', re.S)
_JSON = re.compile(r'<\|open\|>json type="object"<\|sep\|>(?P<val>.*?)<\|close\|>json<\|sep\|>', re.S)


def calls_from_xtml(text: str) -> list[tuple[str, dict]]:
    """Parse K3 XTML tool calls from decoded generation text (special tokens kept)."""
    out = []
    for m in _CALL.finditer(text):
        name = html.unescape(m.group("name"))
        body = m.group("body")
        j = _JSON.search(body)
        if j:
            try:
                args = json.loads(j.group("val"))
            except json.JSONDecodeError:
                args = {"__unparseable__": j.group("val")}
        else:
            args = {}
            for a in _ARG.finditer(body):
                key, typ, val = html.unescape(a.group("key")), a.group("type"), a.group("val")
                if typ == "string":
                    args[key] = val
                else:
                    try:
                        args[key] = json.loads(val)
                    except json.JSONDecodeError:
                        args[key] = val
        out.append((name, args))
    return out


def _std(v):
    if isinstance(v, str):
        return re.sub(r"[ ,./\-_*^'\"]", "", v).lower()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_std(x) for x in v]
    if isinstance(v, dict):
        return {k: _std(x) for k, x in v.items()}
    return v


def _value_ok(got, allowed: list) -> bool:
    g = _std(got)
    return any(g == _std(a) for a in allowed if a != "")


def _call_ok(name: str, args: dict, gt_name: str, gt_params: dict, functions: dict) -> bool:
    if name != safe_name(gt_name):
        return False
    fn = functions[gt_name]
    required = set(fn["parameters"].get("required", []))
    for k in args:
        if k not in gt_params:
            return False  # unknown / hallucinated parameter
    for k, allowed in gt_params.items():
        if k in args:
            if not _value_ok(args[k], allowed):
                return False
        elif "" not in allowed and k in required:
            return False
        elif "" not in allowed:
            return False  # ground truth insists on a value
    return True


def score(item: dict, calls: list[tuple[str, dict]]) -> dict:
    """item: {category, functions, ground_truth}. Returns {correct, n_calls, error?}."""
    functions = {f["name"]: f for f in item["functions"]}
    gt = item["ground_truth"]
    if len(calls) != len(gt):
        return {"correct": False, "n_calls": len(calls), "error": f"expected {len(gt)} call(s), got {len(calls)}"}
    used = set()
    for g in gt:
        (gname, gparams), = g.items()
        hit = next((i for i, (n, a) in enumerate(calls) if i not in used and _call_ok(n, a, gname, gparams, functions)), None)
        if hit is None:
            return {"correct": False, "n_calls": len(calls), "error": f"no matching call for {gname}"}
        used.add(hit)
    return {"correct": True, "n_calls": len(calls)}
