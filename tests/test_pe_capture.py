"""CPU check for tools/pe_capture.py with a fake engine: labels from judged gens, edit applied and restored,
last-k positions, index with decoded tokens, resume skips captured rows."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pe_capture  # noqa: E402


class Tok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def test_capture_with_edit_and_labels_from(tmp_path, monkeypatch):
    from harness import k3chat, refusal, tp

    data = tmp_path / "h.jsonl"
    data.write_text("".join(json.dumps({"id": f"a{i}", "prompt": f"question {i}"}) + "\n" for i in range(3)))
    gens = tmp_path / "gens.jsonl"
    gens.write_text("".join(json.dumps({"id": f"harmful:b:a{i}", "flagged": i == 0}) + "\n" for i in range(3)))
    bank = tmp_path / "bank.pt"
    torch.save({"f": {"mean_diff": torch.randn(187, 8)}}, bank)
    calls = []
    monkeypatch.setattr(k3chat, "prompt_ids", lambda tok, q, effort="max", thinking=True: list(range(10 + len(q))))
    monkeypatch.setattr(tp, "edit_writers", lambda llm, Q, lam: calls.append(("edit", tuple(Q.shape))))
    monkeypatch.setattr(tp, "restore_writers", lambda llm: calls.append(("restore",)))

    def cap(llm, ids, positions, path, streams=None):
        calls.append(("cap", positions[0], len(streams)))
        Path(path).write_bytes(b"x")
    monkeypatch.setattr(tp, "capture_one", cap)
    monkeypatch.setattr("harness.session._file_positions", lambda p, e: None)
    monkeypatch.setattr(refusal, "load_prompts", lambda path, split: [
        {"id": f"{split}:{json.loads(l)['id']}", "question": json.loads(l)["prompt"], "split": split}
        for l in open(data)])
    cfg = {"out": str(tmp_path / "pe"), "sets": [{"tag": "b", "file": "h.jsonl", "split": "harmful"}], "k": 4,
           "streams": [100, 120], "labels_from": str(gens),
           "edit": {"bank": str(bank), "family": "f", "layers": [54, 60], "lam": 1.0}}
    res = pe_capture.run(None, Tok(), lambda *_: None, cfg)
    assert res == {**res, "rows": 3, "edited": True}
    assert calls[0] == ("edit", (187, 1, 8)) and calls[-1] == ("restore",)
    caps = [c for c in calls if c[0] == "cap"]
    assert caps[0] == ("cap", 16, 20)  # prompt "question 0" -> 20 ids, last 4 -> starts at 16; 20 streams
    idx = [json.loads(l) for l in open(tmp_path / "pe" / "index.jsonl")]
    assert [x["flagged"] for x in idx] == [True, False, False] and idx[0]["tokens"] == ["<16>", "<17>", "<18>", "<19>"]


def test_refuse_mass():
    import math
    top = [(" I", math.log(0.6)), ("Sure", math.log(0.3)), ("I", math.log(0.05))]
    assert abs(pe_capture.refuse_mass(top, ["I"]) - 0.65) < 1e-9


def test_index_is_written_row_by_row(tmp_path, monkeypatch):
    """A hard stop mid-capture leaves an index that lists exactly the rows captured so far."""
    from harness import k3chat, refusal, tp

    monkeypatch.setattr(k3chat, "prompt_ids", lambda tok, q, effort="max", thinking=True: list(range(12)))
    n = []

    def cap(llm, ids, positions, path, streams=None):
        if len(n) == 2:
            raise KeyboardInterrupt  # not caught by the per-row handler, like a killed process
        n.append(path)
        Path(path).write_bytes(b"x")
    monkeypatch.setattr(tp, "capture_one", cap)
    monkeypatch.setattr("harness.session._file_positions", lambda p, e: None)
    monkeypatch.setattr(refusal, "load_prompts", lambda path, split: [
        {"id": f"{split}:{i}", "question": "q", "split": split} for i in range(5)])
    monkeypatch.setattr(pe_capture, "label_pass", lambda *a, **k: {})
    cfg = {"out": str(tmp_path / "pe"), "sets": [{"tag": "b", "file": "h.jsonl", "split": "harmful"}], "k": 4}
    try:
        pe_capture.run(None, Tok(), lambda *_: None, cfg)
    except KeyboardInterrupt:
        pass
    idx = [json.loads(l) for l in open(tmp_path / "pe" / "index.jsonl")]
    assert [x["file"] for x in idx] == ["00000.safetensors", "00001.safetensors"]
