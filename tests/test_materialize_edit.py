"""CPU check for tools/materialize_edit.py on a tiny fake checkpoint: only the named writers of the edited layers
change, they equal ablation.ablate_output_weight, every other tensor (other dtypes included) is byte-identical,
non-weight files are copied, and a rerun resumes."""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import materialize_edit as me  # noqa: E402

from harness import ablation  # noqa: E402

D, S = 8, 187
P = "language_model.model.layers"


def make_base(tmp: Path):
    base = tmp / "base"
    base.mkdir()
    g = torch.Generator().manual_seed(0)
    shards = {
        "model-1.safetensors": {f"{P}.48.self_attn.o_proj.weight": torch.randn(D, 12, generator=g).bfloat16(),
                                f"{P}.48.block_sparse_moe.routed_expert_up_proj.weight": torch.randn(D, 6, generator=g).bfloat16(),
                                f"{P}.48.block_sparse_moe.experts.0.w2.weight": torch.randint(0, 255, (D, 4), dtype=torch.uint8),
                                f"{P}.48.self_attn.q_proj.weight": torch.randn(12, D, generator=g).bfloat16()},
        "model-2.safetensors": {f"{P}.48.block_sparse_moe.shared_experts.down_proj.weight": torch.randn(D, 10, generator=g).bfloat16(),
                                f"{P}.30.self_attn.o_proj.weight": torch.randn(D, 12, generator=g).bfloat16()},
        "model-3.safetensors": {f"{P}.10.block_sparse_moe.shared_experts.down_proj.weight": torch.randn(D, 10, generator=g).bfloat16()},
    }
    wmap = {}
    for sh, ts in shards.items():
        save_file(ts, str(base / sh), metadata={"format": "pt"})
        wmap.update({k: sh for k in ts})
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}))
    (base / "config.json").write_text("{}")
    (base / "assets").mkdir()
    (base / "assets" / "a.txt").write_text("x")
    (base / ".cache").mkdir()
    bank = tmp / "bank.pt"
    torch.save({"f": {"v": torch.randn(S, D, generator=g)}}, bank)
    spec = tmp / "spec.json"
    spec.write_text(json.dumps({"bank": str(bank), "family": "f", "variant": "v", "layers": [40, 50], "lam": 1.5}))
    return base, shards, spec


def test_materialize_edits_only_writers_and_verifies(tmp_path):
    base, shards, spec = make_base(tmp_path)
    out = tmp_path / "out"
    me.main(base, out, str(spec), workers=2)
    _, Q, lam = me.load_edit(str(spec))
    edited = {f"{P}.48.self_attn.o_proj.weight", f"{P}.48.block_sparse_moe.routed_expert_up_proj.weight",
              f"{P}.48.block_sparse_moe.shared_experts.down_proj.weight"}
    for sh, ts in shards.items():
        with safe_open(str(out / sh), "pt") as f:
            for name, t in ts.items():
                got = f.get_tensor(name)
                if name in edited:
                    want = ablation.ablate_output_weight(t, Q[me.writer_slot(name)], float(lam[me.writer_slot(name)]))
                    assert torch.equal(got, want) and not torch.equal(got, t)
                else:
                    assert torch.equal(got, t)  # other layers, inputs, experts (uint8) untouched
    e = json.loads((out / "EDIT.json").read_text())
    assert e["edited_tensors"] == 3 and e["verify"]["problems"] == [] and (out / "DONE").exists()
    assert (out / "config.json").exists() and (out / "assets" / "a.txt").exists() and not (out / ".cache").exists()
    n = len((out / "progress.jsonl").read_text().splitlines())
    me.main(base, out, str(spec), workers=2)  # rerun: nothing recopied
    assert len((out / "progress.jsonl").read_text().splitlines()) == n


def test_writer_slot():
    assert me.writer_slot(f"{P}.48.self_attn.o_proj.weight") == 97
    assert me.writer_slot(f"{P}.48.block_sparse_moe.shared_experts.down_proj.weight") == 98
    assert me.writer_slot(f"{P}.48.block_sparse_moe.experts.3.down_proj.weight") is None
    assert me.writer_slot(f"{P}.48.self_attn.q_proj.weight") is None
