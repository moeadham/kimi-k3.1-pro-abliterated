"""CPU checks for tools/pe_bank.py on a fake prompt-end capture: classes, positions, projections, split half."""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pe_bank  # noqa: E402

K, S0, S1, D = 4, 100, 110, 128


def fake_capture(tmp_path, n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    signal = torch.nn.functional.normalize(torch.randn(D, generator=g), dim=0)
    common = torch.randn(D, generator=g) * 20  # large shared mean, like the residual stream
    (tmp_path / "states").mkdir()
    index = []
    for i in range(n):
        kind = ["ref", "refuse", "answer"][i % 3]
        st = common + torch.randn(K, S1 - S0, D, generator=g) * 0.5
        if kind == "refuse":
            st[-1] += 3 * signal  # signal only at the last position
        save_file({"states": st.to(torch.bfloat16)}, str(tmp_path / "states" / f"{i:05d}.safetensors"))
        index.append({"id": f"x:{i}", "set": "reference" if kind == "ref" else "build",
                      "split": "harmful" if kind == "refuse" else "harmless", "file": f"{i:05d}.safetensors",
                      "positions": list(range(30, 30 + K)), "streams": [S0, S1],
                      "p_refuse_start": {"refuse": 0.9, "answer": 0.02, "ref": 0.01}[kind]})
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in index))
    return signal


def test_bank_recovers_signal_and_writes_variants(tmp_path):
    signal = fake_capture(tmp_path)
    out = tmp_path / "bank.pt"
    rep = pe_bank.build(str(tmp_path), str(out), positions=["-1", "-4:-2"], pcs=[2], chunk=4)
    bank = torch.load(out)
    fam = bank["h3/-1"]
    assert set(fam) == {"mean_diff", "projected", "projected_ref", "pc2"}
    assert fam["mean_diff"].shape == (187, D) and fam["mean_diff"][:S0].abs().sum() == 0
    cos = (fam["mean_diff"][S0:S1] @ signal)
    assert float(cos.min()) > 0.8  # the signal at the last position is found on every stream
    assert float((bank["h3/-4:-2"]["mean_diff"][S0:S1] @ signal).abs().mean()) < 0.3  # no signal earlier
    assert rep["refusing"] == 20 and rep["answering"] == 20 and rep["ref_rows"] == 20
    assert min(rep["split_half_cos"]["-1"]) > 0.4 and len(rep["split_half_cos"]["-1"]) == S1 - S0
    assert (tmp_path / "bank.json").exists()


def test_positions_and_labels():
    assert pe_bank.parse_positions("-1", 8) == [7] and pe_bank.parse_positions("-4:-1", 8) == [4, 5, 6, 7] == pe_bank.parse_positions("4:1", 8)
    with pytest.raises(ValueError):
        pe_bank.parse_positions("-9", 8)
    x = {"split": "harmful", "p_refuse_start": 0.3, "flagged": False}
    assert pe_bank.classify(x, "first_token", 0.5, 0.1) is None  # ambiguous: dropped
    assert pe_bank.classify(x, "prompt", 0.5, 0.1) is True and pe_bank.classify(x, "jev", 0.5, 0.1) is False


def test_run1_format_and_pc_projection(tmp_path):
    """Run-1 `refusal` captures: prompt_end picked among other positions, folds as sets, PCs from answering rows."""
    g = torch.Generator().manual_seed(1)
    signal = torch.nn.functional.normalize(torch.randn(D, generator=g), dim=0)
    common = torch.randn(D, generator=g) * 20
    (tmp_path / "states").mkdir()
    index = []
    for i in range(40):
        refused = i % 2 == 0
        st = common + torch.randn(7, 187, D, generator=g) * 0.5  # 7 positions: 4 prompt_end + 3 others
        if refused:
            st[:4] += 3 * signal
        save_file({"states": st.to(torch.bfloat16)}, str(tmp_path / "states" / f"{i:05d}.safetensors"))
        index.append({"key": f"k{i}", "file": f"{i:05d}.safetensors", "positions": [10, 11, 12, 13, 20, 30, 40],
                      "position_sets": {"prompt_end": [10, 11, 12, 13], "answer_head": [20, 30, 40]},
                      "censored": refused, "fold": "val" if i >= 36 else "train", "prompt_id": f"harmful:p{i // 2}",
                      "dataset_split": "harmful" if refused else "harmless"})
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in index))
    out = tmp_path / "b.pt"
    rep = pe_bank.build([str(tmp_path)], str(out), label="jev", positions=["-4:-1"], ref_sets=[], pcs=[2],
                        class_sets=["train"], chunk=64)
    fam = torch.load(out)["h3/-4:-1"]
    assert rep["refusing"] == 18 and rep["answering"] == 18 and rep["ref_from"] == "answering rows"
    assert float((fam["mean_diff"][100] @ signal)) > 0.8 and "pc2" in fam
    common_u = torch.nn.functional.normalize(common, dim=0)
    assert abs(float(fam["pc2"][100] @ common_u)) < 0.05  # the dominant shared direction is projected out


def test_jev_label_drops_ambiguous_cutoff_band():
    import pe_bank
    c = lambda **x: pe_bank.classify(x, "jev", 0.5, 0.1)  # noqa: E731
    assert c(flagged=True) is True and c(flagged=False) is False
    assert c(flagged=True, cutoff=True, score=0.9) is True
    assert c(flagged=True, cutoff=True, score=0.7) is None  # misread-deflection band
    assert c(flagged=False, cutoff=True, score=0.4) is None
    assert c(flagged=False, cutoff=True, score=0.2) is False
