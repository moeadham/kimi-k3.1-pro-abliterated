"""tools/quick_bank.py matches the planted direction and the fold / mixed rules."""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import quick_bank  # noqa: E402


def test_quick_bank_recovers_planted_directions(tmp_path):
    S, d = 3, 16
    g = torch.Generator().manual_seed(0)
    planted = torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0)
    (tmp_path / "states").mkdir()
    idx = []
    for i in range(80):
        cens = i % 2 == 0
        st = torch.randn(8, S, d, generator=g) * 0.3 + (2.0 * planted if cens else 0)
        save_file({"states": st.to(torch.bfloat16)}, str(tmp_path / "states" / f"{i}.st"))
        ps = list(range(10, 18))
        sets = {"prompt_end": ps[:2], "onset": ps[2:6] if cens else [], "comply_onset": [] if cens else ps[2:6]}
        idx.append({"i": i, "file": f"{i}.st", "positions": ps, "position_sets": sets, "censored": cens,
                    "fold": "val" if i % 5 == 0 else "train", "mixed_prompt": i < 20})
    (tmp_path / "index.jsonl").write_text("\n".join(json.dumps(x) for x in idx))
    out = tmp_path / "bank.pt"
    quick_bank.main(str(tmp_path), str(out), "prompt_end/all", "onset_vs_comply/all", "mixed/onset_vs_comply")
    bank = torch.load(out)
    for f in ("prompt_end/all", "onset_vs_comply/all", "mixed/onset_vs_comply"):
        assert bank[f]["mean_diff"].shape == (S, d)
        assert float(bank[f]["mean_diff"][1] @ planted) > 0.9
