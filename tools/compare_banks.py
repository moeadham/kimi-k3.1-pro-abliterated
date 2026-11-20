"""Compare two directions banks (and summaries): mean_diff cosine, PCA subspace overlap, val AUC.

  python tools/compare_banks.py OLD_DIR NEW_DIR
"""
import json
import sys
from pathlib import Path

import torch

old, new = Path(sys.argv[1]), Path(sys.argv[2])
bo, bn = torch.load(old / "directions.pt"), torch.load(new / "directions.pt")
so, sn = (json.loads((p / "summary.json").read_text())["families"] for p in (old, new))
print("families old-only", sorted(set(bo) - set(bn)), "new-only", sorted(set(bn) - set(bo)))
for f in sorted(set(bo) & set(bn)):
    a, b = bo[f]["mean_diff"], bn[f]["mean_diff"]
    nz = a.norm(dim=-1) > 0
    cos = (a[nz] * b[nz]).sum(-1)
    line = f"{f:28s} mean_diff cos min {cos.min():.6f}"
    if bo[f].get("pca") is not None and bn[f].get("pca") is not None:
        ov = [torch.linalg.svdvals(bo[f]["pca"][s] @ bn[f]["pca"][s].T).min().item() for s in range(0, a.shape[0], 10)]
        line += f" | pca subspace overlap min {min(ov):.4f}"
    if f in so and f in sn and "val_auc_max" in so[f]:
        line += f" | val_auc_max {so[f]['val_auc_max']:.6f} vs {sn[f]['val_auc_max']:.6f}, best_s {so[f]['best_s']} vs {sn[f]['best_s']}"
    print(line)
