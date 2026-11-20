"""Copy chosen families of a directions bank into a small file (mean_diff + projected only, float32).

  python tools/extract_bank.py /vol/RUN/directions_refusal/directions.pt /vol/banks/NAME.pt fam1 fam2 ...
Used to hand directions to a running GPU sandbox, whose Volume view does not include files written later by
other sandboxes: the small file is downloaded and pushed with modal_gpu.py put.
"""
import sys

import torch

src, dst, *fams = sys.argv[1:]
bank = torch.load(src, map_location="cpu")
out = {}
for f in fams:
    if f not in bank:
        print(f"skip {f!r}: not in bank ({sorted(bank)[:12]}...)")
        continue
    out[f] = {k: bank[f][k].float().contiguous() for k in ("mean_diff", "projected") if bank[f].get(k) is not None}
torch.save(out, dst)
print(f"{dst}: {sorted(out)}")
