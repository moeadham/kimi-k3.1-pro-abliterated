"""Parameter count of a safetensors checkpoint from its shard headers (no tensor data read).
MXFP4 `*.weight_packed` (U8) hold 2 values per byte; `*.weight_scale` are block scales, counted apart.

  python tools/count_params.py /vol/Kimi-K3.1-Pro
"""
import json
import struct
import sys
from collections import Counter
from math import prod
from pathlib import Path

d = Path(sys.argv[1])
c = Counter()
for sh in sorted(d.glob("*.safetensors")):
    with open(sh, "rb") as f:
        hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        n = prod(v["shape"])
        if k.endswith("weight_packed"):
            c["routed_expert_params"] += 2 * n
        elif k.endswith("weight_scale"):
            c["expert_scales"] += n
        elif k.startswith("vision_tower") or "mm_projector" in k or "multi_modal" in k:
            c["vision_params"] += n
        else:
            c["dense_params_" + v["dtype"]] += n
c["language_params"] = c["routed_expert_params"] + sum(v for k, v in c.items() if k.startswith("dense_params_"))
print(json.dumps(dict(c), indent=1))
