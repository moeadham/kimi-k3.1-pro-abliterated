"""Directions bank for edits from the `forks` anchor captures (all stream indices), for tools/quick_probe.py.

Families (mean difference, refused vs answered, per stream index; train = all rows):
  nt_fork/all        the state producing the fork token (refusal word vs start of the compliance sentence)
  nt_prompt_end/all  the last prompt token in non-thinking mode (it produces the first answer token)
Optionally only rows whose fork the branch test confirmed (--confirmed-only).

  python tools/fork_bank.py /vol/RUN/forks OUT.pt [--confirmed-only]
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file


def main(cap: str, out: str, confirmed_only: bool = False, workers: int = 16) -> dict:
    cap = Path(cap)
    torch.set_num_threads(1)  # each reader thread otherwise starts its own torch pool (863 threads on 32 CPUs)
    index = [json.loads(l) for l in open(cap / "index.jsonl")]
    if confirmed_only:
        index = [x for x in index if x.get("fork_confirmed") or not x.get("flagged")]
    sums: dict = {}

    def one(x):
        st = load_file(str(cap / "states" / x["anchor_file"]))["states"]
        where = {p: j for j, p in enumerate(x["anchor_positions"])}
        got = {"nt_prompt_end/all": st[where[x["prompt_len"] - 1]].double()}
        if x.get("fork_state") is not None and x["fork_state"] in where:
            got["nt_fork/all"] = st[where[x["fork_state"]]].double()
        return bool(x["flagged"]), got

    with ThreadPoolExecutor(workers) as pool:
        for c, got in pool.map(one, index):
            for fam, v in got.items():
                s, n = sums.get((fam, c), (0, 0))
                sums[(fam, c)] = (s + v, n + 1)
    bank = {}
    for fam in ("nt_fork/all", "nt_prompt_end/all"):
        if (fam, True) in sums and (fam, False) in sums:
            (sb, nb), (sg, ng) = sums[(fam, True)], sums[(fam, False)]
            gm = (sg / ng).float()
            r = torch.nn.functional.normalize((sb / nb).float() - gm, dim=-1)
            g = torch.nn.functional.normalize(gm, dim=-1)
            proj = torch.nn.functional.normalize(r - (r * g).sum(-1, keepdim=True) * g, dim=-1)
            bank[fam] = {"mean_diff": r, "projected": proj}
            print(f"{fam}: refused {nb} / answered {ng}")
    torch.save(bank, out)
    return bank


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], confirmed_only="--confirmed-only" in sys.argv)
