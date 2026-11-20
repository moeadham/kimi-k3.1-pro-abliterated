"""Capture specific replay rows at TP>1 and report cross-rank agreement (pod/Modal).

Usage: python tests_gpu/capture_rank_agree.py /vol/tiny-k3/mxfp4 --tp 2 --rows 329 330 331 [--tol 0.05]
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--rows", type=int, nargs="+", default=[330])
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--lengths", type=int, nargs="*", default=[], help="add first row with len >= each L")
    ap.add_argument("--no-custom-ar", action="store_true", help="disable_custom_all_reduce=True")
    a = ap.parse_args()
    os.environ["K3H_STEERING"] = "1"
    os.environ["K3H_STEERING_RANK"] = "8"
    from harness import tp  # noqa: E402
    from harness.session import replay_rows  # noqa: E402
    from harness.vllm_launch import make_llm  # noqa: E402

    llm = make_llm(a.model, eager=True, max_num_seqs=1, gpu_memory_utilization=0.85, tensor_parallel_size=a.tp,
                   max_model_len=a.max_model_len, max_num_batched_tokens=a.max_model_len,
                   **({"disable_custom_all_reduce": True} if a.no_custom_ar else {}))
    rows = replay_rows()
    idx = list(a.rows)
    for L in a.lengths:
        idx.append(min((i for i, r in enumerate(rows) if len(r["ids"]) >= L), key=lambda i: len(rows[i]["ids"])))
    for i in idx:
        row = rows[i]
        pos = sorted({p for v in row["positions"].values() for p in v})
        for rep in range(2):
            try:
                r = tp.capture_one(llm, row["ids"], pos, None, tol=a.tol)
                print(json.dumps({"row": i, "rep": rep, "len": len(row["ids"]), "n_pos": len(pos), "ok": True,
                                  "checksum": r["checksum"], "l1": r["l1"], **r["rank_agreement"]}), flush=True)
            except RuntimeError as e:
                print(json.dumps({"row": i, "rep": rep, "len": len(row["ids"]), "n_pos": len(pos), "ok": False,
                                  "err": str(e)[:600]}), flush=True)


if __name__ == "__main__":
    main()
