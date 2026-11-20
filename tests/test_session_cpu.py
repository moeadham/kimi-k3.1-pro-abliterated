"""CPU checks for the session driver helpers and TP edit helpers (no vLLM)."""
import types

import pytest
import torch

from harness import session, steering, tp


def test_auc():
    assert session._auc(torch.tensor([3.0, 4.0]), torch.tensor([1.0, 2.0])) == 1.0
    assert session._auc(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])) == 0.0
    # ties: identical scores carry no information (a degenerate stream index must not look separating)
    assert session._auc(torch.zeros(5), torch.zeros(7)) == 0.5
    assert session._auc(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0])) == 0.875


def test_trial_basis_shapes():
    S, d = 9, 16
    bank = {"f": {"mean_diff": torch.nn.functional.normalize(torch.randn(S, d), dim=-1),
                  "projected": torch.nn.functional.normalize(torch.randn(S, d), dim=-1),
                  "pca": torch.randn(S, 8, d)}}
    p = dict(family="f", kind="pca", rank=3, direction_index=4, per_layer=False, max_weight=1.0, peak=4,
             min_weight=0.2, falloff=3)
    Q, lam = session._trial_basis(bank, p, S)
    assert Q.shape == (S, 3, d) and torch.equal(Q[0], Q[8]) and lam.shape == (S,)
    assert lam[4] == 1.0 and lam[0] == 0.0
    p.update(kind="mean_diff", per_layer=True)
    Q, _ = session._trial_basis(bank, p, S)
    assert Q.shape == (S, 1, d) and torch.equal(Q[2, 0], bank["f"]["mean_diff"][2])


def test_val_prompts_are_val_and_unique():
    c = session.val_prompts("censored", 1000)
    k = session.val_prompts("control", 1000)
    assert c and k and len({x["id"] for x in c}) == len(c)
    assert all(x["id"].startswith("ctrl-") for x in k) and not any(x["id"].startswith("ctrl-") for x in c)
    train_ids = {r["prompt_id"] for r in session.replay_rows("train")}
    assert not (train_ids & {x["id"] for x in c + k})


class ToyInner(torch.nn.Module):
    """Mimics KimiLinearModel parameter names: embed, o_proj, dense layer-0 MLP, MoE writers."""

    def __init__(self, d=8, n_layers=2):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(20, d)
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.o_proj = torch.nn.Linear(6, d, bias=False)
            layer.mlp = torch.nn.Module()
            if i == 0:
                layer.mlp.down_proj = torch.nn.Linear(5, d, bias=False)
            else:
                layer.mlp.routed_expert_up_proj = torch.nn.Linear(4, d, bias=False)
                layer.mlp.shared_experts = torch.nn.Module()
                layer.mlp.shared_experts.down_proj = torch.nn.Linear(3, d, bias=False)
                layer.mlp.experts = torch.nn.Module()
                layer.mlp.experts.w2 = torch.nn.Parameter(torch.randn(4, 4))  # not a writer
            self.layers.append(layer)


def _worker(inner):
    return types.SimpleNamespace(model_runner=types.SimpleNamespace(model=types.SimpleNamespace(model=inner)))


def test_writer_params_and_edit_restore():
    inner = ToyInner()
    names = set(tp.writer_params(inner))
    assert names == {"embed_tokens.weight", "layers.0.self_attn.o_proj.weight", "layers.0.mlp.down_proj.weight",
                     "layers.1.self_attn.o_proj.weight", "layers.1.mlp.routed_expert_up_proj.weight",
                     "layers.1.mlp.shared_experts.down_proj.weight"}
    w = _worker(inner)
    before = {n: p.detach().clone() for n, p in inner.named_parameters()}
    r = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    assert tp.w_edit_writers(w, r, 1.0)["edited"] == 6
    for n, (_, p) in tp.writer_params(inner).items():
        out_dim = -1 if n.startswith("embed") else 0
        proj = (p.detach() @ r.T) if out_dim == -1 else (r @ p.detach())
        assert proj.abs().max() < 1e-5, n  # writer outputs orthogonal to r
    assert torch.equal(inner.layers[1].mlp.experts.w2, before["layers.1.mlp.experts.w2"])
    tp.w_restore_writers(w)
    assert all(torch.equal(p, before[n]) for n, p in inner.named_parameters())


def test_edit_refuses_sharded_output_dim():
    inner = ToyInner()
    inner.layers[0].self_attn.o_proj = torch.nn.Linear(6, 4, bias=False)  # output dim != d_model
    with pytest.raises(RuntimeError, match="lacks full d_model"):
        tp.w_edit_writers(_worker(inner), torch.nn.functional.normalize(torch.randn(1, 8), dim=-1), 1.0)
    tp._ORIG.clear()


def test_slot_equals_stream_index():
    assert steering.slot_of("embed", 0) == 0
    assert steering.slot_of("attn", 3) == 2 * 3 + 1 and steering.slot_of("mlp", 3) == 2 * 3 + 2


def test_directions_stage_on_synthetic_captures(tmp_path):
    """End-to-end directions stage on fake capture files with a planted direction."""
    import json as _json

    from safetensors.torch import save_file

    S, d = 5, 32
    torch.manual_seed(0)  # independent of test order
    planted = torch.nn.functional.normalize(torch.randn(d), dim=0)
    cap = tmp_path / "capture"
    (cap / "states").mkdir(parents=True)
    idx = []
    g = torch.Generator().manual_seed(0)
    for i in range(80):
        cens = i % 2 == 0
        n = 10
        st = torch.randn(n, S, d, generator=g) + (3.0 * planted if cens else 0)
        st[:, 0] = 1.0  # degenerate s = 0 (like template-token embeddings at prompt_end): same for every row
        save_file({"states": st.to(torch.bfloat16)}, cap / "states" / f"{i:05d}.safetensors")
        ps = list(range(100, 100 + n))
        idx.append({"i": i, "file": f"{i:05d}.safetensors", "positions": ps,
                    "position_sets": {"prompt_end": ps[:4], "reasoning_head": ps[4:], "onset": ps[6:8] if cens else [],
                                      "comply_onset": [] if cens else ps[6:8],
                                      "answer_head": ps[8:]},
                    "censored": cens, "lang": "en" if i % 4 < 2 else "zh", "fold": "val" if i >= 64 else "train",
                    "mixed_prompt": i < 8, "prompt_id": f"p{i // 2}", "dataset_split": "x"})
    (cap / "index.jsonl").write_text("\n".join(_json.dumps(x) for x in idx) + "\n")
    args = types.SimpleNamespace(out=tmp_path)
    session.stage_directions(args)
    bank = torch.load(tmp_path / "directions" / "directions.pt")
    summ = _json.loads((tmp_path / "directions" / "summary.json").read_text())
    assert summ["families"]["prompt_end/all"]["val_auc_max"] > 0.99
    assert summ["families"]["prompt_end/all"]["best_s"] != 0  # the degenerate index is never "best"
    assert float(bank["prompt_end/all"]["mean_diff"][2] @ planted) > 0.95
    assert bank["prompt_end/en"]["pca"].shape == (S, 8, d) and "mixed/reasoning_head" in bank
    assert "mixed/onset_vs_comply" in bank and "onset_vs_comply/all" in bank
    assert float(bank["mixed/onset_vs_comply"]["mean_diff"][2] @ planted) > 0.5  # 8 rows, d=32: noisy


def test_directions_both_labelings_from_one_load_match_separate_runs(tmp_path):
    """--label-by both (one capture load, threaded) writes the same banks as two separate runs."""
    import json as _json

    from safetensors.torch import save_file

    S, d = 4, 16
    g = torch.Generator().manual_seed(1)
    cap = tmp_path / "refusal"
    (cap / "states").mkdir(parents=True)
    idx = []
    for i in range(60):
        harmful, flagged = i % 2 == 0, (i % 2 == 0) != (i % 10 == 0)  # labels disagree on 6 rows
        st = torch.randn(6, S, d, generator=g) + (2.0 if flagged else 0) + (1.0 if harmful else 0)
        save_file({"states": st.to(torch.bfloat16)}, cap / "states" / f"{i}.st")
        ps = list(range(6))
        idx.append({"i": i, "file": f"{i}.st", "positions": ps, "position_sets": {"prompt_end": ps[:3],
                    "reasoning_head": ps[3:]}, "censored": flagged, "lang": "en", "fold": "val" if i >= 45 else "train",
                    "mixed_prompt": False, "prompt_id": f"p{i}", "dataset_split": "harmful" if harmful else "harmless"})
    (cap / "index.jsonl").write_text("\n".join(_json.dumps(x) for x in idx) + "\n")
    base = ["directions", "--model", "unused", "--out", str(tmp_path), "--capture-from", "refusal"]
    session.main(base + ["--label-by", "both"])
    both = {k: torch.load(tmp_path / k / "directions.pt") for k in ("directions_refusal", "directions_refusal_prompt")}
    for label, k in (("flag", "directions_refusal"), ("prompt", "directions_refusal_prompt")):
        session.main(base + ["--label-by", label, "--force"])
        sep = torch.load(tmp_path / k / "directions.pt")
        assert torch.allclose(sep["prompt_end/all"]["mean_diff"], both[k]["prompt_end/all"]["mean_diff"])
    assert not torch.allclose(both["directions_refusal"]["prompt_end/all"]["mean_diff"],
                              both["directions_refusal_prompt"]["prompt_end/all"]["mean_diff"])
    session.main(base + ["--label-by", "both"])  # both DONE: skipped, no reload


def test_capture_take_selects_positions_without_stacking_everything():
    """w_capture_take must equal stack(...)[positions] (bitwise) while never building the full [T, S, d] stack."""
    T, S, d = 50, 7, 4
    calls = [torch.randn(T, d) for _ in range(S)] + [torch.randn(1, d)]  # one decode-sized call is ignored
    want = torch.stack(calls[:S], dim=1)[[3, 10, 49]]
    tp._CAP["calls"], tp._CAP["cm"] = calls, types.SimpleNamespace(__exit__=lambda *a: None)
    real_stack = torch.stack

    def guarded(ts, dim=0):
        assert all(t.shape[0] <= 3 for t in ts), "stacked full-length calls"
        return real_stack(ts, dim=dim)
    torch.stack = guarded
    try:
        r = tp.w_capture_take(None, T, [3, 10, 49], None)
    finally:
        torch.stack = real_stack
    assert r["ok"] and r["shape"] == [3, S, d]
    assert r["checksum"] == float(want.float().sum())


def test_capture_take_failure_returns_not_ok_and_clears_state():
    tp._CAP["calls"] = [torch.randn(5, 4) for _ in range(3)]
    exited = []
    tp._CAP["cm"] = types.SimpleNamespace(__exit__=lambda *a: exited.append(1))
    r = tp.w_capture_take(None, 5, [99], None)  # position out of range -> error inside
    assert r["ok"] is False and "Error" in r["error"] and tp._CAP == {} and exited


def test_pca_lowrank_matches_exact_span():
    from harness.directions import pca_directions

    g = torch.Generator().manual_seed(0)
    n, S, d, k = 300, 2, 200, 4
    basis = torch.linalg.qr(torch.randn(d, k, generator=g))[0].T  # k planted directions
    coef = torch.randn(n, k, generator=g) * torch.tensor([8.0, 6.0, 4.0, 3.0])
    bad = (coef @ basis).unsqueeze(1).expand(n, S, d) + 0.05 * torch.randn(n, S, d, generator=g)
    good = 0.05 * torch.randn(n, S, d, generator=g)
    q = pca_directions(bad, good, k=k)  # n, d > 4k: randomized path
    assert q.shape == (S, k, d)
    assert torch.allclose(q[0] @ q[0].T, torch.eye(k), atol=1e-4)
    # same span as the planted basis
    assert torch.linalg.svdvals(q[0] @ basis.T).min() > 0.999


def test_directions_streaming_is_chunk_invariant(tmp_path):
    """Reading 2 stream indices at a time gives the same bank as reading all at once."""
    import json as _json

    from safetensors.torch import save_file

    S, d = 7, 24
    g = torch.Generator().manual_seed(3)
    planted = torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0)
    cap = tmp_path / "refusal"
    (cap / "states").mkdir(parents=True)
    idx = []
    for i in range(90):
        cens = i % 3 != 0
        st = torch.randn(10, S, d, generator=g) + (1.5 * planted if cens else 0)
        save_file({"states": st.to(torch.bfloat16)}, cap / "states" / f"{i}.st")
        ps = list(range(10))
        idx.append({"i": i, "file": f"{i}.st", "positions": ps, "censored": cens, "lang": "en",
                    "fold": "val" if i % 5 == 0 else "train", "mixed_prompt": i < 30, "prompt_id": f"p{i // 3}",
                    "dataset_split": "harmful" if i % 2 else "harmless",
                    "position_sets": {"prompt_end": ps[:3], "reasoning_head": ps[3:8], "onset": ps[8:] if cens else [],
                                      "comply_onset": [] if cens else ps[8:]}})
    (cap / "index.jsonl").write_text("\n".join(_json.dumps(x) for x in idx) + "\n")
    banks = {}
    for chunk in (2, 100):
        out = tmp_path / f"o{chunk}"
        out.mkdir()
        (out / "refusal").symlink_to(cap)
        session.main(["directions", "--model", "unused", "--out", str(out), "--capture-from", "refusal",
                      "--label-by", "both", "--layer-chunk", str(chunk)])
        banks[chunk] = ({k: torch.load(out / k / "directions.pt") for k in ("directions_refusal",
                                                                           "directions_refusal_prompt")},
                        _json.loads((out / "directions_refusal" / "summary.json").read_text()))
    (b2, s2), (b100, s100) = banks[2], banks[100]
    for k in b2:
        assert set(b2[k]) == set(b100[k]) and "mixed/onset_vs_comply" in b2["directions_refusal"]
        for fam in b2[k]:
            assert torch.allclose(b2[k][fam]["mean_diff"], b100[k][fam]["mean_diff"], atol=1e-6)
            if b2[k][fam]["pca"] is not None:  # same seed per stream index -> same basis
                assert torch.allclose(b2[k][fam]["pca"], b100[k][fam]["pca"], atol=1e-5)
    assert s2["families"] == s100["families"]
    assert float(b2["directions_refusal"]["prompt_end/all"]["mean_diff"][3] @ planted) > 0.9


def test_capture_take_stream_subset_and_metadata(tmp_path):
    import json as _json

    from safetensors import safe_open

    T, S, d = 6, 9, 4
    calls = [torch.randn(T, d) for _ in range(S)]
    tp._CAP["calls"], tp._CAP["cm"] = calls, types.SimpleNamespace(__exit__=lambda *a: None)
    out = tmp_path / "x.st"
    r = tp.w_capture_take(None, T, [1, 4], str(out), streams=[0, 4, 8])
    assert r["ok"] and r["shape"] == [2, 3, d]
    with safe_open(str(out), "pt") as f:
        st = f.get_tensor("states").float()
        assert _json.loads(f.metadata()["streams"]) == [0, 4, 8]
    assert torch.allclose(st[1, 2], calls[8][4].to(torch.bfloat16).float())
