import torch

from harness.tp import rank_agreement


def _res(l1, sample, cs=1.0):
    return {"l1": l1, "sample": torch.tensor(sample), "checksum": cs, "shape": [2, 3, 4]}


def test_identical_ranks_agree_bitwise():
    a = rank_agreement([_res(10.0, [1.0, -2.0, 3.0]), _res(10.0, [1.0, -2.0, 3.0])])
    assert a == {"l1_rel_gap": 0.0, "sample_rel_dist": 0.0, "bitwise_checksums": True}


def test_small_noise_is_small():
    # row 329 at TP2: L1 10043686 vs 10046244, most sampled elements identical
    a = rank_agreement([_res(10043686.0, [1.0, -2.0, 3.0, 4.0], 1.0), _res(10046244.0, [1.0, -2.0, 3.05, 4.0], 2.0)])
    assert a["l1_rel_gap"] < 1e-3 and a["sample_rel_dist"] < 0.05 and not a["bitwise_checksums"]


def test_sharded_stream_is_large():
    a = rank_agreement([_res(10.0, [1.0, -2.0, 3.0, 4.0]), _res(5.0, [0.0, 0.0, -3.0, 9.0])])
    assert a["l1_rel_gap"] > 0.05 and a["sample_rel_dist"] > 0.05
