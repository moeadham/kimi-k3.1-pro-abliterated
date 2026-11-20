import pytest
import torch
import torch.nn.functional as F

from harness.ablation import ablate_embedding, ablate_output_weight, project_activation
from harness.directions import mean_diff, pca_directions, project_out_good, winsorize
from harness.kernel import WeightKernel

torch.manual_seed(0)
D, DIN, V = 64, 48, 100


def unit(d=D):
    return F.normalize(torch.randn(d), dim=0)


def test_ablated_writer_has_no_output_along_r():
    W, r = torch.randn(D, DIN), unit()
    Wp = ablate_output_weight(W, r)
    x = torch.randn(10, DIN)
    assert (x @ Wp.T @ r).abs().max() < 1e-4


def test_lambda_zero_is_noop_and_idempotent_at_one():
    W, r = torch.randn(D, DIN), unit()
    assert torch.equal(ablate_output_weight(W, r, 0.0), W)
    once = ablate_output_weight(W, r)
    assert torch.allclose(ablate_output_weight(once, r), once, atol=1e-5)


def test_partial_lambda_scales_component():
    W, r = torch.randn(D, DIN), unit()
    x = torch.randn(DIN)
    before = (W @ x) @ r
    after = (ablate_output_weight(W, r, 0.25) @ x) @ r
    assert torch.allclose(after, 0.75 * before, atol=1e-4)


def test_weight_edit_equals_activation_projection():
    W, r = torch.randn(D, DIN), unit()
    x = torch.randn(5, DIN)
    lhs = x @ ablate_output_weight(W, r, 0.7).T
    rhs = project_activation(x @ W.T, r, 0.7)
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_multidirection_basis():
    W = torch.randn(D, DIN)
    Q = torch.linalg.qr(torch.randn(D, 3))[0].T  # [3, D]
    Wp = ablate_output_weight(W, Q)
    assert (Q @ Wp).abs().max() < 1e-4


def test_row_norm_preservation():
    W, r = torch.randn(D, DIN), unit()
    Wp = ablate_output_weight(W, r, preserve_row_norms=True)
    assert torch.allclose(Wp.norm(dim=1), W.norm(dim=1), atol=1e-4)


def test_bf16_dtype_preserved():
    W = torch.randn(D, DIN, dtype=torch.bfloat16)
    assert ablate_output_weight(W, unit()).dtype == torch.bfloat16


def test_embedding_rows_orthogonal():
    E, r = torch.randn(V, D), unit()
    assert (ablate_embedding(E, r) @ r).abs().max() < 1e-4


def test_mean_diff_recovers_planted_direction():
    L, n = 3, 400
    r = unit()
    good = torch.randn(n, L, D)
    bad = torch.randn(n, L, D) + 3.0 * r
    d, _, _ = mean_diff(bad, good)
    assert (d @ r).min() > 0.9


def test_project_out_good_is_orthogonal_to_good_mean():
    L, n = 2, 200
    good = torch.randn(n, L, D) + 2.0 * unit()
    bad = torch.randn(n, L, D) + 2.0 * unit()
    d, _, gm = mean_diff(bad, good)
    p = project_out_good(d, gm)
    assert ((p * F.normalize(gm, dim=-1)).sum(-1)).abs().max() < 1e-5
    assert torch.allclose(p.norm(dim=-1), torch.ones(L), atol=1e-5)


def test_pca_basis_orthonormal_and_spans_modes():
    L, n = 2, 300
    a, b = unit(), unit()
    good = torch.randn(n, L, D) * 0.1
    coeff = torch.randn(n, 2) * 3
    bad = good + coeff[:, :1, None] * a + coeff[:, 1:, None] * b
    Q = pca_directions(bad, good, k=2)
    assert Q.shape == (L, 2, D)
    eye = Q @ Q.transpose(1, 2)
    assert torch.allclose(eye, torch.eye(2).expand(L, 2, 2), atol=1e-4)
    for v in (a, b):  # each planted mode lies (almost) in the span
        assert ((Q @ v).norm(dim=-1) > 0.95).all()


def test_winsorize_clamps():
    x = torch.randn(4, 2, D)
    x[0, 0, 0] = 1000.0
    assert winsorize(x, 0.9).abs().max() < 1000.0
    assert torch.equal(winsorize(x, 1.0), x)


def test_weight_kernel_shape():
    k = WeightKernel(max_weight=1.0, peak=5, min_weight=0.2, falloff=3)
    w = k.weights(12)
    assert w[5] == pytest.approx(1.0)
    assert w[8] == pytest.approx(0.2) and w[2] == pytest.approx(0.2)
    assert w[9] == 0 and w[1] == 0
