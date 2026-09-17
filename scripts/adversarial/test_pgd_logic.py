#!/usr/bin/env python3
"""Unit tests for the hand-written PGD + metric logic (CPU, synthetic objectives).

Checks against closed-form optima for a linear objective, where the eps-ball
maximizer is analytic:
    L-inf:  argmax <w,x0+d>, |d|<=eps  ->  d = eps*sign(w),  gain = eps*||w||_1
    L2:     argmax <w,x0+d>, ||d||<=eps ->  d = eps*w/||w||,  gain = eps*||w||_2
with x0 in the interior of [0,1] so the box constraint does not bind.
"""
import numpy as np
import torch

import adv_robustness_attack as A


def test_linf_linear():
    torch.manual_seed(0)
    D = 3 * 8 * 8
    w = torch.randn(D)
    x0 = torch.full((1, 3, 8, 8), 0.5)
    obj = lambda x: (x.reshape(x.size(0), -1) @ w)
    eps = 0.02  # small enough that [0,1] box never binds around 0.5
    clean = obj(x0).item()
    up = A.pgd(obj, x0, eps, "linf", +1.0, steps=100, alpha=2.5 * eps / 100, restarts=1).item()
    dn = A.pgd(obj, x0, eps, "linf", -1.0, steps=100, alpha=2.5 * eps / 100, restarts=1).item()
    gain = eps * w.abs().sum().item()
    assert abs((up - clean) - gain) / gain < 0.02, (up - clean, gain)
    assert abs((clean - dn) - gain) / gain < 0.02, (clean - dn, gain)
    assert up > clean > dn
    print(f"[linf-linear] up-clean={up-clean:.4f} clean-dn={clean-dn:.4f} closed-form={gain:.4f}  OK")


def test_l2_linear():
    torch.manual_seed(1)
    D = 3 * 8 * 8
    w = torch.randn(D)
    x0 = torch.full((1, 3, 8, 8), 0.5)
    obj = lambda x: (x.reshape(x.size(0), -1) @ w)
    eps = 0.3
    clean = obj(x0).item()
    up = A.pgd(obj, x0, eps, "l2", +1.0, steps=100, alpha=2.5 * eps / 100, restarts=1).item()
    gain = eps * w.norm(p=2).item()
    assert abs((up - clean) - gain) / gain < 0.02, (up - clean, gain)
    print(f"[l2-linear] up-clean={up-clean:.4f} closed-form={gain:.4f}  OK")


def test_projection_respects_ball_and_box():
    torch.manual_seed(2)
    x0 = torch.rand(4, 3, 8, 8)
    for norm, eps in [("linf", 0.05), ("l2", 1.0)]:
        x = torch.rand(4, 3, 8, 8) * 3 - 1  # wildly outside
        xp = A._project(x, x0, eps, norm)
        assert xp.min() >= -1e-6 and xp.max() <= 1 + 1e-6, "box violated"
        d = (xp - x0).reshape(4, -1)
        if norm == "linf":
            assert d.abs().max() <= eps + 1e-5, d.abs().max().item()
        else:
            assert d.norm(dim=1).max() <= eps + 1e-4, d.norm(dim=1).max().item()
    print("[projection] L-inf/L2 ball + [0,1] box respected  OK")


def test_batch_independence():
    """PGD on a batch must give the same per-image result as attacking each alone."""
    torch.manual_seed(3)
    D = 3 * 8 * 8
    W = torch.randn(2, D)
    x0 = torch.rand(2, 3, 8, 8)
    def obj(x):  # each image scored by its own weight vector
        return (x.reshape(x.size(0), -1) * W).sum(dim=1)
    eps = 0.03
    batch = A.pgd(obj, x0, eps, "linf", +1.0, 60, 2.5 * eps / 60, 1)
    singles = torch.stack([
        A.pgd(lambda x, i=i: (x.reshape(x.size(0), -1) @ W[i]),
              x0[i:i+1], eps, "linf", +1.0, 60, 2.5 * eps / 60, 1)[0]
        for i in range(2)])
    assert torch.allclose(batch, singles, atol=1e-4), (batch, singles)
    print(f"[batch] batched == per-image  OK  ({batch.tolist()})")


def test_fgsm_linear_matches_closed_form():
    """FGSM = one full-eps step (steps=1, restarts=0, alpha=eps). For a LINEAR objective the
    eps-ball optimum is reached in a single step, so FGSM must equal the closed form exactly
    (and therefore equal PGD)."""
    torch.manual_seed(4)
    D = 3 * 8 * 8
    w = torch.randn(D)
    x0 = torch.full((1, 3, 8, 8), 0.5)
    obj = lambda x: (x.reshape(x.size(0), -1) @ w)
    clean = obj(x0).item()
    for norm, eps, gain in [("linf", 0.02, 0.02 * w.abs().sum().item()),
                            ("l2", 0.30, 0.30 * w.norm(p=2).item())]:
        up = A.pgd(obj, x0, eps, norm, +1.0, steps=1, alpha=eps, restarts=0).item()
        dn = A.pgd(obj, x0, eps, norm, -1.0, steps=1, alpha=eps, restarts=0).item()
        assert abs((up - clean) - gain) / gain < 1e-4, (norm, up - clean, gain)
        assert abs((clean - dn) - gain) / gain < 1e-4, (norm, clean - dn, gain)
    print("[fgsm-linear] single-step FGSM == closed-form eps-ball optimum (L-inf & L2)  OK")


def test_fgsm_le_pgd_nonlinear():
    """On a NONLINEAR objective FGSM is a weaker attack than multi-step PGD: its achieved swing
    must be >= 0 (best-so-far vs clean) and <= PGD's swing.  This is the expected relationship
    that makes the FGSM-vs-PGD comparison a meaningful invariance check (weaker attack, same rank)."""
    torch.manual_seed(5)
    D = 3 * 8 * 8
    w1 = torch.randn(D); w2 = torch.randn(D)
    x0 = torch.rand(4, 3, 8, 8)
    obj = lambda x: (x.reshape(x.size(0), -1) @ w1) + 0.5 * (x.reshape(x.size(0), -1) @ w2) ** 2
    eps = 0.03
    clean = obj(x0)
    fg = A.pgd(obj, x0, eps, "linf", +1.0, steps=1, alpha=eps, restarts=0)
    pg = A.pgd(obj, x0, eps, "linf", +1.0, steps=50, alpha=2.5 * eps / 50, restarts=1)
    assert (fg >= clean - 1e-5).all(), "FGSM best-so-far fell below clean"
    assert (pg >= fg - 1e-4).all(), (pg.tolist(), fg.tolist())   # PGD at least as strong
    print(f"[fgsm<=pgd] nonlinear: 0 <= FGSM swing <= PGD swing  OK "
          f"(mean fgsm={float((fg-clean).mean()):.4f} pgd={float((pg-clean).mean()):.4f})")


def test_grad_sensitivity():
    D = 3 * 8 * 8
    w = torch.randn(D)
    x0 = torch.rand(1, 3, 8, 8)
    obj = lambda x: (x.reshape(x.size(0), -1) @ w)
    g2, g1 = A.grad_sensitivity(obj, x0)
    assert abs(g2.item() - w.norm(p=2).item()) < 1e-4
    assert abs(g1.item() - w.norm(p=1).item()) < 1e-4
    print(f"[grad-sens] L2={g2.item():.3f}(==||w||2 {w.norm(2):.3f}) L1={g1.item():.3f}  OK")


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_linf_linear()
    test_l2_linear()
    test_projection_respects_ball_and_box()
    test_batch_independence()
    test_fgsm_linear_matches_closed_form()
    test_fgsm_le_pgd_nonlinear()
    test_grad_sensitivity()
    print("\nALL PGD/METRIC TESTS PASSED")
