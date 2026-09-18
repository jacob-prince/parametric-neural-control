#!/usr/bin/env python3
"""
adv_visual_attack.py - qualitative "how the attack looks" per model, matched on effect.

For one encoding site (default red unit9, the Fig 7A example) and one seed image, find,
for each model, the MINIMAL L∞ pixel perturbation that raises that model's predicted
response by a common target Δ.  Because the effect (Δ) is held fixed across models, the
required perturbation size and structure reveal robustness directly: robust models need a
large, semantically-structured perturbation; fragile models need imperceptible noise.

Saves per model an .npz with the clean image, the min-ε adversarial image, the perturbation,
and metadata (eps_used, clean_pred, adv_pred, target).  A separate figure script tiles them.

Search modes:
  default          octave-ladder search (LADDER_255): first ε whose PGD max reaches the
                   target. PGD maximizes within the ball, so the achieved Δ can overshoot.
  --bisect         geometric bisection on ε between a failing lo and a reaching hi
                   (brackets seeded via --bracket-lo/--bracket-hi, verified, then bisected)
                   until the achieved Δ is within --tol of the target: the adversarial
                   image honestly achieves ≈ +Δ at the minimal ε. ~8 PGD evals per model.
  --eps-exact      attack at exactly this ε.

Usage (cluster, one model):
    python adv_visual_attack.py --model resnet50 --monkey red --unit 9 --seed-idx 1 \
        --target-delta 3.0 --out <dir>
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import adv_robustness_attack as A   # reuse model loading / readout / projection

LADDER_255 = [0.5, 1, 2, 4, 8, 16, 32, 64]   # L∞ budgets to search for the min that reaches Δ


def pgd_up_image(objective, x0, eps, steps, restarts):
    """PGD to MAXIMIZE objective within an L∞ eps-ball; returns (best_image, best_pred)."""
    best_pred = objective(x0).detach()
    best_x = x0.clone()
    for t in range(restarts + 1):
        x = x0.clone() if t == 0 else A._rand_start(x0, eps, "linf").detach()
        alpha = 2.5 * eps / steps
        for _s in range(steps):
            x.requires_grad_(True)
            pred = objective(x)
            grad, = torch.autograd.grad(pred.sum(), x)
            with torch.no_grad():
                x = A._project(x + alpha * grad.sign(), x0, eps, "linf")
                cur = objective(x)
                better = cur > best_pred
                if better.any():
                    best_pred = torch.where(better, cur, best_pred)
                    bi = better.view(-1, 1, 1, 1)
                    best_x = torch.where(bi, x, best_x)
            x = x.detach()
    return best_x.detach(), best_pred.detach()


def bisect_eps(obj, x0, target, lo255, hi255, steps, restarts, tol, max_iters):
    """Minimal-ε search by geometric bisection: verify the bracket (lo fails, hi reaches),
    then bisect until the smallest reaching ε's achieved pred is within tol of the target.
    Returns (eps255, adv_img, adv_pred, n_evals)."""
    evals = [0]

    def attack(e255):
        evals[0] += 1
        ax, ap = pgd_up_image(obj, x0, e255 / 255.0, steps, restarts)
        ap = float(ap.item())
        print(f"  [bisect] eps={e255:.4g}/255 -> pred {ap:+.3f} "
              f"({'reach' if ap >= target else 'fail'})", flush=True)
        return ax, ap

    # verify hi reaches (walk up if not) and lo fails (walk down if not)
    x_hi, p_hi = attack(hi255)
    while p_hi < target and hi255 < 512:
        lo255 = hi255; hi255 *= 2
        x_hi, p_hi = attack(hi255)
    x_lo, p_lo = attack(lo255)
    while p_lo >= target and lo255 > 1e-3:
        hi255, x_hi, p_hi = lo255, x_lo, p_lo
        lo255 /= 2
        x_lo, p_lo = attack(lo255)

    best = (hi255, x_hi, p_hi)                       # smallest ε known to reach the target
    for _ in range(max_iters):
        if best[2] - target <= tol:
            break
        mid = float(np.sqrt(lo255 * hi255))
        x_m, p_m = attack(mid)
        if p_m >= target:
            hi255 = mid; best = (mid, x_m, p_m)
        else:
            lo255 = mid
    return best[0], best[1], best[2], evals[0]


def run(model_name, monkey, unit, seed_idx, target_delta, steps, restarts, out_dir, device,
        eps_override=None, bisect=False, bracket_lo=None, bracket_hi=None, tol=0.05,
        max_iters=12):
    load_model, featureFetcher = A._import_backbone_deps()
    model, preprocess = load_model(model_name, device=device)
    model = model.eval().to(device); model.requires_grad_(False)
    normalizer = A.make_normalizer(preprocess, device)

    monkey_dir = A.MONKEY_DIRS[monkey]
    r = A.load_readout(monkey_dir, unit, model_name, device)
    if r is None:
        sys.exit(f"no readout for {model_name} {monkey} Ch{unit}")
    fetcher = featureFetcher(model, input_size=(3, A.IMG_PX, A.IMG_PX),
                             print_module=False, store_device=device)
    fetcher.record(r["layer"], ingraph=True, store_device=device)

    seed_paths = A._config_seeds(monkey_dir, unit, model_name)
    x0, names = A.load_canvas([seed_paths[seed_idx]], device)   # single image, shape (1,3,224,224)

    rv, rb = r["readout_vec"], r["readout_bias"]
    def obj(x):
        model(normalizer(x))
        return r["xtransform"](fetcher[r["layer"]]) @ rv + rb

    clean = float(obj(x0).item())
    target = clean + target_delta

    if bisect:
        torch.manual_seed(0)                             # deterministic restarts
        lo = bracket_lo if bracket_lo is not None else 0.5
        hi = bracket_hi if bracket_hi is not None else 64.0
        used_eps, adv_img, adv_pred, n_evals = bisect_eps(
            obj, x0, target, float(lo), float(hi), steps, restarts, tol, max_iters)
        print(f"[bisect] done in {n_evals} PGD evals; eps*={used_eps:.4g}/255 "
              f"achieved {adv_pred - clean:+.3f} (target +{target_delta:g})", flush=True)
    else:
        # ladder search, or a single exact ε (e.g. the fine-sweep ε*)
        ladder = [float(eps_override)] if eps_override is not None else LADDER_255
        used_eps, adv_img, adv_pred = None, None, clean
        for e255 in ladder:
            eps = e255 / 255.0
            ax, ap = pgd_up_image(obj, x0, eps, steps, restarts)
            ap = float(ap.item())
            if ap >= target:
                used_eps, adv_img, adv_pred = e255, ax, ap
                break
            used_eps, adv_img, adv_pred = e255, ax, ap   # keep the strongest so far

    reached = adv_pred >= target
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    clean_np = x0[0].cpu().numpy()
    adv_np = adv_img[0].cpu().numpy()
    out = out_dir / f"advvis_{monkey}_Ch{unit}_seed{seed_idx}_{model_name}.npz"
    np.savez_compressed(out, clean=clean_np.astype(np.float32), adv=adv_np.astype(np.float32),
                        eps_used=np.float32(used_eps), reached=np.array(reached),
                        clean_pred=np.float32(clean), adv_pred=np.float32(adv_pred),
                        target=np.float32(target), target_delta=np.float32(target_delta),
                        model=str(model_name), monkey=str(monkey), unit=np.int64(unit),
                        seed_name=str(names[0]), layer=str(r["layer"]),
                        range_q99q01=np.float32(r["range_q99q01"]))
    print(f"WROTE {out.name}  eps={used_eps}/255  clean={clean:.2f} adv={adv_pred:.2f} "
          f"target={target:.2f} reached={reached}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=A.MODELS)
    ap.add_argument("--monkey", default="red")
    ap.add_argument("--unit", type=int, default=9)
    ap.add_argument("--seed-idx", type=int, default=1)
    ap.add_argument("--target-delta", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--out", default=str(SCRIPT_DIR.parent / "results" / "advvis"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps-exact", type=float, default=None, help="attack at exactly this ε (/255)")
    ap.add_argument("--from-sweep", action="store_true",
                    help="use this model's fine-sweep ε* (results/sweep/) as the exact ε")
    ap.add_argument("--bisect", action="store_true",
                    help="geometric bisection on ε until the achieved Δ is within --tol of the target")
    ap.add_argument("--bracket-lo", type=float, default=None, help="bisect: ε (/255) expected to fail")
    ap.add_argument("--bracket-hi", type=float, default=None, help="bisect: ε (/255) expected to reach")
    ap.add_argument("--tol", type=float, default=0.05, help="bisect: allowed overshoot of the achieved Δ")
    ap.add_argument("--max-iters", type=int, default=12, help="bisect: max bisection steps after bracketing")
    a = ap.parse_args()
    eps_override = a.eps_exact
    if a.from_sweep and eps_override is None:
        sp = (SCRIPT_DIR.parent / "results" / "sweep" /
              f"sweep_{a.monkey}_Ch{a.unit}_seed{a.seed_idx}_{a.model}.npz")
        d = np.load(sp, allow_pickle=True)
        eps_override = float(d["eps_star"])
        print(f"[from-sweep] eps*={eps_override:.3f}/255", flush=True)
    run(a.model, a.monkey, a.unit, a.seed_idx, a.target_delta, a.steps, a.restarts, a.out, a.device,
        eps_override=eps_override, bisect=a.bisect, bracket_lo=a.bracket_lo,
        bracket_hi=a.bracket_hi, tol=a.tol, max_iters=a.max_iters)


if __name__ == "__main__":
    main()
