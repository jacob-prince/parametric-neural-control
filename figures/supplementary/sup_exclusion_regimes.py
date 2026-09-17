#!/usr/bin/env python3
"""Supp fig (exclusion_regimes) — control outcomes across accentuated-image exclusion
regimes.

A synthesized stimulus "fails" when the generating model's own post-hoc prediction for
the saved image misses the intended target by more than tau (as a fraction of the
channel's target range). Dropping each model's own failures ("per-model") would compare
models on unequal footing; this figure lays out the candidate exclusion regimes and
shows that the model ranking of control outcomes is insensitive to the choice.

  a  per-model retention under each regime at both tau (heatmaps), overall retention
     (bars), and presentation completion per monkey (the STS monkeys did not complete
     their stimulus sets; unpresented stimuli are excluded under every regime).
  b  summary table: what each regime keeps, and how much survives.
  c  per-model control r recomputed from the kept stimuli under each regime (tau = 5%),
     one subplot per regime; fixed model order and shared y-axis so shifts are visible.

Masks come precomputed from loader.load_exclusions (exclusions.py); control clouds are
rebuilt from loader.load_predictions + the control-session responses with the firing
floor applied, exactly as loader.control_cloud.
"""
import os
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch

from pnc import paths
from pnc.utils import (apply_figure_style, FONT_FAMILY, FIG_FACECOLOR, save_fig,
                       MM, WIDTH_2COL_MM, MONKEY_COLORS,
                       MODEL_ORDER, MODEL_SHORT_NAMES, get_model_color, ROBUST_MODELS)
from pnc.manifest import fig as _fig, output_name
from pnc.preproc import loader as L

apply_figure_style()
STEM = output_name('exclusion_regimes')

MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']
THRESHOLDS = [0.02, 0.05]           # display order: strict, then lenient
TAU_PRIMARY = 0.05                   # canonical success criterion
MIN_LEVELS = 5                       # a sweep needs >=5 kept levels for a per-sweep metric
MIN_STIM = 8                         # a site-model needs >=8 kept stimuli for a control r
FS_TITLE, FS_AX, FS_TICK, FS_ANNOT = 7, 6.5, 5.5, 5.5
FS_LETTER = 10

# criterion key, display label, definition (table), bar color
CRITERIA = [
    ('none',      'No exclusion',
     'all presented stimuli kept', '#B0B0B0'),
    ('permodel',  'Per-model',
     'each model drops its own failed syntheses\n(stimulus sets differ across models)', '#D1495B'),
    ('drop2',     'Drop 2 levels',
     'the two strongest drive levels dropped\nfrom every sweep (same levels for all models)', '#8E44AD'),
    ('intersect', 'Intersect',
     'a (site, seed, level) kept only if every\nmodel succeeded there (identical sets)', '#E1812C'),
]
CRIT_KEYS = [c[0] for c in CRITERIA]
CRIT_LABEL = {c[0]: c[1] for c in CRITERIA}
CRIT_DEF = {c[0]: c[2] for c in CRITERIA}
CRIT_COLOR = {c[0]: c[3] for c in CRITERIA}
SHORT = dict(MODEL_SHORT_NAMES); SHORT['AlexNet_training_seed_01'] = 'Untrained'


# ── data: precomputed exclusion masks over all 5 monkeys ─────────────────────
def load_all():
    model, unit, seed, level, present = [], [], [], [], []
    masks = {t: {c: [] for c in CRIT_KEYS} for t in THRESHOLDS}
    monkey = []
    for m in MONKEYS:
        ex = L.load_exclusions(m)
        n = len(ex['gen_model'])
        monkey.append(np.full(n, m, object))
        model.append(np.asarray(ex['gen_model']))
        unit.append(np.asarray(ex['gen_unit']).astype(int))
        seed.append(np.asarray(ex['gen_seed']).astype(int))
        level.append(np.asarray(ex['level_ord']).astype(int))
        present.append(np.asarray(ex['present']).astype(bool))
        for t in THRESHOLDS:
            for c in CRIT_KEYS:
                masks[t][c].append(np.asarray(ex['masks'][t][c]).astype(bool))
    return dict(monkey=np.concatenate(monkey), model=np.concatenate(model),
                unit=np.concatenate(unit), seed=np.concatenate(seed),
                level=np.concatenate(level), present=np.concatenate(present),
                masks={t: {c: np.concatenate(v) for c, v in cm.items()} for t, cm in masks.items()})


# Module-level tables used by the panel functions below. take9 filled them at import time;
# here `_setup()` populates them from main() so importing the module does no work.
D = MK = MO = UN = SD = LO = N = SWEEP = SEEDCOND = SEEDCONDS = None
STATS = N_SWEEPS = N_SEEDCONDS = None


def _setup():
    global D, MK, MO, UN, SD, LO, N, SWEEP, SEEDCOND, SEEDCONDS, STATS, N_SWEEPS, N_SEEDCONDS
    D = load_all()
    MK, MO, UN, SD, LO = D['monkey'], D['model'], D['unit'], D['seed'], D['level']
    N = len(MO)
    SWEEP = list(zip(MK, UN, MO, SD))          # (monkey, unit, model, seed) — a sweep
    SEEDCOND = list(zip(MK, UN, SD))           # (monkey, unit, seed) — a seed-condition
    SEEDCONDS = sorted(set(SEEDCOND))
    STATS = {t: compute(t) for t in THRESHOLDS}
    N_SWEEPS = len(set(SWEEP))
    N_SEEDCONDS = len(SEEDCONDS)


def _sweeps_ge(mask, K):
    kept = defaultdict(set)
    for i in np.nonzero(mask)[0]:
        kept[SWEEP[i]].add(LO[i])
    return sum(1 for v in kept.values() if len(v) >= K)


def _fair_seedconds(mask, K=MIN_LEVELS):
    """# seed-conditions where EVERY model retains >= K levels."""
    kept = defaultdict(lambda: defaultdict(set))
    for i in np.nonzero(mask)[0]:
        kept[SEEDCOND[i]][MO[i]].add(LO[i])
    n = 0
    for sc in SEEDCONDS:
        km = kept.get(sc, {})
        if all(len(km.get(m, set())) >= K for m in MODEL_ORDER):
            n += 1
    return n


def compute(t):
    crit = {}
    for k in CRIT_KEYS:
        mask = D['masks'][t][k]
        perpct = {m: 100 * mask[MO == m].sum() / (MO == m).sum() for m in MODEL_ORDER}
        crit[k] = dict(mask=mask, n_stim=int(mask.sum()), pct=100 * mask.sum() / N,
                       sw5=_sweeps_ge(mask, MIN_LEVELS),
                       fairsc=_fair_seedconds(mask, MIN_LEVELS), perpct=perpct)
    return dict(crit=crit)


# ── per-regime control outcomes (tau = TAU_PRIMARY) ──────────────────────────
def control_by_regime():
    """{regime: {(monkey, unit, model): control r over the kept stimuli}} — the control
    cloud rebuilt per regime (generating model's own prediction vs measured response,
    firing floor applied), exactly as loader.control_cloud but with the regime mask."""
    out = {k: {} for k in CRIT_KEYS}
    off = 0
    for mk in MONKEYS:
        P = L.load_predictions(mk); ex = L.load_exclusions(mk)
        n = len(ex['gen_model'])
        ui = {int(u): j for j, u in enumerate(P['target_units'])}
        mi = {m: j for j, m in enumerate(P['pred_models'])}
        gm = np.asarray(ex['gen_model']); gu = np.asarray(ex['gen_unit']).astype(int)
        stim = np.asarray(ex['stim'])
        for u in sorted(set(gu)):
            resp = L._acc_response(mk, int(u)); fl = L.firing_floor(mk, int(u))
            for m in MODEL_ORDER:
                sel = (gm == m) & (gu == u)
                for k in CRIT_KEYS:
                    keep = sel & D['masks'][TAU_PRIMARY][k][off:off + n]
                    names = stim[keep]
                    ok = np.array([nm in resp for nm in names])
                    if ok.sum() < MIN_STIM:
                        continue
                    x = np.maximum(P['pred'][keep][ok, mi[m], ui[int(u)]], fl)
                    y = np.array([resp[nm] for nm in names[ok]])
                    if x.std() > 0 and y.std() > 0:
                        out[k][(mk, int(u), m)] = float(stats.pearsonr(x, y)[0])
        off += n
    return out


# ── panel a: per-model retention heatmaps + retention bars ───────────────────
def draw_footing(ax, t, show_ylabels, show_cbar):
    S = STATS[t]
    M = np.array([[S['crit'][k]['perpct'][m] for m in MODEL_ORDER] for k in CRIT_KEYS])
    im = ax.imshow(M, aspect='auto', cmap='Blues', vmin=0, vmax=100, interpolation='nearest')
    for r in range(len(CRIT_KEYS)):
        for c in range(len(MODEL_ORDER)):
            v = M[r, c]
            ax.text(c, r, f'{v:.0f}', ha='center', va='center', fontsize=FS_ANNOT,
                    color='white' if v > 60 else '#333', fontfamily=FONT_FAMILY)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=45, ha='right',
                       rotation_mode='anchor', fontsize=FS_TICK - 0.5)
    for tl, m in zip(ax.get_xticklabels(), MODEL_ORDER):
        tl.set_color(get_model_color(m))
        if m in ROBUST_MODELS:
            tl.set_fontweight('bold')
    ax.set_yticks(range(len(CRIT_KEYS)))
    if show_ylabels:
        ax.set_yticklabels([CRIT_LABEL[k] for k in CRIT_KEYS], fontsize=FS_TICK)
        for tl, k in zip(ax.get_yticklabels(), CRIT_KEYS):
            tl.set_color(CRIT_COLOR[k])
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_title(f'Per-model retention  ({t:.0%} of range)', fontsize=FS_TITLE,
                 fontfamily=FONT_FAMILY, pad=5)
    if show_cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
        cb.ax.tick_params(labelsize=FS_TICK)


def draw_retention(ax):
    xs = np.arange(len(CRIT_KEYS)); w = 0.38
    tp, tr = THRESHOLDS
    for offx, t, hatch, alpha in [(-w / 2, tp, None, 0.95), (w / 2, tr, '///', 0.62)]:
        vals = [STATS[t]['crit'][k]['pct'] for k in CRIT_KEYS]
        ax.bar(xs + offx, vals, w, color=[CRIT_COLOR[k] for k in CRIT_KEYS],
               alpha=alpha, edgecolor='white', linewidth=0.5, hatch=hatch, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([CRIT_LABEL[k] for k in CRIT_KEYS], rotation=45, ha='right',
                       rotation_mode='anchor', fontsize=FS_TICK)
    for tl, k in zip(ax.get_xticklabels(), CRIT_KEYS):
        tl.set_color(CRIT_COLOR[k])
    ax.set_ylabel('Stimuli kept (%)', fontsize=FS_AX, fontfamily=FONT_FAMILY, labelpad=3)
    ax.set_title(f'Retention (of {N:,} stimuli)', fontsize=FS_TITLE, fontfamily=FONT_FAMILY, pad=6)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(0, 108)
    ax.legend(handles=[Patch(fc='#888', label=f'{THRESHOLDS[0]:.0%} of range'),
                       Patch(fc='#888', hatch='///', alpha=0.62, label=f'{THRESHOLDS[1]:.0%} of range')],
              loc='upper right', fontsize=FS_ANNOT, frameon=False)


def draw_completion(ax):
    disp = {'red': 'R', 'paul': 'P', 'venus': 'V', 'leap': 'L', 'three0': 'T'}
    vals = [100 * D['present'][MK == m].mean() for m in MONKEYS]
    ax.bar(range(len(MONKEYS)), vals, 0.7, color=[MONKEY_COLORS[m] for m in MONKEYS],
           edgecolor='white', linewidth=0.5, zorder=3)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 2.5, f'{v:.0f}', ha='center', va='bottom', fontsize=FS_ANNOT,
                fontfamily=FONT_FAMILY, color='#333')
    ax.set_xticks(range(len(MONKEYS)))
    ax.set_xticklabels([disp[m] for m in MONKEYS], rotation=45, ha='right',
                       rotation_mode='anchor', fontsize=FS_TICK)
    for tl, m in zip(ax.get_xticklabels(), MONKEYS):
        tl.set_color(MONKEY_COLORS[m]); tl.set_fontweight('bold')
    ax.set_ylabel('Stimuli presented (%)', fontsize=FS_AX, fontfamily=FONT_FAMILY, labelpad=3)
    ax.set_xlabel('monkey', fontsize=FS_AX, fontfamily=FONT_FAMILY)
    ax.set_ylim(0, 112)
    ax.set_title('Presentation\ncompletion', fontsize=FS_TITLE, fontfamily=FONT_FAMILY, pad=6)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# ── panel b: summary table ────────────────────────────────────────────────────
def draw_table(ax):
    ax.axis('off')
    tp, tr = THRESHOLDS
    hdr = f'{tp:.0%} / {tr:.0%} of range'
    cols = ['Criterion', 'Definition', f'Stimuli kept\n{hdr}',
            f'Usable sweeps $\\geq$5\n{hdr}', f'Seed-conds all $\\geq$5\n{hdr}']
    rows = []
    for k in CRIT_KEYS:
        a, b = STATS[tp]['crit'][k], STATS[tr]['crit'][k]
        rows.append([
            CRIT_LABEL[k], CRIT_DEF[k],
            f"{a['n_stim']:,} ({a['pct']:.0f}%) / {b['n_stim']:,} ({b['pct']:.0f}%)",
            f"{a['sw5']} / {b['sw5']}  (of {N_SWEEPS:,})",
            f"{a['fairsc']} / {b['fairsc']}  (of {N_SEEDCONDS})",
        ])
    # explicit bbox = exactly fill this axes' cell (figure-fraction [0,1]x[0,1] in axes
    # coords): the table's rendered height is then IDENTICAL to the GridSpec row height,
    # so the outer grid's hspace produces the actual equal gaps above and below it.
    # (tbl.scale() was the earlier bug: it grows the table past its axes cell with no
    # relation to the assigned row height, silently eating the row-b/row-c gap.)
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='upper center',
                   colWidths=[0.11, 0.36, 0.22, 0.17, 0.14], bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(FS_ANNOT)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#DDD')
        if r == 0:
            cell.set_facecolor('#F0F0F0')
            cell.set_text_props(fontweight='bold', fontfamily=FONT_FAMILY)
        else:
            k = CRIT_KEYS[r - 1]
            if c == 0:
                cell.set_text_props(color=CRIT_COLOR[k], fontweight='bold', fontfamily=FONT_FAMILY)
            if k == 'permodel':
                cell.set_facecolor('#FCEBEC')
            elif k == 'none':
                cell.set_facecolor('#F5F5F5')
            elif k == 'intersect':
                cell.set_facecolor('#FDF1E4')
    return tbl


# ── panel c: per-model control r under each regime ───────────────────────────
def draw_regime_outcomes(fig, grid, ctrl):
    """`grid` is anything indexable as grid[0, ci] (a GridSpec or a
    GridSpecFromSubplotSpec) already laid out as 1x4."""
    rng = np.random.default_rng(0)
    allvals = [v for k in CRIT_KEYS for v in ctrl[k].values()]
    ylim = (min(allvals) - 0.05, max(allvals) + 0.05)
    axes = []
    for ci, k in enumerate(CRIT_KEYS):
        ax = fig.add_subplot(grid[0, ci]); axes.append(ax)
        for xi, mdl in enumerate(MODEL_ORDER):
            vals = np.array([v for (mk, u, m), v in ctrl[k].items() if m == mdl])
            ax.scatter(xi + rng.uniform(-0.28, 0.28, len(vals)), vals, s=4,
                       color=get_model_color(mdl), alpha=0.75, edgecolors='white',
                       linewidths=0.15, zorder=4)
            ax.plot([xi - 0.36, xi + 0.36], [vals.mean()] * 2, color='black', lw=1.1, zorder=6)
        ax.axhline(0, color='#cccccc', lw=0.6, zorder=1)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER], rotation=45, ha='right',
                           rotation_mode='anchor', fontsize=FS_TICK - 0.8)
        for tl, m in zip(ax.get_xticklabels(), MODEL_ORDER):
            tl.set_color(get_model_color(m))
            if m in ROBUST_MODELS:
                tl.set_fontweight('bold')
        ax.set_ylim(*ylim); ax.set_xlim(-0.8, len(MODEL_ORDER) - 0.2)
        ax.set_title(CRIT_LABEL[k], fontsize=FS_TITLE, color=CRIT_COLOR[k],
                     fontweight='bold', fontfamily=FONT_FAMILY, pad=4)
        if ci == 0:
            ax.set_ylabel('Control $r$ (kept stimuli)', fontsize=FS_AX, fontfamily=FONT_FAMILY)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.tick_params(axis='x', length=0, pad=2)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    return axes


def main(out_dir):
    _setup()
    ctrl = control_by_regime()

    fig = plt.figure(figsize=(WIDTH_2COL_MM * MM, 148 * MM), facecolor=FIG_FACECOLOR)
    LEFT, RIGHT = 0.09, 0.95

    # Explicit row boundaries (figure-fraction y), NOT a shared-hspace GridSpec: a plain
    # GridSpec's single hspace value reserves equal NOMINAL gaps, but row a's rotated
    # tick labels and row c's panel titles overhang their axes cells by very different
    # amounts (measured at 45 deg ticks: ~0.077 below row a, ~0.023 above row c). Equal
    # nominal gaps therefore produced very unequal RENDERED gaps. These boundaries are
    # back-solved from that measurement so the actual whitespace above and below row b is
    # equal (each nominal gap = target 0.030 + that row's own overhang). Rows keep their
    # heights and slide up; savefig's tight bbox trims the slack left under row c.
    A_TOP, A_BOT = 0.920, 0.6767
    B_TOP, B_BOT = 0.5702, 0.4062
    C_TOP, C_BOT = 0.3532, 0.1099

    # Row a — footing check (both thresholds) + retention bars + completion per monkey
    gsA = GridSpec(1, 4, figure=fig, left=LEFT, right=RIGHT, top=A_TOP, bottom=A_BOT,
                   wspace=0.62, width_ratios=[1.20, 1.20, 0.68, 0.40])
    axA0 = fig.add_subplot(gsA[0])
    draw_footing(axA0, THRESHOLDS[0], show_ylabels=True, show_cbar=False)
    draw_footing(fig.add_subplot(gsA[1]), THRESHOLDS[1], show_ylabels=False, show_cbar=True)
    draw_retention(fig.add_subplot(gsA[2]))
    draw_completion(fig.add_subplot(gsA[3]))

    # Row b — table
    axB = fig.add_axes([LEFT, B_BOT, RIGHT - LEFT, B_TOP - B_BOT])
    draw_table(axB)

    # Row c — per-model control outcomes under each regime
    gsC = GridSpec(1, 4, figure=fig, left=LEFT, right=RIGHT, top=C_TOP, bottom=C_BOT, wspace=0.14)
    axesC = draw_regime_outcomes(fig, gsC, ctrl)

    fig.text(0.09, 0.968, 'Accentuated image exclusion regimes and their effect on control outcomes',
             fontsize=FS_TITLE, fontweight='bold', ha='left', va='center', fontfamily=FONT_FAMILY)

    fig.canvas.draw()
    for ax_ref, lab, dy in [(axA0, 'a', 0.032), (axB, 'b', 0.028), (axesC[0], 'c', 0.034)]:
        fig.text(0.04, ax_ref.get_position().y1 + dy, lab, fontsize=FS_LETTER,
                 fontweight='bold', fontfamily=FONT_FAMILY, va='top', ha='left')

    # console: per-regime family means + rank stability vs no-exclusion
    base = {m: np.mean([v for (mk, u, mm), v in ctrl['none'].items() if mm == m]) for m in MODEL_ORDER}
    for k in CRIT_KEYS:
        pm = {m: np.mean([v for (mk, u, mm), v in ctrl[k].items() if mm == m]) for m in MODEL_ORDER}
        rho = stats.spearmanr([base[m] for m in MODEL_ORDER], [pm[m] for m in MODEL_ORDER])[0]
        nsm = min(sum(1 for kk in ctrl[k] if kk[2] == m) for m in MODEL_ORDER)
        print(f'{CRIT_LABEL[k]:14s} per-model mean r range '
              f'{min(pm.values()):.2f}-{max(pm.values()):.2f} | rank rho vs none = {rho:.2f} '
              f'| min site-models/model = {nsm}')

    out = save_fig(fig, os.path.join(out_dir, STEM))
    plt.close(fig)
    print('Saved ->', out)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', dest='out_dir', default=str(paths.output_dir() / 'figures'), help='output directory')
    args = ap.parse_args()
    main(**vars(args))
