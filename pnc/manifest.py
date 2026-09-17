"""Single source of truth for supplementary-figure NUMBERING and order.

Figures are numbered SEQUENTIALLY from the ORDER of ENTRIES below -- there are NO
hard-coded S-numbers. Each entry consumes `count` numbers (default 1; the per-macaque
galleries use 5). To renumber, just reorder / insert / delete an entry and the numbers
re-flow automatically. Figure scripts derive their S-number, title, and output prefix
from their STABLE `slug`:

    from pnc.manifest import fig
    M = fig('differentiation')
    NUM, TITLE, PREFIX = M['num'], M['title'], M['prefix']

For a span entry (count>1), M['span'] = (first, last) and per-item numbers are
M['num'] + item_index (e.g. sweep_gallery red = num, paul = num+1, ...).

Fields: slug (stable id) . count (# S-numbers consumed) . title . section.
"""

ENTRIES = [
    dict(slug='calibration_mosaic', count=1,
         title='Calibration image set',
         section='Framework'),
    dict(slug='reliability', count=1,
         title='Response reliability and NSD noise ceiling',
         section='Framework'),
    dict(slug='anatomy', count=1,
         title='Array and Neuropixels placement',
         section='Framework'),
    dict(slug='floc_selectivity', count=1,
         title='Site category-selectivity',
         section='Framework'),
    dict(slug='layer_selection', count=1,
         title='Per-channel encoding-layer selection',
         section='Framework'),
    dict(slug='hyperparam_selection', count=1,
         title='Automated synthesis-regularization selection',
         section='FeatAccent'),
    dict(slug='success_rate', count=1,
         title='Feature-accentuation success rate',
         section='FeatAccent'),
    dict(slug='fair_filtering', count=1,
         title='Structure of synthesis failures',
         section='FeatAccent'),
    dict(slug='axis_alignment_stats', count=1,
         title='Encoding-axis alignment of accentuation sweeps',
         section='FeatAccent'),
    dict(slug='axis_alignment_gallery', count=1,
         title='Accentuation sweeps in the read-out-aligned embedding',
         section='FeatAccent'),
    dict(slug='sweep_extremes', count=1,
         title='Suppress and drive extremes across sites',
         section='FeatAccent'),
    dict(slug='sweep_gallery', count=5,
         title='Accentuation sweep galleries',
         section='FeatAccent'),
    dict(slug='control_slope_anova', count=1,
         title='Model differences under the control-slope measure',
         section='StrongEnc'),
    dict(slug='robust_metrics', count=1,
         title='Control outcomes under alternative scoring metrics and per-seed aggregation',
         section='RobustAdv'),
    dict(slug='controllability_variation', count=1,
         title='Controllability variation across sites',
         section='StrongEnc'),
    dict(slug='robust_by_site', count=1,
         title='Per-site control score (all 10 models)',
         section='RobustAdv'),
    dict(slug='cross_area_consistency', count=1,
         title='Cross-area consistency of model rankings',
         section='StrongEnc'),
    dict(slug='high_control_examples', count=5,
         title='High-control example sweeps',
         section='StrongEnc'),
    dict(slug='superstimuli', count=1,
         title='Super-stimulus proportion',
         section='RobustAdv'),
    dict(slug='superstim_scatter_venus', count=1,
         title='Super-stimulus response scatter, Monkey V',
         section='RobustAdv'),
    dict(slug='superstim_scatter_all', count=1,
         title='Super-stimulus response scatter, all sites',
         section='RobustAdv'),
    dict(slug='model_dichotomies', count=1,
         title='Neural control across model inductive bias groups',
         section='Gradient'),
    dict(slug='control_reliability', count=1,
         title='Reliability of control-phase responses',
         section='StrongEnc'),
    dict(slug='robadvctl_regaxis', count=1,
         title='Synthesis regularization and parametric control',
         section='RobustAdv'),
    dict(slug='robadvctl_controls', count=1,
         title='Control outcomes after regressing out the effect of synthesis hyperparameters',
         section='RobustAdv'),
    dict(slug='robadvctl_alignment', count=1,
         title='Relationship between axis alignment and neural control outcomes',
         section='RobustAdv'),
    dict(slug='exclusion_regimes', count=1,
         title='Control outcomes across accentuated-image exclusion regimes',
         section='FeatAccent'),
    dict(slug='controversial', count=1,
         title='Controversial-stimulus experiment',
         section='RobustAdv'),
    dict(slug='controversial_stimuli', count=1,
         title='Controversial-accentuation stimuli',
         section='RobustAdv'),
    dict(slug='controversial_results', count=1,
         title='Controversial-accentuation results',
         section='RobustAdv'),
    dict(slug='adversarial_sensitivity', count=1,
         title='Encoding-axis adversarial sensitivity and neural control',
         section='Gradient'),
    dict(slug='adversarial_robustness', count=1,
         title='Adversarial attack-method comparisons',
         section='Gradient'),
    dict(slug='adv_sensitivity_formulations', count=1,
         title='Alternative metrics for estimating adversarial sensitivity and gradient spectral concentration',
         section='RobustAdv'),
    dict(slug='flatness_pipeline', count=1,
         title='Computing gradient spectral participation ratio (PR)',
         section='Gradient'),
    dict(slug='gradient_gallery', count=1,
         title='Input-gradient map galleries by seed image',
         section='Gradient'),
    dict(slug='gradient_geometry', count=1,
         title='Encoding-gradient geometry and neural control',
         section='Gradient'),
    dict(slug='concentration_image_robustness', count=1,
         title='Robustness of gradient spectral participation ratio to probe images and analysis choices',
         section='Gradient'),
    dict(slug='refit_diet_ladder', count=1,
         title='Gradient spectra and their prediction of control across encoding training diets',
         section='Gradient'),
    dict(slug='flatness_regime_control', count=1,
         title='Gradient spectral participation ratio and control within matched regularization regimes',
         section='Gradient'),
    dict(slug='predicting_cv_trained', count=1,
         title='Cross-validated prediction of parametric control among the trained models',
         section='Models'),
    dict(slug='predicting_summary', count=1,
         title='Predicting neural control: single-predictor leaderboard and cross-validated best model',
         section='Gradient'),
]



def _assign():
    out, n = [], 1
    for e in ENTRIES:
        c = e.get('count', 1)
        e = dict(e)
        e['num'] = n
        e['span'] = (n, n + c - 1) if c > 1 else None
        n += c
        out.append(e)
    return out


_NUMBERED = _assign()
_BY_SLUG = {e['slug']: e for e in _NUMBERED}
TOTAL = _NUMBERED[-1]['num'] + (_NUMBERED[-1].get('count', 1) - 1)


def fig(slug):
    """Manifest entry for a slug, augmented with `num`, `span`, `prefix`, `label`."""
    e = dict(_BY_SLUG[slug])
    n = e['num']
    e['prefix'] = f"figS{n:02d}_{slug}"
    e['label'] = f"Supplementary Figure {n}"
    return e


def all_entries():
    return list(_NUMBERED)


# Per-macaque order of the two span entries (sweep_gallery, high_control_examples).
SPAN_MONKEYS = ['red', 'paul', 'venus', 'leap', 'three0']

# Main-figure names as they appear in the manuscript (figures/<name>.png).
MAIN_FIGURES = {1: 'framework', 2: 'accentuation', 3: 'single_site',
                4: 'divergence', 5: 'predictors', 6: 'benchmarking'}


def output_name(slug, monkey=None):
    """Manuscript file stem for a supplementary figure, e.g. 's17_control_slope_anova' or
    's12_sweep_gallery_red'. This is the name the rendered PNG is saved under."""
    e = fig(slug)
    if e['span']:
        if monkey is None:
            raise ValueError(f"{slug} spans {e['span']}; pass monkey=")
        return f"s{e['num'] + SPAN_MONKEYS.index(monkey):02d}_{slug}_{monkey}"
    return f"s{e['num']:02d}_{slug}"


def all_output_names():
    """Every manuscript stem in numeric order: framework .. benchmarking, s01 .. s49."""
    out = list(MAIN_FIGURES.values())
    for e in _NUMBERED:
        out += [output_name(e['slug'], mk) for mk in SPAN_MONKEYS] if e['span'] else [output_name(e['slug'])]
    return out


def index_markdown():
    lines = ["| S# | slug | title | section |", "|---|---|---|---|"]
    for e in _NUMBERED:
        num = f"S{e['num']}" + (f"-{e['span'][1]}" if e['span'] else "")
        lines.append(f"| {num} | `{e['slug']}` | {e['title']} | {e['section']} |")
    return "\n".join(lines)


if __name__ == '__main__':
    print(index_markdown())
    print(f"\nTOTAL supplementary figures: S1-S{TOTAL}")
