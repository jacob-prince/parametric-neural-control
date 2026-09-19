#!/usr/bin/env python3
"""Build the authoritative per-channel synthesis-hyperparameter table for take8.

Source of truth: the ORIGINAL production run logs. Each control-synthesis job logged
`Best hyperparameters: {'noise':..,'decay':..,'success_rate':..}` per (monkey, model, channel);
these survive in
  /n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/cluster_run_outputs/*.err
and were parsed (on the cluster) into `hp_from_logs.json` (250/250 canonical combos, 0 missing,
0 defaulted). Validated three ways: (a) exact match to the 10 surviving per-unit yamls,
(b) exact match to a from-scratch tuning re-run of leap resnet50_clip Ch306 (0.25/2.4), and
(c) re-generated stimuli match production in CLIP-embedding space (matched 0.92 vs mismatched
0.59; see qc/regen_fidelity.py). Tuning genuinely varied per channel (2-5 of 5 grid regimes used
per model), so a per-monkey-model value is NOT a valid substitute.

Writes intermediate/hp_tuning_selected.{csv,pkl}: monkey, model, channel, noise, decay,
success_rate, source.
"""
import os, json, pickle
import pandas as pd

from pnc import paths

INT = str(paths.preprocessed_data())
src = json.load(open(os.path.join(INT, "sup_hyperparam_hp_from_logs.json")))

df = pd.DataFrame(src)[["monkey", "model", "channel", "noise", "decay", "success_rate"]].copy()
df["source"] = "production_log"
df = df.sort_values(["monkey", "model", "channel"]).reset_index(drop=True)
df.to_csv(os.path.join(INT, "sup_hyperparam_hp_tuning_selected.csv"), index=False)
pickle.dump(df, open(os.path.join(INT, "sup_hyperparam_hp_tuning_selected.pkl"), "wb"))

print("wrote sup_hyperparam_hp_tuning_selected.{csv,pkl}:", len(df), "rows (per-channel, from production logs)")
print("coverage: %d monkeys x models x channels" % len(df),
      "| distinct (noise,decay) regimes:", df[["noise", "decay"]].drop_duplicates().shape[0])
print(df.groupby("model")[["noise", "decay"]].agg(["mean", "nunique"]).round(3).to_string())
