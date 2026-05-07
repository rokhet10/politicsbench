#!/usr/bin/env python3
"""
Statistical test for trait activation (paper-style).

**Primary test (default)** — one hypothesis, no Bonferroni carnival:

  H₀: Mean activation across **scenario stages S₁–S₄** (mean of the four in-scenario
  stage means per model) equals **baseline** activation.

  For each model *m*: compute ``scenario_agg_m = mean(S₁,S₂,S₃,S₄)_m``, then
  ``Δ_m = scenario_agg_m − baseline_m``. Paired across models:
  one-sample *t*-test and Wilcoxon signed-rank on ``Δ`` vs 0.

This matches the claim: *the scripted scenario elicits higher trait activation than the
direct (baseline) prompt.*

**Trajectory** stage-by-stage: descriptive only — use the line plot from
``analyze_trait_activation_stages.py --all-models``; do not significance-test every stage.

Optional ``--exploratory-contrasts``: prints the old stage-vs-baseline table (multiple tests);
use only for exploration, not as primary evidence.

Requires scipy. Install: ``pip install scipy``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TURN_STAGES = ["turn_1", "turn_2", "turn_3", "turn_4"]


def _paired_diffs(
    per_model: List[Dict[str, Any]], a: str, b: str
) -> Tuple[List[float], int]:
    d: List[float] = []
    for row in per_model:
        ma = row.get("mean_activation_by_stage") or {}
        va = ma.get(a)
        vb = ma.get(b)
        if va is None or vb is None:
            continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if math.isnan(float(va)) or math.isnan(float(vb)):
                continue
            d.append(float(va) - float(vb))
    return d, len(d)


def scenario_agg_minus_baseline(per_model: List[Dict[str, Any]]) -> List[float]:
    """Δ_m = mean(S1..S4)_m - baseline_m for each model with complete data."""
    diffs: List[float] = []
    for row in per_model:
        ma = row.get("mean_activation_by_stage") or {}
        b = ma.get("baseline")
        turns = [ma.get(t) for t in TURN_STAGES]
        if b is None or any(x is None for x in turns):
            continue
        try:
            fb = float(b)
            tv = [float(x) for x in turns]
        except (TypeError, ValueError):
            continue
        if any(math.isnan(fb) or math.isnan(x) for x in tv):
            continue
        agg = float(np.mean(tv))
        diffs.append(agg - fb)
    return diffs


def cohen_dz(diffs: List[float]) -> float:
    if len(diffs) < 2:
        return math.nan
    x = np.array(diffs, dtype=float)
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return math.nan
    return float(x.mean() / sd)


def bonferroni(p: float, k: int) -> float:
    return min(1.0, p * k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT / "trait_activation" / "results" / "trait_activation_stages_all_models.json",
    )
    ap.add_argument(
        "--exploratory-contrasts",
        action="store_true",
        help="Also print stage-by-stage vs baseline (multiple tests; not for primary claims).",
    )
    args = ap.parse_args()

    try:
        from scipy import stats
    except ImportError:
        print("Install scipy: pip install scipy", file=sys.stderr)
        return 1

    data = json.loads(args.summary_json.read_text(encoding="utf-8"))
    per_model = data.get("per_model") or []
    if not per_model:
        print("No per_model entries in JSON.", file=sys.stderr)
        return 1

    tau = data.get("tau_primary", "?")
    print(f"File: {args.summary_json}")
    print(f"n_models (in file): {len(per_model)}  tau: {tau}\n")

    # --- Primary test ---
    diffs = scenario_agg_minus_baseline(per_model)
    n = len(diffs)
    print("=" * 72)
    print("PRIMARY TEST:  mean(S₁–S₄) activation  vs.  baseline  (paired across models)")
    print("=" * 72)
    print(
        "Definition: For each model, average the mean #traits≥τ at turn_1…turn_4 (OG), "
        "then subtract the model’s baseline mean. Test whether mean(Δ) ≠ 0.\n"
    )
    if n < 3:
        print(f"Insufficient paired models (n={n}). Need complete S₁–S₄ and baseline.")
        return 1

    d_arr = np.array(diffs, dtype=float)
    mean_d = float(d_arr.mean())
    dz = cohen_dz(diffs)
    tt = stats.ttest_1samp(d_arr, 0.0)
    wn = stats.wilcoxon(d_arr, alternative="two-sided", zero_method="wilcox")
    tp = float(tt.pvalue)
    wp = float(getattr(wn, "pvalue", getattr(wn, "p", math.nan)))

    print(f"  n (paired models):     {n}")
    print(f"  mean Δ (S̄₁₋₄ − base):  {mean_d:+.4f}  (activated-trait count scale, 0–10)")
    print(f"  Cohen's dz:            {dz:.3f}")
    print(f"  paired t-test p:       {tp:.4g}")
    print(f"  Wilcoxon p:            {wp:.4g}")
    print()

    # --- Exploratory ---
    if args.exploratory_contrasts:
        print("=" * 72)
        print("EXPLORATORY (not primary): each stage vs baseline; Bonferroni ×5 for t-tests")
        print("=" * 72)
        k = 5
        hdr = (
            f"{'contrast':<26} {'n':>3} {'meanΔ':>8} {'dz':>7} "
            f"{'p_t':>9} {'p_t*':>9} {'p_w':>9} {'p_w*':>9}"
        )
        print(hdr)
        print("-" * len(hdr))
        for st in ["turn_1", "turn_2", "turn_3", "turn_4", "final"]:
            dlist, nn = _paired_diffs(per_model, st, "baseline")
            label = f"{st} − baseline"
            if nn < 3:
                print(f"{label:<26} {nn:>3}  (skip)")
                continue
            d_a = np.array(dlist, dtype=float)
            tt2 = stats.ttest_1samp(d_a, 0.0)
            w2 = stats.wilcoxon(d_a, alternative="two-sided", zero_method="wilcox")
            tp2 = float(tt2.pvalue)
            wp2 = float(getattr(w2, "pvalue", getattr(w2, "p", math.nan)))
            print(
                f"{label:<26} {nn:>3} {float(d_a.mean()):>8.4f} {cohen_dz(dlist):>7.3f} "
                f"{tp2:>9.4g} {bonferroni(tp2, k):>9.4g} {wp2:>9.4g} {bonferroni(wp2, k):>9.4g}"
            )
        print()

    print("Interpretation: Report the primary test for the paper; use the trajectory figure")
    print("for qualitative pattern (e.g. rise mid-scenario, partial return at debrief).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
