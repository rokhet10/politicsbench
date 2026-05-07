#!/usr/bin/env python3
"""
Trait activation across **all standard evaluation stages**:

  baseline → turn 1–4 → final (debrief trait rubric in ``rubric_scores``).

**Activation (per task, per stage):** count of traits in ``RUBRIC_TRAIT_KEYS`` with mean judge
score ≥ τ (default τ=14; pass ``--tau`` for multiple thresholds).

Writes JSON summary + matplotlib figures:

  **Single run** (default):

  - ``activation_mean_by_stage_<run_key>.png`` — mean count of traits ≥τ (bars + 95% CI across scenarios)
  - ``trait_mean_heatmap_<run_key>.png`` — mean trait score (0–20) × stage
  - ``activation_count_distribution_<run_key>.png`` — histogram of “# traits ≥τ” at final stage

  **All models** (``--all-models``):

  - Defaults to **OG scenarios only** (``.*-og$``); override with ``--scenario-regex``.
  - Per model: mean activation across filtered scenarios. Then **mean across models** with error bars =
    ``std`` / ``sem`` / ``ci95`` **across models** (not across scenarios).
  - **Line plot** (recommended for sequential benchmarks): Baseline → ``S₁``–``S₄`` → Debrief on *x*,
    mean activated traits on *y* — ``activation_trajectory_line_all_models_OG.png``
  - ``trait_activation_stages_all_models.json``, ``trait_mean_heatmap_all_models_OG.png``

No API calls; reads ``eqbench_runs_final.json`` or any runs file shape used by eqbench3.

Complements ``analyze_trait_activation_entropy.py`` (which uses blanket / stage1 / full only and
defaults to ``-og`` regex).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RUBRIC_TRAIT_KEYS  # noqa: E402

STAGE_KEYS = ["baseline", "turn_1", "turn_2", "turn_3", "turn_4", "final"]


def _scenario_ids(iblock: Dict[str, Any], pat: re.Pattern) -> List[str]:
    return sorted(
        [k for k in iblock if isinstance(iblock[k], dict) and pat.search(str(k))],
        key=lambda x: (
            int(m.group(1)),
            str(x),
        )
        if (m := re.match(r"^(\d+)-", str(x)))
        else (9999, str(x)),
    )


def _mean_across_judges(judge_dicts: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not judge_dicts:
        return None
    out: Dict[str, float] = {}
    for k in RUBRIC_TRAIT_KEYS:
        vals: List[float] = []
        for d in judge_dicts:
            if not isinstance(d, dict):
                return None
            v = d.get(k)
            if v is None:
                return None
            vals.append(float(v))
        out[k] = sum(vals) / len(vals)
    return out


def _vec_baseline(task: Dict[str, Any]) -> Optional[Dict[str, float]]:
    agg = task.get("baseline_rubric_scores")
    if isinstance(agg, dict) and all(k in agg for k in RUBRIC_TRAIT_KEYS):
        return {k: float(agg[k]) for k in RUBRIC_TRAIT_KEYS}
    by = task.get("baseline_rubric_scores_by_judge")
    if isinstance(by, list) and by:
        return _mean_across_judges(by)
    return None


def _vec_turn(task: Dict[str, Any], turn_index: int) -> Optional[Dict[str, float]]:
    turns = task.get("turn_rubric_scores") or []
    if len(turns) > turn_index and isinstance(turns[turn_index], dict):
        agg = turns[turn_index]
        if all(k in agg for k in RUBRIC_TRAIT_KEYS):
            return {k: float(agg[k]) for k in RUBRIC_TRAIT_KEYS}
    tj = task.get("turn_rubric_scores_by_judge") or []
    if len(tj) > turn_index and isinstance(tj[turn_index], list) and tj[turn_index]:
        return _mean_across_judges(tj[turn_index])
    return None


def _vec_final(task: Dict[str, Any]) -> Optional[Dict[str, float]]:
    agg = task.get("rubric_scores")
    if isinstance(agg, dict) and all(k in agg for k in RUBRIC_TRAIT_KEYS):
        return {k: float(agg[k]) for k in RUBRIC_TRAIT_KEYS}
    by = task.get("rubric_scores_by_judge")
    if isinstance(by, list) and by:
        return _mean_across_judges(by)
    return None


def activation_count(vec: Dict[str, float], tau: float) -> int:
    return sum(1 for k in RUBRIC_TRAIT_KEYS if vec[k] >= tau)


def _stage_extractors():
    """One callable per stage in STAGE_KEYS order."""
    return (
        _vec_baseline,
        lambda t: _vec_turn(t, 0),
        lambda t: _vec_turn(t, 1),
        lambda t: _vec_turn(t, 2),
        lambda t: _vec_turn(t, 3),
        _vec_final,
    )


def collect_stage_activation_counts(
    iblock: Dict[str, Any],
    scenario_ids: List[str],
    tau: float,
) -> Dict[str, List[float]]:
    extractors = _stage_extractors()
    per_stage: Dict[str, List[float]] = {s: [] for s in STAGE_KEYS}
    for sid in scenario_ids:
        task = iblock[sid]
        for sk, ex in zip(STAGE_KEYS, extractors):
            vec = ex(task)
            if vec is None:
                continue
            per_stage[sk].append(float(activation_count(vec, tau)))
    return per_stage


def mean_activation_by_stage(
    iblock: Dict[str, Any],
    scenario_ids: List[str],
    tau: float,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    per_stage = collect_stage_activation_counts(iblock, scenario_ids, tau)
    means: Dict[str, float] = {}
    ns: Dict[str, int] = {}
    for sk in STAGE_KEYS:
        vals = per_stage[sk]
        means[sk] = float(np.mean(vals)) if vals else math.nan
        ns[sk] = len(vals)
    return means, ns


def trait_mean_matrix(iblock: Dict[str, Any], scenario_ids: List[str]) -> np.ndarray:
    """Mean trait score (10 × 6) across scenarios (nan-aware)."""
    extractors = _stage_extractors()
    chunks: List[np.ndarray] = []
    for sid in scenario_ids:
        task = iblock[sid]
        M = np.full((len(RUBRIC_TRAIT_KEYS), len(STAGE_KEYS)), math.nan)
        for j, ex in enumerate(extractors):
            vec = ex(task)
            if vec is None:
                continue
            for i, trait in enumerate(RUBRIC_TRAIT_KEYS):
                M[i, j] = vec[trait]
        if np.any(~np.isnan(M)):
            chunks.append(M)
    if not chunks:
        return np.full((len(RUBRIC_TRAIT_KEYS), len(STAGE_KEYS)), math.nan)
    return np.nanmean(np.stack(chunks, axis=0), axis=0)


def _pick_run(runs: Dict[str, Any], run_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if run_key:
        if run_key not in runs:
            raise KeyError(f"run_key {run_key!r} not in file")
        return run_key, runs[run_key]
    if len(runs) == 1:
        k = next(iter(runs))
        return k, runs[k]
    best_k: Optional[str] = None
    best_ts = ""
    for k, v in runs.items():
        if not isinstance(v, dict):
            continue
        ts = (
            (v.get("results") or {}).get("end_time")
            or v.get("end_time")
            or v.get("start_time")
            or ""
        )
        if isinstance(ts, str) and ts > best_ts:
            best_ts = ts
            best_k = k
    if best_k:
        return best_k, runs[best_k]
    raise ValueError("Could not pick run; pass --run-key")


def _ci95(vals: List[float]) -> Tuple[float, float, float]:
    arr = np.array([x for x in vals if not math.isnan(x)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    m = float(arr.mean())
    if arr.size < 2:
        return m, m, m
    sem = float(arr.std(ddof=1) / math.sqrt(arr.size))
    h = 1.96 * sem
    return m, m - h, m + h


def _model_level_std_sem(
    values: List[float], error_bar: str
) -> Tuple[float, float]:
    """Return (lower_err, upper_err) for matplotlib symmetric or asymmetric yerr."""
    arr = np.array([x for x in values if not math.isnan(x)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return 0.0, 0.0
    mean_v = float(arr.mean())
    std_v = float(arr.std(ddof=1))
    if error_bar == "std":
        return std_v, std_v
    if error_bar == "sem":
        sem_v = std_v / math.sqrt(arr.size)
        return sem_v, sem_v
    # ci95 across models (treat model means as samples)
    sem_v = std_v / math.sqrt(arr.size)
    h = 1.96 * sem_v
    return h, h


def run_all_models(args: argparse.Namespace) -> int:
    runs = json.loads(args.runs_json.read_text(encoding="utf-8"))
    pat_str = (
        args.scenario_regex
        if args.scenario_regex is not None
        else (r".*-og$" if args.all_models else r".*")
    )
    pat = re.compile(pat_str)
    primary_tau = args.tau[0]

    per_model: List[Dict[str, Any]] = []
    heatmap_stack: List[np.ndarray] = []

    for rk, run_obj in sorted(runs.items()):
        if not isinstance(run_obj, dict):
            continue
        st = run_obj.get("scenario_tasks") or {}
        iblock = st.get(args.iteration) or st.get(str(args.iteration))
        if not isinstance(iblock, dict):
            continue
        scenario_ids = _scenario_ids(iblock, pat)
        if not scenario_ids:
            continue

        means, ns = mean_activation_by_stage(iblock, scenario_ids, primary_tau)
        if all(math.isnan(means[s]) for s in STAGE_KEYS):
            continue

        label = run_obj.get("model_name") or run_obj.get("test_model") or rk
        per_model.append(
            {
                "run_key": rk,
                "model_name": label,
                "n_scenarios": len(scenario_ids),
                "mean_activation_by_stage": means,
                "n_obs_by_stage": ns,
            }
        )
        heatmap_stack.append(trait_mean_matrix(iblock, scenario_ids))

    if not per_model:
        print("No runs with matching scenarios; check --scenario-regex / iteration.", file=sys.stderr)
        return 1

    # Stack model-level means per stage (variation across models)
    stage_model_matrix = np.full((len(per_model), len(STAGE_KEYS)), math.nan)
    for i, row in enumerate(per_model):
        for j, sk in enumerate(STAGE_KEYS):
            stage_model_matrix[i, j] = row["mean_activation_by_stage"][sk]

    grand_mean = np.nanmean(stage_model_matrix, axis=0)
    yerr_lo: List[float] = []
    yerr_hi: List[float] = []
    for j in range(len(STAGE_KEYS)):
        col = [stage_model_matrix[i, j] for i in range(len(per_model))]
        lo, hi = _model_level_std_sem(col, args.error_bar)
        yerr_lo.append(lo)
        yerr_hi.append(hi)

    hm_agg = np.nanmean(np.stack(heatmap_stack, axis=0), axis=0)

    report = {
        "mode": "all_models",
        "runs_file": str(args.runs_json.resolve()),
        "scenario_regex_effective": pat_str,
        "iteration": args.iteration,
        "tau_primary": primary_tau,
        "error_bar_across_models": args.error_bar,
        "n_models": len(per_model),
        "per_model": per_model,
        "across_models_mean_activation_by_stage": {
            STAGE_KEYS[j]: float(grand_mean[j]) for j in range(len(STAGE_KEYS))
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "trait_activation_stages_all_models.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}", file=sys.stderr)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib missing ({e})", file=sys.stderr)
        return 0

    x = np.arange(len(STAGE_KEYS), dtype=float)
    x_labels_pub = [
        "Baseline",
        r"$S_1$",
        r"$S_2$",
        r"$S_3$",
        r"$S_4$",
        "Debrief",
    ]

    fig1, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.errorbar(
        x,
        grand_mean,
        yerr=[yerr_lo, yerr_hi],
        fmt="-o",
        color="darkslateblue",
        capsize=4,
        markersize=9,
        linewidth=2,
        elinewidth=1.2,
        alpha=0.92,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels_pub)
    ax1.set_xlabel("Evaluation point")
    ax1.set_ylabel(f"Mean activated traits (count ≥ {primary_tau:g} of {len(RUBRIC_TRAIT_KEYS)})")
    ax1.set_ylim(3.5, 6.5)
    err_name = {"std": "SD", "sem": "SEM", "ci95": "95% CI"}.get(
        args.error_bar, args.error_bar
    )
    ax1.set_title(
        f"Trait activation trajectory — mean across models (n={len(per_model)}), OG scenarios\n"
        f"Error bars: {err_name} across models"
    )
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.grid(True, axis="x", alpha=0.15)
    fig1.tight_layout()
    p1 = args.out_dir / "activation_trajectory_line_all_models_OG.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"Wrote {p1}", file=sys.stderr)

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    im = ax2.imshow(hm_agg, aspect="auto", cmap="viridis", vmin=0, vmax=20)
    ax2.set_xticks(np.arange(len(STAGE_KEYS)))
    ax2.set_xticklabels(STAGE_KEYS, rotation=25, ha="right")
    ax2.set_yticks(np.arange(len(RUBRIC_TRAIT_KEYS)))
    ax2.set_yticklabels(RUBRIC_TRAIT_KEYS, fontsize=8)
    ax2.set_title(
        f"Mean trait scores (0–20), mean across models — OG (n={len(per_model)} models)"
    )
    fig2.colorbar(im, ax=ax2, label="Mean score")
    fig2.tight_layout()
    p2 = args.out_dir / "trait_mean_heatmap_all_models_OG.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"Wrote {p2}", file=sys.stderr)

    print(f"all_models n={len(per_model)} tau={primary_tau} regex={pat_str!r} error_bar={args.error_bar}")
    for sk, gm in zip(STAGE_KEYS, grand_mean):
        print(f"  {sk}: across-model mean of (within-model OG means) = {gm:.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-json", type=Path, required=True)
    ap.add_argument("--run-key", default=None)
    ap.add_argument("--iteration", default="1")
    ap.add_argument(
        "--scenario-regex",
        default=None,
        help=r"Filter scenario_ids. Default: all scenarios for single-run; OG-only (.*-og$) for --all-models.",
    )
    ap.add_argument(
        "--all-models",
        action="store_true",
        help="Aggregate across every run in the JSON; error bars = spread across models.",
    )
    ap.add_argument(
        "--error-bar",
        choices=["std", "sem", "ci95"],
        default="std",
        help="Error bar type across models (default std).",
    )
    ap.add_argument(
        "--tau",
        type=float,
        nargs="+",
        default=[14.0],
        help="Activation threshold(s); count traits with score >= tau.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "trait_activation" / "results",
    )
    args = ap.parse_args()

    if args.all_models:
        if args.run_key:
            print("Ignoring --run-key because --all-models is set.", file=sys.stderr)
        return run_all_models(args)

    runs = json.loads(args.runs_json.read_text(encoding="utf-8"))
    rk, run_obj = _pick_run(runs, args.run_key)
    pat_str = args.scenario_regex if args.scenario_regex is not None else r".*"
    pat = re.compile(pat_str)

    st = run_obj.get("scenario_tasks") or {}
    iblock = st.get(args.iteration) or st.get(str(args.iteration))
    if not isinstance(iblock, dict):
        print("Missing scenario_tasks for iteration", args.iteration, file=sys.stderr)
        return 1

    scenario_ids = _scenario_ids(iblock, pat)

    stage_keys = STAGE_KEYS
    extractors = _stage_extractors()

    activation_by_tau: Dict[float, Dict[str, List[float]]] = {
        t: {s: [] for s in stage_keys} for t in args.tau
    }
    trait_means: Dict[str, Dict[str, List[float]]] = {
        s: {trait: [] for trait in RUBRIC_TRAIT_KEYS} for s in stage_keys
    }
    per_rows: List[Dict[str, Any]] = []

    for sid in scenario_ids:
        task = iblock[sid]
        row: Dict[str, Any] = {"scenario_id": sid}
        for sk, ex in zip(stage_keys, extractors):
            vec = ex(task)
            if vec is None:
                row[f"missing_{sk}"] = True
                for tau in args.tau:
                    row[f"activation_ge_{tau}_{sk}"] = None
                continue
            row[f"missing_{sk}"] = False
            for tau in args.tau:
                c = float(activation_count(vec, tau))
                activation_by_tau[tau][sk].append(c)
                row[f"activation_ge_{tau}_{sk}"] = c
            for trait in RUBRIC_TRAIT_KEYS:
                trait_means[sk][trait].append(vec[trait])

        per_rows.append(row)

    agg_activation: Dict[str, Any] = {}
    for tau in args.tau:
        agg_activation[f"tau_{tau}"] = {}
        for sk in stage_keys:
            vals = activation_by_tau[tau][sk]
            m, lo, hi = _ci95(vals)
            agg_activation[f"tau_{tau}"][sk] = {
                "n": float(len(vals)),
                "mean": m,
                "ci95_low": lo,
                "ci95_high": hi,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

    heatmap_matrix = np.zeros((len(RUBRIC_TRAIT_KEYS), len(stage_keys)))
    for j, sk in enumerate(stage_keys):
        for i, trait in enumerate(RUBRIC_TRAIT_KEYS):
            xs = trait_means[sk][trait]
            heatmap_matrix[i, j] = float(np.mean(xs)) if xs else math.nan

    report = {
        "run_key": rk,
        "runs_file": str(args.runs_json.resolve()),
        "iteration": args.iteration,
        "scenario_regex": pat_str,
        "n_scenarios": len(scenario_ids),
        "stage_order": stage_keys,
        "tau_values": list(args.tau),
        "activation_aggregate": agg_activation,
        "per_scenario": per_rows,
        "notes": [
            "final uses rubric_scores (trait judge on debrief).",
            "activation_ge_tau counts traits with mean judge score >= tau (default 14).",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"trait_activation_stages_{rk}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}", file=sys.stderr)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib missing; skip plots ({e})", file=sys.stderr)
        print(json.dumps({k: v for k, v in report.items() if k != "per_scenario"}, indent=2))
        return 0

    x = np.arange(len(stage_keys))
    primary_tau = args.tau[0]

    # --- Fig 1: mean activation count at primary tau ---
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    means = [
        agg_activation[f"tau_{primary_tau}"][sk]["mean"] for sk in stage_keys
    ]
    lows = [
        means[j] - agg_activation[f"tau_{primary_tau}"][sk]["ci95_low"]
        for j, sk in enumerate(stage_keys)
    ]
    highs = [
        agg_activation[f"tau_{primary_tau}"][sk]["ci95_high"] - means[j]
        for j, sk in enumerate(stage_keys)
    ]
    ax1.bar(x, means, color="steelblue", alpha=0.85, yerr=[lows, highs], capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_keys, rotation=25, ha="right")
    ax1.set_ylabel(f"Mean # traits ≥ {primary_tau:g} (of {len(RUBRIC_TRAIT_KEYS)})")
    ax1.set_ylim(0, len(RUBRIC_TRAIT_KEYS) + 0.5)
    ax1.set_title(f"Trait activation by stage — {rk}")
    ax1.grid(True, axis="y", alpha=0.3)
    fig1.tight_layout()
    p1 = args.out_dir / f"activation_mean_by_stage_{rk}.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"Wrote {p1}", file=sys.stderr)

    # --- Fig 2: heatmap mean trait score ---
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    im = ax2.imshow(heatmap_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=20)
    ax2.set_xticks(np.arange(len(stage_keys)))
    ax2.set_xticklabels(stage_keys, rotation=25, ha="right")
    ax2.set_yticks(np.arange(len(RUBRIC_TRAIT_KEYS)))
    ax2.set_yticklabels(RUBRIC_TRAIT_KEYS, fontsize=8)
    ax2.set_title(f"Mean trait scores (0–20) — {rk}")
    fig2.colorbar(im, ax=ax2, label="Mean score")
    fig2.tight_layout()
    p2 = args.out_dir / f"trait_mean_heatmap_{rk}.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"Wrote {p2}", file=sys.stderr)

    # --- Fig 3: histogram at final stage for primary tau ---
    final_counts = activation_by_tau[primary_tau]["final"]
    if final_counts:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.hist(
            final_counts,
            bins=np.arange(-0.5, len(RUBRIC_TRAIT_KEYS) + 1.5, 1),
            color="coral",
            edgecolor="white",
            alpha=0.9,
        )
        ax3.set_xlabel(f"# traits ≥ {primary_tau:g} at final (debrief judge)")
        ax3.set_ylabel("Number of scenarios")
        ax3.set_title(f"Distribution — {rk}")
        ax3.set_xticks(range(0, len(RUBRIC_TRAIT_KEYS) + 1))
        fig3.tight_layout()
        p3 = args.out_dir / f"activation_count_distribution_final_{rk}.png"
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)
        print(f"Wrote {p3}", file=sys.stderr)

    # stdout: compact summary for primary tau
    print(f"run_key={rk} n_scenarios={len(scenario_ids)} tau={primary_tau}")
    for sk in stage_keys:
        a = agg_activation[f"tau_{primary_tau}"][sk]
        print(
            f"  {sk}: mean_traits_ge_{primary_tau:g}={a['mean']:.3f} "
            f"n={int(a['n'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
