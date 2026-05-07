#!/usr/bin/env python3
"""
Inter-judge agreement from a runs JSON where each turn/baseline has
``*_rubric_scores_by_judge`` (list of one dict per judge, same order as run ``judge_models``).

Reports mean variance of the weighted ideological scalar and:
  - Fleiss' kappa on ordinal bins of that scalar (3 raters)
  - Mean pairwise quadratic-weighted Cohen's kappa on the same bins (numpy-only)

Filter scenarios with ``--scenario-regex`` (default: ``-og$`` for original-wording arms).

With ``--all-models``, pools OG scenarios across all runs in the file and reports a single
agreement trajectory over stages (baseline, S1–S4, debrief).

Writes an optional figure via ``--out-png`` (defaults next to ``--out-json`` when set).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RUBRIC_CRITERION_WEIGHTS


TRAIT_KEYS = list(RUBRIC_CRITERION_WEIGHTS.keys())


def weighted_scalar(scores: Optional[Dict[str, float]], weights: Dict[str, float]) -> float:
    if not scores:
        return math.nan
    s = 0.0
    for k, w in weights.items():
        v = scores.get(k)
        if v is None:
            return math.nan
        s += float(v) * float(w)
    return s


def scalar_to_bin(z: float, n_bins: int = 11, lo: float = -24.0, hi: float = 24.0) -> int:
    """Map weighted scalar to 0..n_bins-1 for agreement metrics."""
    if math.isnan(z):
        return -1
    span = hi - lo
    if span <= 0:
        return 0
    t = (z - lo) / span * (n_bins - 1)
    return int(max(0, min(n_bins - 1, round(t))))


def cohen_kappa_quadratic_weighted(y1: np.ndarray, y2: np.ndarray, n_cat: int) -> float:
    """
    Quadratic-weighted Cohen's kappa for ordinal ratings (numpy-only).
    Same weight scheme as common implementations: w_ij = ((i-j)^2) / ((K-1)^2).
    """
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = len(y1)
    if n == 0 or len(y2) != n:
        return math.nan
    if np.array_equal(y1, y2):
        return 1.0
    O = np.zeros((n_cat, n_cat), dtype=float)
    for a, b in zip(y1, y2):
        if 0 <= a < n_cat and 0 <= b < n_cat:
            O[a, b] += 1.0
    tot = O.sum()
    if tot == 0:
        return math.nan
    O /= tot
    denom_sq = max((n_cat - 1) ** 2, 1)
    w = np.fromfunction(lambda i, j: (i - j) ** 2 / denom_sq, (n_cat, n_cat), dtype=float)
    row_marg = O.sum(axis=1)
    col_marg = O.sum(axis=0)
    E = np.outer(row_marg, col_marg)
    num = float(np.sum(w * O))
    den = float(np.sum(w * E))
    if den < 1e-12:
        return math.nan
    return 1.0 - num / den


def fleiss_kappa(count_matrix: np.ndarray) -> float:
    """
    Fleiss' kappa. count_matrix shape (n_subjects, n_categories);
    each row sums to the same number of raters R.
    """
    mat = np.asarray(count_matrix, dtype=float)
    if mat.size == 0:
        return math.nan
    n, k = mat.shape
    row_sums = mat.sum(axis=1)
    if not np.allclose(row_sums, row_sums[0]):
        return math.nan
    r = row_sums[0]
    if r < 2 or n < 2:
        return math.nan
    p_j = mat.sum(axis=0) / (n * r)
    p_i = (mat * (mat - 1)).sum(axis=1) / (r * (r - 1))
    p_bar = float(p_i.mean())
    p_e = float(np.sum(p_j**2))
    den = 1.0 - p_e
    if abs(den) < 1e-12:
        return math.nan
    return (p_bar - p_e) / den


def mean_pairwise_quadratic_kappa(
    labels: np.ndarray, n_cat: int
) -> Tuple[float, List[float]]:
    """labels shape (n_items, n_judges), integer ordinal categories in [0, n_cat-1]."""
    j = labels.shape[1]
    if j < 2:
        return math.nan, []
    vals: List[float] = []
    for a in range(j):
        for b in range(a + 1, j):
            y1, y2 = labels[:, a], labels[:, b]
            vals.append(cohen_kappa_quadratic_weighted(y1, y2, n_cat))
    clean = [v for v in vals if not math.isnan(v)]
    return (float(np.mean(clean)) if clean else math.nan, vals)


def _pick_run(runs: Dict[str, Any], run_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if run_key:
        if run_key not in runs:
            raise KeyError(f"run_key {run_key!r} not found; have {sorted(runs)!r}")
        return run_key, runs[run_key]
    if len(runs) == 1:
        k = next(iter(runs))
        return k, runs[k]
    raise ValueError("Multiple runs in JSON; pass --run-key")


def agreement_for_matrix(
    scalar_matrix: np.ndarray,
    n_bins: int = 11,
) -> Dict[str, Any]:
    """
    scalar_matrix: (n_items, n_judges), may contain nan — rows with any nan skipped for that metric block.
    """
    n_items, n_j = scalar_matrix.shape
    valid_rows = ~np.isnan(scalar_matrix).any(axis=1)
    n_ok = int(valid_rows.sum())
    if n_ok < 1:
        return {
            "n_items_used": 0,
            "mean_within_item_var_scalar": math.nan,
            "fleiss_kappa_bins": math.nan,
            "mean_pairwise_quadratic_weighted_cohen_kappa": math.nan,
            "pairwise_quadratic_weighted_cohen_kappas": [],
        }
    sm = scalar_matrix[valid_rows]
    row_vars = np.var(sm, axis=1, ddof=1)
    mean_var = float(np.mean(row_vars))

    if n_ok < 2:
        return {
            "n_items_used": n_ok,
            "mean_within_item_var_scalar": mean_var,
            "fleiss_kappa_bins": math.nan,
            "mean_pairwise_quadratic_weighted_cohen_kappa": math.nan,
            "pairwise_quadratic_weighted_cohen_kappas": [],
        }

    bins = np.array(
        [
            [scalar_to_bin(float(sm[i, k]), n_bins=n_bins) for k in range(n_j)]
            for i in range(sm.shape[0])
        ]
    )
    if (bins < 0).any():
        return {
            "n_items_used": int(sm.shape[0]),
            "mean_within_item_var_scalar": mean_var,
            "fleiss_kappa_bins": math.nan,
            "mean_pairwise_quadratic_weighted_cohen_kappa": math.nan,
            "pairwise_quadratic_weighted_cohen_kappas": [],
        }

    counts = np.zeros((sm.shape[0], n_bins), dtype=int)
    for i in range(sm.shape[0]):
        for k in range(n_j):
            counts[i, int(bins[i, k])] += 1
    fk = fleiss_kappa(counts)
    mq, pairs = mean_pairwise_quadratic_kappa(bins, n_bins)
    return {
        "n_items_used": int(sm.shape[0]),
        "mean_within_item_var_scalar": mean_var,
        "fleiss_kappa_bins": float(fk) if not math.isnan(fk) else math.nan,
        "mean_pairwise_quadratic_weighted_cohen_kappa": mq,
        "pairwise_quadratic_weighted_cohen_kappas": [
            float(x) if not math.isnan(x) else None for x in pairs
        ],
    }


def extract_judge_matrix_for_stage(
    task: Dict[str, Any],
    stage: str,
    n_judges_expected: int,
) -> Optional[np.ndarray]:
    """
    stage: 'baseline' | 'turn_1'..'turn_4' | 'final'
    Returns (1, n_judges) matrix of weighted scalars for this single task, or None if missing.
    """
    if stage == "baseline":
        per = task.get("baseline_rubric_scores_by_judge")
    elif stage == "final":
        per = task.get("rubric_scores_by_judge")
    elif stage.startswith("turn_"):
        try:
            idx = int(stage.split("_", 1)[1]) - 1
        except (ValueError, IndexError):
            raise ValueError(stage)
        tj = task.get("turn_rubric_scores_by_judge") or []
        per = tj[idx] if 0 <= idx < len(tj) else None
    else:
        raise ValueError(stage)

    if not isinstance(per, list) or len(per) != n_judges_expected:
        return None
    scalars = []
    for d in per:
        if not isinstance(d, dict):
            return None
        scalars.append(weighted_scalar(d, RUBRIC_CRITERION_WEIGHTS))
    if any(math.isnan(s) for s in scalars):
        return None
    return np.array([scalars], dtype=float)


def _stage_label_pub(stage: str) -> str:
    if stage == "baseline":
        return "Baseline"
    if stage == "final":
        return "Debrief"
    if stage.startswith("turn_"):
        try:
            n = int(stage.split("_", 1)[1])
            return rf"$S_{n}$"
        except (ValueError, IndexError):
            return stage
    return stage


def _plot_agreement_trajectory(summary: Dict[str, Any], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib not available; skip plot ({e})", file=sys.stderr)
        return

    stage_order = summary.get("stage_order") or []
    if not isinstance(stage_order, list) or not stage_order:
        return
    stages_block = summary.get("stages") or {}
    ys: List[float] = []
    for st in stage_order:
        b = stages_block.get(st) or {}
        v = b.get("mean_pairwise_quadratic_weighted_cohen_kappa")
        ys.append(float(v) if isinstance(v, (int, float)) else math.nan)

    x = np.arange(len(stage_order), dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(x, ys, "-o", color="darkslateblue", linewidth=2, markersize=8, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels([_stage_label_pub(s) for s in stage_order])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Agreement (mean pairwise quadratic-weighted κ)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.15)
    ax.set_title(summary.get("title") or "Inter-judge agreement by stage")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}", file=sys.stderr)


def _plot_metric_trajectory(
    summary: Dict[str, Any],
    out_png: Path,
    metric: str = "kappa",
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib not available; skip plot ({e})", file=sys.stderr)
        return

    metric_key = (
        "mean_pairwise_quadratic_weighted_cohen_kappa"
        if metric == "kappa"
        else "mean_within_item_var_scalar"
    )
    ylabel = (
        "Agreement (mean pairwise quadratic-weighted κ)"
        if metric == "kappa"
        else "Judge disagreement (mean within-item variance)"
    )

    stage_order = summary.get("stage_order") or []
    if not isinstance(stage_order, list) or not stage_order:
        return
    stages_block = summary.get("stages") or {}
    ys: List[float] = []
    for st in stage_order:
        b = stages_block.get(st) or {}
        v = b.get(metric_key)
        ys.append(float(v) if isinstance(v, (int, float)) else math.nan)

    x = np.arange(len(stage_order), dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(x, ys, "-o", color="darkslateblue", linewidth=2, markersize=8, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels([_stage_label_pub(s) for s in stage_order])
    ax.set_xlabel("Stage")
    ax.set_ylabel(ylabel)
    if ymin is not None and ymax is not None:
        ax.set_ylim(float(ymin), float(ymax))
    elif metric == "kappa":
        ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.15)
    title = summary.get("title") or "Inter-judge trajectory"
    ax.set_title(f"{title} ({metric})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}", file=sys.stderr)


def _plot_agreement_overlay_models(
    summary: Dict[str, Any], out_png: Path, ymin: float = 0.7, ymax: float = 1.0
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib not available; skip overlay plot ({e})", file=sys.stderr)
        return

    per_model = summary.get("per_model") or []
    stage_order = summary.get("stage_order") or []
    if not isinstance(per_model, list) or not per_model or not isinstance(stage_order, list):
        return

    x = np.arange(len(stage_order), dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    cmap = plt.get_cmap("tab10")

    for i, row in enumerate(per_model):
        m = row.get("mean_pairwise_quadratic_weighted_cohen_kappa_by_stage") or {}
        ys = [float(m[s]) if isinstance(m.get(s), (int, float)) else math.nan for s in stage_order]
        label = str(row.get("model_name") or row.get("run_key") or f"model_{i+1}")
        if len(label) > 34:
            label = label[:31] + "..."
        ax.plot(x, ys, "-o", linewidth=1.7, markersize=5.5, alpha=0.9, color=cmap(i % 10), label=label)

    pooled = summary.get("stages") or {}
    ys_pool = [
        float((pooled.get(s) or {}).get("mean_pairwise_quadratic_weighted_cohen_kappa"))
        if isinstance((pooled.get(s) or {}).get("mean_pairwise_quadratic_weighted_cohen_kappa"), (int, float))
        else math.nan
        for s in stage_order
    ]
    ax.plot(x, ys_pool, "k-s", linewidth=2.6, markersize=6.5, label="Pooled mean", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels([_stage_label_pub(s) for s in stage_order])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Agreement (mean pairwise quadratic-weighted κ)")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.15)
    ax.set_title("Judge agreement by stage — model overlay (OG only)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}", file=sys.stderr)


def _trait_variance_for_run(
    run_obj: Dict[str, Any],
    *,
    iteration: str,
    scenario_pat: re.Pattern[str],
) -> Optional[Dict[str, Any]]:
    st = run_obj.get("scenario_tasks") or {}
    iblock = st.get(iteration) or st.get(str(iteration))
    if not isinstance(iblock, dict):
        return None

    scenarios = _scenarios_sorted(iblock, scenario_pat)
    if not scenarios:
        return None

    stage_order = ["baseline", "turn_1", "turn_2", "turn_3", "turn_4", "final"]
    out: Dict[str, Dict[str, Any]] = {
        s: {t: {"vals": [], "n_items_used": 0} for t in TRAIT_KEYS} for s in stage_order
    }

    for sid in scenarios:
        task = iblock[sid]
        for stage in stage_order:
            if stage == "baseline":
                per = task.get("baseline_rubric_scores_by_judge")
            elif stage == "final":
                per = task.get("rubric_scores_by_judge")
            else:
                idx = int(stage.split("_", 1)[1]) - 1
                tj = task.get("turn_rubric_scores_by_judge") or []
                per = tj[idx] if 0 <= idx < len(tj) else None
            if not isinstance(per, list) or len(per) < 2:
                continue
            for trait in TRAIT_KEYS:
                xs: List[float] = []
                ok = True
                for j in per:
                    if not isinstance(j, dict) or not isinstance(j.get(trait), (int, float)):
                        ok = False
                        break
                    xs.append(float(j[trait]))
                if not ok:
                    continue
                out[stage][trait]["vals"].append(float(np.var(np.array(xs, dtype=float), ddof=1)))
                out[stage][trait]["n_items_used"] += 1

    # collapse to means
    collapsed: Dict[str, Dict[str, Any]] = {}
    for stage in stage_order:
        collapsed[stage] = {}
        for trait in TRAIT_KEYS:
            vals = out[stage][trait]["vals"]
            collapsed[stage][trait] = {
                "n_items_used": int(out[stage][trait]["n_items_used"]),
                "mean_within_item_var_trait": (float(np.mean(vals)) if vals else math.nan),
            }
    return {"stage_order": stage_order, "traits": TRAIT_KEYS, "by_stage": collapsed}


def _plot_trait_variance_lines(report: Dict[str, Any], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib not available; skip trait plot ({e})", file=sys.stderr)
        return
    stage_order = report.get("stage_order") or []
    traits = report.get("traits") or []
    by_stage = report.get("by_stage") or {}
    if not stage_order or not traits:
        return

    x = np.arange(len(stage_order), dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    cmap = plt.get_cmap("tab20")
    for i, trait in enumerate(traits):
        ys: List[float] = []
        for st in stage_order:
            block = ((by_stage.get(st) or {}).get(trait) or {})
            v = block.get("mean_within_item_var_trait")
            ys.append(float(v) if isinstance(v, (int, float)) else math.nan)
        ax.plot(x, ys, "-o", linewidth=1.4, markersize=4.6, alpha=0.9, color=cmap(i % 20), label=trait)

    ax.set_xticks(x)
    ax.set_xticklabels([_stage_label_pub(s) for s in stage_order])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Trait-level disagreement (mean within-item variance)")
    ax.set_title("Per-trait judge disagreement by stage (single model)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}", file=sys.stderr)


def _scenarios_sorted(iblock: Dict[str, Any], pat: re.Pattern[str]) -> List[str]:
    return sorted(
        (k for k in iblock if isinstance(iblock[k], dict) and pat.search(str(k))),
        key=lambda x: (int(m.group(1)), x) if (m := re.match(r"^(\d+)-", str(x))) else (999, x),
    )


def _agreement_report_for_run(
    run_obj: Dict[str, Any],
    *,
    iteration: str,
    scenario_pat: re.Pattern[str],
    expected_judge_models: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    jmodels = run_obj.get("judge_models") or []
    if not isinstance(jmodels, list) or len(jmodels) < 2:
        return None
    if expected_judge_models is not None and jmodels != expected_judge_models:
        return None

    st = run_obj.get("scenario_tasks") or {}
    iblock = st.get(iteration) or st.get(str(iteration))
    if not isinstance(iblock, dict):
        return None

    scenarios = _scenarios_sorted(iblock, scenario_pat)
    if not scenarios:
        return None

    n_j = len(jmodels)
    stage_order = ["baseline", "turn_1", "turn_2", "turn_3", "turn_4", "final"]
    matrices: Dict[str, List[np.ndarray]] = {s: [] for s in stage_order}
    skipped: Dict[str, List[str]] = {s: [] for s in stage_order}
    for sid in scenarios:
        task = iblock[sid]
        for stage in stage_order:
            m = extract_judge_matrix_for_stage(task, stage, n_judges_expected=n_j)
            if m is None:
                skipped[stage].append(str(sid))
            else:
                matrices[stage].append(m)

    out: Dict[str, Any] = {
        "judge_models": jmodels,
        "scenario_regex": scenario_pat.pattern,
        "scenarios_used": scenarios,
        "skipped_missing_per_judge_scores": skipped,
        "stage_order": stage_order,
        "stages": {},
    }
    for stage in stage_order:
        rows = matrices[stage]
        if not rows:
            out["stages"][stage] = {"label": _stage_label_pub(stage), "error": "no_complete_tasks"}
            continue
        mat = np.vstack(rows)
        out["stages"][stage] = {"label": _stage_label_pub(stage), **agreement_for_matrix(mat)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Judge agreement from per-judge turn/baseline rubrics.")
    ap.add_argument("--runs-json", type=Path, required=True)
    ap.add_argument("--run-key", default=None)
    ap.add_argument("--iteration", default="1")
    ap.add_argument(
        "--all-models",
        action="store_true",
        help="Pool scenarios across all runs in the JSON (requires consistent judge_models ordering).",
    )
    ap.add_argument(
        "--scenario-regex",
        default=r"-og$",
        help="Only include scenario_ids matching this regex (default: original wording).",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Write a stage trajectory plot (y = mean pairwise quadratic-weighted kappa).",
    )
    ap.add_argument(
        "--out-overlay-png",
        type=Path,
        default=None,
        help="For --all-models: write one line per model (x=stage).",
    )
    ap.add_argument(
        "--out-trait-lines-png",
        type=Path,
        default=None,
        help="Single-run mode: one line per trait (variance across judges) over stages.",
    )
    ap.add_argument(
        "--metric",
        choices=["kappa", "variance"],
        default="kappa",
        help="Primary trajectory metric: kappa (categorical agreement) or variance (continuous disagreement).",
    )
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--overlay-ymin", type=float, default=0.7)
    ap.add_argument("--overlay-ymax", type=float, default=1.0)
    args = ap.parse_args()

    runs = json.loads(args.runs_json.read_text(encoding="utf-8"))
    pat = re.compile(args.scenario_regex)

    if args.all_models:
        stage_order = ["baseline", "turn_1", "turn_2", "turn_3", "turn_4", "final"]
        matrices: Dict[str, List[np.ndarray]] = {s: [] for s in stage_order}
        skipped: Dict[str, int] = {s: 0 for s in stage_order}
        per_model_rows: List[Dict[str, Any]] = []

        expected_jmodels: Optional[List[str]] = None
        used_run_keys: List[str] = []

        for rk, run_obj in sorted(runs.items(), key=lambda kv: str(kv[0])):
            if not isinstance(run_obj, dict):
                continue
            report_one = _agreement_report_for_run(
                run_obj,
                iteration=str(args.iteration),
                scenario_pat=pat,
                expected_judge_models=expected_jmodels,
            )
            if report_one is None:
                continue
            if expected_jmodels is None:
                expected_jmodels = report_one["judge_models"]
            used_run_keys.append(str(rk))
            per_model_rows.append(
                {
                    "run_key": str(rk),
                    "model_name": run_obj.get("model_name") or run_obj.get("test_model") or str(rk),
                    "mean_pairwise_quadratic_weighted_cohen_kappa_by_stage": {
                        stg: (report_one.get("stages", {}).get(stg, {}) or {}).get(
                            "mean_pairwise_quadratic_weighted_cohen_kappa"
                        )
                        for stg in stage_order
                    },
                }
            )

            st = run_obj.get("scenario_tasks") or {}
            iblock = st.get(args.iteration) or st.get(str(args.iteration))
            if not isinstance(iblock, dict):
                continue
            scenarios = _scenarios_sorted(iblock, pat)
            n_j = len(expected_jmodels)
            for sid in scenarios:
                task = iblock[sid]
                for stage in stage_order:
                    m = extract_judge_matrix_for_stage(task, stage, n_judges_expected=n_j)
                    if m is None:
                        skipped[stage] += 1
                    else:
                        matrices[stage].append(m)

        if expected_jmodels is None or not used_run_keys:
            print("No compatible runs found for --all-models.", file=sys.stderr)
            return 1

        report: Dict[str, Any] = {
            "mode": "all_models_pooled",
            "title": "Judge agreement — pooled across models (OG only)",
            "runs_json": str(args.runs_json),
            "scenario_regex": args.scenario_regex,
            "iteration": str(args.iteration),
            "n_models_used": len(used_run_keys),
            "run_keys_used": used_run_keys,
            "judge_models": expected_jmodels,
            "stage_order": stage_order,
            "skipped_missing_per_judge_scores_count": skipped,
            "per_model": per_model_rows,
            "stages": {},
        }
        for stage in stage_order:
            rows = matrices[stage]
            if not rows:
                report["stages"][stage] = {"label": _stage_label_pub(stage), "error": "no_complete_tasks"}
                continue
            mat = np.vstack(rows)
            report["stages"][stage] = {"label": _stage_label_pub(stage), **agreement_for_matrix(mat)}
    else:
        rk, run_obj = _pick_run(runs, args.run_key)
        report_one = _agreement_report_for_run(
            run_obj,
            iteration=str(args.iteration),
            scenario_pat=pat,
            expected_judge_models=None,
        )
        if report_one is None:
            print("No usable tasks for this run.", file=sys.stderr)
            return 1
        report = {
            "mode": "single_run",
            "title": f"Judge agreement — {rk}",
            "run_key": rk,
            **report_one,
        }

    print(json.dumps(report, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_json}", file=sys.stderr)
        if args.out_png is None:
            args.out_png = args.out_json.with_suffix(".png")
    if args.out_png:
        _plot_metric_trajectory(
            report,
            args.out_png,
            metric=str(args.metric),
            ymin=args.ymin,
            ymax=args.ymax,
        )
    if args.all_models and args.out_overlay_png:
        _plot_agreement_overlay_models(
            report,
            args.out_overlay_png,
            ymin=float(args.overlay_ymin),
            ymax=float(args.overlay_ymax),
        )
    if (not args.all_models) and args.out_trait_lines_png:
        rk = report.get("run_key")
        run_obj = runs.get(rk) if isinstance(runs, dict) else None
        if isinstance(run_obj, dict):
            trait_report = _trait_variance_for_run(
                run_obj,
                iteration=str(args.iteration),
                scenario_pat=pat,
            )
            if trait_report:
                _plot_trait_variance_lines(trait_report, args.out_trait_lines_png)
                if args.out_json:
                    trait_json = args.out_json.with_name(args.out_json.stem + "_trait_variance.json")
                    trait_json.write_text(json.dumps(trait_report, indent=2), encoding="utf-8")
                    print(f"Wrote {trait_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
