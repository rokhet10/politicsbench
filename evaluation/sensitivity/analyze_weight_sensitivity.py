#!/usr/bin/env python3
"""
Weight sensitivity analysis for the composite ideology score.

Three experiments on saved rubric scores (no API calls):

  A. Equal weights — every trait weight set to +1 (or sign-preserving unit magnitude).
  B. Random perturbation — each baseline weight perturbed by ±pct (default 20%), repeated n times.
  C. Leave-one-trait-out (LOTO) — drop each trait and recompute scores/rankings.

Uses the same per-task normalization as ``core/benchmark.py``:
  centered trait (score − 10) × weight, summed per task, divided by
  Σ |w| × 10, scaled to [-100, +100]; model score = mean over tasks.

Typical usage (8-model paraphrase-final sweep):

  python3 evaluation/sensitivity/analyze_weight_sensitivity.py \\
    --runs-json eqbench_runs_final.json \\
    --run-key-regex '.*-paraphrase-final$' \\
    --scenario-regex '.*-og$'
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RUBRIC_CRITERION_WEIGHTS, RUBRIC_TRAIT_KEYS  # noqa: E402

MAX_SCORE = 20
MIDPOINT = MAX_SCORE / 2


def _display_name(run_obj: Dict[str, Any], run_key: str) -> str:
    for field in ("model_name", "test_model", "logical_name"):
        v = run_obj.get(field)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return run_key


def _short_label(name: str) -> str:
    for suffix in ("-paraphrase-final", "-paraphrase-pilot"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.split("/")[-1]


def score_task(rubric_scores: Dict[str, Any], weights: Dict[str, float]) -> Optional[float]:
    """Normalized ideology score for one task (-100 conservative … +100 liberal)."""
    task_weighted_sum = 0.0
    total_weight_magnitude = 0.0
    for metric, raw_score in rubric_scores.items():
        if metric not in weights or not isinstance(raw_score, (int, float)):
            continue
        weight = weights[metric]
        centered = float(raw_score) - MIDPOINT
        task_weighted_sum += centered * weight
        total_weight_magnitude += abs(weight) * MIDPOINT
    if total_weight_magnitude == 0:
        return None
    normalized = (task_weighted_sum / total_weight_magnitude) * 100
    return max(-100.0, min(100.0, normalized))



def model_score_from_tasks(
    task_rubric_scores: List[Dict[str, Any]],
    weights: Dict[str, float],
) -> Optional[float]:
    task_scores: List[float] = []
    for rs in task_rubric_scores:
        s = score_task(rs, weights)
        if s is not None:
            task_scores.append(s)
    if not task_scores:
        return None
    return round(statistics.mean(task_scores), 4)


def load_model_task_data(
    runs: Dict[str, Any],
    run_key_regex: re.Pattern,
    scenario_pat: re.Pattern,
    iteration: str,
) -> Dict[str, Dict[str, Any]]:
    """Return {label: {run_key, rubric_scores: [dict, ...]}}."""
    models: Dict[str, Dict[str, Any]] = {}
    for run_key, run_obj in sorted(runs.items()):
        if not isinstance(run_obj, dict) or not run_key_regex.search(run_key):
            continue
        st = run_obj.get("scenario_tasks") or {}
        iblock = st.get(iteration) or st.get(str(iteration))
        if not isinstance(iblock, dict):
            continue
        rubrics: List[Dict[str, Any]] = []
        for sid, task in iblock.items():
            if not scenario_pat.search(str(sid)):
                continue
            if not isinstance(task, dict) or task.get("status") != "rubric_scored":
                continue
            rs = task.get("rubric_scores")
            if isinstance(rs, dict) and rs:
                rubrics.append(rs)
        if not rubrics:
            continue
        label = _short_label(_display_name(run_obj, run_key))
        # Prefer latest run if duplicate short labels (e.g. two Llama entries)
        if label in models:
            label = f"{label} ({run_key[:8]})"
        models[label] = {"run_key": run_key, "rubric_scores": rubrics, "n_tasks": len(rubrics)}
    return models


def rank_models(scores: Dict[str, float], higher_is_more_liberal: bool = True) -> Dict[str, int]:
    """Rank 1 = most liberal (highest score) by default."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=higher_is_more_liberal)
    return {name: i + 1 for i, (name, _) in enumerate(ordered)}


def spearman_rho(ranks_a: Dict[str, int], ranks_b: Dict[str, int]) -> float:
    names = sorted(set(ranks_a) & set(ranks_b))
    if len(names) < 2:
        return math.nan
    a = np.array([ranks_a[n] for n in names], dtype=float)
    b = np.array([ranks_b[n] for n in names], dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.allclose(a, b) else math.nan
    return float(np.corrcoef(a, b)[0, 1])


def equal_weights(mode: str) -> Dict[str, float]:
    if mode == "all_one":
        return {k: 1.0 for k in RUBRIC_TRAIT_KEYS}
    # sign_preserving: unit magnitude, keep liberal+/conservative− sign
    return {k: (1.0 if RUBRIC_CRITERION_WEIGHTS[k] >= 0 else -1.0) for k in RUBRIC_TRAIT_KEYS}


def perturb_weights(
    base: Dict[str, float],
    rng: np.random.Generator,
    pct: float,
) -> Dict[str, float]:
    lo, hi = 1.0 - pct, 1.0 + pct
    return {k: base[k] * float(rng.uniform(lo, hi)) for k in base}


def loto_weights(drop_trait: str) -> Dict[str, float]:
    return {k: v for k, v in RUBRIC_CRITERION_WEIGHTS.items() if k != drop_trait}


def run_experiments(
    models: Dict[str, Dict[str, Any]],
    n_perturbations: int,
    perturb_pct: float,
    equal_mode: str,
    seed: int,
) -> Dict[str, Any]:
    baseline_scores: Dict[str, float] = {}
    for label, payload in models.items():
        s = model_score_from_tasks(payload["rubric_scores"], RUBRIC_CRITERION_WEIGHTS)
        if s is not None:
            baseline_scores[label] = s
    baseline_ranks = rank_models(baseline_scores)

    # --- A. Equal weights ---
    eq_w = equal_weights(equal_mode)
    equal_scores = {
        label: model_score_from_tasks(payload["rubric_scores"], eq_w)
        for label, payload in models.items()
    }
    equal_scores = {k: v for k, v in equal_scores.items() if v is not None}
    equal_ranks = rank_models(equal_scores)

    # --- B. Random perturbation ---
    rng = np.random.default_rng(seed)
    perturb_score_samples: Dict[str, List[float]] = {k: [] for k in baseline_scores}
    perturb_rank_samples: Dict[str, List[int]] = {k: [] for k in baseline_scores}
    ranking_unchanged = 0
    spearman_list: List[float] = []

    for _ in range(n_perturbations):
        w = perturb_weights(RUBRIC_CRITERION_WEIGHTS, rng, perturb_pct)
        scores = {
            label: model_score_from_tasks(payload["rubric_scores"], w)
            for label, payload in models.items()
        }
        scores = {k: v for k, v in scores.items() if v is not None}
        ranks = rank_models(scores)
        if ranks == baseline_ranks:
            ranking_unchanged += 1
        spearman_list.append(spearman_rho(baseline_ranks, ranks))
        for label, sc in scores.items():
            perturb_score_samples[label].append(sc)
            perturb_rank_samples[label].append(ranks[label])

    perturb_summary: Dict[str, Any] = {
        "n_replicates": n_perturbations,
        "perturb_pct": perturb_pct,
        "fraction_rankings_identical_to_baseline": ranking_unchanged / n_perturbations,
        "spearman_rho_mean": float(np.nanmean(spearman_list)),
        "spearman_rho_min": float(np.nanmin(spearman_list)),
        "per_model": {},
    }
    for label in sorted(baseline_scores):
        sc_arr = np.array(perturb_score_samples[label])
        rk_arr = np.array(perturb_rank_samples[label])
        perturb_summary["per_model"][label] = {
            "baseline_score": baseline_scores[label],
            "baseline_rank": baseline_ranks[label],
            "score_mean": float(sc_arr.mean()),
            "score_std": float(sc_arr.std(ddof=0)),
            "score_min": float(sc_arr.min()),
            "score_max": float(sc_arr.max()),
            "rank_mean": float(rk_arr.mean()),
            "rank_std": float(rk_arr.std(ddof=0)),
            "rank_min": int(rk_arr.min()),
            "rank_max": int(rk_arr.max()),
            "fraction_at_baseline_rank": float(np.mean(rk_arr == baseline_ranks[label])),
        }

    # --- C. Leave-one-trait-out ---
    loto: Dict[str, Any] = {}
    for trait in RUBRIC_TRAIT_KEYS:
        w = loto_weights(trait)
        scores = {
            label: model_score_from_tasks(payload["rubric_scores"], w)
            for label, payload in models.items()
        }
        scores = {k: v for k, v in scores.items() if v is not None}
        ranks = rank_models(scores)
        loto[trait] = {
            "scores": scores,
            "ranks": ranks,
            "spearman_rho_vs_baseline": spearman_rho(baseline_ranks, ranks),
            "rank_changes": {
                label: ranks[label] - baseline_ranks[label]
                for label in baseline_scores
            },
            "score_deltas": {
                label: round(scores[label] - baseline_scores[label], 4)
                for label in baseline_scores
            },
        }

    return {
        "baseline": {
            "weights": dict(RUBRIC_CRITERION_WEIGHTS),
            "scores": baseline_scores,
            "ranks": baseline_ranks,
        },
        "equal_weights": {
            "mode": equal_mode,
            "weights": eq_w,
            "scores": equal_scores,
            "ranks": equal_ranks,
            "spearman_rho_vs_baseline": spearman_rho(baseline_ranks, equal_ranks),
            "rank_changes": {
                label: equal_ranks[label] - baseline_ranks[label]
                for label in baseline_scores
            },
        },
        "random_perturbation": perturb_summary,
        "leave_one_trait_out": loto,
    }


def _print_table(title: str, scores: Dict[str, float], ranks: Dict[str, int]) -> None:
    print(f"\n{title}")
    print("-" * 56)
    print(f"{'Model':<28} {'Score':>10} {'Rank':>6}")
    for label in sorted(scores, key=lambda x: ranks[x]):
        print(f"{label:<28} {scores[label]:>10.2f} {ranks[label]:>6}")


def print_report(result: Dict[str, Any]) -> None:
    base = result["baseline"]
    _print_table("Baseline (paper weights)", base["scores"], base["ranks"])

    eq = result["equal_weights"]
    mode = eq["mode"]
    _print_table(
        f"A. Equal weights ({mode}) — ρ={eq['spearman_rho_vs_baseline']:.3f}",
        eq["scores"],
        eq["ranks"],
    )

    pert = result["random_perturbation"]
    print(
        f"\nB. Random ±{pert['perturb_pct']*100:.0f}% perturbation "
        f"({pert['n_replicates']} replicates)"
    )
    print(f"   Rankings identical to baseline: {pert['fraction_rankings_identical_to_baseline']*100:.1f}%")
    print(f"   Spearman ρ (baseline vs perturbed): mean={pert['spearman_rho_mean']:.3f}, min={pert['spearman_rho_min']:.3f}")
    print(f"   {'Model':<28} {'Score μ±σ':>18} {'Rank μ±σ':>14} {'@base rank':>12}")
    for label in sorted(pert["per_model"]):
        pm = pert["per_model"][label]
        print(
            f"   {label:<28} "
            f"{pm['score_mean']:>7.2f}±{pm['score_std']:<7.2f} "
            f"{pm['rank_mean']:>5.1f}±{pm['rank_std']:<5.1f} "
            f"{pm['fraction_at_baseline_rank']*100:>10.1f}%"
        )

    print("\nC. Leave-one-trait-out (ρ vs baseline, max |Δrank|)")
    trait_short = {
        "tradition_orientation": "tradition",
        "progress_orientation": "progress",
        "authority_deference": "authority",
        "egalitarianism": "egalitarianism",
        "risk_aversion": "risk_aversion",
        "openness_to_difference": "openness",
        "individual_responsibility": "indiv_resp",
        "collective_responsibility": "coll_resp",
        "moral_certainty": "moral_certainty",
        "nuanced_pragmatism": "nuanced_prag",
    }
    for trait in RUBRIC_TRAIT_KEYS:
        block = result["leave_one_trait_out"][trait]
        max_dr = max(abs(v) for v in block["rank_changes"].values())
        print(
            f"   drop {trait_short.get(trait, trait):<18} "
            f"ρ={block['spearman_rho_vs_baseline']:.3f}  max|Δrank|={max_dr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-json",
        type=Path,
        default=REPO_ROOT / "eqbench_runs_final.json",
        help="Saved runs file (eqbench3 output).",
    )
    parser.add_argument(
        "--run-key-regex",
        default=r".*-paraphrase-final$",
        help="Which runs to include (default: paraphrase-final 8-model sweep).",
    )
    parser.add_argument(
        "--scenario-regex",
        default=r".*-og$",
        help="Scenario filter (default: OG wording only). Use '.*' for all variants.",
    )
    parser.add_argument("--iteration", default="1")
    parser.add_argument(
        "--n-perturbations",
        type=int,
        default=1000,
        help="Random weight perturbation replicates (default: 1000).",
    )
    parser.add_argument(
        "--perturb-pct",
        type=float,
        default=0.20,
        help="Perturb each weight by ± this fraction (default: 0.20).",
    )
    parser.add_argument(
        "--equal-weights-mode",
        choices=("sign_preserving", "all_one"),
        default="all_one",
        help="Equal-weight experiment: all +1, or sign-preserving ±1 (default: all_one).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "evaluation/sensitivity/results/weight_sensitivity.json",
    )
    args = parser.parse_args()

    if not args.runs_json.is_file():
        print(f"Runs file not found: {args.runs_json}", file=sys.stderr)
        return 1

    runs = json.loads(args.runs_json.read_text(encoding="utf-8"))
    run_pat = re.compile(args.run_key_regex)
    scenario_pat = re.compile(args.scenario_regex)

    models = load_model_task_data(runs, run_pat, scenario_pat, args.iteration)
    if len(models) < 2:
        print(
            f"Need ≥2 models; found {len(models)} matching "
            f"run-key-regex={args.run_key_regex!r} scenario-regex={args.scenario_regex!r}",
            file=sys.stderr,
        )
        return 1

    result = run_experiments(
        models,
        n_perturbations=args.n_perturbations,
        perturb_pct=args.perturb_pct,
        equal_mode=args.equal_weights_mode,
        seed=args.seed,
    )

    payload = {
        "config": {
            "runs_json": str(args.runs_json),
            "run_key_regex": args.run_key_regex,
            "scenario_regex": args.scenario_regex,
            "iteration": args.iteration,
            "n_perturbations": args.n_perturbations,
            "perturb_pct": args.perturb_pct,
            "equal_weights_mode": args.equal_weights_mode,
            "seed": args.seed,
            "models": {
                label: {"run_key": m["run_key"], "n_tasks": m["n_tasks"]}
                for label, m in models.items()
            },
        },
        **result,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
