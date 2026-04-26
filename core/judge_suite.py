# core/judge_suite.py
"""Shared helpers for multi-judge rubric and pairwise ELO aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class JudgeSuiteConfig:
    """One or more judge model API ids (same endpoint / credentials)."""

    model_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_ids:
            raise ValueError("Judge suite requires at least one model id")

    @classmethod
    def from_list(cls, model_ids: List[str]) -> "JudgeSuiteConfig":
        return cls(model_ids=tuple(model_ids))


def normalize_judge_model_list(
    judge_models: List[str] | None, fallback_single: str | None
) -> List[str]:
    """Resolve CLI-style input: explicit list or single legacy id."""
    if judge_models:
        out = [m.strip() for m in judge_models if m and str(m).strip()]
        if not out:
            raise ValueError("judge_models list is empty after stripping")
        return out
    if fallback_single and str(fallback_single).strip():
        return [fallback_single.strip()]
    raise ValueError("No judge models configured")


def aggregate_rubric_scores(score_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Mean per criterion. Requires each dict to have the same set of keys
    (strict; otherwise raises ValueError).
    """
    if not score_dicts:
        raise ValueError("No rubric score dicts to aggregate")
    keys = set(score_dicts[0].keys())
    for d in score_dicts[1:]:
        if set(d.keys()) != keys:
            raise ValueError(
                "Rubric dicts from judges must share identical keys; "
                f"got {keys} vs {set(d.keys())}"
            )
    n = len(score_dicts)
    return {k: sum(d[k] for d in score_dicts) / n for k in sorted(keys)}


def pairwise_fraction_for_logical_test(
    judge_result: Dict[str, Any],
    order_str: str,
    scenario_id: str,
) -> Dict[str, Any]:
    """
    Map one judge's pairwise JSON to stats in *logical test model* coordinates.
    Mirrors _judge_scenario_pairs_in_parallel forward/reverse and _recompute_comparison_stats.
    """
    # Lazy import avoids import cycle with pairwise_judging.
    from core.pairwise_judging import interpret_pairwise_result, compute_fraction_for_test
    from core.elo_helpers import downscale_analysis_pair

    outcome_A, plus_A, plus_B = interpret_pairwise_result(judge_result)
    a_is_test = order_str.startswith("A0493:test")

    if a_is_test:
        plus_test, plus_other = plus_A, plus_B
        outcome_test = outcome_A
    else:
        plus_test, plus_other = plus_B, plus_A
        if outcome_A == 1.0:
            outcome_test = 0.0
        elif outcome_A == 0.0:
            outcome_test = 1.0
        else:
            outcome_test = 0.5

    _frac, _diff, _diff_norm, _diff_blend = compute_fraction_for_test(
        outcome_test, plus_test, plus_other
    )
    plus_test, plus_other, diff, diff_norm, diff_blend, frac = downscale_analysis_pair(
        scenario_id, outcome_test, plus_test, plus_other
    )
    return {
        "fraction_for_test": frac,
        "outcome_for_test_model": outcome_test,
        "plus_for_test": plus_test,
        "plus_for_other": plus_other,
        "plus_diff": diff,
        "plus_diff_normalized": diff_norm,
        "plus_diff_blended": diff_blend,
    }


def outcome_from_mean_fraction(mean_frac: float) -> float:
    if mean_frac > 0.5:
        return 1.0
    if mean_frac < 0.5:
        return 0.0
    return 0.5


def aggregate_pairwise_comparison(
    judge_responses: List[Dict[str, Any]],
    order_str: str,
    scenario_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Average fraction_for_test (and other numeric fields) across judges.
    outcome_for_test_model is derived from mean fraction (>0.5 / <0.5 / else tie).
    """
    if not judge_responses:
        raise ValueError("aggregate_pairwise_comparison: empty judge_responses")

    per_judge_stats: List[Dict[str, Any]] = []
    for jr in judge_responses:
        per_judge_stats.append(
            pairwise_fraction_for_logical_test(jr, order_str, scenario_id)
        )

    n = len(per_judge_stats)
    mean_frac = sum(s["fraction_for_test"] for s in per_judge_stats) / n
    mean_outcome = outcome_from_mean_fraction(mean_frac)

    def _avg(key: str) -> float:
        return sum(s[key] for s in per_judge_stats) / n

    aggregated = {
        "fraction_for_test": mean_frac,
        "outcome_for_test_model": mean_outcome,
        "plus_for_test": int(round(_avg("plus_for_test"))),
        "plus_for_other": int(round(_avg("plus_for_other"))),
        "plus_diff": int(round(_avg("plus_diff"))),
        "plus_diff_normalized": _avg("plus_diff_normalized"),
        "plus_diff_blended": _avg("plus_diff_blended"),
    }
    return aggregated, per_judge_stats
