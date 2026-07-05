# core/judge_suite.py
"""Shared helpers for multi-judge rubric aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class JudgeSuiteConfig:
    """One or more judge model API ids (same endpoint / credentials)."""

    model_ids: tuple[str, ...]

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
