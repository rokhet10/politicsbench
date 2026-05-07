#!/usr/bin/env python3
"""
Trait activation count and trait-score entropy across evaluation points:

  * **blanket** — ``baseline_rubric_scores`` (blanket prompt answer)
  * **stage_1_turn0** — ``turn_rubric_scores[0]`` (rubric after first staged prompt only)
  * **stage_4_turn3** — ``turn_rubric_scores[3]`` (rubric after full 4-turn scenario)

**Activation:** count traits in ``RUBRIC_TRAIT_KEYS`` with score ≥ τ (default τ=12 and τ=14).

**Trait entropy:** Shannon entropy (nats) of p_i = s_i / sum_j s_j over the 10 trait scores
(s_i clipped at 0). High entropy ⇒ mass spread across many dimensions; low ⇒ concentrated.

Uses per-task aggregated scores when present; otherwise means ``*_scores_by_judge`` vectors.
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

from utils.constants import RUBRIC_TRAIT_KEYS


def _mean_across_judges(judge_dicts: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not judge_dicts:
        return None
    out: Dict[str, float] = {}
    for k in RUBRIC_TRAIT_KEYS:
        vals = []
        for d in judge_dicts:
            if not isinstance(d, dict):
                return None
            v = d.get(k)
            if v is None:
                return None
            vals.append(float(v))
        out[k] = sum(vals) / len(vals)
    return out


def trait_vector_for_stage(task: Dict[str, Any], stage: str) -> Optional[Dict[str, float]]:
    """
    stage: blanket | stage_1_turn0 | stage_4_turn3
    """
    if stage == "blanket":
        agg = task.get("baseline_rubric_scores")
        if isinstance(agg, dict) and all(k in agg for k in RUBRIC_TRAIT_KEYS):
            return {k: float(agg[k]) for k in RUBRIC_TRAIT_KEYS}
        by = task.get("baseline_rubric_scores_by_judge")
        if isinstance(by, list) and by:
            return _mean_across_judges(by)
        return None

    if stage == "stage_1_turn0":
        turn_idx = 0
    elif stage == "stage_4_turn3":
        turn_idx = 3
    else:
        raise ValueError(stage)

    turns = task.get("turn_rubric_scores") or []
    if len(turns) > turn_idx and isinstance(turns[turn_idx], dict):
        agg = turns[turn_idx]
        if all(k in agg for k in RUBRIC_TRAIT_KEYS):
            return {k: float(agg[k]) for k in RUBRIC_TRAIT_KEYS}

    tj = task.get("turn_rubric_scores_by_judge") or []
    if len(tj) > turn_idx and isinstance(tj[turn_idx], list) and tj[turn_idx]:
        return _mean_across_judges(tj[turn_idx])
    return None


def activation_count(scores: Dict[str, float], tau: float) -> int:
    return sum(1 for k in RUBRIC_TRAIT_KEYS if scores[k] >= tau)


def trait_entropy_nats(scores: Dict[str, float]) -> float:
    """Entropy of normalized trait magnitudes (spread across dimensions)."""
    v = np.array([max(0.0, scores[k]) for k in RUBRIC_TRAIT_KEYS], dtype=float)
    s = float(v.sum())
    if s <= 0:
        return math.nan
    p = v / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _pick_run(runs: Dict[str, Any], run_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if run_key:
        if run_key not in runs:
            raise KeyError(f"run_key {run_key!r} not in file; keys: {sorted(runs)!r}")
        return run_key, runs[run_key]
    if len(runs) == 1:
        k = next(iter(runs))
        return k, runs[k]
    raise ValueError("Multiple runs; pass --run-key")


def _summarize(vals: List[float]) -> Dict[str, float]:
    arr = np.array([x for x in vals if not math.isnan(x)], dtype=float)
    if arr.size == 0:
        return {"n": 0.0, "mean": math.nan, "std": math.nan, "median": math.nan}
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
    }


def _paired_diff(a: List[float], b: List[float]) -> List[float]:
    out = []
    for x, y in zip(a, b):
        if not math.isnan(x) and not math.isnan(y):
            out.append(x - y)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Trait activation & entropy: blanket vs stage1 vs full.")
    ap.add_argument("--runs-json", type=Path, required=True)
    ap.add_argument("--run-key", default=None)
    ap.add_argument("--iteration", default="1")
    ap.add_argument("--scenario-regex", default=r"-og$", help="Filter scenario_ids (default OG arms).")
    ap.add_argument("--tau", type=float, nargs="*", default=[12.0, 14.0], help="Activation thresholds.")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    runs = json.loads(args.runs_json.read_text(encoding="utf-8"))
    rk, run_obj = _pick_run(runs, args.run_key)
    pat = re.compile(args.scenario_regex)

    st = run_obj.get("scenario_tasks") or {}
    iblock = st.get(args.iteration) or st.get(str(args.iteration))
    if not isinstance(iblock, dict):
        print("Missing scenario_tasks for iteration", args.iteration, file=sys.stderr)
        return 1

    scenarios = sorted(
        (k for k in iblock if isinstance(iblock[k], dict) and pat.search(str(k))),
        key=lambda x: (int(m.group(1)), x) if (m := re.match(r"^(\d+)-", str(x))) else (999, x),
    )

    stages = ["blanket", "stage_1_turn0", "stage_4_turn3"]
    stage_labels = {
        "blanket": "blanket_statement",
        "stage_1_turn0": "stage_1_only_turn0_rubric",
        "stage_4_turn3": "full_4_stage_turn3_rubric",
    }

    per_rows: List[Dict[str, Any]] = []
    # collect lists for aggregates
    series: Dict[str, Dict[str, List[float]]] = {
        s: {"entropy": [], **{f"activation_ge_{int(t)}": [] for t in args.tau}} for s in stages
    }

    for sid in scenarios:
        task = iblock[sid]
        row: Dict[str, Any] = {"scenario_id": sid}
        vecs: Dict[str, Optional[Dict[str, float]]] = {}
        for stg in stages:
            vec = trait_vector_for_stage(task, stg)
            vecs[stg] = vec
            if vec is None:
                row[f"missing_{stg}"] = True
                for t in args.tau:
                    row[f"activation_ge_{int(t)}_{stg}"] = None
                row[f"trait_entropy_nats_{stg}"] = None
                continue
            row[f"missing_{stg}"] = False
            ent = trait_entropy_nats(vec)
            row[f"trait_entropy_nats_{stg}"] = ent
            series[stg]["entropy"].append(ent)
            for t in args.tau:
                c = activation_count(vec, t)
                key = f"activation_ge_{int(t)}"
                row[f"{key}_{stg}"] = c
                series[stg][key].append(float(c))

        per_rows.append(row)

    aggregates: Dict[str, Any] = {}
    for stg in stages:
        agg_st: Dict[str, Any] = {"label": stage_labels[stg]}
        agg_st["trait_entropy_nats"] = _summarize(series[stg]["entropy"])
        for t in args.tau:
            key = f"activation_ge_{int(t)}"
            agg_st[key] = _summarize(series[stg][key])
        aggregates[stg] = agg_st

    # Claim-oriented contrasts (same scenario): full vs blanket, stage1 vs blanket
    ent_full, ent_bl = [], []
    act_full: Dict[float, List[float]] = {t: [] for t in args.tau}
    act_bl: Dict[float, List[float]] = {t: [] for t in args.tau}
    ent_s1, ent_bl2 = [], []
    act_s1: Dict[float, List[float]] = {t: [] for t in args.tau}
    act_bl_s1: Dict[float, List[float]] = {t: [] for t in args.tau}

    for row in per_rows:
        vf = trait_vector_for_stage(iblock[row["scenario_id"]], "stage_4_turn3")
        vb = trait_vector_for_stage(iblock[row["scenario_id"]], "blanket")
        vs = trait_vector_for_stage(iblock[row["scenario_id"]], "stage_1_turn0")
        if vf and vb:
            ent_full.append(trait_entropy_nats(vf))
            ent_bl.append(trait_entropy_nats(vb))
            for t in args.tau:
                act_full[t].append(float(activation_count(vf, t)))
                act_bl[t].append(float(activation_count(vb, t)))
        if vs and vb:
            ent_s1.append(trait_entropy_nats(vs))
            ent_bl2.append(trait_entropy_nats(vb))
            for t in args.tau:
                act_s1[t].append(float(activation_count(vs, t)))
                act_bl_s1[t].append(float(activation_count(vb, t)))

    contrasts: Dict[str, Any] = {
        "full_minus_blanket": {
            "trait_entropy_nats_paired_mean_diff": float(
                np.mean(_paired_diff(ent_full, ent_bl))
            )
            if ent_full
            else math.nan,
            "trait_entropy_nats_paired_summary": _summarize(_paired_diff(ent_full, ent_bl)),
        },
        "stage1_minus_blanket": {
            "trait_entropy_nats_paired_mean_diff": float(
                np.mean(_paired_diff(ent_s1, ent_bl2))
            )
            if ent_s1
            else math.nan,
            "trait_entropy_nats_paired_summary": _summarize(_paired_diff(ent_s1, ent_bl2)),
        },
    }
    for t in args.tau:
        tk = f"activation_ge_{int(t)}"
        contrasts["full_minus_blanket"][tk] = {
            "paired_mean_diff": float(np.mean(_paired_diff(act_full[t], act_bl[t])))
            if act_full[t]
            else math.nan,
            "paired_summary": _summarize(_paired_diff(act_full[t], act_bl[t])),
        }
        contrasts["stage1_minus_blanket"][tk] = {
            "paired_mean_diff": float(np.mean(_paired_diff(act_s1[t], act_bl_s1[t])))
            if act_s1[t]
            else math.nan,
            "paired_summary": _summarize(_paired_diff(act_s1[t], act_bl_s1[t])),
        }

    # Optional related-samples t-test (requires scipy)
    try:
        from scipy import stats as scipy_stats

        if len(ent_full) == len(ent_bl) and len(ent_full) >= 2:
            contrasts["full_minus_blanket"]["trait_entropy_nats_paired_ttest_rel"] = {
                "statistic": float(
                    scipy_stats.ttest_rel(ent_full, ent_bl, nan_policy="omit").statistic
                ),
                "pvalue": float(
                    scipy_stats.ttest_rel(ent_full, ent_bl, nan_policy="omit").pvalue
                ),
            }
        for t in args.tau:
            a, b = act_full[t], act_bl[t]
            if len(a) == len(b) and len(a) >= 2:
                tr = scipy_stats.ttest_rel(a, b, nan_policy="omit")
                contrasts["full_minus_blanket"][f"activation_ge_{int(t)}_paired_ttest_rel"] = {
                    "statistic": float(tr.statistic),
                    "pvalue": float(tr.pvalue),
                }
    except Exception:
        pass

    report: Dict[str, Any] = {
        "run_key": rk,
        "runs_file": str(args.runs_json),
        "iteration": args.iteration,
        "scenario_regex": args.scenario_regex,
        "trait_keys": list(RUBRIC_TRAIT_KEYS),
        "activation_definition": "count traits with mean judge score >= tau",
        "entropy_definition": "Shannon entropy (nats) of normalized trait scores s_i/sum(s)",
        "tau_values": list(args.tau),
        "per_scenario": per_rows,
        "aggregate_by_stage": aggregates,
        "paired_contrasts_same_scenario": contrasts,
        "interpretation_notes": [
            "Positive full_minus_blanket mean activation or entropy supports broader trait engagement after the full scenario vs blanket.",
            "stage_1_turn0 isolates the rubric after only the first in-character prompt.",
        ],
    }

    print(json.dumps(report, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
