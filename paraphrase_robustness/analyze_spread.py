#!/usr/bin/env python3
"""
Cross-variant spread: dispersion of rubric scores across variants within each base.

Uses weighted scalars from optional baseline_rubric_scores (blanket question), turn 0, and turn 3.
Manifest pairs may be wording-only (wa/wb) or archived framing (pro/anti); math is the same.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RUBRIC_CRITERION_WEIGHTS


def weighted_scalar(turn: Optional[Dict[str, float]], weights: Dict[str, float]) -> float:
    if not turn:
        return math.nan
    s = 0.0
    for k, w in weights.items():
        v = turn.get(k)
        if v is None:
            return math.nan
        s += float(v) * float(w)
    return s


def load_manifest(path: Path) -> Tuple[int, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    variants = data.get("variants") or []
    ver = int(data.get("version", 1))
    return ver, variants


def pick_run_payload(
    runs: Dict[str, Any],
    run_key: Optional[str],
    *,
    use_latest: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(runs, dict):
        raise TypeError("runs JSON root must be an object mapping run_key -> run data")
    if not runs:
        raise ValueError("runs JSON is empty (no top-level keys)")

    if run_key:
        if run_key not in runs:
            raise KeyError(
                f"run_key {run_key!r} not in runs file. Available: {sorted(runs)!r}"
            )
        return run_key, runs[run_key]

    if use_latest:
        def start_stamp(item: Tuple[str, Any]) -> str:
            _, obj = item
            if not isinstance(obj, dict):
                return ""
            return str(obj.get("start_time") or "")

        rk, run_obj = max(runs.items(), key=start_stamp)
        return rk, run_obj

    if len(runs) == 1:
        k = next(iter(runs))
        return k, runs[k]

    keys = ", ".join(repr(k) for k in sorted(runs.keys()))
    raise ValueError(
        f"runs JSON has {len(runs)} top-level keys; pass --run-key KEY or --latest. "
        f"Available keys: {keys}"
    )


def extract_turn_scores(
    run_obj: Dict[str, Any], scenario_id: str, iteration: str = "1"
) -> Tuple[Optional[List[Optional[Dict[str, float]]]], Optional[Dict[str, float]], str]:
    st = run_obj.get("scenario_tasks") or {}
    iter_block = st.get(iteration) or st.get(str(iteration))
    if not isinstance(iter_block, dict):
        return None, None, "missing_iteration"
    task = iter_block.get(str(scenario_id)) or iter_block.get(scenario_id)
    if not isinstance(task, dict):
        return None, None, "missing_task"
    turns = task.get("turn_rubric_scores")
    if not isinstance(turns, list):
        return None, None, "no_turn_scores"
    baseline = task.get("baseline_rubric_scores")
    if baseline is not None and not isinstance(baseline, dict):
        baseline = None
    return turns, baseline, task.get("status") or ""


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Linear interpolation percentile, p in [0,100]."""
    if not sorted_vals:
        return math.nan
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def spread_stats(values: List[float]) -> Dict[str, float]:
    arr = [v for v in values if not math.isnan(v)]
    n = len(arr)
    if n < 2:
        return {"n": float(n), "std": math.nan, "iqr": math.nan}
    sarr = sorted(arr)
    q25 = _percentile(sarr, 25)
    q75 = _percentile(sarr, 75)
    return {
        "n": float(n),
        "std": float(statistics.stdev(arr)),
        "iqr": float(q75 - q25),
    }


def _wilcoxon_greater_zero(deltas: List[float]) -> Optional[Dict[str, float]]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    clean = [d for d in deltas if not math.isnan(d)]
    if len(clean) < 2:
        return None
    try:
        w = wilcoxon(clean, alternative="greater")
        return {"statistic": float(w.statistic), "pvalue": float(w.pvalue)}
    except ValueError:
        return None


def summarize_run(
    runs_path: Path,
    manifest_path: Path,
    *,
    run_key: Optional[str],
    iteration: str,
    kind_filter: Optional[str],
    use_latest: bool = False,
) -> Dict[str, Any]:
    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    rk, run_obj = pick_run_payload(runs, run_key, use_latest=use_latest)
    _, variants = load_manifest(manifest_path)

    if kind_filter:
        variants = [v for v in variants if v.get("kind") == kind_filter]

    by_base: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for v in variants:
        by_base[v["base_id"]].append(v)

    per_base: List[Dict[str, Any]] = []
    deltas_std: List[float] = []
    deltas_std_b0: List[float] = []

    def _base_sort_key(kv: Tuple[str, Any]) -> Tuple[int, Any]:
        k = kv[0]
        if str(k).isdigit():
            return (0, int(k))
        return (1, str(k))

    for base_id, vlist in sorted(by_base.items(), key=_base_sort_key):
        s0: List[float] = []
        s3: List[float] = []
        sb: List[float] = []
        variant_detail: List[Dict[str, Any]] = []
        for v in vlist:
            sid = v["scenario_id"]
            turns, baseline_scores, st = extract_turn_scores(
                run_obj, sid, iteration=iteration
            )
            if turns is None or len(turns) < 4:
                variant_detail.append(
                    {
                        "scenario_id": sid,
                        "error": st,
                        "turns_len": len(turns or []),
                    }
                )
                continue
            t0 = turns[0] if len(turns) > 0 else None
            t3 = turns[3] if len(turns) > 3 else None
            z0 = weighted_scalar(t0, RUBRIC_CRITERION_WEIGHTS)
            z3 = weighted_scalar(t3, RUBRIC_CRITERION_WEIGHTS)
            zb = (
                weighted_scalar(baseline_scores, RUBRIC_CRITERION_WEIGHTS)
                if baseline_scores
                else math.nan
            )
            if not math.isnan(z0):
                s0.append(z0)
            if not math.isnan(z3):
                s3.append(z3)
            if not math.isnan(zb):
                sb.append(zb)
            variant_detail.append(
                {
                    "scenario_id": sid,
                    "framing": v.get("framing"),
                    "tone": v.get("tone"),
                    "scalar_baseline": zb,
                    "scalar_turn0": z0,
                    "scalar_turn3": z3,
                    "status": st,
                }
            )

        sp0 = spread_stats(s0)
        sp3 = spread_stats(s3)
        spb = spread_stats(sb)
        d_std = (
            sp0["std"] - sp3["std"]
            if not math.isnan(sp0["std"]) and not math.isnan(sp3["std"])
            else math.nan
        )
        if not math.isnan(d_std):
            deltas_std.append(d_std)
        d_std_b0 = (
            sp0["std"] - spb["std"]
            if not math.isnan(sp0["std"]) and not math.isnan(spb["std"])
            else math.nan
        )
        if not math.isnan(d_std_b0):
            deltas_std_b0.append(d_std_b0)

        per_base.append(
            {
                "base_id": base_id,
                "spread_baseline_std": spb["std"],
                "spread_baseline_iqr": spb["iqr"],
                "spread_turn0_std": sp0["std"],
                "spread_turn3_std": sp3["std"],
                "spread_turn0_iqr": sp0["iqr"],
                "spread_turn3_iqr": sp3["iqr"],
                "delta_std": d_std,
                "delta_std_turn0_minus_baseline": d_std_b0,
                "variants": variant_detail,
            }
        )

    clean_d = [d for d in deltas_std if not math.isnan(d)]
    clean_b0 = [d for d in deltas_std_b0 if not math.isnan(d)]
    summary: Dict[str, Any] = {
        "run_key": rk,
        "runs_file": str(runs_path),
        "manifest": str(manifest_path),
        "kind_filter": kind_filter,
        "n_bases": len(per_base),
        "mean_delta_std": float(statistics.mean(clean_d)) if clean_d else math.nan,
        "mean_delta_std_turn0_minus_baseline": (
            float(statistics.mean(clean_b0)) if clean_b0 else math.nan
        ),
        "per_base": per_base,
        "wilcoxon_delta_std_greater_zero": _wilcoxon_greater_zero(clean_d),
        "wilcoxon_delta_std_turn0_minus_baseline_greater_zero": _wilcoxon_greater_zero(
            clean_b0
        ),
    }
    return summary


def plot_paired_spread(per_base: List[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    xsb: List[float] = []
    xs0: List[float] = []
    xs3: List[float] = []
    labels: List[str] = []
    for row in per_base:
        b = row.get("spread_baseline_std", math.nan)
        a = row["spread_turn0_std"]
        c = row["spread_turn3_std"]
        if math.isnan(a) or math.isnan(c):
            continue
        xsb.append(b)
        xs0.append(a)
        xs3.append(c)
        labels.append(row["base_id"])

    if not xs0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    idx = list(range(len(xs0)))
    off_b, off0, off3 = -0.15, 0.0, 0.15
    has_baseline = any(not math.isnan(v) for v in xsb)
    if has_baseline:
        ax.scatter(
            [i + off_b for i in idx],
            xsb,
            label="blanket baseline",
            color="#9b59b6",
            s=40,
            zorder=3,
        )
    ax.scatter(
        [i + off0 for i in idx],
        xs0,
        label="turn 0 (Stage 1)",
        color="#2c3e50",
        s=42,
        zorder=3,
    )
    ax.scatter(
        [i + off3 for i in idx],
        xs3,
        label="turn 3 (Stage 4)",
        color="#3498db",
        s=42,
        zorder=3,
    )
    for i in idx:
        ys = [v for v in (xsb[i], xs0[i], xs3[i]) if not math.isnan(v)]
        if len(ys) >= 2:
            ax.plot([i, i], [min(ys), max(ys)], color="#95a5a6", linewidth=1.0, zorder=1)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cross-variant std (weighted scalar)")
    ax.set_title("Cross-variant spread: blanket baseline vs Stage 1 vs Stage 4")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stage_summary(per_base: List[Dict[str, Any]], out_path: Path) -> None:
    """
    Stage-on-x summary: distribution of per-base spread at each stage.

    Each base contributes one value per stage (std across variants), so this visualizes
    how cross-variant dispersion changes with stage, aggregated over bases.
    """
    import matplotlib.pyplot as plt

    sb: List[float] = []
    s0: List[float] = []
    s3: List[float] = []
    for row in per_base:
        b = row.get("spread_baseline_std", math.nan)
        a = row.get("spread_turn0_std", math.nan)
        c = row.get("spread_turn3_std", math.nan)
        if not math.isnan(float(b)):
            sb.append(float(b))
        if not math.isnan(float(a)):
            s0.append(float(a))
        if not math.isnan(float(c)):
            s3.append(float(c))

    data = [sb, s0, s3]
    labels = ["Baseline", "Stage 1 (turn 0)", "Stage 4 (turn 3)"]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#2c3e50", "linewidth": 2},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    colors = ["#9b59b6", "#2c3e50", "#3498db"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)

    # Jitter points for transparency about sample size.
    rng = np.random.default_rng(0)
    for i, xs in enumerate(data, start=1):
        if not xs:
            continue
        jitter = rng.normal(0, 0.04, size=len(xs))
        ax.scatter(
            np.full(len(xs), i) + jitter,
            xs,
            s=22,
            alpha=0.55,
            color=colors[i - 1],
            edgecolor="none",
            zorder=3,
        )

    ax.set_ylabel("Cross-variant std (weighted scalar)")
    ax.set_title("Cross-variant spread by stage (across bases)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paraphrase spread analysis (turn 0 vs 3).")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--runs-json", type=Path, required=True)
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--run-key", default=None, help="Which run to analyze when the JSON has multiple top-level keys.")
    pick.add_argument(
        "--latest",
        action="store_true",
        help="Pick the run with the lexicographically greatest start_time (for multi-run files).",
    )
    ap.add_argument("--iteration", default="1")
    ap.add_argument(
        "--kind",
        default="main",
        help='Filter manifest by kind (e.g. "main", "control"). Use "all" for no filter.',
    )
    ap.add_argument("--out-dir", type=Path, default=Path("paraphrase_robustness/results"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    kind_filter = None if args.kind == "all" else args.kind
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_run(
        args.runs_json,
        args.manifest,
        run_key=args.run_key,
        iteration=args.iteration,
        kind_filter=kind_filter,
        use_latest=bool(args.latest),
    )

    stem = args.runs_json.stem + ("_" + kind_filter if kind_filter else "_all")
    json_out = args.out_dir / f"spread_summary_{stem}.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "per_base"}, indent=2))
    print(f"Wrote {json_out}")

    if not args.no_plot:
        plot_path = args.out_dir / f"spread_paired_{stem}.png"
        plot_paired_spread(summary["per_base"], plot_path)
        print(f"Wrote {plot_path}")

        stage_path = args.out_dir / f"spread_by_stage_{stem}.png"
        plot_stage_summary(summary["per_base"], stage_path)
        print(f"Wrote {stage_path}")


if __name__ == "__main__":
    main()
