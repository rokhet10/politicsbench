#!/usr/bin/env python3
"""
Summarize and plot commitment_score trajectories from a runs JSON (eqbench3_runs.json or any
path). Stages: optional blanket **baseline**, each **turn** (after staged prompts), then **final**
(debrief or analysis rubric_scores).

Requires tasks judged with the commitment rubric (dicts containing ``commitment_score``). Detects
runs via ``scoring_mode == \"commitment\"`` or by finding ``commitment_score`` on rubric fields.

Writes:
  - ``commitment_summary_<run_key>[_variant-og].json`` — per-stage stats, per-task trajectories, deltas
  - ``commitment_trajectory_mean_<run_key>[_variant-og].png`` — mean ±95% CI across tasks
  - ``commitment_trajectory_per_scenario_<run_key>[_variant-og].png`` — one line per scenario (up to ``--max-lines``)

  With ``--all-models`` (use with ``--only-variant og`` for one wording per base):
  - ``commitment_summary_all_models[_variant-og].json`` — per-model stage means + aggregate
  - ``commitment_trajectory_mean_all_models[_variant-og].png`` — mean trajectory across models, error bars = spread across models (SEM / SD / 95% CI)
  - ``commitment_trajectory_overlay_all_models[_variant-og].png`` — every model’s mean trajectory on one axes

Use ``--only-variant og`` to aggregate only ``*-og`` prompts (one wording per base). Default includes all wordings.
The per-scenario plot's ``n`` in the title is ``min(--max-lines, n_tasks)``, not the sample size for means.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.file_io import load_json_file  # noqa: E402


def _score_from_dict(d: Any) -> Optional[float]:
    if not isinstance(d, dict):
        return None
    v = d.get("commitment_score")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _score_from_judges(by: Any) -> Optional[float]:
    if not isinstance(by, list) or not by:
        return None
    vals: List[float] = []
    for d in by:
        if isinstance(d, dict) and isinstance(d.get("commitment_score"), (int, float)):
            vals.append(float(d["commitment_score"]))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _baseline_commitment(task: Dict[str, Any]) -> Optional[float]:
    s = _score_from_dict(task.get("baseline_commitment_scores"))
    if s is not None:
        return s
    by = _score_from_judges(task.get("baseline_commitment_scores_by_judge"))
    if by is not None:
        return by
    s = _score_from_dict(task.get("baseline_rubric_scores"))
    if s is not None:
        return s
    return _score_from_judges(task.get("baseline_rubric_scores_by_judge"))


def _turn_commitment(task: Dict[str, Any], turn_index: int) -> Optional[float]:
    c_turns = task.get("turn_commitment_scores") or []
    if turn_index < len(c_turns):
        s = _score_from_dict(c_turns[turn_index])
        if s is not None:
            return s
    ctj = task.get("turn_commitment_scores_by_judge") or []
    if turn_index < len(ctj):
        by = _score_from_judges(ctj[turn_index])
        if by is not None:
            return by
    turns = task.get("turn_rubric_scores") or []
    if turn_index < len(turns):
        s = _score_from_dict(turns[turn_index])
        if s is not None:
            return s
    tj = task.get("turn_rubric_scores_by_judge") or []
    if turn_index < len(tj):
        return _score_from_judges(tj[turn_index])
    return None


def _final_commitment(task: Dict[str, Any]) -> Optional[float]:
    s = _score_from_dict(task.get("debrief_commitment_scores"))
    if s is not None:
        return s
    by = _score_from_judges(task.get("debrief_commitment_scores_by_judge"))
    if by is not None:
        return by
    s = _score_from_dict(task.get("rubric_scores"))
    if s is not None:
        return s
    return _score_from_judges(task.get("rubric_scores_by_judge"))


def _task_has_commitment_rubric(task: Dict[str, Any]) -> bool:
    if _baseline_commitment(task) is not None:
        return True
    n_turns = max(
        len(task.get("turn_commitment_scores") or []),
        len(task.get("turn_rubric_scores") or []),
    )
    for i in range(n_turns):
        if _turn_commitment(task, i) is not None:
            return True
    if _final_commitment(task) is not None:
        return True
    return False


def _ordered_stage_names(max_turns: int, has_baseline: bool) -> List[str]:
    out: List[str] = []
    if has_baseline:
        out.append("baseline")
    for i in range(max_turns):
        out.append(f"turn_{i + 1}")
    out.append("final")
    return out


def _extract_row(
    task: Dict[str, Any], stage_names: List[str], has_baseline: bool, n_turns: int
) -> Dict[str, Optional[float]]:
    row: Dict[str, Optional[float]] = {s: None for s in stage_names}
    if has_baseline:
        row["baseline"] = _baseline_commitment(task)
    for i in range(n_turns):
        key = f"turn_{i + 1}"
        if key in row:
            row[key] = _turn_commitment(task, i)
    row["final"] = _final_commitment(task)
    return row


def _pick_latest_run_key(runs: Dict[str, Any]) -> Optional[str]:
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
    return best_k


def _summarize(vals: List[float]) -> Dict[str, float]:
    arr = np.array([x for x in vals if not math.isnan(x)], dtype=float)
    if arr.size == 0:
        return {"n": 0.0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _ci95(vals: List[float]) -> Tuple[float, float, float]:
    """Return mean, low, high (normal approx) for plotting."""
    arr = np.array([x for x in vals if not math.isnan(x)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    m = float(arr.mean())
    if arr.size < 2:
        return m, m, m
    sem = float(arr.std(ddof=1) / math.sqrt(arr.size))
    half = 1.96 * sem
    return m, m - half, m + half


def _variant_suffix(variant: Optional[str]) -> str:
    if not variant:
        return ""
    return f"_variant-{variant}"


def _legend_display_model_name(name: str) -> str:
    """Shorter legend labels (strip common run suffixes)."""
    n = str(name).strip()
    if n.endswith("-paraphrase-final"):
        return n[: -len("-paraphrase-final")]
    if n.endswith("-final"):
        return n[: -len("-final")]
    return n


def _iter_scenario_blocks(
    scenario_tasks: Dict[str, Any], iteration: Optional[str]
) -> List[Tuple[str, Dict[str, Any]]]:
    """If ``iteration`` is set, only that block; else all iteration keys (sorted)."""
    if iteration is not None:
        ib = scenario_tasks.get(iteration) or scenario_tasks.get(str(iteration))
        if not isinstance(ib, dict):
            return []
        return [(str(iteration), ib)]
    out: List[Tuple[str, Dict[str, Any]]] = []
    for it, scen_map in sorted(scenario_tasks.items(), key=lambda x: str(x[0])):
        if isinstance(scen_map, dict):
            out.append((str(it), scen_map))
    return out


def compute_commitment_summary(
    run: Dict[str, Any],
    rk: str,
    runs_path: Path,
    only_variant: Optional[str],
    iteration: Optional[str],
    *,
    log_variant_filter: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Build summary dict for one run (no file I/O). ``iteration=None`` uses all iteration blocks
    (same as legacy single-run behavior). For ``--all-models``, pass e.g. ``iteration=\"1\"``.
    """
    scenario_tasks = run.get("scenario_tasks") or {}
    tasks_flat: List[Tuple[str, str, Dict[str, Any]]] = []
    for it, scen_map in _iter_scenario_blocks(scenario_tasks, iteration):
        for sid, t in sorted(scen_map.items(), key=lambda x: str(x[0])):
            if isinstance(t, dict) and _task_has_commitment_rubric(t):
                tasks_flat.append((str(it), str(sid), t))

    if not tasks_flat:
        return None

    if only_variant:
        ov = only_variant.strip().lower()
        if ov not in ("og", "wa", "wb"):
            raise SystemExit("--only-variant must be one of: og, wa, wb")
        suffix = f"-{ov}"
        before = len(tasks_flat)
        tasks_flat = [
            (it, sid, t)
            for it, sid, t in tasks_flat
            if str(sid).endswith(suffix)
        ]
        if not tasks_flat:
            return None
        if log_variant_filter:
            print(
                f"Filtered to paraphrase variant {ov!r}: {len(tasks_flat)} / {before} tasks.",
                file=sys.stderr,
            )

    max_turns = 0
    for _it, _sid, t in tasks_flat:
        max_turns = max(max_turns, len(t.get("prompts") or []))
        max_turns = max(max_turns, len(t.get("turn_rubric_scores") or []))

    has_baseline = any(_baseline_commitment(t) is not None for _it, _sid, t in tasks_flat)
    stage_names = _ordered_stage_names(max_turns, has_baseline)

    per_task: List[Dict[str, Any]] = []
    matrix: List[List[Optional[float]]] = []

    for it, sid, t in tasks_flat:
        row = _extract_row(t, stage_names, has_baseline, max_turns)
        vec = [row[s] for s in stage_names]
        matrix.append(vec)
        delta_final_baseline: Optional[float] = None
        if row.get("baseline") is not None and row.get("final") is not None:
            delta_final_baseline = float(row["final"]) - float(row["baseline"])
        per_task.append(
            {
                "iteration": it,
                "scenario_id": sid,
                "stages": row,
                "delta_final_minus_baseline": delta_final_baseline,
            }
        )

    per_stage: Dict[str, Any] = {}
    for j, sname in enumerate(stage_names):
        col = [matrix[i][j] for i in range(len(matrix))]
        observed = [float(x) for x in col if x is not None]
        per_stage[sname] = _summarize(observed)
        m, lo, hi = _ci95(observed)
        per_stage[sname] = {**per_stage[sname], "ci95_low": lo, "ci95_high": hi}

    return {
        "run_key": rk,
        "runs_file": str(runs_path.resolve()),
        "only_variant": only_variant,
        "iteration_filter": iteration,
        "n_tasks": len(tasks_flat),
        "stage_order": stage_names,
        "per_stage": per_stage,
        "per_task": per_task,
    }


def _error_bar_across_models(col: List[float], error_bar: str) -> Tuple[float, float]:
    """Symmetric half-width for errorbar(yerr=([lo],[hi]))."""
    arr = np.array([x for x in col if not math.isnan(x)], dtype=float)
    if arr.size < 2:
        return 0.0, 0.0
    std_v = float(arr.std(ddof=1))
    if error_bar == "std":
        return std_v, std_v
    if error_bar == "sem":
        sem_v = std_v / math.sqrt(arr.size)
        return sem_v, sem_v
    sem_v = std_v / math.sqrt(arr.size)
    h = 1.96 * sem_v
    return h, h


def _means_vector_for_stages(
    per_stage: Dict[str, Any], canonical: List[str]
) -> List[float]:
    out: List[float] = []
    for s in canonical:
        block = per_stage.get(s)
        if not isinstance(block, dict) or "mean" not in block:
            out.append(float("nan"))
            continue
        m = block["mean"]
        out.append(float(m) if isinstance(m, (int, float)) and not math.isnan(float(m)) else float("nan"))
    return out


def run_all_models_analysis(
    runs_path: Path,
    out_dir: Path,
    only_variant: Optional[str],
    iteration: str,
    error_bar: str,
) -> None:
    runs = load_json_file(str(runs_path))
    if not isinstance(runs, dict):
        raise SystemExit("Runs file is not a JSON object.")

    if only_variant:
        print(
            f"--all-models: using iteration {iteration!r}, scenarios ending with "
            f"-{only_variant.lower()} only.",
            file=sys.stderr,
        )

    per_model: List[Dict[str, Any]] = []
    for rk, run in sorted(runs.items()):
        if not isinstance(run, dict):
            continue
        summary = compute_commitment_summary(
            run,
            rk,
            runs_path,
            only_variant,
            iteration,
            log_variant_filter=False,
        )
        if not summary:
            continue
        label = run.get("model_name") or run.get("test_model") or rk
        per_model.append(
            {
                "run_key": rk,
                "model_name": label,
                "n_tasks": summary["n_tasks"],
                "stage_order": summary["stage_order"],
                "per_stage": summary["per_stage"],
            }
        )

    if not per_model:
        raise SystemExit("No runs with commitment data after filters.")

    canonical: List[str] = list(per_model[0]["stage_order"])
    for row in per_model[1:]:
        if row["stage_order"] != canonical:
            print(
                f"Warning: stage_order differs for {row['run_key']!r}; "
                f"aligning to first run's stages (missing → NaN).",
                file=sys.stderr,
            )

    n_models = len(per_model)
    mat = np.full((n_models, len(canonical)), math.nan, dtype=float)
    for i, row in enumerate(per_model):
        vec = _means_vector_for_stages(row["per_stage"], canonical)
        mat[i, :] = vec

    grand_mean = np.nanmean(mat, axis=0)
    yerr_lo: List[float] = []
    yerr_hi: List[float] = []
    for j in range(len(canonical)):
        col = [float(mat[i, j]) for i in range(n_models) if not math.isnan(float(mat[i, j]))]
        lo, hi = _error_bar_across_models(col, error_bar)
        yerr_lo.append(lo)
        yerr_hi.append(hi)

    out_tag = _variant_suffix(only_variant)
    aggregate_path = out_dir / f"commitment_summary_all_models{out_tag}.json"
    report = {
        "mode": "all_models",
        "runs_file": str(runs_path.resolve()),
        "only_variant": only_variant,
        "iteration": iteration,
        "error_bar_across_models": error_bar,
        "n_models": n_models,
        "stage_order": canonical,
        "across_models_mean_by_stage": {
            canonical[j]: float(grand_mean[j]) for j in range(len(canonical))
        },
        "per_model": [
            {
                "run_key": m["run_key"],
                "model_name": m["model_name"],
                "n_tasks": m["n_tasks"],
                "mean_commitment_by_stage": {
                    canonical[j]: float(mat[i, j])
                    for j in range(len(canonical))
                    if not math.isnan(float(mat[i, j]))
                },
            }
            for i, m in enumerate(per_model)
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {aggregate_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib not available; skip plots ({e})", file=sys.stderr)
        return

    x = np.arange(len(canonical), dtype=float)
    x_labels_pub = []
    for s in canonical:
        if s == "baseline":
            x_labels_pub.append("Baseline")
        elif s == "final":
            x_labels_pub.append("Debrief")
        elif s.startswith("turn_"):
            try:
                n = int(s.split("_", 1)[1])
                x_labels_pub.append(f"$S_{n}$")
            except (ValueError, IndexError):
                x_labels_pub.append(s)
        else:
            x_labels_pub.append(s)

    err_name = {"std": "SD", "sem": "SEM", "ci95": "95% CI"}.get(error_bar, error_bar)
    variant_note = f", {only_variant.upper()} only" if only_variant else ""

    # --- Fig 1: grand mean ± error across models ---
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.errorbar(
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
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels_pub)
    ax.set_xlabel("Evaluation point")
    ax.set_ylabel("Mean commitment score (0–5)")
    ax.set_ylim(1, 6)
    ax.set_title(
        f"Commitment trajectory — mean across models (n={n_models}){variant_note}\n"
        f"Error bars: {err_name} across models (within-model mean over scenarios per stage)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.15)
    fig.tight_layout()
    p1 = out_dir / f"commitment_trajectory_mean_all_models{out_tag}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Wrote {p1}")

    # --- Fig 2: overlay each model’s within-run mean trajectory ---
    fig2, ax2 = plt.subplots(figsize=(10, 5.2))
    cmap = plt.get_cmap("tab10")
    for i, row in enumerate(per_model):
        y = mat[i, :]
        if np.all(np.isnan(y)):
            continue
        short = _legend_display_model_name(row["model_name"])
        if len(short) > 32:
            short = short[:29] + "…"
        ax2.plot(
            x,
            y,
            "-o",
            color=cmap(i % 10),
            linewidth=1.6,
            markersize=6,
            alpha=0.9,
            label=short,
        )
    ax2.plot(
        x,
        grand_mean,
        "k-",
        linewidth=2.5,
        marker="s",
        markersize=7,
        label="Mean (models)",
        zorder=10,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels_pub)
    ax2.set_xlabel("Evaluation point")
    ax2.set_ylabel("Mean commitment score (0–5)")
    ax2.set_ylim(1, 6)
    ax2.set_title(
        f"Commitment trajectories — all models overlaid (n={n_models}){variant_note}\n"
        f"Each line: mean over scenarios for that model at each stage"
    )
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / f"commitment_trajectory_overlay_all_models{out_tag}.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {p2}")


def run_analysis(
    runs_path: Path,
    run_key: Optional[str],
    out_dir: Path,
    max_lines_plot: int,
    only_variant: Optional[str] = None,
) -> None:
    runs = load_json_file(str(runs_path))
    if not isinstance(runs, dict):
        raise SystemExit("Runs file is not a JSON object.")

    if run_key:
        if run_key not in runs:
            raise SystemExit(f"Unknown run_key {run_key!r}. Keys: {sorted(runs)!r}")
        rk = run_key
    else:
        picked = _pick_latest_run_key(runs)
        if not picked:
            raise SystemExit("No runs in file.")
        rk = picked
        print(f"Using latest run_key={rk!r}")

    run = runs[rk]
    if not isinstance(run, dict):
        raise SystemExit("Run entry is not an object.")

    if run.get("scoring_mode") != "commitment":
        print(
            "Note: scoring_mode is not 'commitment'; still scanning tasks for commitment_score.",
            file=sys.stderr,
        )

    summary = compute_commitment_summary(run, rk, runs_path, only_variant, iteration=None)
    if not summary:
        raise SystemExit("No tasks with commitment_score found in this run.")

    stage_names = summary["stage_order"]
    per_stage = summary["per_stage"]
    per_task = summary["per_task"]
    matrix: List[List[Optional[float]]] = []
    for pt in per_task:
        row = pt["stages"]
        matrix.append([row[s] for s in stage_names])

    out_tag = _variant_suffix(only_variant)
    summary_out = {k: v for k, v in summary.items() if k != "iteration_filter"}
    summary_path = out_dir / f"commitment_summary_{rk}{out_tag}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Wrote {summary_path}")

    # --- plots ---
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib not available; skip plots ({e})", file=sys.stderr)
        return

    x = np.arange(len(stage_names))
    means = [per_stage[s]["mean"] for s in stage_names]
    lows = [per_stage[s]["ci95_low"] for s in stage_names]
    highs = [per_stage[s]["ci95_high"] for s in stage_names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(
        x,
        means,
        yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
        fmt="-o",
        capsize=4,
        color="tab:blue",
        ecolor="tab:blue",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names, rotation=35, ha="right")
    ax.set_ylabel("Commitment score (0–5)")
    ax.set_ylim(1, 6)
    ax.set_title(f"Mean commitment trajectory — {rk}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_mean = out_dir / f"commitment_trajectory_mean_{rk}{out_tag}.png"
    fig.savefig(p_mean, dpi=150)
    plt.close(fig)
    print(f"Wrote {p_mean}")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    n_lines = min(max_lines_plot, len(matrix))
    for i in range(n_lines):
        y = matrix[i]
        yy = [v if v is not None else np.nan for v in y]
        ax2.plot(x, yy, alpha=0.35, linewidth=1)
    ax2.plot(x, means, "-o", color="black", linewidth=2, label="Mean")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_names, rotation=35, ha="right")
    ax2.set_ylabel("Commitment score (0–5)")
    ax2.set_ylim(1, 6)
    ax2.set_title(f"Per-scenario trajectories (n={n_lines}) — {rk}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p_spaghetti = out_dir / f"commitment_trajectory_per_scenario_{rk}{out_tag}.png"
    fig2.savefig(p_spaghetti, dpi=150)
    plt.close(fig2)
    print(f"Wrote {p_spaghetti}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-json",
        type=Path,
        default=REPO_ROOT / "eqbench3_runs.json",
        help="Runs JSON path.",
    )
    ap.add_argument("--run-key", default=None, help="Run key (default: latest by timestamp).")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "commitment" / "results",
        help="Output directory for JSON + PNGs.",
    )
    ap.add_argument(
        "--max-lines",
        type=int,
        default=40,
        help="Max spaghetti lines (scenarios) on per-scenario plot. Use a large value (e.g. 999) to draw all.",
    )
    ap.add_argument(
        "--only-variant",
        choices=["og", "wa", "wb"],
        default=None,
        help="Keep only scenario_ids ending with -og / -wa / -wb (one wording per base). "
        "Omits other wordings from means and plots. Numeric ids (e.g. data/scenario_prompts.txt) are dropped.",
    )
    ap.add_argument(
        "--all-models",
        action="store_true",
        help="Aggregate every run in the JSON: mean trajectory across models + overlay plot. "
        "Uses --iteration (default 1). Ignores --run-key and --max-lines.",
    )
    ap.add_argument(
        "--iteration",
        default="1",
        help="Which scenario_tasks block to use for --all-models (default: 1).",
    )
    ap.add_argument(
        "--error-bar",
        choices=["std", "sem", "ci95"],
        default="sem",
        help="Spread across models on commitment_trajectory_mean_all_models*.png (default: sem).",
    )
    args = ap.parse_args()

    if not args.runs_json.is_file():
        raise SystemExit(f"File not found: {args.runs_json}")

    if args.all_models:
        if args.run_key:
            print("Ignoring --run-key because --all-models is set.", file=sys.stderr)
        run_all_models_analysis(
            args.runs_json,
            args.out_dir,
            args.only_variant,
            str(args.iteration),
            args.error_bar,
        )
        return

    run_analysis(
        args.runs_json,
        args.run_key,
        args.out_dir,
        args.max_lines,
        only_variant=args.only_variant,
    )


if __name__ == "__main__":
    main()
