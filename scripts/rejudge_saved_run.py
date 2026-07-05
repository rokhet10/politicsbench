#!/usr/bin/env python3
"""
Re-run rubric judge calls using saved conversation_history in a local runs JSON file.

Use this after changing judge prompts or per-stage transcript logic. It re-invokes:
  - run_turn_rubric for each scenario stage (when --turns is set, default on)
  - final debrief / analysis rubric via the same path as the benchmark (when --final is set, default on)

The built-in ``eqbench3.py --redo-rubric-judging`` only resets *final* rubric fields and does not
re-call per-turn judges (those run inside run_scenario, which skips completed turns).

Examples:
  python scripts/rejudge_saved_run.py --runs-file eqbench3_runs.json --latest
  python scripts/rejudge_saved_run.py --runs-file eqbench3_runs.json --run-key abc123_model-name --max-tasks 5
  python scripts/rejudge_saved_run.py --runs-file paraphrase_robustness/results/paraphrase_runs.json --run-key RUN --threads 12

  # Commitment judge (0-5) on saved transcripts only — same stages as the benchmark:
  python scripts/rejudge_saved_run.py --runs-file eqbench3_runs.json --run-key YOUR_KEY --commitment-scoring

  # After cloning a run for a judge-prompt ablation (same test outputs, new run_key):
  #   python3 judge_agreement/scripts/clone_run_for_rejudge.py --to-run-key MY_NEW_KEY ...
  #   python scripts/rejudge_saved_run.py --runs-file judge_agreement/results/judge_agreement_runs.json --run-key MY_NEW_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmark import _execute_rubric_scoring_task  # noqa: E402
from core.conversation import ScenarioTask  # noqa: E402
from core.judge_suite import aggregate_rubric_scores  # noqa: E402
from utils.constants import (  # noqa: E402
    ANALYSIS_SCENARIO_IDS,
    ANALYSIS_RUBRIC_CRITERIA_FILE,
    ANALYSIS_RUBRIC_PROMPT_FILE,
    COMMITMENT_ANALYSIS_PROMPT_FILE,
    COMMITMENT_DEBRIEF_PROMPT_FILE,
    COMMITMENT_OUTPUT_FORMAT,
    COMMITMENT_TURN_PROMPT_FILE,
    STANDARD_RUBRIC_CRITERIA_FILE,
    STANDARD_RUBRIC_PROMPT_FILE,
    TURN_RUBRIC_OUTPUT_FORMAT,
    TURN_RUBRIC_PROMPT_TEMPLATE,
)
import utils.constants as C  # noqa: E402
from utils.file_io import load_json_file, update_run_data  # noqa: E402


def _load_output_formats() -> Tuple[str, str, str, str]:
    """Return (std_template, std_fmt, anl_template, anl_fmt)."""
    with open(STANDARD_RUBRIC_CRITERIA_FILE, "r", encoding="utf-8") as f:
        std_crit = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    if not std_crit:
        raise ValueError(f"Empty or missing criteria: {STANDARD_RUBRIC_CRITERIA_FILE}")
    out_std: Dict[str, Any] = {
        "chain_of_thought_reasoning": "detailed chain of thought reasoning about the coming scoring decisions"
    }
    for c in std_crit:
        out_std[c] = 0
    std_fmt = json.dumps(out_std, indent=2).replace(": 0", ": 0-20")
    std_tpl = Path(STANDARD_RUBRIC_PROMPT_FILE).read_text(encoding="utf-8")

    with open(ANALYSIS_RUBRIC_CRITERIA_FILE, "r", encoding="utf-8") as f:
        anl_crit = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    if not anl_crit:
        raise ValueError(f"Empty or missing criteria: {ANALYSIS_RUBRIC_CRITERIA_FILE}")
    out_anl: Dict[str, Any] = {
        "chain_of_thought_reasoning": "detailed chain of thought reasoning about the coming scoring decisions"
    }
    for c in anl_crit:
        out_anl[c] = 0
    anl_fmt = json.dumps(out_anl, indent=2).replace(": 0", ": 0-20")
    anl_tpl = Path(ANALYSIS_RUBRIC_PROMPT_FILE).read_text(encoding="utf-8")
    return std_tpl, std_fmt, anl_tpl, anl_fmt


def _load_commitment_formats() -> Tuple[str, str, str, str, str, str]:
    """
    Return (turn_tpl, turn_fmt, debrief_tpl, debrief_fmt, analysis_tpl, analysis_fmt).
    debrief_fmt and analysis_fmt are the same JSON schema (commitment_score 0-5).
    """
    turn_tpl = Path(COMMITMENT_TURN_PROMPT_FILE).read_text(encoding="utf-8")
    debrief_tpl = Path(COMMITMENT_DEBRIEF_PROMPT_FILE).read_text(encoding="utf-8")
    analysis_tpl = Path(COMMITMENT_ANALYSIS_PROMPT_FILE).read_text(encoding="utf-8")
    fmt = COMMITMENT_OUTPUT_FORMAT
    if "{transcript}" not in turn_tpl or "{output_format}" not in turn_tpl:
        raise ValueError("Commitment turn prompt missing {transcript} or {output_format}.")
    if (
        "{transcript}" not in debrief_tpl
        or "{debrief}" not in debrief_tpl
        or "{output_format}" not in debrief_tpl
    ):
        raise ValueError(
            "Commitment debrief prompt missing {transcript}, {debrief}, or {output_format}."
        )
    if "{transcript}" not in analysis_tpl or "{output_format}" not in analysis_tpl:
        raise ValueError("Commitment analysis prompt missing {transcript} or {output_format}.")
    return turn_tpl, fmt, debrief_tpl, fmt, analysis_tpl, fmt


def _resolve_judge_models(run_data: Dict[str, Any]) -> List[str]:
    jm = run_data.get("judge_models")
    if isinstance(jm, list) and jm:
        return [str(x).strip() for x in jm if str(x).strip()]
    single = run_data.get("judge_model")
    if single and str(single).strip():
        return [str(single).strip()]
    raise ValueError(
        "Run record has no judge_model / judge_models; pass --judge-models explicitly."
    )


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


def _task_eligible(t: Dict[str, Any]) -> bool:
    if not isinstance(t, dict):
        return False
    hist = t.get("conversation_history") or []
    prompts = t.get("prompts") or []
    if not hist or not prompts:
        return False
    if len(hist) < 2 * len(prompts):
        return False
    st = t.get("status")
    return st in ("completed", "rubric_scored", "scenario_completed")


def dry_run_turn_example(
    task_dict: Dict[str, Any],
    turn_index: Optional[int] = None,
    *,
    turn_tpl: str = TURN_RUBRIC_PROMPT_TEMPLATE,
    turn_fmt: str = TURN_RUBRIC_OUTPUT_FORMAT,
) -> None:
    """Print an illustrative turn-rubric judge prompt (default: last stage, or turn_index)."""
    task = ScenarioTask.from_dict(task_dict)
    prompts = task.prompts or []
    if not prompts:
        logging.info("Dry-run example skipped: no prompts on task %s.", task.scenario_id)
        return
    if turn_index is None:
        turn_index = len(prompts) - 1
    if turn_index < 0 or turn_index >= len(prompts):
        logging.warning("Invalid turn_index %s for task %s.", turn_index, task.scenario_id)
        return
    body = task.build_single_stage_rubric_transcript(
        turn_index, truncate_for_rubric=True
    )
    if not body:
        logging.warning("Could not build transcript for turn %s.", turn_index)
        return
    prompt = turn_tpl.format(
        transcript=body,
        output_format=turn_fmt,
    )
    stage_label = turn_index + 1
    print(
        f"\n=== Example: Stage {stage_label} (turn_index={turn_index}) "
        "turn-rubric user message to judge ===\n"
    )
    print(prompt.replace("*", "").replace("#", "")[:12000])
    if len(prompt) > 12000:
        print("\n... [truncated for display] ...\n")


def rejudge_task(
    task_dict: Dict[str, Any],
    api_clients: Dict[str, Any],
    judge_models: List[str],
    trait_std_tpl: str,
    trait_std_fmt: str,
    trait_anl_tpl: str,
    trait_anl_fmt: str,
    do_turns: bool,
    do_final: bool,
    truncate_for_rubric: bool,
    trait_turn_tpl: str,
    trait_turn_fmt: str,
    trait_judging: bool,
    c_turn_tpl: Optional[str],
    c_turn_fmt: Optional[str],
    c_std_tpl: Optional[str],
    c_anl_tpl: Optional[str],
    c_out_fmt: Optional[str],
    commitment_judging: bool,
    commitment_only_storage: bool,
) -> ScenarioTask:
    task = ScenarioTask.from_dict(task_dict)
    if do_turns:
        if task.baseline_prompt:
            bh = task.baseline_conversation_history or []
            if len(bh) >= 2 and bh[-1].get("role") == "assistant":
                if trait_judging:
                    task.run_baseline_rubric(
                        api_clients=api_clients,
                        rubric_prompt_template=trait_turn_tpl,
                        rubric_output_format_str=trait_turn_fmt,
                        judge_models=judge_models,
                    )
                if commitment_judging and c_turn_tpl and c_turn_fmt:
                    if commitment_only_storage:
                        task.run_baseline_rubric(
                            api_clients=api_clients,
                            rubric_prompt_template=c_turn_tpl,
                            rubric_output_format_str=c_turn_fmt,
                            judge_models=judge_models,
                        )
                    else:
                        task.run_baseline_commitment_rubric(
                            api_clients=api_clients,
                            rubric_prompt_template=c_turn_tpl,
                            rubric_output_format_str=c_turn_fmt,
                            judge_models=judge_models,
                        )
        n = len(task.prompts or [])
        for turn_index in range(n):
            if trait_judging:
                task.run_turn_rubric(
                    api_clients=api_clients,
                    rubric_prompt_template=trait_turn_tpl,
                    rubric_output_format_str=trait_turn_fmt,
                    turn_index=turn_index,
                    judge_models=judge_models,
                )
            if commitment_judging and c_turn_tpl and c_turn_fmt:
                if commitment_only_storage:
                    task.run_turn_rubric(
                        api_clients=api_clients,
                        rubric_prompt_template=c_turn_tpl,
                        rubric_output_format_str=c_turn_fmt,
                        turn_index=turn_index,
                        judge_models=judge_models,
                    )
                else:
                    task.run_turn_commitment_rubric(
                        api_clients=api_clients,
                        rubric_prompt_template=c_turn_tpl,
                        rubric_output_format_str=c_turn_fmt,
                        turn_index=turn_index,
                        judge_models=judge_models,
                    )
    if do_final:
        is_analysis = task.scenario_id in ANALYSIS_SCENARIO_IDS
        trait_tmpl = trait_anl_tpl if is_analysis else trait_std_tpl
        trait_fmt = trait_anl_fmt if is_analysis else trait_std_fmt
        c_tmpl = c_anl_tpl if is_analysis else c_std_tpl
        _execute_rubric_scoring_task(
            task,
            api_clients,
            judge_models,
            trait_tmpl if trait_judging else None,
            trait_fmt if trait_judging else None,
            c_tmpl if commitment_judging else None,
            c_out_fmt if commitment_judging else None,
            queue.Queue(),
            "",
            truncate_for_rubric,
            trait_judging,
            commitment_judging,
            commitment_only_storage,
        )
    return task


def _collect_jobs(
    scenario_tasks: Any, max_tasks: int
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Ordered (iteration, scenario_id, task_dict) for eligible tasks."""
    jobs: List[Tuple[str, str, Dict[str, Any]]] = []
    for iter_str, scen_map in sorted(scenario_tasks.items(), key=lambda x: str(x[0])):
        if not isinstance(scen_map, dict):
            continue
        for sid, tdict in sorted(scen_map.items(), key=lambda x: str(x[0])):
            if not _task_eligible(tdict):
                continue
            jobs.append((iter_str, sid, tdict))
            if max_tasks and len(jobs) >= max_tasks:
                return jobs
    return jobs


def _process_one_task(
    job: Tuple[str, str, Dict[str, Any]],
    api_clients: Dict[str, Any],
    judge_models: List[str],
    trait_std_tpl: str,
    trait_std_fmt: str,
    trait_anl_tpl: str,
    trait_anl_fmt: str,
    do_turns: bool,
    do_final: bool,
    truncate_for_rubric: bool,
    trait_turn_tpl: str,
    trait_turn_fmt: str,
    trait_judging: bool,
    c_turn_tpl: Optional[str],
    c_turn_fmt: Optional[str],
    c_std_tpl: Optional[str],
    c_anl_tpl: Optional[str],
    c_out_fmt: Optional[str],
    commitment_judging: bool,
    commitment_only_storage: bool,
) -> Tuple[str, str, ScenarioTask]:
    iter_str, sid, tdict = job
    task = rejudge_task(
        tdict,
        api_clients,
        judge_models,
        trait_std_tpl,
        trait_std_fmt,
        trait_anl_tpl,
        trait_anl_fmt,
        do_turns=do_turns,
        do_final=do_final,
        truncate_for_rubric=truncate_for_rubric,
        trait_turn_tpl=trait_turn_tpl,
        trait_turn_fmt=trait_turn_fmt,
        trait_judging=trait_judging,
        c_turn_tpl=c_turn_tpl,
        c_turn_fmt=c_turn_fmt,
        c_std_tpl=c_std_tpl,
        c_anl_tpl=c_anl_tpl,
        c_out_fmt=c_out_fmt,
        commitment_judging=commitment_judging,
        commitment_only_storage=commitment_only_storage,
    )
    return iter_str, sid, task


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run rubric judges on saved run JSON.")
    parser.add_argument("--runs-file", default=C.DEFAULT_LOCAL_RUNS_FILE)
    parser.add_argument("--run-key", help="Explicit run_key (default: --latest)")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Select the run with the lexicographically greatest end/start timestamp.",
    )
    parser.add_argument(
        "--judge-models",
        help="Comma-separated judge API ids (overrides run metadata).",
    )
    parser.add_argument(
        "--turns-only",
        action="store_true",
        help="Only re-run per-stage turn rubrics.",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Only re-run final rubric (debrief or analysis).",
    )
    parser.add_argument(
        "--truncate-for-rubric",
        action="store_true",
        default=False,
        help="Pass truncate_for_rubric=True to prepare_rubric_prompt_text (default: False, matches eqbench3).",
    )
    parser.add_argument("--max-tasks", type=int, default=0, help="Cap tasks processed (0 = no cap).")
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Parallel workers (each task = all its turn rubrics + final). Default: 8. Use 1 to run sequentially.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one turn-rubric example prompt (see --example-turn) and exit without API calls.",
    )
    parser.add_argument(
        "--example-turn",
        type=int,
        default=None,
        metavar="INDEX",
        help="With --dry-run, which turn_index to show (0-based). Default: last stage.",
    )
    parser.add_argument(
        "--commitment-scoring",
        action="store_true",
        default=False,
        help="Legacy: commitment-only rejudge (overrides run scoring_mode). Default follows run metadata.",
    )
    parser.add_argument("--verbosity", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.verbosity.upper(), logging.INFO))

    do_turns = not args.final_only
    do_final = not args.turns_only
    if args.turns_only and args.final_only:
        logging.error("Choose at most one of --turns-only / --final-only.")
        sys.exit(2)
    if args.threads < 1:
        logging.error("--threads must be >= 1.")
        sys.exit(2)

    runs_path = Path(args.runs_file)
    if not runs_path.is_file():
        logging.error("Runs file not found: %s", runs_path)
        sys.exit(1)

    runs = load_json_file(str(runs_path))
    if not isinstance(runs, dict):
        logging.error("Runs file did not load as a JSON object.")
        sys.exit(1)

    run_key = args.run_key
    if args.latest or not run_key:
        run_key = _pick_latest_run_key(runs)
        if not run_key:
            logging.error("Could not determine a run key (empty file?).")
            sys.exit(1)
        logging.info("Using run_key=%s", run_key)

    run_data = runs.get(run_key)
    if not isinstance(run_data, dict):
        logging.error("Unknown run_key: %s", run_key)
        sys.exit(1)

    sm = run_data.get("scoring_mode", "traits")
    trait_judging = sm in ("traits", "both")
    commitment_judging = sm in ("commitment", "both")
    if args.commitment_scoring:
        trait_judging = False
        commitment_judging = True
    commitment_only_storage = commitment_judging and not trait_judging

    trait_std_tpl = trait_std_fmt = trait_anl_tpl = trait_anl_fmt = ""
    trait_turn_tpl = trait_turn_fmt = ""
    c_turn_tpl = c_turn_fmt = c_std_tpl = c_anl_tpl = c_out_fmt = None

    if trait_judging:
        trait_std_tpl, trait_std_fmt, trait_anl_tpl, trait_anl_fmt = _load_output_formats()
        trait_turn_tpl = TURN_RUBRIC_PROMPT_TEMPLATE
        trait_turn_fmt = TURN_RUBRIC_OUTPUT_FORMAT
    if commitment_judging:
        c_turn_tpl, c_turn_fmt, c_std_tpl, c_out_fmt, c_anl_tpl, c_out2 = (
            _load_commitment_formats()
        )
        assert c_out_fmt == c_out2
    logging.info(
        "Rejudge: traits=%s commitment=%s (scoring_mode=%s)",
        trait_judging,
        commitment_judging,
        sm,
    )

    if args.judge_models:
        judge_models = [x.strip() for x in args.judge_models.split(",") if x.strip()]
    else:
        judge_models = _resolve_judge_models(run_data)

    if args.dry_run:
        scenario_tasks = run_data.get("scenario_tasks") or {}
        want_turn = args.example_turn
        for _it, scen_map in scenario_tasks.items():
            if not isinstance(scen_map, dict):
                continue
            for _sid, tdict in scen_map.items():
                if not _task_eligible(tdict):
                    continue
                n = len(tdict.get("prompts") or [])
                if want_turn is not None and want_turn >= n:
                    continue
                dry_turn_tpl = trait_turn_tpl if trait_judging else (c_turn_tpl or "")
                dry_turn_fmt = trait_turn_fmt if trait_judging else (c_turn_fmt or "")
                dry_run_turn_example(
                    tdict,
                    turn_index=want_turn,
                    turn_tpl=dry_turn_tpl,
                    turn_fmt=dry_turn_fmt,
                )
                return
        logging.warning("No eligible task found for dry-run example.")
        return

    from utils.api import APIClient  # noqa: WPS433

    api_clients = {"judge": APIClient(model_type="judge")}

    scenario_tasks = run_data.get("scenario_tasks") or {}
    jobs = _collect_jobs(scenario_tasks, args.max_tasks)
    if not jobs:
        logging.warning("No eligible tasks to rejudge.")
        return

    logging.info(
        "Rejudging %s task(s) with %s worker thread(s).",
        len(jobs),
        args.threads,
    )

    def _submit(job: Tuple[str, str, Dict[str, Any]]) -> Tuple[str, str, ScenarioTask]:
        return _process_one_task(
            job,
            api_clients,
            judge_models,
            trait_std_tpl,
            trait_std_fmt,
            trait_anl_tpl,
            trait_anl_fmt,
            do_turns,
            do_final,
            args.truncate_for_rubric,
            trait_turn_tpl,
            trait_turn_fmt,
            trait_judging,
            c_turn_tpl,
            c_turn_fmt,
            c_std_tpl,
            c_anl_tpl,
            c_out_fmt,
            commitment_judging,
            commitment_only_storage,
        )

    processed = 0
    if args.threads == 1:
        for job in jobs:
            iter_str, sid, task = _submit(job)
            ok = update_run_data(
                str(runs_path),
                run_key,
                {"scenario_tasks": {iter_str: {sid: task.to_dict()}}},
            )
            if not ok:
                logging.error("Failed to save task %s iter %s", sid, iter_str)
                sys.exit(1)
            processed += 1
            logging.info(
                "Rejudged task scenario=%s iter=%s status=%s", sid, iter_str, task.status
            )
    else:
        workers = min(args.threads, len(jobs))
        errors: List[Exception] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="Rejudge") as ex:
            future_map = {ex.submit(_submit, job): job for job in jobs}
            for fut in as_completed(future_map):
                job = future_map[fut]
                try:
                    iter_str, sid, task = fut.result()
                except Exception as e:
                    logging.exception(
                        "Task failed scenario=%s iter=%s: %s", job[1], job[0], e
                    )
                    errors.append(e)
                    continue
                ok = update_run_data(
                    str(runs_path),
                    run_key,
                    {"scenario_tasks": {iter_str: {sid: task.to_dict()}}},
                )
                if not ok:
                    logging.error("Failed to save task %s iter %s", sid, iter_str)
                    sys.exit(1)
                processed += 1
                logging.info(
                    "Rejudged task scenario=%s iter=%s status=%s",
                    sid,
                    iter_str,
                    task.status,
                )
        if errors:
            logging.error("%s task(s) failed; see logs above.", len(errors))
            sys.exit(1)

    logging.info("Done. Rejudged %s task(s).", processed)


if __name__ == "__main__":
    main()
