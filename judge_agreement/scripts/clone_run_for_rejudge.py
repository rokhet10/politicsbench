#!/usr/bin/env python3
"""
Clone a completed run to a new top-level run_key with the same scenario_tasks
(test model transcripts and debriefs). Use this before re-invoking judges after you
change rubric / turn-judge prompts or truncation so the new scores live in a
separate entry for apples-to-apples comparison.

Next steps (repo root):

  python3 scripts/rejudge_saved_run.py \\
    --runs-file judge_agreement/results/judge_agreement_runs.json \\
    --run-key <NEW_KEY> \\
    [--truncate-for-rubric]

  python3 judge_agreement/scripts/analyze_judge_agreement.py \\
    --runs-json judge_agreement/results/judge_agreement_runs.json \\
    --run-key <NEW_KEY> \\
    --out-json judge_agreement/results/judge_agreement_summary_<suffix>.json
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import constants as C  # noqa: E402
from utils.file_io import load_json_file, save_json_file  # noqa: E402


def _patch_tasks_test_label(tasks: Any, label: str) -> None:
    if not isinstance(tasks, dict):
        return
    for _it, scen_map in tasks.items():
        if not isinstance(scen_map, dict):
            continue
        for _sid, tdict in scen_map.items():
            if isinstance(tdict, dict):
                tdict["test_model"] = label


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Clone a runs-file entry for judge-only reruns (same scenarios, new rubric calls)."
    )
    ap.add_argument(
        "--runs-file",
        type=Path,
        default=Path("judge_agreement/results/judge_agreement_runs.json"),
        help="Path to runs JSON (default: judge_agreement/results/judge_agreement_runs.json).",
    )
    ap.add_argument(
        "--from-run-key",
        default="judge_agree_gemini_gemini-judge-agreement-og",
        help="Source run_key to copy.",
    )
    ap.add_argument("--to-run-key", required=True, help="New run_key (must not exist unless --force).")
    ap.add_argument(
        "--model-name",
        default=None,
        help="Logical model_name / test_model label for the clone (default: <source>-rejudge-<date>).",
    )
    ap.add_argument(
        "--truncate-for-rubric",
        action="store_true",
        help="Record truncate_for_rubric=True on the clone (match rejudge_saved_run.py --truncate-for-rubric).",
    )
    ap.add_argument(
        "--sync-task-test-model-label",
        action="store_true",
        help="Set each task's test_model field to --model-name for cleaner provenance.",
    )
    ap.add_argument(
        "--note",
        default="",
        help="Stored on the clone as judge_prompting_note (e.g. less transcript context).",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite if to-run-key already exists.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions and exit without writing.")
    ap.add_argument("--verbosity", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.verbosity.upper(), logging.INFO))

    runs_path = args.runs_file
    if not runs_path.is_file():
        logging.error("Runs file not found: %s", runs_path)
        return 1

    runs: Dict[str, Any] = load_json_file(str(runs_path))
    if not isinstance(runs, dict):
        logging.error("Runs file did not load as a JSON object.")
        return 1

    src_key = args.from_run_key
    dst_key = args.to_run_key
    if src_key not in runs:
        logging.error("Unknown --from-run-key: %s", src_key)
        return 1
    if dst_key in runs and not args.force:
        logging.error("Target run_key already exists: %s (use --force)", dst_key)
        return 1

    src = runs[src_key]
    if not isinstance(src, dict):
        logging.error("Source run is not an object.")
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    model_name = args.model_name or f"{src.get('model_name', 'model')}-rejudge-{stamp}"

    new_run = copy.deepcopy(src)
    new_run["run_key"] = dst_key
    new_run["model_name"] = model_name
    new_run["test_model"] = model_name
    new_run["truncate_for_rubric"] = bool(args.truncate_for_rubric)
    new_run["status"] = "pending_rejudge"
    new_run["cloned_from_run_key"] = src_key
    new_run["clone_created_at"] = datetime.now(timezone.utc).isoformat()
    if args.note:
        new_run["judge_prompting_note"] = args.note

    # Refresh prompt file paths to current repo defaults (paths may match the old run).
    new_run["rubric_criteria_file_standard"] = C.STANDARD_RUBRIC_CRITERIA_FILE
    new_run["rubric_prompt_file_standard"] = C.STANDARD_RUBRIC_PROMPT_FILE

    for k in ("results",):
        if k in new_run:
            del new_run[k]

    if args.sync_task_test_model_label:
        _patch_tasks_test_label(new_run.get("scenario_tasks"), model_name)

    if args.dry_run:
        print(f"Would write new run_key={dst_key!r} model_name={model_name!r}")
        print(f"cloned_from={src_key!r} truncate_for_rubric={new_run['truncate_for_rubric']}")
        return 0

    runs[dst_key] = new_run
    if not save_json_file(runs, str(runs_path)):
        logging.error("Failed to save runs file.")
        return 1

    logging.info("Added run_key=%s (clone of %s). Next: rejudge_saved_run.py --run-key %s", dst_key, src_key, dst_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
