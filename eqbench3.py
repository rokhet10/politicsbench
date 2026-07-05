# File: ai/eqbench3/eqbench3.py

"""
Main entry point for the EQBench3 benchmark based on multi-turn scenarios.
Includes scenario simulation, debriefing, and optional rubric scoring.
"""
import argparse
import sys
import signal
import logging
from datetime import datetime, timezone
import os # For path joining
from typing import Dict, Any
# Added import for leaderboard printing
import unicodedata

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

from utils.logging_setup import setup_logging, get_verbosity
from utils.file_io import load_json_file, update_run_data, save_json_file # save_json_file needed for reset
# Logging is not configured until main() — imports below can take a while on cold start.
print("EQBench3: loading modules (cold start can take a while before logs appear)...", flush=True)
from core.benchmark import run_eq_bench3 # Import the main benchmark runner
import utils.constants as C # Import constants for default file paths


def signal_handler(signum, frame):
    """Handles graceful shutdown on signals like Ctrl+C."""
    print(f"\n[INFO] Signal {signum} received. Shutting down gracefully...")
    logging.info(f"Shutdown signal {signum} received.")
    # Perform any necessary cleanup here if needed
    sys.exit(1)

def print_summary_box(run_key: str, local_runs_file: str, run_rubric: bool):
    """
    Prints a formatted summary box of the benchmark run.
    Reads data ONLY from the local runs file.
    """
    try:
        # Load only local runs data for the summary box
        runs = load_json_file(local_runs_file)
        run_data = runs.get(run_key)
        if not run_data:
            print(f"\nError: Could not find run data for key {run_key} in local runs file {local_runs_file}")
            logging.warning(f"Summary box generation failed: Run key {run_key} not found in {local_runs_file}")
            return

        # Use model_name if available, fallback to test_model (legacy), then api_model_id
        model_name = run_data.get("model_name", run_data.get("test_model", "N/A"))
        api_model_id = run_data.get("api_model_id", "N/A")
        judge_models_list = run_data.get("judge_models")
        if isinstance(judge_models_list, list) and judge_models_list:
            judge_model = " / ".join(str(m) for m in judge_models_list)
        else:
            judge_model = run_data.get("judge_model", "N/A")
        start_time_str = run_data.get("start_time")
        end_time_str = run_data.get("end_time")
        run_status = run_data.get("status", "Unknown")

        duration_str = "N/A"
        if start_time_str and end_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                if start_time.tzinfo is None: start_time = start_time.replace(tzinfo=timezone.utc)
                if end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)
                duration = end_time - start_time
                total_seconds = duration.total_seconds()
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
            except ValueError as e:
                duration_str = "Error parsing time"
                logging.error(f"Error parsing run duration times: {start_time_str}, {end_time_str} - {e}")

        results = run_data.get("results", {})

        # --- Rubric Score (Calculate 0-100) ---
        rubric_score_0_20 = results.get("average_rubric_score", "N/A") # This is the 0-20 score
        rubric_error = results.get("rubric_error")
        rubric_score_100_str = "N/A" # Initialize the display string

        if not run_rubric:
            rubric_score_100_str = "Skipped"
        elif rubric_error:
            rubric_score_100_str = f"Error ({rubric_error[:20]}...)"
        elif isinstance(rubric_score_0_20, (int, float)):
            # Scale the 0-20 score to 0-100
            rubric_score_0_100 = rubric_score_0_20
            # * 5.0
            rubric_score_100_str = f"{rubric_score_0_100:.2f}" # Format the 0-100 score
        else:
            # Handle cases where score might be "N/A" or other non-numeric string
             rubric_score_100_str = str(rubric_score_0_20)

        def cell_width(s: str) -> int:
            w = 0
            for ch in s:
                w += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
            return w

        def crop_to_width(s: str, max_w: int, ellipsis="…") -> str:
            if cell_width(s) <= max_w:
                return s + " " * (max_w - cell_width(s))
            keep_w = max_w - cell_width(ellipsis)
            out = ""
            for ch in s:
                ch_w = 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
                if cell_width(out) + ch_w > keep_w:
                    break
                out += ch
            return out + ellipsis

        BOX_W = 80
        PAD = 1
        LINE = "─"
        TOP = "┌" + LINE * (BOX_W - 2) + "┐"
        BOTTOM = "└" + LINE * (BOX_W - 2) + "┘"
        ROW_SEP = "├" + LINE * (BOX_W - 2) + "┤"
        TITLE_SEP = "╞" + LINE * (BOX_W - 2) + "╡"

        rows = [
            ("Run Key:",               run_key),
            ("Model Name:",            model_name),
            ("API Model ID:",          api_model_id),
        ]
        if run_rubric:
            rows.append(("Judge (Rubric):", judge_model))

        rows.extend([
            ("Status:",                run_status),
            ("Duration:",              duration_str),
            ("Rubric Score (0‑100):",  rubric_score_100_str),
        ])

        # ──────────  calculate column widths  ──────────
        max_label = max(cell_width(lbl) for lbl, _ in rows)
        # cap label col so value column is at least 15‑char wide
        # constant characters per row:
        #   • 3 border glyphs  │ │ │
        #   • 4 padding spaces (PAD on each side of each cell)
        CONST_CHARS = 3 + 4*PAD          # = 7 when PAD == 1

        label_col = min(max(cell_width(lbl) for lbl, _ in rows),
                        BOX_W - CONST_CHARS - 15)     # leave ≥15 for value col
        value_col = BOX_W - CONST_CHARS - label_col   # <-- corrected here

        def make_row(lbl: str, val: str) -> str:
            lbl_fmt = crop_to_width(lbl,  label_col)
            val_fmt = crop_to_width(val,  value_col)
            return (f"│{' '*PAD}{lbl_fmt}{' '*PAD}│"
                    f"{' '*PAD}{val_fmt}{' '*PAD}│")

        # ──────────  render  ──────────
        print("\n" + TOP)
        print(make_row("", "EQBench3 Results Summary".center(value_col)))
        print(TITLE_SEP)
        for (lbl, val) in rows:
            if lbl == "Duration:":
                print(ROW_SEP)
            print(make_row(lbl, str(val)))
        print(BOTTOM)


    except Exception as e:
        print(f"\nError generating summary box: {e}")
        logging.error(f"Error generating summary box for run {run_key}", exc_info=True)


def print_rubric_summary(runs_data: Dict[str, Any], highlight_model: str):
    """Prints a formatted leaderboard summary based on Rubric Scores."""
    if not runs_data:
        print("\n[INFO] No run data available to display Rubric leaderboard.")
        return

    # --- Prepare data ---
    rubric_entries = []
    for run_key, run_data in runs_data.items():
        if not isinstance(run_data, dict):
            continue

        model_name = run_data.get("model_name", run_data.get("test_model", "Unknown"))
        results = run_data.get("results", {})
        rubric_score_0_20 = results.get("average_rubric_score") # 0-20 scale
        rubric_error = results.get("rubric_error")

        score_100 = 0.0 # Default for sorting if missing/error
        score_disp = "N/A"

        if rubric_error:
            score_disp = "Error"
        elif rubric_score_0_20 == "Skipped":
             score_disp = "Skipped"
        elif isinstance(rubric_score_0_20, (int, float)):
            score_100 = rubric_score_0_20
            # * 5.0
            score_disp = f"{score_100:.1f}" # Display with 1 decimal place
        elif rubric_score_0_20 is not None: # Handle other non-numeric strings like "N/A"
             score_disp = str(rubric_score_0_20)

        # Only include entries that have a valid score attempt (not skipped and has results)
        if "results" in run_data and rubric_score_0_20 != "Skipped":
            rubric_entries.append({
                "name": model_name,
                "score_100": score_100,
                "score_disp": score_disp,
            })

    # Sort by Rubric Score (0-100) descending
    rubric_entries.sort(key=lambda x: x["score_100"], reverse=True)

    # --- Formatting Helpers (copied from print_leaderboard_summary) ---
    def cell_width(s: str) -> int:
        w = 0
        for ch in s: w += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
        return w

    def crop_to_width(s: str, max_w: int, ellipsis="…") -> str:
        s = str(s) # Ensure string conversion
        if cell_width(s) <= max_w: return s + " " * (max_w - cell_width(s))
        keep_w = max_w - cell_width(ellipsis)
        out = ""
        current_w = 0
        for ch in s:
            ch_w = 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
            if current_w + ch_w > keep_w: break
            out += ch
            current_w += ch_w
        return out + ellipsis + " " * (max_w - cell_width(out + ellipsis))

    # --- Table Parameters ---
    COL_PAD = 1
    RANK_W = 4
    MODEL_W = 50 # Wider for model names
    SCORE_W = 15 # Width for "XXX.X / 100"

    cols = [RANK_W, MODEL_W, SCORE_W]
    LINE = "─"

    # --- Separator and Box Width Calculation ---
    segments = [LINE * (w + 2 * COL_PAD) for w in cols]
    ROW_SEP_INNER = "┼".join(segments)
    ROW_SEP = f"├{ROW_SEP_INNER}┤"
    BOX_W = len(ROW_SEP)

    # --- Recalculate other lines based on correct BOX_W ---
    TOP = "┌" + LINE * (BOX_W - 2) + "┐"
    BOTTOM = "└" + LINE * (BOX_W - 2) + "┘"
    TITLE_SEP = "╞" + LINE * (BOX_W - 2) + "╡"

    # --- Header Row ---
    headers = ["Rank", "Model Name", "Rubric (0-100)"]
    header_cells = [
        crop_to_width(h, w).center(w)
        for h, w in zip(headers, cols)
    ]
    header_parts = [f"{' ' * COL_PAD}{cell_content}{' ' * COL_PAD}" for cell_content in header_cells]
    header_row = f"│{'│'.join(header_parts)}│"

    # --- Print Table ---
    print("\n" + TOP)
    print(f"│{'EQBench3 Rubric Score Summary'.center(BOX_W - 2)}│")
    print(TITLE_SEP)
    print(header_row)
    print(ROW_SEP)

    for rank, entry in enumerate(rubric_entries, 1):
        is_highlighted = entry["name"] == highlight_model
        prefix = ">" if is_highlighted else " "
        rank_str = f"{prefix}{rank}"

        cells = [
            crop_to_width(rank_str, RANK_W),
            crop_to_width(entry["name"], MODEL_W),
            crop_to_width(entry["score_disp"], SCORE_W).rjust(SCORE_W),
        ]
        row_parts = [f"{' ' * COL_PAD}{cell_content}{' ' * COL_PAD}" for cell_content in cells]
        row_str = f"│{'│'.join(row_parts)}│"
        print(row_str)

    print(BOTTOM)

def _reset_model_data(model_name: str, local_runs_file: str):
    """Remove all entries for the logical `model_name` from the local runs file."""
    runs = load_json_file(local_runs_file) or {}
    to_delete = [k for k,v in runs.items()
                 if isinstance(v, dict) and v.get("model_name", v.get("test_model")) == model_name]
    deleted_count = 0
    for k in to_delete:
        if k in runs:
            del runs[k]
            deleted_count += 1
    if deleted_count > 0:
        save_json_file(runs, local_runs_file)
        logging.info(f"[RESET] Removed {deleted_count} run(s) for '{model_name}' from local runs file: {local_runs_file}")
    else:
        logging.info(f"[RESET] No runs found for '{model_name}' in {local_runs_file}")


def main():
    parser = argparse.ArgumentParser(description="Run EQBench3 Scenario Benchmark.")
    # --- Model Identifiers ---
    parser.add_argument("--test-model", required=True, help="Identifier for the model sent to the API (e.g., 'openai/gpt-4o'). This is the API Model ID.")
    parser.add_argument("--model-name", help="Logical identifier for the model (e.g., 'gpt-4o-june-2024') used for tracking and leaderboards. Defaults to the value of --test-model if not provided.")
    parser.add_argument("--judge-model", help="Single judge model id (used if --judge-models is not set).")
    parser.add_argument(
        "--judge-models",
        help="Comma-separated judge model ids; scores are averaged across judges. Overrides --judge-model when set.",
    )
    # --- File Paths ---
    parser.add_argument("--runs-file", default=C.DEFAULT_LOCAL_RUNS_FILE, help=f"File to store local run data (default: {C.DEFAULT_LOCAL_RUNS_FILE}).")
    parser.add_argument("--leaderboard-runs-file", default=C.CANONICAL_LEADERBOARD_RUNS_FILE, help=f"Path to the canonical leaderboard runs file (read-only, default: {C.CANONICAL_LEADERBOARD_RUNS_FILE}).")
    # --- Run Control ---
    parser.add_argument("--run-id", help="Optional: Resume or specify a run ID prefix for the local run.")
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel threads for API calls.")
    parser.add_argument("--verbosity", choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default="INFO", help="Logging verbosity level.")
    parser.add_argument("--save-interval", type=int, default=2, help="How often (in tasks) to save partial progress to local files.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of times to run each scenario (for assessing variance).")
    # --- Feature Flags ---
    parser.add_argument(
        "--ignore-canonical",
        action="store_true",
        default=False,
        help="If set, do not load or use default canonical leaderboard files. Runs will be based on local files only."
    )
    parser.add_argument("--no-rubric", action="store_true", default=False, help="Disable the Rubric scoring step.")
    parser.add_argument("--redo-rubric-judging", action="store_true", default=False,
                        help="If set, tasks in the local runs file that have completed rubric scoring will be reset so the rubric step is re-run.")
    parser.add_argument(
        "--reset-model",
        action="store_true",
        default=False,
        help="Delete all existing run data for the logical model name from the local runs file before running.",
    )
    # --- Removed file path arguments (now handled via constants) ---
    parser.add_argument(
        "--scenario-prompts-file",
        default=None,
        help="Path to scenario prompts .txt (default: scenario_prompts.txt at repo root via constants).",
    )
    parser.add_argument(
        "--paraphrase-manifest",
        default=None,
        help="Optional JSON manifest for paraphrase experiments; file SHA-256 is stored on the run. "
        "Baseline blanket questions: loaded from manifest baseline_questions when present; keys from "
        "scenario_prompts.txt (######## BASELINE_QUESTIONS JSON trailer) override the manifest when both are used.",
    )
    parser.add_argument(
        "--no-trait-judging",
        action="store_true",
        default=False,
        help="Skip 10-trait value rubrics (baseline, per-stage, final). Default is to run traits.",
    )
    parser.add_argument(
        "--no-commitment-judging",
        action="store_true",
        default=False,
        help="Skip commitment/stance judge (0-5). Default is to run commitment scoring alongside traits.",
    )
    parser.add_argument(
        "--commitment-scoring",
        action="store_true",
        default=False,
        help="Legacy: commitment-only run (same as --no-trait-judging). Prefer explicit --no-trait-judging.",
    )
    # parser.add_argument("--debrief-prompt-file", ...)
    # parser.add_argument("--pairwise-prompt-file", ...)
    # parser.add_argument("--rubric-criteria-file", ...)
    # parser.add_argument("--rubric-prompt-file", ...)

    args = parser.parse_args()

    # Determine the logical model name and API model ID
    api_model_id = args.test_model
    logical_model_name = args.model_name if args.model_name else api_model_id

    # Setup logging first
    setup_logging(get_verbosity(args.verbosity))

    actual_leaderboard_runs_file = args.leaderboard_runs_file

    if args.ignore_canonical:
        logging.info("--ignore-canonical flag is set. Canonical leaderboard files will not be loaded or used.")
        actual_leaderboard_runs_file = None
    else:
        if not os.path.exists(args.leaderboard_runs_file):
            logging.warning(f"Canonical leaderboard runs file not found: {args.leaderboard_runs_file}. Will proceed as if empty. Use --ignore-canonical to run purely locally and suppress this warning.")

    logging.info("--- EQBench3 Run Start ---")
    logging.info(f"Logical Model Name: {logical_model_name}")
    logging.info(f"API Model ID: {api_model_id}")
    logging.info(f"Local Runs File: {args.runs_file}")
    logging.info(f"Leaderboard Runs File (effective): {actual_leaderboard_runs_file if actual_leaderboard_runs_file else 'Ignored'}")
    logging.debug(f"Full Arguments: {args}")

    if args.reset_model:
        # Reset using the logical model name and LOCAL file paths
        _reset_model_data(logical_model_name, args.runs_file)

    run_rubric_flag = not args.no_rubric

    judge_models_resolved = None
    if args.judge_models and args.judge_models.strip():
        judge_models_resolved = [
            x.strip() for x in args.judge_models.split(",") if x.strip()
        ]
    elif args.judge_model and args.judge_model.strip():
        judge_models_resolved = [args.judge_model.strip()]

    if run_rubric_flag and not judge_models_resolved:
        parser.error(
            "Provide --judge-model and/or --judge-models unless --no-rubric is set."
        )
        sys.exit(1)

    if judge_models_resolved and len(judge_models_resolved) != len(
        set(judge_models_resolved)
    ):
        logging.warning(
            "Duplicate judge model ids in suite; keeping order as given."
        )

    trait_judging = not args.no_trait_judging
    commitment_judging = not args.no_commitment_judging
    if args.commitment_scoring:
        trait_judging = False
        commitment_judging = True
    if run_rubric_flag and not trait_judging and not commitment_judging:
        parser.error(
            "Rubric scoring is enabled but both trait and commitment judging are disabled. "
            "Remove --no-trait-judging and --no-commitment-judging (or use --no-rubric)."
        )
        sys.exit(1)

    # Hook signals for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    run_key = None
    try:
        # Call run_eq_bench3 with logical name, API ID, and all file paths
        run_key = run_eq_bench3(
            model_name=logical_model_name,
            api_model_id=api_model_id,
            judge_models=judge_models_resolved if run_rubric_flag else None,
            local_runs_file=args.runs_file,
            leaderboard_runs_file=actual_leaderboard_runs_file,
            num_threads=args.threads,
            run_id=args.run_id,
            save_interval=args.save_interval,
            iterations=args.iterations,
            run_rubric=run_rubric_flag,
            redo_judging=args.redo_rubric_judging,
            truncate_for_rubric=False, # Hardcoded for now, could be arg
            scenario_prompts_file=args.scenario_prompts_file,
            paraphrase_manifest_file=args.paraphrase_manifest,
            trait_judging=trait_judging,
            commitment_judging=commitment_judging,
        )

        logging.info(f"EQBench3 run completed. Run key: {run_key}")
        print(f"\nEQBench3 benchmark completed. Run key: {run_key}")

    except Exception as e:
        logging.critical(f"An unhandled error occurred during the benchmark run: {e}", exc_info=True)
        print(f"\nFATAL ERROR during benchmark run: {e}")
        if run_key and args.runs_file:
            try:
                # Update status in the LOCAL runs file
                update_run_data(args.runs_file, run_key, {
                    "status": "error",
                    "error": f"Unhandled exception: {str(e)}",
                    "end_time": datetime.now(timezone.utc).isoformat()
                })
                logging.info(f"Marked run {run_key} as errored in {args.runs_file}")
            except Exception as update_e:
                logging.error(f"Could not update run status to error for {run_key}: {update_e}")
        sys.exit(1)

    # Print Summary Box if run completed (even with errors), using LOCAL runs file
    if run_key:        
        # Print Rubric Leaderboard Summary if Rubric scoring was run
        if run_rubric_flag and run_key:
            try:
                logging.info("Loading merged run data for Rubric leaderboard display...")
                # Load final data from both sources, respecting --ignore-canonical
                final_leaderboard_runs = {}
                if not args.ignore_canonical:
                    final_leaderboard_runs = load_json_file(actual_leaderboard_runs_file)
                else:
                    logging.info("Rubric summary: Not loading canonical runs due to --ignore-canonical.")
                final_local_runs = load_json_file(args.runs_file)
                final_merged_runs = {**final_leaderboard_runs, **final_local_runs}

                if final_merged_runs:
                    print_rubric_summary(final_merged_runs, logical_model_name)
                else:
                    logging.warning("Could not load merged run data for Rubric leaderboard display.")
            except Exception as e:
                logging.error(f"Failed to load or print Rubric leaderboard summary: {e}", exc_info=True)
        elif not run_rubric_flag:
            logging.info("Skipping Rubric leaderboard display because Rubric scoring was disabled.")

        print_summary_box(run_key, args.runs_file, run_rubric_flag)


if __name__ == "__main__":
    main()