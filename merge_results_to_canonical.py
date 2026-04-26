# File: ai/eqbench3/utils/merge_candidates.py

import os
import sys

# Add the parent directory (ai/eqbench3) to sys.path to allow imports like 'from utils. ...'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


import logging

import copy
from datetime import datetime, timezone
import argparse
from collections import defaultdict
from core.elo import (
    get_solver_comparisons,
    models_in_comparisons,
)
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Tuple
from core.trueskill_solver import solve_with_trueskill, normalize_elo_scores
from core.pairwise_judging import _recompute_comparison_stats
# from core.elo import models_in_comparisons           # helper already written # Redundant import
from core.elo_config import DEFAULT_ELO, WIN_MARGIN_BIN_SIZE, WIN_MARGIN_BIN_SIZE_FOR_CI, RANK_WINDOW
from utils.file_io import load_json_file, save_json_file
import utils.constants as C
from core.elo_helpers import should_ignore_scenario
import uuid, shutil # For atomic save

# --- Logging Setup ---
def setup_merge_logging(level_str):
    log_level = getattr(logging, level_str.upper(), logging.INFO)

    # Re-initialise logging even if it was configured earlier
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True              # <-- key line
    )

    # If other modules grabbed the root logger before this, make sure they honour the new level.
    logging.getLogger().setLevel(log_level)

    logging.debug(f"Logging level set to {level_str.upper()}")
# --- End Logging Setup ---


# --- End Imports ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configured in setup_merge_logging

# =========================================================================
# Merge Functionality
# =========================================================================

def find_merge_candidates(local_runs, local_elo, canonical_runs, canonical_elo):
    """Identifies runs in local files that meet merge criteria."""
    candidates = []
    processed_model_names = set() # Track models already added as candidates

    # Build set of model names already in canonical data (runs or elo)
    canonical_model_names = set(k for k in canonical_elo if k != "__metadata__")
    for run_data in canonical_runs.values():
        if isinstance(run_data, dict):
            model_name = run_data.get("model_name", run_data.get("test_model"))
            if model_name:
                canonical_model_names.add(model_name)

    logging.info(f"Found {len(canonical_model_names)} unique model names in canonical data.")
    logging.debug(f"Total items found in local_runs: {len(local_runs)}")

    for run_key, run_data in local_runs.items():
        if run_key == "__metadata__": continue # Skip metadata entry
        logging.debug(f"Processing local run key: '{run_key}'")

        if not isinstance(run_data, dict):
            logging.debug(f"Skipping '{run_key}': Run data is not a dictionary.")
            continue

        model_name = run_data.get("model_name", run_data.get("test_model"))
        if not model_name:
            logging.warning(f"Skipping run {run_key}: Missing model name.")
            continue

        # Avoid adding the same model multiple times if it has multiple local runs
        if model_name in processed_model_names:
            logging.debug(f"Skipping '{run_key}' (model '{model_name}'): Model name already processed from a previous run key.")
            continue

        # --- Check Criteria ---
        # 1. Exists in local ELO?
        if model_name not in local_elo or not isinstance(local_elo.get(model_name), dict):
            logging.debug(f"Skipping {model_name} ({run_key}): Missing or invalid entry in local ELO file.")
            continue

        # 2. Completeness (Rubric, ELO scores, Matchups)
        results = run_data.get("results", {})
        rubric_score = results.get("average_rubric_score")
        elo_raw = results.get("elo_raw")
        has_rubric = rubric_score is not None and rubric_score != "Skipped" and results.get("rubric_error") is None
        has_elo = elo_raw is not None and elo_raw != "Skipped" and results.get("elo_error") is None

        if not has_rubric:
            logging.debug(f"Skipping {model_name} ({run_key}): Missing valid Rubric score.")
            continue
        if not has_elo:
            logging.debug(f"Skipping {model_name} ({run_key}): Missing valid ELO score.")
            continue

        # Check for matchups involving this model in local ELO comparisons
        has_matchups = False
        local_comps = local_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
        for comp in local_comps:
            pair = comp.get("pair", {})
            if model_name in (pair.get("test_model"), pair.get("neighbor_model")):
                has_matchups = True
                break
        if not has_matchups:
            logging.debug(f"Skipping {model_name} ({run_key}): No matchups found in local ELO comparisons.")
            continue

        # 3. No Name Collision in Canonical Data
        if model_name in canonical_model_names:
            logging.debug(f"Skipping {model_name} ({run_key}): Name already exists in canonical data.")
            continue

        # --- Candidate Found ---
        candidates.append({
            "run_key": run_key,
            "model_name": model_name,
            "rubric_score": rubric_score * 5.0 if isinstance(rubric_score, (int, float)) else "N/A",
            "elo_norm": results.get("elo_normalized", "N/A")
        })
        processed_model_names.add(model_name)
        logging.info(f"Found potential merge candidate: {model_name} (from run {run_key})")

    return candidates

def merge_data(selected_candidates, local_runs, local_elo, canonical_runs, canonical_elo):
    """Moves selected run data and relevant comparisons from local to canonical files."""
    if not selected_candidates:
        return False # Indicate nothing was merged

    merged_model_names = set(c['model_name'] for c in selected_candidates)
    logging.info(f"Preparing to merge {len(merged_model_names)} models: {', '.join(merged_model_names)}")

    # Determine the final set of models that will be in the canonical ELO file
    final_canonical_models = set(k for k in canonical_elo if k != "__metadata__")
    final_canonical_models.update(merged_model_names)
    logging.info(f"Final canonical ELO file will contain {len(final_canonical_models)} models.")

    # --- Process Local ELO Comparisons ---
    local_comps = local_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    comps_to_move = []
    comps_to_keep_local = []
    moved_comp_count = 0

    for comp in local_comps:
        pair = comp.get("pair", {})
        model_a = pair.get("test_model")
        model_b = pair.get("neighbor_model")

        # Check if this comparison involves one of the models being merged
        comp_involves_merged_model = model_a in merged_model_names or model_b in merged_model_names

        if comp_involves_merged_model:
            # Check if BOTH models in the pair will be in the final canonical set
            if model_a in final_canonical_models and model_b in final_canonical_models:
                comps_to_move.append(comp)
                moved_comp_count += 1
                logging.debug(f"Moving comparison to canonical: {model_a} vs {model_b}")
            else:
                # Keep comparison locally if it involves a merged model but the other model isn't canonical
                comps_to_keep_local.append(comp)
                logging.debug(f"Keeping comparison locally (one model not canonical): {model_a} vs {model_b}")
        else:
            # Keep comparison locally if it doesn't involve any model being merged now
            comps_to_keep_local.append(comp)

    logging.info(f"Identified {moved_comp_count} comparisons to move to canonical ELO.")
    logging.info(f"{len(comps_to_keep_local)} comparisons will remain in local ELO.")

    # Update local ELO comparisons
    if "__metadata__" not in local_elo: local_elo["__metadata__"] = {}
    local_elo["__metadata__"]["global_pairwise_comparisons"] = comps_to_keep_local

    # Add comparisons to canonical ELO
    if "__metadata__" not in canonical_elo: canonical_elo["__metadata__"] = {}
    if "global_pairwise_comparisons" not in canonical_elo["__metadata__"]:
        canonical_elo["__metadata__"]["global_pairwise_comparisons"] = []
    canonical_elo["__metadata__"]["global_pairwise_comparisons"].extend(comps_to_move)

    # --- Move Run Data and ELO Entries ---
    moved_run_keys = set()
    for candidate in selected_candidates:
        run_key = candidate["run_key"]
        model_name = candidate["model_name"]

        # Move run data (handle multiple runs for the same model)
        # Find all run keys associated with the model name in local_runs
        run_keys_for_model = [
            r_key for r_key, r_data in local_runs.items()
            if r_key != "__metadata__" and isinstance(r_data, dict) and
               r_data.get("model_name", r_data.get("test_model")) == model_name
        ]

        for r_key_to_move in run_keys_for_model:
            if r_key_to_move in local_runs and r_key_to_move not in moved_run_keys:
                canonical_runs[r_key_to_move] = local_runs[r_key_to_move]
                del local_runs[r_key_to_move]
                moved_run_keys.add(r_key_to_move)
                logging.info(f"Moved run data for {r_key_to_move} (model {model_name}) to canonical runs.")
            elif r_key_to_move in moved_run_keys:
                 logging.debug(f"Run key {r_key_to_move} already moved.")
            else:
                logging.warning(f"Run key {r_key_to_move} (for model {model_name}) not found in local runs data during merge.")


        # Move ELO entry (only once per model)
        if model_name in local_elo:
            if model_name != "__metadata__": # Safety check
                canonical_elo[model_name] = local_elo[model_name]
                del local_elo[model_name]
                logging.info(f"Moved ELO entry for {model_name} to canonical ELO.")
        else:
            # This might happen if the ELO entry was already moved by another run of the same model (shouldn't if logic is correct, but good to check)
            logging.warning(f"Model name {model_name} not found in local ELO data during merge (might indicate an issue or already moved).")

    return True # Indicate merging occurred

# =========================================================================
# Unmerge Functionality
# =========================================================================

def find_unmerge_candidates(canonical_runs, canonical_elo):
    """Identifies models present in canonical files that can be moved to local."""
    candidates = []
    model_names_in_elo = set(k for k in canonical_elo if k != "__metadata__")
    model_names_in_runs = set()
    run_key_map = defaultdict(list) # model_name -> list of run_keys

    for run_key, run_data in canonical_runs.items():
        if run_key == "__metadata__": continue
        if isinstance(run_data, dict):
            model_name = run_data.get("model_name", run_data.get("test_model"))
            if model_name:
                model_names_in_runs.add(model_name)
                run_key_map[model_name].append(run_key)

    # Consider models present in either file for unmerging
    all_canonical_models = sorted(list(model_names_in_elo | model_names_in_runs))

    logging.info(f"Found {len(all_canonical_models)} unique model names in canonical data for potential unmerging.")

    for model_name in all_canonical_models:
        elo_data = canonical_elo.get(model_name, {})
        elo_norm = elo_data.get("elo_norm", "N/A") if isinstance(elo_data, dict) else "N/A"
        run_keys = run_key_map.get(model_name, ["N/A"])

        candidates.append({
            "model_name": model_name,
            "elo_norm": elo_norm,
            "run_keys": run_keys # Store all associated run keys
        })

    return candidates

def unmerge_data(selected_candidates: List[Dict[str, Any]],
                 local_runs: Dict[str, Any],
                 local_elo: Dict[str, Any],
                 canonical_runs: Dict[str, Any],
                 canonical_elo: Dict[str, Any]) -> bool:
    """Moves selected run data and relevant comparisons from canonical to local files."""
    if not selected_candidates:
        logging.info("No candidates selected for unmerge. Nothing to do.")
        return False # Indicate nothing was unmerged because nothing was selected

    unmerged_model_names = set(c['model_name'] for c in selected_candidates)
    logging.info(f"Preparing to unmerge {len(unmerged_model_names)} models: {', '.join(sorted(list(unmerged_model_names)))}")

    # Determine the sets of models in each location *after* the move
    # Models remaining canonical are those currently canonical MINUS those being unmerged.
    current_canonical_models = set(k for k in canonical_elo if k != "__metadata__")
    remaining_canonical_models = current_canonical_models - unmerged_model_names

    # Models ending up in local are those currently local PLUS those being unmerged.
    current_local_models = set(k for k in local_elo if k != "__metadata__")
    final_local_models = current_local_models | unmerged_model_names

    logging.info(f"Canonical ELO file will contain {len(remaining_canonical_models)} models after unmerge.")
    logging.info(f"Local ELO file will contain {len(final_local_models)} models after unmerge.")


    # --- Process Canonical ELO Comparisons ---
    canonical_comps = canonical_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    comps_to_move_to_local = []
    comps_to_keep_canonical = []
    moved_comp_count = 0
    kept_comp_count = 0

    logging.debug(f"Processing {len(canonical_comps)} canonical comparisons...")
    for comp in canonical_comps:
        pair = comp.get("pair", {})
        model_a = pair.get("test_model")
        model_b = pair.get("neighbor_model")

        if not model_a or not model_b:
            logging.warning(f"Skipping comparison with missing model names: {comp}")
            comps_to_keep_canonical.append(comp) # Keep malformed ones? Or discard? Keeping seems safer.
            kept_comp_count += 1
            continue

        # Determine the final destination of each model in the pair
        model_a_stays_canonical = model_a in remaining_canonical_models
        model_b_stays_canonical = model_b in remaining_canonical_models

        # *** Corrected Logic ***
        # A comparison stays canonical ONLY if BOTH models involved are staying canonical.
        # Otherwise, it moves to local (because at least one model involved is moving).
        if model_a_stays_canonical and model_b_stays_canonical:
            comps_to_keep_canonical.append(comp)
            kept_comp_count += 1
            logging.debug(f"Keeping comparison canonical (both models remain): {model_a} vs {model_b}")
        else:
            # If either model_a or model_b (or both) are NOT staying canonical (i.e., they are being unmerged),
            # the comparison moves to local.
            comps_to_move_to_local.append(comp)
            moved_comp_count += 1
            logging.debug(f"Moving comparison to local (at least one model unmerging): {model_a} vs {model_b}")


    logging.info(f"Identified {moved_comp_count} comparisons to move to local ELO.")
    logging.info(f"{kept_comp_count} comparisons will remain in canonical ELO.")

    # *** Requirement 1: Fail if 0 comparisons moved when models were selected ***
    if moved_comp_count == 0 and len(unmerged_model_names) > 0:
        logging.error(f"Attempted to unmerge {len(unmerged_model_names)} models, but found 0 relevant comparisons to move.")
        logging.error("This likely indicates an issue. Aborting unmerge to prevent data inconsistency.")
        # Provide more debug info if possible
        logging.error(f"Models selected for unmerge: {unmerged_model_names}")
        logging.error(f"Models remaining canonical: {remaining_canonical_models}")
        return False # Indicate failure

    # Update canonical ELO comparisons
    if "__metadata__" not in canonical_elo: canonical_elo["__metadata__"] = {}
    canonical_elo["__metadata__"]["global_pairwise_comparisons"] = comps_to_keep_canonical

    # Add comparisons to local ELO
    if "__metadata__" not in local_elo: local_elo["__metadata__"] = {}
    if "global_pairwise_comparisons" not in local_elo["__metadata__"]:
        local_elo["__metadata__"]["global_pairwise_comparisons"] = []
    # Avoid duplicates if running unmerge multiple times without cleaning? Check first.
    local_elo["__metadata__"].setdefault("global_pairwise_comparisons", []) \
            .extend(comps_to_move_to_local)
    added_count = len(comps_to_move_to_local)
    logging.info(f"Added {added_count} comparisons to local ELO metadata.")

    logging.info(f"Added {added_count} comparisons to local ELO metadata.")


    # --- Move Run Data and ELO Entries ---
    moved_run_keys = set()
    for candidate in selected_candidates:
        model_name = candidate["model_name"]
        # Find *all* run keys associated with this model in canonical_runs
        run_keys_for_model = [
            r_key for r_key, r_data in canonical_runs.items()
            if r_key != "__metadata__" and isinstance(r_data, dict) and
               r_data.get("model_name", r_data.get("test_model")) == model_name
        ]

        if not run_keys_for_model:
             logging.warning(f"No run entries found in canonical runs for model '{model_name}' during unmerge.")

        # Move run data
        for r_key_to_move in run_keys_for_model:
            if r_key_to_move in canonical_runs and r_key_to_move not in moved_run_keys:
                # Check for collision in local_runs
                if r_key_to_move in local_runs:
                    logging.warning(f"Run key '{r_key_to_move}' already exists in local runs! Overwriting during unmerge for model '{model_name}'.")
                local_runs[r_key_to_move] = canonical_runs[r_key_to_move]
                del canonical_runs[r_key_to_move]
                moved_run_keys.add(r_key_to_move)
                logging.info(f"Moved run data for {r_key_to_move} (model {model_name}) to local runs.")
            elif r_key_to_move in moved_run_keys:
                 logging.debug(f"Run key {r_key_to_move} already processed.")
            # else: # Should not happen if run_keys_for_model was built correctly
            #    logging.warning(f"Run key {r_key_to_move} (for model {model_name}) not found in canonical runs data during unmerge step (unexpected).")

        # Move ELO entry
        if model_name in canonical_elo:
            if model_name != "__metadata__": # Safety check
                 # Check for collision in local_elo
                if model_name in local_elo:
                     logging.warning(f"ELO entry for model '{model_name}' already exists in local ELO! Overwriting during unmerge.")
                local_elo[model_name] = canonical_elo[model_name]
                del canonical_elo[model_name]
                logging.info(f"Moved ELO entry for {model_name} to local ELO.")
        else:
            # This could happen if the model only existed in runs but not ELO, which is weird but possible.
            logging.warning(f"Model name '{model_name}' not found in canonical ELO data during unmerge (might indicate inconsistency).")

    return True # Indicate unmerging occurred successfully


# =========================================================================
# Delete Functionality
# =========================================================================

def find_delete_candidates(canonical_runs, canonical_elo):
    """Identifies models present in canonical files that can be deleted."""
    candidates = []
    model_names_in_elo = set(k for k in canonical_elo if k != "__metadata__")
    model_names_in_runs = set()
    run_key_map = defaultdict(list) # model_name -> list of run_keys

    for run_key, run_data in canonical_runs.items():
        if run_key == "__metadata__": continue
        if isinstance(run_data, dict):
            model_name = run_data.get("model_name", run_data.get("test_model"))
            if model_name:
                model_names_in_runs.add(model_name)
                run_key_map[model_name].append(run_key)

    # Consider models present in either file for deletion
    all_canonical_models = sorted(list(model_names_in_elo | model_names_in_runs))

    logging.info(f"Found {len(all_canonical_models)} unique model names in canonical data for potential deletion.")

    for model_name in all_canonical_models:
        elo_data = canonical_elo.get(model_name, {})
        elo_norm = elo_data.get("elo_norm", "N/A") if isinstance(elo_data, dict) else "N/A"
        # Find an associated run key (just for info, not strictly needed for deletion logic)
        run_key_example = run_key_map.get(model_name, ["N/A"])[0] # Just show the first one

        candidates.append({
            "model_name": model_name,
            "elo_norm": elo_norm,
            "run_key_example": run_key_example # For display purposes
        })

    return candidates


def delete_data(selected_models_to_delete, canonical_runs, canonical_elo):
    """Removes selected models and their associated data from canonical files."""
    if not selected_models_to_delete:
        return False # Indicate nothing was deleted

    deleted_model_names = set(c['model_name'] for c in selected_models_to_delete)
    logging.info(f"Preparing to delete {len(deleted_model_names)} models: {', '.join(deleted_model_names)}")

    # --- Remove ELO Entries ---
    deleted_elo_count = 0
    for model_name in deleted_model_names:
        if model_name in canonical_elo:
            if model_name != "__metadata__": # Safety check
                del canonical_elo[model_name]
                logging.info(f"Removed ELO entry for {model_name} from canonical ELO.")
                deleted_elo_count += 1
        else:
            logging.warning(f"Model name {model_name} not found in canonical ELO data during delete.")
    logging.info(f"Removed {deleted_elo_count} ELO entries.")

    # --- Remove Associated Run Data ---
    run_keys_to_delete = set()
    for run_key, run_data in canonical_runs.items():
        if run_key == "__metadata__":
            continue
        if isinstance(run_data, dict):
            model_name = run_data.get("model_name", run_data.get("test_model"))
            if model_name in deleted_model_names:
                run_keys_to_delete.add(run_key)

    deleted_run_count = 0
    for run_key in run_keys_to_delete:
        if run_key in canonical_runs:
            del canonical_runs[run_key]
            logging.info(f"Removed run data for {run_key} (model in {deleted_model_names}) from canonical runs.")
            deleted_run_count += 1
    logging.info(f"Removed {deleted_run_count} run entries.")


    # --- Remove Comparisons Involving Deleted Models ---
    if "__metadata__" in canonical_elo and "global_pairwise_comparisons" in canonical_elo["__metadata__"]:
        original_comps = canonical_elo["__metadata__"]["global_pairwise_comparisons"]
        comps_to_keep = []
        removed_comp_count = 0
        for comp in original_comps:
            pair = comp.get("pair", {})
            model_a = pair.get("test_model")
            model_b = pair.get("neighbor_model")
            if model_a not in deleted_model_names and model_b not in deleted_model_names:
                comps_to_keep.append(comp)
            else:
                removed_comp_count += 1
                logging.debug(f"Removing comparison involving deleted model: {model_a} vs {model_b}")

        canonical_elo["__metadata__"]["global_pairwise_comparisons"] = comps_to_keep
        logging.info(f"Removed {removed_comp_count} comparisons involving deleted models from canonical ELO.")
    else:
        logging.info("No comparisons found in canonical ELO metadata to filter.")

    return True # Indicate deletion occurred

# =========================================================================
# Common Functionality (Selection, ELO Recalc, Saving)
# =========================================================================

def select_models_from_list(candidates: List[Dict[str, Any]], action: str) -> List[Dict[str, Any]]:
    """Prompts the user to select models from a list for a given action (merge/delete/unmerge)."""
    if not candidates:
        return []

    print(f"\n--- Candidates for {action.capitalize()} ---")
    if action == "merge":
        for i, cand in enumerate(candidates):
             rubric_str = f"{cand['rubric_score']:.1f}" if isinstance(cand['rubric_score'], (int, float)) else cand['rubric_score']
             print(f"{i+1: >3}. {cand['model_name']} (Run: {cand['run_key']}, Rubric: {rubric_str}, ELO Norm: {cand['elo_norm']})")
    elif action == "delete":
         for i, cand in enumerate(candidates):
             print(f"{i+1: >3}. {cand['model_name']} (ELO Norm: {cand['elo_norm']}, Example Run: {cand.get('run_key_example', 'N/A')})")
    elif action == "unmerge":
         for i, cand in enumerate(candidates):
             run_keys_str = ', '.join(cand.get('run_keys', ['N/A']))
             print(f"{i+1: >3}. {cand['model_name']} (ELO Norm: {cand['elo_norm']}, Run(s): {run_keys_str})")
    else:
        logging.error(f"Unknown action '{action}' in select_models_from_list")
        return [] # Should not happen
    print("----------------------")

    while True:
        try:
            prompt = f"Enter numbers of models to {action} (e.g., 1,3,4), 'all', or 'none': "
            selection = input(prompt).strip().lower()
            if selection == 'none':
                return []
            if selection == 'all':
                return candidates # Return all candidate dicts

            selected_indices = set()
            parts = selection.split(',')
            for part in parts:
                part = part.strip()
                if not part: continue
                index = int(part) - 1
                if 0 <= index < len(candidates):
                    selected_indices.add(index)
                else:
                    print(f"Invalid number: {part}. Please enter numbers between 1 and {len(candidates)}.")
                    raise ValueError("Invalid index")

            # Return the selected candidate dicts
            return [candidates[i] for i in sorted(list(selected_indices))]

        except ValueError:
            print("Invalid input. Please use the specified format.")
        except Exception as e:
            print(f"An error occurred: {e}")


# --- ELO Re-calculation (Refactored) ------------------------------------
def _recalculate_elo_ratings(elo_data: Dict[str, Any]) -> bool:
    """
    Recalculate ELO ratings for the given ELO data structure (modifies in place).
    Uses the same comparison-filter pipeline as run_elo_analysis_eqbench3.
    Returns True on success, False on failure.
    """
    logging.info("Recalculating ELO ratings...")

    TS_DEFAULT_SIGMA = 350 / 3

    try:
        meta            = elo_data.get("__metadata__", {})
        all_comparisons = meta.get("global_pairwise_comparisons", [])
        all_models      = [m for m in elo_data if m != "__metadata__"]

        if not all_models:
            logging.warning("No models found in ELO data. Clearing ELO entries.")
            # Clear all model entries but keep metadata structure
            keys_to_del = [k for k in elo_data if k != "__metadata__"]
            for k in keys_to_del:
                del elo_data[k]
            if "global_pairwise_comparisons" in meta:
                 meta["global_pairwise_comparisons"] = [] # Clear comparisons too
            return True # Successful in the sense that the state is valid (empty)

        if not all_comparisons:
            logging.warning("No comparisons found; assigning default ELO to all models.")
            for m in all_models:
                elo_data[m] = {
                    "elo":      DEFAULT_ELO,
                    "elo_norm": DEFAULT_ELO,
                    "sigma":    TS_DEFAULT_SIGMA,
                    "ci_low":   DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high":  DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                    "ci_low_norm": DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA, # Add normalized defaults too
                    "ci_high_norm": DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                }
            return True

        # --- make sure every record has fraction_for_test etc. -------------
        changed = 0
        for comp in all_comparisons:
            if "error" not in comp and (
                "judge_response" in comp or "judge_responses" in comp
            ):
                before = comp.get("fraction_for_test")
                _recompute_comparison_stats(comp)      # adds / refreshes fields
                if comp.get("fraction_for_test") != before:
                    changed += 1
        logging.info(f"Recomputed stats for {changed} comparisons")

        # ---------- identical pipeline to main ELO run ------------------
        initial_snapshot = {
            m: elo_data.get(m, {}).get("elo", DEFAULT_ELO) for m in all_models
        }
        comps_for_solver = get_solver_comparisons(
            all_comparisons,
            initial_snapshot,
            rank_window=RANK_WINDOW,
        )
        logging.info(f"Kept {len(comps_for_solver)}/{len(all_comparisons)} comparisons after rank-window filter.")

        if not comps_for_solver:
            logging.warning("No comparisons after rank-window filter. Assigning default ELO to remaining models.")
            # Assign defaults only to models still present
            current_models = set(elo_data.keys()) - {"__metadata__"}
            for m in current_models:
                 elo_data[m] = {
                    "elo":      DEFAULT_ELO,
                    "elo_norm": DEFAULT_ELO,
                    "sigma":    TS_DEFAULT_SIGMA,
                    "ci_low":   DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high":  DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                    "ci_low_norm": DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high_norm": DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                 }
            return True


        # Models to solve for: union of models in blob and models in filtered comparisons
        models_in_blob = set(all_models)
        models_in_filtered_comps = models_in_comparisons(comps_for_solver)
        models_for_solver_set = models_in_blob | models_in_filtered_comps
        models_for_solver = sorted(list(models_for_solver_set))

        # Ensure we only calculate for models actually remaining in elo_data keys
        models_for_solver = [m for m in models_for_solver if m in elo_data]

        if not models_for_solver:
             logging.warning("No models left to solve for after filtering. Clearing ELO data.")
             keys_to_del = [k for k in elo_data if k != "__metadata__"]
             for k in keys_to_del:
                 del elo_data[k]
             if "global_pairwise_comparisons" in meta:
                  meta["global_pairwise_comparisons"] = []
             return True

        logging.info(f"Solving ELO for {len(models_for_solver)} models.")

        mu_map, _ = solve_with_trueskill(
            models_for_solver,
            comps_for_solver,
            {m: DEFAULT_ELO for m in models_for_solver},
            debug=False,
            use_fixed_initial_ratings=True,
            bin_size=WIN_MARGIN_BIN_SIZE,
            return_sigma=True,
        )

        # Recalc again just for sigma, using smaller bin size
        _, sigma_map = solve_with_trueskill(
            models_for_solver,
            comps_for_solver,
            {m: DEFAULT_ELO for m in models_for_solver},
            debug=False,
            use_fixed_initial_ratings=True,
            bin_size=WIN_MARGIN_BIN_SIZE_FOR_CI,
            return_sigma=True,
        )

        mu_norm_map = normalize_elo_scores(mu_map)

        # ---------- write results back (in place) -----------------------
        all_remaining_models = set(elo_data.keys()) - {"__metadata__"}
        calculated_models = set(models_for_solver)

        for m in all_remaining_models:
            if m in calculated_models:
                mu_raw  = mu_map.get(m, DEFAULT_ELO)
                sigma   = sigma_map.get(m, TS_DEFAULT_SIGMA)
                ci_low  = mu_raw - 1.96 * sigma
                ci_high = mu_raw + 1.96 * sigma
                elo_data[m] = {
                    "elo":       round(mu_raw, 2),
                    "elo_norm":  round(mu_norm_map.get(m, DEFAULT_ELO), 2),
                    "sigma":     round(sigma, 2),
                    "ci_low":    round(ci_low, 2),
                    "ci_high":   round(ci_high, 2),
                }
            else: # Model exists but had no comparisons after filtering
                 logging.warning(f"Model '{m}' had no comparisons after filtering, assigning default ELO.")
                 elo_data[m] = {
                    "elo":      DEFAULT_ELO,
                    "elo_norm": DEFAULT_ELO,
                    "sigma":    TS_DEFAULT_SIGMA,
                    "ci_low":   DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high":  DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                    "ci_low_norm": DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high_norm": DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                 }


        # ---------- normalise CI bounds ---------------------------------
        models_with_valid_elo = {m for m, d in elo_data.items() if m != "__metadata__" and "elo" in d}

        if models_with_valid_elo:
            raw_plus_bounds = {}
            raw_plus_bounds.update({m: d["elo"] for m, d in elo_data.items() if m in models_with_valid_elo})
            raw_plus_bounds.update({f"{m}__low":  d["ci_low"]  for m, d in elo_data.items() if m in models_with_valid_elo})
            raw_plus_bounds.update({f"{m}__high": d["ci_high"] for m, d in elo_data.items() if m in models_with_valid_elo})

            if raw_plus_bounds: # Check if there's anything to normalize
                norm_bounds = normalize_elo_scores(raw_plus_bounds)
                for m, d in elo_data.items():
                    if m in models_with_valid_elo:
                        # Use elo_norm as default if bound normalization fails for some reason
                        d["ci_low_norm"]  = round(norm_bounds.get(f"{m}__low",  d.get("elo_norm", DEFAULT_ELO)), 2)
                        d["ci_high_norm"] = round(norm_bounds.get(f"{m}__high", d.get("elo_norm", DEFAULT_ELO)), 2)
            else:
                 logging.warning("No valid ELO scores found to normalize CI bounds.")
        else:
             logging.warning("No models with valid ELO scores after recalculation.")

        # Update timestamp in metadata
        elo_data.setdefault("__metadata__", {})
        elo_data["__metadata__"]["last_updated"] = datetime.now(timezone.utc).isoformat()

        logging.info("ELO recalculation complete.")
        return True

    except Exception:
        logging.error("ELO recalculation failed", exc_info=True)
        return False


# --- Atomic Saving ------------------------------------------------------

def _atomic_multi_save(path_to_data: Dict[str, Any]) -> bool:
    """
    Transactionally write several JSON blobs. If anything fails,
    originals are untouched and all temp files are removed.
    """
    temps = {}
    success = True
    written_temps = []

    try:
        # ---- 1. write each temp file ---------------------------------------
        for final_path_str, data in path_to_data.items():
            final_path = Path(final_path_str)
            dir_name = final_path.parent
            stem = final_path.stem # Includes .json if extension is .json.gz
            suffix = final_path.suffix # .gz or .json

            # Handle double extensions like .json.gz correctly
            if suffix == ".gz" and stem.endswith(".json"):
                stem = stem[:-5] # Remove .json
                ext = ".json.gz"
            else:
                ext = suffix

            tmp_name = f"{stem}.tmp.{uuid.uuid4().hex}{ext}"
            tmp_path = dir_name / tmp_name

            dir_name.mkdir(parents=True, exist_ok=True) # Ensure directory exists

            if not save_json_file(data, str(tmp_path)):
                logging.error(f"Failed to write temporary file: {tmp_path}")
                success = False
                # Attempt to remove the failed temp file if it exists
                if tmp_path.exists():
                    try: tmp_path.unlink()
                    except OSError as e: logging.warning(f"Could not remove failed temp file {tmp_path}: {e}")
                break # Stop trying to write more temps
            else:
                temps[str(final_path)] = str(tmp_path)
                written_temps.append(str(tmp_path)) # Keep track of successfully written temps

        # ---- 2. rename temps onto finals (only if all temps written) -----
        if success:
            try:
                for final_path, tmp_path in temps.items():
                    os.replace(tmp_path, final_path) # os.replace is atomic on most systems
                    written_temps.remove(tmp_path) # Remove from list if successfully renamed
                logging.debug("All temporary files successfully renamed.")
                return True
            except Exception as e:
                logging.error(f"Multi-save rename phase failed: {e}", exc_info=True)
                success = False
                # NOTE: If os.replace is truly atomic, no finals were touched.
                # Temps that were *not* successfully renamed are still in written_temps.

    except Exception as e:
        logging.error(f"Error during atomic save process (before rename): {e}", exc_info=True)
        success = False

    finally:
        # ---- Cleanup: Remove any remaining temp files ----
        if written_temps:
             logging.warning(f"Cleaning up {len(written_temps)} temporary files due to failed save.")
             for t_path_str in written_temps:
                 t_path = Path(t_path_str)
                 if t_path.exists():
                     try:
                         t_path.unlink()
                         logging.debug(f"Removed temp file: {t_path}")
                     except OSError as e:
                         logging.error(f"Failed to remove temporary file {t_path}: {e}")
                 else:
                      logging.warning(f"Expected temp file {t_path} not found for cleanup.")

    return success


# ─────────────────────────────────────────────────────────────────────
# Recalc Action Specific Logic (Kept separate as it only saves canonical)
# ─────────────────────────────────────────────────────────────────────
def _refresh_comparison_fields(comps: List[Dict[str, Any]]) -> None:
    """Populate margin / fraction fields so every record is usable."""
    changed = 0
    for c in comps:
        if "error" not in c and (
            "judge_response" in c or "judge_responses" in c
        ):
            before = c.get("fraction_for_test")
            _recompute_comparison_stats(c)
            if c.get("fraction_for_test") != before:
                changed += 1
    logging.info(f"[recalc] refreshed stats for {changed} comparisons")

# ─────────────────────────────────────────────────────────────────────
def _dual_solve_with_window(all_models: Set[str],
                            comps: List[Dict[str, Any]]
                           ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Replicates benchmark logic: rough solve → rank-window → dual solve."""

    # --- 0. quick rough solve to get a ladder --------------------------
    init = {m: DEFAULT_ELO for m in all_models}
    rough_mu, _ = solve_with_trueskill(
        list(all_models), comps, init,
        debug=False, use_fixed_initial_ratings=True,
        bin_size=WIN_MARGIN_BIN_SIZE, return_sigma=True)

    # --- 1. filter comparisons with the same ±8 window -----------------
    filtered = get_solver_comparisons(
        comps, rough_mu, rank_window=RANK_WINDOW)
    logging.info(f"[recalc] kept {len(filtered)}/{len(comps)} comparisons after rank-window")

    if not filtered:
        # Return empty maps if no comparisons left
        logging.warning("All comparisons filtered out – cannot solve.")
        return {}, {}


    # --- 2. final μ  (bin 20) -----------------------------------------
    models_for_solve = sorted(list(all_models | models_in_comparisons(filtered)))
    init_final = {m: DEFAULT_ELO for m in models_for_solve}

    mu_map, _ = solve_with_trueskill(
        models_for_solve, filtered, init_final,
        debug=False, use_fixed_initial_ratings=True,
        bin_size=WIN_MARGIN_BIN_SIZE, return_sigma=True)

    # --- 3. σ / CI  (bin 5) -------------------------------------------
    _, sigma_map = solve_with_trueskill(
        models_for_solve, filtered, init_final,
        debug=False, use_fixed_initial_ratings=True,
        bin_size=WIN_MARGIN_BIN_SIZE_FOR_CI, return_sigma=True)

    return mu_map, sigma_map

# ─────────────────────────────────────────────────────────────────────
def _write_back_recalc(elo_path: Path, old_blob: Dict[str, Any],
                       mu: Dict[str, float], sigma: Dict[str, float]) -> bool:
    """Writes the recalculated ELO data back to the specified file."""

    ts_env_sigma = 350 / 3
    new_blob     = copy.deepcopy(old_blob) # Start with old metadata, comparisons etc.

    # Clear existing model entries before writing new ones
    models_to_clear = [k for k in new_blob if k != "__metadata__"]
    for k in models_to_clear:
        del new_blob[k]

    if not mu: # Handle case where solving failed or yielded no results
        logging.warning("[recalc] No ELO scores generated. Writing empty model data.")
        # Keep metadata, but no model entries
    else:
        # normalise μ and CI bounds exactly like benchmark
        norm_mu  = normalize_elo_scores(mu)

        raw_bounds = {}
        for m in mu:
            sig = sigma.get(m, ts_env_sigma)
            raw_bounds[f"{m}_low"] = mu[m] - 1.96 * sig
            raw_bounds[f"{m}_hi"]  = mu[m] + 1.96 * sig
        norm_bounds = normalize_elo_scores(raw_bounds)

        for m in mu:
            sig     = sigma.get(m, ts_env_sigma)
            ci_low  = raw_bounds.get(f"{m}_low", mu[m]) # Default to mu if bound missing
            ci_high = raw_bounds.get(f"{m}_hi", mu[m])
            norm_ci_low = norm_bounds.get(f"{m}_low", norm_mu[m])
            norm_ci_high = norm_bounds.get(f"{m}_hi", norm_mu[m])

            new_blob[m] = {
                "elo":          round(mu[m],            2),
                "elo_norm":     round(norm_mu[m],       2),
                "sigma":        round(sig,              2),
                "ci_low":       round(ci_low,           2),
                "ci_high":      round(ci_high,          2),
                "ci_low_norm":  round(norm_ci_low, 2),
                "ci_high_norm": round(norm_ci_high,  2),
            }

    new_blob.setdefault("__metadata__", {})
    new_blob["__metadata__"]["last_updated"] = datetime.now(
        timezone.utc).isoformat()

    # Use atomic save for single file write as well
    if _atomic_multi_save({str(elo_path): new_blob}):
        logging.info(f"[recalc] wrote refreshed ratings to {elo_path}")
        return True
    else:
        logging.error(f"[recalc] FAILED to save {elo_path}")
        return False

# ─────────────────────────────────────────────────────────────────────
def action_recalc(args, canonical_elo_data):
    """Full re-solve + overwrite canonical file so it matches pipeline."""
    logging.info("Starting full recalculation of canonical ELO...")
    canonical_elo_copy = copy.deepcopy(canonical_elo_data) # Work on a copy

    comps = canonical_elo_copy.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    if not comps:
        logging.error("No comparisons found in canonical data – aborting recalc.")
        return False

    # 1. refresh per-comparison stats (modifies copy in place)
    _refresh_comparison_fields(comps)

    # 2. union of models: those with comparisons + those already in blob
    models_comp = models_in_comparisons(comps)
    models_blob = {m for m in canonical_elo_copy if m != "__metadata__"}
    all_models  = models_comp.union(models_blob)
    if not all_models:
        logging.warning("[recalc] No models found in data after loading. Aborting.")
        return False
    logging.info(f"[recalc] solving for {len(all_models)} models")

    # 3. dual solve with rank-window logic
    try:
        mu_map, sigma_map = _dual_solve_with_window(all_models, comps)
    except Exception as e:
        logging.error(f"[recalc] Solving failed: {e}", exc_info=True)
        return False

    # 4. write back (overwrites original file path using the copy's data)
    if not _write_back_recalc(Path(args.canonical_elo), canonical_elo_copy, mu_map, sigma_map):
        return False # Write back failed

    logging.info("Recalculation action completed successfully.")
    return True


# =========================================================================
# Main Execution
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge/Unmerge local runs/ELO with canonical files OR delete models from canonical files OR recalculate canonical ELO."
    )
    parser.add_argument(
        "--action",
        choices=["merge", "delete", "recalc", "unmerge"], # Added unmerge
        required=True, # Make action required
        help="Action to perform: 'merge' local->canonical, 'unmerge' canonical->local, 'delete' from canonical, 'recalc' canonical.",
    )
    parser.add_argument(
        "--local-runs",
        default=C.DEFAULT_LOCAL_RUNS_FILE,
        help="Path to the local runs JSON file (used for merge/unmerge).",
    )
    parser.add_argument(
        "--local-elo",
        default=C.DEFAULT_LOCAL_ELO_FILE,
        help="Path to the local ELO JSON file (used for merge/unmerge).",
    )
    parser.add_argument(
        "--canonical-runs",
        default=C.CANONICAL_LEADERBOARD_RUNS_FILE,
        help="Path to the canonical runs file (source/target).",
    )
    parser.add_argument(
        "--canonical-elo",
        default=C.CANONICAL_LEADERBOARD_ELO_FILE,
        help="Path to the canonical ELO file (source/target).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically confirm the selected action without prompting.",
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity level.",
    )

    args = parser.parse_args()
    setup_merge_logging(args.verbosity)

    # --- Load Data ---
    logging.info(f"Loading canonical runs: {args.canonical_runs}")
    canonical_runs = load_json_file(args.canonical_runs)
    logging.info(f"Loading canonical ELO: {args.canonical_elo}")
    canonical_elo = load_json_file(args.canonical_elo)

    local_runs = {}
    local_elo = {}
    # Load local files if needed for merge or unmerge
    if args.action in ["merge", "unmerge"]:
        logging.info(f"Loading local runs: {args.local_runs}")
        local_runs = load_json_file(args.local_runs)
        logging.info(f"Loading local ELO: {args.local_elo}")
        local_elo = load_json_file(args.local_elo)
        if not isinstance(local_runs, dict) or not isinstance(local_elo, dict):
             logging.error("Failed to load local data files correctly for %s. Exiting.", args.action)
             sys.exit(1)

    if not isinstance(canonical_runs, dict) or not isinstance(canonical_elo, dict):
        logging.error("Failed to load canonical data files correctly. Exiting.")
        sys.exit(1)

    # Ensure metadata dicts exist
    canonical_runs.setdefault("__metadata__", {})
    canonical_elo.setdefault("__metadata__", {})
    if args.action in ["merge", "unmerge"]:
        local_runs.setdefault("__metadata__", {})
        local_elo.setdefault("__metadata__", {})

    files_to_save = None
    action_performed = False

    # --- Perform Action ---
    if args.action == "merge":
        candidates = find_merge_candidates(local_runs, local_elo, canonical_runs, canonical_elo)
        if not candidates:
            logging.info("No suitable candidates found for merging.")
            sys.exit(0)

        selected_candidates = select_models_from_list(candidates, "merge")
        if not selected_candidates:
            logging.info("No candidates selected for merging.")
            sys.exit(0)

        print("\n--- Summary of Merge ---")
        print(f"Models to merge: {', '.join(c['model_name'] for c in selected_candidates)}")
        print(f"Local Runs File:      {args.local_runs} (will be modified)")
        print(f"Local ELO File:       {args.local_elo} (will be modified)")
        print(f"Canonical Runs File:  {args.canonical_runs} (will be modified)")
        print(f"Canonical ELO File:   {args.canonical_elo} (will be modified)")
        print("------------------------")

        if not (args.yes or input("Proceed with merge? (y/n): ").strip().lower().startswith("y")):
            logging.info("Merge cancelled by user.")
            sys.exit(0)

        logging.info("Starting merge process…")
        local_runs_copy = copy.deepcopy(local_runs)
        local_elo_copy = copy.deepcopy(local_elo)
        canonical_runs_copy = copy.deepcopy(canonical_runs)
        canonical_elo_copy = copy.deepcopy(canonical_elo)

        if not merge_data(selected_candidates, local_runs_copy, local_elo_copy, canonical_runs_copy, canonical_elo_copy):
            logging.error("No data moved during merge; aborting.")
            sys.exit(1)

        logging.info("Recalculating canonical ELO after merge...")
        if not _recalculate_elo_ratings(canonical_elo_copy): # Use refactored function
            logging.error("Canonical ELO recalculation failed; no files will be written.")
            sys.exit(1)
        # Note: Local ELO doesn't need recalculation after merge as only removed items

        logging.info("Saving modified files…")
        files_to_save = {
            args.canonical_runs: canonical_runs_copy,
            args.canonical_elo : canonical_elo_copy,
            args.local_runs    : local_runs_copy,
            args.local_elo     : local_elo_copy,
        }
        action_performed = True

    elif args.action == "unmerge":
        candidates = find_unmerge_candidates(canonical_runs, canonical_elo)
        if not candidates:
            logging.info("No suitable candidates found for unmerging.")
            sys.exit(0)

        selected_candidates = select_models_from_list(candidates, "unmerge")
        if not selected_candidates:
            logging.info("No candidates selected for unmerging.")
            sys.exit(0)

        print("\n--- Summary of Unmerge ---")
        print(f"Models to unmerge: {', '.join(c['model_name'] for c in selected_candidates)}")
        print(f"Local Runs File:      {args.local_runs} (will be modified)")
        print(f"Local ELO File:       {args.local_elo} (will be modified)")
        print(f"Canonical Runs File:  {args.canonical_runs} (will be modified)")
        print(f"Canonical ELO File:   {args.canonical_elo} (will be modified)")
        print("--------------------------")

        if not (args.yes or input("Proceed with unmerge? (y/n): ").strip().lower().startswith("y")):
            logging.info("Unmerge cancelled by user.")
            sys.exit(0)

        logging.info("Starting unmerge process…")
        local_runs_copy = copy.deepcopy(local_runs)
        local_elo_copy = copy.deepcopy(local_elo)
        canonical_runs_copy = copy.deepcopy(canonical_runs)
        canonical_elo_copy = copy.deepcopy(canonical_elo)

        if not unmerge_data(selected_candidates, local_runs_copy, local_elo_copy, canonical_runs_copy, canonical_elo_copy):
            logging.error("No data moved during unmerge; aborting.")
            sys.exit(1)

        logging.info("Recalculating canonical ELO after unmerge...")
        recalc_canon_ok = _recalculate_elo_ratings(canonical_elo_copy)
        logging.info("Recalculating local ELO after unmerge...")
        recalc_local_ok = _recalculate_elo_ratings(local_elo_copy)

        if not (recalc_canon_ok and recalc_local_ok):
            logging.error("ELO recalculation failed for one or both files; no files will be written.")
            sys.exit(1)

        logging.info("Saving modified files…")
        files_to_save = {
            args.canonical_runs: canonical_runs_copy,
            args.canonical_elo : canonical_elo_copy,
            args.local_runs    : local_runs_copy,
            args.local_elo     : local_elo_copy,
        }
        action_performed = True

    elif args.action == "delete":
        candidates = find_delete_candidates(canonical_runs, canonical_elo)
        if not candidates:
            logging.info("No models found in canonical files to delete.")
            sys.exit(0)

        selected_models = select_models_from_list(candidates, "delete")
        if not selected_models:
            logging.info("No models selected for deletion.")
            sys.exit(0)

        print("\n--- Summary of Deletion ---")
        print(f"Models to delete: {', '.join(c['model_name'] for c in selected_models)}")
        print(f"Canonical Runs File:  {args.canonical_runs} (will be modified)")
        print(f"Canonical ELO File:   {args.canonical_elo} (will be modified)")
        print("---------------------------")

        if not (args.yes or input("Proceed with deletion? (y/n): ").strip().lower().startswith("y")):
            logging.info("Deletion cancelled by user.")
            sys.exit(0)

        logging.info("Starting deletion process…")
        canonical_runs_copy = copy.deepcopy(canonical_runs)
        canonical_elo_copy = copy.deepcopy(canonical_elo)

        if not delete_data(selected_models, canonical_runs_copy, canonical_elo_copy):
            logging.error("No data removed during deletion; aborting.")
            sys.exit(1)

        logging.info("Recalculating canonical ELO after deletion...")
        if not _recalculate_elo_ratings(canonical_elo_copy): # Use refactored function
            logging.error("Canonical ELO recalculation failed; no files will be written.")
            sys.exit(1)

        logging.info("Saving modified canonical files…")
        files_to_save = {
            args.canonical_runs: canonical_runs_copy,
            args.canonical_elo : canonical_elo_copy,
        }
        action_performed = True

    elif args.action == "recalc":
        # Recalc action handles its own saving internally via _write_back_recalc
        if not action_recalc(args, canonical_elo):
             logging.error("Recalculation action failed.")
             sys.exit(1)
        # No need to set files_to_save or action_performed here
        logging.info("Recalculation action completed.")
        sys.exit(0) # Exit after recalc action

    else:
        # Should be caught by argparse choices
        logging.error(f"Invalid action specified: {args.action}")
        sys.exit(1)

    # --- Perform Save (for merge, unmerge, delete) ---
    if action_performed and files_to_save:
        if _atomic_multi_save(files_to_save):
            logging.info(f"{args.action.capitalize()} process completed successfully.")
        else:
            logging.error(f"{args.action.capitalize()} aborted – atomic save failed, no files should have been overwritten.")
            sys.exit(1)
    elif action_performed and not files_to_save:
         logging.error("Action was marked as performed, but no files were set to be saved. This indicates an internal logic error.")
         sys.exit(1)
    # else: action was not performed (e.g., user cancelled) or was 'recalc' which saves itself.


if __name__ == "__main__":
    main()