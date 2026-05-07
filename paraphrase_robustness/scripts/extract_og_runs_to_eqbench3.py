#!/usr/bin/env python3
"""
Copy paraphrase pilot runs from paraphrase_robustness/results/*.json into eqbench3_runs.json,
keeping only scenario_ids whose manifest entry has framing == \"og\" (verbatim originals).

Runs with no matching tasks (e.g. framing-only pro/anti pilots, wording32 wa/wb/wc) are skipped.
Top-level ``results`` on each copied run is dropped because it reflects the full scenario set.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.file_io import load_json_file, save_json_file  # noqa: E402
import utils.constants as C  # noqa: E402


def _og_ids_from_manifest(manifest_path: Path) -> Set[str]:
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    out = {
        v["scenario_id"]
        for v in m.get("variants", [])
        if isinstance(v, dict) and v.get("framing") == "og"
    }
    return out


def _filter_run(
    run_dict: Dict[str, Any], og_ids: Set[str], source_file: str
) -> Optional[Tuple[Dict[str, Any], int]]:
    out = copy.deepcopy(run_dict)
    st = out.get("scenario_tasks") or {}
    new_st: Dict[str, Dict[str, Any]] = {}
    kept = 0
    for iter_k, scen_map in st.items():
        if not isinstance(scen_map, dict):
            continue
        filt = {sid: task for sid, task in scen_map.items() if sid in og_ids}
        if filt:
            new_st[str(iter_k)] = filt
            kept += len(filt)
    if not new_st:
        return None
    out["scenario_tasks"] = new_st
    out.pop("results", None)
    prov = {
        "extracted_from": source_file,
        "filter": "manifest framing == og",
        "tasks_kept": kept,
    }
    existing = out.get("og_extract_provenance")
    if isinstance(existing, dict):
        existing.update(prov)
        out["og_extract_provenance"] = existing
    else:
        out["og_extract_provenance"] = prov
    return out, kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=ROOT / "paraphrase_robustness" / "results" / "paraphrase_runs.json",
        help="Paraphrase runs JSON to read.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "paraphrase_robustness" / "manifest.json",
        help="Manifest listing variants (og / wa / wb).",
    )
    ap.add_argument(
        "--target",
        type=Path,
        default=ROOT / C.DEFAULT_LOCAL_RUNS_FILE,
        help=f"Merge destination (default: {C.DEFAULT_LOCAL_RUNS_FILE}).",
    )
    ap.add_argument(
        "--key-suffix",
        default="_og-only",
        help="If a run_key already exists in the target file, append this suffix.",
    )
    args = ap.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    if not args.source.is_file():
        raise SystemExit(f"Source runs not found: {args.source}")

    og_ids = _og_ids_from_manifest(args.manifest)
    if not og_ids:
        raise SystemExit("No framing==og scenario_ids in manifest.")

    src = load_json_file(str(args.source))
    if not isinstance(src, dict):
        raise SystemExit("Source did not load as a JSON object.")

    tgt_path = args.target
    tgt: Dict[str, Any] = {}
    if tgt_path.is_file():
        loaded = load_json_file(str(tgt_path))
        if isinstance(loaded, dict):
            tgt = loaded

    try:
        source_tag = str(args.source.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        source_tag = str(args.source)
    merged = 0
    skipped = 0
    for run_key, run_data in src.items():
        if not isinstance(run_data, dict):
            continue
        filtered = _filter_run(run_data, og_ids, source_tag)
        if filtered is None:
            print(f"skip (no og tasks): {run_key}")
            skipped += 1
            continue
        new_run, n_tasks = filtered
        dest_key = str(run_key)
        if dest_key in tgt:
            dest_key = f"{run_key}{args.key_suffix}"
        if dest_key in tgt:
            raise SystemExit(
                f"Key collision after suffix: {dest_key!r} already in {tgt_path}"
            )
        tgt[dest_key] = new_run
        print(f"merged {n_tasks} og tasks as {dest_key!r} (from {run_key!r})")
        merged += 1

    if not save_json_file(tgt, str(tgt_path)):
        raise SystemExit(f"Failed to write {tgt_path}")
    print(f"Wrote {tgt_path} ({len(tgt)} total run keys); merged {merged}, skipped {skipped}.")


if __name__ == "__main__":
    main()
