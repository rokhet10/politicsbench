#!/usr/bin/env python3
"""
Copy selected run_key entries from judge_agreement/results/judge_agreement_runs.json
into trait_activation/results/trait_activation_runs.json for a self-contained
trait-activation analysis (no judge API calls; numbers only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.file_io import load_json_file, save_json_file  # noqa: E402

DEFAULT_SRC = "judge_agreement/results/judge_agreement_runs.json"
DEFAULT_OUT = "trait_activation/results/trait_activation_runs.json"
DEFAULT_KEYS = [
    "judge_agree_gemini_gemini-judge-agreement-og",
    "judge_agree_gemini_gemini-judge-agreement-lessctx",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--source", type=Path, default=Path(DEFAULT_SRC))
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument(
        "--keys",
        nargs="*",
        default=DEFAULT_KEYS,
        help="Run keys to copy (default: OG + lessctx judge-agreement runs).",
    )
    args = ap.parse_args()

    data = load_json_file(str(args.source))
    if not isinstance(data, dict):
        print("Source is not a JSON object.", file=sys.stderr)
        return 1

    out: Dict[str, Any] = {}
    missing: List[str] = []
    for k in args.keys:
        if k not in data:
            missing.append(k)
        else:
            out[k] = data[k]
    if missing:
        print("Missing run keys in source:", ", ".join(missing), file=sys.stderr)
        return 1
    if not out:
        print("No runs to write.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not save_json_file(out, str(args.out)):
        return 1
    print(f"Wrote {args.out} ({len(out)} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
