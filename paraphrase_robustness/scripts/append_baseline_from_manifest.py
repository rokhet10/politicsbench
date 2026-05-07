#!/usr/bin/env python3
"""Strip any existing ``######## BASELINE_QUESTIONS`` trailer, then append from manifest.

Reads ``baseline_questions`` from ``paraphrase_robustness/manifest.json`` by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.scenario_prompts import split_baseline_questions_block  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prompts",
        type=Path,
        default=Path("scenario_prompts.txt"),
        help="Scenario file to update (default: repo-root scenario_prompts.txt)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("paraphrase_robustness/manifest.json"),
        help="Manifest containing baseline_questions",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite --prompts)",
    )
    args = ap.parse_args()
    text = args.prompts.read_text(encoding="utf-8")
    body, _old = split_baseline_questions_block(text)
    mf = json.loads(args.manifest.read_text(encoding="utf-8"))
    bq = mf.get("baseline_questions")
    if not isinstance(bq, dict):
        print("ERROR: manifest has no baseline_questions object", file=sys.stderr)
        return 1
    blob = json.dumps(bq, indent=2, ensure_ascii=False)
    out_text = body.rstrip() + "\n\n######## BASELINE_QUESTIONS\n" + blob + "\n"
    dest = args.out or args.prompts
    dest.write_text(out_text, encoding="utf-8")
    print(f"Wrote {dest} ({len(bq)} baseline keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
