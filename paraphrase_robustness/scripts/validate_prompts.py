#!/usr/bin/env python3
"""Validate a scenario prompts file: each scenario has exactly four prompts.

Uses ``utils.scenario_prompts.parse_scenario_prompts`` (handles optional
``######## BASELINE_QUESTIONS`` JSON trailer).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on PYTHONPATH when run as ``python3 paraphrase_robustness/scripts/validate_prompts.py``
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.scenario_prompts import parse_scenario_prompts  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: validate_prompts.py <scenario_prompts.txt>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    scenarios = parse_scenario_prompts(path)
    if not scenarios:
        print(f"ERROR: no scenarios parsed from {path}", file=sys.stderr)
        return 1
    bad = []
    for sid, prompts in sorted(scenarios.items(), key=lambda x: str(x[0])):
        if len(prompts) != 4:
            bad.append((sid, len(prompts)))
    if bad:
        for sid, n in bad:
            print(f"ERROR: scenario {sid!r} has {n} prompts, expected 4", file=sys.stderr)
        return 1
    print(f"OK: {len(scenarios)} scenarios, each with 4 prompts — {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
