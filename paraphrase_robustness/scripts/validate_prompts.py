#!/usr/bin/env python3
"""Validate a scenario prompts file: each scenario has exactly four prompts.

Standalone parser (mirrors core.benchmark.parse_scenario_prompts) so this script
does not import the full benchmark stack.
"""
from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional


def parse_scenario_prompts(file_path: str) -> Dict[str, List[str]]:
    scenarios: Dict[str, List[str]] = {}
    current_scenario_id: Optional[str] = None
    current_prompts_for_scenario: List[str] = []
    current_prompt_lines: List[str] = []
    in_prompt_content = False

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            scenario_match = re.match(r"^########\s*(\S+)", line)
            prompt_match = re.match(r"^#######\s*Prompt(\d+)", line)

            if scenario_match:
                if current_prompt_lines:
                    prompt_text = "\n".join(current_prompt_lines).strip()
                    if prompt_text:
                        current_prompts_for_scenario.append(prompt_text)
                    current_prompt_lines = []
                if current_scenario_id and current_prompts_for_scenario:
                    scenarios[current_scenario_id] = current_prompts_for_scenario
                current_scenario_id = scenario_match.group(1)
                current_prompts_for_scenario = []
                in_prompt_content = False
                continue

            if prompt_match:
                if current_scenario_id is None:
                    continue
                if current_prompt_lines:
                    prompt_text = "\n".join(current_prompt_lines).strip()
                    if prompt_text:
                        current_prompts_for_scenario.append(prompt_text)
                current_prompt_lines = []
                in_prompt_content = True
                continue

            if current_scenario_id and in_prompt_content:
                current_prompt_lines.append(raw_line.rstrip("\n\r"))

        if current_prompt_lines:
            prompt_text = "\n".join(current_prompt_lines).strip()
            if prompt_text:
                current_prompts_for_scenario.append(prompt_text)
        if current_scenario_id and current_prompts_for_scenario:
            scenarios[current_scenario_id] = current_prompts_for_scenario

    return scenarios


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
