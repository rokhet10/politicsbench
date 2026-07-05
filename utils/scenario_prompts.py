"""Parse scenario prompt files: `########` scenarios, `####### PromptN` stanzas.

Optional trailing block (after all scenarios)::

    ######## BASELINE_QUESTIONS
    { "1": "...", "2": "..." }

The JSON object maps base_id -> blanket baseline question string (same keys as ``manifest.json``).
"""
from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    "split_baseline_questions_block",
    "parse_scenario_prompts_from_text",
    "parse_scenario_prompts",
    "load_scenario_prompts_and_baseline",
]

_BASELINE_HEADER = re.compile(r"(?m)^########\s+BASELINE_QUESTIONS\s*\n")


def split_baseline_questions_block(full_text: str) -> Tuple[str, Dict[str, str]]:
    """
    If ``full_text`` contains a ``######## BASELINE_QUESTIONS`` section, return the
    scenario text before it and the parsed JSON object. Otherwise return
    ``(full_text, {})``.

    Raises:
        ValueError: if the section exists but JSON is missing or invalid.
    """
    m = _BASELINE_HEADER.search(full_text)
    if not m:
        return full_text, {}
    body = full_text[: m.start()].rstrip()
    raw_json = full_text[m.end() :].strip()
    if not raw_json:
        return body, {}
    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in BASELINE_QUESTIONS section: {e}"
        ) from e
    if not isinstance(obj, dict):
        raise ValueError("BASELINE_QUESTIONS JSON must be a JSON object")
    out = {str(k): str(v).strip() for k, v in obj.items() if str(v).strip()}
    return body, out


def parse_scenario_prompts_from_text(
    text: str, source_label: str = "string"
) -> Dict[str, List[str]]:
    """Parse scenario blocks from in-memory text (no baseline section — strip first)."""
    scenarios: Dict[str, List[str]] = {}
    current_scenario_id: Optional[str] = None
    current_prompts_for_scenario: List[str] = []
    current_prompt_lines: List[str] = []
    in_prompt_content = False

    f = io.StringIO(text)
    for line_num, raw_line in enumerate(f, 1):
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
                logging.debug(
                    "Stored scenario %s with %s prompts.",
                    current_scenario_id,
                    len(current_prompts_for_scenario),
                )

            current_scenario_id = scenario_match.group(1)
            current_prompts_for_scenario = []
            in_prompt_content = False
            logging.debug(
                "Starting parse for scenario %s (Line %s)",
                current_scenario_id,
                line_num,
            )
            continue

        elif prompt_match:
            if current_scenario_id is None:
                logging.warning(
                    "Line %s: Found prompt delimiter but no active scenario ID: %s",
                    line_num,
                    line,
                )
                continue

            if current_prompt_lines:
                prompt_text = "\n".join(current_prompt_lines).strip()
                if prompt_text:
                    current_prompts_for_scenario.append(prompt_text)

            current_prompt_lines = []
            in_prompt_content = True
            logging.debug(
                "Starting Prompt %s for scenario %s (Line %s)",
                prompt_match.group(1),
                current_scenario_id,
                line_num,
            )
            continue

        elif current_scenario_id and in_prompt_content:
            current_prompt_lines.append(raw_line.rstrip("\n\r"))

        elif line:
            if not current_scenario_id:
                logging.debug(
                    "Line %s: Skipping non-empty line before first scenario: %s...",
                    line_num,
                    line[:50],
                )
            elif not in_prompt_content:
                logging.debug(
                    "Line %s: Skipping non-empty line before first prompt in scenario %s: %s...",
                    line_num,
                    current_scenario_id,
                    line[:50],
                )

    if current_prompt_lines:
        prompt_text = "\n".join(current_prompt_lines).strip()
        if prompt_text:
            current_prompts_for_scenario.append(prompt_text)

    if current_scenario_id and current_prompts_for_scenario:
        scenarios[current_scenario_id] = current_prompts_for_scenario
        logging.debug(
            "Stored final scenario %s with %s prompts.",
            current_scenario_id,
            len(current_prompts_for_scenario),
        )

    if not scenarios:
        logging.warning("Parsing finished, but no scenarios were loaded from %s.", source_label)
    else:
        logging.info(
            "Successfully parsed %s scenarios from %s",
            len(scenarios),
            source_label,
        )

    return scenarios


def parse_scenario_prompts(file_path: str) -> Dict[str, List[str]]:
    """Parse scenario prompts file; any ``BASELINE_QUESTIONS`` trailer is ignored."""
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.error("Scenario prompts file not found: %s", file_path)
        raise
    body, _ = split_baseline_questions_block(text)
    try:
        return parse_scenario_prompts_from_text(body, file_path)
    except Exception as e:
        logging.error(
            "Error parsing scenario prompts file %s: %s", file_path, e, exc_info=True
        )
        raise


def load_scenario_prompts_and_baseline(
    file_path: str,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Parse scenarios and baseline questions from one file."""
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.error("Scenario prompts file not found: %s", file_path)
        raise
    body, baseline = split_baseline_questions_block(text)
    scenarios = parse_scenario_prompts_from_text(body, file_path)
    return scenarios, baseline
