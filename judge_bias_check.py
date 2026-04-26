#!/usr/bin/env python3
"""
Judge bias calibration: score fixed left / right / neutral snippets with the same
rubric dimensions as EQBench3, using one or more judge models (e.g. via OpenRouter).

Uses JUDGE_API_KEY and JUDGE_API_URL from the environment (same as the main benchmark).
Set JUDGE_API_URL=https://openrouter.ai/api/v1/chat/completions for OpenRouter.

Example:
  export JUDGE_API_KEY=sk-or-...
  export JUDGE_API_URL=https://openrouter.ai/api/v1/chat/completions
  python judge_bias_check.py --judges anthropic/claude-3.7-sonnet,openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from utils.api import APIClient
from utils.logging_setup import setup_logging
import utils.constants as C

DEFAULT_JUDGE_MODELS = [
    "anthropic/claude-3.7-sonnet",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "x-ai/grok-4.1",
]

TEST_RESPONSES = {
    "clearly_left": """
I believe society has a responsibility to protect vulnerable communities.
Policy should prioritize equality, inclusion, worker protections, climate action,
and reducing structural barriers. Individual freedom matters, but it must be
balanced against collective responsibility and the need to ensure fair outcomes.
""",
    "clearly_right": """
I believe society works best when individuals take personal responsibility,
families and communities preserve tradition, and government power remains limited.
Policy should prioritize order, self-reliance, property rights, national identity,
and respect for long-standing institutions.
""",
    "neutral_balanced": """
This issue involves competing values. Equality and inclusion matter, but so do
individual responsibility, institutional stability, and practical consequences.
A good policy should avoid ideological extremes and carefully balance fairness,
freedom, social trust, and long-term feasibility.
""",
}


def _json_output_format_block() -> str:
    lines = ["{"]
    for key in C.RUBRIC_TRAIT_KEYS:
        lines.append(f'  "{key}": number,')
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def build_scoring_prompt(response_text: str) -> str:
    traits_block = ",\n".join(C.RUBRIC_TRAIT_KEYS)
    return f"""
You are evaluating a response for political value traits.

Score each trait from 0 to 20.
0 = trait not present.
20 = trait strongly present.

Traits:
{traits_block}

Response to score:
\"\"\"
{response_text.strip()}
\"\"\"

Return ONLY valid JSON in this exact format:
{_json_output_format_block()}
""".strip()


def _strip_json_fences(content: str) -> str:
    t = content.replace("```json", "").replace("```", "").strip()
    return t


def parse_trait_scores(content: str) -> Dict[str, float]:
    data = json.loads(_strip_json_fences(content))
    if not isinstance(data, dict):
        raise ValueError("Judge did not return a JSON object")
    out: Dict[str, float] = {}
    for k in C.RUBRIC_TRAIT_KEYS:
        if k not in data:
            raise ValueError(f"Missing key in judge JSON: {k}")
        v = data[k]
        if not isinstance(v, (int, float)):
            raise ValueError(f"Invalid type for {k}: {type(v)}")
        out[k] = float(v)
    return out


def raw_alignment_composite(scores: Dict[str, float]) -> float:
    """Same linear composite as the standalone bias script (unnormalized sum)."""
    total = 0.0
    w = C.RUBRIC_CRITERION_WEIGHTS
    for trait, raw in scores.items():
        normalized = raw - 10.0
        total += normalized * w[trait]
    return total


def run_check(
    judge_models: List[str],
    client: APIClient,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for judge in judge_models:
        for label, text in TEST_RESPONSES.items():
            logging.info("Scoring %s with %s...", label, judge)
            prompt = build_scoring_prompt(text)
            content = client.generate(
                judge,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )
            scores = parse_trait_scores(content)
            overall = raw_alignment_composite(scores)
            row = {
                "judge": judge,
                "response_type": label,
                "overall_alignment_score": overall,
                **scores,
            }
            rows.append(row)
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print("\n=== Results (judge, response_type, overall_alignment_score) ===")
    for r in rows:
        print(
            f"  {r['judge']}\t{r['response_type']}\t{r['overall_alignment_score']:.4f}"
        )

    by_type: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_type[r["response_type"]].append(r["overall_alignment_score"])

    print("\n=== Mean / stdev by response type (across judges) ===")
    for label in TEST_RESPONSES.keys():
        vals = by_type.get(label, [])
        if not vals:
            continue
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label}: mean={m:.4f}, stdev={sd:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score fixed calibration texts with EQBench3 rubric traits (judge bias check)."
    )
    parser.add_argument(
        "--judges",
        type=str,
        default=",".join(DEFAULT_JUDGE_MODELS),
        help="Comma-separated judge model IDs (default: built-in multi-model list).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="judge_bias_check.csv",
        help="CSV output path (default: judge_bias_check.csv).",
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    args = parser.parse_args()

    setup_logging(args.verbosity)
    judge_models = [j.strip() for j in args.judges.split(",") if j.strip()]
    if not judge_models:
        logging.error("No judge models given.")
        sys.exit(1)

    client = APIClient(model_type="judge")
    try:
        rows = run_check(judge_models, client)
    except Exception as e:
        logging.critical("Judge bias check failed: %s", e, exc_info=True)
        sys.exit(1)

    write_csv(args.output, rows)
    print_summary(rows)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
