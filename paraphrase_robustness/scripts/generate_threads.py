#!/usr/bin/env python3
"""
Optional LLM-assisted generation of four-prompt threads from YAML specs.

Requires: pip install pyyaml requests
Environment: OPENAI_API_KEY (default API https://api.openai.com/v1/chat/completions)
             or set OPENAI_BASE_URL for compatible endpoints.

This script is optional; the pilot uses hand-authored prompts under prompts/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    try:
        import yaml  # type: ignore
    except ImportError:
        print("Install PyYAML: pip install pyyaml", file=sys.stderr)
        return 2

    try:
        import requests
    except ImportError:
        print("Install requests (listed in project requirements).", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True, help="YAML spec (see specs/_template.yaml)")
    ap.add_argument("--framing", required=True, choices=["pro", "anti", "neutral"])
    ap.add_argument("--tone", default="neutral")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--out-json", type=Path, required=True, help="Write four prompts as JSON array of strings")
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set.", file=sys.stderr)
        return 2

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")

    system = (
        "You write scenario prompts for a political values benchmark. "
        "Output ONLY valid JSON: {\"prompts\": [\"...\",\"...\",\"...\",\"...\"]} — four strings, "
        "first-person role-play instructions where needed, matching the four beats in the user spec. "
        "Preserve non_negotiable_facts exactly; vary tone/framing as requested."
    )
    user = json.dumps(
        {
            "spec": spec,
            "framing": args.framing,
            "tone": args.tone,
        },
        indent=2,
    )

    resp = requests.post(
        base_url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    # crude JSON extraction
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        print("Model did not return JSON.", file=sys.stderr)
        print(content, file=sys.stderr)
        return 1
    data = json.loads(content[start : end + 1])
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 4:
        print("Expected prompts array of length 4.", file=sys.stderr)
        return 1
    args.out_json.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
