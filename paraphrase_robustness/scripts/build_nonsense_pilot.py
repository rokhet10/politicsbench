#!/usr/bin/env python3
"""
Emit a tiny nonsense four-turn scenario file (negative-control style).

The main pilot already includes ctrl-nonsense-a/b in scenario_prompts_pilot.txt.
Run this to regenerate a standalone file or to tweak wording without hand-editing the main pilot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE = """######## {sid} | Control | Bean ritual ({label})
####### Prompt1
 [This is a role-play, with you playing a human who is, essentially, yourself. Treat it like a real situation. Always respond in first person.] A cousin texts: you must decide by tonight whether the family will adopt a yearly ritual of counting **{color}** jellybeans on the porch for luck. Your sibling replies-all: "This is absurd—there is no moral weight to bean color." What are you thinking and feeling?
####### Prompt2
 The group chat explodes. One aunt insists tradition binds the family; another says spreadsheets about candy are a waste of moral energy. Which loyalty weighs more for you—harmless ritual cohesion or refusing nonsense tasks?
####### Prompt3
 Your parent asks you to break the tie publicly at dinner. What is non-negotiable: protecting the parent’s feelings or refusing to endorse a meaningless ceremony?
####### Prompt4
 The family adopts the ritual anyway. You are assigned to buy three pounds of beans out of pocket. Describe the trade-off of going along versus opting out, and why the worse option is worse—for you.

"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "prompts" / "scenario_prompts_nonsense_only.txt",
    )
    args = ap.parse_args()
    body = TEMPLATE.format(sid="ctrl-nonsense-a", label="variant A", color="red")
    body += TEMPLATE.format(sid="ctrl-nonsense-b", label="variant B", color="green")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
