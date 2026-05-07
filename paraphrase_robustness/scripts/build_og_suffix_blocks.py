#!/usr/bin/env python3
"""
Rewrite canonical scenario headers from data/scenario_prompts.txt:

  ######## 11 | Housing | ...
→ ######## 11-og | Housing | ... (original wording)

Usage:
  python3 paraphrase_robustness/scripts/build_og_suffix_blocks.py \\
    --source data/scenario_prompts.txt \\
    --bases 11-20 \\
    --out paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_bases11_20_og.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _parse_base_range(s: str) -> range:
    if "-" in s:
        a, b = s.split("-", 1)
        return range(int(a.strip()), int(b.strip()) + 1)
    n = int(s.strip())
    return range(n, n + 1)


def _rewrite_header(line: str) -> str:
    line = line.strip()
    m = re.match(r"^########\s+(\d+)(\s+.*)$", line)
    if not m:
        raise ValueError(f"Bad header: {line!r}")
    sid, rest = m.group(1), m.group(2).strip()
    if "(original wording)" in rest:
        return f"######## {sid}-og {rest}\n"
    return f"######## {sid}-og {rest} (original wording)\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument(
        "--bases",
        action="append",
        required=True,
        help='e.g. "11-20" or "11-11"',
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    want: set[int] = set()
    for br in args.bases:
        want.update(_parse_base_range(br))

    raw_lines = args.source.read_text(encoding="utf-8").splitlines(keepends=True)
    buf: list[str] = []
    out_chunks: list[str] = []
    emitting = False

    header_start = re.compile(r"^########\s+(\d+)(\s+.*)$")

    for line in raw_lines:
        m = header_start.match(line.rstrip("\n\r"))
        if m:
            if emitting and buf:
                out_chunks.append("".join(buf))
                buf = []
            sid = int(m.group(1))
            emitting = sid in want
            if emitting:
                buf.append(_rewrite_header(line))
            continue
        if emitting:
            buf.append(line)

    if emitting and buf:
        out_chunks.append("".join(buf))

    if not out_chunks:
        print("ERROR: no blocks written; check --bases and source.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(out_chunks) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_chunks)} scenario block(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
