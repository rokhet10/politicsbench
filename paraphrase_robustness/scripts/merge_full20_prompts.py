#!/usr/bin/env python3
"""
Concatenate pilot prompts (bases 1–10), OG blocks for 11–20, and generated wa/wb for 11–20.

Expected scenario order per base: N-og, N-wa, N-wb. The wa/wb file should contain only
11-wa..20-wb (from generate_wordings.py).

Default inputs live under ``archive/pre_unified_prompts/`` (see that README). Output is the unified repo-root **`scenario_prompts.txt`** (60 scenarios). Re-append
baseline JSON with **`scripts/append_baseline_from_manifest.py`**.

Usage:
  python3 paraphrase_robustness/scripts/merge_full20_prompts.py \\
    --pilot paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_pilot.txt \\
    --og11 paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_bases11_20_og.txt \\
    --wa-wb11 paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_bases11_20_wa_wb.txt \\
    --out scenario_prompts.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _read_blocks(path: Path) -> dict[str, str]:
    """Map scenario_id -> full block text."""
    text = path.read_text(encoding="utf-8").strip()
    chunks = re.split(r"(?m)^(?=########\s)", text)
    out: dict[str, str] = {}
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        first_line = ch.split("\n", 1)[0]
        sid = first_line.replace("########", "").strip().split()[0]
        out[sid] = ch
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, required=True, help="Bases 1–10 og/wa/wb.")
    ap.add_argument("--og11", type=Path, required=True, help="Bases 11–20 *-og only.")
    ap.add_argument("--wa-wb11", type=Path, required=True, help="Bases 11–20 *-wa and *-wb.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pilot_txt = args.pilot.read_text(encoding="utf-8").strip()
    og_map = _read_blocks(args.og11)
    ww_map = _read_blocks(args.wa_wb11)

    merged_chunks: list[str] = [pilot_txt]
    for n in range(11, 21):
        og_key = f"{n}-og"
        wa_key = f"{n}-wa"
        wb_key = f"{n}-wb"
        for k in (og_key, wa_key, wb_key):
            if k == og_key:
                chunk = og_map.get(k)
            else:
                chunk = ww_map.get(k)
            if not chunk:
                raise SystemExit(f"Missing block for {k}")
            merged_chunks.append(chunk)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(merged_chunks) + "\n", encoding="utf-8")
    n_headers = merged_chunks[0].count("########") + sum(
        c.count("########") for c in merged_chunks[1:]
    )
    print(f"Wrote {args.out} ({n_headers} scenario headers: pilot blob + bases 11–20 og/wa/wb)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
