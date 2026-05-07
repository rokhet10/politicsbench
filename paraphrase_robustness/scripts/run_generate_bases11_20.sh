#!/usr/bin/env bash
# Regenerate surface paraphrases (wa/wb) for bases 11–20 and merge into repo-root scenario_prompts.txt.
# Requires OPENAI_API_KEY. Run from repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ARCH="paraphrase_robustness/archive/pre_unified_prompts"
OG_OUT="$ARCH/scenario_prompts_bases11_20_og.txt"
WW_OUT="$ARCH/scenario_prompts_bases11_20_wa_wb.txt"
LOG="paraphrase_robustness/results/wording_generate_bases11_20.jsonl"
BUNDLE="paraphrase_robustness/results/wording_generated_bundle_bases11_20.json"
MERGED="scenario_prompts.txt"
PILOT="$ARCH/scenario_prompts_pilot.txt"

echo "==> build 11-og .. 20-og"
python3 paraphrase_robustness/scripts/build_og_suffix_blocks.py \
  --source data/scenario_prompts.txt \
  --bases 11-20 \
  --out "$OG_OUT"

echo "==> GPT paraphrases 11-wa..20-wb (80 API calls)"
python3 paraphrase_robustness/scripts/generate_wordings.py \
  --prompts-file data/scenario_prompts.txt \
  --scenarios 11,12,13,14,15,16,17,18,19,20 \
  --num-variants 2 \
  --variant-suffixes wa,wb \
  --temperature 1.0 \
  --variant-min-temperature 1.0 \
  --workers 8 \
  --sleep 0 \
  --log "$LOG" \
  --out-json "$BUNDLE" \
  --out-txt "$WW_OUT"

echo "==> merge pilot (1–10) + og/wa/wb (11–20)"
python3 paraphrase_robustness/scripts/merge_full20_prompts.py \
  --pilot "$PILOT" \
  --og11 "$OG_OUT" \
  --wa-wb11 "$WW_OUT" \
  --out "$MERGED"

python3 paraphrase_robustness/scripts/append_baseline_from_manifest.py --prompts "$MERGED"

python3 paraphrase_robustness/scripts/validate_prompts.py "$MERGED"
echo "Done: $MERGED"
