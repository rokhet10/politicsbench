#!/usr/bin/env bash
# Example: run PoliticsBench on paraphrase pilot prompts with manifest provenance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Use python3: on many systems `python` is still Python 2 (SyntaxError on f-strings).
python3 eqbench3.py \
  --test-model "${TEST_MODEL:-openai/gpt-4.1-mini}" \
  --model-name "${MODEL_NAME:-gpt-4.1-mini-paraphrase-pilot}" \
  --judge-model "${JUDGE_MODEL:-anthropic/claude-3.7-sonnet}" \
  --no-elo \
  --iterations 1 \
  --scenario-prompts-file paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_pilot.txt \
  --paraphrase-manifest paraphrase_robustness/manifest.json \
  --runs-file "${RUNS_FILE:-paraphrase_robustness/results/paraphrase_runs.json}"
