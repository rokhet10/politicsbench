#!/usr/bin/env bash
# Optional remainder batch (e.g. Qwen + Llama) or Llama-only: edit the run_one / python3 blocks below.
# Same setup as the full 8-model sweep: repo-root scenario_prompts.txt (60 scenarios + baseline),
# appends to one runs JSON.
#
# IMPORTANT:
#   • --threads only speeds up *one* model's run (parallel scenario tasks).
#   • Run this script once at a time. Do not start a second copy while the first is writing
#     to the same RUNS_FILE — concurrent processes can corrupt the JSON.
#   • For Llama on OpenRouter free tier, use low concurrency, e.g. THREADS=1 or THREADS=2.
#
# Usage (from repo root):
#   export OPENROUTER_API_KEY=...
#   THREADS=6 bash scripts/run_eqbench_final_8models.sh
# Qwen steps use $THREADS; Llama uses $LLAMA_THREADS (default 2). Example:
#   THREADS=8 LLAMA_THREADS=1 bash scripts/run_eqbench_final_8models.sh
#
# Optional:
#   RUNS_FILE=eqbench_runs_final.json
#   JUDGE_MODELS="anthropic/claude-3.7-sonnet,openai/gpt-4.1-mini,x-ai/grok-4.1-fast"
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUNS_FILE="${RUNS_FILE:-eqbench_runs_final.json}"
THREADS="${THREADS:-6}"
# Default: Claude + GPT + Grok (comma-separated; scores are aggregated across judges).
JUDGE_MODELS="${JUDGE_MODELS:-anthropic/claude-3.7-sonnet,openai/gpt-4.1-mini,x-ai/grok-4.1-fast}"

run_one() {
  local api_id="$1"
  local logical_name="$2"
  echo "=============================================="
  echo "Starting: $logical_name ($api_id)"
  echo "=============================================="
  python3 eqbench3.py \
    --test-model "$api_id" \
    --model-name "$logical_name" \
    --judge-models "$JUDGE_MODELS" \
    --threads "$THREADS" \
    --iterations 1 \
    --ignore-canonical \
    --scenario-prompts-file scenario_prompts.txt \
    --paraphrase-manifest paraphrase_robustness/manifest.json \
    --runs-file "$RUNS_FILE"
}

# API id (OpenRouter-style) | logical model_name for this run (stored in JSON)
# Order: both Qwens first, Llama last (free tier 429s under high parallel load).

# run_one "qwen/qwen3-235b-a22b-2507"               "qwen3-235b-a22b-2507-paraphrase-final"
# run_one "qwen/qwen3-235b-a22b"                    "qwen3-235b-a22b-paraphrase-final"

# Llama: override threads for this step only unless LLAMA_THREADS is unset (default 2).
_LLAMA_THREADS="${LLAMA_THREADS:-2}"
echo "=============================================="
echo "Llama (meta-llama/...:free): using --threads $_LLAMA_THREADS (set LLAMA_THREADS to override)"
echo "=============================================="
python3 eqbench3.py \
  --test-model "meta-llama/llama-3.3-70b-instruct" \
  --model-name "llama-3.3-70b-instruct-paraphrase-final" \
  --judge-models "$JUDGE_MODELS" \
  --threads "$_LLAMA_THREADS" \
  --iterations 1 \
  --ignore-canonical \
  --scenario-prompts-file scenario_prompts.txt \
  --paraphrase-manifest paraphrase_robustness/manifest.json \
  --runs-file "$RUNS_FILE"

echo "Llama-only batch finished. Results in $RUNS_FILE."
