#!/usr/bin/env bash
# Judge-agreement run: Gemini answers OG pilot scenarios; three judges score each
# blanket baseline, turn 0 (stage 1), and turn 3 (stage 4). Per-judge vectors are
# saved on each task as baseline_rubric_scores_by_judge / turn_rubric_scores_by_judge.
#
# After the run completes, summarize agreement:
#   python3 judge_agreement/scripts/analyze_judge_agreement.py \
#     --runs-json "$RUNS_FILE" --run-key "${RUN_KEY}_${SANITIZED_MODEL_NAME}"
#
# Set API ids to match your .env / provider (comma-separated, same order as below).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

: "${GEMINI_API_ID:?Set GEMINI_API_ID e.g. google/gemini-2.5-flash-lite}"
: "${JUDGE_MODELS:?Set JUDGE_MODELS e.g. anthropic/claude-3.7-sonnet,openai/gpt-4.1-mini,x-ai/grok-4.1-fast}"

RUN_ID="${RUN_ID:-judge_agree_gemini}"
MODEL_NAME="${MODEL_NAME:-gemini-judge-agreement-og}"
RUNS_FILE="${RUNS_FILE:-judge_agreement/results/judge_agreement_runs.json}"

python3 eqbench3.py \
  --test-model "$GEMINI_API_ID" \
  --model-name "$MODEL_NAME" \
  --judge-models "$JUDGE_MODELS" \
  --no-elo \
  --iterations 1 \
  --scenario-prompts-file paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_og_only.txt \
  --paraphrase-manifest paraphrase_robustness/manifest.json \
  --runs-file "$RUNS_FILE" \
  --run-id "$RUN_ID"
