#!/usr/bin/env bash
# Summarize commitment trajectories for every run in a runs JSON (post-hoc; no API calls).
#
# Usage (from repo root):
#   bash commitment/scripts/run_commitment_analysis_all_models.sh
#
# Optional:
#   RUNS_FILE=eqbench_runs_final.json
#   ONLY_VARIANT=og    # keep only *-og scenarios (20 tasks per model vs 60)
#   MAX_LINES=999      # draw all scenarios on the spaghetti plot (default in script is 40)
#
# After per-model outputs, runs one aggregate pass:
#   commitment_summary_all_models[_variant-og].json
#   commitment_trajectory_mean_all_models[_variant-og].png
#   commitment_trajectory_overlay_all_models[_variant-og].png
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RUNS_FILE="${RUNS_FILE:-eqbench_runs_final.json}"
ONLY_VARIANT="${ONLY_VARIANT:-}"
MAX_LINES="${MAX_LINES:-40}"

extra=()
if [[ -n "$ONLY_VARIANT" ]]; then
  extra+=(--only-variant "$ONLY_VARIANT")
fi

while IFS= read -r key; do
  echo "=============================================="
  echo "Commitment analysis: $key"
  echo "=============================================="
  python3 commitment/scripts/analyze_commitment.py \
    --runs-json "$RUNS_FILE" \
    --run-key "$key" \
    --max-lines "$MAX_LINES" \
    "${extra[@]}"
done < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); [print(k) for k in sorted(d)]" "$RUNS_FILE")

echo "=============================================="
echo "All models: mean trajectory + overlay"
echo "=============================================="
python3 commitment/scripts/analyze_commitment.py \
  --runs-json "$RUNS_FILE" \
  --all-models \
  --error-bar sem \
  "${extra[@]}"

echo "Done. Outputs under commitment/results/"
