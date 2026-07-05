#!/usr/bin/env bash
# Weight sensitivity analysis on the 8-model paraphrase-final sweep (OG scenarios).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RUNS_FILE="${RUNS_FILE:-eqbench_runs_final.json}"

python3 evaluation/sensitivity/analyze_weight_sensitivity.py \
  --runs-json "$RUNS_FILE" \
  --run-key-regex '.*-paraphrase-final$' \
  --scenario-regex '.*-og$' \
  "$@"
