# Trait activation & entropy (post-hoc on judge runs)

This folder holds a **snapshot** of judge-agreement runs and **derived** metrics (trait activation counts and trait-score entropy at blanket / stage-1 / full scenario). No API calls in the analysis step.

## Data layout

| File | Role |
|------|------|
| `results/trait_activation_runs.json` | Subset of `judge_agreement/results/judge_agreement_runs.json` (by default: OG + less-judge-context run keys). Regenerate after you update judge scores upstream. |
| `results/trait_activation_entropy.json` | Primary report; currently aligned with the **lessctx** judge run (see `run_key` inside the JSON). |
| `results/trait_activation_entropy_og.json` | Same analysis for the original judge-agreement run (baseline). |
| `results/trait_activation_entropy_lessctx.json` | Copy of the lessctx report (same as `trait_activation_entropy.json` when that is the active variant). |

## Refresh snapshot + rerun analysis

From repo root, after `judge_agreement_runs.json` has been updated:

```bash
python3 trait_activation/scripts/snapshot_runs_from_judge_agreement.py

python3 trait_activation/scripts/analyze_trait_activation_entropy.py \
  --runs-json trait_activation/results/trait_activation_runs.json \
  --run-key judge_agree_gemini_gemini-judge-agreement-lessctx \
  --out-json trait_activation/results/trait_activation_entropy_lessctx.json

python3 trait_activation/scripts/analyze_trait_activation_entropy.py \
  --runs-json trait_activation/results/trait_activation_runs.json \
  --run-key judge_agree_gemini_gemini-judge-agreement-og \
  --out-json trait_activation/results/trait_activation_entropy_og.json

cp trait_activation/results/trait_activation_entropy_lessctx.json \
   trait_activation/results/trait_activation_entropy.json
```

Use `--keys` on the snapshot script if you only want one run in `trait_activation_runs.json`.
