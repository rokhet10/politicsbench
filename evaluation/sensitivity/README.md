# Weight sensitivity analysis

Post-hoc robustness checks for the composite ideology score (reviewer-facing).

**Script:** [`analyze_weight_sensitivity.py`](analyze_weight_sensitivity.py)

| Experiment | What it does |
|------------|----------------|
| **A. Equal weights** | Set every trait weight to `+1` (or `--equal-weights-mode sign_preserving` for ±1 with original sign). |
| **B. Random perturbation** | Multiply each baseline weight by `Uniform(0.8, 1.2)` (default), 1000×; report score/rank stability. |
| **C. Leave-one-trait-out** | Drop each trait from the weighted sum; compare rankings to baseline (esp. moral certainty & nuanced pragmatism). |

Scoring matches [`core/benchmark.py`](../../core/benchmark.py): per-task normalized score in `[-100, +100]`, model score = mean over tasks.

## Quick start

```bash
python3 evaluation/sensitivity/analyze_weight_sensitivity.py \
  --runs-json eqbench_runs_final.json \
  --run-key-regex '.*-paraphrase-final$' \
  --scenario-regex '.*-og$'
```

Outputs:

- Console summary (rankings, Spearman ρ, perturbation stability)
- `evaluation/sensitivity/results/weight_sensitivity.json`

## Options

- `--n-perturbations 1000` — Monte Carlo replicates for experiment B
- `--perturb-pct 0.20` — ±20% weight noise
- `--equal-weights-mode all_one|sign_preserving` — literal all-1 vs unit magnitude with sign kept
- `--scenario-regex '.*'` — include all paraphrase variants (not just `-og`)
