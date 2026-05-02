# Paraphrase robustness experiment (Stage 1 vs Stage 4)

This folder holds **PoliticsBench paraphrase-robustness** assets: scenario specs, a variant manifest, pilot prompt files, helper scripts, and analysis outputs.

## Layout

| Path | Purpose |
|------|---------|
| `specs/` | YAML specs per base scenario (moral spine, beats, facts) for generation or QC |
| `manifest.json` | Maps each `scenario_id` in the prompts file → `base_id`, `framing`, `tone`, `kind` |
| `prompts/` | `.txt` files in the same `########` / `####### PromptN` format as `data/scenario_prompts.txt` |
| `scripts/` | Validation, optional LLM thread generation, nonsense pilot builder, run helper |
| `qc/` | Human QC checklist before large API spend |
| `results/` | Plots and tables from `analyze_spread.py` (large JSON runs usually stay outside or in `../`) |

## Running the benchmark on pilot variants

From the repo root, use the standard entrypoint with **custom prompts** and optional **manifest provenance**:

```bash
python eqbench3.py \
  --test-model openai/gpt-4.1-mini \
  --model-name gpt-4.1-mini-paraphrase-pilot \
  --judge-model anthropic/claude-3.7-sonnet \
  --no-elo \
  --iterations 1 \
  --scenario-prompts-file paraphrase_robustness/prompts/scenario_prompts_pilot.txt \
  --paraphrase-manifest paraphrase_robustness/manifest.json \
  --runs-file paraphrase_robustness/results/paraphrase_runs.json
```

The run record stores `scenario_prompts_file`, `paraphrase_manifest_file`, and `paraphrase_manifest_sha256`.

## Analysis

PoliticsBench trait weights sum to **zero** on purpose (`RUBRIC_CRITERION_WEIGHTS` in `utils/constants.py`), so if every trait is the **same constant** in a turn, the weighted composite is **0** for every variant and spread is meaningless. Use **asymmetric** trait profiles (or per-trait spread) when synthesizing fixtures or sanity checks.

```bash
python paraphrase_robustness/analyze_spread.py \
  --manifest paraphrase_robustness/manifest.json \
  --runs-json path/to/eqbench3_runs.json \
  --run-key OPTIONAL_KEY_IF_FILE_HAS_MULTIPLE \
  --out-dir paraphrase_robustness/results
```

Use `--kind main` (default) to restrict to non-control variants, or `--kind control` for nonsense pilots. Omit `--kind` to pool all entries in the manifest.

## Generating new threads

- **LLM-assisted:** `python paraphrase_robustness/scripts/generate_threads.py --help` (requires API env vars).
- **Negative control prompts:** `python paraphrase_robustness/scripts/build_nonsense_pilot.py`.

## Validation

```bash
python paraphrase_robustness/scripts/validate_prompts.py paraphrase_robustness/prompts/scenario_prompts_pilot.txt
```

## Cost / completion counts (3 judges)

Use **`--judge-models modelA,modelB,modelC`** (e.g. GPT + Grok + Claude on OpenRouter). Each rubric step runs **all** judges: **5 × 3 = 15** judge completions per scenario per test-model run, plus **5** test completions.

Compare **paraphrase40** (40 scenario IDs = 20 bases × 2 framings) vs **standard20** (full default deck in `data/scenario_prompts.txt`):

```bash
python paraphrase_robustness/scripts/estimate_cost.py \
  --preset paraphrase40 --preset standard20 \
  --models 8 --judges 3 \
  --usd-per-mtok-input 0.15 --usd-per-mtok-output 0.60
```

Replace token defaults (`--test-in-tokens-per-scenario`, `--judge-in-tokens-per-scenario`, …) using totals from **one pilot** on your provider. If judges are priced differently than the test model, set `--judge-usd-per-mtok-input` / `--judge-usd-per-mtok-output` separately (e.g. a rough average across the three).
