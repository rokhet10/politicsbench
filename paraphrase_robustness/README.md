# Surface-wording robustness (Stage 1 vs Stage 4)

This folder holds **PoliticsBench wording-robustness** assets: scenario specs, a variant manifest, pilot prompt files, helper scripts, and analysis outputs.

## Coverage

| Asset | Description |
|-------|-------------|
| **`manifest.json` (v8+)** | Maps each `scenario_id` → `base_id`, `framing` (`og` / `wa` / `wb`), `tone`, `kind`. Still ships **`baseline_questions`** (duplicated in the unified prompts file for convenience). |
| **Unified paraphrase deck (canonical)** | Repo-root **`scenario_prompts.txt`** (mirrored at **`prompts/scenario_prompts.txt`** via symlink) — **60** scenarios plus a trailing **`######## BASELINE_QUESTIONS`** JSON block (one blanket question per base **`1`–`20`**). |
| **10-base pilot (smaller / cheaper runs)** | **`archive/pre_unified_prompts/scenario_prompts_pilot.txt`** — bases **`1`–`10`** only (**30** rows). |

**Human QC** is recommended so `wa`/`wb` stay strict paraphrases (no fact drift). Use `qc/CHECKLIST.md` on a sample before large API spend; for bases **11–20**, QC the new families the same way as 1–10.

The **first wording snapshot** (`wa`/`wb`/`wc` pilot + manifest version frozen there) is **archived** at `archive/wording_1/`.

## Layout

| Path | Purpose |
|------|---------|
| `specs/` | YAML specs per base scenario (`01_*.yaml` … `20_*.yaml`): moral spine, beats, non‑negotiable facts — for generation and QC |
| `manifest.json` | Maps each `scenario_id` → metadata; includes `baseline_questions` for bases **1–20** |
| `prompts/` | **`scenario_prompts.txt`** symlink → repo-root canonical file |
| `scripts/` | Validation, LLM paraphrase generation, merge helpers, nonsense pilot, cost estimates |
| `qc/` | Human QC checklist |
| `results/` | Plots/tables from `analyze_spread.py`, generation logs (`wording_generate_*.jsonl`), bundles |

## Prompt files (which to use)

| File | When |
|------|------|
| **`scenario_prompts.txt`** (repo root; **`utils.constants.STANDARD_SCENARIO_PROMPTS_FILE`**) | **Default** — **60** staged scenarios + **`######## BASELINE_QUESTIONS`** JSON. Benchmark reads baseline from this file first (keys override manifest when both are passed). |
| `archive/pre_unified_prompts/scenario_prompts_pilot.txt` | Bases 1–10 only (30 scenarios). |
| `archive/pre_unified_prompts/scenario_prompts_bases11_20_og.txt` | Intermediate merge piece: **11-og … 20-og** (from `build_og_suffix_blocks.py`). |
| `archive/pre_unified_prompts/scenario_prompts_bases11_20_wa_wb.txt` | Intermediate merge piece: **11-wa … 20-wb** (`generate_wordings.py`). |

Older split filenames and a duplicate full merge live under **`archive/pre_unified_prompts/`** (see `README.md` there).

## Generating / refreshing surface paraphrases

**Paraphrase generator** (`scripts/generate_wordings.py`): one OpenAI Chat Completions call per prompt line (Prompt1–4), with optional **two** surface variants per base (`--num-variants 2 --variant-suffixes wa,wb`). Logs to JSONL; can write a consolidated `.txt`.

### Regenerate **bases 11–20** wa/wb (from canonical `data/scenario_prompts.txt`)

Requires `OPENAI_API_KEY` (and optional `OPENAI_API_URL`). From repo root:

```bash
python3 paraphrase_robustness/scripts/generate_wordings.py \
  --prompts-file data/scenario_prompts.txt \
  --scenarios 11,12,13,14,15,16,17,18,19,20 \
  --num-variants 2 --variant-suffixes wa,wb \
  --temperature 1.0 --variant-min-temperature 1.0 \
  --workers 8 --sleep 0 \
  --log paraphrase_robustness/results/wording_generate_bases11_20.jsonl \
  --out-json paraphrase_robustness/results/wording_generated_bundle_bases11_20.json \
  --out-txt paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_bases11_20_wa_wb.txt
```

Use `--resume` to continue from an existing `--log` without redoing successful rows.

### Build **11-og … 20-og** headers (canonical text, `-og` ids)

```bash
python3 paraphrase_robustness/scripts/build_og_suffix_blocks.py \
  --source data/scenario_prompts.txt \
  --bases 11-20 \
  --out paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_bases11_20_og.txt
```

### Merge into **`scenario_prompts.txt`** (repo root)

After `scenario_prompts_pilot.txt` (bases 1–10), **11–20 og**, and **11–20 wa/wb** exist under `archive/pre_unified_prompts/`:

```bash
python3 paraphrase_robustness/scripts/merge_full20_prompts.py \
  --pilot paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_pilot.txt \
  --og11 paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_bases11_20_og.txt \
  --wa-wb11 paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_bases11_20_wa_wb.txt \
  --out scenario_prompts.txt

python3 paraphrase_robustness/scripts/append_baseline_from_manifest.py --prompts scenario_prompts.txt

python3 paraphrase_robustness/scripts/validate_prompts.py scenario_prompts.txt
```

Expect **60** scenarios, each with **4** prompts.

**One-shot helper** (regenerate wa/wb then merge — same paths as above):

```bash
bash paraphrase_robustness/scripts/run_generate_bases11_20.sh
```

Other generators:

- **LLM-assisted threads:** `python3 paraphrase_robustness/scripts/generate_threads.py --help`
- **Negative control prompts:** `python3 paraphrase_robustness/scripts/build_nonsense_pilot.py`

## Running the benchmark on variants

From the repo root, use the standard entrypoint with **custom prompts** and optional **manifest provenance**:

**Full 20-base paraphrase deck:**

```bash
python3 eqbench3.py \
  --test-model openai/gpt-4.1-mini \
  --model-name gpt-4.1-mini-paraphrase-full20 \
  --judge-model anthropic/claude-3.7-sonnet \
  --iterations 1 \
  --scenario-prompts-file scenario_prompts.txt \
  --paraphrase-manifest paraphrase_robustness/manifest.json \
  --runs-file paraphrase_robustness/results/paraphrase_runs.json
```

**Smaller 10-base pilot** (unchanged):

```bash
python3 eqbench3.py \
  --scenario-prompts-file paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_pilot.txt \
  --paraphrase-manifest paraphrase_robustness/manifest.json \
  ...
```

The run record stores `scenario_prompts_file`, `paraphrase_manifest_file`, and `paraphrase_manifest_sha256`.

## Analysis

PoliticsBench trait weights sum to **zero** on purpose (`RUBRIC_CRITERION_WEIGHTS` in `utils/constants.py`), so if every trait is the **same constant** in a turn, the weighted composite is **0** for every variant and spread is meaningless. Use **asymmetric** trait profiles (or per-trait spread) when synthesizing fixtures or sanity checks.

```bash
python3 paraphrase_robustness/analyze_spread.py \
  --manifest paraphrase_robustness/manifest.json \
  --runs-json paraphrase_robustness/results/paraphrase_runs.json \
  --run-key YOUR_RUN_KEY \
  --out-dir paraphrase_robustness/results
```

If the runs JSON has **multiple** top-level keys, pass **`--run-key KEY`** or **`--latest`**.

Use `--kind main` (default) to restrict to non-control variants, or `--kind control` for nonsense pilots.

## Validation

```bash
python3 paraphrase_robustness/scripts/validate_prompts.py scenario_prompts.txt
```

## Cost / completion counts (3 judges)

Use **`--judge-models modelA,modelB,modelC`** (e.g. GPT + Grok + Claude on OpenRouter). Each rubric step runs **all** judges: **5 × 3 = 15** judge completions per scenario per test-model run (trait-only; double if dual trait+commitment judging), plus **5** test completions.

Compare **wording60** (60 scenario IDs, full paraphrase deck) vs **standard20**:

```bash
python3 paraphrase_robustness/scripts/estimate_cost.py \
  --preset wording60 --preset standard20 \
  --models 8 --judges 3 \
  --usd-per-mtok-input 0.15 --usd-per-mtok-output 0.60
```

You can also pass **`--prompts-file scenario_prompts.txt`** to count scenarios directly.

Replace token defaults (`--test-in-tokens-per-scenario`, `--judge-in-tokens-per-scenario`, …) using totals from **one pilot** on your provider. If judges are priced differently than the test model, set `--judge-usd-per-mtok-input` / `--judge-usd-per-mtok-output` separately.
