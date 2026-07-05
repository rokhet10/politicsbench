# Evaluation (post-hoc on saved runs)

This index covers **paper and experiment analysis** — scripts that read finished `*_runs.json` files and produce tables, plots, or summary JSON. They do **not** call the test model; most make no API calls at all.

To **run the benchmark** (generate transcripts and judge scores), see the [top-level README](../README.md).

---

## What counts as evaluation vs the benchmark

| | Benchmark | Evaluation (this index) |
|---|-----------|------------------------|
| **Entry point** | [`eqbench3.py`](../eqbench3.py) | Scripts below |
| **API calls** | Test model + judge(s) | Usually none (rejudge is an exception) |
| **Input** | Scenario deck + prompts | Saved run JSON |
| **Output** | `eqbench3_runs.json` (or custom path) | Figures, summaries, CSV/JSON reports |

**Benchmark assets** (not evaluation): repo-root [`scenario_prompts.txt`](../scenario_prompts.txt), [`data/`](../data/) templates, [`commitment/`](../commitment/) judge prompts, [`paraphrase_robustness/manifest.json`](../paraphrase_robustness/manifest.json).

**Deck maintenance** (generate paraphrases, validate prompts): [`paraphrase_robustness/`](../paraphrase_robustness/README.md) — mostly pre-benchmark tooling, with `analyze_spread.py` as post-hoc eval.

---

## Analyses

### Trait activation

Folder: [`trait_activation/`](../trait_activation/README.md)

- **Activation trajectories** — mean activated traits (τ threshold) across baseline → stages → debrief.
- **Entropy** — trait-score dispersion at blanket / stage-1 / full scenario.
- **Significance** — paired tests on activation summaries.

Typical input: `eqbench_runs_final.json` or a snapshot under `trait_activation/results/`.

```bash
python3 trait_activation/scripts/analyze_trait_activation_stages.py \
  --runs-json eqbench_runs_final.json --all-models

python3 trait_activation/scripts/test_activation_significance.py
```

---

### Commitment trajectories

Folder: [`commitment/scripts/`](../commitment/scripts/)

- **0–5 commitment judge** summaries over baseline → stages → debrief (requires runs scored with commitment judging enabled).

```bash
bash commitment/scripts/run_commitment_analysis_all_models.sh
# or: python3 commitment/scripts/analyze_commitment.py --help
```

Commitment **prompts** used at benchmark time live in `commitment/turn_scoring_prompt.txt` and `commitment/debrief_scoring_prompt.txt`.

---

### Inter-judge agreement

Folder: [`judge_agreement/`](../judge_agreement/README.md)

- Multi-judge rubric runs; Cohen's κ (and optional variance mode) across judges and stages.
- Clone + [`scripts/rejudge_saved_run.py`](../scripts/rejudge_saved_run.py) workflow for prompt ablations on fixed transcripts.

```bash
python3 judge_agreement/scripts/analyze_judge_agreement.py \
  --runs-json judge_agreement/results/judge_agreement_runs.json \
  --all-models --out-json judge_agreement/results/summary.json
```

---

### Paraphrase robustness (spread)

Folder: [`paraphrase_robustness/`](../paraphrase_robustness/README.md) — see **Post-hoc spread** section.

- Compares scores across `og` / `wa` / `wb` variants per base scenario (no new API runs).

```bash
python3 paraphrase_robustness/analyze_spread.py \
  --runs-json eqbench_runs_final.json --all-models
```

---

### Judge bias calibration

Script: [`judge_bias_check.py`](../judge_bias_check.py)

- Scores fixed left/right/neutral snippets with the same rubric dimensions as the benchmark (judge API only).

```bash
python3 judge_bias_check.py --judges anthropic/claude-3.7-sonnet,openai/gpt-4.1-mini
```

---

## Utilities

| Tool | Role |
|------|------|
| [`scripts/rejudge_saved_run.py`](../scripts/rejudge_saved_run.py) | Re-run judge calls on saved `conversation_history` (after prompt changes). Uses current rubric/commitment templates. |
| [`viewer.html`](../viewer.html) | Browser viewer for run JSON (local static server may be required). |

---

## Local-only helpers

These may exist in your working tree but are not required to run the benchmark:

- [`analysis/`](../analysis/) — misc plotting/stats helpers (`misc.py`, `stats.py`, `viz.py`).
- Root-level figures, `table5_trait_scores.*`, and other paper exports.

---

## Suggested workflow

1. Run models with [`eqbench3.py`](../eqbench3.py) (or [`scripts/run_eqbench_final_8models.sh`](../scripts/run_eqbench_final_8models.sh)).
2. Point evaluation scripts at the same `--runs-file` (e.g. `eqbench_runs_final.json`).
3. See each subfolder README for outputs under `results/`.
