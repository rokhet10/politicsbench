# PoliticsBench

[![Paper (arXiv)](https://img.shields.io/badge/paper-arXiv%3A2603.23841-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.23841)
[![PDF](https://img.shields.io/badge/PDF-2603.23841-EC1C24?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2603.23841.pdf)

**PoliticsBench** is a multi-turn benchmark for studying how language models express **political and value-laden traits** in scenario-driven dialogue. Methodology and empirical results are described in [*PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay*](https://arxiv.org/abs/2603.23841) (Khetan & Khetan, 2026).

A **test model** completes structured multi-stage roleplay scenarios; **judge** LLM(s) score each stage and the debrief on:

- **Trait rubric** — ten political–psychological dimensions (0–20 each), combined with signed weights into a composite alignment score on roughly **[-100, 100]** ([`core/benchmark.py`](./core/benchmark.py), [`utils/constants.py`](./utils/constants.py)).
- **Commitment rubric** (default on) — single 0–5 scalar per stage and debrief ([`commitment/`](./commitment/) prompts).

This repository **builds on EQ-Bench 3** (scenario engine, debrief, judging pipeline). Please cite the **PoliticsBench paper** and **EQ-Bench** ([citation](#citation)).

**Post-hoc analysis** (trait activation, commitment curves, judge agreement, paraphrase spread, etc.) lives under separate folders — see **[Evaluation](./evaluation/README.md)**. That code reads saved run JSON; it is not part of the benchmark harness itself.

---

## Credits (EQ-Bench)

- **EQ-Bench 3**: [github.com/EQ-bench/eqbench3](https://github.com/EQ-bench/eqbench3)
- **EQ-Bench paper**: [arXiv:2312.06281](https://arxiv.org/abs/2312.06281) · [eqbench.com](https://eqbench.com/)

---

## What is measured

| Piece | Role |
|--------|------|
| **Scenario deck** | Repo-root [`scenario_prompts.txt`](./scenario_prompts.txt) — 60 paraphrase scenarios (`1-og` … `20-wb`) plus optional baseline questions in a JSON trailer. |
| **Templates** | [`data/`](./data/) — master roleplay prompt, debrief, trait rubric criteria/prompt. |
| **Trait rubric** | Ten dimensions scored 0–20; keys in `RUBRIC_TRAIT_KEYS`. |
| **Commitment** | 0–5 stance/commitment score at baseline, each stage, and debrief. |
| **Composite** | Weighted trait blend (`RUBRIC_CRITERION_WEIGHTS`) → liberal–conservative summary score. |

Scores depend on the **judge model(s)** and prompts — they measure behavior under that setup, not ground-truth human ideology.

---

## Table of contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Running the benchmark](#running-the-benchmark)
4. [Results files](#results-files)
5. [Evaluation](#evaluation)
6. [Repository layout](#repository-layout)
7. [Limitations](#limitations)
8. [License](#license)
9. [Citation](#citation)

---

## Installation

```bash
git clone <your-repo-url>
cd politicsbench
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`:

- `TEST_API_KEY` / `TEST_API_URL` — model under test
- `JUDGE_API_KEY` / `JUDGE_API_URL` — judge for rubric scoring

---

## Quickstart

Default run: **traits + commitment**, full 60-scenario deck, one iteration.

```bash
python3 eqbench3.py \
  --test-model openai/gpt-4.1-mini \
  --model-name gpt-4.1-mini-demo-run \
  --judge-models anthropic/claude-3.7-sonnet,openai/gpt-4.1-mini,x-ai/grok-4.1-fast \
  --threads 6 \
  --iterations 1 \
  --ignore-canonical \
  --scenario-prompts-file scenario_prompts.txt \
  --paraphrase-manifest paraphrase_robustness/manifest.json \
  --runs-file eqbench_runs_final.json
```

Output: transcripts and judge scores in `--runs-file` (default `eqbench3_runs.json`).

Batch helper for the paper’s eight final models: [`scripts/run_eqbench_final_8models.sh`](./scripts/run_eqbench_final_8models.sh).

---

## Running the benchmark

Entry point: [`eqbench3.py`](./eqbench3.py).

Pipeline per scenario: **simulate turns → debrief → judge** (per-stage trait + commitment judges, then debrief trait + commitment).

### Key paths

| Path | Purpose |
|------|---------|
| [`scenario_prompts.txt`](./scenario_prompts.txt) | Canonical scenario deck (default via `STANDARD_SCENARIO_PROMPTS_FILE`). |
| [`data/scenario_master_prompt.txt`](./data/scenario_master_prompt.txt) | Roleplay wrapper template. |
| [`data/debrief_prompt.txt`](./data/debrief_prompt.txt) | Debrief user message. |
| [`data/rubric_scoring_*.txt`](./data/) | Trait judge criteria and prompt. |
| [`commitment/turn_scoring_prompt.txt`](./commitment/turn_scoring_prompt.txt) | Commitment judge (per stage). |
| [`commitment/debrief_scoring_prompt.txt`](./commitment/debrief_scoring_prompt.txt) | Commitment judge (debrief). |
| [`paraphrase_robustness/manifest.json`](./paraphrase_robustness/manifest.json) | Variant metadata + baseline questions (SHA stored on run). |

Symlink: `paraphrase_robustness/prompts/scenario_prompts.txt` → repo-root deck.

### Scoring modes (commitment flags)

By default both **trait** and **commitment** judging run (`scoring_mode: both` on the run record).

| Flag | Effect |
|------|--------|
| *(default)* | Trait rubrics (0–20) **and** commitment (0–5) at baseline, each stage, and debrief. |
| `--no-commitment-judging` | Trait rubrics only (`scoring_mode: traits`). |
| `--no-trait-judging` | Commitment only (`scoring_mode: commitment`). |
| `--commitment-scoring` | Legacy alias for `--no-trait-judging`. |
| `--no-rubric` | Skip all judging (scenarios + debrief only). |

Trait-only or commitment-only runs still use the same scenario/debrief pipeline; only which judge prompts fire changes.

### Other CLI flags

| Argument | Description |
|----------|-------------|
| `--test-model` **(required)** | API id for the model under test. |
| `--model-name` | Logical name in JSON (defaults to `--test-model`). |
| `--judge-model` | Single judge id (if `--judge-models` unset). |
| `--judge-models` | Comma-separated judges; scores averaged per criterion. |
| `--runs-file` | Output JSON (default: `eqbench3_runs.json`). |
| `--scenario-prompts-file` | Override deck path (default: `scenario_prompts.txt` at repo root). |
| `--paraphrase-manifest` | Record manifest path + SHA on the run. |
| `--threads` | Parallel workers (default: 4). |
| `--iterations` | Repeats per scenario (default: 1). |
| `--ignore-canonical` | Do not load bundled canonical leaderboard runs. |
| `--leaderboard-runs-file` | Optional read-only reference runs. |
| `--redo-rubric-judging` | Reset completed tasks and re-run judges. |
| `--reset-model` | Delete local runs for `--model-name` before starting. |
| `--run-id` | Optional run-key prefix for resume. |

Run `python3 eqbench3.py --help` for the full list.

### Re-judging saved transcripts

After changing judge prompts, re-score existing runs without re-running the test model:

```bash
python3 scripts/rejudge_saved_run.py \
  --runs-file eqbench_runs_final.json \
  --run-key YOUR_RUN_KEY \
  --threads 8
```

See [`evaluation/README.md`](./evaluation/README.md) for clone workflows (judge agreement ablations).

---

## Results files

| File | Contents |
|------|----------|
| `eqbench3_runs.json` (default) | Per-run metadata, `scenario_tasks` (histories, per-stage rubrics, debrief scores), aggregated `results`. |
| Custom path (e.g. `eqbench_runs_final.json`) | Same schema; use for multi-model sweeps. |

Each task stores turn-level trait/commitment scores, debrief scores, and status (`rubric_scored` when complete).

---

## Evaluation

Paper analyses — trait activation, commitment trajectories, judge agreement, paraphrase spread, bias checks — are documented in **[evaluation/README.md](./evaluation/README.md)**.

Those scripts consume saved run JSON produced by the benchmark above. They do not replace `eqbench3.py`.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `eqbench3.py` | Benchmark CLI. |
| `core/` | Scenario loop, judge aggregation. |
| `utils/` | Constants, API client, I/O. |
| `data/` | Master/debrief/rubric templates (not the scenario deck). |
| `scenario_prompts.txt` | Canonical 60-scenario deck. |
| `commitment/` | Commitment judge prompts (+ analysis scripts in `scripts/`). |
| `paraphrase_robustness/` | Manifest, deck symlink, paraphrase specs/generation, spread analysis. |
| `scripts/` | Batch runs, rejudge helper. |
| `evaluation/README.md` | Index of post-hoc analysis folders. |
| `trait_activation/`, `judge_agreement/` | Evaluation packages (see evaluation index). |
| `viewer.html` | Optional run JSON viewer. |

---

## Limitations

1. **Judge dependence** — Scores reflect the judge and prompt wording.
2. **Prompt dependence** — Different decks or master prompts change what is measured.
3. **Cost** — Multi-turn generation plus many judge calls add up at scale.
4. **Composite** — The summary score is a weighted linear blend of traits.

---

## License

Derived from EQ-Bench 3. See [`LICENSE`](./LICENSE).

---

## Citation

**PoliticsBench:**

Rohan Khetan and Ashna Khetan, *PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay*, arXiv:2603.23841, 2026.  
[https://arxiv.org/abs/2603.23841](https://arxiv.org/abs/2603.23841)

```bibtex
@misc{khetan2026politicsbench,
  title         = {PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay},
  author        = {Rohan Khetan and Ashna Khetan},
  year          = {2026},
  eprint        = {2603.23841},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2603.23841}
}
```

**EQ-Bench 3:**

```bibtex
@misc{eqbench3_repo_2025,
  author       = {Samuel J. Paech},
  title        = {EQ-Bench 3: Emotional Intelligence Benchmark},
  year         = {2025},
  howpublished = {\url{https://github.com/EQ-bench/eqbench3}}
}
```

**Original EQ-Bench:**

```bibtex
@misc{paech2023eqbench,
  title        = {EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models},
  author       = {Samuel J. Paech},
  year         = {2023},
  eprint       = {2312.06281},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}
```
