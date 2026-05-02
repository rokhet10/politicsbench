# PoliticsBench

[![Paper (arXiv)](https://img.shields.io/badge/paper-arXiv%3A2603.23841-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.23841)
[![PDF](https://img.shields.io/badge/PDF-2603.23841-EC1C24?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2603.23841.pdf)

**PoliticsBench** is a multi-turn benchmark for studying how language models express **political and value-laden traits** in scenario-driven dialogue. Methodology and empirical results are described in [*PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay*](https://arxiv.org/abs/2603.23841) (Khetan & Khetan, 2026). A tested model completes the same style of structured, multi-stage interactions used in [EQ-Bench 3](https://github.com/EQ-bench/eqbench3); an auxiliary **judge** LLM then scores outputs on a **ten-dimension political–psychological rubric** (0–20 per trait). Trait scores are combined with fixed polarity weights into a single **composite alignment score** roughly on **[-100, 100]** (see [`core/benchmark.py`](./core/benchmark.py) and [`utils/constants.py`](./utils/constants.py) for the exact definition).

This repository **builds on EQ-Bench 3** (scenario engine, optional debrief, rubric pass, optional pairwise **ELO / TrueSkill**). We are grateful to **Samuel J. Paech** and the EQ-Bench project for releasing the upstream codebase and benchmark design. Please cite the **PoliticsBench paper** and **EQ-Bench** when you use this stack ([citation](#citation)).

---

## Credits (EQ-Bench)

- **EQ-Bench 3** (software architecture, multi-turn harness, judging pipeline): [https://github.com/EQ-bench/eqbench3](https://github.com/EQ-bench/eqbench3)  
- **EQ-Bench** (original benchmark and paper): [arXiv:2312.06281](https://arxiv.org/abs/2312.06281) · [https://eqbench.com/](https://eqbench.com/)

PoliticsBench-specific pieces include the **scenario prompts** under `data/`, the **trait rubric** (`RUBRIC_TRAIT_KEYS`, `RUBRIC_CRITERION_WEIGHTS` in `utils/constants.py`), and analysis scripts / figures in this repo.

---

## What is measured

| Piece | Role |
|--------|------|
| **Scenarios** | Multi-turn (or related) tasks defined in `data/` — e.g. `scenario_prompts.txt`, `scenario_master_prompt.txt`, and experiment variants you may add alongside them. |
| **Rubric traits** | Ten dimensions (e.g. tradition vs progress orientation, authority deference, egalitarianism, …) scored 0–20 by the judge. Keys: `RUBRIC_TRAIT_KEYS` in [`utils/constants.py`](./utils/constants.py). |
| **Weights** | Each trait has a signed weight (`RUBRIC_CRITERION_WEIGHTS`) so that higher rubric values on some traits push the composite toward one pole and on others toward the opposite. |
| **Composite** | Aggregated run-level score interpreted as a **liberal–conservative axis** in code comments (−100 conservative … +100 liberal). This is **not** the original EQ-Bench “EQ 0–100” scale. |
| **ELO (optional)** | Same machinery as EQ-Bench 3: pairwise judge comparisons and TrueSkill-style aggregation for **relative** ranking of models on transcript quality / criteria (useful for comparisons; interpret separately from the ideology composite). |

**Important:** All rubric and ELO outputs depend on a **single (or small suite of) judge model(s)**. They measure **behavior under that judge and these prompts**, not ground-truth human ideology.

---

## Table of contents

1. [Installation](#installation)  
2. [Quickstart](#quickstart)  
3. [Running the benchmark](#running-the-benchmark)  
4. [Rubric vs ELO](#rubric-vs-elo)  
5. [Results files](#results-files)  
6. [Analysis and viewer](#analysis-and-viewer)  
7. [Repository layout](#repository-layout)  
8. [Limitations](#limitations)  
9. [License](#license)  
10. [Citation](#citation)

---

## Installation

1. Clone this repository and enter the project directory (adjust the URL to your fork or remote).

   ```bash
   git clone <your-repo-url>
   cd politicsbench
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure API keys in `.env`:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` so that, typically:

   - `TEST_API_KEY` / `TEST_API_URL` — model under test  
   - `JUDGE_API_KEY` / `JUDGE_API_URL` — judge used for rubric (and ELO if enabled)

---

## Quickstart

**Rubric only** (one iteration, no ELO; good default for PoliticsBench-style runs):

```bash
python eqbench3.py \
  --test-model openai/gpt-4.1-mini \
  --model-name gpt-4.1-mini-demo-run \
  --judge-model anthropic/claude-3.7-sonnet \
  --no-elo \
  --iterations 1
```

Transcripts and scores are written to the local runs file (default `eqbench3_runs.json`).

**Rubric + ELO** (after scenarios finish, pairwise comparisons vs leaderboard and/or local runs):

```bash
python eqbench3.py \
  --test-model openai/gpt-4.1-mini \
  --model-name my-gpt4-run \
  --judge-model anthropic/claude-3.7-sonnet
```

ELO state defaults to `elo_results_eqbench3.json`.

For **PoliticsBench-only** comparisons (no bundled EQ leaderboard), use `--ignore-canonical` and/or point `--leaderboard-runs-file` / `--leaderboard-elo-file` at your own baseline JSON files.

---

## Running the benchmark

Entry point: [`eqbench3.py`](./eqbench3.py). It orchestrates scenario execution, optional debrief, **rubric** judging on the political trait dimensions, and optional **ELO** analysis.

### Command-line arguments

| Argument | Description |
|----------|-------------|
| `--test-model` **(required)** | API model id for the model under test (e.g. `openai/gpt-4.1-mini`). |
| `--model-name` | Logical name for storage and ELO (defaults to `--test-model`). Should be unique per run line you care about. |
| `--judge-model` | Single judge model id (used if `--judge-models` is not set). |
| `--judge-models` | Comma-separated judge ids; scores can be averaged across judges. Overrides `--judge-model` when set. |
| `--runs-file` | Local runs JSON (default: `eqbench3_runs.json`). |
| `--elo-results-file` | Local ELO JSON (default: `elo_results_eqbench3.json`). |
| `--leaderboard-runs-file` | Read-only reference runs (default: `data/canonical_leaderboard_results.json.gz`). |
| `--leaderboard-elo-file` | Read-only reference ELO (default: `data/canonical_leaderboard_elo_results.json.gz`). |
| `--run-id` | Optional run key prefix; random if omitted. |
| `--threads` | Parallel API workers (default: 4). |
| `--verbosity` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`). |
| `--save-interval` | Save progress every N tasks (default: 2). |
| `--iterations` | Repeat each scenario this many times (default: 1). |
| `--no-elo` | Skip ELO. |
| `--no-rubric` | Skip rubric. |
| `--ignore-canonical` | Do not load default canonical leaderboard files (local-only ELO context). |
| `--redo-rubric-judging` | Force rubric re-scoring for completed tasks. |
| `--reset-model` | Remove local runs and ELO entries for the logical `--model-name` before starting. |

Run `python eqbench3.py --help` for the full list.

Scenario and rubric **file paths** are wired through [`utils/constants.py`](./utils/constants.py) (e.g. `data/scenario_prompts.txt`, `data/rubric_scoring_prompt.txt`). To try alternate prompt sets, swap or symlink the files under `data/` consistently with those constants, or extend the code to accept overrides if you add that feature.

---

## Rubric vs ELO

- **Rubric** — Absolute trait scores per scenario/stage and the weighted **PoliticsBench composite**. Best when you want a fixed scale per model under one judge.  
- **ELO** — Relative placement via pairwise comparisons (upstream EQ-Bench 3 design). More discriminative for ranking, typically more judge calls and cost. Canonical leaderboard files, when present, come from the **EQ-Bench** ecosystem; for pure politics experiments you often want `--ignore-canonical` or custom leaderboard paths.

You may run **either or both**.

---

## Results files

| File | Contents |
|------|----------|
| `eqbench3_runs.json` (default) | Per-run transcripts, task status, rubric breakdowns, aggregated results. |
| `elo_results_eqbench3.json` (default) | Pairwise comparisons and ELO-related metadata. |
| `data/canonical_leaderboard_*.json.gz` | Optional upstream-style reference data (may be absent in a politics-only checkout). |

---

## Analysis and viewer

- Optional Python utilities under [`analysis/`](./analysis/) (e.g. stats, plots).  
- [`viewer.html`](./viewer.html) — local HTML viewer for exploring run JSON (open in a browser; may require a local static server depending on browser file policies).  
- Paper-style tables or exports may live at the repo root (e.g. `table5_trait_scores.md`).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `eqbench3.py` | CLI entry point. |
| `core/` | Scenario loop, rubric aggregation, ELO, pairwise judging, TrueSkill. |
| `utils/` | Constants (trait keys, weights, paths), API helpers, I/O, logging. |
| `data/` | Scenario text, master prompts, rubric templates, optional canonical archives. |
| `merge_results_to_canonical.py` | Upstream helper to merge local runs into canonical leaderboard files (EQ-Bench workflow). |
| `analysis/` | Supplementary analysis scripts. |

---

## Limitations

1. **Judge dependence** — Scores reflect the judge model and prompt wording, not an objective political “truth.”  
2. **Prompt dependence** — Different scenario files or master prompts change what is measured.  
3. **Truncation** — Pairwise (ELO) judging may truncate transcripts; rubric behavior is configured in code.  
4. **Cost** — Multi-turn generation plus many judge calls can be expensive at scale.  
5. **Composite** — The [-100, 100]-style summary is a **weighted linear blend** of traits; it simplifies a high-dimensional response into one number.

---

## License

This project includes code derived from EQ-Bench 3. See [`LICENSE`](./LICENSE) in this repository for terms applying here; respect upstream licenses and attribution when redistributing.

---

## Citation

If you use this benchmark or code in research, please cite the **PoliticsBench** paper and the **EQ-Bench** sources it builds on.

**PoliticsBench (this work):**

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

**EQ-Bench 3 (upstream repository):**

```bibtex
@misc{eqbench3_repo_2025,
  author       = {Samuel J. Paech},
  title        = {EQ-Bench 3: Emotional Intelligence Benchmark},
  year         = {2025},
  howpublished = {\url{https://github.com/EQ-bench/eqbench3}},
  note         = {Commit or release tag}
}
```

**Original EQ-Bench paper:**

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

If you extend the benchmark (new scenarios, judges, or weights), cite the **PoliticsBench** paper and **EQ-Bench** above, and describe your scenario set, judge model(s), and any changes to `RUBRIC_CRITERION_WEIGHTS` or prompts.
