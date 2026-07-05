# PoliticsBench

[![Paper (arXiv)](https://img.shields.io/badge/paper-arXiv%3A2603.23841-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.23841)
[![PDF](https://img.shields.io/badge/PDF-2603.23841-EC1C24?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2603.23841.pdf)

**PoliticsBench** is a multi-turn benchmark for studying how language models express **political and value-laden traits** in scenario-driven dialogue. Methodology and empirical results are described in [*PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay*](https://arxiv.org/abs/2603.23841) (Khetan & Khetan, 2026). A tested model completes the same style of structured, multi-stage interactions used in [EQ-Bench 3](https://github.com/EQ-bench/eqbench3); an auxiliary **judge** LLM then scores outputs on a **ten-dimension political–psychological rubric** (0–20 per trait). Trait scores are combined with fixed polarity weights into a single **composite alignment score** roughly on **[-100, 100]** (see [`core/benchmark.py`](./core/benchmark.py) and [`utils/constants.py`](./utils/constants.py) for the exact definition).

This repository **builds on EQ-Bench 3** (scenario engine, debrief, rubric pass). We are grateful to **Samuel J. Paech** and the EQ-Bench project for releasing the upstream codebase and benchmark design. Please cite the **PoliticsBench paper** and **EQ-Bench** when you use this stack ([citation](#citation)).

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

**Important:** All rubric outputs depend on a **single (or small suite of) judge model(s)**. They measure **behavior under that judge and these prompts**, not ground-truth human ideology.

---

## Table of contents

1. [Installation](#installation)  
2. [Quickstart](#quickstart)  
3. [Running the benchmark](#running-the-benchmark)  
4. [Results files](#results-files)  
5. [Analysis and viewer](#analysis-and-viewer)  
6. [Repository layout](#repository-layout)  
7. [Limitations](#limitations)  
8. [License](#license)  
9. [Citation](#citation)

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
   - `JUDGE_API_KEY` / `JUDGE_API_URL` — judge used for rubric scoring

---

## Quickstart

```bash
python eqbench3.py \
  --test-model openai/gpt-4.1-mini \
  --model-name gpt-4.1-mini-demo-run \
  --judge-model anthropic/claude-3.7-sonnet \
  --iterations 1
```

Transcripts and scores are written to the local runs file (default `eqbench3_runs.json`).

For **PoliticsBench-only** runs (no bundled EQ leaderboard), use `--ignore-canonical` and/or point `--leaderboard-runs-file` at your own baseline JSON.

---

## Running the benchmark

Entry point: [`eqbench3.py`](./eqbench3.py). It orchestrates scenario execution, debrief, and **rubric** judging on the political trait dimensions.

### Command-line arguments

| Argument | Description |
|----------|-------------|
| `--test-model` **(required)** | API model id for the model under test (e.g. `openai/gpt-4.1-mini`). |
| `--model-name` | Logical name for storage (defaults to `--test-model`). |
| `--judge-model` | Single judge model id (used if `--judge-models` is not set). |
| `--judge-models` | Comma-separated judge ids; scores can be averaged across judges. Overrides `--judge-model` when set. |
| `--runs-file` | Local runs JSON (default: `eqbench3_runs.json`). |
| `--leaderboard-runs-file` | Read-only reference runs (default: `data/canonical_leaderboard_results.json.gz`). |
| `--run-id` | Optional run key prefix; random if omitted. |
| `--threads` | Parallel API workers (default: 4). |
| `--verbosity` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`). |
| `--save-interval` | Save progress every N tasks (default: 2). |
| `--iterations` | Repeat each scenario this many times (default: 1). |
| `--no-rubric` | Skip rubric. |
| `--ignore-canonical` | Do not load default canonical leaderboard files. |
| `--redo-rubric-judging` | Force rubric re-scoring for completed tasks. |
| `--reset-model` | Remove local runs for the logical `--model-name` before starting. |
| `--scenario-prompts-file` | Override the scenario prompts `.txt` path (default: `data/scenario_prompts.txt` via constants). Stored on the run and reused on resume. |
| `--paraphrase-manifest` | Optional JSON manifest (e.g. paraphrase experiment); file SHA-256 is recorded on the run for provenance. |

Run `python eqbench3.py --help` for the full list.

Scenario and rubric **file paths** default through [`utils/constants.py`](./utils/constants.py) (e.g. `data/scenario_prompts.txt`, `data/rubric_scoring_prompt.txt`). For ad-hoc prompt sets, pass `--scenario-prompts-file` instead of editing `data/` in place. Paraphrase-robustness assets and scripts live under [`paraphrase_robustness/`](./paraphrase_robustness/README.md).

---

## Results files

| File | Contents |
|------|----------|
| `eqbench3_runs.json` (default) | Per-run transcripts, task status, rubric breakdowns, aggregated results. |
| `data/canonical_leaderboard_*.json.gz` | Optional upstream-style reference data (may be absent in a politics-only checkout). |

---

## Analysis and viewer

- Optional Python utilities under [`analysis/`](./analysis/) (e.g. stats, plots).  
- [`paraphrase_robustness/`](./paraphrase_robustness/README.md) — paraphrase pilot prompts, manifest, validation, spread analysis (Stage 1 vs Stage 4), and negative-control helpers.  
- [`viewer.html`](./viewer.html) — local HTML viewer for exploring run JSON (open in a browser; may require a local static server depending on browser file policies).  
- Paper-style tables or exports may live at the repo root (e.g. `table5_trait_scores.md`).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `eqbench3.py` | CLI entry point. |
| `core/` | Scenario loop and rubric aggregation. |
| `utils/` | Constants (trait keys, weights, paths), API helpers, I/O, logging. |
| `data/` | Scenario text, master prompts, rubric templates, optional canonical archives. |
| `analysis/` | Supplementary analysis scripts. |
| `paraphrase_robustness/` | Paraphrase robustness experiment (specs, pilot prompts, `analyze_spread.py`, QC checklist). |

---

## Limitations

1. **Judge dependence** — Scores reflect the judge model and prompt wording, not an objective political “truth.”  
2. **Prompt dependence** — Different scenario files or master prompts change what is measured.  
3. **Cost** — Multi-turn generation plus many judge calls can be expensive at scale.  
4. **Composite** — The [-100, 100]-style summary is a **weighted linear blend** of traits; it simplifies a high-dimensional response into one number.

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
