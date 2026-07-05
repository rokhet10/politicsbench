# Inter-judge agreement pilot

Scripts and results for the multi-judge rubric agreement experiment (OG scenarios, three judges).

## New run entry + rerun judges (same test model, new judge prompting)

After you change judge-side prompts or truncation in code / `data/rubric_*.txt` / turn-rubric templates, keep the **same** Gemini (or other) transcripts but store fresh judge scores under a **new** `run_key`:

1. **Clone** the finished run (copies `scenario_tasks` only; metadata is updated):

   ```bash
   python3 judge_agreement/scripts/clone_run_for_rejudge.py \
     --runs-file judge_agreement/results/judge_agreement_runs.json \
     --from-run-key judge_agree_gemini_gemini-judge-agreement-og \
     --to-run-key judge_agree_gemini_gemini-judge-agreement-lessctx \
     --model-name gemini-judge-agreement-lessctx \
     --note "Judges see reduced context (prompt update YYYY-MM-DD)"
   ```

   Add `--truncate-for-rubric` if you will rejudge with `scripts/rejudge_saved_run.py --truncate-for-rubric` so the record matches.

2. **Reinvoke judges** on saved histories (uses **current** rubric prompt files and `TURN_RUBRIC_*` in `utils/constants.py`):

   ```bash
   python3 scripts/rejudge_saved_run.py \
     --runs-file judge_agreement/results/judge_agreement_runs.json \
     --run-key judge_agree_gemini_gemini-judge-agreement-lessctx \
     --threads 8
   ```

   Pass `--truncate-for-rubric` here when you recorded it on the clone.

3. **Remeasure agreement**:

   ```bash
   python3 judge_agreement/scripts/analyze_judge_agreement.py \
     --runs-json judge_agreement/results/judge_agreement_runs.json \
     --run-key judge_agree_gemini_gemini-judge-agreement-lessctx \
     --out-json judge_agreement/results/judge_agreement_summary_lessctx.json
   ```

## Fresh benchmark run (optional)

To collect everything from scratch instead of cloning + rejudge:

```bash
python3 judge_agreement/scripts/run_judge_agreement_gemini.sh
```

Adjust `RUNS_FILE`, `GEMINI_API_ID`, and `JUDGE_MODELS` as needed.
