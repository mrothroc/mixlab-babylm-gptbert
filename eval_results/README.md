# eval_results

Per-component scores backing the table in the top-level [README](../README.md). Both models were scored
through one identical harness — [`babylm-org/babylm-eval`](https://github.com/babylm-org/babylm-eval) @ commit
`3bf5142`, mntp backend, full subsets.

- **`reference/`** — within-harness re-evaluation of the official BabyLM 2025 GPT-BERT masked-focus baseline
  ([`BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus`](https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus)).
- **`reproduction/`** — the mixlab run (full corpus, 9.87 epochs).
- **`summary.json`** — machine-readable table; the zero-shot values are extracted directly from the report
  files in this directory.

Each zero-shot `*.txt` is the eval pipeline's own score report: BLiMP / BLiMP-supplement / EWoK / COMPS /
entity-tracking show per-field detail plus an `AVERAGE ACCURACY`; `reading.txt` shows the EYE and SPR scores,
and `reading = mean(EYE, SPR)`.

**GLUE** (`reproduction/glue.json`): the reproduction's per-task fine-tune results — macro **64.18** (mean of
each task's primary metric: accuracy; F1 for MRPC/QQP), reproducible via [`../scripts/run_glue.sh`](../scripts/run_glue.sh).
The **reference GLUE 66.20** is the official published baseline number (baselines paper, Table 2); the baseline
was **not** re-fine-tuned locally, so there is no reference GLUE breakdown here. A Space-scored variant of the
reproduction gave 64.03.

## Not included here
- **Raw per-example `predictions.json`** (5–7 MB per zero-shot task) are omitted for size; the score reports
  carry the headline numbers.
- **AoA** is forfeited ≈0 for **both** models (a known upstream scorer issue for 16k-vocab models) and is
  excluded from the per-component comparison.
