# Reproducing the BabyLM 2025 GPT-BERT baseline on a single Mac, with mixlab

A faithful, from-scratch reproduction of the **BabyLM 2025 GPT-BERT *masked-focus* Strict-Small baseline**
([`BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus`](https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus))
trained entirely on a **single Apple-silicon GPU** with [**mixlab**](https://github.com/mrothroc/mixlab), an
open Metal/MLX language-model trainer.

This is not a leaderboard entry. It shows that a GPU-poor, Mac-runnable open pipeline can reproduce a strong
2025 reference baseline closely enough to build on. GPT-BERT masked-focus is not a vanilla transformer: it is a
masked+causal hybrid with disentangled relative attention, a gated attention output, dense layer aggregation,
and a BERT-style MLM head. Reproducing it end to end is a real test of the trainer.

## Result: faithful per-component reproduction

We did not enter the competition; this is a faithfulness demonstration, so we report the **per-component
comparison**, not an aggregate score. For the six zero-shot components, both the reference baseline and the
mixlab reproduction were scored through one identical eval harness:

| Component | Reference baseline | mixlab reproduction | Δ |
|---|---:|---:|---:|
| BLiMP | 70.36 | 70.64 | +0.28 |
| BLiMP-supplement | 63.71 | 61.79 | −1.92 |
| EWoK | 51.63 | 50.88 | −0.75 |
| entity tracking | 40.14 | 40.33 | +0.19 |
| COMPS | 53.55 | 52.85 | −0.70 |
| reading (eye+SPR) | 6.39 | 7.30 | +0.91 |
| GLUE † | 66.20 | 64.18 | −2.02 |

† The six zero-shot rows are both re-scored in our harness. For **GLUE**, the reproduction was fine-tuned and
scored here, but the reference value (66.20) is the **official published baseline** (baselines paper, Table 2) —
we did not re-fine-tune the baseline — so the GLUE row alone is not a within-harness re-eval of the reference.

The per-task eval reports backing this table (both models, one harness) are in
[`eval_results/`](eval_results/). The reproduction is within ≈2 points on every component, and slightly above
the reference on BLiMP and entity tracking. The **reference column is our within-harness re-evaluation of the official 2025 baseline model**, and
it tracks the official published numbers (baselines paper, Table 2): essentially exact on BLiMP (official 70.4),
supplement (63.7), entity (40.0), COMPS (53.5), reading (≈6.4), and GLUE (66.2), with EWoK the one outlier at
51.6 vs the official 50.0 (harness variance). (AoA is forfeited ≈0 here: age-of-acquisition is scored from a per-checkpoint surprisal trajectory submitted with the predictions, not from the final model, and these artifacts don't include one. It is *not* a 16k-vocab scorer bug — that earlier belief is incorrect — but a noise-dominated metric (scored with the leaderboard's own code, a single fixed model spans tens of points depending on extraction settings), so we report per-component, not a macro.)

## The architecture (read from the weights, not the card)

The prose model card is sparse (it lists a bias-free ≈31M network); the config and the custom model code are
the precise source. Reading the config, the custom code, and the served weights gives the real topology, which
mixlab reproduces:

- 12 layers, hidden 384, 6 heads, seq up to 512, vocab 16,384, tied embeddings (≈33M params)
- **DeBERTa-style disentangled relative attention** (shared rel-embedding, reuse content QK projections, windowed)
- **GeGLU** feed-forward (intermediate 1280) with internal LayerNorm
- **Affine-free LayerNorm** (eps 1e-7), pre- and post-attention (post before the out-projection)
- **Attention value-gate** + attention biases
- **Dense layer aggregation (DWA)** — a learned weighted sum over all prior sublayer outputs
- **BERT-style MLM head**
- **Per-example hybrid CLM+MNTP** objective (masked-focus = 6.25% causal), mask schedule 0.30→0.15

> Note: the model trains with bidirectional attention, but the exported HF `config.json` lists the attention
> blocks as `causal`. That's expected — `export-hf` emits a forced-choice (causal) head for the zero-shot
> evaluation; it is not config drift.

## Requirements

- **mixlab** — the Metal/MLX trainer. On macOS it installs via Homebrew; **read the
  [mixlab repo's README](https://github.com/mrothroc/mixlab) for the exact `brew` command and other install
  options**. This reproduction was produced with **mixlab v0.33.3** (MLX/Metal build).
- **BabyLM evaluation pipeline** — [`babylm-org/babylm-eval`](https://github.com/babylm-org/babylm-eval),
  pinned to commit `3bf5142` (it installs its own Python deps). Point `BABYLM_PIPELINE` at the strict-eval
  directory of your clone; `scripts/run_zero_shot.sh` uses it. Before evaluating, **populate its
  `evaluation_data/` via the pipeline's data-download step** (see the pipeline's `strict/` README) — note
  **EWoK requires accepting the dataset terms on HuggingFace** plus a filter step.
- **Training corpus** — the 2026 detoxified Strict-Small set,
  [`BabyLM-community/BabyLM-2026-Strict-Small`](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small)
  (HF Datasets). Put the `*.txt` files in `./corpus/`.
- **Tokenizer** — the reference 16k tokenizer: `tokenizer.json` from
  [`BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus`](https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus).
  Put it in `./tokenizer/`.
- **Python** (for the data-prep script): `pip install -r requirements.txt`.

## Reproduce it

```bash
# 0. install the dependencies listed under Requirements above
# 1. prepare the corpus with <s> segment markers in the packed token stream (reference convention)
#    args: <tokenizer_dir (has tokenizer.json)> <corpus_dir (*.txt)> <out_shard_dir>
python scripts/prep_s_markers.py ./tokenizer ./corpus ./data/shards   # -> shards with <s> ~0.8% of tokens

# 2. train (single Apple GPU, ~5.5h on an M1 Max)
MIXLAB_MLX_CACHE_LIMIT_MB=8192 mixlab -mode arch \
  -config configs/gptbert_masked_focus.json \
  -train 'data/shards/train_*.bin' \
  -safetensors final.safetensors -checkpoint-dir checkpoints -checkpoint-every 2000

# 3. export to HuggingFace + verify native-vs-HF parity (must be ~1e-7)
mixlab -mode export-hf -config configs/gptbert_masked_focus.json \
  -safetensors-load final.safetensors -tokenizer-path ./tokenizer/tokenizer.json -output hf_export
mixlab -mode parity -config configs/gptbert_masked_focus.json \
  -safetensors-load final.safetensors -hf hf_export -train 'data/shards/val_*.bin' -parity-python python

# 4. evaluate the component tasks via the official BabyLM pipeline
bash scripts/run_zero_shot.sh hf_export results logs   # the 6 zero-shot tasks (runs on the Mac)
# (Super)GLUE is a fine-tune step (7 tasks). Same harness, same hyperparams, but it
# needs a GPU (RunPod Pod / Colab / local CUDA), not the Mac:
bash scripts/run_glue.sh hf_export glue_results glue_logs
```

The exact training configuration is in [`configs/gptbert_masked_focus.json`](configs/gptbert_masked_focus.json).

## What it took in mixlab

Reproducing this baseline required ≈15 mixlab features (all built into the trainer): affine-free LayerNorm with
configurable placement, GeGLU with internal norm, DeBERTa disentangled relative attention with the
reuse-content-QK parameterization, the attention value-gate, dense (DWA) layer aggregation, a BERT-style
masked-LM export head, per-example hybrid CLM+MNTP mixing, z-loss, warmup/cosine/cooldown scheduling, a
long-run memory fix, and an MLX cache cap. One data-presentation detail mattered as much as the architecture:
inserting `<s>` segment markers into the packed training stream (the reference convention) was needed to
recover sentence-pair GLUE performance.

## Implementation Notes

- This reproduces the **2025** baseline. (The official BabyLM **2026** baselines are GPT-2, a different architecture.)
- **Corpus:** we trained on the **2026 detoxified** Strict-Small corpus, not the original 2025 corpus the reference model used (detoxification was introduced in the 2026 round). So this reproduces the 2025 architecture and recipe on the current, debiased data rather than a data-identical rerun. The six sources are unchanged and the filter removes only a small fraction of sentences, so the effect on these (non-toxicity) benchmarks should be minimal; we kept the detoxified corpus as the better, current dataset.
- We compare both models within one identical harness. Absolute scores can differ across eval harnesses for the same model (e.g. BLiMP scoring varies between pipeline versions), so the right comparison is per-component within one harness, which is what's shown.
- Single training seed; several components (reading, entity tracking) are noisy at this scale.
- Reproduced on Apple-silicon; we have not characterized numerical differences from a CUDA run.

## Artifacts

- Model: [`mrothroc/mixlab-gptbert-masked-focus-replica`](https://huggingface.co/mrothroc/mixlab-gptbert-masked-focus-replica) (HF)
- Trainer: [mixlab](https://github.com/mrothroc/mixlab)
- Reference: [`BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus`](https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus)
