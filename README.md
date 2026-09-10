# latent-study

Minimal, corpus-specific latent memory for strict offline Machine Studying. The experiment compares one learned global soft prefix with a frozen textual PEEK map while keeping Qwen3.5-9B, DSPy corpus access, search implementation, root-agent loop, and evaluation budgets fixed.

This repository is an executable MVP scaffold, not a completed benchmark claim. The source/corpus audit and deterministic smoke artifacts are checked in; no full corpus record generation or 9B training was started.

## Pinned scope

- Frozen model: `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Authorized corpus: DSPy code and tests at `9cdb0aac28b2a04b064e40697ccd301872cf6a43`.
- Evaluation data: StudyBench at `11e27d7c4e7ac98515ef383fe4bf6700ff6821f2` (evaluation-only).
- PEEK: `8b109771b51126284ea337f23827facde1db05ed`.
- Qwen3.5 support: Transformers `5.3.0`, tag commit `aad13b87ed59f2afcfaebc985f403301887a35fc`.

Exact hashes and dates are in [upstream_revisions.json](artifacts/audit/upstream_revisions.json). PEEK is a pinned VCS dependency. Its Apache-2.0 license and NOTICE, plus DSPy's MIT license, are preserved in `third_party/licenses/`.

## Isolation boundary

Use these non-overlapping paths:

```text
data/corpus/dspy/                 authorized study input
artifacts/audit/                  corpus-derived manifest
artifacts/study/                  records, probes, study outputs
data/evaluation/studybench/       downstream questions/rubrics; evaluation only
artifacts/evaluation/             frozen-condition results
```

Study-data generation is evidence-first and deterministic unless a frozen base model is explicitly supplied. It never reads StudyBench questions, answers, rubrics, trajectories, rewards, or prompt patterns. All study artifacts must be frozen before evaluation. StudyBench is currently public rather than hidden, which makes filesystem and process isolation especially important.

## Install

Python 3.11 is recommended. Qwen3.5 requires the pinned Transformers 5.3 implementation; the older 4.57 environment used for CPU tests cannot load it.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install -e .
```

The PyTorch wheel should be chosen for the host CUDA version when the generic lock entry is unsuitable.

## Reproduce source and corpus audit

```bash
git clone https://github.com/stanfordnlp/dspy.git data/corpus/dspy
git -C data/corpus/dspy checkout 9cdb0aac28b2a04b064e40697ccd301872cf6a43

latent-study audit-corpus \
  --corpus data/corpus/dspy \
  --output artifacts/audit/dspy_manifest.json
```

The manifest uses Python functions/classes and prose sections as semantic units. IDs include path, semantic span, and content hash. AST-derived definitions, references, and call relations are extracted; arbitrary overlapping windows are not counted as independent coverage.

## Build the study bank

Production record generation must sample `N=4` actions from the frozen base model without a latent and execute every result through the shared search tool:

```bash
latent-study generate-records \
  --manifest artifacts/audit/dspy_manifest.json \
  --output artifacts/study/dspy_records.jsonl \
  --coverage-report artifacts/study/dspy_coverage.json \
  --probes artifacts/study/dspy_probes.json \
  --candidate-model Qwen/Qwen3.5-9B \
  --actions 4 --seed 17
```

Generation can be split into independent deterministic workers with `--num-workers K --worker-index i`; concatenate their JSONL outputs and sort/deduplicate by `record_id` before training.

Without `--candidate-model`, the command fails closed unless `--deterministic-smoke` is explicitly given. The latter is only for tests and is labeled in every record. Multi-fact records carry separate required evidence groups. Full visible evidence after normal truncation—not path overlap—drives reward. Semantic and within-document position distances remain disabled.

Add `--shuffle-correspondence` to create the compute/coverage-matched prompt–evidence negative control. Query-only and rank-only ablations use `--lambda-rank 0` and `--query-weight 0`, respectively; latent length accepts 16, 64, or 256 (and 2–4 for smoke).

## Train the latent

Create the untrained-random L64 control with the same initializer and corpus provenance:

```bash
latent-study init-latent --model Qwen/Qwen3.5-9B --length 64 --seed 17 \
  --corpus-hash <manifest.corpus_hash> \
  --output artifacts/checkpoints/dspy_random_L64.pt
```

```bash
CUDA_VISIBLE_DEVICES=0 latent-study train-latent \
  --manifest artifacts/audit/dspy_manifest.json \
  --records artifacts/study/dspy_records.jsonl \
  --model Qwen/Qwen3.5-9B \
  --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --length 64 --dtype bfloat16 --replay 0.5 \
  --output artifacts/checkpoints/dspy_L64.pt \
  --report artifacts/study/dspy_L64_training.json
```

Only `Z_D` is passed to AdamW. Base parameters have `requires_grad=False`, but forward computation is not wrapped in `no_grad`, so gradients reach input-prefix embeddings. Query loss is a reward-weighted, length-normalized pairwise action loss over action tokens only; it is not DPO. Ranking uses the same frozen LM's next-token `A-B` logit difference, with actual-tokenizer single-token validation and loss averaged over every required evidence group.

The prefix is placed before the chat-template token embeddings. Attention masks, cache positions, and text/multimodal RoPE positions include it. Tool observations extend the same hybrid Qwen cache without inserting the prefix again. Checkpoints contain the prefix tensor and provenance independently of the LM.

## Study PEEK offline

These commands invoke upstream `peek.CachePolicy`; local code only converts the same corpus records into PEEK trajectories and freezes the resulting map. The Qwen tokenizer enforces the map budget.

```bash
latent-study peek-study --records artifacts/study/dspy_records.jsonl \
  --output artifacts/study/offline_peek_64.json --token-budget 64 \
  --tokenizer Qwen/Qwen3.5-9B --replay 0.5

latent-study peek-study --records artifacts/study/dspy_records.jsonl \
  --output artifacts/study/offline_peek_1024.json --token-budget 1024 \
  --tokenizer Qwen/Qwen3.5-9B --replay 0.5
```

The 64-token condition starts from PEEK's valid section syntax without its explanatory initial filler, because the stock template alone exceeds 64 tokens. The Distiller/Cartographer/priority Evictor policy remains upstream code.

## Replay, evaluation, and tests

```bash
latent-study replay-report --records artifacts/study/dspy_records.jsonl \
  --replay 0.5 --output artifacts/study/replay_50.json
latent-study replay-report --records artifacts/study/dspy_records.jsonl \
  --replay 0 --output artifacts/study/replay_0.json

latent-study smoke-eval --manifest artifacts/audit/dspy_manifest.json \
  --records artifacts/smoke/study_records.jsonl \
  --output artifacts/smoke/tool_evaluation.json

pytest -q

CUDA_VISIBLE_DEVICES=0 python scripts/real_model_smoke.py \
  --model data/models/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --length 3 --seed 17 \
  --checkpoint artifacts/smoke/real_model_latent_L3.pt \
  --output artifacts/smoke/real_model_smoke.json

latent-study expertise --point 5000:10 --point 10000:20 \
  --point 20000:30 --point 100000:40
```

The expertise command implements the primary specification exactly: best score achieved at or below each generated-token budget; log-token anchor 3k; `w(x)=ln(10) 10^-x`; zero below the first measured point; last score carried to infinity. The example above returns 10.8.

Primary downstream conditions are no study, random latent L64, trained latent L64, offline PEEK 64, and offline PEEK 1024. They must use identical root-agent, search limits, corpus, output constraints, and inference budgets. The evidence scorer is training/diagnostic-only and is not an evaluation-time reranker.

## Current audit and smoke status

The pinned DSPy snapshot produced 371 decoded text documents, 2,571 non-overlapping semantic units, an estimated 513,596 eligible lexical tokens, 4,567 symbol occurrences, and 10,284 structurally extracted relations in 2.05 seconds at 81,324 KiB peak RSS. These are tokenizer-independent audit estimates; actual model-token exposure must be reported during training.

The checked-in smoke bank contains 6 records (4 definition, 2 relation/navigation), two actions each, and deliberately covers only 6/2,571 units. The pinned official tokenizer/config verified `d=4096`, `A`=`token 32`, and `B`=`token 33`; both labels are exactly one token. CPU/mock tests cover isolation, prefix/cache, losses, exact evidence visibility, deterministic replay, and the official metric.

The real Qwen3.5-9B L=3 smoke passed on one RTX 6000 Ada: LM parameters frozen, prefix gradient nonzero, zero LM gradient buffers, bit-exact latent save/load logits, one prefix insertion across a two-token tool observation and two generated tokens. It took 11.00s total (7.14s load, 2.37s paired forward/backward) with 19,441,522,176 bytes peak allocated GPU memory; the standalone latent artifact is 50,589 bytes. See [real_model_smoke.json](artifacts/smoke/real_model_smoke.json) and [DEVIATIONS.md](docs/DEVIATIONS.md) before interpreting any result.
