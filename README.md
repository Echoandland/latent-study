# latent-study

Minimal, corpus-specific latent memory for strict offline Machine Studying. The experiment compares one learned global soft prefix with a frozen textual PEEK map while keeping Qwen3.5-9B, DSPy corpus access, search implementation, root-agent loop, and evaluation budgets fixed.

This repository is an executable MVP scaffold, not a completed benchmark claim. The source/corpus audit, a bounded 50-record frozen-base generation, and L=3 GPU pilots are retained as repair evidence; no full corpus training or StudyBench evaluation was started.

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

The PyTorch wheel should be chosen for the host CUDA version when the generic lock entry is unsuitable. The repair host used PyTorch 2.6.0+cu124, Transformers 5.3.0, and four RTX 6000 Ada GPUs; the pinned `fla`/`causal-conv1d` fast kernels were unavailable, so Qwen timing uses the Transformers torch fallback.

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

Generation can be split into independent deterministic workers with `--num-workers K --worker-index i`. Candidate seeds are derived from the global seed and record ID, not worker-local order. Merge shards only through `latent-study merge-records --expected-workers K --input ... --output ...`; it rejects missing/duplicate shards, duplicate record IDs, and corpus/config/model/tool mismatches, then emits deterministic record-ID order.

Without `--candidate-model`, the command fails closed unless `--deterministic-smoke` is explicitly given. The latter is only for tests and is labeled in every record. Multi-fact records carry separate required evidence groups. Full visible evidence after normal truncation—not path overlap—drives reward. Semantic and within-document position distances remain disabled.

Before freezing a study artifact, run the separate evaluation-contamination audit. It may read evaluation files, but no study/training command does:

```bash
latent-study audit-contamination \
  --study artifacts/study/dspy_records.jsonl \
  --corpus data/corpus/dspy \
  --evaluation data/evaluation/studybench \
  --output artifacts/study/contamination_audit.json
```

Add `--shuffle-correspondence` to create the compute/coverage-matched prompt–evidence negative control. Query-only and rank-only ablations use `--lambda-rank 0` and `--query-weight 0`, respectively; latent length accepts 16, 64, or 256 (and 2–4 for smoke).

## Train the latent

Create the untrained-random L64 control with the same initializer and corpus provenance:

```bash
latent-study init-latent --model Qwen/Qwen3.5-9B --length 64 --seed 17 \
  --manifest artifacts/audit/dspy_manifest.json \
  --corpus-hash <manifest.corpus_hash> \
  --contamination-audit artifacts/study/contamination_audit.json \
  --output artifacts/checkpoints/dspy_random_L64.pt
```

```bash
CUDA_VISIBLE_DEVICES=0 latent-study train-latent \
  --manifest artifacts/audit/dspy_manifest.json \
  --records artifacts/study/dspy_records.jsonl \
  --contamination-audit artifacts/study/contamination_audit.json \
  --model Qwen/Qwen3.5-9B \
  --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --length 64 --dtype bfloat16 --replay 0.5 \
  --output artifacts/checkpoints/dspy_L64.pt \
  --report artifacts/study/dspy_L64_training.json
```

Only `Z_D` is passed to AdamW. Base parameters have `requires_grad=False`, but forward computation is not wrapped in `no_grad`, so gradients reach input-prefix embeddings. Query loss is a reward-weighted, length-normalized pairwise action loss over action tokens only; it is not DPO. Ranking uses the same frozen LM's next-token `A-B` logit difference, with actual-tokenizer single-token validation and loss averaged over every required evidence group.

One shared serializer defines the logical prompt order `system/tool instructions -> corpus memory slot -> user/history`. Textual PEEK tokens occupy that slot; the soft prefix embeddings are spliced at the slot's exact token boundary. Training, candidate generation, ranking, and the root loop use this serializer. Attention masks and positions include the soft memory once. Qwen3.5 cached continuation is not currently claimed: the exact path is explicitly named `full_recompute_fallback` and root runs require `--acknowledge-full-recompute-fallback`.

## Study PEEK offline

These commands invoke upstream `peek.CachePolicy`; local code only converts the same corpus records into PEEK trajectories and freezes the resulting map. The Qwen tokenizer enforces the map budget.

```bash
latent-study peek-study --manifest artifacts/audit/dspy_manifest.json \
  --records artifacts/study/dspy_records.jsonl \
  --contamination-audit artifacts/study/contamination_audit.json \
  --output artifacts/study/offline_peek_64.json --token-budget 64 \
  --model Qwen/Qwen3.5-9B --internal-max-new-tokens 1024 --replay 0.5

latent-study peek-study --manifest artifacts/audit/dspy_manifest.json \
  --records artifacts/study/dspy_records.jsonl \
  --contamination-audit artifacts/study/contamination_audit.json \
  --output artifacts/study/offline_peek_1024.json --token-budget 1024 \
  --model Qwen/Qwen3.5-9B --internal-max-new-tokens 1024 --replay 0.5
```

The 64-token condition starts from PEEK's valid section syntax without its explanatory initial filler, because the stock template alone exceeds 64 tokens. The Distiller/Cartographer/priority Evictor policy remains upstream code. The adapter validates Distiller and Cartographer schemas separately, retries deterministic generations with the exact expected schema, and fails without writing a result if an update fails or the final map is empty, header-only, non-readable, or lacks navigation content.

## Replay, evaluation, and tests

```bash
latent-study replay-report --manifest artifacts/audit/dspy_manifest.json \
  --records artifacts/study/dspy_records.jsonl \
  --replay 0.5 --output artifacts/study/replay_50.json
latent-study replay-report --manifest artifacts/audit/dspy_manifest.json \
  --records artifacts/study/dspy_records.jsonl \
  --replay 0 --output artifacts/study/replay_0.json
```

The production `evaluate` command loads a separate frozen JSON/JSONL dataset, validates all five frozen conditions and their exact portable dependency hashes, and verifies that every memory's contamination audit names that exact evaluation snapshot. It loads one frozen backbone and leases one condition runner at a time, so five condition objects cannot retain five model copies. Omit `--budget` to run every configured budget, or repeat it to select a subset, for example `--budget direct --budget max5`. `--judge module:function` injects the evaluation-only strict/lenient scorer. The command requires explicit acknowledgement of the conservative Qwen `full_recompute_fallback`.

```bash
latent-study evaluate --manifest artifacts/audit/dspy_manifest.json \
  --dataset data/evaluation/studybench \
  --random-latent artifacts/checkpoints/dspy_random_L64.pt \
  --trained-latent artifacts/checkpoints/dspy_L64.pt \
  --peek64 artifacts/study/offline_peek_64.json \
  --peek1024 artifacts/study/offline_peek_1024.json \
  --budget direct --budget max5 --budget max20 --judge my_eval:score \
  --acknowledge-full-recompute-fallback --output artifacts/evaluation/mvp.json
```

```bash
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
  --point 20000:30 --point 100000:40  # debugging only
```

Production evaluation does not require manually entered points. For each condition and evaluation budget it emits one aggregate curve point: x is mean generated tokens per example over the benchmark at that budget, and y is aggregate strict or lenient score. The historical 3k-token anchor and integration rule are preserved. The standalone `expertise --point` command remains only as a debugging utility.

Primary downstream conditions are no study, random latent L64, trained latent L64, offline PEEK 64, and offline PEEK 1024. They must use identical root-agent, search limits, corpus, output constraints, and inference budgets. The evidence scorer is training/diagnostic-only and is not an evaluation-time reranker.

## Current repair validation status

The current schema-v3 provenance contract cryptographically binds artifact content, protocol and command configuration identities, canonical corpus/evaluation snapshots, and portable parent dependencies. The historical corpus audit produced 371 decoded text documents, 2,571 non-overlapping semantic units, an estimated 513,596 eligible lexical tokens, 4,567 symbol occurrences, 14,048 syntactic call sites, and 1,733 conservatively verified call relations. It reports 90 ambiguous, 2,953 lexically shadowed, and 9,272 otherwise excluded call sites. These are tokenizer-independent historical audit estimates; actual model-token exposure must be reported during training.

CPU/mock validation currently collects 58 tests. It additionally covers protocol-versus-command configuration compatibility, portable dependency relocation, unified semantic corpus identity, current-dataset contamination binding, sequential shared-backbone evaluation, budget-level aggregation, manifest-authoritative PEEK setup, and per-run CUDA peak reset.

All older checked-in record, replay, latent, PEEK, root-agent, and real-Qwen result artifacts predate the current artifact contract and are historical only. Production loaders reject them rather than silently treating them as current. In particular, PEEK-64 and the real-Qwen latent optimization smoke remain blocked; Round 3 does not weaken either gate. The current infrastructure status is in [REPAIR_ROUND_3_REPORT.md](docs/REPAIR_ROUND_3_REPORT.md).
