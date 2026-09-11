# Repair report — 2026-09-11 UTC (Repair Round 1, historical)

> Superseded by [REPAIR_ROUND_2_REPORT.md](REPAIR_ROUND_2_REPORT.md). The schema-v1/source hashes and GPU results below are retained as historical evidence only; current readers enforce the Round 2 schema-v2 contract.

## Outcome

The repository now fails closed on stale artifacts, evaluation contamination, incompatible memories, failed/empty PEEK maps, unacknowledged Qwen full recomputation, and non-decreasing training objectives. CPU/mock validation passes. A bounded GPU validation completed: 50 frozen-base records (N=4) and matched L=3 replay pilots were produced. The real PEEK-64 run failed its structured-output gate, and the strengthened real-Qwen smoke failed its query-only overfit gate; no invalid latent/map is promoted and no full training or StudyBench evaluation was started.

The validation stop is scientific, not a hardware absence: four RTX 6000 Ada GPUs were visible outside the conversation sandbox. PEEK-1024 and all downstream conditions remain unrun because the required 64-token PEEK condition could not reliably produce structured updates.

## Environment and exact commands

- Host: `Linux COE-CS-sv003 5.15.0-179-generic x86_64`
- Python 3.11.16; PyTorch 2.6.0+cu124; Transformers 5.3.0; pytest 9.1.1. `nvidia-smi` reported four RTX 6000 Ada Generation GPUs (49,140 MiB each, roughly 48,639 MiB free at discovery).
- Repository HEAD: `b97f17531035302de83dd6a6a94a92110e7ee786` (working-tree repair changes are uncommitted).
- Relevant source-tree hash for current artifacts: `cd08762d5dab8a8fe1484d44584c5e177fe79ea85add5e8ba36b539f42008d80`.
- Pinned PEEK source inspected at `8b109771b51126284ea337f23827facde1db05ed`; upstream files were not modified.
- Model files identify revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; config loads as Qwen3.5 with hidden size 4096.

Commands completed:

```bash
PYTHONPATH=src:/tmp/latent-study-peek/src .venv/bin/python -m pytest -q
PYTHONPATH=src:/tmp/latent-study-peek/src .venv/bin/python -m compileall -q src tests scripts
PYTHONPATH=src .venv/bin/python -m latent_study.cli audit-corpus \
  --config configs/dspy_mvp.json --corpus data/corpus/dspy \
  --output artifacts/repair/corpus_manifest_v10.json
git diff --check

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -m latent_study.cli generate-records \
  --config configs/dspy_mvp.json \
  --manifest artifacts/repair/corpus_manifest_v10.json \
  --output artifacts/repair/dspy_records_50_v10.jsonl \
  --coverage-report artifacts/repair/dspy_coverage_v10.json \
  --probes artifacts/repair/dspy_probes_v10.json \
  --candidate-model data/models/Qwen3.5-9B \
  --candidate-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --candidate-device cuda:0 --candidate-dtype bfloat16 --local-files-only \
  --actions 4 --seed 17 --limit 25 --family-limit 15 --relation-limit 10
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -m latent_study.cli train-latent \
  --config configs/dspy_mvp.json \
  --manifest artifacts/repair/corpus_manifest_v10.json \
  --records artifacts/repair/dspy_records_50_v10.jsonl \
  --model data/models/Qwen3.5-9B --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --device cuda:0 --dtype bfloat16 --length 3 --max-records 6 \
  --replay 0 --batch-size 2 --updates-per-source 1 --learning-rate 0.001 \
  --output artifacts/repair/latent_pilot_v10_replay0.pt \
  --report artifacts/repair/latent_pilot_v10_replay0.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -m latent_study.cli train-latent \
  --config configs/dspy_mvp.json \
  --manifest artifacts/repair/corpus_manifest_v10.json \
  --records artifacts/repair/dspy_records_50_v10.jsonl \
  --model data/models/Qwen3.5-9B --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --device cuda:0 --dtype bfloat16 --length 3 --max-records 6 \
  --replay 0.5 --batch-size 2 --updates-per-source 1 --learning-rate 0.001 \
  --output artifacts/repair/latent_pilot_v10_replay50.pt \
  --report artifacts/repair/latent_pilot_v10_replay50.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:/tmp/latent-study-peek/src .venv/bin/python -m latent_study.cli peek-study \
  --config configs/dspy_mvp.json \
  --records artifacts/repair/dspy_records_50_v10.jsonl \
  --output artifacts/repair/offline_peek_64_v10.json \
  --model data/models/Qwen3.5-9B --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --device cuda:0 --local-files-only --max-records 6 --token-budget 64 \
  --replay 0.5 --retries 2  # failed closed
```

The current manifest's resolved config hash is `1d7f0b1cb8de8ad0cda3183a2efc125788d6c37749ae7067a25c953b20470f32`; candidate-bank hash is `b6409c40368afe762667a1c6ae798c9f38dc0d7e0885f686c6c41eb904b8bd01`; final replay=0 and replay=50% pilot hashes are `873a0ea54905e5b99c2c4a73a94a02fe18cf2eabe0e5d6ba135d510ccef17f73` and `0e433a61f2e8186ecb0f1a6844d225bf4d27daadb214a93bc864ead69ccf3733`; PEEK-64 hash is `72bfe1334f0a0e16ed4db49ba9accb376b45e586bf119517d7d0ce57219e6577`. Corpus hash is `55af7e1e61f774069507505603743e271e11ee6808b6cc10b53c47cbf352fe6e`; tool-schema hash is `75c3c6a62d9ce650e2f6989c440e8586fdc97d82ba0be82c9caa526b0c89a69e`.

## Tests and compatibility

- CPU/mock: 32 passed, 0 failed, 0 skipped by pytest.
- GPU bounded validation ran on `CUDA_VISIBLE_DEVICES=0`; the independent real-Qwen smoke and PEEK-64 gate failed closed as described below.
- New artifact schema: version 1. JSON artifacts embed provenance; JSONL banks use a required `.provenance.json` sidecar; latent checkpoints embed the same contract.
- Provenance includes repository commit, source-tree hash, resolved config hash, model/revision, corpus hash, tool-schema hash, command, and relevant CLI overrides.
- Production readers reject all older checked-in repair record, coverage, replay, PEEK, root-agent, and latent/result artifacts because they lack current provenance or have a stale source-tree hash. They remain untouched as historical evidence.
- The fresh manifest is compatible with the current source. The corpus directory lacks its own `.git`, so its claimed upstream Git revision cannot be independently checked from that directory; the content hash matches both older manifests.

## Corpus and record audit

Fresh corpus audit:

- 371 documents; 2,571 semantic units; 513,596 lexical audit tokens; 4,567 symbol occurrences.
- 14,048 syntactic call sites.
- 1,733 accepted uniquely verified calls; 90 ambiguous; 2,953 shadowed; 9,272 otherwise excluded.
- Shadowed breakdown: 697 name calls, 2,255 receiver calls, and 1 reassigned `self`/`cls` receiver.
- Duplicate import exclusions: 22 symbol bindings and 58 module bindings.

The resolver now accounts conservatively for parameters, local/annotated assignments, loops, with/exception bindings, comprehensions, nested definitions, imported aliases, and global/nonlocal declarations. Regression cases cover parameter shadowing, reassignment, nested definitions, duplicate imports, ambiguous methods, and a valid unique call.

Current v10 bank (frozen Qwen3.5-9B, N=4, 50 records) has family counts comprehension 25, relation_induction 5, navigation 5, evidence_interpretation 2, misconception_correction 13; validator counts are atomic_structural_definition 25, uniquely_resolved_ast_call 10, ast_assignment_or_import 2, and single_verified_name_substitution 13. It contains 200 candidate actions: valid rate 0.750, global unique rate 0.770, within-record unique rate 0.845, preference-pair rate 0.860, exact visible-hit rate 0.305, and required-group coverage 0.3475. All 50 records passed complete rendered evidence exposeability. The Qwen generation used 74,920 input and 12,596 output tokens over 50 model calls (182.23 s model latency; 193.48 s wall time). Relation records now carry a caller observation and a declared callee follow-up sequence; the deterministic sequence exposes both required groups before base-model candidate replacement. These are exposure/coverage measurements, not evaluation accuracy.

The prior two-record deterministic CLI smoke remains explicitly `base_model_candidates=false` historical smoke. Its tool-loop result (8 actions, zero invalid, 2/2 full evidence) is retained separately and is not merged with the v10 bank.

## Training and replay

The deterministic four-record CPU production-path test used AdamW (`lr=0.01`, `weight_decay=0.01`, 10 updates/source):

| Objective | Before | After |
|---|---:|---:|
| query | 5.5197752419 | 5.4462088091 |
| rank | 0.5124232583 | 0.5124179162 |
| combined | 6.0321985002 | 5.9586267253 |

The prefix changed, the frozen LM parameter digest was bitwise unchanged, and LM gradient-buffer count was zero. The production `train()` path now computes aggregate diagnostics over up to eight records and refuses to save when the active objective fails to decrease. Query-only, rank-only, and combined checks are explicit; the combined update is the real optimizer path. The prior synthetic line search was removed from the real-Qwen smoke.

The current v10 Qwen L=3 pilots used the exact first six records of the v10 bank, AdamW (`lr=0.001`, `weight_decay=0.01`), batch size 2, one update/source, and the actual bfloat16 model. Both passed all aggregate checks and preserved the frozen LM:

| Pilot | Query before → after | Rank before → after | Combined before → after | Peak GPU | Deployable prefix |
|---|---:|---:|---:|---:|---:|
| replay=0 | 2.633669 → 2.526374 | 0.095998 → 0.041899 | 2.729668 → 2.568273 | 34,737,531,904 B | 49,152 B |
| replay=50% | 2.633669 → 2.518186 | 0.095998 → 0.026646 | 2.729668 → 2.544831 | 34,737,531,904 B | 49,152 B |

Each had `prefix_changed=true`, `frozen_lm_bitwise_unchanged=true`, and `lm_gradient_buffers=0`. Replay=0 current-source exposures were `{dspy/dsp/utils/utils.py: 2, dspy/propose/utils.py: 2, tests/adapters/test_tool.py: 2, tests/clients/test_cache.py: 2, tests/primitives/test_python_interpreter.py: 4}`; replay=50% previous exposures were `{dspy/dsp/utils/utils.py: 2, dspy/propose/utils.py: 1, tests/adapters/test_tool.py: 1, tests/clients/test_cache.py: 1}`, with the last source having no later opportunity. The separate strengthened synthetic smoke still fails closed in query-only (1.987137 → 2.053502); this is recorded as a smoke blocker, while the real six-record combined pilots are the valid aggregate production evidence.

The dynamic replay integration adds seven sources sequentially, exactly as production does. Previous-source exposures were `[1,1,1,1,1,1,0]`: min 0, max 1, range 1, population standard deviation 0.3499271061. The final source had no later replay opportunity; zero-exposure eligible sources were empty. The scheduler is deterministic least-exposed selection among currently eligible previous sources. Current and previous ledgers are reported separately. Replay=0 retains matched update/batch compute and cycles the current source, preserving unique current-source coverage.

No current real-Qwen L=64 pilot was run; the bounded L=3 pilots above are the only real-Qwen training runs.

## PEEK

The adapter retains upstream `CachePolicy`, `ContextMap`, `Distiller`, `Cartographer`, edit, scoring, and eviction behavior. It now validates Distiller and Cartographer outputs separately, gives concise schema-specific retry feedback, decodes deterministically, records calls/tokens/retries/failures/latency/peak memory, separates map tokens/bytes from total artifact bytes, and refuses failed, empty, header-only, non-readable, or non-navigable maps.

Historical, rejected evidence:

- PEEK-64: 6 updates, 18 calls, 12 malformed outputs, 6 retries, 6 failed outputs; 61,108 actual client input tokens, 4,009 output tokens, 207.23 s client latency. Final map: `## CONTEXT ROADMAP` — 5 actual Qwen tokens and 19 bytes. This is a failed baseline.
- PEEK-1024 semantic smoke: only one update over one record (not the same bank); 2 calls, 0 malformed outputs, 6,160 input tokens, 745 output tokens, 37.09 s client latency. Its 125-token/433-byte map begins: `Core utility classes (e.g., dotdict, dotdict_lax) are defined in dspy/dsp/utils/utils.py...`. It is historical smoke evidence, not a paired baseline.

The current paired subset was the first six records of `dspy_records_50_v10.jsonl`, with the same replay=50% schedule and actual Qwen tokenizer. PEEK-64 failed closed on the third Distiller stage after two completed policy updates: the client reported 7 model calls, 22,721 input tokens, 1,974 output tokens, 97.20 s client model latency, 3 malformed Distiller outputs, 2 retries, and 1 failed update. The diagnostic artifact is `offline_peek_64_v10.json` with `status=failed`, `phase=study_failed`, and no frozen result. It records 19,144,202,752 B peak GPU allocation, 58 tokenizer tokens/258 map bytes, and 6,469 complete artifact bytes; that partial map is not a baseline. The client records malformed-output rate 3/7, retry rate 2/7, and failure rate 0.25 per attempted update pair. The malformed samples show truncated/invalid JSON despite deterministic decoding and schema-specific retries. Per protocol, PEEK-1024 was not launched after the required 64-token condition failed; no meaningful excerpt exists for either current budget, and the older unpaired 1024 smoke remains historical only.

## Serialization, root loop, and cache status

The shared serializer exposes exact `memory_start`/`memory_end` token boundaries in this order:

```text
system and tool instructions -> <corpus_memory> memory slot </corpus_memory> -> user/history
```

PEEK map tokens fill the interval; latent embeddings splice at its start. Tests assert identical role-delimited tokens before and after the slot and exercise training, candidate, ranking, tool continuation, and all five root-agent memory conditions. The five-condition synthetic test emitted a typed `read_file`, executed it, appended the exact clipped observation as a tool response, continued, and returned an answer supported by visible `fact.py:1-2` evidence. Output/tool/observation budgets were enforced; latent insertion count was one.

Pinned Transformers 5.3 Qwen3.5 uses its recurrent DeltaNet branch only when the appended sequence length is one. A multi-token observation passed as a block takes the non-recurrent branch with `initial_state=None`; earlier artifacts also document non-equivalent chunk/recurrent behavior. Native cache equivalence is therefore not claimed. The exact path is named `full_recompute_fallback`, reports latency/scaling separately, and production root runs require `--acknowledge-full-recompute-fallback`. Earlier zero-error “cache” numbers compared full recomputation with full recomputation and are rejected.

## Blocked validation commands

The host GPU was available outside the sandbox. Candidate generation and both bounded latent pilots therefore ran. The strengthened real-Qwen smoke was attempted with the pinned model and stopped fail-closed in its query-only objective gate:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:/tmp/latent-study-peek/src .venv/bin/python scripts/real_model_smoke.py \
  --model data/models/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --length 3 --seed 17 --overfit-steps 2 --optimizer AdamW \
  --checkpoint artifacts/checkpoints/repair_smoke_v10_l3.pt \
  --output artifacts/repair/real_qwen_smoke_v10.json
```

Observed current-source smoke failure: query-only aggregate objective `1.987137 → 2.053502` (three optimizer updates; nonzero prefix gradients), so no smoke checkpoint/report was written. The six-record production pilots did pass aggregate query/rank/combined checks, but this independent smoke gate remains unresolved and must be reviewed before full L=64 training. PEEK-64 similarly failed after its allowed retries; PEEK-1024 and real five-condition Qwen root runs are intentionally blocked. No OpenClaw, full L=64 run, or public StudyBench evaluation was started during this review gate.

The L=3 pilots provide the study-token, latency, peak-GPU, deployable-prefix, and checkpoint-byte measurements shown above. No deployable L=64 latent, successful current PEEK map, or five-condition real-Qwen root measurement exists because those runs were blocked by the gates above. The code records latent tensor/checkpoint bytes, PEEK map token/bytes, complete artifact bytes, generated-token cost, latency, and peak memory as separate fields; it does not equate 64 FP32 soft tokens with 64 text tokens.
