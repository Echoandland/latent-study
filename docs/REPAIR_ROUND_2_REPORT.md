# Repair Round 2 report — 2026-09-11 UTC

## Status

The infrastructure repairs are complete and the repository is ready for review. CPU/mock validation passes. The expensive experiment was intentionally not started: no full L=64 training, full-corpus mining, OpenClaw run, or official StudyBench evaluation was launched.

The working tree is uncommitted on branch `repair/offline-machine-studying`. `HEAD` is `5a4792d8949c000fa22b34c1569cb95d864c5d7f`; the repaired source-tree hash (computed independently of CWD) is `108c2c5f1e38975774db469a7777da49f8e60a05e96b7a51fb9b11f3979fc97d`.

Resolved base configuration hashes are:

| configuration | hash |
|---|---|
| `configs/dspy_mvp.json` | `ab0e4f0710c3de3ebe63e99b07cf43bf9c674cca776ba25068c9f16efce68a20` |
| `configs/repair_smoke.json` | `f3e8056f81dc9999a3a87335bea159397ac1eec4705f2947fc88acbe198d239f` |

The pinned model is `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; the tool-schema hash is `75c3c6a62d9ce650e2f6989c440e8586fdc97d82ba0be82c9caa526b0c89a69e`. The tokenizer is the model tokenizer at the same revision whenever a tokenizer-dependent artifact is produced.

## 1. Issues fixed

- Artifact provenance is now schema v2. Persisted JSON/JSONL/checkpoint artifacts carry an `artifact_sha256` content digest and exact dependency descriptors. Dependencies are content-hashed, not merely named. Provenance also records repository commit, deterministic source-tree hash, resolved configuration hash, model/revision, tokenizer identity/revision, corpus hash, tool-schema hash, command, and CLI overrides.
- JSON/JSONL mutation and dependency mutation fail closed in production readers. `read_sidecar`, `validate_json_artifact`, latent loading, map loading, condition preflight, and evaluation validate the primary content digest and dependency digests. `resolved_config_hash` remains a configuration identity field; it is checked whenever an expected configuration identity is supplied and is never presented as the artifact hash.
- Artifact schema versions are present in persisted reports/maps/manifests and in the provenance schema. Older checked-in artifacts remain untouched and are rejected as stale/incompatible rather than silently reinterpreted.
- A single live-corpus verifier now compares the frozen manifest against the filesystem before study, tool initialization, or evaluation. It detects modified, missing, newly eligible, excluded-set, root, path, and corpus-hash drift without regenerating the manifest.
- Tool actions use one schema-derived parser for model output and runtime execution. Unknown properties, wrong types, invalid integer minima, invalid ranges, and unknown tools become controlled invalid-action/tool-error results.
- Evaluation contamination is audited in a separate pre-freeze command. Only an audit with `status=pass` and its own content digest can attest a study artifact as `frozen_before_evaluation`.
- PEEK internal structured-generation allowance is independent of the final map budget. Distiller and Cartographer have separate validation/statistics and schema-specific retries; retries use a larger allowance and deterministic decoding. Empty/header-only/non-navigable/over-budget maps and failed updates remain ineligible.
- `source_tree_hash()` resolves the repository from the module/project root (or an explicit project-root override), not from the caller's CWD.
- README commands and CLI names are synchronized (`peek-study` uses `--model`; the stale `--tokenizer` spelling is gone), and help parsing is covered by a regression test.
- The production evaluation harness now loads a separate frozen JSON/JSONL dataset, requires exactly the five MVP conditions, applies explicit direct/max-5/max-20/exact-20 budgets, accepts an evaluation-only strict/lenient judge, and emits per-example performance-versus-compute data without manual `--point` entry.
- Root-agent metrics now separate serialized input tokens, memory tokens, generated tokens, tool calls, post-truncation observation bytes, latency, prefill/decode latency, and peak accelerator memory. Backend-unavailable measurements are `null` with an explanation. Exact-20/no-early-return profiles force a continuation request until the required tool-call count or fail the budget; they do not silently accept an early final answer.

## 2. Files changed

The implementation changes are in `src/latent_study/artifacts.py`, `io.py`, `corpus.py`, `search.py`, `isolation.py`, `agent.py`, `latent.py`, `peek_baseline.py`, `evaluation.py`, `cli.py`, `config.py`, `replay.py`, and the compute-matched scheduling path in `train.py`. The pinned training method and optimizer remain in place. Configuration/schema updates are in `configs/dspy_mvp.json`, `configs/repair_smoke.json`, and `schemas/artifact_provenance.schema.json`. CLI/documentation and smoke tooling are in `README.md`, `scripts/real_model_smoke.py`, `scripts/reload_latent_smoke.py`, and `scripts/round2_peek_smoke.py`. Focused regressions are in `tests/test_repair_round2.py` and the replay integration assertions in `tests/test_integration.py`.

Existing generated artifacts, including old repair artifacts, were not deleted. Their old provenance/schema is treated as historical and incompatible by current loaders.

## 3. Provenance and dependency model

`artifact_content_hash()` uses canonical semantic JSON/JSONL content, tensor/metadata content for `.pt` files, and deterministic recursive file hashes for directories. The self-referential provenance digest is the only field excluded while calculating an artifact's own digest. A downstream artifact stores exact parent paths and SHA256 values, for example:

```text
trained_latent checkpoint
  -> study record-bank SHA256
  -> corpus-manifest SHA256
  -> probe-bank SHA256 (when used)
  -> contamination-audit SHA256

offline PEEK map
  -> study record-bank SHA256
  -> corpus-manifest SHA256
  -> contamination-audit SHA256
  -> model/tokenizer identity and revision
```

The acceptance case was exercised: a valid JSONL bank plus sidecar loaded successfully, one record was modified without changing the sidecar, and strict production loading raised `ArtifactCompatibilityError` for a content-digest mismatch. A modified dependency similarly failed before use.

## 4. Corpus integrity

`verify_manifest_against_live_corpus()` rebuilds an in-memory index with the manifest's exact inclusion/exclusion rules and compares root, eligible paths, file hashes, excluded paths/reasons, and corpus hash. It never writes a replacement manifest. `CodingTools(..., manifest=manifest)` invokes this check before registering tools; all production study/evaluation commands invoke it before model construction. The regression test modifies a corpus file and confirms both direct verification and tool initialization fail before the loop runs.

## 5. Tool-schema validation

`TOOL_SCHEMAS` is the model-visible definition and the source of runtime validation. The following all reject deterministically and return a controlled tool error at the runtime boundary:

```json
{"tool":"grep","query":"x","extra":1}
{"tool":"grep","query":"x","path":123}
{"tool":"glob","pattern":"*.py","max_results":-1}
```

Malformed JSON, malformed actions embedded in model output, and tool implementation type errors are recorded as `invalid_action`/tool errors rather than escaping the study or root-agent loop.

## 6. Evaluation-contamination boundary

`latent-study audit-contamination` is the only command that receives evaluation paths. It fingerprints normalized evaluation IDs, questions/prompts, answer/rubric/content fields, and sufficiently distinctive six-token content shingles. It reports pass/fail, collision type, study artifact location, evaluation location, and evaluation fingerprint. Training, latent initialization, and PEEK receive only the resulting attestation; their audit loader verifies the immutable attestation and study/corpus dependencies without reopening evaluation files.

Synthetic tests cover a clean case, copied evaluation ID, and copied evaluation question. Frozen artifact gates require the passing audit's SHA256 in addition to the artifact's own provenance digest, `phase=frozen_before_evaluation`, current schema, and `evaluation_inputs_seen=false`.

## 7. PEEK repair and smoke result

The adapter retains the pinned upstream `CachePolicy`, `ContextMap`, `Distiller`, `Cartographer`, edit, scoring, and eviction behavior. Internal structured-output generation defaults to 1024 tokens and retries can use 1536; the final map remains constrained by the actual Qwen tokenizer to exactly the requested 64- or 1024-token budget. Distiller and Cartographer malformed/retry/failure counters, input/output tokens, calls, latency, and peak memory are recorded separately. PEEK prefill/decode latency is stored as `null` because this backend reports one combined `generate` latency, with an explicit unavailability reason.

The bounded paired smoke used the same two synthetic records, same replay schedule, same local Qwen model/tokenizer, and both budgets:

| budget | result | diagnostics |
|---|---|---|
| 64 | failed closed | Structured calls were valid (2 calls, 0 malformed, internal allowances `[1024, 1024]`), but the final map was empty/non-navigable; diagnostic phase `study_failed`, map accounting 11 tokens/45 bytes. No baseline was frozen. |
| 1024 | generated but unattested | 2 calls, 0 malformed, 4,605 input tokens, 499 output tokens, about 23.6 s model latency; map 75 actual tokenizer tokens/301 UTF-8 bytes. The excerpt was `## CONTEXT UNDERSTANDING [cu-00001] The corpus contains facts.py, a utility file defining atomic mathematical functions. Key entities include lunar_checksum (adds 7) and solar_checksum (adds 9).` The artifact is `study_complete_unattested` because this development smoke intentionally supplied no contamination attestation, so it is not an evaluation baseline. |

The reproducible bounded harness is `scripts/round2_peek_smoke.py`. Its expected exit is nonzero when the 64-token gate fails; it still prints both budget diagnostics. This result is conservative: the Round 2 adapter no longer turns a failed/empty map into a successful baseline. The historical truncated-JSON PEEK artifacts remain historical evidence and are rejected by the current schema/source contract.

## 8. Evaluation harness status

`run_evaluation()` and the `evaluate` CLI require exactly:

```text
no_study
random_latent_L64
trained_latent_L64
offline_peek_64
offline_peek_1024
```

All conditions share model/revision, corpus snapshot, tool schema, root prompt revision, decoding, tool limits, and the selected budget. The condition preflight validates latent length/hidden size/model/tokenizer/corpus/phase and PEEK budget/tokenizer/model/corpus/phase. Budget profiles distinguish model output-token limits from tool-iteration limits; direct, <=5, <=20, and exact-20/no-early-return are explicit configuration entries. A judge can return a number or `{strict, lenient}`; absent judge scores are null, never fabricated.

The evaluator emits input/output tokens, tool calls, observation bytes, latency, unavailable metric explanations, score records, condition summaries, performance-versus-compute points, and an expertise-ready artifact. The CLI also requires the evaluation dataset to be under a configured evaluation root and rejects overlap with authorized study roots. No public StudyBench dataset or judge was used during this repair, so no official accuracy or expertise result is claimed.

## 9. Validation executed

Environment:

```text
Python 3.11 (.venv)
PyTorch 2.6.0+cu124
Transformers 5.3.0
PEEK pinned source 8b109771b51126284ea337f23827facde1db05ed
```

Commands and results:

```bash
./.venv/bin/python -m pytest -q
# 49 passed, 0 failed, 0 skipped

python3 -m compileall -q src tests scripts
# passed

git diff --check
# passed

./.venv/bin/python -m pytest --collect-only -q
# 5 + 2 + 3 + 1 + 15 + 17 + 6 = 49 tests

PYTHONPATH=src ./.venv/bin/python -m latent_study.cli --help
PYTHONPATH=src ./.venv/bin/python -m latent_study.cli peek-study --help
# passed; peek-study exposes --model and no --tokenizer

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ./.venv/bin/python \
  scripts/real_model_smoke.py \
  --model data/models/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --length 3 --seed 17 --overfit-steps 2 --optimizer AdamW \
  --checkpoint artifacts/repair/round2_real_qwen_smoke_l3.pt \
  --output artifacts/repair/round2_real_qwen_smoke.json
# fail-closed in train(): query-only objective 1.9871366620 -> 2.0824265281

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:/tmp/latent-study-peek/src \
  ./.venv/bin/python scripts/round2_peek_smoke.py \
  --model data/models/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --device cuda:0
# expected nonzero: PEEK-64 failed closed; paired PEEK-1024 produced the
# 75-token/301-byte unattested map described above
```

`nvidia-smi` on the host showed four RTX 6000 Ada GPUs (49,140 MiB each). A host-side `CUDA_VISIBLE_DEVICES=0` Python check reported `cuda_available=True`, one visible device, and `NVIDIA RTX 6000 Ada Generation`; an un-escalated conversation `.venv` process reports no device because NVML is not exposed there. The bounded Qwen commands above were run with the host-side device access. The fast `fla`/`causal-conv1d` kernels were unavailable, so Transformers used its torch fallback.

## 10. Retained research validation and metrics

The prior bounded Round 1 evidence is retained but is not silently promoted: its artifacts predate the Round 2 source hash and are rejected by current readers. For review context, that bank contained 50 records with family counts comprehension 25, relation induction 5, navigation 5, evidence interpretation 2, and misconception correction 13; validator counts were atomic structural 25, uniquely resolved relation 10, assignment/import 2, and single-name substitution 13. It had 200 actions, valid-action rate 0.750, preference-pair rate 0.860, exact visible-hit rate 0.305, and required-group coverage 0.3475. Its relation resolver audit was 1,733 accepted, 90 ambiguous, 2,953 shadowed, and 9,272 excluded calls. These values are historical exposure/coverage measurements, not Round 2 evaluation results.

The CPU production-optimizer regression still reports aggregate query/rank/combined decreases (5.5197752419→5.4462088091, 0.5124232583→0.5124179162, and 6.0321985002→5.9586267253), frozen LM bitwise identity, changed prefix, and zero LM gradient buffers. The new real-Qwen length-3 smoke failed its query-only gate as reported above; no checkpoint was promoted. The retained bounded length-3 pilot measurements and dynamic replay distributions are historical only; the replay integration still reports deterministic least-exposed scheduling, separate current/previous exposure ledgers, zero-exposure eligible sources, min/max/range/stdev, and the final late-source no-opportunity reason. Both replay controls now use the same compute-matched shard-step schedule; replay=0 fills the matched batch with current records without reducing optimizer updates or unique current-record coverage.

Synthetic root-loop tests exercise all five memory conditions: typed tool action, shared tool execution, exact post-truncation tool observation, continued reasoning, final visible evidence, budget enforcement, one latent insertion, and map-only shared-slot insertion. Native Qwen cache equivalence remains unverified; the explicit production status is `full_recompute_fallback`, with no cache-correctness or efficiency claim. Latent tensor/checkpoint bytes, PEEK map tokens/bytes, complete artifact bytes, generated tokens, latency, and peak memory remain separate measurements; a 64-token FP32 prefix is not storage-equivalent to 64 text tokens.

## 11. Unresolved blockers and deviations

- The strengthened real-Qwen optimizer smoke fails the required query-only overfit gate despite exercising the actual production AdamW path. This must be resolved before full L=64 training.
- The paired real PEEK-64 smoke fails the meaningful/navigation-map gate. The 1024-token paired smoke is readable but deliberately unattested. Consequently neither is an eligible frozen evaluation baseline.
- Native cached Qwen3.5 continuation equivalence is not available; full recomputation requires explicit acknowledgement and is not presented as cache support.
- No official evaluation dataset/judge, full corpus mining, full L=64 training, or StudyBench result is claimed.
- The pinned model ran through the Transformers torch fallback because optional fast kernels were not installed; timings should not be generalized as optimized-kernel performance.

## 12. Reproduction and handoff

Run the CPU commands in Section 9 from the repository root. For host GPU checks, use the pinned model path/revision and explicit `CUDA_VISIBLE_DEVICES=0`; do not remove the length/record limits from the smoke scripts. Before any evaluation, run `audit-corpus`, `generate-records`, `audit-contamination`, both study artifact commands, and `validate-conditions`; all five memories must pass current content/dependency/provenance checks. Stop after review of this report; no expensive experiment is launched automatically.
