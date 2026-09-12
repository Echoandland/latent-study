# Repair Round 3 report — 2026-09-11 UTC

> Historical report. Repair Round 3.1 supersedes its expertise-axis, effective CLI configuration, local model snapshot, and manifest-root portability claims. See `docs/REPAIR_ROUND_3_1_REPORT.md`.

## Status

Round 3 infrastructure and evaluation repairs are complete and pass the feasible CPU/mock gate: **58 passed, 0 failed, 0 skipped**. No full L=64 training, official StudyBench evaluation, or expensive PEEK study was run. The real-Qwen latent optimization and PEEK-64 quality failures remain blocked exactly as requested.

The work is uncommitted on branch `repair/offline-machine-studying`, based on commit `3bb381c1fe9dc638d4e68d0a3a714e19f2dbd526`. The repaired implementation source-tree hash is `e43325bb412ee62691a3913b754f3b1e161cf70c7df7587aac1f1331effefaa2`; the same value was computed from the repository root and `/tmp`.

## Issues fixed

1. **Evaluation model lifetime.** Production evaluation loads the tokenizer and frozen Qwen backbone once. `SequentialRunnerFactory` creates one ephemeral condition/budget runner at a time around that shared object, rejects nested leases, and releases the runner before the next is built. It no longer constructs five closures retaining five models. A weak-reference model-spy regression proves the previous runner is dead before the next factory call and all runners see the same backbone identity. CUDA cache clearing remains cleanup only, not the lifetime mechanism.

2. **Performance versus compute.** Evaluation now runs the same examples at every selected budget and emits one aggregate point per `condition × budget`. Each point records example count, mean strict/lenient score, valid count/rate, and mean/total/measured-count compute fields. Expertise x is **mean generated tokens per example for the complete benchmark at that budget**; y is the benchmark mean strict or lenient score. Total generated tokens are retained for auditing but are not the x coordinate. The historical 3,000-token anchor remains. Per-example rows remain available for diagnosis but never become curve points. A four-example fixture producing 1,000 tokens each yields one 1,000-token point, not four points.

3. **Evaluation-dataset contamination binding.** JSON/JSONL evaluation inputs now have a canonical content snapshot identity. A passing contamination audit embeds that identity. Frozen memory attestation binds both the audit SHA256 and evaluation snapshot SHA256. Before loading the backbone, evaluation resolves and validates the portable audit dependency, its own provenance/content hash, the memory's attestation, and equality with the current dataset snapshot. Tests cover a match, A-versus-B mismatch, post-audit dataset mutation, and audit tampering.

4. **Configuration compatibility.** Provenance schema 3 carries both `protocol_config_hash` and command-level `resolved_config_hash`. `protocol_config_view()` is the single canonical projection. It includes protocol identity; model ID/revision/frozen state/dtype; corpus revision; memory placement, serializer revision, latent length/init; candidate-generation semantics; tool interface and limits; reward; objective and weights; root prompt revision; decoding; and evaluation conditions/budgets. It deliberately excludes paths/output locations, worker/shard placement, batch sizing, retry counts, and similar command-local controls. Those remain recorded by the resolved hash and dependencies, but do not impose false cross-command equality. Production manifest, study-bank, latent, PEEK, and evaluation readers receive the caller's expected protocol hash rather than trusting an artifact's own value.

5. **Unified corpus identity.** `_scan_corpus_files()` is now the sole eligibility policy used by manifest creation, `canonical_corpus_snapshot()`, `corpus_snapshot_hash()`, and live verification. Identity is the canonical ordered eligible relative path/content-hash set. Changes under `.git`, virtual environments, caches, unsupported extensions, and non-UTF-8 files do not alter it. Eligible modifications, additions, and removals fail a closed-snapshot check.

6. **Manifest-authoritative PEEK setup.** The PEEK command loads and live-verifies the manifest first, then passes its corpus hash as the expected value when loading and validating records. A D2 record bank cannot bootstrap its own claimed identity while running against D1. Both a direct study-function regression and CLI setup regression prove failure occurs before client/model construction. Equivalent self-referential record validation in replay setup was corrected.

7. **Per-run accelerator metrics.** Each root-agent example resets CUDA peak-memory statistics immediately before the measured run and reads peak memory afterward. The reported scope includes the live shared backbone and that example's generation/tool loop. If CUDA or reset support is unavailable, the value is null with a reason. A mocked regression verifies reset → inference → peak-read ordering.

8. **Portable dependency identity.** Artifact dependencies use a logical role, artifact/experiment-relative path, semantic hash kind, content SHA256, artifact type, and embedded provenance identity. Absolute paths may not be compatibility identity. Schema-3 readers resolve relative paths from the dependent artifact/repository and verify content plus embedded provenance. A complete bundle copied to a different temporary root remains valid with unchanged content and relative layout.

9. **Documentation and related-pattern audit.** `README.md` and `docs/PROTOCOL.md` now describe schema 3, dual config hashes, semantic corpus/evaluation snapshots, dataset-bound contamination, sequential shared-backbone evaluation, repeatable budgets, budget-level aggregation, and the exact expertise input. The Round 2 report is explicitly historical. Production searches also corrected portable dependency construction, manifest/record ordering, command readers that trusted stored hashes, and canonical model/tokenizer identity when a local load path is used.

## Provenance/dependency model

Every schema-3 persisted artifact binds its primary content digest independently of the provenance sidecar. Dependencies are role-based and portable. Supported semantic dependency kinds include `artifact_content`, `corpus_snapshot`, and `evaluation_dataset_snapshot`; corpus identity is never computed with a generic recursive directory hash. Compatibility checks jointly enforce artifact schema/type, content, embedded parent provenance identity, source-tree hash when current source is required, protocol hash, corpus/tool/model/tokenizer fields, phase, and contamination chain where applicable. `resolved_config_hash` remains command provenance and is not mislabeled as an artifact hash.

## Evaluation budgets and metrics

`evaluate --budget` is repeatable; omitting it selects every configured profile. Tool/agent iteration limits (`direct`, `max5`, `max20`, `exact20`) remain distinct from model output-token limits. Exact/no-early-return constraints are validated on each result. For each condition/budget the output reports mean and total input tokens, generated tokens, tool calls, observation bytes, total latency, prefill latency, and decode latency, with measured counts. Peak accelerator memory is the maximum of the per-example reset-bounded measurements. Unsupported backend metrics are null with an explanation rather than estimates presented as measurements.

The standalone `expertise --point` utility remains for debugging only. Normal production evaluation emits `performance_vs_compute` and `expertise_input_points` automatically.

## Files changed

- Core: `src/latent_study/{agent,artifacts,cli,config,corpus,evaluation,isolation,latent,peek_baseline}.py`
- Schema/config-facing tools: `schemas/artifact_provenance.schema.json`, `scripts/real_model_smoke.py`, `scripts/reload_latent_smoke.py`
- Tests: `tests/test_repair_round3.py`, plus schema-3 fixture updates in `tests/test_repair_round2.py` and `tests/test_protocol_repairs.py`
- Documentation: `README.md`, `docs/PROTOCOL.md`, `docs/REPAIR_ROUND_2_REPORT.md`, and this report

Existing untracked historical/generated repair artifacts were preserved and were not promoted to schema 3.

## Validation executed

Environment: Ubuntu/Linux host, repository `/nas03/yucheng/latent-study`; system Python 3.10.12; project virtual environment Python 3.11.16. `nvidia-smi` could not communicate with the NVIDIA driver in this session, so no real-GPU smoke is reported.

```bash
python3 -m compileall -q src tests scripts
PYTHONPATH=src ./.venv/bin/python -m pytest -q
# 58 passed, 0 failed, 0 skipped

PYTHONPATH=src ./.venv/bin/python -m pytest -q tests/test_repair_round3.py
# 9 passed

python3 -m json.tool schemas/artifact_provenance.schema.json >/dev/null
for f in configs/*.json; do python3 -m json.tool "$f" >/dev/null; done
git diff --check

PYTHONPATH=src ./.venv/bin/python -m latent_study.cli evaluate --help
# Confirms repeatable --budget and default-all behavior.

PYTHONPATH=/nas03/yucheng/latent-study/src \
  ./.venv/bin/python -c 'from latent_study.artifacts import source_tree_hash; print(source_tree_hash())'
cd /tmp
PYTHONPATH=/nas03/yucheng/latent-study/src \
  /nas03/yucheng/latent-study/.venv/bin/python -c \
  'from latent_study.artifacts import source_tree_hash; print(source_tree_hash())'
# Both: e43325bb412ee62691a3913b754f3b1e161cf70c7df7587aac1f1331effefaa2
```

The full suite includes synthetic end-to-end root-agent evaluation across all five memory conditions and Round 3 multi-condition/multi-budget evaluation. The PEEK mismatch checks require no expensive generation. No real PEEK study or Qwen training/evaluation was attempted.

## Acceptance statements

- **Yes:** production evaluation can run all five conditions without retaining five Qwen backbones; it reuses one backbone and permits at most one live runner.
- **Yes:** expertise uses budget-level benchmark aggregate points, never individual examples.
- **Yes:** a memory audited against evaluation dataset A fails evaluation on dataset B.
- **Yes:** scientifically meaningful projected protocol changes invalidate incompatible artifacts.
- **Yes:** an output-path or excluded command-local change alone does not invalidate protocol-compatible artifacts.
- **Yes:** corpus provenance and live verification use one semantic snapshot definition.
- **Yes:** PEEK rejects records from a corpus different from the authoritative manifest before model execution.
- **Yes:** CUDA peak statistics are reset and measured per root-agent run when CUDA supports them; otherwise the metric is explicitly unavailable.
- **Yes:** an unchanged artifact/dependency bundle remains valid after relocation when its relative layout is preserved.

## Remaining blockers and project classification

| Category | Status | Evidence/consequence |
|---|---|---|
| A. Infrastructure/evaluation correctness | **Pass for CPU/mock scope** | 58/58 tests plus compile, schema/config parse, CLI, relocation, and CWD-independence checks pass. Real GPU execution was unavailable in this session. |
| B. Real latent optimization validation | **Blocked** | The retained real-Qwen L=3 query-only smoke previously worsened about 1.9871 → 2.0824. No loss/objective change was made in Round 3. |
| C. PEEK-64 baseline validation | **Blocked** | Structured output is valid, but the current ≤64-token smoke map remains empty/non-navigable. Its budget/gates were not weakened. |
| D. Full L=64 / official benchmark readiness | **Blocked** | B and C have not independently passed; no official StudyBench result is claimed. |

Review this repair before any expensive experiment. The exact next validation commands, once a working GPU/pinned model environment is available, remain the bounded real-model smoke commands in the README; do not expand them into full training or official evaluation until categories B and C pass.
