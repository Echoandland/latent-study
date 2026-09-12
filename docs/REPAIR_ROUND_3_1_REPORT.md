# Repair Round 3.1 report — 2026-09-12 UTC

> Historical report. Repair Round 3.1.1 supersedes its claim that anchor-spanning configured caps alone establish runtime reachability. See `docs/REPAIR_ROUND_3_1_1_REPORT.md` for the current budget and study-bank-consumer gates.

## Status

The narrowly scoped Round 3.1 infrastructure repairs are complete in the working tree. Feasible CPU/mock validation passes: **62 passed, 0 failed, 0 skipped**. No full L=64 training, official benchmark evaluation, or expensive real-model/PEEK run was performed.

The branch is `repair/offline-machine-studying`, based on commit `fdd41c0f7d78b5391bfc7f6531c31a737ecdfa76`. The current implementation source-tree hash is `8d3c1fbdb6e0b9faa8676e60ca4d9cf8cfb25d0935edb3ffb6e24de10038a8f3`; it was reproduced from both the repository root and `/tmp`.

## Exact fixes

### Expertise budget semantics

Expertise remains the historical best-so-far weighted curve with a 3,000-token anchor. Its x coordinate is now defined as the **configured generated-token cap per evaluation example** for a budget profile. Its y coordinate is the benchmark-wide mean strict or lenient score for the same condition and budget. Thus four examples under a 3,000-token cap produce one `(3000, aggregate_score)` point, not four points and not a 12,000-token point.

The generated-token cap and tool/agent iteration limit are separate fields. Actual mean and total generated tokens, tool calls, observations, latency, and memory remain separate efficiency measurements and do not silently replace the declared x coordinate.

The production `dspy_mvp` profiles are now:

| Profile | Tool-call semantics | Generated-token cap per example |
|---|---:|---:|
| `direct` | 0 | 1,000 |
| `max5` | ≤5 | 3,000 |
| `max20` | ≤20 | 6,000 |
| `exact20` | exactly 20/no early return | 10,000 |

`budgets_from_config()` and production `run_evaluation()` validate anchor coverage. Selecting only a non-spanning subset such as `direct` fails before runner execution with an explicit error. Production evaluation still emits one automatic expertise-ready point per `condition × budget`; manual `--point` input is only a debugging utility. The small repair-smoke configuration explicitly disables the production anchor-coverage gate so unit/synthetic work is not turned into an expensive run.

### Effective configuration and CLI overrides

The configuration flow is now:

```text
base config -> apply protocol-relevant CLI values -> effective config
            -> protocol_config_view/effective protocol hash
            -> command config hash plus complete CLI provenance
```

The centralized `PROTOCOL_CLI_OVERRIDE_PATHS` mapping prevents a command from using one scientific setting while recording a hash for another. Production readers continue to receive the caller's expected effective protocol hash; they never use an artifact's stored hash as its own expectation.

Protocol/scientific identity includes:

- canonical model revision/dtype/frozen state;
- corpus revision and semantic snapshot;
- latent length/init and shared memory placement;
- candidate count, generation/sampling controls and candidate dtype;
- tool interface and search/result/query limits;
- reward and query/rank objective weights, margin/delta, gradient balance and rank labels;
- replay fraction/policy, seed, optimizer, learning rate, weight decay, batch/accumulation and updates per source;
- PEEK structured-generation/retry settings;
- root prompt/decoding semantics;
- evaluation conditions, complete tool/generated-token budget profiles and expertise definition.

Purely operational/local CLI values include output and input locations, the local model load path, device, local-cache mode, worker assignment/count, and acknowledgement flags. Bounded smoke/data-selection controls such as `max_records`, shard/limit selectors and temporary paths remain command-local; their exact effect is still bound by the command hash, artifact content and dependency hashes. Selecting a subset of already-declared evaluation budgets is command-local, while changing any budget definition is protocol identity.

Tests prove `--actions 99` changes the effective candidate count and protocol hash, changing only the output path does not, and an artifact carrying the former hash fails against the latter.

### Actual model/tokenizer snapshot identity

Provenance schema is now version 4 and adds required nullable fields:

- `model_snapshot_sha256`
- `tokenizer_snapshot_sha256`

Whenever a model is actually loaded, both are non-null. The canonical model ID and claimed revision remain separately recorded.

`resolve_model_snapshot()` resolves an explicit local directory or an immutable Hugging Face revision to the directory that will actually be passed to Transformers. It hashes the contents and relative names of recognized model files (configuration, indexes and weight shards) separately from recognized tokenizer files (tokenizer configuration/data, vocabulary, merges and chat templates). A process-local cache avoids rehashing unchanged large files repeatedly; its key includes path, size, mtime and ctime metadata, while the persisted identity itself is based on file bytes, not metadata or absolute path.

Candidate generation, latent training/initialization, PEEK study, root-agent execution, production evaluation and the real-model smoke/reload scripts persist these identities where they load a model. Latent/PEEK/evaluation readers compare the current actual snapshot hashes. The five-condition preflight also requires all four memory artifacts to carry one identical non-null model/tokenizer snapshot pair. Synthetic tests mutate weights and tokenizer content while retaining the same claimed ID/revision and observe separate fail-closed mismatches.

### Real experiment-bundle relocation

Production corpus manifests no longer persist their original absolute corpus root as identity. They store an artifact-relative logical root and an artifact-relative `corpus_snapshot` dependency. `_read_manifest()` resolves that dependency relative to the current manifest location and uses the resolved live root at runtime. Live verification no longer requires equality with an original machine path; it still rebuilds the canonical eligible-file snapshot and rejects every eligible modification, addition or removal.

The integration test constructs a realistic bundle containing a corpus, provenance-bearing manifest and dependent study artifact, copies the whole bundle to another temporary root, and validates both the relocated dependency chain and live corpus. It then separately modifies, adds and removes eligible source files and confirms fail-closed behavior.

## Equivalent-pattern audit

The audit also covered model loads in candidate generation, PEEK, latent initialization/training, root-agent evaluation, production evaluation and real-model smoke scripts; worker merge compatibility now compares snapshot identities as well. The evaluation preflight propagates the common snapshot pair into its own provenance. No latent loss, evidence/reward behavior, PEEK update algorithm, or five-condition definition was changed.

## Files changed

- `src/latent_study/config.py`, `cli.py`, `evaluation.py`, `artifacts.py`, `corpus.py`, `latent.py`
- new `src/latent_study/snapshots.py`
- `configs/dspy_mvp.json`, `configs/repair_smoke.json`
- `schemas/artifact_provenance.schema.json`
- `scripts/real_model_smoke.py`, `scripts/reload_latent_smoke.py`
- new `tests/test_repair_round31.py` and compatibility updates to Round 2/3/config tests
- `README.md`, `docs/PROTOCOL.md`, historical Round 3 banner, and this report

Historical untracked repair artifacts were preserved and not promoted to schema 4.

## Validation and exact results

```bash
python3 -m compileall -q src tests scripts
PYTHONPATH=src ./.venv/bin/python -m pytest -q
# 62 passed, 0 failed, 0 skipped

PYTHONPATH=src ./.venv/bin/python -m pytest -q \
  tests/test_repair_round31.py tests/test_repair_round3.py tests/test_repair_round2.py
# 30 passed

python3 -m json.tool schemas/artifact_provenance.schema.json >/dev/null
for f in configs/*.json; do python3 -m json.tool "$f" >/dev/null; done
git diff --check

PYTHONPATH=src ./.venv/bin/python - <<'PY'
from latent_study.config import load_config
from latent_study.evaluation import budgets_from_config
c = load_config("configs/dspy_mvp.json")
print([(p.name, p.max_tool_calls, p.max_output_tokens) for p in budgets_from_config(c)])
PY
# [('direct', 0, 1000), ('max5', 5, 3000),
#  ('max20', 20, 6000), ('exact20', 20, 10000)]
```

The test suite uses tiny synthetic model/tokenizer files; it does not hash or load the real 9B model and makes no real-model performance claim. No optional test was skipped. Real GPU validation was outside this repair's authorized scope.

## Explicit acceptance verification

1. **Yes — expertise can be computed with configured production budgets.** The budget axis reaches 3,000 and extends to 10,000 tokens, and aggregate strict/lenient points are emitted automatically. A non-spanning selected curve fails explicitly.
2. **Yes — protocol-relevant CLI overrides change `protocol_config_hash`.** This is tested with candidate action count, while an output-path-only change leaves it unchanged; incompatible artifacts fail closed.
3. **Yes — local model/tokenizer contents are bound to provenance.** Separate byte-content snapshot hashes are persisted and checked; changing either resource under an unchanged claimed ID/revision invalidates compatibility.
4. **Yes — a complete unchanged experiment bundle can move to another filesystem root.** The relocated corpus, manifest and dependent study artifact pass, while eligible corpus drift after relocation fails.

## Unresolved research blockers

- **Real Qwen L=3 latent optimization remains blocked.** Its prior tiny real-model validation has not passed; Round 3.1 did not change the latent objective or optimizer method.
- **PEEK-64 useful-map validation remains blocked.** Round 3.1 did not change PEEK's algorithm, final 64-token budget, or meaningful-map gate.
- Consequently the project is **not declared ready** for full L=64 training or official benchmark evaluation.
