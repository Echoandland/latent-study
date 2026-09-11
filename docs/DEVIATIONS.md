# Verified setup, deviations, and unresolved assumptions

## Reused from PEEK

PEEK is pinned at `8b109771b51126284ea337f23827facde1db05ed`. `offline_peek_map` imports and executes its `CachePolicy`, `ContextMap`, Distiller, Cartographer, structured edits, score update, and priority/age eviction. Its Apache-2.0 license and NOTICE are preserved. The local adapter converts already-verified corpus records and actual tool observations into the trajectory text accepted by PEEK.

## Adapted locally

- PEEK's normal protocol is online and uses inference-time task trajectories. Here all updates use corpus-derived study records before evaluation; the map is then frozen. This is the primary fair baseline, not original online/transductive PEEK.
- The PEEK initial annotated map is larger than a 64-token budget. Both baseline sizes therefore start from `## CONTEXT ROADMAP` with no filler, retaining upstream map syntax and update/eviction behavior.
- Historical smoke used a whitespace token approximation. Production `peek-study` now requires the actual Qwen tokenizer; the current PEEK attempt used it and is reported below.

## Implemented locally

Corpus semantic-unit/hash manifests, AST evidence validation, shared discrete search/truncation, required evidence groups, action reward/preferences, soft-prefix integration, pairwise action/ranking losses, source-stratified replay, isolation checks, official expertise calculation, and experiment reporting are local.

PEEK does not provide StudyBench, its corpus checkout, a root coding-agent harness, corpus-study record tooling, or a differentiable model interface. None is attributed to PEEK here.

## Machine Studying and StudyBench verification

The primary blog/paper define expertise as a normalized weighted area over the best-achieved performance curve on a log generated-token axis. This implementation uses the specified 3k anchor and `w(x)=ln(10)10^-x`. Its worked 5k/10k/20k/100k example is regression-tested at 10.8%.

The official Hugging Face dataset revision `11e27d7...` contains 30 DSPy and 20 OpenClaw questions, gold answers, weighted rubrics, and source excerpts. It declares DSPy commit `9cdb0aac...`. It does not ship an executable agent loop or judge implementation. The published grading description is implemented once claim decisions exist, but generating those decisions still requires a declared judge. Consequently exact downstream reproduction is not currently possible from the released repository alone.

The release is fully public, not hidden. This project treats it as evaluation-only; no evaluation rows were copied into the repository or opened by study commands. Public availability also means future results must be described as open-benchmark results, not hidden-exam generalization.

## Smoke deviations

- Qwen3.5-9B was loaded at the pinned revision on an RTX 6000 Ada GPU. The bounded frozen-base run generated 50 records with N=4; L=3 replay=0 and replay=50% pilots passed the aggregate objective gate. The strengthened synthetic real-Qwen smoke's query-only branch still rises (and therefore fails closed); no latent is promoted from that smoke. Fast `fla` and `causal_conv1d` kernels were not installed, so all measurements use Transformers' slower PyTorch fallback.
- The 6-record smoke uses deterministic structural candidates with `N=2`, not frozen-base sampling. Every record marks `base_model_candidates=false`.
- No full record bank, full latent training, checkpoint selection, OpenClaw run, literature task, or expensive experiment was started.
- The current trainer is single-process/single-GPU. Records and shards are deterministic and generation workers can be run independently, but DDP synchronization is not implemented yet. Tensor parallelism and distributed frameworks were intentionally not added.
- Corpus “eligible tokens” in the audit are deterministic lexical estimates. Actual Qwen tokenizer counts must replace them in final exposure reports.
- Fixed corpus probes are held-out prompt templates within the same fully trained corpus, not unseen-corpus generalization. They are not used for checkpoint selection.

## Repair status (2026-09-11)

- Older checked-in result artifacts are retained but fail provenance-schema validation and are historical only.
- The prior 64-token PEEK artifact is an empty header-only map and is explicitly a failed baseline. The 1024-token semantic smoke used a different tiny bank, so it is not a paired comparison.
- The pinned Qwen3.5 implementation only uses the recurrent DeltaNet path for one-token cached inputs. Multi-token tool observations cannot be appended as a chunk without discarding linear-attention history. Earlier runs also observed non-equivalent torch chunk-versus-recurrent logits. The repaired code therefore retains exact full recomputation, labels it `full_recompute_fallback`, makes no cache/efficiency claim, and requires explicit acknowledgement.
- Real-Qwen candidate generation and both L=3 pilots were rerun. PEEK-64 was attempted on the exact six-record subset and failed after three malformed Distiller attempts; per protocol the paired PEEK-1024 run and downstream evaluation remain blocked rather than emitting a false 64-token baseline.

## Remaining requirements before a full run

1. Identify or freeze a concrete root coding-agent and judge harness; record its exact revision and prompts as evaluation-only artifacts.
2. Generate the complete study bank with base-model `N=4` actions, review coverage/exclusions and skipped preferences, then obtain review before full training.
3. Add synchronized DDP only if one-GPU throughput makes it necessary.
