# Protocol invariants

1. Study reads only the pinned corpus tree and corpus-derived artifacts.
2. Evaluation questions, answers, rubrics, trajectories, rewards, and prompt patterns never enter generation, training, replay, tuning, or selection.
3. Latent and PEEK map learning finishes before evaluation; both artifacts carry `phase=frozen_before_evaluation` and are immutable across questions.
4. The external corpus and the same search implementation remain available during evaluation.
5. Every condition uses the same frozen LM, root loop, corpus, tool/result limits, output constraints, and inference budget.
6. Candidate action observations are the actual post-limit/post-truncation strings. Evidence counts only if the verified span is visible.
7. Alternative spans for one fact share a group; different necessary facts use different groups. Complete reward requires all groups.
8. Only verified negatives are trained. Similarity or non-overlap alone is never negative proof.
9. Downstream evaluation never changes study data, hyperparameters, checkpoints, prompts, or coverage priorities.
10. Every current artifact uses provenance schema version 3. Its content SHA256, portable dependency roles/relative paths/content hashes, repository commit, source-tree hash, `protocol_config_hash`, command-level `resolved_config_hash`, model/tokenizer identity and revision, corpus snapshot hash, tool-schema hash, command, and CLI overrides are recorded. Missing, mutated, or incompatible metadata is a hard error.
11. The only memory location is inside the initial system role after tool/root instructions and before the user/history. PEEK text fills this token interval; latent embeddings are spliced at its start boundary.
12. Qwen3.5 `full_recompute_fallback` is correctness-preserving recomputation, not cache validation or an efficiency claim. Evaluation requires explicit acknowledgement while native cached equivalence remains unverified.
13. Corpus identity is the canonical ordered set of eligible UTF-8 protocol files and their relative-path/content hashes. `.git`, virtual environments, caches, unsupported extensions, and non-UTF-8 files are outside that identity; every eligible modification/addition/removal invalidates the closed snapshot.
14. A frozen memory's contamination audit is cryptographically bound to the canonical evaluation-dataset snapshot. Production evaluation verifies memory -> audit -> current dataset before loading the backbone.
15. Evaluation runs one shared frozen backbone and leases one condition/budget runner at a time. Performance-versus-compute contains one benchmark aggregate point per condition and budget. Expertise uses mean generated tokens per example as compute and the benchmark mean strict/lenient score as performance; per-example rows are never curve points.
