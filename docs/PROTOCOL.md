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

