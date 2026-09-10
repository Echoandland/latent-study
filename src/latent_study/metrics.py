from __future__ import annotations

import math


def expertise(points: list[tuple[float, float]], anchor_tokens: float = 3000.0) -> float:
    """Official Machine Studying best-so-far staircase weighted AUC.

    `points` are (generated_tokens, performance) with performance on any consistent
    scale. Weight mass from b_i to b_{i+1} is anchor/b_i-anchor/b_{i+1}; the tail
    is anchor/b_last. Scores below the first measurement are zero.
    """
    if anchor_tokens <= 0 or not points:
        raise ValueError("positive anchor and at least one point required")
    merged: dict[float, float] = {}
    for budget, score in points:
        if budget < anchor_tokens or not math.isfinite(budget) or not math.isfinite(score):
            raise ValueError("budgets must be finite and at least the anchor; scores finite")
        merged[budget] = max(score, merged.get(budget, -math.inf))
    ordered = sorted(merged.items())
    best, total = -math.inf, 0.0
    for index, (budget, score) in enumerate(ordered):
        best = max(best, score)
        next_budget = ordered[index + 1][0] if index + 1 < len(ordered) else math.inf
        mass = anchor_tokens / budget - (0.0 if math.isinf(next_budget) else anchor_tokens / next_budget)
        total += best * mass
    return total


def studybench_weighted_score(rubric: list[dict], satisfied_claim_ids: set[str], *, core_gate: bool) -> float:
    """Compute StudyBench's published claim-weighted rubric score after judging claims."""
    if not rubric or sum(int(c["weight"]) for c in rubric) != 100:
        raise ValueError("StudyBench rubric weights must sum to 100")
    if core_gate and any(c["claim_type"] == "core" and c["claim_id"] not in satisfied_claim_ids
                         for c in rubric):
        return 0.0
    return float(sum(int(c["weight"]) for c in rubric if c["claim_id"] in satisfied_claim_ids))
