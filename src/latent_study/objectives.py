from __future__ import annotations


def query_pairwise_loss(pos_mean_logp, neg_mean_logp, reward_gap: float, *, beta: float = 1.0,
                        max_gap: float | None = None):
    """Length-normalized pairwise action loss (not DPO)."""
    import torch.nn.functional as F
    gap = min(reward_gap, max_gap) if max_gap is not None else reward_gap
    if gap <= 0:
        raise ValueError("reward_gap must be positive")
    return gap * F.softplus(-beta * (pos_mean_logp - neg_mean_logp))


def select_preference(outcomes, delta: float):
    ordered = sorted(outcomes, key=lambda o: (o.reward, o.action.query))
    if len(ordered) < 2 or ordered[-1].reward - ordered[0].reward < delta:
        return None
    return ordered[-1], ordered[0]


def ranking_group_loss(positive_scores, negative_scores, *, margin: float = 1.0,
                       aggregation: str = "max"):
    import torch
    import torch.nn.functional as F
    if not positive_scores or not negative_scores:
        raise ValueError("each group requires positives and verified negatives")
    positives = torch.stack(positive_scores)
    if aggregation == "max":
        positive = positives.max()
    elif aggregation == "logsumexp":
        positive = torch.logsumexp(positives, dim=0)
    else:
        raise ValueError("aggregation must be max or logsumexp")
    return torch.stack([F.softplus(margin - positive + negative) for negative in negative_scores]).mean()


def ranking_loss(groups, *, margin: float = 1.0, aggregation: str = "max"):
    import torch
    losses = [ranking_group_loss(pos, neg, margin=margin, aggregation=aggregation) for pos, neg in groups]
    if not losses:
        raise ValueError("ranking record has no required evidence groups")
    return torch.stack(losses).mean()

