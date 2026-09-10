from __future__ import annotations

from dataclasses import dataclass

from .schema import ActionOutcome, EvidenceGroup, ToolAction, ToolHit


@dataclass(frozen=True)
class RewardConfig:
    hit_weight: float = 1.0
    group_coverage_weight: float = 2.0
    first_rank_weight: float = 1.0
    valid_weight: float = 0.2
    broad_result_penalty: float = 0.08
    returned_kib_penalty: float = 0.02
    semantic_distance_enabled: bool = False
    positional_distance_enabled: bool = False


def visible_groups(hits: tuple[ToolHit, ...], groups: tuple[EvidenceGroup, ...]) -> tuple[str, ...]:
    visible = []
    for group in groups:
        matched = False
        for alternative in group.alternatives:
            for hit in hits:
                # Evidence must survive truncation, not merely share a chunk ID.
                if _span_visible(hit, alternative):
                    matched = True
        if matched:
            visible.append(group.group_id)
    return tuple(visible)


def _span_visible(hit: ToolHit, span) -> bool:
    """An evidence hit counts only if its exact path/range/text survived rendering."""
    from .io import sha256_text
    if span.source_path and hit.source_path != span.source_path:
        return False
    if hit.start_line > span.start_line or hit.end_line < span.end_line:
        return False
    lines = hit.text.splitlines()
    relative_start = span.start_line - hit.start_line
    relative_end = span.end_line - hit.start_line + 1
    if relative_start < 0 or relative_end > len(lines):
        return False
    visible_text = "\n".join(lines[relative_start:relative_end])
    return sha256_text(visible_text) == span.text_hash


def score_action(action: ToolAction, valid: bool, hits: tuple[ToolHit, ...], observation: str,
                 groups: tuple[EvidenceGroup, ...], config: RewardConfig = RewardConfig(),
                 previsible_group_ids: tuple[str, ...] = ()) -> ActionOutcome:
    known_group_ids = {group.group_id for group in groups}
    if not set(previsible_group_ids) <= known_group_ids:
        raise ValueError("previsible group must be one of the record's required groups")
    group_ids = tuple(sorted(set(visible_groups(hits, groups)) | set(previsible_group_ids)))
    visible_chunk_ids = set()
    for group in groups:
        for alternative in group.alternatives:
            for hit in hits:
                if _span_visible(hit, alternative):
                    visible_chunk_ids.add(hit.chunk_id)
    required = len(groups)
    coverage = len(group_ids) / required if required else 0.0
    matching_ranks = [h.rank for h in hits if h.chunk_id in visible_chunk_ids]
    first_rank = (1.0 / min(matching_ranks)) if matching_ranks else 0.0
    exact_hits = len(visible_chunk_ids)
    returned_bytes = len(observation.encode("utf-8"))
    components = {
        "valid": config.valid_weight if valid else -1.0,
        "exact_visible_hit": config.hit_weight * exact_hits,
        "required_group_coverage": config.group_coverage_weight * coverage,
        "first_valid_rank": config.first_rank_weight * first_rank,
        "broad_results": -config.broad_result_penalty * max(0, len(hits) - required),
        "returned_bytes": -config.returned_kib_penalty * returned_bytes / 1024.0,
    }
    reward = sum(components.values())
    return ActionOutcome(action, valid, hits, group_ids, reward, components, observation, returned_bytes)
