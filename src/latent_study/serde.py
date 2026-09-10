from __future__ import annotations

from .schema import (ActionOutcome, CorpusUnit, EvidenceGroup, EvidenceSpan,
                     StudyRecord, ToolAction, ToolHit)


def unit_from_dict(value: dict) -> CorpusUnit:
    value = dict(value)
    for key in ("symbols", "definitions", "references"):
        value[key] = tuple(value.get(key, ()))
    value["relations"] = tuple(tuple(x) for x in value.get("relations", ()))
    return CorpusUnit(**value)


def record_from_dict(value: dict) -> StudyRecord:
    groups = tuple(EvidenceGroup(g["group_id"], tuple(EvidenceSpan(**a) for a in g["alternatives"]))
                   for g in value["evidence_groups"])
    actions = tuple(ToolAction(**a) for a in value["candidate_actions"])
    outcomes = []
    for outcome in value["outcomes"]:
        hits = tuple(ToolHit(**hit) for hit in outcome["hits"])
        outcomes.append(ActionOutcome(ToolAction(**outcome["action"]), outcome["valid"], hits,
                                      tuple(outcome["visible_group_ids"]), outcome["reward"],
                                      outcome["reward_components"], outcome["observation"],
                                      outcome["returned_bytes"]))
    return StudyRecord(record_id=value["record_id"], family=value["family"],
                       source_id=value["source_id"], prompt=value["prompt"],
                       observation=value["observation"], evidence_groups=groups,
                       positive_chunk_ids=tuple(value["positive_chunk_ids"]),
                       verified_negative_chunk_ids=tuple(value["verified_negative_chunk_ids"]),
                       candidate_actions=actions, outcomes=tuple(outcomes),
                       validation=value["validation"], corpus_hash=value["corpus_hash"],
                       source_hash=value["source_hash"], template_id=value["template_id"],
                       random_seed=value["random_seed"], weak_label=value.get("weak_label", False),
                       observation_action=(ToolAction(**value["observation_action"])
                                           if value.get("observation_action") else None))
