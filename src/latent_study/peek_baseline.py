from __future__ import annotations

import json
import math
from pathlib import Path

from .io import write_json
from .replay import SourceReplay
from .schema import StudyRecord


class EvidenceTrajectoryClient:
    """Deterministic LMClient adapter feeding corpus-derived records to PEEK policy code."""

    def __init__(self):
        self.pending_candidates: list[dict] = []

    def completion(self, messages):
        text = messages[-1]["content"]
        marker = "LATENT_STUDY_RECORD="
        if marker in text:
            payload = json.loads(text.split(marker, 1)[1].splitlines()[0])
            self.pending_candidates = payload["cache_candidates"]
            return json.dumps({"diagnosis": "offline corpus-derived trajectory",
                               "item_tags": {}, "cache_candidates": self.pending_candidates})
        operations = [{"type": "ADD", "section": c["section"], "content": c["value"]}
                      for c in self.pending_candidates]
        self.pending_candidates = []
        return json.dumps({"reasoning": "deterministic evidence-grounded offline update",
                           "operations": operations})

    def last_usage(self):
        from peek.core.types import Usage
        return Usage(0, 0)


def trajectory_from_record(record: StudyRecord) -> str:
    candidates = [{"section": "context_roadmap",
                   "value": f"{record.source_id}: evidence units {', '.join(record.positive_chunk_ids)}"}]
    payload = {"record_id": record.record_id, "source_id": record.source_id,
               "prompt": record.prompt, "observation": record.observation,
               "outcomes": [{"query": o.action.query, "reward": o.reward,
                             "visible_groups": o.visible_group_ids,
                             "actual_observation": o.observation} for o in record.outcomes],
               "cache_candidates": candidates}
    return "Offline corpus study trajectory.\nLATENT_STUDY_RECORD=" + json.dumps(payload, separators=(",", ":"))


def study_offline_peek(records: list[StudyRecord], output: str | Path, *, token_budget: int,
                       replay_fraction: float = 0.5, seed: int = 0, batch_size: int = 4,
                       updates_per_source: int = 2, token_counter=None,
                       counter_name: str = "unspecified") -> dict:
    try:
        from peek import CachePolicy, ContextMap
    except ImportError as exc:
        raise RuntimeError("install the pinned PEEK dependency with `pip install -e '.[peek]'`") from exc
    if not records:
        raise ValueError("PEEK study bank is empty")
    from .isolation import validate_study_bank
    validate_study_bank(records, records[0].corpus_hash)
    client = EvidenceTrajectoryClient()
    if token_counter is None:
        raise ValueError("PEEK requires the actual deployment tokenizer; smoke must opt into its approximation")
    # PEEK's stock annotated template itself exceeds 64 tokens. This corpus-only
    # adaptation starts from its valid section syntax with no explanatory filler.
    policy = CachePolicy(client=client, token_budget=token_budget, evolve_steps=None,
                         token_counter=token_counter, cmap=ContextMap("## CONTEXT ROADMAP\n"))
    by_source: dict[str, list[StudyRecord]] = {}
    for record in records:
        by_source.setdefault(record.source_id, []).append(record)
    replay = SourceReplay(seed, replay_fraction)
    update_count = 0
    for source in sorted(by_source):
        replay.add_source(source, by_source[source])
        current_slots = batch_size if len(replay.bank) == 1 else batch_size - batch_size // 2
        shard_steps = max(updates_per_source, math.ceil(len(by_source[source]) / current_slots))
        for step in range(shard_steps):
            batch = replay.batch(source, batch_size, step)
            for record in batch.records:
                policy.update(trajectory=trajectory_from_record(record), question=record.prompt)
                update_count += 1
    output = Path(output)
    policy.save(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload.update({"phase": "frozen_before_evaluation", "evaluation_inputs_seen": False,
                    "protocol": "offline_peek_map", "upstream_policy": True,
                    "update_count": update_count, "replay_fraction": replay_fraction,
                    "token_counter": counter_name,
                    "map_tokens": token_counter(policy.current_map_text),
                    "exposure_by_record": replay.ledger.by_record,
                    "exposure_by_source": replay.ledger.by_source})
    # JSON size depends on the digits of this field; converge to the actual final size.
    payload["serialized_bytes"] = 0
    for _ in range(4):
        write_json(output, payload)
        payload["serialized_bytes"] = output.stat().st_size
    write_json(output, payload)
    return payload
