from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .objectives import query_pairwise_loss, ranking_loss, select_preference
from .replay import SourceReplay


def train(prefix_lm, records, units_by_id, output_path, *, corpus_hash: str, model_id: str,
          learning_rate: float = 1e-3, query_weight: float = 1.0,
          lambda_rank: float = 1.0, delta: float = 0.25,
          replay_fraction: float = 0.5, seed: int = 0, batch_size: int = 2,
          updates_per_source: int = 2, relevant_label: str = "A", irrelevant_label: str = "B"):
    import torch
    prefix_lm.assert_frozen()
    rel_id, irr_id = prefix_lm.validate_labels(relevant_label, irrelevant_label)
    optimizer = torch.optim.AdamW(prefix_lm.trainable_parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(prefix_lm.prefix.device)
    by_source = {}
    for record in records:
        by_source.setdefault(record.source_id, []).append(record)
    replay = SourceReplay(seed, replay_fraction)
    report = {"shards": [], "skipped_preferences": 0, "losses": [], "started": time.time()}
    for source in sorted(by_source):
        replay.add_source(source, by_source[source])
        current_losses, previous_losses = [], []
        # Use the 50%-replay schedule for both interventions. replay=0 fills
        # those matched slots with current records instead of reducing updates.
        current_slots = batch_size if len(replay.bank) == 1 else batch_size - batch_size // 2
        shard_steps = max(updates_per_source, math.ceil(len(by_source[source]) / current_slots))
        for step in range(shard_steps):
            batch = replay.batch(source, batch_size, step)
            optimizer.zero_grad(set_to_none=True)
            pairs = [select_preference(record.outcomes, delta) if query_weight > 0 else None
                     for record in batch.records]
            query_den = sum(pair is not None for pair in pairs)
            rank_den = sum(bool(record.verified_negative_chunk_ids) and lambda_rank > 0
                           for record in batch.records)
            query_values, rank_values = [], []
            for index, record in enumerate(batch.records):
                pair = pairs[index]
                if pair is not None and query_weight > 0:
                    positive, negative = pair
                    context = prefix_lm.chat_ids([{"role": "user", "content": record.prompt}],
                                                 add_generation_prompt=True)
                    pos_ids = prefix_lm.tokenizer(json.dumps({"tool": "search", "query": positive.action.query}),
                                                  add_special_tokens=False, return_tensors="pt").input_ids.to(prefix_lm.prefix.device)
                    neg_ids = prefix_lm.tokenizer(json.dumps({"tool": "search", "query": negative.action.query}),
                                                  add_special_tokens=False, return_tensors="pt").input_ids.to(prefix_lm.prefix.device)
                    query_loss = query_weight * query_pairwise_loss(
                        prefix_lm.mean_action_logprob(context, pos_ids),
                        prefix_lm.mean_action_logprob(context, neg_ids),
                        positive.reward - negative.reward)
                    (query_loss / query_den).backward()
                    query_values.append(float(query_loss.detach()))
                    (current_losses if record.source_id == source else previous_losses).append(
                        float(query_loss.detach()))
                elif query_weight > 0:
                    report["skipped_preferences"] += 1
                if record.verified_negative_chunk_ids and lambda_rank > 0:
                    negative_units = [units_by_id[c] for c in record.verified_negative_chunk_ids]
                    group_scores = []
                    for group in record.evidence_groups:
                        positives = []
                        for alternative in group.alternatives:
                            unit = units_by_id[alternative.chunk_id]
                            prompt = _rank_prompt(record.prompt, unit.text, relevant_label, irrelevant_label)
                            positives.append(prefix_lm.label_score(prefix_lm.chat_ids(
                                [{"role": "user", "content": prompt}], add_generation_prompt=True), rel_id, irr_id))
                        negatives = []
                        for unit in negative_units:
                            prompt = _rank_prompt(record.prompt, unit.text, relevant_label, irrelevant_label)
                            negatives.append(prefix_lm.label_score(prefix_lm.chat_ids(
                                [{"role": "user", "content": prompt}], add_generation_prompt=True), rel_id, irr_id))
                        group_scores.append((positives, negatives))
                    rank_value = lambda_rank * ranking_loss(group_scores)
                    (rank_value / rank_den).backward()
                    rank_values.append(float(rank_value.detach()))
                    (current_losses if record.source_id == source else previous_losses).append(
                        float(rank_value.detach()))
            if not query_values and not rank_values:
                continue
            if prefix_lm.prefix.grad is None or not torch.count_nonzero(prefix_lm.prefix.grad):
                raise RuntimeError("soft prefix did not receive a nonzero gradient")
            if any(p.grad is not None for p in prefix_lm.model.parameters()):
                raise RuntimeError("a frozen LM parameter received a gradient buffer")
            optimizer.step()
            report["losses"].append((_mean(query_values) or 0.0) +
                                    (_mean(rank_values) or 0.0))
        report["shards"].append({"source": source, "current_source_loss": _mean(current_losses),
                                 "mixed_replay_loss": _mean(previous_losses),
                                 "fixed_corpus_probe": "not used for checkpoint selection"})
    prefix_lm.save(output_path, corpus_hash=corpus_hash, model_id=model_id)
    report.update({"elapsed_seconds": time.time() - report["started"],
                   "exposure_by_record": replay.ledger.by_record,
                   "exposure_by_source": replay.ledger.by_source,
                   "replay_fraction": replay_fraction,
                   "serialized_bytes": Path(output_path).stat().st_size,
                   "training_peak_memory_bytes": (torch.cuda.max_memory_allocated(prefix_lm.prefix.device)
                                                   if torch.cuda.is_available() else None)})
    return report


def _rank_prompt(question: str, chunk: str, a: str, b: str) -> str:
    return (f"Question/state:\n{question}\n\nCandidate evidence:\n{chunk}\n\n"
            f"Output exactly {a} if relevant or {b} if irrelevant.")


def _mean(values):
    return sum(values) / len(values) if values else None
