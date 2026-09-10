from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .objectives import query_pairwise_loss, ranking_loss, select_preference
from .replay import SourceReplay
from .agent import ranking_state_messages, study_state_messages
from .rewards import score_action
from .schema import EvidenceGroup, EvidenceSpan, StudyRecord, ToolAction


def train(prefix_lm, records, units_by_id, output_path, *, corpus_hash: str, model_id: str,
          learning_rate: float = 1e-3, query_weight: float = 1.0,
          lambda_rank: float = 1.0, delta: float = 0.25,
          beta: float = 1.0, rank_margin: float = 1.0,
          replay_fraction: float = 0.5, seed: int = 0, batch_size: int = 2,
          updates_per_source: int = 2, relevant_label: str = "A", irrelevant_label: str = "B",
          probes: list[dict] | None = None, tools=None, gradient_accumulation_steps: int = 1):
    import torch
    prefix_lm.assert_frozen()
    if gradient_accumulation_steps != 1:
        raise ValueError("MVP currently supports gradient_accumulation_steps=1 only")
    rel_id, irr_id = prefix_lm.validate_labels(relevant_label, irrelevant_label)
    optimizer = torch.optim.AdamW(prefix_lm.trainable_parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(prefix_lm.prefix.device)
    by_source = {}
    for record in records:
        by_source.setdefault(record.source_id, []).append(record)
    replay = SourceReplay(seed, replay_fraction)
    report = {"shards": [], "skipped_preferences": 0, "losses": [], "started": time.time()}
    diagnostic_record = records[0]
    report["objective_diagnostics_before"] = _record_objectives(
        prefix_lm, diagnostic_record, units_by_id, rel_id, irr_id, relevant_label, irrelevant_label,
        query_weight=query_weight, rank_weight=lambda_rank, delta=delta, beta=beta, margin=rank_margin)
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
                    context = prefix_lm.chat_ids(study_state_messages(record), add_generation_prompt=True)
                    pos_ids = prefix_lm.tokenizer(json.dumps(positive.action.to_payload(), sort_keys=True),
                                                  add_special_tokens=False, return_tensors="pt").input_ids.to(prefix_lm.prefix.device)
                    neg_ids = prefix_lm.tokenizer(json.dumps(negative.action.to_payload(), sort_keys=True),
                                                  add_special_tokens=False, return_tensors="pt").input_ids.to(prefix_lm.prefix.device)
                    query_loss = query_weight * query_pairwise_loss(
                        prefix_lm.mean_action_logprob(context, pos_ids),
                        prefix_lm.mean_action_logprob(context, neg_ids),
                        positive.reward - negative.reward, beta=beta)
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
                            positives.append(prefix_lm.label_score(prefix_lm.chat_ids(
                                ranking_state_messages(record, alternative.text or unit.text,
                                                       relevant_label, irrelevant_label),
                                add_generation_prompt=True), rel_id, irr_id))
                        negatives = []
                        for unit in negative_units:
                            negatives.append(prefix_lm.label_score(prefix_lm.chat_ids(
                                ranking_state_messages(record, unit.text, relevant_label, irrelevant_label),
                                add_generation_prompt=True), rel_id, irr_id))
                        group_scores.append((positives, negatives))
                    rank_value = lambda_rank * ranking_loss(group_scores, margin=rank_margin)
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
        studied = sorted(replay.bank)
        probe_metrics = (evaluate_probes(prefix_lm, probes, units_by_id, tools, rel_id, irr_id,
                                         relevant_label, irrelevant_label, current_source=source,
                                         studied_sources=studied) if probes and tools else {})
        report["shards"].append({"source": source, "current_source_loss": _mean(current_losses),
                                 "previous_source_loss": _mean(previous_losses),
                                 "fixed_corpus_probes": probe_metrics,
                                 "probe_use": "measurement only; never checkpoint selection"})
    prefix_lm.save(output_path, corpus_hash=corpus_hash, model_id=model_id)
    report["objective_diagnostics_after"] = _record_objectives(
        prefix_lm, diagnostic_record, units_by_id, rel_id, irr_id, relevant_label, irrelevant_label,
        query_weight=query_weight, rank_weight=lambda_rank, delta=delta, beta=beta, margin=rank_margin)
    report.update({"elapsed_seconds": time.time() - report["started"],
                   "exposure_by_record": replay.ledger.by_record,
                   "exposure_by_source": replay.ledger.by_source,
                   "source_exposure_imbalance": replay.source_imbalance(),
                   "replay_exposure_by_source": replay.previous_ledger.by_source,
                   "replay_source_imbalance": replay.source_imbalance(kind="previous"),
                   "replay_fraction": replay_fraction,
                   "batch_size": batch_size, "gradient_accumulation_steps": gradient_accumulation_steps,
                   "serialized_bytes": Path(output_path).stat().st_size,
                   "training_peak_memory_bytes": (torch.cuda.max_memory_allocated(prefix_lm.prefix.device)
                                                   if torch.cuda.is_available() else None)})
    return report


def _mean(values):
    return sum(values) / len(values) if values else None


def _record_objectives(prefix_lm, record, units_by_id, rel_id, irr_id, relevant_label, irrelevant_label,
                       *, query_weight, rank_weight, delta, beta, margin):
    import torch
    query_value = rank_value = 0.0
    with torch.no_grad():
        pair = select_preference(record.outcomes, delta) if query_weight else None
        if pair:
            positive, negative = pair; context = prefix_lm.chat_ids(study_state_messages(record))
            action_ids = [prefix_lm.tokenizer(json.dumps(item.action.to_payload(), sort_keys=True),
                                              add_special_tokens=False, return_tensors="pt").input_ids.to(prefix_lm.prefix.device)
                          for item in (positive, negative)]
            query_value = float((query_weight * query_pairwise_loss(
                prefix_lm.mean_action_logprob(context, action_ids[0]),
                prefix_lm.mean_action_logprob(context, action_ids[1]),
                positive.reward-negative.reward, beta=beta)).detach())
        if rank_weight and record.verified_negative_chunk_ids:
            group_scores = []
            for group in record.evidence_groups:
                positives = [prefix_lm.label_score(prefix_lm.chat_ids(ranking_state_messages(
                    record, alternative.text or units_by_id[alternative.chunk_id].text,
                    relevant_label, irrelevant_label)), rel_id, irr_id) for alternative in group.alternatives]
                negatives = [prefix_lm.label_score(prefix_lm.chat_ids(ranking_state_messages(
                    record, units_by_id[c].text, relevant_label, irrelevant_label)), rel_id, irr_id)
                             for c in record.verified_negative_chunk_ids]
                group_scores.append((positives, negatives))
            rank_value = float((rank_weight * ranking_loss(group_scores, margin=margin)).detach())
    return {"record_id": record.record_id, "query": query_value, "rank": rank_value,
            "combined": query_value + rank_value}


def evaluate_probes(prefix_lm, probes, units_by_id, tools, rel_id, irr_id, relevant_label,
                    irrelevant_label, *, current_source: str, studied_sources: list[str]):
    import torch
    early = set(studied_sources[:max(1, len(studied_sources) // 2)]) - {current_source}
    recent = set(studied_sources[-2:-1])
    buckets = {"current_source": {current_source}, "previous_early_sources": early,
               "previous_recent_sources": recent, "all_probe_sources": set(studied_sources)}
    raw = []
    with torch.no_grad():
        for probe in probes:
            if probe["source_id"] not in set(studied_sources): continue
            span = EvidenceSpan(**probe["evidence_span"])
            positive, negative = units_by_id[probe["gold_chunk_id"]], units_by_id[probe["verified_negative_chunk_id"]]
            record = StudyRecord(probe["probe_id"], "comprehension", probe["source_id"], probe["prompt"], "",
                                 (EvidenceGroup("probe_fact", (span,)),), (positive.unit_id,), (negative.unit_id,),
                                 (), (), {"method": "atomic_structural_definition"}, "probe", positive.source_hash,
                                 probe["template_id"], 0)
            pos = prefix_lm.label_score(prefix_lm.chat_ids(
                ranking_state_messages(record, span.text, relevant_label, irrelevant_label)), rel_id, irr_id)
            neg = prefix_lm.label_score(prefix_lm.chat_ids(
                ranking_state_messages(record, negative.text, relevant_label, irrelevant_label)), rel_id, irr_id)
            actions = [ToolAction(tool="read_file", path=span.source_path, start_line=span.start_line,
                                  end_line=span.end_line),
                       ToolAction(tool="grep", query=negative.name.split(".")[-1], path=negative.source_path)]
            context = prefix_lm.chat_ids(study_state_messages(record))
            action_scores = []
            for action in actions:
                ids = prefix_lm.tokenizer(json.dumps(action.to_payload(), sort_keys=True), add_special_tokens=False,
                                          return_tensors="pt").input_ids.to(prefix_lm.prefix.device)
                action_scores.append(prefix_lm.mean_action_logprob(context, ids))
            chosen = actions[int(action_scores[1] > action_scores[0])]
            outcome = score_action(chosen, *tools.execute(chosen), record.evidence_groups)
            exact = float(outcome.reward_components["exact_visible_hit"] > 0)
            coverage = len(outcome.visible_group_ids) / len(record.evidence_groups)
            raw.append({"source": probe["source_id"], "margin": float((pos-neg).detach()),
                        "rank_correct": float(pos > neg), "query_exact_hit": exact,
                        "query_group_coverage": coverage, "valid_action": 1.0,
                        "query_probe_mode": "argmax_over_fixed_legal_action_pair"})
    result = {}
    for name, sources in buckets.items():
        selected = [item for item in raw if item["source"] in sources]
        result[name] = {"records": len(selected), **{key: (_mean([x[key] for x in selected]) if selected else None)
                                                       for key in ("margin", "rank_correct", "query_exact_hit",
                                                                   "query_group_coverage", "valid_action")}}
    return result
