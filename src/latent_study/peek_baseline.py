from __future__ import annotations

import json
import math
from pathlib import Path
import time

from .io import write_json
from .replay import SourceReplay
from .schema import StudyRecord


class QwenPeekClient:
    """Production PEEK LMClient backed by the same frozen Qwen deployment model."""

    def __init__(self, model, tokenizer, *, max_new_tokens: int = 384, retries: int = 1):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.retries = max_new_tokens, retries
        self._usage = None
        self.stats = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "latency_seconds": 0.0, "malformed_outputs": 0,
                      "retry_calls": 0, "failed_outputs": 0, "malformed_examples": []}

    def _once(self, messages):
        import torch
        from peek.core.types import Usage
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        try:
            ids = self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(ids, "keys"): ids = ids["input_ids"]
        ids = ids.to(self.model.device); started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        elapsed = time.perf_counter() - started
        generated = output[0, ids.shape[1]:]
        self._usage = Usage(int(ids.numel()), int(generated.numel()))
        self.stats["model_calls"] += 1; self.stats["input_tokens"] += int(ids.numel())
        self.stats["output_tokens"] += int(generated.numel()); self.stats["latency_seconds"] += elapsed
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def completion(self, messages):
        from peek._io import extract_json
        current = list(messages)
        for attempt in range(self.retries + 1):
            raw = self._once(current)
            if isinstance(extract_json(raw), dict): return raw
            self.stats["malformed_outputs"] += 1
            if len(self.stats["malformed_examples"]) < 3:
                self.stats["malformed_examples"].append(raw[-1000:])
            if attempt < self.retries:
                self.stats["retry_calls"] += 1
                current = current + [{"role": "assistant", "content": raw},
                                     {"role": "user", "content": "Return one valid JSON object matching the requested schema. JSON only."}]
        self.stats["failed_outputs"] += 1
        return raw

    def last_usage(self):
        if self._usage is None:
            from peek.core.types import Usage
            return Usage()
        return self._usage


def trajectory_from_record(record: StudyRecord) -> str:
    """Expose the same verified study information as latent training, in readable form."""
    groups = []
    for group in record.evidence_groups:
        groups.append({"group": group.group_id, "necessary_fact": True,
                       "alternative_valid_spans": [{"path": span.source_path, "lines": [span.start_line, span.end_line],
                                                     "text": span.text} for span in group.alternatives]})
    outcomes = []
    for outcome in record.outcomes:
        outcomes.append({"action": outcome.action.to_payload(), "valid": outcome.valid,
                         "visible_observation": outcome.observation, "visible_required_groups": outcome.visible_group_ids,
                         "deterministic_reward": outcome.reward, "reward_components": outcome.reward_components,
                         "derived_feedback": ("useful" if outcome.reward > 0 and outcome.visible_group_ids else
                                              "invalid" if not outcome.valid else "insufficient evidence")})
    payload = {"study_prompt": record.prompt,
               "existing_tool_state": ({"action": record.observation_action.to_payload() if record.observation_action else None,
                                         "visible_observation": record.observation} if record.observation else None),
               "required_evidence_groups": groups, "executed_candidate_actions": outcomes,
               "validation": {"method": record.validation.get("method"),
                              "resolution_rule": record.validation.get("resolution_rule")}}
    return ("Offline corpus-only study trajectory. Distill human-readable corpus knowledge, relations, and "
            "navigation hints. Internal record/chunk identifiers are deliberately omitted.\n" +
            json.dumps(payload, ensure_ascii=False, sort_keys=True))


def trajectory_from_records(records) -> str:
    return "\n\n===== NEXT STUDY RECORD =====\n\n".join(trajectory_from_record(record) for record in records)


def study_offline_peek(records: list[StudyRecord], output: str | Path, *, token_budget: int,
                       client, replay_fraction: float = 0.5, seed: int = 0, batch_size: int = 4,
                       updates_per_source: int = 2, token_counter=None,
                       counter_name: str = "unspecified", model_provenance: dict | None = None) -> dict:
    try:
        from peek import CachePolicy, ContextMap
    except ImportError as exc:
        raise RuntimeError("install the pinned PEEK dependency with `pip install -e '.[peek]'`") from exc
    if not records: raise ValueError("PEEK study bank is empty")
    from .isolation import validate_study_bank
    validate_study_bank(records, records[0].corpus_hash)
    if isinstance(client, type) or client.__class__.__name__.startswith("Fake"):
        raise ValueError("production PEEK study refuses fake clients")
    if token_counter is None: raise ValueError("PEEK requires the actual deployment tokenizer")
    policy = CachePolicy(client=client, token_budget=token_budget, evolve_steps=None,
                         token_counter=token_counter, cmap=ContextMap("## CONTEXT ROADMAP\n"))
    by_source: dict[str, list[StudyRecord]] = {}
    for record in records: by_source.setdefault(record.source_id, []).append(record)
    replay, update_count, usage_in, usage_out = SourceReplay(seed, replay_fraction), 0, 0, 0
    started = time.perf_counter()
    for source in sorted(by_source):
        replay.add_source(source, by_source[source])
        current_slots = batch_size if len(replay.bank) == 1 else batch_size - batch_size // 2
        shard_steps = max(updates_per_source, math.ceil(len(by_source[source]) / current_slots))
        for step in range(shard_steps):
            batch = replay.batch(source, batch_size, step)
            result = policy.update(trajectory=trajectory_from_records(batch.records),
                                   question="\n".join(record.prompt for record in batch.records))
            if result is not None:
                usage_in += result.usage.input_tokens; usage_out += result.usage.output_tokens
            update_count += 1
    output = Path(output); policy.save(output); payload = json.loads(output.read_text(encoding="utf-8"))
    map_text = policy.current_map_text
    payload.update({"phase": "frozen_before_evaluation", "evaluation_inputs_seen": False,
                    "protocol": "offline_peek_map", "upstream_policy": True,
                    "upstream_revision": "8b109771b51126284ea337f23827facde1db05ed",
                    "update_count": update_count, "replay_fraction": replay_fraction,
                    "token_counter": counter_name, "map_text_tokens": token_counter(map_text),
                    "map_text_bytes": len(map_text.encode("utf-8")),
                    "peek_reported_input_tokens_excluding_retries": usage_in,
                    "peek_reported_output_tokens_excluding_retries": usage_out,
                    "study_elapsed_seconds": time.perf_counter() - started,
                    "client_statistics": getattr(client, "stats", {}), "model": model_provenance or {},
                    "exposure_by_record": replay.ledger.by_record,
                    "exposure_by_source": replay.ledger.by_source,
                    "source_exposure_imbalance": replay.source_imbalance()})
    payload["replay_exposure_by_source"] = replay.previous_ledger.by_source
    payload["replay_source_imbalance"] = replay.source_imbalance(kind="previous")
    calls = max(1, payload["client_statistics"].get("model_calls", 0))
    payload["study_input_tokens"] = payload["client_statistics"].get("input_tokens", usage_in)
    payload["study_output_tokens"] = payload["client_statistics"].get("output_tokens", usage_out)
    try:
        import torch
        payload["study_peak_memory_bytes"] = (torch.cuda.max_memory_allocated(client.model.device)
                                               if torch.cuda.is_available() else None)
    except (AttributeError, RuntimeError):
        payload["study_peak_memory_bytes"] = None
    payload["client_statistics"]["malformed_output_rate"] = payload["client_statistics"].get("malformed_outputs", 0) / calls
    payload["client_statistics"]["retry_rate"] = payload["client_statistics"].get("retry_calls", 0) / calls
    payload["client_statistics"]["failure_rate_per_update"] = payload["client_statistics"].get("failed_outputs", 0) / max(1, update_count * 2)
    payload["complete_artifact_bytes"] = 0
    for _ in range(4):
        write_json(output, payload); payload["complete_artifact_bytes"] = output.stat().st_size
    write_json(output, payload)
    return payload
