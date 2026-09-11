from __future__ import annotations

import json
from pathlib import Path
import time
import re

from .io import write_json
from .replay import SourceReplay, matched_shard_steps
from .schema import StudyRecord


_MAP_SECTIONS = {
    "context_roadmap", "context_understanding", "domain_constants",
    "parsing_schema", "error_patterns", "reusable_results",
}


class QwenPeekClient:
    """Production PEEK LMClient backed by the same frozen Qwen deployment model."""

    def __init__(self, model, tokenizer, *, max_new_tokens: int = 1024,
                 internal_max_new_tokens: int | None = None,
                 retry_max_new_tokens: int | None = None, retries: int = 1):
        self.model, self.tokenizer = model, tokenizer
        # This is the internal structured Distiller/Cartographer generation
        # allowance.  The final PEEK map budget is enforced separately by the
        # pinned CachePolicy token counter.
        self.max_new_tokens = int(internal_max_new_tokens if internal_max_new_tokens is not None
                                  else max_new_tokens)
        self.retry_max_new_tokens = int(
            retry_max_new_tokens if retry_max_new_tokens is not None
            else max(self.max_new_tokens * 2, 1024))
        if self.max_new_tokens < 1 or self.retry_max_new_tokens < self.max_new_tokens:
            raise ValueError("PEEK internal generation budgets are invalid")
        self.retries = int(retries)
        if self.retries < 0:
            raise ValueError("PEEK retries must be non-negative")
        self._usage = None
        self._attempt = 0
        self.stats = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "latency_seconds": 0.0, "malformed_outputs": 0,
                      "retry_calls": 0, "failed_outputs": 0, "malformed_examples": [],
                      "internal_generation_budgets": [],
                      "distiller": {"calls": 0, "malformed": 0, "retries": 0, "failures": 0},
                      "cartographer": {"calls": 0, "malformed": 0, "retries": 0, "failures": 0}}

    @staticmethod
    def _stage(messages) -> str:
        prompt = messages[-1].get("content", "") if messages else ""
        return "cartographer" if "Available Operations" in prompt else "distiller"

    @staticmethod
    def _schema(stage: str) -> str:
        if stage == "distiller":
            return ('{"diagnosis":"string","item_tags":{"<item_id>":'
                    '"helpful|harmful|neutral|stale"},"cache_candidates":'
                    '[{"section":"context_roadmap|context_understanding|domain_constants|parsing_schema|error_patterns|reusable_results",'
                    '"value":"string","transferability":"string","rationale":"string"}]}')
        return ('{"reasoning":"string","operations":[{"type":"ADD","section":'
                '"context_roadmap|context_understanding|domain_constants|parsing_schema|error_patterns|reusable_results",'
                '"content":"string"}|{"type":"DELETE","item_id":"string"}|'
                '{"type":"REPLACE","item_id":"string","content":"string"}]}')

    @staticmethod
    def _valid(stage: str, value) -> tuple[bool, str]:
        if not isinstance(value, dict): return False, "top level is not an object"
        if stage == "distiller":
            if set(value) != {"diagnosis", "item_tags", "cache_candidates"}: return False, "fields must match exactly"
            if not isinstance(value["diagnosis"], str): return False, "diagnosis must be a string"
            if not isinstance(value["item_tags"], dict): return False, "item_tags must be an object"
            if any(v not in {"helpful", "harmful", "neutral", "stale"}
                   for v in value["item_tags"].values()): return False, "invalid item tag"
            if not isinstance(value["cache_candidates"], list): return False, "cache_candidates must be a list"
            for candidate in value["cache_candidates"]:
                required = {"section", "value", "transferability", "rationale"}
                if (not isinstance(candidate, dict) or set(candidate) != required
                        or candidate["section"] not in _MAP_SECTIONS
                        or any(not isinstance(candidate[key], str) or not candidate[key].strip()
                               for key in required)):
                    return False, "each cache candidate must match the required fields and section enum"
            return True, ""
        if set(value) != {"reasoning", "operations"}: return False, "fields must match exactly"
        if not isinstance(value["reasoning"], str) or not isinstance(value["operations"], list):
            return False, "reasoning must be a string and operations a list"
        for op in value["operations"]:
            if not isinstance(op, dict) or op.get("type") not in {"ADD", "DELETE", "REPLACE"}:
                return False, "each operation needs a valid type"
            required = ({"type", "section", "content"} if op["type"] == "ADD" else
                        {"type", "item_id"} if op["type"] == "DELETE" else
                        {"type", "item_id", "content"})
            if set(op) != required or any(not isinstance(op[k], str) or not op[k].strip()
                                          for k in required - {"type"}):
                return False, f"invalid {op['type']} fields"
            if op["type"] == "ADD" and op["section"] not in _MAP_SECTIONS:
                return False, "ADD section is outside the pinned map schema"
        return True, ""

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
            allowance = self.max_new_tokens if self._attempt == 0 else self.retry_max_new_tokens
            self.stats["internal_generation_budgets"].append(allowance)
            output = self.model.generate(ids, max_new_tokens=allowance, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        elapsed = time.perf_counter() - started
        generated = output[0, ids.shape[1]:]
        self._usage = Usage(int(ids.numel()), int(generated.numel()))
        self.stats["model_calls"] += 1; self.stats["input_tokens"] += int(ids.numel())
        self.stats["output_tokens"] += int(generated.numel()); self.stats["latency_seconds"] += elapsed
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def completion(self, messages):
        from peek._io import extract_json
        stage = self._stage(messages)
        self.stats[stage]["calls"] += 1
        current = list(messages)
        for attempt in range(self.retries + 1):
            self._attempt = attempt
            raw = self._once(current)
            valid, error = self._valid(stage, extract_json(raw))
            if valid: return raw
            self.stats["malformed_outputs"] += 1
            self.stats[stage]["malformed"] += 1
            if len(self.stats["malformed_examples"]) < 3:
                self.stats["malformed_examples"].append(raw[-1000:])
            if attempt < self.retries:
                self.stats["retry_calls"] += 1
                self.stats[stage]["retries"] += 1
                current = current + [{"role": "assistant", "content": raw},
                                     {"role": "user", "content":
                                      f"Invalid {stage} JSON: {error}. Return JSON only, exactly matching this schema: {self._schema(stage)}"}]
        self.stats["failed_outputs"] += 1
        self.stats[stage]["failures"] += 1
        raise RuntimeError(f"PEEK {stage} failed structured output after {self.retries + 1} attempts")

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
                       counter_name: str = "unspecified", model_provenance: dict | None = None,
                       artifact_provenance: dict | None = None) -> dict:
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
    output = Path(output)
    started = time.perf_counter()
    try:
        for source in sorted(by_source):
            replay.add_source(source, by_source[source])
            shard_steps = matched_shard_steps(
                len(by_source[source]), batch_size, updates_per_source,
                has_previous_sources=bool(replay.bank))
            for step in range(shard_steps):
                batch = replay.batch(source, batch_size, step)
                result = policy.update(trajectory=trajectory_from_records(batch.records),
                                       question="\n".join(record.prompt for record in batch.records))
                if result is not None:
                    usage_in += result.usage.input_tokens; usage_out += result.usage.output_tokens
                update_count += 1
        map_items = policy.cmap.items()
        map_tokens = token_counter(policy.current_map_text)
        meaningful = [item for item in map_items
                      if len(re.findall(r"[A-Za-z]{2,}", item.content)) >= 4 and len(item.content) >= 20]
        navigation = [item for item in meaningful if re.search(
            r"(?:\.py\b|\bfile\b|\bmodule\b|\bclass\b|\bfunction\b|\bsection\b|\blocated\b|\bunder\b)",
            item.content, re.IGNORECASE)]
        if map_tokens > token_budget:
            raise RuntimeError(f"PEEK context map exceeds token budget ({map_tokens}>{token_budget})")
        if not map_items or len(meaningful) != len(map_items) or not navigation:
            raise RuntimeError("PEEK produced an empty/header-only or non-navigable context map")
        if getattr(client, "stats", {}).get("failed_outputs", 0):
            raise RuntimeError("PEEK update had failed structured outputs")
    except Exception as exc:
        try:
            import torch
            peak_memory = (torch.cuda.max_memory_allocated(client.model.device)
                           if torch.cuda.is_available() else None)
        except (AttributeError, RuntimeError):
            peak_memory = None
        failed_stats = getattr(client, "stats", {})
        calls = max(1, failed_stats.get("model_calls", 0))
        failed_stats["malformed_output_rate"] = failed_stats.get("malformed_outputs", 0) / calls
        failed_stats["retry_rate"] = failed_stats.get("retry_calls", 0) / calls
        failed_stats["failure_rate_per_update"] = failed_stats.get("failed_outputs", 0) / max(1, update_count * 2)
        failed = {"artifact_schema_version": 2, "status": "failed", "phase": "study_failed",
                  "evaluation_inputs_seen": False, "protocol": "offline_peek_map",
                  "provenance": artifact_provenance or {}, "token_budget": token_budget,
                  "token_counter": counter_name, "completed_updates": update_count,
                  "failure": f"{type(exc).__name__}: {exc}",
                  "client_statistics": failed_stats,
                  "study_elapsed_seconds": time.perf_counter() - started,
                  "study_total_latency_seconds": failed_stats.get("latency_seconds"),
                  "study_prefill_latency_seconds": None,
                  "study_decode_latency_seconds": None,
                  "study_metric_unavailable": {
                      "prefill_latency_seconds": "PEEK backend reports one generate call latency, not a separate prefill phase",
                      "decode_latency_seconds": "PEEK backend reports one generate call latency, not a separate decode phase",
                  },
                  "study_input_tokens": failed_stats.get("input_tokens", usage_in),
                  "study_output_tokens": failed_stats.get("output_tokens", usage_out),
                  "study_peak_memory_bytes": peak_memory,
                  "map_text_tokens": token_counter(policy.current_map_text),
                  "map_text_bytes": len(policy.current_map_text.encode("utf-8")),
                  "exposure_by_source": replay.ledger.by_source,
                  "replay_audit": replay.replay_audit()}
        failed["complete_artifact_bytes"] = 0
        for _ in range(4):
            write_json(output, failed)
            failed["complete_artifact_bytes"] = output.stat().st_size
        write_json(output, failed)
        raise RuntimeError(f"PEEK study failed; diagnostics written to {output}") from exc
    policy.save(output); payload = json.loads(output.read_text(encoding="utf-8"))
    map_text = policy.current_map_text
    contamination = (artifact_provenance or {}).get("contamination_audit", {})
    attested = (isinstance(contamination, dict) and contamination.get("status") == "pass"
                and isinstance(contamination.get("artifact_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", contamination["artifact_sha256"]))
    payload.update({"phase": "frozen_before_evaluation" if attested else "study_complete_unattested",
                    "evaluation_inputs_seen": False,
                    "artifact_schema_version": 2, "provenance": artifact_provenance or {},
                    "protocol": "offline_peek_map", "upstream_policy": True,
                    "upstream_revision": "8b109771b51126284ea337f23827facde1db05ed",
                    "update_count": update_count, "replay_fraction": replay_fraction,
                    "token_counter": counter_name, "map_text_tokens": token_counter(map_text),
                    "map_text_bytes": len(map_text.encode("utf-8")),
                    "peek_reported_input_tokens_excluding_retries": usage_in,
                    "peek_reported_output_tokens_excluding_retries": usage_out,
                    "study_elapsed_seconds": time.perf_counter() - started,
                    "study_total_latency_seconds": getattr(client, "stats", {}).get("latency_seconds"),
                    "study_prefill_latency_seconds": None,
                    "study_decode_latency_seconds": None,
                    "study_metric_unavailable": {
                        "prefill_latency_seconds": "PEEK backend reports one generate call latency, not a separate prefill phase",
                        "decode_latency_seconds": "PEEK backend reports one generate call latency, not a separate decode phase",
                    },
                    "client_statistics": getattr(client, "stats", {}), "model": model_provenance or {},
                    "exposure_by_record": replay.ledger.by_record,
                    "exposure_by_source": replay.ledger.by_source,
                    "source_exposure_imbalance": replay.source_imbalance()})
    payload["replay_exposure_by_source"] = replay.previous_ledger.by_source
    payload["replay_source_imbalance"] = replay.source_imbalance(kind="previous")
    payload["replay_audit"] = replay.replay_audit()
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
