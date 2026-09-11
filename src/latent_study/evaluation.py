from __future__ import annotations

"""Evaluation-only plumbing.

This module deliberately knows nothing about StudyBench question semantics. It
loads frozen examples, runs the five memory conditions under one explicit
budget, and delegates scoring to an injected judge. Training never receives the
evaluation dataset.
"""

import importlib
import json
import math
import resource
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


MVP_CONDITIONS = (
    "no_study", "random_latent_L64", "trained_latent_L64",
    "offline_peek_64", "offline_peek_1024",
)


@dataclass(frozen=True)
class EvaluationExample:
    example_id: str
    question: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationBudgetProfile:
    name: str
    max_tool_calls: int
    max_output_tokens: int
    max_observation_bytes: int
    exact_tool_calls: int | None = None
    allow_early_return: bool = True


def _json_values(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("examples", "data", "records"):
            if isinstance(value.get(key), list):
                return value[key]
    return value if isinstance(value, list) else [value]


def load_evaluation_dataset(path: str | Path) -> list[EvaluationExample]:
    """Load evaluation-only examples with stable IDs."""
    root = Path(path)
    files = sorted(item for item in (root.rglob("*") if root.is_dir() else [root])
                   if item.is_file() and item.suffix.lower() in {".json", ".jsonl"})
    if not files:
        raise ValueError(f"evaluation dataset has no JSON/JSONL files: {root}")
    examples: list[EvaluationExample] = []
    seen: set[str] = set()
    for file in files:
        for index, row in enumerate(_json_values(file)):
            if not isinstance(row, dict):
                raise ValueError(f"evaluation row is not an object: {file}:{index + 1}")
            example_id = row.get("example_id", row.get("id", row.get("question_id")))
            question = row.get("question", row.get("prompt"))
            if not isinstance(example_id, str) or not example_id.strip():
                raise ValueError(f"evaluation row lacks stable example_id: {file}:{index + 1}")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"evaluation row lacks question text: {file}:{index + 1}")
            example_id = example_id.strip()
            if example_id in seen:
                raise ValueError(f"duplicate evaluation example ID: {example_id}")
            seen.add(example_id)
            examples.append(EvaluationExample(example_id, question,
                                              {str(key): value for key, value in row.items()}))
    if not examples:
        raise ValueError("evaluation dataset is empty")
    return examples


def budget_from_config(config: Mapping[str, Any], name: str | None = None) -> EvaluationBudgetProfile:
    evaluation = config.get("evaluation", {})
    profile_name = name or evaluation.get("default_budget", "max5")
    profiles = evaluation.get("budgets", {})
    if profile_name not in profiles:
        raise ValueError(f"unknown evaluation budget profile: {profile_name}")
    raw = profiles[profile_name]
    required = ("max_tool_calls", "max_output_tokens", "max_observation_bytes")
    if any(key not in raw for key in required):
        raise ValueError(f"evaluation budget {profile_name} is incomplete")
    values = {key: int(raw[key]) for key in required}
    if any(value < 0 for value in values.values()):
        raise ValueError(f"evaluation budget {profile_name} contains a negative limit")
    exact = raw.get("exact_tool_calls")
    if exact is not None:
        exact = int(exact)
        if exact < 0 or exact > values["max_tool_calls"]:
            raise ValueError(f"evaluation budget {profile_name} has invalid exact_tool_calls")
    return EvaluationBudgetProfile(profile_name, **values, exact_tool_calls=exact,
                                    allow_early_return=bool(raw.get("allow_early_return", True)))


def load_judge(spec: str | None) -> Callable | None:
    """Load an evaluation-only ``module:function`` scorer, if supplied."""
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("judge must be specified as module:function")
    module_name, function_name = spec.split(":", 1)
    judge = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(judge):
        raise ValueError(f"judge is not callable: {spec}")
    return judge


def _score(scorer: Callable | None, example: EvaluationExample, result: dict) -> dict:
    if scorer is None:
        return {"strict": None, "lenient": None, "available": False,
                "reason": "no evaluation judge was configured"}
    value = scorer(example, result)
    if isinstance(value, Mapping):
        strict, lenient = value.get("strict"), value.get("lenient")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        strict = lenient = float(value)
    else:
        raise ValueError("evaluation judge must return a number or {strict,lenient} mapping")
    for name, score in (("strict", strict), ("lenient", lenient)):
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                  or not math.isfinite(float(score))):
            raise ValueError(f"judge returned a non-finite {name} score")
    return {"strict": None if strict is None else float(strict),
            "lenient": None if lenient is None else float(lenient),
            "available": strict is not None or lenient is not None}


def _budget_check(result: Mapping[str, Any], budget: EvaluationBudgetProfile) -> dict:
    calls = int(result.get("tool_calls", 0) or 0)
    output = int(result.get("model_output_tokens", result.get("output_tokens", 0)) or 0)
    observations = int(result.get("returned_observation_bytes", result.get("observation_bytes", 0)) or 0)
    violations = []
    if calls > budget.max_tool_calls:
        violations.append("tool_calls_exceeded")
    if output > budget.max_output_tokens:
        violations.append("output_tokens_exceeded")
    if observations > budget.max_observation_bytes:
        violations.append("observation_bytes_exceeded")
    if budget.exact_tool_calls is not None and calls != budget.exact_tool_calls:
        violations.append("exact_tool_calls_not_met")
    if not budget.allow_early_return and budget.exact_tool_calls is not None and calls < budget.exact_tool_calls:
        violations.append("early_return_forbidden")
    return {"valid": not violations, "violations": violations,
            "tool_calls": calls, "model_output_tokens": output,
            "observation_bytes": observations}


def validate_budget_result(result: Mapping[str, Any], budget: EvaluationBudgetProfile) -> dict:
    """Public budget validator used by tests and production callers."""
    return _budget_check(result, budget)


def _compute_metrics(result: Mapping[str, Any]) -> dict:
    fields = {
        "input_tokens": result.get("model_input_tokens", result.get("input_tokens")),
        "generated_tokens": result.get("model_output_tokens", result.get("output_tokens")),
        "tool_calls": result.get("tool_calls"),
        "observation_bytes": result.get("returned_observation_bytes", result.get("observation_bytes")),
        "total_latency_seconds": result.get("inference_latency_seconds", result.get("latency_seconds")),
        "prefill_latency_seconds": result.get("prefill_latency_seconds"),
        "decode_latency_seconds": result.get("decode_latency_seconds"),
        "peak_memory_bytes": result.get("peak_memory_bytes"),
    }
    unavailable = dict(result.get("metric_unavailable", {}))
    for name, value in list(fields.items()):
        if value is None:
            unavailable.setdefault(name, "backend did not expose this measurement")
        elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
            fields[name] = None
            unavailable.setdefault(name, "backend returned a non-finite measurement")
    fields["unavailable"] = unavailable
    return fields


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _expertise(points: list[tuple[float, float]]) -> tuple[float | None, str | None]:
    if not points:
        return None, "no judge score with a measured generated-token budget"
    if any(tokens < 3000 for tokens, _ in points):
        return None, "points do not reach the 3,000-token expertise anchor"
    from .metrics import expertise
    return expertise(points), None


def run_evaluation(examples: list[EvaluationExample], condition_runners: Mapping[str, Callable], *,
                   budget: EvaluationBudgetProfile, scorer: Callable | None = None,
                   conditions: Iterable[str] = MVP_CONDITIONS, output: str | Path | None = None,
                   provenance: dict | None = None, fair_conditions: list[Any] | None = None) -> dict:
    """Run all MVP conditions and emit an expertise-ready structured artifact."""
    selected = tuple(conditions)
    if selected != MVP_CONDITIONS:
        raise ValueError(f"evaluation must compare exactly the five MVP conditions: {MVP_CONDITIONS}")
    if set(condition_runners) != set(selected):
        raise ValueError("condition runners do not cover exactly the five MVP conditions")
    if fair_conditions is not None:
        from .agent import assert_fair_conditions
        assert_fair_conditions(fair_conditions)
    if not examples:
        raise ValueError("evaluation dataset is empty")
    if output is not None and (not isinstance(provenance, dict) or
                               not provenance.get("artifact_type")):
        raise ValueError("persisted evaluation results require complete provenance")
    started = time.perf_counter()
    records, summaries = [], {}
    for condition in selected:
        runner = condition_runners[condition]
        condition_rows = []
        for example in examples:
            begin = time.perf_counter()
            result = runner(example, budget)
            if not isinstance(result, Mapping):
                raise ValueError(f"condition runner returned a non-object: {condition}/{example.example_id}")
            result = dict(result)
            elapsed = time.perf_counter() - begin
            compute = _compute_metrics(result)
            if compute.get("total_latency_seconds") is None:
                compute["total_latency_seconds"] = elapsed
            budget_result = _budget_check(result, budget)
            score = _score(scorer, example, result)
            row = {"condition": condition, "example_id": example.example_id,
                   "question": example.question, "score": score,
                   "budget": budget_result, "compute": compute,
                   "answer": result.get("answer"), "transcript": result.get("transcript"),
                   "raw_result_fields": {key: value for key, value in result.items()
                                         if key not in {"transcript", "answer"}}}
            records.append(row); condition_rows.append(row)
        strict = _mean_or_none(row["score"]["strict"] for row in condition_rows)
        lenient = _mean_or_none(row["score"]["lenient"] for row in condition_rows)
        points_strict = [(float(row["compute"]["generated_tokens"]), row["score"]["strict"])
                         for row in condition_rows
                         if row["compute"]["generated_tokens"] is not None and row["score"]["strict"] is not None
                         and row["budget"]["valid"]]
        points_lenient = [(float(row["compute"]["generated_tokens"]), row["score"]["lenient"])
                          for row in condition_rows
                          if row["compute"]["generated_tokens"] is not None and row["score"]["lenient"] is not None
                          and row["budget"]["valid"]]
        strict_expertise, strict_reason = _expertise(points_strict)
        lenient_expertise, lenient_reason = _expertise(points_lenient)
        summaries[condition] = {
            "examples": len(condition_rows), "strict_score_mean": strict,
            "lenient_score_mean": lenient, "budget_valid_rate": _mean_or_none(
                float(row["budget"]["valid"]) for row in condition_rows),
            "performance_vs_compute": {"strict": points_strict, "lenient": points_lenient},
            "expertise_strict": strict_expertise, "expertise_lenient": lenient_expertise,
            "expertise_unavailable_reason": strict_reason or lenient_reason,
        }
    payload = {
        "artifact_schema_version": 2, "artifact_type": "evaluation_result",
        "phase": "evaluation_complete", "evaluation_inputs_seen": True,
        "status": "complete", "conditions": list(selected),
        "budget_profile": budget.__dict__, "examples": [example.__dict__ for example in examples],
        "results": records, "condition_summaries": summaries,
        "expertise_metric": {"definition": "best-so-far staircase weighted AUC",
                              "anchor_generated_tokens": 3000,
                              "manual_points_required": False},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "metric_definitions": {
            "input_tokens": "serialized root prompt tokens including textual map when present; latent prefix reported separately",
            "generated_tokens": "model output tokens generated by the root agent",
            "tool_calls": "typed tool actions attempted",
            "observation_bytes": "UTF-8 bytes after final tool-output truncation",
            "prefill_latency_seconds": "backend-reported initial prompt forward latency, unavailable when not exposed",
            "decode_latency_seconds": "backend-reported continuation latency, unavailable when not exposed",
            "peak_memory_bytes": "backend-reported accelerator peak allocation, null on CPU/unavailable backends",
        },
        "provenance": provenance or {},
    }
    if output is not None:
        from .io import write_json
        write_json(output, payload)
    return payload


def tool_smoke(records, search) -> dict:
    """Re-execute cached actions; this is a tool/reward smoke, not accuracy."""
    from .rewards import score_action
    latencies, warm_latencies, calls, invalid, result_count, returned_bytes = [], [], 0, 0, 0, 0
    exact, full_group, group_coverage = 0, 0, []
    for record in records:
        record_groups = set(g.group_id for g in record.evidence_groups)
        best = None
        for action in record.candidate_actions:
            start = time.perf_counter(); valid, hits, observation = search.execute(action)
            latencies.append(time.perf_counter() - start)
            start = time.perf_counter(); search.execute(action)
            warm_latencies.append(time.perf_counter() - start)
            previsible = (record.evidence_groups[0].group_id,) if record.family == "navigation" and record.observation else ()
            outcome = score_action(action, valid, hits, observation, record.evidence_groups,
                                   previsible_group_ids=previsible)
            calls += 1; invalid += int(not valid); result_count += len(hits); returned_bytes += outcome.returned_bytes
            best = outcome if best is None or outcome.reward > best.reward else best
        if best is not None:
            exact += int(bool(best.visible_group_ids))
            full_group += int(set(best.visible_group_ids) == record_groups)
            group_coverage.append(len(best.visible_group_ids) / len(record_groups) if record_groups else 0)
    return {
        "scope": "study-record action/tool smoke; not downstream StudyBench evaluation",
        "records": len(records), "tool_calls": calls, "invalid_actions": invalid,
        "result_count": result_count, "returned_observation_bytes": returned_bytes,
        "records_with_visible_evidence": exact,
        "records_with_all_required_groups": full_group,
        "mean_required_group_coverage": statistics.fmean(group_coverage) if group_coverage else 0.0,
        "cold_cache_latency_seconds": sum(latencies),
        "warm_cache_latency_seconds": sum(warm_latencies),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model_input_tokens": None, "model_output_tokens": None,
        "downstream_accuracy": None, "expertise": None,
        "reason_unmeasured": "official model checkpoint and judge/harness were not available",
    }
