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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .io import canonical_hash


MVP_CONDITIONS = (
    "no_study", "random_latent_L64", "trained_latent_L64",
    "offline_peek_64", "offline_peek_1024",
)
EXPERTISE_ANCHOR_TOKENS = 3000.0


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


def _evaluation_files(path: Path) -> list[Path]:
    return sorted(item for item in (path.rglob("*") if path.is_dir() else [path])
                  if item.is_file() and item.suffix.lower() in {".json", ".jsonl"})


def evaluation_dataset_snapshot(paths: str | Path | Iterable[str | Path]) -> dict:
    """Canonical identity of all evaluation content, independent of root path.

    Parsed JSON values (including answers and rubrics) and paths relative to
    each supplied dataset root are hashed.  Absolute checkout locations and
    JSON whitespace are deliberately excluded from the identity.
    """
    if isinstance(paths, (str, Path)):
        roots = [Path(paths)]
    else:
        roots = [Path(path) for path in paths]
    entries = []
    for root in roots:
        files = _evaluation_files(root)
        if not files:
            raise ValueError(f"evaluation dataset has no JSON/JSONL files: {root}")
        for file in files:
            relative = file.relative_to(root).as_posix() if root.is_dir() else file.name
            entries.append({"path": relative, "content": _json_values(file)})
    entries.sort(key=lambda item: (item["path"], canonical_hash(item["content"])))
    return {"schema_version": 1, "sha256": canonical_hash(entries),
            "file_count": len(entries), "paths": [item["path"] for item in entries]}


def evaluation_dataset_snapshot_hash(path: str | Path | Iterable[str | Path]) -> str:
    return evaluation_dataset_snapshot(path)["sha256"]


def load_evaluation_dataset(path: str | Path) -> list[EvaluationExample]:
    """Load evaluation-only examples with stable IDs."""
    root = Path(path)
    files = _evaluation_files(root)
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


def budgets_from_config(config: Mapping[str, Any], names: Iterable[str] | None = None) -> tuple[EvaluationBudgetProfile, ...]:
    """Resolve a deterministic set of distinct evaluation compute budgets."""
    selected = tuple(names) if names is not None else tuple(config.get("evaluation", {}).get("budgets", {}))
    if not selected:
        raise ValueError("at least one evaluation budget must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("evaluation budget names must not be duplicated")
    profiles = tuple(budget_from_config(config, name) for name in selected)
    expertise_config = config.get("evaluation", {}).get("expertise", {})
    if expertise_config.get("require_anchor_coverage", False):
        validate_expertise_budget_coverage(
            profiles, float(expertise_config.get("anchor_generated_tokens", EXPERTISE_ANCHOR_TOKENS)))
    return profiles


def validate_expertise_budget_coverage(
        profiles: Iterable[EvaluationBudgetProfile],
        anchor_tokens: float = EXPERTISE_ANCHOR_TOKENS) -> None:
    """Reject a curve whose configured per-example generation caps miss the anchor."""
    values = [profile.max_output_tokens for profile in profiles]
    if not values or not math.isfinite(anchor_tokens) or anchor_tokens <= 0:
        raise ValueError("expertise requires a positive finite generated-token anchor")
    if max(values) < anchor_tokens:
        raise ValueError(
            f"evaluation generated-token budgets do not reach the {anchor_tokens:g}-token expertise anchor")


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


def _expertise(points: list[tuple[float, float]], anchor_tokens: float) -> tuple[float | None, str | None, list[tuple[float, float]]]:
    eligible = [(tokens, score) for tokens, score in points if tokens >= anchor_tokens]
    if not points:
        return None, "no budget-level aggregate judge score", []
    if not eligible:
        return None, f"no budget-level aggregate point reaches the {anchor_tokens:g}-token expertise anchor", []
    from .metrics import expertise
    return expertise(eligible, anchor_tokens=anchor_tokens), None, eligible


def _sum_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) if clean else None


def _aggregate_compute(rows: list[dict]) -> dict:
    fields = ("input_tokens", "generated_tokens", "tool_calls", "observation_bytes",
              "total_latency_seconds", "prefill_latency_seconds", "decode_latency_seconds")
    result = {}
    for field in fields:
        values = [row["compute"].get(field) for row in rows]
        result[field] = {"mean_per_example": _mean_or_none(values),
                         "total": _sum_or_none(values),
                         "measured_examples": sum(value is not None for value in values)}
    peaks = [row["compute"].get("peak_memory_bytes") for row in rows
             if row["compute"].get("peak_memory_bytes") is not None]
    result["peak_memory_bytes"] = {
        "max_across_example_runs": max(peaks) if peaks else None,
        "measured_examples": len(peaks),
        "scope": "each example run resets CUDA peak statistics before inference",
    }
    return result


class SequentialRunnerFactory:
    """Lease exactly one condition/budget runner at a time.

    The factory retains only its builder.  A released runner is never stored,
    which prevents condition-specific agents or model wrappers from
    accumulating across the five-condition evaluation.
    """
    def __init__(self, build: Callable, release: Callable | None = None, *,
                 shared_backbone: Any | None = None):
        self._build = build
        self._release = release
        self.shared_backbone = shared_backbone
        self._active = False

    @contextmanager
    def lease(self, condition: str, budget: EvaluationBudgetProfile):
        if self._active:
            raise RuntimeError("condition runners must execute sequentially")
        self._active = True
        runner = self._build(condition, budget)
        try:
            yield runner
        finally:
            try:
                if self._release is not None:
                    self._release(runner)
                elif hasattr(runner, "close"):
                    runner.close()
            finally:
                runner = None
                self._active = False


def run_evaluation(examples: list[EvaluationExample], condition_runners: Mapping[str, Callable] | None = None, *,
                   budget: EvaluationBudgetProfile | None = None,
                   budgets: Iterable[EvaluationBudgetProfile] | None = None,
                   runner_factory: SequentialRunnerFactory | None = None,
                   scorer: Callable | None = None,
                   conditions: Iterable[str] = MVP_CONDITIONS, output: str | Path | None = None,
                   provenance: dict | None = None, fair_conditions: list[Any] | None = None,
                   dataset_snapshot: dict | None = None,
                   expertise_anchor_tokens: float = EXPERTISE_ANCHOR_TOKENS,
                   require_expertise_budget_coverage: bool = True) -> dict:
    """Run all MVP conditions and emit an expertise-ready structured artifact."""
    selected = tuple(conditions)
    if selected != MVP_CONDITIONS:
        raise ValueError(f"evaluation must compare exactly the five MVP conditions: {MVP_CONDITIONS}")
    if (condition_runners is None) == (runner_factory is None):
        raise ValueError("provide exactly one of condition_runners or runner_factory")
    if condition_runners is not None and set(condition_runners) != set(selected):
        raise ValueError("condition runners do not cover exactly the five MVP conditions")
    profiles = tuple(budgets or (() if budget is None else (budget,)))
    if not profiles:
        raise ValueError("evaluation requires at least one compute budget")
    if len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("evaluation budget names must be unique")
    if require_expertise_budget_coverage:
        validate_expertise_budget_coverage(profiles, expertise_anchor_tokens)
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
        budget_summaries = {}
        points_strict, points_lenient = [], []
        curve_strict, curve_lenient = [], []
        for profile in profiles:
            condition_rows = []
            scope = (runner_factory.lease(condition, profile) if runner_factory is not None
                     else nullcontext(condition_runners[condition]))
            with scope as runner:
                for example in examples:
                    begin = time.perf_counter()
                    result = runner(example, profile)
                    if not isinstance(result, Mapping):
                        raise ValueError(
                            f"condition runner returned a non-object: {condition}/{example.example_id}")
                    result = dict(result)
                    elapsed = time.perf_counter() - begin
                    compute = _compute_metrics(result)
                    if compute.get("total_latency_seconds") is None:
                        compute["total_latency_seconds"] = elapsed
                        compute["unavailable"].pop("total_latency_seconds", None)
                    budget_result = _budget_check(result, profile)
                    score = _score(scorer, example, result)
                    row = {"condition": condition, "example_id": example.example_id,
                           "question": example.question, "score": score,
                           "budget_profile": profile.name, "budget": budget_result,
                           "compute": compute, "answer": result.get("answer"),
                           "transcript": result.get("transcript"),
                           "raw_result_fields": {key: value for key, value in result.items()
                                                 if key not in {"transcript", "answer"}}}
                    records.append(row); condition_rows.append(row)
            # A ``with`` target remains bound in Python after __exit__.  Drop
            # it before the next lease so the previous agent/prefix cannot be
            # live while the following condition runner is constructed.
            del runner
            strict = _mean_or_none(row["score"]["strict"] for row in condition_rows)
            lenient = _mean_or_none(row["score"]["lenient"] for row in condition_rows)
            compute_aggregate = _aggregate_compute(condition_rows)
            generated_mean = compute_aggregate["generated_tokens"]["mean_per_example"]
            all_valid = all(row["budget"]["valid"] for row in condition_rows)
            summary = {
                "examples": len(condition_rows), "strict_score_mean": strict,
                "lenient_score_mean": lenient,
                "budget_valid_examples": sum(row["budget"]["valid"] for row in condition_rows),
                "budget_valid_rate": _mean_or_none(
                    float(row["budget"]["valid"]) for row in condition_rows),
                "compute": compute_aggregate,
            }
            budget_summaries[profile.name] = summary
            for score_name, score_value, point_list, curve in (
                    ("strict", strict, points_strict, curve_strict),
                    ("lenient", lenient, points_lenient, curve_lenient)):
                point = {"budget_profile": profile.name, "examples": len(condition_rows),
                         "compute_quantity": "configured_generated_token_budget_per_example",
                         "generated_token_budget_per_example": profile.max_output_tokens,
                         "mean_actual_generated_tokens_per_example": generated_mean,
                         "total_generated_tokens": compute_aggregate["generated_tokens"]["total"],
                         "aggregate_score": score_value, "all_examples_budget_valid": all_valid}
                curve.append(point)
                if score_value is not None and all_valid:
                    point_list.append((float(profile.max_output_tokens), float(score_value)))
        strict_expertise, strict_reason, strict_inputs = _expertise(points_strict, expertise_anchor_tokens)
        lenient_expertise, lenient_reason, lenient_inputs = _expertise(points_lenient, expertise_anchor_tokens)
        summaries[condition] = {
            "budgets": budget_summaries,
            "performance_vs_compute": {
                "compute_quantity": "configured_generated_token_budget_per_example",
                "strict": curve_strict, "lenient": curve_lenient},
            "expertise_strict": strict_expertise, "expertise_lenient": lenient_expertise,
            "expertise_input_points": {"strict": strict_inputs, "lenient": lenient_inputs},
            "expertise_unavailable_reason": strict_reason or lenient_reason,
        }
    from .artifacts import ARTIFACT_SCHEMA_VERSION
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "artifact_type": "evaluation_result",
        "phase": "evaluation_complete", "evaluation_inputs_seen": True,
        "status": "complete", "conditions": list(selected),
        "model_lifetime": {
            "strategy": ("single_shared_backbone" if runner_factory is not None
                         and runner_factory.shared_backbone is not None
                         else "sequential_condition_runner_leases"),
            "max_live_condition_runners": 1,
        },
        "budget_profiles": [profile.__dict__ for profile in profiles],
        "budget_profile": profiles[0].__dict__ if len(profiles) == 1 else None,
        "examples": [example.__dict__ for example in examples],
        "evaluation_dataset_snapshot": dataset_snapshot,
        "results": records, "condition_summaries": summaries,
        "expertise_metric": {"definition": "best-so-far staircase weighted AUC",
                              "anchor_generated_tokens": expertise_anchor_tokens,
                              "input_compute_quantity": "configured generated-token budget per example",
                              "input_performance_quantity": "benchmark mean strict/lenient score at that budget",
                              "below_anchor_points": "reported in the curve but excluded from expertise integration",
                              "manual_points_required": False},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "metric_definitions": {
            "input_tokens": "serialized root prompt tokens including textual map when present; latent prefix reported separately",
            "generated_tokens": "model output tokens generated by the root agent",
            "expertise_compute_axis": "configured per-example generated-token cap; actual mean/total generation is reported separately",
            "tool_calls": "typed tool actions attempted",
            "observation_bytes": "UTF-8 bytes after final tool-output truncation",
            "total_latency_seconds": "per-example wall-clock runner latency; backend measurement preferred and harness timer used otherwise",
            "prefill_latency_seconds": "backend-reported initial prompt forward latency, unavailable when not exposed",
            "decode_latency_seconds": "backend-reported continuation latency, unavailable when not exposed",
            "peak_memory_bytes": "maximum accelerator allocation after resetting peak statistics at the start of each example run; null on CPU/unavailable backends",
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
