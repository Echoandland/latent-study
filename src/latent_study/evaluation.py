from __future__ import annotations

import resource
import statistics
import time


def tool_smoke(records, search) -> dict:
    """Re-execute cached actions; this is a tool/reward smoke, not downstream accuracy."""
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
