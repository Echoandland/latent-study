#!/usr/bin/env python3
"""Strengthened tiny pinned-Qwen smoke; never a benchmark or full study run."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_study.agent import PrefixAgentSession, ranking_state_messages, study_state_messages, tool_continuation_ids
from latent_study.corpus import build_manifest
from latent_study.io import write_json
from latent_study.io import canonical_hash
from latent_study.artifacts import ARTIFACT_SCHEMA_VERSION, artifact_payload_hash, provenance
from latent_study.search import TOOL_SCHEMA_HASH
from latent_study.latent import load_qwen
from latent_study.records import generate_coverage_records
from latent_study.search import CodingTools
from latent_study.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--revision")
    parser.add_argument("--output", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17); parser.add_argument("--overfit-steps", type=int, default=2)
    parser.add_argument("--optimizer", choices=("SGD", "AdamW"), default="SGD")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--objectives", choices=("all", "query_only", "rank_only", "combined"),
                        default="all")
    args = parser.parse_args()
    smoke_config_hash = canonical_hash(vars(args))
    base_provenance = provenance(
        artifact_type="real_qwen_smoke", resolved_config_hash=smoke_config_hash,
        protocol_config_hash=canonical_hash({"protocol": "real_qwen_smoke_v1",
                                             "model_revision": args.revision,
                                             "latent_length": args.length}),
        model_id="Qwen/Qwen3.5-9B", model_revision=args.revision or "unresolved",
        corpus_hash="real-smoke-only", tool_schema_hash=TOOL_SCHEMA_HASH,
        command="python scripts/real_model_smoke.py", cli_overrides=vars(args),
        repository_root=ROOT, tokenizer_id=args.model,
        tokenizer_revision=args.revision or "unresolved")

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter(); torch_device = torch.device(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch_device); torch.cuda.reset_peak_memory_stats()
    wrapper = load_qwen(args.model, length=args.length, dtype="bfloat16", revision=args.revision, device=args.device)
    loaded = time.perf_counter(); wrapper.assert_frozen()
    rel_id, irr_id = wrapper.validate_labels("A", "B")

    with tempfile.TemporaryDirectory(prefix="latent-study-real-smoke-") as tmp:
        corpus = Path(tmp) / "corpus"; corpus.mkdir()
        # Keep both independent records in one source so this controlled
        # overfit isolates optimizer behavior from continual-source forgetting.
        definitions = [
            ("alpha", "plus one", "+ 1"), ("beta", "minus one", "- 1"),
            ("gamma", "plus two", "+ 2"), ("delta", "minus two", "- 2"),
            ("epsilon", "plus three", "+ 3"), ("zeta", "minus three", "- 3"),
        ]
        (corpus / "facts.py").write_text("\n\n".join(
            f'def {name}(value):\n    """Return value {description}."""\n    return value {expression}'
            for name, description, expression in definitions) + "\n")
        manifest = build_manifest(corpus); tools = CodingTools(corpus, manifest["units"])
        records = generate_coverage_records(manifest["units"], manifest["corpus_hash"], search=tools)
        smoke_records = records[:6]
        if len(smoke_records) < 4:
            raise RuntimeError("controlled smoke corpus did not produce enough independent records")
        record = smoke_records[0]; units = {u.unit_id: u for u in manifest["units"]}
        action_text = json.dumps(record.candidate_actions[0].to_payload(), sort_keys=True)
        _, _, cached_observation = tools.execute(record.candidate_actions[0])

        first_prefix = wrapper.prefix.detach().clone()
        common = dict(corpus_hash=manifest["corpus_hash"], model_id="Qwen/Qwen3.5-9B",
                      model_revision=args.revision or "", learning_rate=args.learning_rate,
                      optimizer_name=args.optimizer, delta=0.01, replay_fraction=0.5,
                      seed=args.seed, batch_size=2, updates_per_source=args.overfit_steps,
                      artifact_provenance={**base_provenance, "artifact_type": "trained_latent"})
        # Exercise three real production optimizer paths from the exact same
        # initializer. The combined path is not replaced by a synthetic line
        # search; its own aggregate combined objective must decrease.
        trainer_reports = {}
        objective_runs = (("query_only", (1.0, 0.0)), ("rank_only", (0.0, 1.0)),
                          ("combined", (1.0, 1.0)))
        if args.objectives != "all":
            objective_runs = tuple(item for item in objective_runs if item[0] == args.objectives)
        for condition, weights in objective_runs:
            with torch.no_grad(): wrapper.prefix.copy_(first_prefix)
            trainer_reports[condition] = train(
                wrapper, smoke_records, units, Path(tmp) / f"trainer-{condition}.pt",
                query_weight=weights[0], lambda_rank=weights[1], **common)
        if args.objectives != "all":
            print(json.dumps({args.objectives: {
                "before": trainer_reports[args.objectives]["objective_diagnostics_before"],
                "after": trainer_reports[args.objectives]["objective_diagnostics_after"],
                "checks": trainer_reports[args.objectives]["objective_decrease_checks"]}}, indent=2))
            return
        trainer_report = trainer_reports["combined"]
        optimizer_step_changed_prefix = not torch.equal(first_prefix, wrapper.prefix.detach())

    # Cached role-delimited continuation versus complete recomputation.
    base_messages = study_state_messages(record); initial_ids = wrapper.chat_ids(base_messages)
    memory_boundary = int(getattr(initial_ids, "_memory_slot_start"))
    action_ids = wrapper.tokenizer(action_text, add_special_tokens=False, return_tensors="pt").input_ids.to(wrapper.prefix.device)
    with torch.no_grad():
        session = PrefixAgentSession(wrapper, initial_ids); session.append_tool_turn(action_ids)
        consumed = torch.cat((initial_ids, action_ids), dim=1)
        suffix = tool_continuation_ids(wrapper.tokenizer, action_ids, cached_observation).to(wrapper.prefix.device)
        session.append_tool_turn(suffix)
        target_ids = torch.cat((consumed, suffix), dim=1)
        full_state = wrapper.prefill_ids(target_ids, use_cache=False, memory_boundary=memory_boundary)
        errors = [float((session.state.next_logits.float() - full_state.next_logits.float()).abs().max().cpu())]
        continuation = wrapper.tokenizer(" inspect more evidence now", add_special_tokens=False,
                                         return_tensors="pt").input_ids.to(wrapper.prefix.device)
        for index in range(continuation.shape[1]):
            session.append_tool_turn(continuation[:, index:index+1])
            recomputed = wrapper.prefill_ids(torch.cat((target_ids, continuation[:, :index+1]), dim=1),
                                             use_cache=False, memory_boundary=memory_boundary)
            errors.append(float((session.state.next_logits.float() - recomputed.next_logits.float()).abs().max().cpu()))
    wrapper.save(args.checkpoint, corpus_hash="real-smoke-only", model_id="Qwen/Qwen3.5-9B",
                 model_revision=args.revision or "", provenance={**base_provenance,
                                                                 "artifact_type": "real_smoke_latent"})
    reference_ids = wrapper.chat_ids([{"role": "system", "content": "Synthetic reload check."},
                                      {"role": "user", "content": "reload comparison"}])
    reference_logits = wrapper.prefill_ids(reference_ids, use_cache=False).next_logits.detach().float().cpu()
    reference_provenance = {**base_provenance, "artifact_type": "reload_reference",
                            "artifact_sha256": None}
    reference_payload = {"input_ids": reference_ids.cpu(), "logits": reference_logits,
                         "memory_boundary": int(getattr(reference_ids, "_memory_slot_start")),
                         "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                         "provenance": reference_provenance}
    reference_provenance["artifact_sha256"] = artifact_payload_hash(reference_payload)
    torch.save(reference_payload, str(args.checkpoint) + ".reference.pt")
    if args.device.startswith("cuda"): torch.cuda.synchronize()
    report = {
        "scope": "strengthened real-model smoke; synthetic corpus only", "model": "Qwen/Qwen3.5-9B",
        "revision": args.revision, "hidden_size": wrapper.hidden_size, "latent_length": wrapper.length,
        "ranking_labels": {"A": rel_id, "B": irr_id}, "all_lm_parameters_frozen": not any(p.requires_grad for p in wrapper.model.parameters()),
        "lm_parameters_bitwise_unchanged": trainer_report["frozen_lm_bitwise_unchanged"],
        "optimizer_step_changed_prefix": optimizer_step_changed_prefix,
        "lm_gradient_buffers": trainer_report["lm_gradient_buffers"],
        "controlled_overfit": {
            "steps": args.overfit_steps,
            "query_only": {"before": trainer_reports["query_only"]["objective_diagnostics_before"],
                           "after": trainer_reports["query_only"]["objective_diagnostics_after"],
                           "checks": trainer_reports["query_only"]["objective_decrease_checks"]},
            "rank_only": {"before": trainer_reports["rank_only"]["objective_diagnostics_before"],
                          "after": trainer_reports["rank_only"]["objective_diagnostics_after"],
                          "checks": trainer_reports["rank_only"]["objective_decrease_checks"]},
            "combined": {"before": trainer_report["objective_diagnostics_before"],
                         "after": trainer_report["objective_diagnostics_after"],
                         "checks": trainer_report["objective_decrease_checks"]}},
        "actual_train_path": {"optimizer": trainer_report["optimizer"],
                              "updates": len(trainer_report["losses"]),
                              "final_loss": trainer_report["losses"][-1]},
        "cache_equivalence": {"observation_tokens": int(target_ids.shape[1] - consumed.shape[1]),
                              "continuation_tokens": int(continuation.shape[1]), "per_step_max_abs_error": errors,
                              "max_abs_error": max(errors), "tolerance": 0.0,
                              "status": "full_recompute_fallback",
                              "cache_correctness_claimed": False,
                              "equivalence_verified": False,
                              "fallback_internal_consistency": max(errors) == 0.0,
                              "tolerance_basis": "fallback compared with identical full recomputation; not a cache claim"},
        "hybrid_state_mode": session.state.cache_mode,
        "prefix_insertions": session.state.prefix_insertions, "model_load_seconds": loaded - started,
        "total_seconds": time.perf_counter() - started,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "latent_serialized_bytes": Path(args.checkpoint).stat().st_size,
        "reload_reference": str(args.checkpoint) + ".reference.pt",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "provenance": base_provenance}
    # A full-recompute fallback can be internally self-consistent, but that is
    # not evidence that a cached continuation is equivalent. Keep the public
    # cache gate explicitly unverified/failed until a native Qwen path exists.
    report["cache_equivalence"]["passed"] = False
    write_json(args.output, report); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
