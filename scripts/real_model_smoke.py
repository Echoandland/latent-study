#!/usr/bin/env python3
"""Strengthened tiny pinned-Qwen smoke; never a benchmark or full study run."""
from __future__ import annotations

import argparse
import hashlib
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
from latent_study.latent import load_qwen
from latent_study.objectives import query_pairwise_loss, ranking_loss, select_preference
from latent_study.records import generate_coverage_records
from latent_study.search import CodingTools
from latent_study.train import train


def model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().contiguous().view(-1).view(__import__("torch").uint8).cpu().numpy())
    return digest.hexdigest()


def query_loss(wrapper, record):
    pair = select_preference(record.outcomes, 0.01)
    if pair is None: raise RuntimeError("controlled smoke record has no preference pair")
    positive, negative = pair
    context = wrapper.chat_ids(study_state_messages(record))
    pos_ids = wrapper.tokenizer(json.dumps(positive.action.to_payload(), sort_keys=True), add_special_tokens=False,
                                return_tensors="pt").input_ids.to(wrapper.prefix.device)
    neg_ids = wrapper.tokenizer(json.dumps(negative.action.to_payload(), sort_keys=True), add_special_tokens=False,
                                return_tensors="pt").input_ids.to(wrapper.prefix.device)
    return query_pairwise_loss(wrapper.mean_action_logprob(context, pos_ids),
                               wrapper.mean_action_logprob(context, neg_ids),
                               positive.reward - negative.reward)


def rank_loss(wrapper, record, units, rel_id, irr_id):
    positives, negatives = [], []
    for alternative in record.evidence_groups[0].alternatives:
        positives.append(wrapper.label_score(wrapper.chat_ids(ranking_state_messages(record, alternative.text, "A", "B")),
                                             rel_id, irr_id))
    for chunk_id in record.verified_negative_chunk_ids:
        negatives.append(wrapper.label_score(wrapper.chat_ids(ranking_state_messages(record, units[chunk_id].text, "A", "B")),
                                             rel_id, irr_id))
    return ranking_loss([(positives, negatives)])


def scaled_sgd_step(optimizer, wrapper, step_norm: float = 0.05):
    norm = float(wrapper.prefix.grad.float().norm())
    optimizer.param_groups[0]["lr"] = step_norm / max(norm, 1e-12)
    optimizer.step()


def synthetic_line_step(wrapper, evaluate, current: float):
    """Tiny synthetic-only line search to demonstrate objective learnability."""
    import torch
    base = wrapper.prefix.detach().clone(); direction = wrapper.prefix.grad.detach()
    direction = direction / direction.float().norm().clamp_min(1e-12)
    best_value, best_prefix = current, base
    for signed_scale in (-0.02, -0.005, -0.001, 0.001, 0.005, 0.02):
        with torch.no_grad(): wrapper.prefix.copy_(base + signed_scale * direction)
        with torch.no_grad(): value = float(evaluate())
        if value < best_value: best_value, best_prefix = value, wrapper.prefix.detach().clone()
        if best_value < current: break
    with torch.no_grad(): wrapper.prefix.copy_(best_prefix)
    return best_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--revision")
    parser.add_argument("--output", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17); parser.add_argument("--overfit-steps", type=int, default=2)
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter(); torch_device = torch.device(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch_device); torch.cuda.reset_peak_memory_stats()
    wrapper = load_qwen(args.model, length=args.length, dtype="bfloat16", revision=args.revision, device=args.device)
    loaded = time.perf_counter(); wrapper.assert_frozen(); digest_before = model_digest(wrapper.model)
    rel_id, irr_id = wrapper.validate_labels("A", "B")

    with tempfile.TemporaryDirectory(prefix="latent-study-real-smoke-") as tmp:
        corpus = Path(tmp) / "corpus"; corpus.mkdir()
        (corpus / "alpha.py").write_text("def alpha(value):\n    \"\"\"Return value plus one.\"\"\"\n    return value + 1\n")
        (corpus / "beta.py").write_text("def beta(value):\n    \"\"\"Return value minus one.\"\"\"\n    return value - 1\n")
        manifest = build_manifest(corpus); tools = CodingTools(corpus, manifest["units"])
        records = generate_coverage_records(manifest["units"], manifest["corpus_hash"], search=tools)
        record = records[0]; units = {u.unit_id: u for u in manifest["units"]}
        action_text = json.dumps(record.candidate_actions[0].to_payload(), sort_keys=True)
        _, _, cached_observation = tools.execute(record.candidate_actions[0])

        with torch.no_grad():
            query_before = float(query_loss(wrapper, record))
            rank_before = float(rank_loss(wrapper, record, units, rel_id, irr_id))
        first_prefix = wrapper.prefix.detach().clone()
        ranking_gradient_nonzero = False
        for _ in range(args.overfit_steps):
            wrapper.prefix.grad = None
            query = query_loss(wrapper, record); current = float(query.detach()); query.backward(); del query
            query_only_after = synthetic_line_step(wrapper, lambda: query_loss(wrapper, record), current)
        with torch.no_grad(): wrapper.prefix.copy_(first_prefix)
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1.0)
        for _ in range(args.overfit_steps):
            optimizer.zero_grad(set_to_none=True)
            rank = rank_loss(wrapper, record, units, rel_id, irr_id); rank.backward(); del rank
            ranking_gradient_nonzero |= bool(torch.count_nonzero(wrapper.prefix.grad).item())
            scaled_sgd_step(optimizer, wrapper)
        with torch.no_grad(): rank_only_after = float(rank_loss(wrapper, record, units, rel_id, irr_id))
        with torch.no_grad(): wrapper.prefix.copy_(first_prefix)
        optimizer = torch.optim.SGD(wrapper.trainable_parameters(), lr=1.0)
        for _ in range(args.overfit_steps):
            optimizer.zero_grad(set_to_none=True)
            query = query_loss(wrapper, record); query.backward(); del query
            rank = rank_loss(wrapper, record, units, rel_id, irr_id); rank.backward(); del rank
            scaled_sgd_step(optimizer, wrapper)
        with torch.no_grad():
            combined_query_after = float(query_loss(wrapper, record))
            combined_rank_after = float(rank_loss(wrapper, record, units, rel_id, irr_id))
        if any(p.grad is not None for p in wrapper.model.parameters()):
            raise RuntimeError("frozen model acquired a gradient buffer")
        optimizer_step_changed_prefix = not torch.equal(first_prefix, wrapper.prefix.detach())
        wrapper.model.zero_grad(set_to_none=True); wrapper.prefix.grad = None

        # Actual trainer path (query-only here; real ranking backward was exercised above).
        trainer_checkpoint = Path(tmp) / "trainer.pt"
        trainer_report = train(wrapper, records[:2], units, trainer_checkpoint,
                               corpus_hash=manifest["corpus_hash"], model_id="Qwen/Qwen3.5-9B",
                               learning_rate=1e-3, query_weight=1.0, lambda_rank=0.0,
                               delta=0.01, replay_fraction=0.5, seed=args.seed,
                               batch_size=2, updates_per_source=1)

    # Cached role-delimited continuation versus complete recomputation.
    base_messages = study_state_messages(record); initial_ids = wrapper.chat_ids(base_messages)
    action_ids = wrapper.tokenizer(action_text, add_special_tokens=False, return_tensors="pt").input_ids.to(wrapper.prefix.device)
    with torch.no_grad():
        session = PrefixAgentSession(wrapper, initial_ids); session.append_tool_turn(action_ids)
        consumed = torch.cat((initial_ids, action_ids), dim=1)
        suffix = tool_continuation_ids(wrapper.tokenizer, action_ids, cached_observation).to(wrapper.prefix.device)
        session.append_tool_turn(suffix)
        target_ids = torch.cat((consumed, suffix), dim=1)
        full_state = wrapper.prefill_ids(target_ids, use_cache=False)
        errors = [float((session.state.next_logits.float() - full_state.next_logits.float()).abs().max().cpu())]
        continuation = wrapper.tokenizer(" inspect more evidence now", add_special_tokens=False,
                                         return_tensors="pt").input_ids.to(wrapper.prefix.device)
        for index in range(continuation.shape[1]):
            session.append_tool_turn(continuation[:, index:index+1])
            recomputed = wrapper.prefill_ids(torch.cat((target_ids, continuation[:, :index+1]), dim=1), use_cache=False)
            errors.append(float((session.state.next_logits.float() - recomputed.next_logits.float()).abs().max().cpu()))
    wrapper.save(args.checkpoint, corpus_hash="real-smoke-only", model_id="Qwen/Qwen3.5-9B")
    reference_ids = wrapper.chat_ids([{"role": "user", "content": "reload comparison"}])
    reference_logits = wrapper.prefill_ids(reference_ids, use_cache=False).next_logits.detach().float().cpu()
    torch.save({"input_ids": reference_ids.cpu(), "logits": reference_logits}, str(args.checkpoint) + ".reference.pt")
    digest_after = model_digest(wrapper.model)
    if args.device.startswith("cuda"): torch.cuda.synchronize()
    report = {
        "scope": "strengthened real-model smoke; synthetic corpus only", "model": "Qwen/Qwen3.5-9B",
        "revision": args.revision, "hidden_size": wrapper.hidden_size, "latent_length": wrapper.length,
        "ranking_labels": {"A": rel_id, "B": irr_id}, "all_lm_parameters_frozen": not any(p.requires_grad for p in wrapper.model.parameters()),
        "lm_parameters_bitwise_unchanged": digest_before == digest_after,
        "optimizer_step_changed_prefix": optimizer_step_changed_prefix,
        "ranking_forward_backward_prefix_gradient_nonzero": ranking_gradient_nonzero,
        "controlled_overfit": {"steps_per_control": args.overfit_steps,
                               "query_only": {"before": query_before, "after": query_only_after},
                               "rank_only": {"before": rank_before, "after": rank_only_after},
                               "combined": {"before": query_before + rank_before,
                                            "after": combined_query_after + combined_rank_after,
                                            "query_after": combined_query_after,
                                            "rank_after": combined_rank_after}},
        "actual_train_path": {"updates": len(trainer_report["losses"]), "final_loss": trainer_report["losses"][-1]},
        "cache_equivalence": {"observation_tokens": int(target_ids.shape[1] - consumed.shape[1]),
                              "continuation_tokens": int(continuation.shape[1]), "per_step_max_abs_error": errors,
                              "max_abs_error": max(errors), "tolerance": 0.0,
                              "tolerance_basis": "Qwen3.5 torch DeltaNet uses explicit full-recompute fallback"},
        "hybrid_state_mode": session.state.cache_mode,
        "prefix_insertions": session.state.prefix_insertions, "model_load_seconds": loaded - started,
        "total_seconds": time.perf_counter() - started,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "latent_serialized_bytes": Path(args.checkpoint).stat().st_size,
        "reload_reference": str(args.checkpoint) + ".reference.pt"}
    if report["cache_equivalence"]["max_abs_error"] > report["cache_equivalence"]["tolerance"]:
        report["cache_equivalence"]["passed"] = False
    else: report["cache_equivalence"]["passed"] = True
    write_json(args.output, report); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
