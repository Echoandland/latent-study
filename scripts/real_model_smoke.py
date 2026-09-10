#!/usr/bin/env python3
"""Tiny pinned Qwen3.5 forward/backward/cache smoke; never a benchmark run."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_study.agent import PrefixAgentSession
from latent_study.io import write_json
from latent_study.latent import load_qwen
from latent_study.objectives import query_pairwise_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    torch_device = torch.device(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch_device)
        torch.cuda.reset_peak_memory_stats()
    wrapper = load_qwen(args.model, length=args.length, dtype="bfloat16",
                        revision=args.revision, device=args.device)
    loaded = time.perf_counter()
    relevant_id, irrelevant_id = wrapper.validate_labels("A", "B")
    context = wrapper.chat_ids([{"role": "user", "content": "Search for alpha."}],
                               add_generation_prompt=True)
    positive = wrapper.tokenizer('{"tool":"search","query":"alpha"}', add_special_tokens=False,
                                 return_tensors="pt").input_ids.to(wrapper.prefix.device)
    negative = wrapper.tokenizer('{"tool":"search","query":"other"}', add_special_tokens=False,
                                 return_tensors="pt").input_ids.to(wrapper.prefix.device)
    pos_logp = wrapper.mean_action_logprob(context, positive)
    neg_logp = wrapper.mean_action_logprob(context, negative)
    loss = query_pairwise_loss(pos_logp, neg_logp, 1.0)
    loss.backward()
    gradient_nonzero = bool(torch.count_nonzero(wrapper.prefix.grad).item())
    frozen_grad_buffers = sum(p.grad is not None for p in wrapper.model.parameters())
    backward_done = time.perf_counter()

    wrapper.save(args.checkpoint, corpus_hash="real-smoke-only", model_id="Qwen/Qwen3.5-9B")
    before = wrapper.prefill_ids(context, use_cache=False).next_logits.detach().float().cpu()
    with torch.no_grad():
        wrapper.prefix.add_(0.5)
    wrapper.load(args.checkpoint, expected_corpus_hash="real-smoke-only")
    after = wrapper.prefill_ids(context, use_cache=False).next_logits.detach().float().cpu()
    logits_preserved = bool(torch.equal(before, after))

    with torch.no_grad():
        session = PrefixAgentSession(wrapper, context)
        tool_tokens = wrapper.tokenizer("tool result", add_special_tokens=False,
                                        return_tensors="pt").input_ids.to(wrapper.prefix.device)
        session.append_tool_turn(tool_tokens)
        generated, final_state = wrapper.generate_from_state(session.state, max_new_tokens=2)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    finished = time.perf_counter()
    report = {
        "scope": "tiny real-model compatibility smoke; no corpus training or downstream evaluation",
        "model": "Qwen/Qwen3.5-9B", "revision": args.revision,
        "model_path": str(Path(args.model).resolve()), "model_type": wrapper.model.config.model_type,
        "hidden_size": wrapper.hidden_size, "latent_length": wrapper.length,
        "seed": args.seed,
        "ranking_labels": {"A": relevant_id, "B": irrelevant_id},
        "all_lm_parameters_frozen": not any(p.requires_grad for p in wrapper.model.parameters()),
        "prefix_gradient_nonzero": gradient_nonzero,
        "lm_gradient_buffers": frozen_grad_buffers,
        "latent_save_load_logits_exact": logits_preserved,
        "prefix_insertions_after_tool_and_generation": final_state.prefix_insertions,
        "tool_observation_tokens_appended": int(tool_tokens.shape[1]),
        "generated_tokens": int(generated.shape[1]),
        "query_loss": float(loss.detach()),
        "model_load_seconds": loaded - started,
        "forward_backward_seconds": backward_done - loaded,
        "total_seconds": finished - started,
        "gpu_peak_memory_bytes": (torch.cuda.max_memory_allocated()
                                  if args.device.startswith("cuda") else None),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "latent_serialized_bytes": Path(args.checkpoint).stat().st_size,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
