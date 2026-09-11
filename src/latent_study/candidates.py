from __future__ import annotations

import time
from .io import canonical_hash
from .agent import parse_tool_actions, serialize_chat_ids, study_state_messages
from .rewards import score_action
from .schema import StudyRecord, ToolAction


def record_seed(global_seed: int, record_id: str) -> int:
    """Stable across worker count, worker assignment, and local enumeration."""
    return int(canonical_hash({"seed": global_seed, "record_id": record_id})[:15], 16)


def sample_base_actions(model, tokenizer, record: StudyRecord, *, n: int, seed: int,
                        max_new_tokens: int = 96, do_sample: bool = True,
                        temperature: float = 0.8, top_p: float = 0.95) -> tuple[ToolAction, ...]:
    """Sample actions from the frozen base LM, explicitly without any learned latent."""
    import torch
    messages = study_state_messages(record)
    messages.append({"role": "user", "content": "Return one legal coding-tool action as JSON only."})
    ids = serialize_chat_ids(tokenizer, messages, add_generation_prompt=True).to(model.device)
    actions = []
    torch.manual_seed(seed)
    if str(model.device).startswith("cuda"): torch.cuda.manual_seed_all(seed)
    batch_ids = ids.repeat(n, 1)
    started = time.perf_counter()
    generation = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                  "pad_token_id": tokenizer.eos_token_id}
    if do_sample:
        generation.update({"temperature": temperature, "top_p": top_p})
    with torch.no_grad():
        output = model.generate(batch_ids, **generation)
    elapsed = time.perf_counter() - started
    stats = getattr(model, "_latent_study_candidate_stats", {"model_calls": 0, "input_tokens": 0,
                                                             "output_tokens": 0, "latency_seconds": 0.0})
    stats["model_calls"] += 1; stats["input_tokens"] += int(batch_ids.numel())
    stats["output_tokens"] += int(output[:, ids.shape[1]:].numel()); stats["latency_seconds"] += elapsed
    model._latent_study_candidate_stats = stats
    for index in range(n):
        text = tokenizer.decode(output[index, ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_tool_actions(text)
        # Preserve invalid generations for scoring/logging with an invalid empty action.
        actions.append(parsed[0] if parsed else ToolAction(query="", tool="grep", max_results=5))
    return tuple(actions)


def replace_actions(record: StudyRecord, actions: tuple[ToolAction, ...], search) -> StudyRecord:
    from dataclasses import replace
    outcomes = []
    for action in actions:
        valid, hits, observation = search.execute(action)
        previsible = tuple(record.validation.get("previsible_group_ids", ()))
        outcomes.append(score_action(action, valid, hits, observation, record.evidence_groups,
                                     previsible_group_ids=previsible))
    validation = dict(record.validation)
    validation["base_model_candidates"] = True
    return replace(record, candidate_actions=actions, outcomes=tuple(outcomes), validation=validation)
