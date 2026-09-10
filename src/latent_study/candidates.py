from __future__ import annotations

import json

from .agent import parse_tool_actions
from .rewards import score_action
from .schema import StudyRecord, ToolAction


def sample_base_actions(model, tokenizer, prompt: str, *, n: int, seed: int) -> tuple[ToolAction, ...]:
    """Sample actions from the frozen base LM, explicitly without any learned latent."""
    import torch
    messages = [{"role": "system", "content": (
        "Return one search action as JSON only: "
        '{"tool":"search","query":"specific query","max_results":5}.')},
        {"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_tensors="pt").to(model.device)
    actions = []
    for index in range(n):
        generator = torch.Generator(device=model.device).manual_seed(seed + index)
        with torch.no_grad():
            output = model.generate(ids, max_new_tokens=80, do_sample=True, temperature=0.8,
                                    top_p=0.95, generator=generator,
                                    pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_tool_actions(text)
        # Preserve invalid generations for scoring/logging with an invalid empty action.
        actions.append(parsed[0] if parsed else ToolAction("", 5))
    return tuple(actions)


def replace_actions(record: StudyRecord, actions: tuple[ToolAction, ...], search) -> StudyRecord:
    from dataclasses import replace
    outcomes = []
    for action in actions:
        valid, hits, observation = search.execute(action)
        previsible = (record.evidence_groups[0].group_id,) if record.family == "navigation" and record.observation else ()
        outcomes.append(score_action(action, valid, hits, observation, record.evidence_groups,
                                     previsible_group_ids=previsible))
    validation = dict(record.validation)
    validation["base_model_candidates"] = True
    return replace(record, candidate_actions=actions, outcomes=tuple(outcomes), validation=validation)
