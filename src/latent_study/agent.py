from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .schema import ToolAction
from .search import TOOL_SCHEMAS, TOOL_SCHEMA_HASH, CodingTools, serialize_tool_observation


_ACTION_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_tool_actions(text: str) -> list[ToolAction]:
    """Parse every legal typed coding-tool JSON object in a multi-turn response."""
    actions = []
    for blob in _ACTION_RE.findall(text):
        try:
            value = json.loads(blob)
        except json.JSONDecodeError:
            continue
        maximum = value.get("max_results", 5)
        if not isinstance(maximum, int):
            continue
        tool = value.get("tool")
        if tool == "grep" and isinstance(value.get("query"), str):
            actions.append(ToolAction(query=value["query"], max_results=maximum, tool="grep",
                                      path=value.get("path", ".")))
        elif tool == "glob" and isinstance(value.get("pattern"), str):
            actions.append(ToolAction(max_results=maximum, tool="glob", pattern=value["pattern"]))
        elif (tool == "read_file" and isinstance(value.get("path"), str)
              and isinstance(value.get("start_line"), int) and isinstance(value.get("end_line"), int)):
            actions.append(ToolAction(tool="read_file", path=value["path"],
                                      start_line=value["start_line"], end_line=value["end_line"]))
    return actions


ROOT_PROMPT_REVISION = "local-coding-agent-v2"
ROOT_SYSTEM_PROMPT = (
    "You are a frozen corpus coding agent. Use only these JSON tool actions when evidence is needed: "
    + json.dumps(TOOL_SCHEMAS, sort_keys=True)
    + " Return a final answer as JSON {\"final\":\"...\"}. Evidence must be visible in tool output."
)


def study_state_messages(record, *, memory_text: str = "") -> list[dict[str, str]]:
    system = ROOT_SYSTEM_PROMPT
    if memory_text:
        system += "\n\nFrozen corpus memory map:\n" + memory_text
    messages = [{"role": "system", "content": system}, {"role": "user", "content": record.prompt}]
    if record.observation:
        action = record.observation_action or ToolAction(tool="grep", query="prior evidence")
        messages.append({"role": "assistant", "content": json.dumps(action.to_payload(), sort_keys=True)})
        messages.append({"role": "tool", "content": serialize_tool_observation(action, record.observation)})
    return messages


def ranking_state_messages(record, candidate_text: str, relevant_label: str, irrelevant_label: str):
    messages = study_state_messages(record)
    messages.append({"role": "user", "content":
                     f"Candidate evidence:\n{candidate_text}\n\nOutput exactly {relevant_label} if relevant or {irrelevant_label} if irrelevant."})
    return messages


def serialize_chat_ids(tokenizer, messages, *, add_generation_prompt: bool = True):
    """Prefix-stable Qwen role serialization; falls back to the tokenizer template."""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>") if hasattr(tokenizer, "convert_tokens_to_ids") else None
    unk = getattr(tokenizer, "unk_token_id", None)
    if im_start is None or im_start == unk:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                            return_tensors="pt")
        return ids["input_ids"] if hasattr(ids, "keys") else ids
    pieces = []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "tool":
            pieces.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")
        elif role == "assistant":
            pieces.append(f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n")
        else:
            pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt: pieces.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return tokenizer("".join(pieces), add_special_tokens=False, return_tensors="pt").input_ids


def tool_continuation_ids(tokenizer, generated_ids, observation: str):
    """Append a tool observation without retokenizing or duplicating prior cache content."""
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ended = bool(generated_ids.shape[1]) and int(generated_ids[0, -1]) == im_end
    prefix = "\n" if ended else "<|im_end|>\n"
    text = (prefix + f"<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


@dataclass(frozen=True)
class InferenceBudget:
    max_tool_calls: int
    max_output_tokens: int
    max_observation_bytes: int


@dataclass(frozen=True)
class Condition:
    name: str
    memory_kind: str
    memory_path: str | None
    budget: InferenceBudget
    search_limits: dict
    model_revision: str = ""
    corpus_hash: str = ""
    tool_schema_hash: str = TOOL_SCHEMA_HASH
    decoding: dict | None = None
    root_prompt_revision: str = ROOT_PROMPT_REVISION


def assert_fair_conditions(conditions: list[Condition]) -> None:
    if not conditions:
        raise ValueError("at least one evaluation condition required")
    reference = conditions[0]
    for condition in conditions[1:]:
        if condition.budget != reference.budget:
            raise ValueError("evaluation inference budgets differ across conditions")
        if condition.search_limits != reference.search_limits:
            raise ValueError("evaluation tool limits differ across conditions")
        for field in ("model_revision", "corpus_hash", "tool_schema_hash", "decoding", "root_prompt_revision"):
            if getattr(condition, field) != getattr(reference, field):
                raise ValueError(f"evaluation {field} differs across conditions")


class PrefixAgentSession:
    """Maintains one prefix insertion while tool observations extend the same cache."""

    def __init__(self, prefix_lm, initial_ids):
        self.prefix_lm = prefix_lm
        self.state = prefix_lm.prefill_ids(initial_ids, use_cache=True)

    def append_tool_turn(self, token_ids) -> None:
        self.state = self.prefix_lm.append_ids(self.state, token_ids)
        if self.state.prefix_insertions != 1:
            raise RuntimeError("soft prefix was duplicated across a tool turn")


class FrozenRootAgent:
    """Minimal fixed-budget root loop shared by all memory conditions."""

    def __init__(self, model, tokenizer, tools: CodingTools, condition: Condition, *,
                 prefix_lm=None, map_text: str = "", max_new_tokens_per_turn: int = 128):
        self.model, self.tokenizer, self.tools, self.condition = model, tokenizer, tools, condition
        self.prefix_lm, self.map_text = prefix_lm, map_text
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        if condition.memory_kind == "latent" and prefix_lm is None:
            raise ValueError("latent condition requires prefix_lm")
        if condition.memory_kind == "map" and not map_text:
            raise ValueError("map condition requires frozen map_text")

    def _ids(self, messages):
        return serialize_chat_ids(self.tokenizer, messages, add_generation_prompt=True).to(self.model.device)

    def run(self, question: str) -> dict:
        import torch
        system = ROOT_SYSTEM_PROMPT + (("\n\nFrozen corpus memory map:\n" + self.map_text)
                                      if self.condition.memory_kind == "map" else "")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        tool_calls, output_tokens, observation_bytes, invalid = 0, 0, 0, 0
        transcript = list(messages); cached_ids = None; session = None
        if self.prefix_lm is not None:
            cached_ids = self.prefix_lm.chat_ids(messages, add_generation_prompt=True)
            session = PrefixAgentSession(self.prefix_lm, cached_ids)
        elif getattr(self.model.config, "model_type", "").startswith("qwen3_5"):
            cached_ids = self._ids(messages)
        while output_tokens < self.condition.budget.max_output_tokens:
            allowance = min(self.max_new_tokens_per_turn,
                            self.condition.budget.max_output_tokens - output_tokens)
            if session is not None:
                generated, session.state = self.prefix_lm.generate_from_state(session.state,
                                                                               max_new_tokens=allowance)
            else:
                ids = cached_ids if cached_ids is not None else self._ids(messages)
                if cached_ids is not None:
                    pieces = []
                    with torch.inference_mode():
                        history = ids
                        for _ in range(allowance):
                            logits = self.model(input_ids=history, use_cache=False, return_dict=True,
                                                logits_to_keep=1).logits[:, -1, :]
                            token = logits.argmax(dim=-1, keepdim=True); pieces.append(token)
                            history = torch.cat((history, token), dim=1)
                            if self.tokenizer.eos_token_id is not None and int(token.item()) == self.tokenizer.eos_token_id: break
                    generated = torch.cat(pieces, dim=1)
                else:
                    with torch.inference_mode():
                        full = self.model.generate(ids, max_new_tokens=allowance, do_sample=False,
                                                   pad_token_id=self.tokenizer.eos_token_id)
                    generated = full[:, ids.shape[1]:]
            output_tokens += int(generated.shape[1])
            text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            transcript.append({"role": "assistant", "content": text})
            try:
                final = next((json.loads(blob)["final"] for blob in _ACTION_RE.findall(text)
                              if isinstance(json.loads(blob), dict) and "final" in json.loads(blob)), None)
            except (json.JSONDecodeError, KeyError):
                final = None
            if final is not None:
                return self._result(str(final), transcript, tool_calls, output_tokens,
                                    observation_bytes, invalid, session)
            actions = parse_tool_actions(text)
            if not actions or tool_calls >= self.condition.budget.max_tool_calls:
                invalid += int(not actions)
                return self._result(text, transcript, tool_calls, output_tokens,
                                    observation_bytes, invalid, session)
            action = actions[0]; valid, _, observation = self.tools.execute(action)
            if not valid: invalid += 1
            remaining = self.condition.budget.max_observation_bytes - observation_bytes
            observation = CodingTools._clip_utf8(observation, remaining)
            observation_bytes += len(observation.encode("utf-8")); tool_calls += 1
            messages.extend([{"role": "assistant", "content": text},
                             {"role": "tool", "content": serialize_tool_observation(action, observation)}])
            transcript.append({"role": "tool", "content": serialize_tool_observation(action, observation)})
            if session is not None:
                suffix = tool_continuation_ids(self.tokenizer, generated,
                                               serialize_tool_observation(action, observation)).to(self.model.device)
                session.append_tool_turn(suffix)
                cached_ids = torch.cat((cached_ids, generated, suffix), dim=1)
            elif cached_ids is not None:
                suffix = tool_continuation_ids(self.tokenizer, generated,
                                               serialize_tool_observation(action, observation)).to(self.model.device)
                cached_ids = torch.cat((cached_ids, generated, suffix), dim=1)
        return self._result("", transcript, tool_calls, output_tokens, observation_bytes, invalid, session)

    @staticmethod
    def _result(answer, transcript, calls, output_tokens, observation_bytes, invalid, session):
        return {"answer": answer, "transcript": transcript, "tool_calls": calls,
                "model_output_tokens": output_tokens, "returned_observation_bytes": observation_bytes,
                "invalid_actions": invalid,
                "prefix_insertions": session.state.prefix_insertions if session is not None else 0}
