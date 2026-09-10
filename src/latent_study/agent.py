from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .schema import ToolAction


_ACTION_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_tool_actions(text: str) -> list[ToolAction]:
    """Parse every visible search JSON object in a multi-turn response."""
    actions = []
    for blob in _ACTION_RE.findall(text):
        try:
            value = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if value.get("tool") != "search" or not isinstance(value.get("query"), str):
            continue
        maximum = value.get("max_results", 5)
        if not isinstance(maximum, int):
            continue
        actions.append(ToolAction(value["query"], maximum))
    return actions


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


def assert_fair_conditions(conditions: list[Condition]) -> None:
    if not conditions:
        raise ValueError("at least one evaluation condition required")
    reference = conditions[0]
    for condition in conditions[1:]:
        if condition.budget != reference.budget:
            raise ValueError("evaluation inference budgets differ across conditions")
        if condition.search_limits != reference.search_limits:
            raise ValueError("evaluation tool limits differ across conditions")


class PrefixAgentSession:
    """Maintains one prefix insertion while tool observations extend the same cache."""

    def __init__(self, prefix_lm, initial_ids):
        self.prefix_lm = prefix_lm
        self.state = prefix_lm.prefill_ids(initial_ids, use_cache=True)

    def append_tool_turn(self, token_ids) -> None:
        self.state = self.prefix_lm.append_ids(self.state, token_ids)
        if self.state.prefix_insertions != 1:
            raise RuntimeError("soft prefix was duplicated across a tool turn")

