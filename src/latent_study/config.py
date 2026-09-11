from __future__ import annotations

import copy
import json
from pathlib import Path

from .io import canonical_hash, write_json


REQUIRED_PATHS = (
    "model.id", "model.revision", "corpus.authorized_roots", "latent.length", "latent.init_std",
    "candidate_generation.n", "candidate_generation.do_sample", "candidate_generation.temperature",
    "candidate_generation.top_p", "candidate_generation.max_new_tokens", "tools.max_results",
    "tools.max_bytes_per_hit", "tools.max_total_bytes", "tools.max_query_chars",
    "objective.beta", "objective.preference_delta", "objective.rank_margin",
    "objective.query_weight", "objective.ranking_weight", "objective.gradient_balance",
    "replay.fraction", "seed",
    "training.optimizer", "training.learning_rate", "training.weight_decay",
    "training.steps_per_source", "training.batch_size", "training.gradient_accumulation_steps",
    "peek.max_new_tokens", "peek.retries", "decoding.do_sample", "decoding.temperature",
    "decoding.max_new_tokens_per_turn", "root_agent.max_tool_calls",
    "root_agent.max_output_tokens", "root_agent.max_observation_bytes",
)


def _get(config: dict, path: str):
    value = config
    for key in path.split("."): value = value[key]
    return value


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = []
    for key in REQUIRED_PATHS:
        try: _get(config, key)
        except KeyError: missing.append(key)
    if missing: raise ValueError("resolved experiment config is missing: " + ", ".join(missing))
    fixed = {"candidate_generation.source": "frozen_base_without_latent",
             "tools.interface": "pinned_local_coding_tools_v2",
             "training.optimizer": "AdamW",
             "objective.gradient_balance": "rms_norm_sum",
             "root_agent.prompt_revision": "local-coding-agent-v2"}
    incompatible = [key for key, expected in fixed.items() if _get(config, key) != expected]
    if incompatible:
        raise ValueError("unsupported behavior-changing configuration: " + ", ".join(incompatible))
    resolved = copy.deepcopy(config)
    resolved["config_source"] = str(Path(path).resolve())
    resolved["config_hash"] = canonical_hash(config)
    return resolved


def save_resolved(config: dict, artifact_path: str | Path) -> Path:
    destination = Path(str(artifact_path) + ".resolved-config.json")
    write_json(destination, config)
    return destination


def value(config: dict, path: str):
    return _get(config, path)
