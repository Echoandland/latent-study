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
    "objective.relevant_label", "objective.irrelevant_label",
    "replay.fraction", "seed",
    "training.optimizer", "training.learning_rate", "training.weight_decay",
    "training.steps_per_source", "training.batch_size", "training.gradient_accumulation_steps",
    "peek.max_new_tokens", "peek.retries", "decoding.do_sample", "decoding.temperature",
    "decoding.max_new_tokens_per_turn", "root_agent.max_tool_calls",
    "root_agent.max_output_tokens", "root_agent.max_observation_bytes",
    "peek_internal.max_new_tokens", "peek_internal.retry_max_new_tokens",
    "evaluation.conditions", "evaluation.default_budget", "evaluation.budgets",
)


PROTOCOL_CONFIG_SCHEMA_VERSION = 2


# CLI values that alter a scientific/protocol setting are applied to the
# effective configuration before either configuration hash is computed. CLI
# fields absent here are command-local/operational and remain recorded by the
# command hash and cli_overrides only.
PROTOCOL_CLI_OVERRIDE_PATHS = {
    "generate-records": {
        "seed": "seed", "actions": "candidate_generation.n",
        "candidate_revision": "model.revision",
        "candidate_dtype": "candidate_generation.dtype",
        "max_results": "tools.max_results",
        "max_bytes_per_hit": "tools.max_bytes_per_hit",
        "max_total_bytes": "tools.max_total_bytes",
    },
    "peek-study": {
        "model_revision": "model.revision", "seed": "seed",
        "internal_max_new_tokens": "peek_internal.max_new_tokens",
        "retry_max_new_tokens": "peek_internal.retry_max_new_tokens",
        "retries": "peek.retries", "replay": "replay.fraction",
        "batch_size": "training.batch_size",
        "updates_per_source": "training.steps_per_source",
    },
    "train-latent": {
        "model_revision": "model.revision", "length": "latent.length",
        "dtype": "model.dtype", "learning_rate": "training.learning_rate",
        "query_weight": "objective.query_weight",
        "lambda_rank": "objective.ranking_weight",
        "delta": "objective.preference_delta", "replay": "replay.fraction",
        "seed": "seed", "batch_size": "training.batch_size",
        "updates_per_source": "training.steps_per_source",
        "relevant_label": "objective.relevant_label",
        "irrelevant_label": "objective.irrelevant_label",
    },
    "init-latent": {
        "model_revision": "model.revision", "length": "latent.length",
        "dtype": "model.dtype", "seed": "seed",
    },
    "replay-report": {
        "replay": "replay.fraction", "seed": "seed",
        "batch_size": "training.batch_size",
        "updates_per_source": "training.steps_per_source",
    },
    "smoke-eval": {
        "max_results": "tools.max_results",
        "max_bytes_per_hit": "tools.max_bytes_per_hit",
        "max_total_bytes": "tools.max_total_bytes",
    },
    "root-agent-smoke": {"model_revision": "model.revision"},
    "evaluate": {"model_revision": "model.revision"},
}


def _set(config: dict, path: str, value) -> None:
    target = config
    pieces = path.split(".")
    for key in pieces[:-1]:
        target = target.setdefault(key, {})
    target[pieces[-1]] = value


def effective_config(base_config: dict, command: str, cli_values: dict) -> dict:
    """Apply protocol-relevant CLI overrides to a clean base configuration."""
    effective = copy.deepcopy(base_config)
    for argument, path in PROTOCOL_CLI_OVERRIDE_PATHS.get(command, {}).items():
        value = cli_values.get(argument)
        if value is not None:
            _set(effective, path, value)
    return effective


def protocol_config_view(config: dict) -> dict:
    """Return the canonical cross-command scientific compatibility view.

    This projection deliberately excludes filesystem locations, output paths,
    worker assignment, batch sizing, retry counts, and other command-local
    execution controls.  Those remain bound by ``resolved_config_hash`` and
    artifact dependencies, but do not make an otherwise identical experiment
    incompatible merely because it was run from another checkout or with a
    different sharding arrangement.
    """
    corpus = config.get("corpus", {})
    model = config.get("model", {})
    latent = config.get("latent", {})
    tools = config.get("tools", {})
    candidate = config.get("candidate_generation", {})
    reward = config.get("reward", {})
    objective = config.get("objective", {})
    decoding = config.get("decoding", {})
    root_agent = config.get("root_agent", {})
    evaluation = config.get("evaluation", {})
    return {
        "schema_version": PROTOCOL_CONFIG_SCHEMA_VERSION,
        "protocol": config.get("protocol"),
        "model": {key: model.get(key) for key in ("id", "revision", "frozen", "dtype")},
        "corpus": {"revision": corpus.get("revision")},
        "memory": {
            "placement": "after_system_and_tool_instructions_before_user_history",
            "serializer_revision": "shared_memory_slot_v1",
            "latent_length": latent.get("length"),
            "latent_init_std": latent.get("init_std"),
        },
        "candidate_generation": {
            key: candidate.get(key)
            for key in ("n", "source", "do_sample", "temperature", "top_p", "max_new_tokens", "dtype")
        },
        "tools": {
            key: tools.get(key)
            for key in ("interface", "max_results", "max_bytes_per_hit", "max_total_bytes",
                        "grep_context_lines", "max_query_chars")
        },
        "reward": reward,
        "objective": {
            key: objective.get(key)
            for key in ("beta", "preference_delta", "rank_margin", "query_weight",
                        "ranking_weight", "gradient_balance", "relevant_label",
                        "irrelevant_label")
        },
        "replay": {
            key: config.get("replay", {}).get(key)
            for key in ("fraction", "source_stratified", "keep_full_bank")
        },
        "training": {
            key: config.get("training", {}).get(key)
            for key in ("optimizer", "learning_rate", "weight_decay", "steps_per_source",
                        "batch_size", "gradient_accumulation_steps")
        },
        "peek": config.get("peek", {}),
        "peek_internal": config.get("peek_internal", {}),
        "seed": config.get("seed"),
        "root_agent": {
            "prompt_revision": root_agent.get("prompt_revision"),
        },
        "decoding": {
            key: decoding.get(key)
            for key in ("do_sample", "temperature", "max_new_tokens_per_turn")
        },
        "evaluation": {
            "conditions": evaluation.get("conditions"),
            "memory_frozen": evaluation.get("memory_frozen"),
            "same_tools_and_budgets": evaluation.get("same_tools_and_budgets"),
            "budgets": evaluation.get("budgets"),
            "expertise": evaluation.get("expertise"),
        },
    }


def protocol_config_hash(config: dict) -> str:
    """Hash only the centralized cross-command compatibility projection."""
    return canonical_hash(protocol_config_view(config))


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
    resolved["protocol_config_hash"] = protocol_config_hash(config)
    resolved["resolved_config_hash"] = canonical_hash(config)
    # Compatibility alias for callers predating the explicit distinction.
    resolved["config_hash"] = resolved["resolved_config_hash"]
    return resolved


def save_resolved(config: dict, artifact_path: str | Path) -> Path:
    destination = Path(str(artifact_path) + ".resolved-config.json")
    write_json(destination, config)
    return destination


def value(config: dict, path: str):
    return _get(config, path)
