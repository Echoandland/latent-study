from __future__ import annotations

import copy
import json
from pathlib import Path

from .io import canonical_hash, write_json


REQUIRED_PATHS = (
    "model.id", "model.revision", "corpus.authorized_roots", "latent.length", "latent.init_std",
    "candidate_generation.n", "tools.max_results", "tools.max_bytes_per_hit", "tools.max_total_bytes",
    "objective.beta", "objective.preference_delta", "objective.rank_margin",
    "objective.query_weight", "objective.ranking_weight", "replay.fraction", "seed",
    "training.steps_per_source", "training.batch_size", "training.gradient_accumulation_steps",
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
