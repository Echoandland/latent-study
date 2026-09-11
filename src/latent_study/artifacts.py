from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .io import canonical_hash, sha256_bytes, write_json


ARTIFACT_SCHEMA_VERSION = 1
PROVENANCE_SUFFIX = ".provenance.json"
SOURCE_PATHS = ("src", "scripts", "configs", "schemas", "pyproject.toml", "requirements.lock")


class ArtifactCompatibilityError(RuntimeError):
    pass


def repository_commit(root: str | Path = ".") -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_tree_hash(root: str | Path = ".") -> str:
    root = Path(root).resolve()
    entries = []
    for item in SOURCE_PATHS:
        candidate = root / item
        paths = sorted(candidate.rglob("*")) if candidate.is_dir() else [candidate]
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts:
                entries.append({"path": path.relative_to(root).as_posix(),
                                "sha256": sha256_bytes(path.read_bytes())})
    return canonical_hash(entries)


def provenance(*, artifact_type: str, resolved_config_hash: str, model_id: str,
               model_revision: str, corpus_hash: str, tool_schema_hash: str,
               command: str, cli_overrides: dict, repository_root: str | Path = ".") -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "repository_commit": repository_commit(repository_root),
        "source_tree_hash": source_tree_hash(repository_root),
        "resolved_config_hash": resolved_config_hash,
        "model_id": model_id,
        "model_revision": model_revision,
        "corpus_hash": corpus_hash,
        "tool_schema_hash": tool_schema_hash,
        "command": command,
        "cli_overrides": cli_overrides,
    }


def sidecar_path(path: str | Path) -> Path:
    return Path(str(path) + PROVENANCE_SUFFIX)


def write_sidecar(path: str | Path, metadata: dict) -> Path:
    destination = sidecar_path(path)
    write_json(destination, metadata)
    return destination


def validate_provenance(metadata: dict, *, artifact_type: str | None = None,
                        resolved_config_hash: str | None = None,
                        model_id: str | None = None, model_revision: str | None = None,
                        corpus_hash: str | None = None,
                        tool_schema_hash: str | None = None,
                        require_current_source: bool = True,
                        repository_root: str | Path = ".") -> dict:
    required = {"artifact_schema_version", "artifact_type", "repository_commit",
                "source_tree_hash", "resolved_config_hash", "model_id", "model_revision",
                "corpus_hash", "tool_schema_hash", "command", "cli_overrides"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ArtifactCompatibilityError("stale artifact lacks provenance fields: " + ", ".join(missing))
    if metadata["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("unsupported artifact schema version")
    if require_current_source and metadata["source_tree_hash"] != source_tree_hash(repository_root):
        raise ArtifactCompatibilityError("artifact source-tree hash is stale")
    expected = {"artifact_type": artifact_type, "resolved_config_hash": resolved_config_hash,
                "model_id": model_id, "model_revision": model_revision,
                "corpus_hash": corpus_hash, "tool_schema_hash": tool_schema_hash}
    for field, value in expected.items():
        if value is not None and metadata.get(field) != value:
            raise ArtifactCompatibilityError(f"artifact {field} mismatch")
    return metadata


def read_sidecar(path: str | Path, **expected) -> dict:
    candidate = sidecar_path(path)
    if not candidate.is_file():
        raise ArtifactCompatibilityError(f"stale artifact has no provenance sidecar: {candidate}")
    return validate_provenance(json.loads(candidate.read_text(encoding="utf-8")), **expected)


def validate_json_artifact(path: str | Path, **expected) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload.get("provenance")
    if not isinstance(metadata, dict):
        raise ArtifactCompatibilityError("stale JSON artifact has no embedded provenance")
    validate_provenance(metadata, **expected)
    return payload
