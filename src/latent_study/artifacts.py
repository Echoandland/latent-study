from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .io import canonical_hash, sha256_bytes, write_json


ARTIFACT_SCHEMA_VERSION = 3
PROVENANCE_SUFFIX = ".provenance.json"
SOURCE_PATHS = ("src", "scripts", "configs", "schemas", "pyproject.toml", "requirements.lock")
INTEGRITY_FIELD = "artifact_sha256"
DEPENDENCY_FIELD = "dependencies"


class ArtifactCompatibilityError(RuntimeError):
    pass


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def repository_root(root: str | Path | None = None) -> Path:
    """Resolve the project root independently of the caller's working directory.

    The source checkout is the authoritative root when this package is imported
    from ``src/latent_study``.  An explicit root or ``LATENT_STUDY_REPO_ROOT``
    is still supported for installed/relocated deployments.
    """
    if root is not None:
        return Path(root).resolve()
    configured = os.environ.get("LATENT_STUDY_REPO_ROOT")
    if configured:
        return Path(configured).resolve()
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "src").is_dir() and (package_root / "pyproject.toml").is_file():
        return package_root
    try:
        return Path(subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        return package_root


def repository_commit(root: str | Path | None = None) -> str:
    root = repository_root(root)
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_tree_hash(root: str | Path | None = None) -> str:
    root = repository_root(root)
    entries = []
    for item in SOURCE_PATHS:
        candidate = root / item
        paths = sorted(candidate.rglob("*")) if candidate.is_dir() else [candidate]
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts:
                entries.append({"path": path.relative_to(root).as_posix(),
                                "sha256": sha256_bytes(path.read_bytes())})
    return canonical_hash(entries)


def _hashable(value: Any, *, in_provenance: bool = False) -> Any:
    """Convert JSON/tensor payloads to a deterministic hash representation."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            key = str(key)
            # The integrity field is deliberately excluded from the digest so
            # storing the digest inside JSON/Torch metadata is not recursive.
            if in_provenance and key == INTEGRITY_FIELD:
                continue
            converted[key] = _hashable(item, in_provenance=(key == "provenance"))
        return converted
    if isinstance(value, (list, tuple)):
        return [_hashable(item, in_provenance=in_provenance) for item in value]
    if isinstance(value, bytes):
        return {"__bytes_sha256__": sha256_bytes(value), "length": len(value)}
    # Avoid importing torch during ordinary CLI startup.  Tensor payloads are
    # represented by dtype/shape and a byte-level digest, so prefix edits are
    # detected without putting a second model-sized tensor in memory.
    try:
        import torch
        if isinstance(value, torch.Tensor):
            raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            return {"__tensor_sha256__": sha256_bytes(raw),
                    "dtype": str(value.dtype), "shape": list(value.shape)}
    except (ImportError, RuntimeError, TypeError):
        pass
    return value


def artifact_payload_hash(payload: Any) -> str:
    """SHA256 of canonical artifact content, excluding only its own digest."""
    return canonical_hash(_hashable(payload))


def artifact_content_hash(path: str | Path) -> str:
    """Return the content hash used by artifact provenance.

    JSON and JSONL are hashed as canonical parsed content (so formatting is not
    mistaken for a semantic change); Torch checkpoints are hashed from all
    tensor and metadata payload content.  Other files use their raw-byte SHA256.
    This is intentionally distinct from ``resolved_config_hash``.
    """
    path = Path(path)
    if not path.exists():
        raise ArtifactCompatibilityError(f"artifact does not exist: {path}")
    if path.is_dir():
        entries = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and "__pycache__" not in child.parts:
                entries.append({"path": child.relative_to(path).as_posix(),
                                "sha256": sha256_bytes(child.read_bytes())})
        return artifact_payload_hash(entries)
    try:
        if path.suffix == ".json":
            return artifact_payload_hash(json.loads(path.read_text(encoding="utf-8")))
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            return artifact_payload_hash(rows)
        if path.suffix == ".pt":
            import torch
            return artifact_payload_hash(torch.load(path, map_location="cpu", weights_only=False))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ImportError, RuntimeError, ValueError) as exc:
        raise ArtifactCompatibilityError(f"cannot hash artifact content: {path}: {exc}") from exc
    return sha256_bytes(path.read_bytes())


_CORPUS_DEPENDENCY_NAMES = {"corpus", "corpus_snapshot", "live_corpus"}
_EVALUATION_DEPENDENCY_NAMES = {"evaluation", "evaluation_dataset", "evaluation_snapshot"}


def _dependency_hash(path: Path, hash_kind: str) -> str:
    if hash_kind == "corpus_snapshot":
        from .corpus import corpus_snapshot_hash
        return corpus_snapshot_hash(path)
    if hash_kind == "evaluation_dataset_snapshot":
        from .evaluation import evaluation_dataset_snapshot_hash
        return evaluation_dataset_snapshot_hash(path)
    if hash_kind != "artifact_content":
        raise ArtifactCompatibilityError(f"unknown dependency hash kind: {hash_kind}")
    return artifact_content_hash(path)


def _embedded_provenance(path: Path) -> dict | None:
    """Read dependency identity without treating its local path as identity."""
    try:
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".jsonl":
            candidate = sidecar_path(path)
            payload = json.loads(candidate.read_text(encoding="utf-8")) if candidate.is_file() else None
        elif path.suffix == ".pt":
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=True)
        else:
            payload = None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ImportError, RuntimeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else payload
    return metadata if isinstance(metadata, dict) and metadata.get("artifact_type") else None


def _portable_path(path: Path, artifact_path: str | Path | None) -> tuple[str, str]:
    if artifact_path is not None:
        base = Path(artifact_path).resolve().parent
        return Path(os.path.relpath(path, base)).as_posix(), "artifact"
    root = repository_root()
    try:
        return path.relative_to(root).as_posix(), "repository"
    except ValueError:
        # Callers that create artifacts outside a checkout should pass
        # artifact_path.  The adjacent-file fallback keeps old library-level
        # fixtures portable rather than embedding a machine-specific path.
        return path.name, "artifact"


def dependency_descriptor(path: str | Path, *, name: str | None = None,
                          artifact_path: str | Path | None = None,
                          hash_kind: str | None = None) -> dict:
    path = Path(path).resolve()
    role = str(name or path.name)
    if hash_kind is None:
        if role in _CORPUS_DEPENDENCY_NAMES and path.is_dir():
            hash_kind = "corpus_snapshot"
        elif role in _EVALUATION_DEPENDENCY_NAMES:
            hash_kind = "evaluation_dataset_snapshot"
        else:
            hash_kind = "artifact_content"
    relative, path_base = _portable_path(path, artifact_path)
    metadata = _embedded_provenance(path)
    identity_view = None
    artifact_type = ("corpus_snapshot" if hash_kind == "corpus_snapshot" else
                     "evaluation_dataset_snapshot" if hash_kind == "evaluation_dataset_snapshot" else
                     metadata.get("artifact_type") if metadata else
                     "directory" if path.is_dir() else path.suffix.lstrip(".") or "file")
    if metadata:
        identity_view = {
            key: metadata.get(key) for key in (
                "artifact_schema_version", "artifact_type", INTEGRITY_FIELD,
                "protocol_config_hash", "corpus_hash", "model_id", "model_revision",
                "tokenizer_id", "tokenizer_revision", "tool_schema_hash")
        }
    return {"name": role, "path": relative, "path_base": path_base,
            "hash_kind": hash_kind, "artifact_type": artifact_type,
            "provenance_identity": canonical_hash(identity_view) if identity_view else None,
            INTEGRITY_FIELD: _dependency_hash(path, hash_kind)}


def resolve_dependency_path(dependency: dict, *, artifact_path: str | Path | None = None,
                            project_root: str | Path | None = None) -> Path:
    relative = Path(str(dependency.get("path", "")))
    if relative.is_absolute():
        raise ArtifactCompatibilityError("absolute dependency paths are not portable")
    path_base = dependency.get("path_base")
    if path_base == "repository":
        return (repository_root(project_root) / relative).resolve()
    if path_base == "artifact" and artifact_path is not None:
        return (Path(artifact_path).resolve().parent / relative).resolve()
    raise ArtifactCompatibilityError("dependency path cannot be resolved without its declared portable base")


def _normalise_dependencies(dependencies: Iterable[Any] | None, *,
                            artifact_path: str | Path | None = None) -> list[dict]:
    out = []
    for dependency in dependencies or ():
        if isinstance(dependency, dict):
            item = dict(dependency)
            path = item.get("path")
            if not path:
                raise ArtifactCompatibilityError("dependency descriptor lacks path")
            if not {"path_base", "hash_kind", "artifact_type", INTEGRITY_FIELD} <= set(item):
                item = dependency_descriptor(path, name=item.get("name"),
                                             artifact_path=artifact_path,
                                             hash_kind=item.get("hash_kind"))
            else:
                item.setdefault("name", Path(path).name)
                item.setdefault("provenance_identity", None)
                item["path"] = Path(str(path)).as_posix()
        elif isinstance(dependency, (tuple, list)) and len(dependency) == 2:
            item = dependency_descriptor(dependency[1], name=str(dependency[0]),
                                         artifact_path=artifact_path)
        else:
            item = dependency_descriptor(dependency, artifact_path=artifact_path)
        out.append(item)
    return sorted(out, key=lambda item: (str(item.get("name", "")), str(item["path"])))


def provenance(*, artifact_type: str, resolved_config_hash: str, protocol_config_hash: str,
               model_id: str,
               model_revision: str, corpus_hash: str, tool_schema_hash: str,
               command: str, cli_overrides: dict, repository_root: str | Path | None = None,
               dependencies: Iterable[Any] | None = None,
               tokenizer_id: str | None = None, tokenizer_revision: str | None = None,
               artifact_path: str | Path | None = None) -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "repository_commit": repository_commit(repository_root),
        "source_tree_hash": source_tree_hash(repository_root),
        INTEGRITY_FIELD: None,
        DEPENDENCY_FIELD: _normalise_dependencies(dependencies, artifact_path=artifact_path),
        "resolved_config_hash": resolved_config_hash,
        "protocol_config_hash": protocol_config_hash,
        "model_id": model_id,
        "model_revision": model_revision,
        "corpus_hash": corpus_hash,
        "tool_schema_hash": tool_schema_hash,
        "command": command,
        "cli_overrides": cli_overrides,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
    }


def sidecar_path(path: str | Path) -> Path:
    return Path(str(path) + PROVENANCE_SUFFIX)


def write_sidecar(path: str | Path, metadata: dict, *, dependencies: Iterable[Any] | None = None) -> Path:
    metadata = dict(metadata)
    if dependencies is not None:
        metadata[DEPENDENCY_FIELD] = _normalise_dependencies(dependencies, artifact_path=path)
    elif DEPENDENCY_FIELD in metadata:
        metadata[DEPENDENCY_FIELD] = _normalise_dependencies(metadata[DEPENDENCY_FIELD], artifact_path=path)
    metadata[INTEGRITY_FIELD] = artifact_content_hash(path)
    destination = sidecar_path(path)
    write_json(destination, metadata)
    return destination


def validate_provenance(metadata: dict, *, artifact_type: str | None = None,
                        resolved_config_hash: str | None = None,
                        protocol_config_hash: str | None = None,
                        model_id: str | None = None, model_revision: str | None = None,
                        corpus_hash: str | None = None,
                        tool_schema_hash: str | None = None,
                        tokenizer_id: str | None = None,
                        tokenizer_revision: str | None = None,
                        artifact_path: str | Path | None = None,
                        dependencies: Iterable[Any] | None = None,
                        skip_dependency_names: Iterable[str] | None = None,
                        require_current_source: bool = True,
                        repository_root: str | Path | None = None) -> dict:
    if not isinstance(metadata, dict):
        raise ArtifactCompatibilityError("artifact provenance must be an object")
    required = {"artifact_schema_version", "artifact_type", "repository_commit",
                "source_tree_hash", "resolved_config_hash", "protocol_config_hash",
                "model_id", "model_revision",
                "corpus_hash", "tool_schema_hash", "command", "cli_overrides",
                INTEGRITY_FIELD, DEPENDENCY_FIELD}
    missing = sorted(required - set(metadata))
    if missing:
        raise ArtifactCompatibilityError("stale artifact lacks provenance fields: " + ", ".join(missing))
    if metadata["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("unsupported artifact schema version")
    string_fields = ("artifact_type", "repository_commit", "source_tree_hash",
                     "resolved_config_hash", "protocol_config_hash", "model_id", "model_revision",
                     "corpus_hash", "tool_schema_hash", "command")
    if any(not isinstance(metadata.get(field), str) or not metadata[field].strip()
           for field in string_fields):
        raise ArtifactCompatibilityError("artifact provenance has an empty or non-string identity field")
    if not _SHA256_RE.fullmatch(metadata["protocol_config_hash"]):
        raise ArtifactCompatibilityError("artifact protocol_config_hash is not a SHA256")
    if not isinstance(metadata.get("cli_overrides"), dict):
        raise ArtifactCompatibilityError("artifact cli_overrides must be an object")
    if (not isinstance(metadata[INTEGRITY_FIELD], str)
            or not _SHA256_RE.fullmatch(metadata[INTEGRITY_FIELD])):
        raise ArtifactCompatibilityError("artifact lacks a valid content SHA256")
    if not isinstance(metadata[DEPENDENCY_FIELD], list):
        raise ArtifactCompatibilityError("artifact dependencies must be a list")
    for dependency in metadata[DEPENDENCY_FIELD]:
        if (not isinstance(dependency, dict)
                or not isinstance(dependency.get("name"), str) or not dependency["name"].strip()
                or not isinstance(dependency.get("path"), str) or not dependency["path"].strip()
                or Path(dependency.get("path", "")).is_absolute()
                or dependency.get("path_base") not in {"artifact", "repository"}
                or dependency.get("hash_kind") not in {
                    "artifact_content", "corpus_snapshot", "evaluation_dataset_snapshot"}
                or not isinstance(dependency.get("artifact_type"), str)
                or (dependency.get("provenance_identity") is not None
                    and (not isinstance(dependency.get("provenance_identity"), str)
                         or not _SHA256_RE.fullmatch(dependency["provenance_identity"])))
                or not isinstance(dependency.get(INTEGRITY_FIELD), str)
                or not _SHA256_RE.fullmatch(dependency[INTEGRITY_FIELD])):
            raise ArtifactCompatibilityError("artifact dependency descriptor is malformed")
    for field in ("tokenizer_id", "tokenizer_revision"):
        if field in metadata and metadata[field] is not None and not isinstance(metadata[field], str):
            raise ArtifactCompatibilityError(f"artifact {field} must be a string or null")
    if require_current_source and metadata["source_tree_hash"] != source_tree_hash(repository_root):
        raise ArtifactCompatibilityError("artifact source-tree hash is stale")
    expected = {"artifact_type": artifact_type, "resolved_config_hash": resolved_config_hash,
                "protocol_config_hash": protocol_config_hash,
                "model_id": model_id, "model_revision": model_revision,
                "corpus_hash": corpus_hash, "tool_schema_hash": tool_schema_hash,
                "tokenizer_id": tokenizer_id, "tokenizer_revision": tokenizer_revision}
    for field, value in expected.items():
        if value is not None and metadata.get(field) != value:
            raise ArtifactCompatibilityError(f"artifact {field} mismatch")
    if artifact_path is not None:
        actual = artifact_content_hash(artifact_path)
        if metadata[INTEGRITY_FIELD] != actual:
            raise ArtifactCompatibilityError("artifact content SHA256 mismatch")
    expected_dependencies = _normalise_dependencies(dependencies, artifact_path=artifact_path)
    if expected_dependencies:
        actual_dependencies = _normalise_dependencies(metadata[DEPENDENCY_FIELD])
        if actual_dependencies != expected_dependencies:
            raise ArtifactCompatibilityError("artifact dependency hash mismatch")
    skipped_dependencies = {str(name) for name in (skip_dependency_names or ())}
    for dependency in metadata[DEPENDENCY_FIELD]:
        if dependency.get("name") in skipped_dependencies:
            continue
        path = resolve_dependency_path(dependency, artifact_path=artifact_path,
                                       project_root=repository_root)
        if not path.exists():
            raise ArtifactCompatibilityError(f"artifact dependency is missing: {path}")
        if dependency.get(INTEGRITY_FIELD) != _dependency_hash(path, dependency["hash_kind"]):
            raise ArtifactCompatibilityError(f"artifact dependency content SHA256 mismatch: {path}")
        current = dependency_descriptor(path, name=dependency["name"], artifact_path=artifact_path,
                                        hash_kind=dependency["hash_kind"])
        if dependency.get("artifact_type") != current.get("artifact_type"):
            raise ArtifactCompatibilityError(f"artifact dependency type mismatch: {path}")
        if (dependency.get("provenance_identity") is not None
                and dependency.get("provenance_identity") != current.get("provenance_identity")):
            raise ArtifactCompatibilityError(f"artifact dependency provenance identity mismatch: {path}")
    return metadata


def read_sidecar(path: str | Path, *, require_current: bool | None = None,
                 strict: bool | None = None, **expected) -> dict:
    candidate = sidecar_path(path)
    if not candidate.is_file():
        raise ArtifactCompatibilityError(f"stale artifact has no provenance sidecar: {candidate}")
    if require_current is not None:
        expected.setdefault("require_current_source", bool(require_current))
    if strict is not None:
        expected.setdefault("require_current_source", bool(strict))
    expected.setdefault("artifact_path", path)
    try:
        metadata = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"stale artifact provenance is unreadable: {candidate}") from exc
    return validate_provenance(metadata, **expected)


def validate_json_artifact(path: str | Path, *, require_current: bool | None = None,
                           strict: bool | None = None, **expected) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"stale JSON artifact is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactCompatibilityError("stale JSON artifact must be an object")
    if "artifact_schema_version" in payload \
            and payload["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("unsupported artifact schema version")
    metadata = payload.get("provenance")
    if not isinstance(metadata, dict):
        raise ArtifactCompatibilityError("stale JSON artifact has no embedded provenance")
    if require_current is not None:
        expected.setdefault("require_current_source", bool(require_current))
    if strict is not None:
        expected.setdefault("require_current_source", bool(strict))
    expected.setdefault("artifact_path", path)
    validate_provenance(metadata, **expected)
    return payload
