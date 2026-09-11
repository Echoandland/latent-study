from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = jsonable(value)
    # Artifact payloads embed their provenance.  Compute the content digest in
    # two passes with only the self-referential field blanked; ordinary JSON
    # helpers remain unchanged for non-artifact metadata.
    if isinstance(payload, dict) and isinstance(payload.get("provenance"), dict):
        from .artifacts import artifact_payload_hash
        metadata = dict(payload["provenance"])
        metadata["artifact_sha256"] = None
        payload["provenance"] = metadata
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        metadata["artifact_sha256"] = artifact_payload_hash(payload)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True, ensure_ascii=False) + "\n")
