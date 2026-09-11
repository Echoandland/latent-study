from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path


class IsolationError(RuntimeError):
    pass


def resolved_under(path: str | Path, root: str | Path) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def enforce_study_inputs(
    inputs: list[str | Path], *, corpus_root: str | Path, evaluation_root: str | Path
) -> None:
    """Fail closed if study can see evaluation data or anything outside its corpus."""
    for item in inputs:
        path = Path(item).resolve()
        if resolved_under(path, evaluation_root):
            raise IsolationError(f"evaluation-derived input forbidden during study: {path}")
        if not resolved_under(path, corpus_root):
            raise IsolationError(f"study input is outside authorized corpus root: {path}")


def assert_frozen_artifact(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".pt":
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("provenance") if isinstance(payload, dict) else None
    if isinstance(metadata, dict) and metadata.get("artifact_sha256"):
        from .artifacts import validate_provenance
        validate_provenance(metadata, artifact_path=path)
    return assert_frozen_payload(payload)


def assert_frozen_payload(payload: dict) -> dict:
    from .artifacts import ARTIFACT_SCHEMA_VERSION
    if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise IsolationError("evaluation requires the current artifact schema version")
    if payload.get("phase") != "frozen_before_evaluation":
        raise IsolationError("evaluation requires a pre-frozen study artifact")
    if payload.get("evaluation_inputs_seen"):
        raise IsolationError("artifact metadata reports evaluation contamination")
    audit = payload.get("contamination_audit") or payload.get("provenance", {}).get("contamination_audit")
    if (not isinstance(audit, dict) or audit.get("status") != "pass"
            or not isinstance(audit.get("artifact_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", audit["artifact_sha256"])
            or not isinstance(audit.get("evaluation_dataset_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", audit["evaluation_dataset_sha256"])):
        raise IsolationError("frozen artifact lacks a passing evaluation-contamination audit")
    metadata = payload.get("provenance")
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("artifact_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", metadata["artifact_sha256"])):
        raise IsolationError("frozen artifact lacks a cryptographic provenance digest")
    return payload


def validate_study_bank(records, corpus_hash: str) -> None:
    allowed = {"atomic_structural_definition", "uniquely_resolved_ast_call",
               "ast_assignment_or_import", "single_verified_name_substitution"}
    for record in records:
        if record.corpus_hash != corpus_hash:
            raise IsolationError(f"record {record.record_id} has a foreign corpus hash")
        if record.validation.get("method") not in allowed:
            raise IsolationError(f"record {record.record_id} lacks an allowed corpus-only validator")
        forbidden = {"evaluation_question", "gold_answer", "rubric", "evaluation_reward"}
        if forbidden & set(record.validation):
            raise IsolationError(f"record {record.record_id} contains evaluation-derived fields")


_SKIP_KEYS = {"provenance", "dependencies", "artifact_sha256", "source_tree_hash",
              "corpus_hash", "resolved_config_hash", "protocol_config_hash",
              "tool_schema_hash", "evaluation_dataset_snapshot"}
_CONTENT_KEYS = {"question", "prompt", "answer", "rubric", "content", "response",
                 "reference", "gold", "trajectory"}


def _normalise_fingerprint(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s./:-]", " ", value)).strip()


def _fingerprint(value: str) -> str:
    return hashlib.sha256(_normalise_fingerprint(value).encode("utf-8")).hexdigest()


def _walk_material(value, *, key: str = "", location: str = ""):
    if isinstance(value, dict):
        for name, item in value.items():
            if name in _SKIP_KEYS:
                continue
            yield from _walk_material(item, key=str(name).casefold(),
                                      location=f"{location}.{name}" if location else str(name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_material(item, key=key,
                                      location=f"{location}[{index}]")
    elif isinstance(value, str) and value.strip():
        normalized = _normalise_fingerprint(value)
        if not normalized:
            return
        category = "id" if key.endswith("id") or key in {"id", "example_id", "question_id"} else "content"
        if key not in _CONTENT_KEYS and category != "id" and len(normalized.split()) < 3:
            return
        yield {"category": category, "value": normalized, "fingerprint": _fingerprint(value),
               "location": location}


def _read_material(path: Path) -> list[dict]:
    items = []
    candidates = sorted(path.rglob("*") if path.is_dir() else [path])
    for candidate in candidates:
        if not candidate.is_file() or candidate.name.startswith("."):
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if candidate.suffix == ".jsonl":
            values = []
            for line in text.splitlines():
                try: values.append(json.loads(line))
                except json.JSONDecodeError: values.append(line)
            items.extend(_walk_material(values, location=str(candidate)))
        elif candidate.suffix == ".json":
            try: value = json.loads(text)
            except json.JSONDecodeError: value = text
            items.extend(_walk_material(value, location=str(candidate)))
        else:
            normalized = _normalise_fingerprint(text)
            if normalized:
                items.append({"category": "content", "value": normalized,
                              "fingerprint": _fingerprint(text), "location": str(candidate)})
            for match in re.findall(r"\b(?:eval|example|question)[-_][A-Za-z0-9_.:-]+\b", text,
                                    flags=re.IGNORECASE):
                items.append({"category": "id", "value": _normalise_fingerprint(match),
                              "fingerprint": _fingerprint(match), "location": str(candidate)})
    return items


def audit_evaluation_contamination(study_paths, evaluation_paths, *, corpus_paths=()) -> dict:
    """Compare study-side fingerprints against evaluation-only material.

    This audit is intentionally a separate command/phase.  Training code does
    not receive evaluation rows; it only consumes the resulting pass attestation
    when an artifact is later frozen.
    """
    study_paths = list(study_paths)
    evaluation_paths = list(evaluation_paths)
    corpus_paths = list(corpus_paths)
    study_items = []
    for path in study_paths:
        study_items.extend(_read_material(Path(path).resolve()))
    for path in corpus_paths:
        study_items.extend(_read_material(Path(path).resolve()))
    evaluation_items = []
    for path in evaluation_paths:
        evaluation_items.extend(_read_material(Path(path).resolve()))
    eval_ids = {item["value"]: item for item in evaluation_items if item["category"] == "id"}
    eval_content = {item["value"]: item for item in evaluation_items if item["category"] == "content"
                    and len(item["value"]) >= 12}
    eval_shingles = {}
    for value in eval_content:
        words = value.split()
        for start in range(max(0, len(words) - 5)):
            shingle = " ".join(words[start:start + 6])
            if len(shingle) >= 32:
                eval_shingles[shingle] = value
    collisions = []
    seen = set()
    for item in study_items:
        collision_type, evaluation = None, None
        if item["category"] == "id" and item["value"] in eval_ids:
            collision_type, evaluation = "evaluation_id", eval_ids[item["value"]]
        elif item["category"] == "content" and item["value"] in eval_content:
            collision_type, evaluation = "evaluation_content", eval_content[item["value"]]
        elif item["category"] == "content":
            words = item["value"].split()
            for start in range(max(0, len(words) - 5)):
                shingle = " ".join(words[start:start + 6])
                if shingle in eval_shingles:
                    collision_type, evaluation = "evaluation_content_shingle", {
                        "value": eval_shingles[shingle]}
                    break
        if collision_type:
            key = (collision_type, item["location"], evaluation.get("value", ""))
            if key in seen:
                continue
            seen.add(key)
            collisions.append({"type": collision_type, "study_location": item["location"],
                               "evaluation_location": evaluation.get("location"),
                               "evaluation_fingerprint": _fingerprint(evaluation.get("value", ""))})
    from .evaluation import evaluation_dataset_snapshot
    return {"status": "pass" if not collisions else "fail", "collision_count": len(collisions),
            "collisions": collisions, "algorithm": "normalized_exact_and_six_token_fingerprints",
            "evaluation_dataset_snapshot": evaluation_dataset_snapshot(evaluation_paths)}


def assert_contamination_free(audit: dict) -> dict:
    if not isinstance(audit, dict) or audit.get("status") != "pass":
        raise IsolationError("evaluation-contamination audit failed")
    return audit


def contamination_attestation(audit: dict) -> dict:
    """Minimal immutable attestation carried by a frozen memory artifact."""
    assert_contamination_free(audit)
    digest = audit.get("provenance", {}).get("artifact_sha256")
    dataset_digest = audit.get("evaluation_dataset_snapshot", {}).get("sha256")
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(dataset_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", dataset_digest)):
        raise IsolationError("contamination audit lacks a cryptographic dataset binding")
    return {"status": "pass", "artifact_sha256": digest,
            "evaluation_dataset_sha256": dataset_digest}


def verify_memory_contamination_binding(payload: dict, artifact_path: str | Path, *,
                                        evaluation_dataset_sha256: str,
                                        protocol_config_hash: str) -> dict:
    """Resolve and verify memory -> audit -> current evaluation snapshot."""
    from .artifacts import (resolve_dependency_path, validate_json_artifact)
    assert_frozen_payload(payload)
    metadata = payload.get("provenance", {})
    dependencies = [item for item in metadata.get("dependencies", ())
                    if item.get("name") == "contamination_audit"]
    if len(dependencies) != 1:
        raise IsolationError("memory artifact must identify exactly one contamination audit")
    audit_path = resolve_dependency_path(dependencies[0], artifact_path=artifact_path)
    try:
        audit = validate_json_artifact(
            audit_path, artifact_type="evaluation_contamination_audit",
            protocol_config_hash=protocol_config_hash)
    except Exception as exc:
        raise IsolationError(f"contamination audit dependency is invalid: {exc}") from exc
    assert_contamination_free(audit)
    audit_digest = audit.get("provenance", {}).get("artifact_sha256")
    audit_dataset = audit.get("evaluation_dataset_snapshot", {}).get("sha256")
    attestation = payload.get("contamination_audit") or metadata.get("contamination_audit")
    if audit_digest != dependencies[0].get("artifact_sha256"):
        raise IsolationError("memory contamination-audit dependency digest mismatch")
    if not isinstance(attestation, dict) or attestation.get("artifact_sha256") != audit_digest:
        raise IsolationError("memory contamination attestation does not identify its audit")
    if (audit_dataset != evaluation_dataset_sha256
            or attestation.get("evaluation_dataset_sha256") != evaluation_dataset_sha256):
        raise IsolationError("memory was audited against a different evaluation dataset snapshot")
    return audit
