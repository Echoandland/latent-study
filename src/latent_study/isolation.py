from __future__ import annotations

import json
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("phase") != "frozen_before_evaluation":
        raise IsolationError("evaluation requires a pre-frozen study artifact")
    if payload.get("evaluation_inputs_seen"):
        raise IsolationError("artifact metadata reports evaluation contamination")
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
