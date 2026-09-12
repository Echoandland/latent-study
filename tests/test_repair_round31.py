import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from latent_study.artifacts import (ARTIFACT_SCHEMA_VERSION,
                                    ArtifactCompatibilityError, provenance,
                                    validate_json_artifact)
from latent_study.cli import _config, _read_manifest
from latent_study.config import load_config, protocol_config_hash
from latent_study.corpus import (CorpusIntegrityError, build_manifest,
                                 verify_manifest_against_live_corpus)
from latent_study.evaluation import (EXPERTISE_ANCHOR_TOKENS,
                                     EvaluationBudgetProfile,
                                     budgets_from_config,
                                     validate_expertise_budget_coverage)
from latent_study.io import write_json
from latent_study.search import TOOL_SCHEMA_HASH
from latent_study.snapshots import resolve_model_snapshot


PROTOCOL_HASH = "a" * 64


def _metadata(kind, *, artifact_path, dependencies=(), model_snapshot=None,
              tokenizer_snapshot=None):
    return provenance(
        artifact_type=kind, resolved_config_hash="command", protocol_config_hash=PROTOCOL_HASH,
        model_id="model", model_revision="revision", corpus_hash="corpus",
        tool_schema_hash=TOOL_SCHEMA_HASH, command="test", cli_overrides={},
        dependencies=dependencies, artifact_path=artifact_path,
        tokenizer_id="model", tokenizer_revision="revision",
        model_snapshot_sha256=model_snapshot,
        tokenizer_snapshot_sha256=tokenizer_snapshot)


def test_production_budgets_span_expertise_anchor_and_nonspanning_fails():
    config = load_config("configs/dspy_mvp.json")
    profiles = budgets_from_config(config)
    caps = [profile.max_output_tokens for profile in profiles]
    assert min(caps) < EXPERTISE_ANCHOR_TOKENS <= max(caps)
    assert caps == [1000, 3000, 6000, 10000]
    validate_expertise_budget_coverage(profiles)

    with pytest.raises(ValueError, match="do not reach the 3000-token expertise anchor"):
        validate_expertise_budget_coverage([
            EvaluationBudgetProfile("too-small", 20, 2999, 6000)])
    with pytest.raises(ValueError, match="do not reach the 3000-token expertise anchor"):
        budgets_from_config(config, ["direct"])


def test_effective_cli_protocol_override_changes_hash_but_output_does_not(tmp_path):
    common = dict(command="generate-records", config="configs/dspy_mvp.json",
                  actions=99, output=str(tmp_path / "one.jsonl"))
    changed = _config(SimpleNamespace(**common), common["output"])
    base = load_config("configs/dspy_mvp.json")
    assert changed["candidate_generation"]["n"] == 99
    assert changed["protocol_config_hash"] != base["protocol_config_hash"]

    other_output = dict(common, output=str(tmp_path / "elsewhere" / "two.jsonl"))
    same_protocol = _config(SimpleNamespace(**other_output), other_output["output"])
    assert same_protocol["protocol_config_hash"] == changed["protocol_config_hash"]
    assert same_protocol["resolved_config_hash"] == changed["resolved_config_hash"]

    artifact = tmp_path / "bound.json"
    write_json(artifact, {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                          "provenance": provenance(
                              artifact_type="bound", resolved_config_hash="command",
                              protocol_config_hash=base["protocol_config_hash"],
                              model_id="model", model_revision="revision", corpus_hash="corpus",
                              tool_schema_hash=TOOL_SCHEMA_HASH, command="test", cli_overrides={},
                              artifact_path=artifact)})
    with pytest.raises(ArtifactCompatibilityError, match="protocol_config_hash mismatch"):
        validate_json_artifact(artifact, require_current_source=False,
                               protocol_config_hash=changed["protocol_config_hash"])


def _fake_model_snapshot(root: Path):
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3_5"}\n', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights-v1")
    (root / "tokenizer.json").write_text('{"version":"v1"}\n', encoding="utf-8")
    (root / "tokenizer_config.json").write_text('{"chat_template":"v1"}\n', encoding="utf-8")


def test_local_model_and_tokenizer_contents_are_bound(tmp_path):
    local = tmp_path / "Qwen3.5-9B"
    _fake_model_snapshot(local)
    first = resolve_model_snapshot(local, "claimed-revision")

    artifact = tmp_path / "memory.json"
    write_json(artifact, {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "provenance": _metadata(
            "offline_peek_map", artifact_path=artifact,
            model_snapshot=first["model_snapshot_sha256"],
            tokenizer_snapshot=first["tokenizer_snapshot_sha256"]),
    })
    validate_json_artifact(
        artifact, require_current_source=False,
        model_snapshot_sha256=first["model_snapshot_sha256"],
        tokenizer_snapshot_sha256=first["tokenizer_snapshot_sha256"])

    (local / "model.safetensors").write_bytes(b"weights-v2")
    second = resolve_model_snapshot(local, "claimed-revision")
    assert second["model_snapshot_sha256"] != first["model_snapshot_sha256"]
    assert second["tokenizer_snapshot_sha256"] == first["tokenizer_snapshot_sha256"]
    with pytest.raises(ArtifactCompatibilityError, match="model_snapshot_sha256 mismatch"):
        validate_json_artifact(artifact, require_current_source=False,
                               model_snapshot_sha256=second["model_snapshot_sha256"])

    (local / "tokenizer.json").write_text('{"version":"v2"}\n', encoding="utf-8")
    third = resolve_model_snapshot(local, "claimed-revision")
    assert third["tokenizer_snapshot_sha256"] != first["tokenizer_snapshot_sha256"]
    with pytest.raises(ArtifactCompatibilityError, match="tokenizer_snapshot_sha256 mismatch"):
        validate_json_artifact(artifact, require_current_source=False,
                               tokenizer_snapshot_sha256=third["tokenizer_snapshot_sha256"])


def _realistic_bundle(root: Path):
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    source = corpus / "facts.py"
    source.write_text("def stable_fact():\n    return 7\n", encoding="utf-8")
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir()
    manifest_path = artifact_dir / "manifest.json"
    manifest = build_manifest(corpus)
    manifest["root"] = "../corpus"
    manifest["root_semantics"] = "artifact_relative_diagnostic_only"
    manifest["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    manifest["provenance"] = provenance(
        artifact_type="corpus_manifest", resolved_config_hash="command",
        protocol_config_hash=PROTOCOL_HASH, model_id="model", model_revision="revision",
        corpus_hash=manifest["corpus_hash"], tool_schema_hash=TOOL_SCHEMA_HASH,
        command="test manifest", cli_overrides={},
        dependencies=[("corpus_snapshot", corpus)], artifact_path=manifest_path)
    write_json(manifest_path, manifest)

    study_path = artifact_dir / "study.json"
    write_json(study_path, {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "records": [{"record_id": "r1", "corpus_hash": manifest["corpus_hash"]}],
        "provenance": provenance(
            artifact_type="study_bundle", resolved_config_hash="command",
            protocol_config_hash=PROTOCOL_HASH, model_id="model", model_revision="revision",
            corpus_hash=manifest["corpus_hash"], tool_schema_hash=TOOL_SCHEMA_HASH,
            command="test study", cli_overrides={},
            dependencies=[("corpus_manifest", manifest_path)], artifact_path=study_path),
    })
    return corpus, manifest_path, study_path


def test_complete_corpus_manifest_study_bundle_relocates_and_remains_closed(tmp_path):
    original = tmp_path / "original"
    _, _, _ = _realistic_bundle(original)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    manifest_path = relocated / "artifacts" / "manifest.json"
    study_path = relocated / "artifacts" / "study.json"
    config = {"model": {"id": "model", "revision": "revision"},
              "protocol_config_hash": PROTOCOL_HASH}

    manifest = _read_manifest(manifest_path, require_current=True, config=config)
    assert Path(manifest["root"]) == (relocated / "corpus").resolve()
    assert manifest["diagnostic_original_root"] == "../corpus"
    assert verify_manifest_against_live_corpus(manifest, manifest["root"])["verified"]
    validate_json_artifact(study_path, require_current_source=True,
                           artifact_type="study_bundle", protocol_config_hash=PROTOCOL_HASH)

    source = relocated / "corpus" / "facts.py"
    source.write_text("def stable_fact():\n    return 8\n", encoding="utf-8")
    with pytest.raises((ArtifactCompatibilityError, CorpusIntegrityError)):
        _read_manifest(manifest_path, require_current=True, config=config)
    source.write_text("def stable_fact():\n    return 7\n", encoding="utf-8")
    added = relocated / "corpus" / "added.py"
    added.write_text("ADDED = True\n", encoding="utf-8")
    with pytest.raises((ArtifactCompatibilityError, CorpusIntegrityError)):
        _read_manifest(manifest_path, require_current=True, config=config)
    added.unlink()
    source.unlink()
    with pytest.raises((ArtifactCompatibilityError, CorpusIntegrityError)):
        _read_manifest(manifest_path, require_current=True, config=config)
