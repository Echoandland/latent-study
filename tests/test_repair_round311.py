import json
from types import SimpleNamespace

import pytest
import torch

import latent_study.cli as cli
from latent_study.agent import Condition, FrozenRootAgent, InferenceBudget
from latent_study.artifacts import (ArtifactCompatibilityError, provenance,
                                    write_sidecar)
from latent_study.config import load_config
from latent_study.evaluation import (EvaluationBudgetProfile,
                                     budgets_from_config,
                                     validate_evaluation_budget_reachability)
from latent_study.search import CodingTools, TOOL_SCHEMA_HASH


MODEL_SNAPSHOT = "1" * 64
TOKENIZER_SNAPSHOT = "2" * 64


def _bank(tmp_path, *, mode="production_frozen_base", model=MODEL_SNAPSHOT,
          tokenizer=TOKENIZER_SNAPSHOT):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "records.jsonl"
    path.write_text("", encoding="utf-8")
    config = load_config("configs/dspy_mvp.json")
    metadata = provenance(
        artifact_type="study_record_bank", resolved_config_hash="command",
        protocol_config_hash=config["protocol_config_hash"],
        model_id=config["model"]["id"], model_revision=config["model"]["revision"],
        corpus_hash="corpus", tool_schema_hash=TOOL_SCHEMA_HASH, command="test",
        cli_overrides={}, model_snapshot_sha256=model,
        tokenizer_snapshot_sha256=tokenizer, artifact_path=path)
    metadata["study_bank_mode"] = mode
    write_sidecar(path, metadata)
    return path, config


def test_production_record_bank_requires_exact_snapshots_and_mode(tmp_path):
    path, config = _bank(tmp_path)
    assert cli._read_records(
        path, require_current=True, config=config, corpus_hash="corpus",
        model_snapshot_sha256=MODEL_SNAPSHOT,
        tokenizer_snapshot_sha256=TOKENIZER_SNAPSHOT,
        require_production=True) == []

    with pytest.raises(ArtifactCompatibilityError, match="model_snapshot_sha256 mismatch"):
        cli._read_records(path, require_current=True, config=config, corpus_hash="corpus",
                          model_snapshot_sha256="3" * 64,
                          tokenizer_snapshot_sha256=TOKENIZER_SNAPSHOT,
                          require_production=True)
    with pytest.raises(ArtifactCompatibilityError, match="tokenizer_snapshot_sha256 mismatch"):
        cli._read_records(path, require_current=True, config=config, corpus_hash="corpus",
                          model_snapshot_sha256=MODEL_SNAPSHOT,
                          tokenizer_snapshot_sha256="4" * 64,
                          require_production=True)

    missing, missing_config = _bank(tmp_path / "missing", model=None, tokenizer=None)
    with pytest.raises(ArtifactCompatibilityError, match="lacks model_snapshot_sha256"):
        cli._read_records(missing, require_current=True, config=missing_config,
                          corpus_hash="corpus", model_snapshot_sha256=MODEL_SNAPSHOT,
                          tokenizer_snapshot_sha256=TOKENIZER_SNAPSHOT,
                          require_production=True)

    synthetic, synthetic_config = _bank(
        tmp_path / "synthetic", mode="synthetic_deterministic_smoke", model=None,
        tokenizer=None)
    with pytest.raises(ArtifactCompatibilityError, match="production_frozen_base"):
        cli._read_records(synthetic, require_current=True, config=synthetic_config,
                          corpus_hash="corpus", model_snapshot_sha256=MODEL_SNAPSHOT,
                          tokenizer_snapshot_sha256=TOKENIZER_SNAPSHOT,
                          require_production=True)
    # Development smoke paths opt out explicitly; provenance/content checks remain active.
    assert cli._read_records(synthetic, require_current=True, config=synthetic_config,
                             corpus_hash="corpus", require_production=False) == []


def test_train_and_peek_commands_enter_the_production_record_gate(monkeypatch, tmp_path):
    config = load_config("configs/dspy_mvp.json")
    manifest = {"corpus_hash": "corpus", "root": str(tmp_path), "units": []}
    snapshot = {"resolved_path": str(tmp_path),
                "model_snapshot_sha256": MODEL_SNAPSHOT,
                "tokenizer_snapshot_sha256": TOKENIZER_SNAPSHOT}

    class GateReached(RuntimeError):
        pass

    def gated(_path, **kwargs):
        assert kwargs["require_production"] is True
        assert kwargs["model_snapshot_sha256"] == MODEL_SNAPSHOT
        assert kwargs["tokenizer_snapshot_sha256"] == TOKENIZER_SNAPSHOT
        raise GateReached

    monkeypatch.setattr(cli, "_config", lambda *_args: config)
    monkeypatch.setattr(cli, "_read_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(cli, "_verify_live_manifest", lambda *_args: None)
    monkeypatch.setattr(cli, "_resolve_model_files", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(cli, "_read_records", gated)

    common = dict(config="configs/dspy_mvp.json", output=str(tmp_path / "out"),
                  records=str(tmp_path / "records"), manifest=str(tmp_path / "manifest"),
                  contamination_audit=str(tmp_path / "audit"), model=str(tmp_path),
                  model_revision=None, max_records=None)
    with pytest.raises(GateReached):
        cli.command_peek(SimpleNamespace(
            **common, command="peek-study", local_files_only=True, token_budget=64,
            device="cpu", internal_max_new_tokens=None, retry_max_new_tokens=None,
            retries=None, replay=None, seed=None, batch_size=None, updates_per_source=None))
    with pytest.raises(GateReached):
        cli.command_train(SimpleNamespace(
            **common, command="train-latent", report=str(tmp_path / "report"), probes=None,
            max_probes=None, length=None, dtype=None, device="cpu", learning_rate=None,
            query_weight=None, lambda_rank=None, delta=None, replay=None, seed=None,
            batch_size=None, updates_per_source=None, relevant_label="A",
            irrelevant_label="B"))


def test_official_memory_dependency_rejects_synthetic_bank(tmp_path):
    bank, config = _bank(
        tmp_path / "bank", mode="synthetic_deterministic_smoke", model=None,
        tokenizer=None)
    memory = tmp_path / "memory.json"
    memory.write_text("{}", encoding="utf-8")
    metadata = provenance(
        artifact_type="offline_peek_map", resolved_config_hash="command",
        protocol_config_hash=config["protocol_config_hash"],
        model_id=config["model"]["id"], model_revision=config["model"]["revision"],
        corpus_hash="corpus", tool_schema_hash=TOOL_SCHEMA_HASH, command="test",
        cli_overrides={}, dependencies=[("study_record_bank", bank)],
        model_snapshot_sha256=MODEL_SNAPSHOT,
        tokenizer_snapshot_sha256=TOKENIZER_SNAPSHOT, artifact_path=memory)
    with pytest.raises(ArtifactCompatibilityError, match="production_frozen_base"):
        cli._assert_memory_uses_production_bank(
            metadata, memory, config=config, corpus_hash="corpus",
            model_snapshot_sha256=MODEL_SNAPSHOT,
            tokenizer_snapshot_sha256=TOKENIZER_SNAPSHOT)


def test_production_budgets_are_reachable_and_impossible_budget_fails():
    config = load_config("configs/dspy_mvp.json")
    profiles = budgets_from_config(config)
    report = validate_evaluation_budget_reachability(
        profiles, config["decoding"]["max_new_tokens_per_turn"])
    assert report["direct"]["maximum_reachable_generated_tokens"] == 1000
    assert report["max5"]["maximum_reachable_generated_tokens"] == 6000
    assert report["max20"]["maximum_reachable_generated_tokens"] == 21000
    assert report["exact20"]["declared_generated_token_budget"] == 10000
    with pytest.raises(ValueError, match="can reach at most 768"):
        validate_evaluation_budget_reachability(
            [EvaluationBudgetProfile("impossible", 5, 3000, 6000)], 128)


def test_root_agent_can_consume_declared_cap_under_real_loop_semantics(tmp_path):
    (tmp_path / "fact.py").write_text("VALUE = 7\n", encoding="utf-8")

    class Tokenizer:
        eos_token_id = None
        calls = 0

        def __call__(self, _text, **_kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, _ids, **_kwargs):
            self.calls += 1
            if self.calls <= 5:
                return json.dumps({"tool": "read_file", "path": "fact.py",
                                   "start_line": 1, "end_line": 1})
            return json.dumps({"final": "VALUE is 7"})

    class Model:
        device = torch.device("cpu")
        config = SimpleNamespace(model_type="toy")

        def generate(self, ids, *, max_new_tokens, **_kwargs):
            return torch.cat((ids, torch.ones((1, max_new_tokens), dtype=torch.long)), dim=1)

    budget = InferenceBudget(max_tool_calls=5, max_output_tokens=3000,
                             max_observation_bytes=6000)
    condition = Condition("no_study", "none", None, budget, {})
    result = FrozenRootAgent(Model(), Tokenizer(), CodingTools(tmp_path, ()), condition,
                             max_new_tokens_per_turn=500).run("What is VALUE?")
    assert result["tool_calls"] == 5
    assert result["model_output_tokens"] == 3000
    assert result["answer"] == "VALUE is 7"
