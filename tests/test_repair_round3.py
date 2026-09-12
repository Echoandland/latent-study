import copy
import gc
import json
import shutil
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latent_study.agent import (Condition, FrozenRootAgent, InferenceBudget,
                                reset_cuda_peak_memory_for_run)
from latent_study.artifacts import (ARTIFACT_SCHEMA_VERSION,
                                    ArtifactCompatibilityError, provenance,
                                    validate_json_artifact)
from latent_study.config import protocol_config_hash, protocol_config_view
from latent_study.corpus import (CorpusIntegrityError, build_manifest,
                                 corpus_snapshot_hash,
                                 verify_manifest_against_live_corpus)
from latent_study.evaluation import (EvaluationBudgetProfile, MVP_CONDITIONS,
                                     SequentialRunnerFactory,
                                     evaluation_dataset_snapshot_hash,
                                     load_evaluation_dataset, run_evaluation)
from latent_study.io import write_json
from latent_study.isolation import (IsolationError,
                                    audit_evaluation_contamination,
                                    contamination_attestation,
                                    verify_memory_contamination_binding)
from latent_study.peek_baseline import study_offline_peek
from latent_study.records import generate_coverage_records
from latent_study.search import CodingTools, TOOL_SCHEMA_HASH


PROTOCOL_HASH = "a" * 64


def _provenance(kind, *, artifact_path=None, dependencies=(), protocol_hash=PROTOCOL_HASH,
                corpus_hash="corpus"):
    return provenance(
        artifact_type=kind, resolved_config_hash="command-config",
        protocol_config_hash=protocol_hash, model_id="model", model_revision="revision",
        corpus_hash=corpus_hash, tool_schema_hash=TOOL_SCHEMA_HASH,
        command="test", cli_overrides={}, dependencies=dependencies,
        artifact_path=artifact_path)


def test_protocol_config_projection_separates_scientific_and_local_fields(tmp_path):
    config = json.loads(Path("configs/dspy_mvp.json").read_text(encoding="utf-8"))
    original = protocol_config_hash(config)
    local_change = copy.deepcopy(config)
    local_change["output_path"] = str(tmp_path / "elsewhere" / "result.json")
    assert protocol_config_hash(local_change) == original

    training_change = copy.deepcopy(config)
    training_change["training"]["batch_size"] = 99
    assert protocol_config_hash(training_change) != original

    scientific_change = copy.deepcopy(config)
    scientific_change["model"]["revision"] = "different-revision"
    changed = protocol_config_hash(scientific_change)
    assert changed != original
    assert "authorized_roots" not in protocol_config_view(config)["corpus"]

    artifact = tmp_path / "result.json"
    write_json(artifact, {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                          "provenance": _provenance(
                              "evaluation_result", artifact_path=artifact,
                              protocol_hash=original)})
    validate_json_artifact(artifact, require_current_source=False,
                           protocol_config_hash=original)
    with pytest.raises(ArtifactCompatibilityError, match="protocol_config_hash mismatch"):
        validate_json_artifact(artifact, require_current_source=False,
                               protocol_config_hash=changed)


def test_dependency_bundle_remains_valid_after_relocation(tmp_path):
    source = tmp_path / "source_bundle"
    source.mkdir()
    dependency = source / "parent.json"
    dependency.write_text('{"stable":true}\n', encoding="utf-8")
    child = source / "child.json"
    write_json(child, {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                       "value": 7,
                       "provenance": _provenance(
                           "portable_child", artifact_path=child,
                           dependencies=[("parent", dependency)])})
    stored = json.loads(child.read_text(encoding="utf-8"))["provenance"]["dependencies"][0]
    assert not Path(stored["path"]).is_absolute()
    assert stored["path_base"] == "artifact"

    relocated = tmp_path / "relocated_bundle"
    shutil.copytree(source, relocated)
    payload = validate_json_artifact(relocated / "child.json", require_current_source=False,
                                     artifact_type="portable_child",
                                     protocol_config_hash=PROTOCOL_HASH)
    assert payload["value"] == 7


def test_corpus_snapshot_uses_manifest_eligibility_rules(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "a.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git_dir = corpus / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("one\n", encoding="utf-8")
    manifest = build_manifest(corpus)
    original = corpus_snapshot_hash(corpus)

    (git_dir / "HEAD").write_text("two\n", encoding="utf-8")
    (corpus / "ignored.bin").write_bytes(b"ignored")
    assert corpus_snapshot_hash(corpus) == original
    assert verify_manifest_against_live_corpus(manifest)["verified"]

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(CorpusIntegrityError, match="modified_files"):
        verify_manifest_against_live_corpus(manifest)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    added = corpus / "added.py"
    added.write_text("ADDED = 1\n", encoding="utf-8")
    with pytest.raises(CorpusIntegrityError, match="unexpected_eligible_files"):
        verify_manifest_against_live_corpus(manifest)
    added.unlink()
    source.unlink()
    with pytest.raises(CorpusIntegrityError, match="missing_files"):
        verify_manifest_against_live_corpus(manifest)


def _contamination_bundle(root: Path, *, question="Which value is audited?"):
    root.mkdir()
    evaluation = root / "evaluation.jsonl"
    evaluation.write_text(json.dumps({"id": "eval-1", "question": question,
                                      "answer": "seven"}) + "\n", encoding="utf-8")
    study = root / "study.jsonl"
    study.write_text(json.dumps({"record_id": "study-1",
                                 "prompt": "Unrelated corpus navigation"}) + "\n",
                     encoding="utf-8")
    audit_path = root / "audit.json"
    audit = audit_evaluation_contamination([study], [evaluation])
    audit.update({"artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                  "phase": "audit_complete", "evaluation_inputs_seen": True,
                  "provenance": _provenance(
                      "evaluation_contamination_audit", artifact_path=audit_path,
                      dependencies=[("study", study), ("evaluation", evaluation)],
                      corpus_hash="audit_only")})
    write_json(audit_path, audit)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    memory_path = root / "memory.json"
    memory = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
              "phase": "frozen_before_evaluation", "evaluation_inputs_seen": False,
              "contamination_audit": contamination_attestation(audit),
              "map_text": "facts.py contains navigation",
              "provenance": _provenance(
                  "offline_peek_map", artifact_path=memory_path,
                  dependencies=[("contamination_audit", audit_path)])}
    write_json(memory_path, memory)
    return evaluation, audit_path, memory_path


def test_contamination_audit_is_bound_to_current_evaluation_snapshot(tmp_path):
    evaluation, _, memory_path = _contamination_bundle(tmp_path / "matching")
    memory = validate_json_artifact(memory_path, require_current_source=False,
                                    protocol_config_hash=PROTOCOL_HASH)
    audit = verify_memory_contamination_binding(
        memory, memory_path,
        evaluation_dataset_sha256=evaluation_dataset_snapshot_hash(evaluation),
        protocol_config_hash=PROTOCOL_HASH)
    assert audit["status"] == "pass"


def test_contamination_binding_rejects_other_mutated_and_tampered_datasets(tmp_path):
    evaluation, audit_path, memory_path = _contamination_bundle(tmp_path / "bundle")
    memory = validate_json_artifact(memory_path, require_current_source=False,
                                    protocol_config_hash=PROTOCOL_HASH)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"id": "eval-2", "question": "Different dataset?"}) + "\n")
    with pytest.raises(IsolationError, match="different evaluation dataset"):
        verify_memory_contamination_binding(
            memory, memory_path,
            evaluation_dataset_sha256=evaluation_dataset_snapshot_hash(other),
            protocol_config_hash=PROTOCOL_HASH)

    evaluation.write_text(json.dumps({"id": "eval-1", "question": "Mutated after audit?"}) + "\n")
    with pytest.raises(IsolationError, match="dependency is invalid"):
        verify_memory_contamination_binding(
            memory, memory_path,
            evaluation_dataset_sha256=evaluation_dataset_snapshot_hash(evaluation),
            protocol_config_hash=PROTOCOL_HASH)

    evaluation.write_text(json.dumps({"id": "eval-1", "question": "Which value is audited?",
                                      "answer": "seven"}) + "\n")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_payload["collision_count"] = 99
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")
    with pytest.raises(IsolationError, match="dependency is invalid"):
        verify_memory_contamination_binding(
            memory, memory_path,
            evaluation_dataset_sha256=evaluation_dataset_snapshot_hash(evaluation),
            protocol_config_hash=PROTOCOL_HASH)


def test_evaluation_is_sequential_and_expertise_uses_budget_aggregates(tmp_path):
    dataset = tmp_path / "evaluation.jsonl"
    dataset.write_text("".join(json.dumps({"id": f"e{index}", "question": "Q"}) + "\n"
                               for index in range(4)), encoding="utf-8")
    examples = load_evaluation_dataset(dataset)
    budgets = (EvaluationBudgetProfile("direct", 0, 3000, 100),
               EvaluationBudgetProfile("max5", 5, 5000, 100))
    shared_backbone = object()
    state = {"live": 0, "max_live": 0, "built": 0, "released": 0}
    seen_backbones = []
    prior_runners = []

    class Runner:
        def __init__(self, profile):
            self.profile = profile
            self.backbone = shared_backbone

        def __call__(self, _example, _profile):
            generated = 1000 if self.profile.name == "direct" else 4000
            return {"answer": "ok", "tool_calls": 0 if self.profile.name == "direct" else 1,
                    "model_input_tokens": 10, "model_output_tokens": generated,
                    "returned_observation_bytes": 0,
                    "strict": .25 if self.profile.name == "direct" else .75,
                    "lenient": .5 if self.profile.name == "direct" else 1.0}

    def build(_condition, profile):
        gc.collect()
        assert all(reference() is None for reference in prior_runners)
        state["live"] += 1
        state["max_live"] = max(state["max_live"], state["live"])
        state["built"] += 1
        seen_backbones.append(id(shared_backbone))
        runner = Runner(profile)
        prior_runners.append(weakref.ref(runner))
        return runner

    def release(_runner):
        state["live"] -= 1
        state["released"] += 1

    factory = SequentialRunnerFactory(build, release, shared_backbone=shared_backbone)
    payload = run_evaluation(
        examples, runner_factory=factory, budgets=budgets,
        scorer=lambda _example, result: {"strict": result["strict"],
                                         "lenient": result["lenient"]})
    assert state == {"live": 0, "max_live": 1, "built": 10, "released": 10}
    gc.collect()
    assert all(reference() is None for reference in prior_runners)
    assert set(seen_backbones) == {id(shared_backbone)}
    assert payload["model_lifetime"]["strategy"] == "single_shared_backbone"
    assert len(payload["results"]) == len(MVP_CONDITIONS) * len(budgets) * 4

    summary = payload["condition_summaries"]["no_study"]
    direct = summary["budgets"]["direct"]
    assert direct["examples"] == 4
    assert direct["compute"]["generated_tokens"]["mean_per_example"] == 1000
    assert direct["compute"]["generated_tokens"]["total"] == 4000
    assert len(summary["performance_vs_compute"]["strict"]) == 2
    # Four examples form one score at each configured per-example generation
    # budget; actual total/mean generation never becomes the x coordinate.
    assert summary["expertise_input_points"]["strict"] == [(3000.0, 0.25), (5000.0, 0.75)]
    assert summary["expertise_strict"] == pytest.approx(0.25 * (1 - 3000 / 5000) + 0.75 * 3000 / 5000)


def test_cuda_peak_memory_is_reset_and_read_within_each_agent_run(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                        lambda device: events.append(("reset", device)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                        lambda device: events.append(("read", device)) or 1234)

    class Tokenizer:
        eos_token_id = None

        def __call__(self, text, **_kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

        def decode(self, _ids, **_kwargs):
            return '{"final":"done"}'

    class Model:
        device = torch.device("cpu")
        config = SimpleNamespace(model_type="toy")

        def generate(self, ids, **_kwargs):
            events.append(("generate", self.device))
            return torch.cat((ids, torch.tensor([[3]])), dim=1)

    model = Model()
    assert reset_cuda_peak_memory_for_run(model) is None
    events.clear()
    condition = Condition("no_study", "none", None, InferenceBudget(0, 1, 10), {})
    agent = FrozenRootAgent(model, Tokenizer(), CodingTools(tmp_path, ()), condition,
                            max_new_tokens_per_turn=1)
    result = agent.run("question")
    assert [event[0] for event in events] == ["reset", "generate", "read"]
    assert result["peak_memory_bytes"] == 1234
    assert result["peak_memory_scope"].startswith("per example run")


def test_peek_rejects_record_bank_from_another_manifest_before_client_use(tmp_path):
    corpus_one = tmp_path / "one"
    corpus_two = tmp_path / "two"
    corpus_one.mkdir(); corpus_two.mkdir()
    (corpus_one / "a.py").write_text("def alpha():\n    return 1\n")
    (corpus_one / "a_other.py").write_text("def alpha_other():\n    return 3\n")
    (corpus_two / "b.py").write_text("def beta():\n    return 2\n")
    (corpus_two / "b_other.py").write_text("def beta_other():\n    return 4\n")
    manifest_one, manifest_two = build_manifest(corpus_one), build_manifest(corpus_two)
    records_two = generate_coverage_records(manifest_two["units"], manifest_two["corpus_hash"])

    class NeverCalledClient:
        calls = 0

        def completion(self, _messages):
            self.calls += 1
            raise AssertionError("model/client must not run on a corpus mismatch")

    client = NeverCalledClient()
    with pytest.raises(IsolationError, match="foreign corpus hash"):
        study_offline_peek(records_two, tmp_path / "peek.json", token_budget=64,
                           client=client, token_counter=lambda text: len(text.split()),
                           expected_corpus_hash=manifest_one["corpus_hash"])
    assert client.calls == 0


def test_peek_cli_uses_manifest_hash_before_model_setup(tmp_path, monkeypatch):
    import latent_study.cli as cli
    import transformers

    corpus_one = tmp_path / "manifest-corpus"
    corpus_two = tmp_path / "record-corpus"
    corpus_one.mkdir(); corpus_two.mkdir()
    for root, prefix in ((corpus_one, "one"), (corpus_two, "two")):
        (root / "a.py").write_text(f"def {prefix}_a():\n    return 1\n")
        (root / "b.py").write_text(f"def {prefix}_b():\n    return 2\n")
    manifest_one, manifest_two = build_manifest(corpus_one), build_manifest(corpus_two)
    records_two = generate_coverage_records(manifest_two["units"], manifest_two["corpus_hash"])
    seen = {}

    monkeypatch.setattr(cli, "_config", lambda _args, _artifact: {
        "protocol_config_hash": PROTOCOL_HASH,
        "model": {"id": "model", "revision": "revision", "dtype": "float32"},
        "peek_internal": {"max_new_tokens": 8, "retry_max_new_tokens": 16},
        "peek": {"retries": 0}, "replay": {"fraction": .5}, "seed": 1,
        "training": {"batch_size": 2, "steps_per_source": 1}})
    monkeypatch.setattr(cli, "_reject_evaluation_paths", lambda *_args: None)
    monkeypatch.setattr(cli, "_read_manifest", lambda *_args, **_kwargs: manifest_one)
    monkeypatch.setattr(cli, "_verify_live_manifest", lambda _manifest: None)

    def read_records(_path, **kwargs):
        seen["expected_corpus_hash"] = kwargs.get("corpus_hash")
        return records_two

    monkeypatch.setattr(cli, "_read_records", read_records)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *_args, **_kwargs: pytest.fail("model loaded before corpus rejection"))
    args = SimpleNamespace(
        output=str(tmp_path / "peek.json"), records="records.jsonl", manifest="manifest.json",
        contamination_audit="audit.json", max_records=None, model="model",
        model_revision="revision", local_files_only=True, device="cpu", token_budget=64,
        internal_max_new_tokens=None, retry_max_new_tokens=None, retries=None,
        replay=None, seed=None, batch_size=None, updates_per_source=None)
    with pytest.raises(IsolationError, match="foreign corpus hash"):
        cli.command_peek(args)
    assert seen["expected_corpus_hash"] == manifest_one["corpus_hash"]
