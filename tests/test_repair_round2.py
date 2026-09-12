import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latent_study.agent import parse_tool_actions
from latent_study.artifacts import (ARTIFACT_SCHEMA_VERSION, ArtifactCompatibilityError, provenance,
                                    read_sidecar, write_sidecar)
from latent_study.corpus import (CorpusIntegrityError, build_manifest,
                                 verify_manifest_against_live_corpus)
from latent_study.evaluation import (EvaluationBudgetProfile, MVP_CONDITIONS,
                                     load_evaluation_dataset, run_evaluation,
                                     validate_budget_result)
from latent_study.isolation import (IsolationError, assert_frozen_payload,
                                    audit_evaluation_contamination)
from latent_study.peek_baseline import QwenPeekClient
from latent_study.search import (CodingTools, SearchLimits, TOOL_SCHEMA_HASH,
                                 ToolActionValidationError,
                                 parse_and_validate_tool_action)
from latent_study.artifacts import source_tree_hash


def _metadata(kind="study_record_bank"):
    return provenance(artifact_type=kind, resolved_config_hash="cfg",
                      protocol_config_hash="a" * 64, model_id="m",
                      model_revision="r", corpus_hash="c", tool_schema_hash=TOOL_SCHEMA_HASH,
                      command="test", cli_overrides={})


def test_sidecar_content_hash_fails_closed_after_jsonl_mutation(tmp_path):
    bank = tmp_path / "bank.jsonl"
    bank.write_text('{"record_id":"r1","prompt":"stable"}\n', encoding="utf-8")
    write_sidecar(bank, _metadata())
    assert read_sidecar(bank, require_current=False)["artifact_sha256"]
    bank.write_text('{"record_id":"r1","prompt":"mutated"}\n', encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="content SHA256 mismatch"):
        read_sidecar(bank, require_current=False)


def test_dependency_hash_is_checked_even_when_primary_is_unchanged(tmp_path):
    dependency = tmp_path / "manifest.json"
    dependency.write_text('{"corpus_hash":"one"}\n', encoding="utf-8")
    bank = tmp_path / "bank.jsonl"
    bank.write_text('{"record_id":"r1"}\n', encoding="utf-8")
    write_sidecar(bank, _metadata(), dependencies=[("manifest", dependency)])
    dependency.write_text('{"corpus_hash":"two"}\n', encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="dependency content SHA256 mismatch"):
        read_sidecar(bank, require_current_source=False)


def test_live_manifest_verification_precedes_tool_initialization(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_manifest(root)
    (root / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(CorpusIntegrityError, match="modified_files"):
        verify_manifest_against_live_corpus(manifest)
    with pytest.raises(CorpusIntegrityError):
        CodingTools(root, manifest["units"], SearchLimits(), manifest=manifest)


@pytest.mark.parametrize("value", [
    {"tool": "grep", "query": "x", "extra": 1},
    {"tool": "grep", "query": "x", "path": 123},
    {"tool": "glob", "pattern": "*.py", "max_results": -1},
])
def test_tool_schema_rejects_malformed_actions_without_runtime_crash(value):
    with pytest.raises(ToolActionValidationError):
        parse_and_validate_tool_action(value)
    # The runtime boundary receives arbitrary model data and returns a
    # controlled invalid result rather than propagating a Python exception.
    tools = CodingTools(Path("/tmp"), ())
    valid, hits, observation = tools.execute(value)
    assert valid is False and hits == () and observation.startswith("ERROR: invalid tool action")


def test_parse_tool_actions_reports_invalid_json_as_controlled_error():
    actions, errors = parse_tool_actions('{"tool":"grep","query":"x","extra":1}',
                                         return_errors=True)
    assert actions == [] and errors and "unknown properties" in errors[0]
    actions, errors = parse_tool_actions('{"tool":"grep","query":"x","query":"y"}',
                                         return_errors=True)
    assert actions == [] and errors and "duplicate property" in errors[0]
    actions, errors = parse_tool_actions(None, return_errors=True)
    assert actions == [] and errors and "string" in errors[0]
    with pytest.raises(ToolActionValidationError, match="duplicate property"):
        parse_and_validate_tool_action('{"tool":"grep","query":"x","query":"y"}')


def test_contamination_audit_clean_id_and_question_collision(tmp_path):
    study = tmp_path / "study.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    evaluation.write_text(json.dumps({"example_id": "eval-17",
                                      "question": "What is the distinctive lunar checksum?"}) + "\n")
    study.write_text(json.dumps({"record_id": "record-1", "prompt": "unrelated corpus fact"}) + "\n")
    assert audit_evaluation_contamination([study], [evaluation])["status"] == "pass"
    study.write_text(json.dumps({"record_id": "eval-17", "prompt": "unrelated corpus fact"}) + "\n")
    failed_id = audit_evaluation_contamination([study], [evaluation])
    assert failed_id["status"] == "fail" and any(c["type"] == "evaluation_id" for c in failed_id["collisions"])
    study.write_text(json.dumps({"record_id": "record-1",
                                 "prompt": "What is the distinctive lunar checksum?"}) + "\n")
    failed_question = audit_evaluation_contamination([study], [evaluation])
    assert failed_question["status"] == "fail"
    assert any(c["type"] == "evaluation_content" for c in failed_question["collisions"])

    # Study-side consumers must not reopen evaluation dependencies while
    # loading an already-issued audit attestation.
    from latent_study.cli import _read_contamination_audit
    from latent_study.io import write_json
    audit_path = tmp_path / "audit.json"
    audit_payload = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "status": "pass",
                     "phase": "audit_complete", "evaluation_inputs_seen": True,
                     "provenance": provenance(
                         artifact_type="evaluation_contamination_audit",
                         resolved_config_hash="cfg", protocol_config_hash="a" * 64,
                         model_id="m", model_revision="r",
                         corpus_hash="audit-only", tool_schema_hash=TOOL_SCHEMA_HASH,
                         command="latent-study audit-contamination", cli_overrides={},
                         dependencies=[("evaluation", evaluation)])}
    write_json(audit_path, audit_payload)
    evaluation.unlink()
    assert _read_contamination_audit(audit_path)["status"] == "pass"


def test_frozen_payload_requires_passing_contamination_attestation():
    base = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "phase": "frozen_before_evaluation",
            "evaluation_inputs_seen": False,
            "provenance": {"artifact_sha256": "a" * 64}}
    with pytest.raises(IsolationError):
        assert_frozen_payload(base)
    assert assert_frozen_payload({**base, "contamination_audit": {
        "status": "pass", "artifact_sha256": "b" * 64,
        "evaluation_dataset_sha256": "c" * 64}})["phase"]


def test_source_tree_hash_is_cwd_independent(tmp_path):
    repo = Path(__file__).parents[1].resolve()
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    code = "from latent_study.artifacts import source_tree_hash; print(source_tree_hash())"
    values = []
    for cwd in (repo, tmp_path):
        values.append(subprocess.check_output([sys.executable, "-c", code], cwd=cwd,
                                              env=env, text=True).strip())
    assert values[0] == values[1] == source_tree_hash()


class _CaptureTokenizer:
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        return torch.tensor([[1, 2]])

    def decode(self, ids, **kwargs):
        return getattr(self, "text", "{}")


class _CaptureModel:
    device = torch.device("cpu")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def generate(self, ids, **kwargs):
        self.calls.append(kwargs["max_new_tokens"])
        self.tokenizer.text = (json.dumps({"diagnosis": "ok", "item_tags": {},
                                           "cache_candidates": []})
                               if len(self.calls) > 1 else "{\"diagnosis\":")
        return torch.cat((ids, torch.tensor([[3]])), dim=1)


def test_peek_retry_uses_larger_internal_budget_than_initial_attempt():
    tokenizer = _CaptureTokenizer()
    model = _CaptureModel(tokenizer)
    client = QwenPeekClient(model, tokenizer, internal_max_new_tokens=384,
                            retry_max_new_tokens=1536, retries=1)
    result = client.completion([{"role": "user", "content": "trajectory"}])
    assert json.loads(result)["diagnosis"] == "ok"
    assert model.calls == [384, 1536]


def test_peek_final_map_budget_is_separate_and_overbudget_fails_closed(tmp_path):
    from peek.core.types import Usage
    from latent_study.records import generate_coverage_records
    from latent_study.peek_baseline import study_offline_peek

    class SemanticClient:
        def completion(self, messages):
            if "Available Operations" in messages[-1]["content"]:
                return json.dumps({"reasoning": "retain shared navigation", "operations": [{
                    "type": "ADD", "section": "context_understanding",
                    "content": "facts.py function alpha is located here"}]})
            return json.dumps({"diagnosis": "shared corpus structure", "item_tags": {},
                               "cache_candidates": [{"section": "context_understanding",
                                                      "value": "facts.py function alpha is located here",
                                                      "transferability": "future lookup",
                                                      "rationale": "file navigation"}]})

        def last_usage(self):
            return Usage(3, 2)

    corpus = tmp_path / "peek-budget"; corpus.mkdir()
    (corpus / "facts.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (corpus / "other.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    manifest = build_manifest(corpus)
    records = generate_coverage_records(manifest["units"], manifest["corpus_hash"], n_actions=2)
    counter = lambda text: len(text.split())
    success = tmp_path / "success.json"
    payload = study_offline_peek(records, success, token_budget=64, client=SemanticClient(),
                                 token_counter=counter, counter_name="synthetic", updates_per_source=1)
    assert payload["map_text_tokens"] <= 64
    assert payload["map_text_bytes"] < payload["complete_artifact_bytes"]
    assert payload["phase"] == "study_complete_unattested"

    failed = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="diagnostics written"):
        study_offline_peek(records, failed, token_budget=1, client=SemanticClient(),
                           token_counter=counter, counter_name="synthetic", updates_per_source=1)
    assert json.loads(failed.read_text(encoding="utf-8"))["phase"] == "study_failed"


def test_evaluation_harness_emits_five_conditions_and_compute_artifact(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"id": "e1", "question": "Use the corpus."}) + "\n")
    examples = load_evaluation_dataset(dataset)
    budget = EvaluationBudgetProfile("max5", 5, 100, 100)
    runners = {name: (lambda example, _budget: {
        "answer": "supported", "tool_calls": 1, "model_input_tokens": 10,
        "model_output_tokens": 20, "returned_observation_bytes": 4,
        "inference_latency_seconds": 0.001,
    }) for name in MVP_CONDITIONS}
    output = tmp_path / "evaluation.json"
    payload = run_evaluation(examples, runners, budget=budget,
                             scorer=lambda _example, _result: {"strict": 1, "lenient": 1},
                             output=output, provenance=_metadata("evaluation_result"),
                             require_expertise_budget_coverage=False)
    assert payload["conditions"] == list(MVP_CONDITIONS)
    assert len(payload["results"]) == 5
    assert payload["results"][0]["compute"]["input_tokens"] == 10
    assert payload["expertise_metric"]["manual_points_required"] is False
    assert len(json.loads(output.read_text())["provenance"]["artifact_sha256"]) == 64


def test_budget_profiles_keep_exact_tool_iteration_semantics_explicit():
    exact = EvaluationBudgetProfile("exact20", 20, 100, 100, exact_tool_calls=20,
                                    allow_early_return=False)
    assert validate_budget_result({"tool_calls": 19, "model_output_tokens": 4,
                                   "returned_observation_bytes": 1}, exact)["valid"] is False
    assert validate_budget_result({"tool_calls": 20, "model_output_tokens": 4,
                                   "returned_observation_bytes": 1}, exact)["valid"] is True


def test_cli_help_exposes_model_not_stale_tokenizer_argument():
    from latent_study.cli import parser
    root = parser()
    sub = next(action for action in root._actions if getattr(action, "dest", None) == "command")
    peek = sub.choices["peek-study"]
    option_strings = {option for action in peek._actions for option in action.option_strings}
    assert "--model" in option_strings and "--tokenizer" not in option_strings


def test_exact_budget_forces_continuation_and_counts_text_memory_once(tmp_path):
    """The exact-iteration profile must not accept a zero-step final answer."""
    from latent_study.agent import (Condition, FrozenRootAgent, InferenceBudget,
                                    ROOT_SYSTEM_PROMPT, serialize_chat_ids)

    class Tokenizer:
        eos_token_id = None

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[ord(char) % 97 for char in text]], dtype=torch.long))

        def encode(self, text, **kwargs):
            return [ord(char) % 97 for char in text]

        def convert_tokens_to_ids(self, token):
            return -1

        def decode(self, ids, **kwargs):
            marker = int(ids[0])
            if marker == 901:
                return '{"tool":"read_file","path":"fact.py","start_line":1,"end_line":1}'
            return '{"final":"grounded VALUE = 7"}'

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.config = SimpleNamespace(model_type="toy")
            self.calls = 0

        def generate(self, ids, **kwargs):
            self.calls += 1
            # The first answer tries to return early, the second performs the
            # required tool action, and the third can finish.
            marker = 902 if self.calls == 1 or self.calls >= 3 else 901
            return torch.cat((ids, torch.tensor([[marker]])), dim=1)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "fact.py").write_text("VALUE = 7\n", encoding="utf-8")
    manifest = build_manifest(corpus)
    tools = CodingTools(corpus, manifest["units"])
    tokenizer, model = Tokenizer(), Model()
    budget = InferenceBudget(1, 3, 200, exact_tool_calls=1, allow_early_return=False)
    condition = __import__("latent_study.agent", fromlist=["Condition"]).Condition(
        "offline_peek_64", "map", "map.json", budget, {"max_results": 5})
    agent = FrozenRootAgent(model, tokenizer, tools, condition,
                            map_text="fact.py contains VALUE = 7", max_new_tokens_per_turn=1)
    result = agent.run("Use one tool before answering.")
    assert result["tool_calls"] == 1
    assert result["answer"] == "grounded VALUE = 7"
    assert any(turn["role"] == "user" and "exactly 1 tool calls" in turn["content"]
               for turn in result["transcript"])
    ids = serialize_chat_ids(tokenizer,
                             [{"role": "system", "content": ROOT_SYSTEM_PROMPT},
                              {"role": "user", "content": "Use one tool before answering."}],
                             memory_text="fact.py contains VALUE = 7")
    assert result["model_input_tokens"] == ids.shape[1]
    assert result["memory_tokens"] == len(tokenizer.encode("fact.py contains VALUE = 7"))


def test_misconception_records_state_a_real_single_identifier_swap(tmp_path):
    from latent_study.records import generate_family_records

    corpus = tmp_path / "family-corpus"
    corpus.mkdir()
    (corpus / "facts.py").write_text(
        "def alpha():\n    return 1\n\ndef beta():\n    return 2\n", encoding="utf-8")
    manifest = build_manifest(corpus)
    records = generate_family_records(manifest["units"], manifest["corpus_hash"],
                                      n_actions=2, search=CodingTools(corpus, manifest["units"]))
    misconception = next(record for record in records if record.family == "misconception_correction")
    validation = misconception.validation
    assert validation["modified_fact_count"] == 1
    assert validation["incorrect_symbol"] != validation["correct_symbol"]
    assert validation["incorrect_symbol"] in misconception.prompt
    assert validation["correct_symbol"] not in misconception.prompt
