import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latent_study.agent import (Condition, FrozenRootAgent, InferenceBudget,
                                ROOT_SYSTEM_PROMPT, serialize_chat)
from latent_study.artifacts import (ArtifactCompatibilityError, provenance,
                                    read_sidecar, write_sidecar)
from latent_study.corpus import build_manifest
from latent_study.candidates import record_seed
from latent_study.latent import SoftPrefixLM
from latent_study.peek_baseline import QwenPeekClient, study_offline_peek
from latent_study.records import generate_coverage_records
from latent_study.records import generate_relation_records
from latent_study.search import CodingTools, SearchLimits, TOOL_SCHEMA_HASH
from latent_study.train import train
from latent_study.cli import command_merge_records

from test_latent import ToyLM, ToyTokenizer


class CharacterTokenizer:
    eos_token_id = None

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[ord(char) for char in text]], dtype=torch.long))

    def decode(self, ids, **kwargs):
        return "".join(chr(int(value)) for value in ids)


def test_shared_memory_slot_has_exact_role_delimited_boundaries():
    tokenizer = CharacterTokenizer()
    messages = [{"role": "system", "content": ROOT_SYSTEM_PROMPT},
                {"role": "user", "content": "Where is alpha?"}]
    empty = serialize_chat(tokenizer, messages)
    mapped = serialize_chat(tokenizer, messages, memory_text="alpha lives in a.py")
    assert mapped.input_ids[0, :mapped.memory_start].tolist() == empty.input_ids[0, :empty.memory_start].tolist()
    assert mapped.input_ids[0, mapped.memory_end:].tolist() == empty.input_ids[0, empty.memory_end:].tolist()
    assert tokenizer.decode(mapped.input_ids[0, mapped.memory_start:mapped.memory_end]) == "alpha lives in a.py"
    rendered = tokenizer.decode(mapped.input_ids[0])
    assert rendered.index("<|im_start|>system") < rendered.index("<corpus_memory>")
    assert rendered.index("</corpus_memory>") < rendered.index("<|im_start|>user")


def test_soft_prefix_is_spliced_at_memory_slot_not_before_system():
    tokenizer = CharacterTokenizer()
    lm = ToyLM()
    # Character IDs exceed ToyLM's vocabulary, so use a bounded exact tokenizer.
    class Bounded(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[ord(c) % 31 for c in text]], dtype=torch.long))
    bounded = Bounded(); wrapper = SoftPrefixLM(lm, bounded, length=2)
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]
    ids = wrapper.chat_ids(messages)
    boundary = ids._memory_slot_start
    captured = {}
    def capture(_module, args, kwargs):
        captured["embeds"] = kwargs["inputs_embeds"].detach().clone()
    handle = lm.register_forward_pre_hook(capture, with_kwargs=True)
    wrapper.prefill_ids(ids, use_cache=False)
    handle.remove()
    token_embeds = lm.get_input_embeddings()(ids).detach()
    assert boundary > 0
    assert torch.equal(captured["embeds"][:, :boundary], token_embeds[:, :boundary])
    assert torch.equal(captured["embeds"][:, boundary:boundary + 2],
                       wrapper.prefix.detach().unsqueeze(0))
    assert torch.equal(captured["embeds"][:, boundary + 2:], token_embeds[:, boundary:])


def test_artifact_schema_rejects_legacy_and_source_stale(tmp_path):
    legacy = tmp_path / "legacy.jsonl"; legacy.write_text("{}\n")
    with pytest.raises(ArtifactCompatibilityError, match="no provenance sidecar"):
        read_sidecar(legacy)
    metadata = provenance(artifact_type="study_record_bank", resolved_config_hash="cfg",
                          protocol_config_hash="a" * 64,
                          model_id="m", model_revision="r", corpus_hash="c",
                          tool_schema_hash=TOOL_SCHEMA_HASH, command="test", cli_overrides={})
    write_sidecar(legacy, metadata)
    assert read_sidecar(legacy, artifact_type="study_record_bank")["corpus_hash"] == "c"
    metadata["source_tree_hash"] = "old"
    write_sidecar(legacy, metadata)
    with pytest.raises(ArtifactCompatibilityError, match="source-tree hash is stale"):
        read_sidecar(legacy)


def test_candidate_seed_is_worker_assignment_invariant():
    ids = [f"record-{index}" for index in range(12)]
    one_worker = {record_id: record_seed(17, record_id) for record_id in ids}
    shards = [{record_id: record_seed(17, record_id) for record_id in ids[worker::3]}
              for worker in range(3)]
    assert one_worker == {key: value for shard in shards for key, value in shard.items()}


def test_worker_merge_validates_shards_duplicates_and_order(tmp_path):
    paths = []
    for worker, record_id in ((0, "z"), (1, "a")):
        path = tmp_path / f"worker-{worker}.jsonl"
        path.write_text(json.dumps({"record_id": record_id}) + "\n")
        meta = provenance(artifact_type="study_record_bank", resolved_config_hash="same",
                          protocol_config_hash="a" * 64,
                          model_id="m", model_revision="r", corpus_hash="c",
                          tool_schema_hash=TOOL_SCHEMA_HASH, command="generate", cli_overrides={})
        meta["shard"] = {"num_workers": 2, "worker_index": worker}
        write_sidecar(path, meta); paths.append(str(path))
    output = tmp_path / "merged.jsonl"
    command_merge_records(SimpleNamespace(input=paths, expected_workers=2, output=str(output)))
    assert [json.loads(line)["record_id"] for line in output.read_text().splitlines()] == ["a", "z"]
    Path(paths[1]).write_text(json.dumps({"record_id": "z"}) + "\n")
    with pytest.raises(ValueError, match="duplicate record IDs"):
        command_merge_records(SimpleNamespace(input=paths, expected_workers=2,
                                              output=str(tmp_path / "bad.jsonl")))


class RetryingClient(QwenPeekClient):
    def __init__(self, outputs, retries=1):
        self.outputs = list(outputs); self.seen = []
        super().__init__(SimpleNamespace(), SimpleNamespace(), retries=retries)

    def _once(self, messages):
        from peek.core.types import Usage
        self.seen.append(messages)
        self._usage = Usage(3, 2)
        self.stats["model_calls"] += 1; self.stats["input_tokens"] += 3
        self.stats["output_tokens"] += 2
        return self.outputs.pop(0)


def test_peek_distiller_and_cartographer_failures_are_separate_and_schema_guided():
    valid_distiller = json.dumps({"diagnosis": "ok", "item_tags": {}, "cache_candidates": []})
    client = RetryingClient(["not json", valid_distiller])
    assert client.completion([{"role": "user", "content": "Agent trajectory and trace"}]) == valid_distiller
    assert client.stats["distiller"]["malformed"] == 1
    assert "diagnosis" in client.seen[-1][-1]["content"]
    bad_cartographer = RetryingClient(["{}", "{}"])
    with pytest.raises(RuntimeError, match="cartographer failed structured output"):
        bad_cartographer.completion([{"role": "user", "content": "Available Operations"}])
    assert bad_cartographer.stats["cartographer"] == {"calls": 1, "malformed": 2,
                                                       "retries": 1, "failures": 1}


def test_peek_rejects_malformed_cache_candidate_schema():
    malformed = json.dumps({"diagnosis": "ok", "item_tags": {},
                            "cache_candidates": [{"section": "context_roadmap"}]})
    valid = json.dumps({"diagnosis": "ok", "item_tags": {}, "cache_candidates": []})
    client = RetryingClient([malformed, valid], retries=1)
    assert client.completion([{"role": "user", "content": "Agent trajectory"}]) == valid
    assert client.stats["distiller"]["malformed"] == 1


def test_failed_peek_update_writes_non_frozen_diagnostics_and_exits_nonzero(tmp_path):
    corpus = tmp_path / "peek-corpus"; corpus.mkdir()
    (corpus / "a.py").write_text("def alpha():\n    return 1\n\ndef beta():\n    return 2\n")
    manifest = build_manifest(corpus)
    records = generate_coverage_records(manifest["units"], manifest["corpus_hash"], n_actions=2)
    output = tmp_path / "peek.json"
    with pytest.raises(RuntimeError, match="diagnostics written"):
        study_offline_peek(records, output, token_budget=64,
                           client=RetryingClient(["{}"], retries=0),
                           token_counter=lambda text: len(text.split()), counter_name="test")
    diagnostic = json.loads(output.read_text())
    assert diagnostic["status"] == "failed" and diagnostic["phase"] == "study_failed"
    assert diagnostic["client_statistics"]["distiller"]["failures"] == 1
    assert diagnostic["client_statistics"]["malformed_output_rate"] == 1.0


def test_relation_induction_has_a_modeled_caller_then_callee_sequence(tmp_path):
    corpus = tmp_path / "relation-corpus"; corpus.mkdir()
    (corpus / "a.py").write_text(
        "def target():\n    return 1\n\ndef caller():\n    return target()\n\ndef unrelated():\n    return 3\n")
    manifest = build_manifest(corpus)
    records = generate_relation_records(manifest, manifest["corpus_hash"], limit=1,
                                        search=CodingTools(corpus, manifest["units"]))
    assert records and records[0].family == "relation_induction"
    record = records[0]
    assert record.observation_action is not None
    assert record.validation["previsible_group_ids"] == ["caller_call"]
    assert any(set(outcome.visible_group_ids) == {"caller_call", "callee_declaration"}
               for outcome in record.outcomes)


def test_production_train_path_decreases_combined_components_and_freezes_lm(tmp_path):
    torch.manual_seed(0)
    corpus = tmp_path / "corpus"; corpus.mkdir()
    for index, name in enumerate(("alpha", "beta", "gamma", "delta")):
        (corpus / f"{name}.py").write_text(f"def {name}(x):\n    return x + {index}\n")
    manifest = build_manifest(corpus)
    records = generate_coverage_records(manifest["units"], manifest["corpus_hash"], n_actions=4)
    wrapper = SoftPrefixLM(ToyLM(), ToyTokenizer(), length=2)
    report = train(wrapper, records, {u.unit_id: u for u in manifest["units"]}, tmp_path / "z.pt",
                   corpus_hash=manifest["corpus_hash"], model_id="toy", learning_rate=.01,
                   updates_per_source=10, batch_size=2)
    assert report["objective_decrease_checks"] == {"query_decreased": True,
                                                    "rank_decreased": True,
                                                    "combined_decreased": True}
    assert report["prefix_changed"] and report["frozen_lm_bitwise_unchanged"]
    assert report["lm_gradient_buffers"] == 0
    assert report["objective_diagnostics_before"]["record_count"] == len(records) == 4


class LoopTokenizer(CharacterTokenizer):
    eos_token_id = None

    def convert_tokens_to_ids(self, token):
        return 0

    def decode(self, ids, **kwargs):
        marker = int(ids[0])
        if marker == 901:
            return '{"tool":"read_file","path":"fact.py","start_line":1,"end_line":2}'
        return '{"final":"The visible fact.py evidence says VALUE = 7."}'


class LoopModel:
    def __init__(self):
        self.device = torch.device("cpu"); self.config = SimpleNamespace(model_type="toy")
        self.calls = 0

    def generate(self, ids, **kwargs):
        self.calls += 1
        marker = 901 if self.calls == 1 else 902
        return torch.cat((ids, torch.tensor([[marker]])), dim=1)


class LoopPrefix:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer, self.calls = model, tokenizer, 0

    def chat_ids(self, messages, **kwargs):
        value = serialize_chat(self.tokenizer, messages)
        value.input_ids._memory_slot_start = value.memory_start
        return value.input_ids

    def prefill_ids(self, ids, **kwargs):
        return SimpleNamespace(prefix_insertions=1, cache_mode="native")

    def generate_from_state(self, state, **kwargs):
        self.calls += 1
        return torch.tensor([[901 if self.calls == 1 else 902]]), state

    def append_ids(self, state, ids):
        return state


@pytest.mark.parametrize("name,kind", [
    ("no_study", "none"), ("random_latent_L64", "latent"),
    ("trained_latent_L64", "latent"), ("offline_peek_64", "map"),
    ("offline_peek_1024", "map")])
def test_synthetic_root_loop_uses_typed_tool_visible_observation_and_one_memory_slot(tmp_path, name, kind):
    corpus = tmp_path / name; corpus.mkdir()
    (corpus / "fact.py").write_text("VALUE = 7\nDETAIL = 'grounded'\n")
    manifest = build_manifest(corpus)
    tools = CodingTools(corpus, manifest["units"], SearchLimits(max_total_bytes=80))
    tokenizer, model = LoopTokenizer(), LoopModel()
    prefix = LoopPrefix(model, tokenizer) if kind == "latent" else None
    condition = Condition(name, kind, "memory" if kind != "none" else None,
                          InferenceBudget(2, 4, 80), {"max_results": 5})
    agent = FrozenRootAgent(model, tokenizer, tools, condition, prefix_lm=prefix,
                            map_text="fact.py contains project constants" if kind == "map" else "",
                            max_new_tokens_per_turn=1)
    result = agent.run("Use the corpus tool to identify VALUE.")
    assert result["tool_calls"] == 1 and result["invalid_actions"] == 0
    assert result["returned_observation_bytes"] <= 80
    tool_turn = next(turn for turn in result["transcript"] if turn["role"] == "tool")
    assert "fact.py:1-2" in tool_turn["content"] and "VALUE = 7" in tool_turn["content"]
    assert "VALUE = 7" in result["answer"]
    assert result["prefix_insertions"] == (1 if kind == "latent" else 0)
