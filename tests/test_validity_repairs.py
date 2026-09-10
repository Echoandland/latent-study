from types import SimpleNamespace

import torch

from latent_study.agent import ranking_state_messages, study_state_messages
from latent_study.candidates import sample_base_actions
from latent_study.corpus import build_manifest
from latent_study.latent import SoftPrefixLM
from latent_study.records import generate_relation_records
from latent_study.rewards import visible_groups
from latent_study.schema import EvidenceGroup, EvidenceSpan, StudyRecord, ToolAction
from latent_study.search import CodingTools, SearchLimits
from latent_study.config import load_config

from test_latent import ToyLM, ToyTokenizer


def test_conservative_relation_resolution_excludes_ambiguous_receivers(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    (root / "a.py").write_text(
        "def duplicate():\n    return 1\n\n"
        "def caller(items):\n    items.append(1)\n    return duplicate()\n")
    (root / "b.py").write_text("def duplicate():\n    return 2\n\nclass C:\n    def append(self, x):\n        return x\n")
    manifest = build_manifest(root)
    accepted = manifest["verified_relations"]
    assert any(r["resolution_rule"] == "direct_local_function" and r["callee_path"] == "a.py" for r in accepted)
    assert not any(r["callee_symbol"] == "append" for r in accepted)
    assert manifest["relation_audit"]["excluded_by_reason"]["unresolved_receiver_call"] >= 1
    records = generate_relation_records(manifest, manifest["corpus_hash"],
                                        search=CodingTools(root, manifest["units"]))
    assert all(r.validation["resolution_rule"] in {"direct_local_function", "from_import_symbol",
                                                    "import_alias_attribute", "class_local_method"}
               for r in records)


def test_atomic_evidence_late_in_long_function_is_visible(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    filler = "".join(f"    value_{i} = {i}  # filler filler filler filler\n" for i in range(80))
    (root / "long.py").write_text("def target():\n    return 7\n\ndef caller():\n" + filler + "    return target()\n")
    manifest = build_manifest(root); relation = manifest["verified_relations"][0]
    assert relation["call_span"]["start_line"] > 40
    span_data = relation["call_span"]
    caller = next(u for u in manifest["units"] if u.unit_id == relation["caller_unit_id"])
    span = EvidenceSpan(caller.unit_id, span_data["start_line"], span_data["end_line"],
                        span_data["text_hash"], caller.source_path, span_data["text"])
    tools = CodingTools(root, manifest["units"], SearchLimits(max_bytes_per_hit=1600, grep_context_lines=2))
    valid, hits, _ = tools.execute(ToolAction(tool="grep", query=r"target\(\)", path="long.py"))
    assert valid and visible_groups(hits, (EvidenceGroup("late", (span,)),)) == ("late",)


def _record(observation):
    return StudyRecord("r", "navigation", "x.py", "find beta", "OBS=" + observation, (), (), (), (), (),
                       {"method": "uniquely_resolved_ast_call"}, "c", "s", "t", 1,
                       observation_action=ToolAction(tool="read_file", path="x.py", start_line=1, end_line=2))


def test_observation_is_real_tool_role_and_changes_action_logits():
    wrapper = SoftPrefixLM(ToyLM(), ToyTokenizer(), length=2)
    one, two = _record("alpha"), _record("omega")
    messages = study_state_messages(one)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    action = torch.tensor([[7, 8]])
    score_one = wrapper.mean_action_logprob(wrapper.chat_ids(messages), action)
    score_two = wrapper.mean_action_logprob(wrapper.chat_ids(study_state_messages(two)), action)
    assert not torch.equal(score_one, score_two)
    assert any("OBS=alpha" in m["content"] for m in ranking_state_messages(one, "candidate", "A", "B"))


def test_candidate_generation_receives_observation():
    class CapturingTokenizer:
        eos_token_id = 0
        def __init__(self): self.messages = None
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages; return torch.tensor([[1, 2]])
        def decode(self, ids, **kwargs): return '{"tool":"grep","query":"beta","path":"x.py"}'
    class FakeModel:
        device = torch.device("cpu")
        def generate(self, ids, **kwargs): return torch.cat((ids, torch.tensor([[3]])), dim=1)
    tokenizer = CapturingTokenizer()
    actions = sample_base_actions(FakeModel(), tokenizer, _record("seen caller evidence"), n=1, seed=3)
    assert actions[0].tool == "grep"
    assert any(m["role"] == "tool" and "seen caller evidence" in m["content"] for m in tokenizer.messages)


def test_resolved_config_contains_every_execution_control():
    config = load_config("configs/dspy_mvp.json")
    assert config["candidate_generation"]["n"] == 4
    assert config["tools"]["interface"] == "pinned_local_coding_tools_v2"
    assert config["objective"] == {"beta": 1.0, "preference_delta": 0.25, "rank_margin": 1.0,
                                    "query_weight": 1.0, "ranking_weight": 1.0}
    assert config["training"]["gradient_accumulation_steps"] == 1
