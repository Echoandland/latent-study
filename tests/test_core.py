import json
from pathlib import Path

import pytest

from latent_study.agent import Condition, InferenceBudget, assert_fair_conditions, parse_tool_actions
from latent_study.corpus import build_manifest
from latent_study.io import sha256_text
from latent_study.isolation import IsolationError, enforce_study_inputs, validate_study_bank
from latent_study.metrics import expertise, studybench_weighted_score
from latent_study.records import coverage_report, generate_coverage_records
from latent_study.replay import SourceReplay
from latent_study.rewards import score_action
from latent_study.schema import EvidenceGroup, EvidenceSpan, ToolAction
from latent_study.search import CorpusSearch, SearchLimits


def make_corpus(tmp_path: Path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.py").write_text("def alpha(x):\n    return beta(x)\n\ndef beta(x):\n    return x + 1\n")
    (root / "b.py").write_text("class Gamma:\n    pass\n")
    return root


def test_deterministic_manifest_records_negatives_rewards_and_replay(tmp_path):
    root = make_corpus(tmp_path)
    one, two = build_manifest(root), build_manifest(root)
    assert one["corpus_hash"] == two["corpus_hash"]
    assert [u.unit_id for u in one["units"]] == [u.unit_id for u in two["units"]]
    by_file = {}
    for unit in one["units"]: by_file.setdefault(unit.source_path, []).append(unit)
    for file_units in by_file.values():
        spans = sorted((u.start_line, u.end_line) for u in file_units)
        assert all(left[1] < right[0] for left, right in zip(spans, spans[1:]))
    records1 = generate_coverage_records(one["units"], one["corpus_hash"], seed=7)
    records2 = generate_coverage_records(two["units"], two["corpus_hash"], seed=7)
    assert [r.to_dict() for r in records1] == [r.to_dict() for r in records2]
    assert all(r.validation["negative_method"] == "distinct_definition_no_symbol_overlap" for r in records1)
    assert all(r.verified_negative_chunk_ids for r in records1)
    validate_study_bank(records1, one["corpus_hash"])
    with pytest.raises(IsolationError): validate_study_bank(records1, "wrong")
    replay1, replay2 = SourceReplay(11, .5), SourceReplay(11, .5)
    grouped = {}
    for record in records1:
        grouped.setdefault(record.source_id, []).append(record)
    for source in sorted(grouped):
        replay1.add_source(source, grouped[source]); replay2.add_source(source, grouped[source])
    last = sorted(grouped)[-1]
    assert [r.record_id for r in replay1.batch(last, 6, 3).records] == [r.record_id for r in replay2.batch(last, 6, 3).records]


def test_visible_evidence_after_truncation_and_group_reward(tmp_path):
    manifest = build_manifest(make_corpus(tmp_path))
    alpha = next(u for u in manifest["units"] if u.name == "alpha")
    group_a = EvidenceGroup("a", (EvidenceSpan(alpha.unit_id, alpha.start_line, alpha.end_line,
                                                sha256_text(alpha.text)),))
    search = CorpusSearch(manifest["units"], SearchLimits(5, 40, 40))
    valid, hits, observation = search.execute(ToolAction("alpha"))
    truncated = score_action(ToolAction("alpha"), valid, hits, observation, (group_a,))
    assert truncated.visible_group_ids == ()
    assert truncated.reward_components["exact_visible_hit"] == 0
    assert truncated.reward_components["first_valid_rank"] == 0
    search = CorpusSearch(manifest["units"], SearchLimits(5, 2000, 4000, grep_context_lines=1))
    valid, hits, observation = search.execute(ToolAction("alpha"))
    complete = score_action(ToolAction("alpha"), valid, hits, observation, (group_a,))
    assert complete.visible_group_ids == ("a",)
    beta = next(u for u in manifest["units"] if u.name == "beta")
    group_b = EvidenceGroup("b", (EvidenceSpan(beta.unit_id, beta.start_line, beta.end_line,
                                                sha256_text(beta.text)),))
    partial = score_action(ToolAction("alpha"), valid, hits, observation, (group_a, group_b))
    assert partial.reward_components["required_group_coverage"] == pytest.approx(1.0)
    with_prior = score_action(ToolAction("alpha"), valid, hits, observation, (group_a, group_b),
                              previsible_group_ids=("b",))
    assert with_prior.reward_components["required_group_coverage"] == pytest.approx(2.0)


def test_isolation_tool_parser_and_fairness(tmp_path):
    corpus, evaluation = tmp_path / "corpus", tmp_path / "evaluation"
    corpus.mkdir(); evaluation.mkdir()
    source = corpus / "x.py"; source.write_text("x=1")
    exam = evaluation / "exam.jsonl"; exam.write_text("{}")
    enforce_study_inputs([source], corpus_root=corpus, evaluation_root=evaluation)
    with pytest.raises(IsolationError):
        enforce_study_inputs([exam], corpus_root=corpus, evaluation_root=evaluation)
    text = 'first {"tool":"grep","query":"alpha","max_results":2}\nthen {"tool":"grep","query":"beta"}'
    assert [a.query for a in parse_tool_actions(text)] == ["alpha", "beta"]
    budget = InferenceBudget(5, 1000, 6000)
    a = Condition("none", "none", None, budget, {"max_results": 5})
    b = Condition("latent", "latent", "z.pt", budget, {"max_results": 5})
    assert_fair_conditions([a, b])
    with pytest.raises(ValueError):
        assert_fair_conditions([a, Condition("bad", "map", None, InferenceBudget(6, 1000, 6000), {"max_results": 5})])
    with pytest.raises(ValueError):
        assert_fair_conditions([a, Condition("bad_revision", "map", None, budget, {"max_results": 5},
                                             model_revision="different")])


def test_official_expertise_worked_example_and_best_so_far():
    assert expertise([(5000, 10), (10000, 20), (20000, 30), (100000, 40)]) == pytest.approx(10.8)
    # A later regression cannot lower p(x), which is defined as best score at <= budget.
    assert expertise([(5000, 10), (10000, 5)]) == pytest.approx(6.0)
    rubric = [{"claim_id": "c1", "claim_type": "core", "weight": 60},
              {"claim_id": "c2", "claim_type": "supporting", "weight": 40}]
    assert studybench_weighted_score(rubric, {"c2"}, core_gate=False) == 40
    assert studybench_weighted_score(rubric, {"c2"}, core_gate=True) == 0


def test_coverage_dimensions(tmp_path):
    manifest = build_manifest(make_corpus(tmp_path))
    records = generate_coverage_records(manifest["units"], manifest["corpus_hash"])
    report = coverage_report(manifest, records)
    assert set(report) >= {"document_file_coverage", "semantic_unit_coverage",
                           "eligible_token_evidence_coverage", "symbol_coverage",
                           "verified_relation_coverage", "units_seen_only_as_negatives"}
