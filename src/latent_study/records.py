from __future__ import annotations

from dataclasses import replace
import random
from typing import Iterable

from .io import canonical_hash, sha256_text
from .rewards import RewardConfig, score_action
from .schema import CorpusUnit, EvidenceGroup, EvidenceSpan, StudyRecord, ToolAction
from .search import CorpusSearch


def _span(unit: CorpusUnit) -> EvidenceSpan:
    return EvidenceSpan(unit.unit_id, unit.start_line, unit.end_line, sha256_text(unit.text))


def _verified_negative(unit: CorpusUnit, ordered: tuple[CorpusUnit, ...]) -> CorpusUnit | None:
    """Choose an AST-proved different definition, never a similarity pseudo-negative."""
    if not unit.definitions:
        return None
    names = set(unit.definitions)
    candidates = [u for u in ordered if u.unit_id != unit.unit_id and u.definitions
                  and names.isdisjoint(u.definitions)]
    return candidates[0] if candidates else None


def _actions(unit: CorpusUnit, negative: CorpusUnit | None, n: int) -> tuple[ToolAction, ...]:
    actions = [ToolAction(unit.name, 5), ToolAction(f"{unit.source_path} {unit.name}", 5),
               ToolAction(unit.source_path, 5)]
    if negative is not None:
        actions.append(ToolAction(negative.name, 5))
    actions.extend(ToolAction(ref, 5) for ref in unit.references[:2])
    # Legal candidate generation is deterministic in audit/smoke mode. The real-model
    # command replaces this list with N base-model samples before executing all actions.
    return tuple(actions[:n])


def generate_coverage_records(units: Iterable[CorpusUnit], corpus_hash: str, *, seed: int = 0,
                              n_actions: int = 4, limit: int | None = None,
                              search: CorpusSearch | None = None) -> list[StudyRecord]:
    ordered = tuple(sorted(units, key=lambda u: u.unit_id))
    search = search or CorpusSearch(ordered)
    records = []
    for unit in ordered:
        if limit is not None and len(records) >= limit:
            break
        if not unit.definitions:
            continue
        negative = _verified_negative(unit, ordered)
        if negative is None:
            continue
        prompt = (f"Find the authoritative definition of `{unit.name}` in the corpus and expose "
                  "the defining evidence before interpreting it.")
        group = EvidenceGroup("definition", (_span(unit),))
        actions = _actions(unit, negative, n_actions)
        outcomes = []
        for action in actions:
            valid, hits, observation = search.execute(action)
            outcomes.append(score_action(action, valid, hits, observation, (group,), RewardConfig()))
        source_hash = unit.source_hash
        payload = {"unit": unit.unit_id, "template": "definition-v1", "seed": seed}
        records.append(StudyRecord(
            record_id=f"rec_{canonical_hash(payload)[:20]}", family="comprehension",
            source_id=unit.source_path, prompt=prompt, observation="", evidence_groups=(group,),
            positive_chunk_ids=(unit.unit_id,), verified_negative_chunk_ids=(negative.unit_id,),
            candidate_actions=actions, outcomes=tuple(outcomes),
            validation={"method": "python_ast_definition", "negative_method": "distinct_ast_definition",
                        "generator": "deterministic_evidence_first", "base_model_candidates": False},
            corpus_hash=corpus_hash, source_hash=source_hash, template_id="definition-v1",
            random_seed=seed,
        ))
    return records


def generate_relation_records(units: Iterable[CorpusUnit], corpus_hash: str, *, seed: int = 0,
                              n_actions: int = 4, limit: int | None = None,
                              search: CorpusSearch | None = None) -> list[StudyRecord]:
    ordered = tuple(sorted(units, key=lambda u: u.unit_id))
    by_leaf = {}
    for unit in ordered:
        for definition in unit.definitions:
            by_leaf.setdefault(definition.split(".")[-1], unit)
    search = search or CorpusSearch(ordered)
    records = []
    for caller in ordered:
        for subject, relation, target in caller.relations:
            callee = by_leaf.get(target)
            if callee is None or callee.unit_id == caller.unit_id:
                continue
            groups = (EvidenceGroup("caller", (_span(caller),)), EvidenceGroup("callee", (_span(callee),)))
            negative = next((u for u in ordered if u.unit_id not in {caller.unit_id, callee.unit_id}
                             and u.definitions and target not in u.definitions), None)
            if negative is None:
                continue
            prompt = (f"After inspecting `{subject}`, navigate to the authoritative definition of "
                      f"the `{target}` it calls and verify both sides of that relation.")
            observation = caller.text
            actions = tuple([ToolAction(target, 5), ToolAction(f"{target} {callee.source_path}", 5),
                             ToolAction(subject, 5), ToolAction(caller.source_path, 5)][:n_actions])
            outcomes = []
            for action in actions:
                valid, hits, visible = search.execute(action)
                outcomes.append(score_action(action, valid, hits, visible, groups,
                                             previsible_group_ids=("caller",)))
            payload = {"edge": [caller.unit_id, relation, callee.unit_id], "template": "call-navigation-v1", "seed": seed}
            records.append(StudyRecord(
                record_id=f"rec_{canonical_hash(payload)[:20]}", family="navigation",
                source_id=caller.source_path, prompt=prompt, observation=observation,
                evidence_groups=groups, positive_chunk_ids=(caller.unit_id, callee.unit_id),
                verified_negative_chunk_ids=(negative.unit_id,), candidate_actions=actions, outcomes=tuple(outcomes),
                validation={"method": "python_ast_call_edge", "relation": relation,
                            "generator": "deterministic_evidence_first", "base_model_candidates": False},
                corpus_hash=corpus_hash, source_hash=caller.source_hash,
                template_id="call-navigation-v1", random_seed=seed,
            ))
            if limit is not None and len(records) >= limit:
                return records
    return records


def coverage_report(manifest: dict, records: list[StudyRecord]) -> dict:
    units = manifest["units"]
    positive = {chunk for r in records for chunk in r.positive_chunk_ids}
    negative = {chunk for r in records for chunk in r.verified_negative_chunk_ids} - positive
    supervised = [u for u in units if u.unit_id in positive]
    docs_all = {u.document_id for u in units}
    docs_seen = {u.document_id for u in supervised}
    sections_all = {u.unit_id for u in units}
    symbols_all = {s for u in units for s in u.symbols}
    symbols_seen = {s for u in supervised for s in u.symbols}
    rel_all = {(a, b, c) for u in units for a, b, c in u.relations}
    relation_records = sum(r.family == "navigation" for r in records)
    eligible = sum(u.eligible_tokens for u in units)
    evidenced = sum(u.eligible_tokens for u in supervised)
    return {
        "document_file_coverage": {"covered": len(docs_seen), "eligible": len(docs_all),
                                   "fraction": len(docs_seen) / len(docs_all) if docs_all else 0.0},
        "semantic_unit_coverage": {"covered": len(positive), "eligible": len(sections_all),
                                   "fraction": len(positive) / len(sections_all) if sections_all else 0.0},
        "eligible_token_evidence_coverage": {"covered": evidenced, "eligible": eligible,
                                             "fraction": evidenced / eligible if eligible else 0.0},
        "symbol_coverage": {"covered": len(symbols_seen), "eligible": len(symbols_all),
                            "fraction": len(symbols_seen) / len(symbols_all) if symbols_all else 0.0},
        "verified_relation_coverage": {"covered_records": relation_records, "eligible_edges": len(rel_all)},
        "units_supervised_as_positive_evidence": len(positive),
        "units_seen_only_as_negatives": len(negative),
        "excluded_units": len(sections_all - positive - negative),
        "excluded_reasons": {"not_selected_or_no_strong_structural_label": len(sections_all - positive - negative)},
        "warning": "Appearance is not evidence of learning; these are exposure/coverage measurements only.",
    }


def independent_probes(units: Iterable[CorpusUnit], training: list[StudyRecord], *, limit: int = 32) -> list[dict]:
    training_prompts = {sha256_text(r.prompt) for r in training}
    probes = []
    for unit in sorted(units, key=lambda u: u.unit_id):
        if not unit.definitions:
            continue
        prompt = f"Which corpus location establishes the implementation contract for `{unit.name}`?"
        if sha256_text(prompt) in training_prompts:
            continue
        probes.append({"probe_id": f"probe_{sha256_text(unit.unit_id + ':probe-v1')[:20]}",
                       "prompt": prompt, "gold_chunk_id": unit.unit_id,
                       "provenance": "held-out prompt within same corpus", "template_id": "probe-v1"})
        if len(probes) >= limit:
            break
    return probes


def shuffled_correspondence_control(records: list[StudyRecord], *, seed: int) -> list[StudyRecord]:
    """Negative control: permute prompts/observations while preserving coverage and compute."""
    if len(records) < 2:
        raise ValueError("shuffled control needs at least two records")
    offset = random.Random(seed).randrange(1, len(records))
    order = [(i + offset) % len(records) for i in range(len(records))]
    out = []
    for record, source_index in zip(records, order):
        source = records[source_index]
        validation = dict(record.validation)
        validation.update({"negative_control": "shuffled_prompt_evidence_correspondence",
                           "shuffle_seed": seed, "prompt_source_record": source.record_id})
        out.append(replace(record, prompt=source.prompt, observation=source.observation,
                           template_id=record.template_id + "+shuffled", validation=validation))
    return out
