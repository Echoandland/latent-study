from __future__ import annotations

import ast
from dataclasses import replace
import random
import re
from typing import Iterable

from .io import canonical_hash, sha256_text
from .rewards import RewardConfig, score_action, visible_groups
from .schema import CorpusUnit, EvidenceGroup, EvidenceSpan, StudyRecord, ToolAction
from .search import CodingTools, CorpusSearch


def _span(unit: CorpusUnit, start: int, end: int, text: str) -> EvidenceSpan:
    return EvidenceSpan(unit.unit_id, start, end, sha256_text(text), unit.source_path, text)


def atomic_definition_span(unit: CorpusUnit) -> EvidenceSpan | None:
    """Smallest bounded declaration/assignment/paragraph that establishes the fact."""
    lines = unit.text.splitlines()
    if not lines:
        return None
    relative_start, relative_end = 1, 1
    if unit.kind in {"function", "class"}:
        try:
            node = ast.parse(unit.text).body[0]
            body = getattr(node, "body", ())
            relative_end = max(1, int(getattr(body[0], "lineno", 2)) - 1) if body else 1
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                relative_end = min(int(getattr(body[0], "end_lineno", relative_end)), 13)
        except (SyntaxError, IndexError):
            relative_end = 1
    elif unit.kind == "module_section":
        try:
            node = next(n for n in ast.parse(unit.text).body
                        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)))
            relative_start = int(node.lineno); relative_end = int(getattr(node, "end_lineno", node.lineno))
        except (SyntaxError, StopIteration):
            return None
    elif unit.kind in {"section", "file"}:
        # Heading plus the first non-empty paragraph, bounded to keep it exposeable.
        end = 1
        seen_content = bool(lines[0].strip() and not lines[0].lstrip().startswith("#"))
        for index, line in enumerate(lines[1:13], 2):
            if not line.strip() and seen_content: break
            seen_content = seen_content or bool(line.strip())
            end = index
        relative_end = end
    text = "\n".join(lines[relative_start - 1:relative_end])
    if not text.strip(): return None
    return _span(unit, unit.start_line + relative_start - 1,
                 unit.start_line + relative_end - 1, text)


def _span_from_relation(unit: CorpusUnit, value: dict) -> EvidenceSpan:
    return _span(unit, int(value["start_line"]), int(value["end_line"]), value["text"])


def _verified_negative(unit: CorpusUnit, ordered: tuple[CorpusUnit, ...], forbidden: Iterable[str] = ()) -> CorpusUnit | None:
    names = set(unit.definitions); forbidden_terms = {x for x in forbidden if x}
    for candidate in ordered:
        if candidate.unit_id == unit.unit_id or not candidate.definitions: continue
        if not names.isdisjoint(candidate.definitions): continue
        hay = candidate.text
        if any(re.search(rf"\b{re.escape(term)}\b", hay) for term in forbidden_terms): continue
        return candidate
    return None


def _actions(unit: CorpusUnit, span: EvidenceSpan, negative: CorpusUnit | None, n: int) -> tuple[ToolAction, ...]:
    actions = [
        ToolAction(query=re.escape(unit.name.split(".")[-1]), max_results=5, tool="grep", path=unit.source_path),
        ToolAction(tool="read_file", path=unit.source_path, start_line=span.start_line, end_line=span.end_line),
        ToolAction(tool="glob", pattern=unit.source_path, max_results=5),
    ]
    if negative is not None: actions.append(ToolAction(query=re.escape(negative.name.split(".")[-1]), tool="grep"))
    return tuple(actions[:n])


def _outcomes(actions, search, groups, previsible=()):
    return tuple(score_action(action, *search.execute(action), groups,
                              RewardConfig(), previsible_group_ids=previsible) for action in actions)


def _group_exposeable(search: CodingTools, group: EvidenceGroup) -> bool:
    """Prove visibility by executing a legal action through final rendering."""
    for span in group.alternatives:
        action = ToolAction(tool="read_file", path=span.source_path,
                            start_line=span.start_line, end_line=span.end_line)
        valid, hits, observation = search.execute(action)
        header = f"{span.source_path}:{span.start_line}-{span.end_line}\n"
        if (valid and header in observation and group.group_id in visible_groups(hits, (group,))):
            return True
    return False


def _record(unit, family, prompt, group, negative, corpus_hash, seed, template, search, n_actions,
            *, validation, observation="", observation_action=None):
    actions = _actions(unit, group.alternatives[0], negative, n_actions)
    return StudyRecord(
        record_id=f"rec_{canonical_hash({'unit': unit.unit_id, 'template': template, 'seed': seed})[:20]}",
        family=family, source_id=unit.source_path, prompt=prompt, observation=observation,
        evidence_groups=(group,), positive_chunk_ids=(unit.unit_id,),
        verified_negative_chunk_ids=((negative.unit_id,) if negative else ()), candidate_actions=actions,
        outcomes=_outcomes(actions, search, (group,)), validation=validation,
        corpus_hash=corpus_hash, source_hash=unit.source_hash, template_id=template,
        random_seed=seed, observation_action=observation_action)


def generate_coverage_records(units: Iterable[CorpusUnit], corpus_hash: str, *, seed: int = 0,
                              n_actions: int = 4, limit: int | None = None,
                              search: CodingTools | None = None) -> list[StudyRecord]:
    ordered = tuple(sorted(units, key=lambda u: u.unit_id)); search = search or CorpusSearch(ordered)
    records = []
    for unit in ordered:
        if limit is not None and len(records) >= limit: break
        if not unit.definitions: continue
        span = atomic_definition_span(unit)
        if span is None or len(span.text.encode("utf-8")) > search.limits.max_bytes_per_hit: continue
        negative = _verified_negative(unit, ordered, unit.definitions)
        if negative is None: continue
        group = EvidenceGroup("definition", (span,))
        if not _group_exposeable(search, group): continue
        records.append(_record(
            unit, "comprehension",
            f"Find and interpret the authoritative declaration of `{unit.name}` in the corpus.",
            group, negative, corpus_hash, seed, "atomic-definition-v2", search, n_actions,
            validation={"method": "atomic_structural_definition", "negative_method": "distinct_definition_no_symbol_overlap",
                        "generator": "deterministic_evidence_first", "evidence_exposeable": True,
                        "base_model_candidates": False}))
    return records


def generate_family_records(units: Iterable[CorpusUnit], corpus_hash: str, *, seed: int = 0,
                            n_actions: int = 4, limit: int | None = None,
                            search: CodingTools | None = None) -> list[StudyRecord]:
    """Minimal deterministic evidence-interpretation and one-fact misconception families."""
    ordered = tuple(sorted(units, key=lambda u: u.unit_id)); search = search or CorpusSearch(ordered)
    out = []
    for index, unit in enumerate(ordered):
        if limit is not None and len(out) >= limit: break
        span = atomic_definition_span(unit)
        if span is None or not unit.definitions or len(span.text.encode()) > search.limits.max_bytes_per_hit: continue
        negative = _verified_negative(unit, ordered, unit.definitions)
        if negative is None: continue
        group = EvidenceGroup("interpreted_fact", (span,))
        if not _group_exposeable(search, group): continue
        if unit.kind == "module_section":
            prompt = f"Which exact corpus statement establishes the configured symbol `{unit.definitions[0]}`?"
            family, template, validator = "evidence_interpretation", "atomic-assignment-interpretation-v1", "ast_assignment_or_import"
        else:
            wrong = negative.definitions[0]
            if wrong == unit.name: continue
            prompt = ("A corpus-navigation claim mistakenly says the declaration at "
                      f"`{unit.source_path}:{span.start_line}` is `{wrong}`. Correct that single identifier "
                      "using the visible declaration and keep the answer grounded in the corpus.")
            family, template, validator = "misconception_correction", "single-name-misconception-v1", "single_verified_name_substitution"
        out.append(_record(unit, family, prompt, group, negative, corpus_hash, seed + index, template,
                           search, n_actions, validation={"method": validator, "evidence_exposeable": True,
                                                         "generator": "deterministic_evidence_first",
                                                         "modified_fact_count": 1 if family == "misconception_correction" else 0,
                                                         **({"incorrect_symbol": wrong,
                                                             "correct_symbol": unit.name}
                                                            if family == "misconception_correction" else {})}))
    return out


def generate_relation_records(manifest_or_units, corpus_hash: str, *, seed: int = 0, n_actions: int = 4,
                              limit: int | None = None, search: CodingTools | None = None,
                              include_navigation: bool = True) -> list[StudyRecord]:
    if isinstance(manifest_or_units, dict):
        ordered = tuple(sorted(manifest_or_units["units"], key=lambda u: u.unit_id))
        relations = manifest_or_units.get("verified_relations", ())
    else:
        ordered, relations = tuple(sorted(manifest_or_units, key=lambda u: u.unit_id)), ()
    search = search or CorpusSearch(ordered); by_id = {u.unit_id: u for u in ordered}; records = []
    for rel_index, relation in enumerate(relations):
        caller, callee = by_id[relation["caller_unit_id"]], by_id[relation["callee_unit_id"]]
        caller_span = _span_from_relation(caller, relation["call_span"])
        callee_span = _span_from_relation(callee, relation["callee_declaration"])
        if max(len(caller_span.text.encode()), len(callee_span.text.encode())) > search.limits.max_bytes_per_hit: continue
        forbidden = (relation["caller_symbol"], relation["callee_symbol"])
        negative = _verified_negative(callee, ordered, forbidden)
        if negative is None: continue
        groups = (EvidenceGroup("caller_call", (caller_span,)), EvidenceGroup("callee_declaration", (callee_span,)))
        if not all(_group_exposeable(search, group) for group in groups): continue
        # Both relation families use the same modeled two-step sequence: the
        # caller span is exposed first, then the agent navigates to the callee.
        # Keeping the caller group pre-visible makes it possible for a legal
        # second action to satisfy all required evidence groups.
        observation_action = ToolAction(tool="read_file", path=caller.source_path,
                                        start_line=caller_span.start_line, end_line=caller_span.end_line)
        valid_observation, observation_hits, observation = search.execute(observation_action)
        previsible = visible_groups(observation_hits, groups) if valid_observation else ()
        if "caller_call" not in previsible: continue
        families = ("relation_induction", "navigation") if include_navigation else ("relation_induction",)
        for family in families:
            if limit is not None and len(records) >= limit: return records
            prompt = (f"Use the visible caller evidence to verify the call relation from "
                      f"`{relation['caller_symbol']}` to `{relation['callee_symbol']}` and locate the callee declaration.")
            if family == "navigation":
                prompt = f"Continue from the observed caller evidence and navigate to `{relation['callee_symbol']}`."
            actions = _actions(callee, callee_span, negative, n_actions)
            outcomes = _outcomes(actions, search, groups, previsible)
            if not any(set(group.group_id for group in groups)
                       <= set(outcome.visible_group_ids) for outcome in outcomes):
                # A relation record is only eligible when its modeled second
                # hop can actually expose every required group after the real
                # tool renderer/truncation.
                continue
            template = f"verified-call-{family}-v2"
            records.append(StudyRecord(
                record_id=f"rec_{canonical_hash({'relation': relation['relation_id'], 'template': template, 'seed': seed})[:20]}",
                family=family, source_id=caller.source_path, prompt=prompt, observation=observation,
                observation_action=observation_action, evidence_groups=groups,
                positive_chunk_ids=tuple(dict.fromkeys((caller.unit_id, callee.unit_id))),
                verified_negative_chunk_ids=(negative.unit_id,), candidate_actions=actions, outcomes=outcomes,
                validation={"method": "uniquely_resolved_ast_call", "resolution_rule": relation["resolution_rule"],
                            "relation_provenance": relation, "evidence_exposeable": True,
                            "previsible_group_ids": list(previsible),
                            "generator": "deterministic_evidence_first"}, corpus_hash=corpus_hash,
                source_hash=caller.source_hash, template_id=template, random_seed=seed + rel_index))
    return records


def coverage_report(manifest: dict, records: list[StudyRecord]) -> dict:
    from collections import Counter
    units = manifest["units"]; positive = {c for r in records for c in r.positive_chunk_ids}
    negative = {c for r in records for c in r.verified_negative_chunk_ids} - positive
    supervised = [u for u in units if u.unit_id in positive]
    docs_all, docs_seen = {u.document_id for u in units}, {u.document_id for u in supervised}
    symbols_all = {s for u in units for s in u.symbols}; symbols_seen = {s for u in supervised for s in u.symbols}
    eligible, evidenced = sum(u.eligible_tokens for u in units), sum(u.eligible_tokens for u in supervised)
    verified_total = manifest.get("relation_audit", {}).get("uniquely_verified", 0)
    relation_ids = {r.validation.get("relation_provenance", {}).get("relation_id") for r in records}
    relation_ids.discard(None)
    family_counts = {family: sum(r.family == family for r in records) for family in
                     ("comprehension", "relation_induction", "navigation", "evidence_interpretation", "misconception_correction")}
    validator_counts = Counter(r.validation.get("method", "missing") for r in records)
    source_counts = Counter(r.source_id for r in records)
    evidence_group_counts = Counter(str(len(r.evidence_groups)) for r in records)
    exposeable = sum(bool(r.validation.get("evidence_exposeable")) for r in records)
    return {
        "document_file_coverage": {"covered": len(docs_seen), "eligible": len(docs_all), "fraction": len(docs_seen)/len(docs_all) if docs_all else 0},
        "semantic_unit_coverage": {"covered": len(positive), "eligible": len(units), "fraction": len(positive)/len(units) if units else 0},
        "eligible_token_evidence_coverage": {"covered": evidenced, "eligible": eligible, "fraction": evidenced/eligible if eligible else 0},
        "symbol_coverage": {"covered": len(symbols_seen), "eligible": len(symbols_all), "fraction": len(symbols_seen)/len(symbols_all) if symbols_all else 0},
        "verified_relation_coverage": {"covered": len(relation_ids), "eligible": verified_total,
                                       "fraction": len(relation_ids)/verified_total if verified_total else 0},
        "relation_audit": manifest.get("relation_audit", {}), "task_family_counts": family_counts,
        "validator_counts": dict(sorted(validator_counts.items())),
        "source_record_counts": dict(sorted(source_counts.items())),
        "evidence_group_count_distribution": dict(sorted(evidence_group_counts.items())),
        "atomic_evidence_exposeability": {"exposeable": exposeable, "records": len(records),
                                           "fraction": exposeable/len(records) if records else 0},
        "units_supervised_as_positive_evidence": len(positive), "units_seen_only_as_negatives": len(negative),
        "excluded_units": len(set(u.unit_id for u in units) - positive - negative),
        "excluded_reasons": {"not_selected_or_no_strong_structural_label": len(set(u.unit_id for u in units)-positive-negative)},
        "warning": "Appearance is not evidence of learning; these are exposure/coverage measurements only."}


def independent_probes(units: Iterable[CorpusUnit], training: list[StudyRecord], *, limit: int = 32) -> list[dict]:
    hashes = {sha256_text(r.prompt) for r in training}; probes = []; ordered = tuple(sorted(units, key=lambda u: u.unit_id))
    for unit in ordered:
        span = atomic_definition_span(unit)
        if span is None or not unit.definitions: continue
        negative = _verified_negative(unit, ordered, unit.definitions)
        if negative is None: continue
        prompt = f"Locate the evidence that establishes `{unit.name}`."
        if sha256_text(prompt) in hashes: continue
        probes.append({"probe_id": f"probe_{sha256_text(unit.unit_id + ':probe-v2')[:20]}", "prompt": prompt,
                       "source_id": unit.source_path, "evidence_span": span.__dict__,
                       "gold_chunk_id": unit.unit_id, "verified_negative_chunk_id": negative.unit_id,
                       "provenance": "deduplicated held-out prompt within same corpus",
                       "template_id": "probe-v2"})
        if len(probes) >= limit: break
    return probes


def shuffled_correspondence_control(records: list[StudyRecord], *, seed: int) -> list[StudyRecord]:
    if len(records) < 2: raise ValueError("shuffled control needs at least two records")
    offset = random.Random(seed).randrange(1, len(records)); out = []
    for i, record in enumerate(records):
        source = records[(i + offset) % len(records)]; validation = dict(record.validation)
        validation.update({"negative_control": "shuffled_prompt_evidence_correspondence", "shuffle_seed": seed,
                           "prompt_source_record": source.record_id})
        out.append(replace(record, prompt=source.prompt, observation=source.observation,
                           observation_action=source.observation_action,
                           template_id=record.template_id + "+shuffled", validation=validation))
    return out
