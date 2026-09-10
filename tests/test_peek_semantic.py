from dataclasses import replace

from peek import CachePolicy, ContextMap

from fakes import FakeSemanticPeekClient
from latent_study.corpus import build_manifest
from latent_study.peek_baseline import trajectory_from_record
from latent_study.records import generate_coverage_records


def test_semantic_distillation_map_depends_on_human_readable_evidence(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    (root / "a.py").write_text("def alpha():\n    \"\"\"Returns the lunar checksum.\"\"\"\n    return 7\n")
    (root / "b.py").write_text("def beta():\n    return 9\n")
    manifest = build_manifest(root)
    record = generate_coverage_records(manifest["units"], manifest["corpus_hash"])[0]

    def map_for(item):
        policy = CachePolicy(client=FakeSemanticPeekClient(), token_budget=1024,
                             token_counter=lambda text: len(text.split()),
                             cmap=ContextMap("## CONTEXT ROADMAP\n"))
        policy.update(trajectory=trajectory_from_record(item), question=item.prompt)
        return policy.current_map_text

    first = map_for(record)
    span = record.evidence_groups[0].alternatives[0]
    changed_span = replace(span, text=span.text.splitlines()[0] + "  # changed semantic evidence")
    changed_group = replace(record.evidence_groups[0], alternatives=(changed_span,))
    second = map_for(replace(record, evidence_groups=(changed_group,)))
    assert span.text.split("(", 1)[0].split()[-1] in first and "Corpus evidence" in first
    assert first != second
