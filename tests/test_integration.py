from latent_study.corpus import build_manifest
from latent_study.records import coverage_report, generate_coverage_records, generate_relation_records
from latent_study.replay import SourceReplay


def test_small_coverage_replay_integration(tmp_path):
    corpus = tmp_path / "corpus"; corpus.mkdir()
    (corpus / "one.py").write_text("def one():\n    return two()\n\ndef two():\n    return 2\n")
    (corpus / "three.py").write_text("def three():\n    return 3\n")
    manifest = build_manifest(corpus)
    records = generate_coverage_records(manifest["units"], manifest["corpus_hash"])
    records += generate_relation_records(manifest["units"], manifest["corpus_hash"])
    report = coverage_report(manifest, records)
    assert report["semantic_unit_coverage"]["fraction"] == 1.0
    by_source = {}
    for record in records: by_source.setdefault(record.source_id, []).append(record)
    replay = SourceReplay(seed=2, replay_fraction=.5)
    batches = []
    for source in sorted(by_source):
        replay.add_source(source, by_source[source])
        batches.append(replay.batch(source, 4, 0))
    assert batches[0].current_count == 4 and batches[0].previous_count == 0
    assert batches[-1].current_count == 2 and batches[-1].previous_count == 2
    no_replay = SourceReplay(seed=2, replay_fraction=0)
    for source in sorted(by_source): no_replay.add_source(source, by_source[source])
    matched = no_replay.batch(sorted(by_source)[-1], 4, 0)
    assert len(matched.records) == 4 and matched.current_count == 4


def test_batch_two_replay_balances_the_real_dynamically_growing_bank(tmp_path):
    corpus = tmp_path / "many"; corpus.mkdir()
    for index in range(7):
        (corpus / f"s{index}.py").write_text(f"def unique_{index}():\n    return {index}\n")
    manifest = build_manifest(corpus)
    records = generate_coverage_records(manifest["units"], manifest["corpus_hash"])
    grouped = {}
    for record in records:
        grouped.setdefault(record.source_id, []).append(record)
    replay = SourceReplay(seed=19, replay_fraction=.5)
    sources = sorted(grouped)
    for step, source in enumerate(sources):
        replay.add_source(source, grouped[source])
        batch = replay.batch(source, batch_size=2, step=step)
        assert (batch.current_count, batch.previous_count) == ((2, 0) if step == 0 else (1, 1))
    audit = replay.replay_audit()
    counts = {source: replay.previous_ledger.by_source.get(source, 0) for source in sources[:-1]}
    assert set(counts.values()) == {1}
    assert audit["zero_exposure_eligible_sources"] == []
    assert audit["no_replay_opportunity_sources"] == [sources[-1]]
    assert audit["imbalance"]["population_stdev"] > 0  # includes the justified late-source zero
