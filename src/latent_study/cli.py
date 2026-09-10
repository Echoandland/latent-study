from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from .corpus import build_manifest
from .io import write_json, write_jsonl
from .records import (coverage_report, generate_coverage_records,
                      generate_relation_records, independent_probes)
from .serde import record_from_dict, unit_from_dict


def _read_manifest(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["units"] = [unit_from_dict(u) for u in raw["units"]]
    return raw


def _read_records(path):
    return [record_from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def command_audit(args):
    started = time.perf_counter()
    manifest = build_manifest(args.corpus)
    write_json(args.output, manifest)
    print(json.dumps({"output": args.output, **manifest["counts"],
                      "runtime_seconds": time.perf_counter() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, indent=2))


def command_records(args):
    manifest = _read_manifest(args.manifest)
    from .isolation import IsolationError, resolved_under
    if resolved_under(manifest["root"], args.evaluation_root):
        raise IsolationError("refusing to generate study records from the evaluation tree")
    from .search import CorpusSearch, SearchLimits
    search = CorpusSearch(manifest["units"], SearchLimits(args.max_results, args.max_bytes_per_hit,
                                                           args.max_total_bytes))
    records = generate_coverage_records(manifest["units"], manifest["corpus_hash"], seed=args.seed,
                                        n_actions=args.actions, limit=args.limit, search=search)
    records += generate_relation_records(manifest["units"], manifest["corpus_hash"], seed=args.seed,
                                         n_actions=args.actions, limit=args.relation_limit, search=search)
    if not 0 <= args.worker_index < args.num_workers:
        raise SystemExit("worker-index must be in [0, num-workers)")
    records = records[args.worker_index::args.num_workers]
    if args.candidate_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .candidates import replace_actions, sample_base_actions
        tokenizer = AutoTokenizer.from_pretrained(args.candidate_model, revision=args.candidate_revision,
                                                  local_files_only=args.local_files_only)
        model = AutoModelForCausalLM.from_pretrained(args.candidate_model,
                                                     revision=args.candidate_revision,
                                                     device_map=args.candidate_device,
                                                     dtype=args.candidate_dtype,
                                                     local_files_only=args.local_files_only)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval()
        records = [replace_actions(r, sample_base_actions(model, tokenizer, r.prompt, n=args.actions,
                                                          seed=args.seed + i), search)
                   for i, r in enumerate(records)]
    elif not args.deterministic_smoke:
        raise SystemExit("real record generation requires --candidate-model; use --deterministic-smoke only for smoke data")
    if args.shuffle_correspondence:
        from .records import shuffled_correspondence_control
        records = shuffled_correspondence_control(records, seed=args.seed)
    write_jsonl(args.output, records)
    report = coverage_report(manifest, records)
    write_json(args.coverage_report, report)
    write_json(args.probes, independent_probes(manifest["units"], records))
    print(json.dumps({"records": len(records), "output": args.output, "coverage": report}, indent=2))


def command_peek(args):
    from .peek_baseline import study_offline_peek
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
        token_counter = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
        counter_name = f"{args.tokenizer_provenance or args.tokenizer}@{args.tokenizer_revision}"
    elif args.smoke_whitespace_counter:
        token_counter = lambda value: len(value.split())
        counter_name = "whitespace_smoke_approximation"
    else:
        raise SystemExit("pass --tokenizer for a valid PEEK budget, or --smoke-whitespace-counter for smoke only")
    payload = study_offline_peek(_read_records(args.records), args.output,
                                 token_budget=args.token_budget, replay_fraction=args.replay,
                                 seed=args.seed, batch_size=args.batch_size,
                                 updates_per_source=args.updates_per_source,
                                 token_counter=token_counter, counter_name=counter_name)
    print(json.dumps({k: payload[k] for k in ("protocol", "update_count", "serialized_bytes")}, indent=2))


def command_train(args):
    import torch
    from .latent import load_qwen
    from .train import train
    manifest = _read_manifest(args.manifest)
    records = _read_records(args.records)
    from .isolation import validate_study_bank
    validate_study_bank(records, manifest["corpus_hash"])
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    prefix = load_qwen(args.model, length=args.length, dtype=args.dtype,
                       revision=args.model_revision, device=args.device)
    report = train(prefix, records, {u.unit_id: u for u in manifest["units"]}, args.output,
                   corpus_hash=manifest["corpus_hash"], model_id=args.model,
                   learning_rate=args.learning_rate, query_weight=args.query_weight,
                   lambda_rank=args.lambda_rank,
                   delta=args.delta, replay_fraction=args.replay, seed=args.seed,
                   batch_size=args.batch_size, updates_per_source=args.updates_per_source,
                   relevant_label=args.relevant_label, irrelevant_label=args.irrelevant_label)
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


def command_init_latent(args):
    import torch
    from .latent import load_qwen
    torch.manual_seed(args.seed)
    prefix = load_qwen(args.model, length=args.length, dtype=args.dtype,
                       revision=args.model_revision, device=args.device)
    prefix.save(args.output, corpus_hash=args.corpus_hash, model_id=args.model)
    print(json.dumps({"condition": "untrained_random_latent", "length": args.length,
                      "seed": args.seed, "output": args.output,
                      "serialized_bytes": Path(args.output).stat().st_size}, indent=2))


def command_expertise(args):
    from .metrics import expertise
    points = []
    for item in args.point:
        budget, score = item.split(":", 1)
        points.append((float(budget), float(score)))
    print(json.dumps({"expertise": expertise(points), "anchor_tokens": 3000}))


def command_replay_report(args):
    import math
    from .replay import SourceReplay
    records = _read_records(args.records)
    by_source = {}
    for record in records: by_source.setdefault(record.source_id, []).append(record)
    replay = SourceReplay(args.seed, args.replay)
    batches = []
    for source in sorted(by_source):
        replay.add_source(source, by_source[source])
        current_slots = args.batch_size if len(replay.bank) == 1 else args.batch_size - args.batch_size // 2
        shard_steps = max(args.updates_per_source, math.ceil(len(by_source[source]) / current_slots))
        for step in range(shard_steps):
            batch = replay.batch(source, args.batch_size, step)
            batches.append({"source": source, "step": step, "current": batch.current_count,
                            "previous": batch.previous_count,
                            "record_ids": [r.record_id for r in batch.records]})
    report = {"replay_fraction": args.replay, "compute_matched_updates": len(batches) * args.batch_size,
              "batches": batches, "exposure_by_record": replay.ledger.by_record,
              "exposure_by_source": replay.ledger.by_source,
              "control_note": "replay=0 fills all matched slots from the current source"}
    write_json(args.output, report); print(json.dumps(report, indent=2))


def command_smoke_eval(args):
    from .evaluation import tool_smoke
    from .search import CorpusSearch, SearchLimits
    manifest = _read_manifest(args.manifest); records = _read_records(args.records)
    report = tool_smoke(records, CorpusSearch(manifest["units"], SearchLimits(
        args.max_results, args.max_bytes_per_hit, args.max_total_bytes)))
    write_json(args.output, report); print(json.dumps(report, indent=2))


def parser():
    root = argparse.ArgumentParser(prog="latent-study")
    sub = root.add_subparsers(required=True)
    audit = sub.add_parser("audit-corpus")
    audit.add_argument("--corpus", required=True); audit.add_argument("--output", required=True)
    audit.set_defaults(func=command_audit)
    records = sub.add_parser("generate-records")
    records.add_argument("--manifest", required=True); records.add_argument("--output", required=True)
    records.add_argument("--evaluation-root", default="data/evaluation")
    records.add_argument("--coverage-report", required=True); records.add_argument("--probes", required=True)
    records.add_argument("--candidate-model"); records.add_argument("--local-files-only", action="store_true")
    records.add_argument("--candidate-revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    records.add_argument("--candidate-device", default="cuda:0")
    records.add_argument("--candidate-dtype", default="auto")
    records.add_argument("--deterministic-smoke", action="store_true")
    records.add_argument("--shuffle-correspondence", action="store_true")
    records.add_argument("--seed", type=int, default=0); records.add_argument("--actions", type=int, default=4)
    records.add_argument("--num-workers", type=int, default=1)
    records.add_argument("--worker-index", type=int, default=0)
    records.add_argument("--limit", type=int); records.add_argument("--relation-limit", type=int, default=32)
    records.add_argument("--max-results", type=int, default=5)
    records.add_argument("--max-bytes-per-hit", type=int, default=1600)
    records.add_argument("--max-total-bytes", type=int, default=6000)
    records.set_defaults(func=command_records)
    peek = sub.add_parser("peek-study")
    peek.add_argument("--records", required=True); peek.add_argument("--output", required=True)
    peek.add_argument("--token-budget", type=int, choices=(64, 1024), required=True)
    peek.add_argument("--tokenizer"); peek.add_argument("--tokenizer-revision",
        default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    peek.add_argument("--tokenizer-provenance", help="canonical model ID when --tokenizer is a local snapshot")
    peek.add_argument("--smoke-whitespace-counter", action="store_true")
    peek.add_argument("--replay", type=float, choices=(0.0, 0.5), default=0.5)
    peek.add_argument("--seed", type=int, default=0); peek.add_argument("--batch-size", type=int, default=4)
    peek.add_argument("--updates-per-source", type=int, default=2); peek.set_defaults(func=command_peek)
    train_p = sub.add_parser("train-latent")
    train_p.add_argument("--manifest", required=True); train_p.add_argument("--records", required=True)
    train_p.add_argument("--model", required=True); train_p.add_argument("--output", required=True)
    train_p.add_argument("--model-revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    train_p.add_argument("--device", default=None, help="default: cuda:0 when available, else cpu")
    train_p.add_argument("--report", required=True); train_p.add_argument("--length", type=int, default=64)
    train_p.add_argument("--dtype", default="bfloat16"); train_p.add_argument("--learning-rate", type=float, default=1e-3)
    train_p.add_argument("--query-weight", type=float, default=1.0)
    train_p.add_argument("--lambda-rank", type=float, default=1.0); train_p.add_argument("--delta", type=float, default=.25)
    train_p.add_argument("--replay", type=float, choices=(0.0, .5), default=.5)
    train_p.add_argument("--seed", type=int, default=0); train_p.add_argument("--batch-size", type=int, default=2)
    train_p.add_argument("--updates-per-source", type=int, default=2)
    train_p.add_argument("--relevant-label", default="A"); train_p.add_argument("--irrelevant-label", default="B")
    train_p.set_defaults(func=command_train)
    init_p = sub.add_parser("init-latent")
    init_p.add_argument("--model", required=True); init_p.add_argument("--output", required=True)
    init_p.add_argument("--corpus-hash", required=True); init_p.add_argument("--length", type=int, default=64)
    init_p.add_argument("--seed", type=int, default=17); init_p.add_argument("--dtype", default="bfloat16")
    init_p.add_argument("--device", default=None); init_p.add_argument("--model-revision",
        default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    init_p.set_defaults(func=command_init_latent)
    metric = sub.add_parser("expertise")
    metric.add_argument("--point", action="append", required=True, help="generated_tokens:score")
    metric.set_defaults(func=command_expertise)
    replay = sub.add_parser("replay-report")
    replay.add_argument("--records", required=True); replay.add_argument("--output", required=True)
    replay.add_argument("--replay", type=float, choices=(0.0, .5), default=.5)
    replay.add_argument("--seed", type=int, default=0); replay.add_argument("--batch-size", type=int, default=4)
    replay.add_argument("--updates-per-source", type=int, default=2); replay.set_defaults(func=command_replay_report)
    smoke = sub.add_parser("smoke-eval")
    smoke.add_argument("--manifest", required=True); smoke.add_argument("--records", required=True)
    smoke.add_argument("--output", required=True); smoke.add_argument("--max-results", type=int, default=5)
    smoke.add_argument("--max-bytes-per-hit", type=int, default=1600)
    smoke.add_argument("--max-total-bytes", type=int, default=6000); smoke.set_defaults(func=command_smoke_eval)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
