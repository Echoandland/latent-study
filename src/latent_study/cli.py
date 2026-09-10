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
                      generate_family_records, generate_relation_records, independent_probes)
from .serde import record_from_dict, unit_from_dict


def _read_manifest(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["units"] = [unit_from_dict(u) for u in raw["units"]]
    return raw


def _read_records(path):
    return [record_from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _config(args, artifact):
    from .config import load_config, save_resolved
    config = load_config(args.config)
    save_resolved(config, artifact)
    return config


def _pick(cli_value, config, *keys):
    value = config
    for key in keys: value = value[key]
    return value if cli_value is None else cli_value


def command_audit(args):
    config = _config(args, args.output)
    started = time.perf_counter()
    manifest = build_manifest(args.corpus)
    write_json(args.output, manifest)
    print(json.dumps({"output": args.output, **manifest["counts"],
                      "runtime_seconds": time.perf_counter() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, indent=2))


def command_records(args):
    started = time.perf_counter()
    config = _config(args, args.output)
    manifest = _read_manifest(args.manifest)
    from .isolation import IsolationError, resolved_under
    if resolved_under(manifest["root"], args.evaluation_root):
        raise IsolationError("refusing to generate study records from the evaluation tree")
    from .search import CodingTools, SearchLimits
    seed = _pick(args.seed, config, "seed"); actions = _pick(args.actions, config, "candidate_generation", "n")
    limits = SearchLimits(_pick(args.max_results, config, "tools", "max_results"),
                          _pick(args.max_bytes_per_hit, config, "tools", "max_bytes_per_hit"),
                          _pick(args.max_total_bytes, config, "tools", "max_total_bytes"),
                          config["tools"]["grep_context_lines"])
    search = CodingTools(manifest["root"], manifest["units"], limits)
    records = generate_coverage_records(manifest["units"], manifest["corpus_hash"], seed=seed,
                                        n_actions=actions, limit=args.limit, search=search)
    records += generate_family_records(manifest["units"], manifest["corpus_hash"], seed=seed,
                                       n_actions=actions, limit=args.family_limit, search=search)
    records += generate_relation_records(manifest, manifest["corpus_hash"], seed=seed,
                                         n_actions=actions, limit=args.relation_limit, search=search)
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
        records = [replace_actions(r, sample_base_actions(model, tokenizer, r, n=actions,
                                                          seed=seed + i), search)
                   for i, r in enumerate(records)]
    elif not args.deterministic_smoke:
        raise SystemExit("real record generation requires --candidate-model; use --deterministic-smoke only for smoke data")
    if args.shuffle_correspondence:
        from .records import shuffled_correspondence_control
        records = shuffled_correspondence_control(records, seed=seed)
    write_jsonl(args.output, records)
    report = coverage_report(manifest, records)
    outcomes = [outcome for record in records for outcome in record.outcomes]
    action_texts = [json.dumps(outcome.action.to_payload(), sort_keys=True) for outcome in outcomes]
    report["study_bank_audit"] = {
        "records": len(records), "candidate_actions": len(outcomes),
        "valid_action_rate": sum(outcome.valid for outcome in outcomes) / len(outcomes) if outcomes else 0,
        "global_unique_action_rate": len(set(action_texts)) / len(action_texts) if action_texts else 0,
        "within_record_unique_action_rate": (sum(len({json.dumps(o.action.to_payload(), sort_keys=True)
                                                        for o in record.outcomes}) for record in records) /
                                              len(outcomes) if outcomes else 0),
        "preference_pair_rate": (sum(max(o.reward for o in r.outcomes) - min(o.reward for o in r.outcomes)
                                     >= config["objective"]["preference_delta"] for r in records) / len(records)
                                 if records else 0),
        "exact_evidence_hit_rate": (sum(o.reward_components["exact_visible_hit"] > 0 for o in outcomes) /
                                    len(outcomes) if outcomes else 0),
        "required_group_coverage": (sum(len(o.visible_group_ids) / len(r.evidence_groups)
                                         for r in records for o in r.outcomes) / len(outcomes) if outcomes else 0),
        "generation_compute": getattr(locals().get("model", None), "_latent_study_candidate_stats", {}),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    report["resolved_config_hash"] = config["config_hash"]
    write_json(args.coverage_report, report)
    write_json(args.probes, independent_probes(manifest["units"], records))
    print(json.dumps({"records": len(records), "output": args.output, "coverage": report}, indent=2))


def command_peek(args):
    config = _config(args, args.output)
    from .peek_baseline import QwenPeekClient, study_offline_peek
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = args.model or config["model"]["id"]; revision = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, device_map=args.device,
                                                 dtype=config["model"]["dtype"], local_files_only=args.local_files_only)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.eval(); client = QwenPeekClient(model, tokenizer, max_new_tokens=args.max_new_tokens)
    token_counter = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
    counter_name = f"{config['model']['id']}@{revision}"
    records = _read_records(args.records)
    if args.max_records is not None: records = records[:args.max_records]
    payload = study_offline_peek(records, args.output,
                                 token_budget=args.token_budget, client=client,
                                 replay_fraction=_pick(args.replay, config, "replay", "fraction"),
                                 seed=_pick(args.seed, config, "seed"),
                                 batch_size=_pick(args.batch_size, config, "training", "batch_size"),
                                 updates_per_source=_pick(args.updates_per_source, config, "training", "steps_per_source"),
                                 token_counter=token_counter, counter_name=counter_name,
                                 model_provenance=config["model"])
    print(json.dumps({k: payload[k] for k in ("protocol", "update_count", "map_text_tokens", "map_text_bytes", "complete_artifact_bytes")}, indent=2))


def command_train(args):
    config = _config(args, args.output)
    import torch
    from .latent import load_qwen
    from .train import train
    manifest = _read_manifest(args.manifest)
    records = _read_records(args.records)
    if args.max_records is not None: records = records[:args.max_records]
    probes = json.loads(Path(args.probes).read_text(encoding="utf-8")) if args.probes else None
    if probes is not None and args.max_probes is not None: probes = probes[:args.max_probes]
    from .search import CodingTools, SearchLimits
    tool_cfg = config["tools"]
    tools = CodingTools(manifest["root"], manifest["units"], SearchLimits(
        tool_cfg["max_results"], tool_cfg["max_bytes_per_hit"], tool_cfg["max_total_bytes"],
        tool_cfg["grep_context_lines"]))
    from .isolation import validate_study_bank
    validate_study_bank(records, manifest["corpus_hash"])
    seed = _pick(args.seed, config, "seed"); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model_path = args.model or config["model"]["id"]
    prefix = load_qwen(model_path, length=_pick(args.length, config, "latent", "length"),
                       dtype=args.dtype or config["model"]["dtype"], revision=config["model"]["revision"],
                       device=args.device, init_std=config["latent"]["init_std"])
    report = train(prefix, records, {u.unit_id: u for u in manifest["units"]}, args.output,
                   corpus_hash=manifest["corpus_hash"], model_id=config["model"]["id"],
                   learning_rate=_pick(args.learning_rate, config, "training", "learning_rate"),
                   query_weight=_pick(args.query_weight, config, "objective", "query_weight"),
                   lambda_rank=_pick(args.lambda_rank, config, "objective", "ranking_weight"),
                   delta=_pick(args.delta, config, "objective", "preference_delta"),
                   beta=config["objective"]["beta"], rank_margin=config["objective"]["rank_margin"],
                   replay_fraction=_pick(args.replay, config, "replay", "fraction"), seed=seed,
                   batch_size=_pick(args.batch_size, config, "training", "batch_size"),
                   updates_per_source=_pick(args.updates_per_source, config, "training", "steps_per_source"),
                   relevant_label=args.relevant_label, irrelevant_label=args.irrelevant_label,
                   probes=probes, tools=tools,
                   gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"])
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
              "source_exposure_imbalance": replay.source_imbalance(),
              "replay_exposure_by_source": replay.previous_ledger.by_source,
              "replay_source_imbalance": replay.source_imbalance(kind="previous"),
              "control_note": "replay=0 fills all matched slots from the current source"}
    write_json(args.output, report); print(json.dumps(report, indent=2))


def command_smoke_eval(args):
    from .evaluation import tool_smoke
    from .search import CodingTools, SearchLimits
    manifest = _read_manifest(args.manifest); records = _read_records(args.records)
    report = tool_smoke(records, CodingTools(manifest["root"], manifest["units"], SearchLimits(
        args.max_results, args.max_bytes_per_hit, args.max_total_bytes)))
    write_json(args.output, report); print(json.dumps(report, indent=2))


def command_root_agent(args):
    config = _config(args, args.output)
    from .agent import Condition, FrozenRootAgent, InferenceBudget
    from .latent import load_qwen
    from .search import CodingTools, SearchLimits
    manifest = _read_manifest(args.manifest); model_path = args.model or config["model"]["id"]
    latent = None; map_text = ""; memory_kind = "none"
    if args.condition in {"random_latent_L64", "trained_latent_L64"}:
        memory_kind = "latent"
        latent = load_qwen(model_path, length=config["latent"]["length"], dtype=config["model"]["dtype"],
                           revision=config["model"]["revision"], device=args.device,
                           init_std=config["latent"]["init_std"])
        if args.condition == "trained_latent_L64":
            if not args.memory: raise SystemExit("trained latent condition requires --memory")
            latent.load(args.memory, expected_corpus_hash=manifest["corpus_hash"])
        model, tokenizer = latent.model, latent.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=config["model"]["revision"],
                                                  local_files_only=args.local_files_only)
        model = AutoModelForCausalLM.from_pretrained(model_path, revision=config["model"]["revision"],
                                                     device_map=args.device, dtype=config["model"]["dtype"],
                                                     local_files_only=args.local_files_only)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval()
        if args.condition.startswith("offline_peek"):
            if not args.memory: raise SystemExit("PEEK condition requires --memory")
            map_text = json.loads(Path(args.memory).read_text(encoding="utf-8"))["map_text"]; memory_kind = "map"
    tools_cfg, root_cfg = config["tools"], config["root_agent"]
    limits = SearchLimits(tools_cfg["max_results"], tools_cfg["max_bytes_per_hit"],
                          tools_cfg["max_total_bytes"], tools_cfg["grep_context_lines"])
    tools = CodingTools(manifest["root"], manifest["units"], limits)
    condition = Condition(args.condition, memory_kind, args.memory,
                          InferenceBudget(root_cfg["max_tool_calls"], root_cfg["max_output_tokens"],
                                          root_cfg["max_observation_bytes"]), tools_cfg,
                          config["model"]["revision"], manifest["corpus_hash"],
                          decoding=config["decoding"], root_prompt_revision=root_cfg["prompt_revision"])
    agent = FrozenRootAgent(model, tokenizer, tools, condition, prefix_lm=latent, map_text=map_text,
                            max_new_tokens_per_turn=config["decoding"]["max_new_tokens_per_turn"])
    from .io import jsonable
    report = agent.run(args.question); report["condition"] = jsonable(condition)
    report["scope"] = "synthetic corpus-derived prompts only; not official StudyBench"
    write_json(args.output, report); print(json.dumps(report, indent=2))


def parser():
    root = argparse.ArgumentParser(prog="latent-study")
    sub = root.add_subparsers(required=True)
    audit = sub.add_parser("audit-corpus")
    audit.add_argument("--config", default="configs/dspy_mvp.json")
    audit.add_argument("--corpus", required=True); audit.add_argument("--output", required=True)
    audit.set_defaults(func=command_audit)
    records = sub.add_parser("generate-records")
    records.add_argument("--config", default="configs/dspy_mvp.json")
    records.add_argument("--manifest", required=True); records.add_argument("--output", required=True)
    records.add_argument("--evaluation-root", default="data/evaluation")
    records.add_argument("--coverage-report", required=True); records.add_argument("--probes", required=True)
    records.add_argument("--candidate-model"); records.add_argument("--local-files-only", action="store_true")
    records.add_argument("--candidate-revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    records.add_argument("--candidate-device", default="cuda:0")
    records.add_argument("--candidate-dtype", default="auto")
    records.add_argument("--deterministic-smoke", action="store_true")
    records.add_argument("--shuffle-correspondence", action="store_true")
    records.add_argument("--seed", type=int); records.add_argument("--actions", type=int)
    records.add_argument("--num-workers", type=int, default=1)
    records.add_argument("--worker-index", type=int, default=0)
    records.add_argument("--limit", type=int); records.add_argument("--relation-limit", type=int, default=32)
    records.add_argument("--family-limit", type=int, default=32)
    records.add_argument("--max-results", type=int)
    records.add_argument("--max-bytes-per-hit", type=int)
    records.add_argument("--max-total-bytes", type=int)
    records.set_defaults(func=command_records)
    peek = sub.add_parser("peek-study")
    peek.add_argument("--config", default="configs/dspy_mvp.json")
    peek.add_argument("--records", required=True); peek.add_argument("--output", required=True)
    peek.add_argument("--max-records", type=int)
    peek.add_argument("--token-budget", type=int, choices=(64, 1024), required=True)
    peek.add_argument("--model"); peek.add_argument("--device", default="cuda:0")
    peek.add_argument("--local-files-only", action="store_true"); peek.add_argument("--max-new-tokens", type=int, default=384)
    peek.add_argument("--replay", type=float, choices=(0.0, 0.5))
    peek.add_argument("--seed", type=int); peek.add_argument("--batch-size", type=int)
    peek.add_argument("--updates-per-source", type=int); peek.set_defaults(func=command_peek)
    train_p = sub.add_parser("train-latent")
    train_p.add_argument("--config", default="configs/dspy_mvp.json")
    train_p.add_argument("--manifest", required=True); train_p.add_argument("--records", required=True)
    train_p.add_argument("--max-records", type=int)
    train_p.add_argument("--probes")
    train_p.add_argument("--max-probes", type=int)
    train_p.add_argument("--model"); train_p.add_argument("--output", required=True)
    train_p.add_argument("--model-revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    train_p.add_argument("--device", default=None, help="default: cuda:0 when available, else cpu")
    train_p.add_argument("--report", required=True); train_p.add_argument("--length", type=int)
    train_p.add_argument("--dtype"); train_p.add_argument("--learning-rate", type=float)
    train_p.add_argument("--query-weight", type=float)
    train_p.add_argument("--lambda-rank", type=float); train_p.add_argument("--delta", type=float)
    train_p.add_argument("--replay", type=float, choices=(0.0, .5))
    train_p.add_argument("--seed", type=int); train_p.add_argument("--batch-size", type=int)
    train_p.add_argument("--updates-per-source", type=int)
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
    root_agent = sub.add_parser("root-agent-smoke")
    root_agent.add_argument("--config", default="configs/dspy_mvp.json")
    root_agent.add_argument("--manifest", required=True); root_agent.add_argument("--output", required=True)
    root_agent.add_argument("--question", required=True); root_agent.add_argument("--model")
    root_agent.add_argument("--device", default="cuda:0"); root_agent.add_argument("--local-files-only", action="store_true")
    root_agent.add_argument("--condition", required=True,
                            choices=("no_study", "random_latent_L64", "trained_latent_L64",
                                     "offline_peek_64", "offline_peek_1024"))
    root_agent.add_argument("--memory"); root_agent.set_defaults(func=command_root_agent)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
