from __future__ import annotations

import argparse
import copy
import json
import resource
import sys
import time
from pathlib import Path

from .corpus import build_manifest
from .io import canonical_hash, write_json, write_jsonl
from .records import (coverage_report, generate_coverage_records,
                      generate_family_records, generate_relation_records, independent_probes)
from .serde import record_from_dict, unit_from_dict


def _read_manifest(path, *, require_current: bool = False, config: dict | None = None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if require_current:
        from .artifacts import validate_provenance
        from .search import TOOL_SCHEMA_HASH
        validate_provenance(raw.get("provenance", {}), artifact_type="corpus_manifest",
                            model_id=config["model"]["id"] if config else None,
                            model_revision=config["model"]["revision"] if config else None,
                            corpus_hash=raw.get("corpus_hash"), tool_schema_hash=TOOL_SCHEMA_HASH)
    raw["units"] = [unit_from_dict(u) for u in raw["units"]]
    return raw


def _read_records(path, *, require_current: bool = False, config: dict | None = None,
                  corpus_hash: str | None = None):
    if require_current:
        from .artifacts import read_sidecar
        from .search import TOOL_SCHEMA_HASH
        read_sidecar(path, artifact_type="study_record_bank",
                     model_id=config["model"]["id"] if config else None,
                     model_revision=config["model"]["revision"] if config else None,
                     corpus_hash=corpus_hash, tool_schema_hash=TOOL_SCHEMA_HASH)
    return [record_from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _config(args, artifact):
    from .config import load_config, save_resolved
    config = load_config(args.config)
    ignored = {"func", "command", "config", "output", "report", "manifest", "records", "probes",
               "coverage_report", "question", "memory", "worker_index"}
    overrides = {key: value for key, value in vars(args).items()
                 if key not in ignored and value is not None}
    base = copy.deepcopy(config)
    base.pop("config_hash", None); base.pop("config_source", None)
    config["cli_overrides"] = overrides
    config["config_hash"] = canonical_hash({"config": base, "cli_overrides": overrides})
    save_resolved(config, artifact)
    return config


def _provenance(config, args, artifact_type: str, corpus_hash: str) -> dict:
    from .artifacts import provenance
    from .search import TOOL_SCHEMA_HASH
    revision = (getattr(args, "model_revision", None)
                or getattr(args, "candidate_revision", None)
                or config["model"]["revision"])
    return provenance(artifact_type=artifact_type, resolved_config_hash=config["config_hash"],
                      model_id=config["model"]["id"], model_revision=revision,
                      corpus_hash=corpus_hash, tool_schema_hash=TOOL_SCHEMA_HASH,
                      command="latent-study " + getattr(args, "command", artifact_type),
                      cli_overrides=config.get("cli_overrides", {}))


def _reject_evaluation_paths(config: dict, paths) -> None:
    from .isolation import IsolationError, resolved_under
    roots = config.get("corpus", {}).get("evaluation_roots", ["data/evaluation"])
    for path in paths:
        if path and any(resolved_under(path, root) for root in roots):
            raise IsolationError(f"study command refuses evaluation-derived path: {Path(path).resolve()}")


def _pick(cli_value, config, *keys):
    value = config
    for key in keys: value = value[key]
    return value if cli_value is None else cli_value


def command_audit(args):
    config = _config(args, args.output)
    from .isolation import IsolationError, resolved_under
    authorized = [Path(root).resolve() for root in config["corpus"]["authorized_roots"]]
    if not any(resolved_under(args.corpus, root) for root in authorized):
        raise IsolationError("corpus is outside every configured authorized root")
    started = time.perf_counter()
    manifest = build_manifest(args.corpus)
    manifest["provenance"] = _provenance(config, args, "corpus_manifest", manifest["corpus_hash"])
    write_json(args.output, manifest)
    print(json.dumps({"output": args.output, **manifest["counts"],
                      "runtime_seconds": time.perf_counter() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, indent=2))


def command_records(args):
    started = time.perf_counter()
    config = _config(args, args.output)
    _reject_evaluation_paths(config, [args.manifest])
    manifest = _read_manifest(args.manifest, require_current=True, config=config)
    from .isolation import IsolationError, enforce_study_inputs
    authorized = next((root for root in config["corpus"]["authorized_roots"]
                       if Path(manifest["root"]).resolve().is_relative_to(Path(root).resolve())), None)
    if authorized is None: raise IsolationError("manifest root is outside configured authorized roots")
    enforce_study_inputs([manifest["root"]], corpus_root=authorized, evaluation_root=args.evaluation_root)
    from .search import CodingTools, SearchLimits
    seed = _pick(args.seed, config, "seed"); actions = _pick(args.actions, config, "candidate_generation", "n")
    limits = SearchLimits(_pick(args.max_results, config, "tools", "max_results"),
                          _pick(args.max_bytes_per_hit, config, "tools", "max_bytes_per_hit"),
                          _pick(args.max_total_bytes, config, "tools", "max_total_bytes"),
                          config["tools"]["grep_context_lines"], config["tools"]["max_query_chars"])
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
        from .candidates import record_seed, replace_actions, sample_base_actions
        candidate_revision = args.candidate_revision or config["model"]["revision"]
        tokenizer = AutoTokenizer.from_pretrained(args.candidate_model, revision=candidate_revision,
                                                  local_files_only=args.local_files_only)
        model = AutoModelForCausalLM.from_pretrained(args.candidate_model,
                                                     revision=candidate_revision,
                                                     device_map=args.candidate_device,
                                                     dtype=args.candidate_dtype,
                                                     local_files_only=args.local_files_only)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval()
        generation = config["candidate_generation"]
        records = [replace_actions(r, sample_base_actions(
            model, tokenizer, r, n=actions,
            seed=record_seed(seed, r.record_id),
            max_new_tokens=generation["max_new_tokens"], do_sample=generation["do_sample"],
            temperature=generation["temperature"], top_p=generation["top_p"]), search)
                   for r in records]
    elif not args.deterministic_smoke:
        raise SystemExit("real record generation requires --candidate-model; use --deterministic-smoke only for smoke data")
    if args.shuffle_correspondence:
        from .records import shuffled_correspondence_control
        records = shuffled_correspondence_control(records, seed=seed)
    write_jsonl(args.output, records)
    from .artifacts import write_sidecar
    bank_meta = _provenance(config, args, "study_record_bank", manifest["corpus_hash"])
    bank_meta["shard"] = {"num_workers": args.num_workers, "worker_index": args.worker_index}
    write_sidecar(args.output, bank_meta)
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
    report["provenance"] = _provenance(config, args, "study_bank_audit", manifest["corpus_hash"])
    write_json(args.coverage_report, report)
    write_json(args.probes, independent_probes(manifest["units"], records))
    write_sidecar(args.probes, _provenance(config, args, "corpus_probe_bank", manifest["corpus_hash"]))
    print(json.dumps({"records": len(records), "output": args.output, "coverage": report}, indent=2))


def command_peek(args):
    config = _config(args, args.output)
    _reject_evaluation_paths(config, [args.records])
    from .peek_baseline import QwenPeekClient, study_offline_peek
    from transformers import AutoModelForCausalLM, AutoTokenizer
    records = _read_records(args.records)
    corpus_hash = records[0].corpus_hash if records else ""
    _read_records(args.records, require_current=True, config=config, corpus_hash=corpus_hash)
    from .isolation import validate_study_bank
    validate_study_bank(records, corpus_hash)
    if args.max_records is not None: records = records[:args.max_records]
    model_path = args.model or config["model"]["id"]
    revision = args.model_revision or config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, device_map=args.device,
                                                 dtype=config["model"]["dtype"], local_files_only=args.local_files_only)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.eval(); client = QwenPeekClient(
        model, tokenizer, max_new_tokens=_pick(args.max_new_tokens, config, "peek", "max_new_tokens"),
        retries=_pick(args.retries, config, "peek", "retries"))
    token_counter = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
    counter_name = f"{config['model']['id']}@{revision}"
    payload = study_offline_peek(records, args.output,
                                 token_budget=args.token_budget, client=client,
                                 replay_fraction=_pick(args.replay, config, "replay", "fraction"),
                                 seed=_pick(args.seed, config, "seed"),
                                 batch_size=_pick(args.batch_size, config, "training", "batch_size"),
                                 updates_per_source=_pick(args.updates_per_source, config, "training", "steps_per_source"),
                                 token_counter=token_counter, counter_name=counter_name,
                                 model_provenance={**config["model"], "revision": revision},
                                 artifact_provenance=_provenance(
                                     config, args, "offline_peek_map", corpus_hash))
    print(json.dumps({k: payload[k] for k in ("protocol", "update_count", "map_text_tokens", "map_text_bytes", "complete_artifact_bytes")}, indent=2))


def command_train(args):
    config = _config(args, args.output)
    _reject_evaluation_paths(config, [args.manifest, args.records, args.probes])
    import torch
    from .latent import load_qwen
    from .train import train
    manifest = _read_manifest(args.manifest, require_current=True, config=config)
    records = _read_records(args.records, require_current=True, config=config,
                            corpus_hash=manifest["corpus_hash"])
    from .isolation import validate_study_bank
    validate_study_bank(records, manifest["corpus_hash"])
    if args.max_records is not None: records = records[:args.max_records]
    if args.probes:
        from .artifacts import read_sidecar
        from .search import TOOL_SCHEMA_HASH
        read_sidecar(args.probes, artifact_type="corpus_probe_bank",
                     model_id=config["model"]["id"], model_revision=config["model"]["revision"],
                     corpus_hash=manifest["corpus_hash"], tool_schema_hash=TOOL_SCHEMA_HASH)
    probes = json.loads(Path(args.probes).read_text(encoding="utf-8")) if args.probes else None
    if probes is not None and args.max_probes is not None: probes = probes[:args.max_probes]
    from .search import CodingTools, SearchLimits
    tool_cfg = config["tools"]
    tools = CodingTools(manifest["root"], manifest["units"], SearchLimits(
        tool_cfg["max_results"], tool_cfg["max_bytes_per_hit"], tool_cfg["max_total_bytes"],
        tool_cfg["grep_context_lines"], tool_cfg["max_query_chars"]))
    seed = _pick(args.seed, config, "seed"); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model_path = args.model or config["model"]["id"]
    revision = args.model_revision or config["model"]["revision"]
    prefix = load_qwen(model_path, length=_pick(args.length, config, "latent", "length"),
                       dtype=args.dtype or config["model"]["dtype"], revision=revision,
                       device=args.device, init_std=config["latent"]["init_std"])
    report = train(prefix, records, {u.unit_id: u for u in manifest["units"]}, args.output,
                   corpus_hash=manifest["corpus_hash"], model_id=config["model"]["id"],
                   learning_rate=_pick(args.learning_rate, config, "training", "learning_rate"),
                   weight_decay=config["training"]["weight_decay"],
                   optimizer_name=config["training"]["optimizer"],
                   query_weight=_pick(args.query_weight, config, "objective", "query_weight"),
                   lambda_rank=_pick(args.lambda_rank, config, "objective", "ranking_weight"),
                   delta=_pick(args.delta, config, "objective", "preference_delta"),
                   beta=config["objective"]["beta"], rank_margin=config["objective"]["rank_margin"],
                   replay_fraction=_pick(args.replay, config, "replay", "fraction"), seed=seed,
                   batch_size=_pick(args.batch_size, config, "training", "batch_size"),
                   updates_per_source=_pick(args.updates_per_source, config, "training", "steps_per_source"),
                   relevant_label=args.relevant_label, irrelevant_label=args.irrelevant_label,
                   probes=probes, tools=tools,
                   gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
                   model_revision=revision, artifact_provenance=_provenance(
                       config, args, "trained_latent", manifest["corpus_hash"]))
    report["provenance"] = _provenance(config, args, "latent_training_report", manifest["corpus_hash"])
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


def command_init_latent(args):
    config = _config(args, args.output)
    import torch
    from .latent import load_qwen
    torch.manual_seed(args.seed)
    revision = args.model_revision or config["model"]["revision"]
    prefix = load_qwen(args.model, length=args.length, dtype=args.dtype,
                       revision=revision, device=args.device, init_std=config["latent"]["init_std"])
    prefix.save(args.output, corpus_hash=args.corpus_hash, model_id=config["model"]["id"],
                model_revision=revision, seed=args.seed,
                provenance=_provenance(config, args, "random_latent", args.corpus_hash))
    print(json.dumps({"condition": "untrained_random_latent", "length": args.length,
                      "seed": args.seed, "output": args.output,
                      "deployable_latent_tensor_bytes": prefix.prefix.numel() * prefix.prefix.element_size(),
                      "latent_checkpoint_artifact_bytes": Path(args.output).stat().st_size}, indent=2))


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
    config = _config(args, args.output)
    _reject_evaluation_paths(config, [args.records])
    records = _read_records(args.records)
    corpus_hash = records[0].corpus_hash if records else ""
    _read_records(args.records, require_current=True, config=config, corpus_hash=corpus_hash)
    from .isolation import validate_study_bank
    validate_study_bank(records, corpus_hash)
    replay_fraction = _pick(args.replay, config, "replay", "fraction")
    seed = _pick(args.seed, config, "seed")
    batch_size = _pick(args.batch_size, config, "training", "batch_size")
    updates = _pick(args.updates_per_source, config, "training", "steps_per_source")
    by_source = {}
    for record in records: by_source.setdefault(record.source_id, []).append(record)
    replay = SourceReplay(seed, replay_fraction)
    batches = []
    for source in sorted(by_source):
        replay.add_source(source, by_source[source])
        current_slots = batch_size if len(replay.bank) == 1 else batch_size - batch_size // 2
        shard_steps = max(updates, math.ceil(len(by_source[source]) / current_slots))
        for step in range(shard_steps):
            batch = replay.batch(source, batch_size, step)
            batches.append({"source": source, "step": step, "current": batch.current_count,
                            "previous": batch.previous_count,
                            "record_ids": [r.record_id for r in batch.records]})
    report = {"replay_fraction": replay_fraction, "compute_matched_updates": len(batches) * batch_size,
              "batches": batches, "exposure_by_record": replay.ledger.by_record,
              "exposure_by_source": replay.ledger.by_source,
              "source_exposure_imbalance": replay.source_imbalance(),
              "replay_exposure_by_source": replay.previous_ledger.by_source,
              "replay_source_imbalance": replay.source_imbalance(kind="previous"),
              "replay_audit": replay.replay_audit(),
              "control_note": "replay=0 fills all matched slots from the current source",
              "provenance": _provenance(config, args, "replay_report", corpus_hash)}
    write_json(args.output, report); print(json.dumps(report, indent=2))


def command_merge_records(args):
    from .artifacts import read_sidecar, write_sidecar
    if len(args.input) != args.expected_workers:
        raise ValueError("missing shard paths: input count differs from expected workers")
    rows, metas, indexes = [], [], set()
    for path in args.input:
        meta = read_sidecar(path, artifact_type="study_record_bank")
        shard = meta.get("shard", {})
        if shard.get("num_workers") != args.expected_workers:
            raise ValueError("worker-count mismatch")
        index = shard.get("worker_index")
        if not isinstance(index, int) or index in indexes:
            raise ValueError("duplicate or invalid worker index")
        indexes.add(index); metas.append(meta)
        rows.extend(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line)
    if indexes != set(range(args.expected_workers)):
        raise ValueError("missing worker shard index")
    comparable = ("resolved_config_hash", "model_id", "model_revision", "corpus_hash", "tool_schema_hash")
    if any(any(meta[field] != metas[0][field] for field in comparable) for meta in metas[1:]):
        raise ValueError("worker shard corpus/config/model/tool mismatch")
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate record IDs across worker shards")
    rows.sort(key=lambda row: row["record_id"])
    write_jsonl(args.output, rows)
    merged = dict(metas[0]); merged["shard"] = {"num_workers": 1, "worker_index": 0,
                                                "merged_from_workers": args.expected_workers}
    merged["command"] = "latent-study merge-records"
    write_sidecar(args.output, merged)
    print(json.dumps({"records": len(rows), "workers": args.expected_workers,
                      "output": args.output}, indent=2))


def command_validate_conditions(args):
    """Fail-closed preflight for the complete five-condition evaluation."""
    config = _config(args, args.output)
    manifest = _read_manifest(args.manifest, require_current=True, config=config)
    from .agent import Condition, InferenceBudget, assert_fair_conditions
    from .artifacts import validate_json_artifact, validate_provenance
    from .isolation import assert_frozen_payload
    from .search import TOOL_SCHEMA_HASH
    import torch
    corpus_hash, model_cfg = manifest["corpus_hash"], config["model"]
    latent_payloads = {}
    for name, path in (("random_latent_L64", args.random_latent),
                       ("trained_latent_L64", args.trained_latent)):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert_frozen_payload(payload)
        validate_provenance(payload.get("provenance", {}),
                            artifact_type="random_latent" if name.startswith("random") else "trained_latent",
                            model_id=model_cfg["id"], model_revision=model_cfg["revision"],
                            corpus_hash=corpus_hash, tool_schema_hash=TOOL_SCHEMA_HASH)
        if (payload.get("length") != config["latent"]["length"]
                or payload["prefix"].shape[0] != config["latent"]["length"]
                or payload.get("hidden_size") != payload["prefix"].shape[1]):
            raise ValueError(f"{name} latent shape metadata mismatch")
        if name.startswith("random") and not isinstance(payload.get("seed"), int):
            raise ValueError("random latent must carry its reproducible initializer seed")
        latent_payloads[name] = {k: v for k, v in payload.items() if k != "prefix"}
    peek_payloads = {}
    for name, path, budget in (("offline_peek_64", args.peek64, 64),
                               ("offline_peek_1024", args.peek1024, 1024)):
        payload = validate_json_artifact(path, artifact_type="offline_peek_map",
                                         model_id=model_cfg["id"], model_revision=model_cfg["revision"],
                                         corpus_hash=corpus_hash, tool_schema_hash=TOOL_SCHEMA_HASH)
        assert_frozen_payload(payload)
        if payload.get("token_budget") != budget:
            raise ValueError(f"{name} budget mismatch")
        if payload.get("token_counter") != f"{model_cfg['id']}@{model_cfg['revision']}":
            raise ValueError(f"{name} tokenizer/revision mismatch")
        peek_payloads[name] = {"token_budget": budget, "map_text_tokens": payload.get("map_text_tokens")}
    tools_cfg, root_cfg = config["tools"], config["root_agent"]
    budget = InferenceBudget(root_cfg["max_tool_calls"], root_cfg["max_output_tokens"],
                             root_cfg["max_observation_bytes"])
    conditions = [Condition(name, kind, path, budget, tools_cfg, model_cfg["revision"], corpus_hash,
                            decoding={**config["decoding"], "seed": config["seed"]},
                            root_prompt_revision=root_cfg["prompt_revision"],
                            allow_full_recompute_fallback=args.acknowledge_full_recompute_fallback)
                  for name, kind, path in (
                      ("no_study", "none", None), ("random_latent_L64", "latent", args.random_latent),
                      ("trained_latent_L64", "latent", args.trained_latent),
                      ("offline_peek_64", "map", args.peek64),
                      ("offline_peek_1024", "map", args.peek1024))]
    assert_fair_conditions(conditions)
    report = {"status": "compatible", "phase": "preflight_only_no_evaluation_started",
              "conditions": [condition.name for condition in conditions],
              "latent_artifacts": latent_payloads, "peek_artifacts": peek_payloads,
              "full_recompute_fallback_acknowledged": args.acknowledge_full_recompute_fallback,
              "provenance": _provenance(config, args, "evaluation_preflight", corpus_hash)}
    write_json(args.output, report); print(json.dumps(report, indent=2))


def command_smoke_eval(args):
    config = _config(args, args.output)
    from .evaluation import tool_smoke
    from .search import CodingTools, SearchLimits
    manifest = _read_manifest(args.manifest, require_current=True, config=config)
    records = _read_records(args.records, require_current=True, config=config,
                            corpus_hash=manifest["corpus_hash"])
    from .isolation import validate_study_bank
    validate_study_bank(records, manifest["corpus_hash"])
    report = tool_smoke(records, CodingTools(manifest["root"], manifest["units"], SearchLimits(
        args.max_results, args.max_bytes_per_hit, args.max_total_bytes)))
    report["provenance"] = _provenance(config, args, "tool_smoke_report", manifest["corpus_hash"])
    write_json(args.output, report); print(json.dumps(report, indent=2))


def command_root_agent(args):
    config = _config(args, args.output)
    if not args.acknowledge_full_recompute_fallback:
        raise RuntimeError("root-agent Qwen run requires --acknowledge-full-recompute-fallback")
    from .agent import Condition, FrozenRootAgent, InferenceBudget
    from .latent import load_qwen
    from .search import CodingTools, SearchLimits
    manifest = _read_manifest(args.manifest, require_current=True, config=config)
    model_path = args.model or config["model"]["id"]
    latent = None; map_text = ""; memory_kind = "none"
    if args.condition in {"random_latent_L64", "trained_latent_L64"}:
        if not args.memory:
            raise SystemExit(f"{args.condition} requires a frozen seeded --memory artifact")
        import torch
        from .artifacts import validate_provenance
        from .isolation import assert_frozen_payload
        checkpoint_meta = torch.load(args.memory, map_location="cpu", weights_only=True)
        assert_frozen_payload(checkpoint_meta)
        validate_provenance(checkpoint_meta.get("provenance", {}),
                            artifact_type=("random_latent" if args.condition.startswith("random")
                                           else "trained_latent"),
                            model_id=config["model"]["id"], model_revision=config["model"]["revision"],
                            corpus_hash=manifest["corpus_hash"])
        if checkpoint_meta.get("length") != config["latent"]["length"]:
            raise ValueError("latent length mismatch")
        if args.condition.startswith("random") and not isinstance(checkpoint_meta.get("seed"), int):
            raise ValueError("random latent artifact lacks a reproducible seed")
        memory_kind = "latent"
        latent = load_qwen(model_path, length=config["latent"]["length"], dtype=config["model"]["dtype"],
                           revision=config["model"]["revision"], device=args.device,
                           init_std=config["latent"]["init_std"])
        latent_meta = latent.load(args.memory, expected_corpus_hash=manifest["corpus_hash"],
                                  expected_model_id=config["model"]["id"],
                                  expected_model_revision=config["model"]["revision"])
        validate_provenance(latent_meta.get("provenance", {}),
                            artifact_type=("random_latent" if args.condition.startswith("random")
                                           else "trained_latent"),
                            model_id=config["model"]["id"],
                            model_revision=config["model"]["revision"],
                            corpus_hash=manifest["corpus_hash"])
        model, tokenizer = latent.model, latent.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        peek = None
        if args.condition.startswith("offline_peek"):
            if not args.memory: raise SystemExit("PEEK condition requires --memory")
            from .artifacts import validate_json_artifact
            expected_budget = 64 if args.condition.endswith("64") else 1024
            peek = validate_json_artifact(args.memory, artifact_type="offline_peek_map",
                                          model_id=config["model"]["id"],
                                          model_revision=config["model"]["revision"],
                                          corpus_hash=manifest["corpus_hash"])
            from .isolation import assert_frozen_payload
            assert_frozen_payload(peek)
            if peek.get("token_budget") != expected_budget:
                raise ValueError("PEEK map token budget is incompatible with condition")
            if peek.get("token_counter") != f"{config['model']['id']}@{config['model']['revision']}":
                raise ValueError("PEEK tokenizer/model revision mismatch")
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=config["model"]["revision"],
                                                  local_files_only=args.local_files_only)
        model = AutoModelForCausalLM.from_pretrained(model_path, revision=config["model"]["revision"],
                                                     device_map=args.device, dtype=config["model"]["dtype"],
                                                     local_files_only=args.local_files_only)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval()
        if args.condition.startswith("offline_peek"):
            assert peek is not None
            map_text = peek["map_text"]; memory_kind = "map"
    tools_cfg, root_cfg = config["tools"], config["root_agent"]
    limits = SearchLimits(tools_cfg["max_results"], tools_cfg["max_bytes_per_hit"],
                          tools_cfg["max_total_bytes"], tools_cfg["grep_context_lines"],
                          tools_cfg["max_query_chars"])
    tools = CodingTools(manifest["root"], manifest["units"], limits)
    condition = Condition(args.condition, memory_kind, args.memory,
                          InferenceBudget(root_cfg["max_tool_calls"], root_cfg["max_output_tokens"],
                                          root_cfg["max_observation_bytes"]), tools_cfg,
                          config["model"]["revision"], manifest["corpus_hash"],
                          decoding={**config["decoding"], "seed": config["seed"]},
                          root_prompt_revision=root_cfg["prompt_revision"],
                          allow_full_recompute_fallback=args.acknowledge_full_recompute_fallback)
    agent = FrozenRootAgent(model, tokenizer, tools, condition, prefix_lm=latent, map_text=map_text,
                            max_new_tokens_per_turn=config["decoding"]["max_new_tokens_per_turn"])
    from .io import jsonable
    report = agent.run(args.question); report["condition"] = jsonable(condition)
    report["scope"] = "synthetic corpus-derived prompts only; not official StudyBench"
    report["provenance"] = _provenance(config, args, "synthetic_root_agent_run",
                                        manifest["corpus_hash"])
    write_json(args.output, report); print(json.dumps(report, indent=2))


def parser():
    root = argparse.ArgumentParser(prog="latent-study")
    sub = root.add_subparsers(dest="command", required=True)
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
    records.add_argument("--candidate-revision")
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
    peek.add_argument("--model"); peek.add_argument("--model-revision"); peek.add_argument("--device", default="cuda:0")
    peek.add_argument("--local-files-only", action="store_true"); peek.add_argument("--max-new-tokens", type=int)
    peek.add_argument("--retries", type=int)
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
    train_p.add_argument("--model-revision")
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
    init_p.add_argument("--config", default="configs/dspy_mvp.json")
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
    replay.add_argument("--config", default="configs/dspy_mvp.json")
    replay.add_argument("--records", required=True); replay.add_argument("--output", required=True)
    replay.add_argument("--replay", type=float, choices=(0.0, .5))
    replay.add_argument("--seed", type=int); replay.add_argument("--batch-size", type=int)
    replay.add_argument("--updates-per-source", type=int); replay.set_defaults(func=command_replay_report)
    smoke = sub.add_parser("smoke-eval")
    smoke.add_argument("--config", default="configs/dspy_mvp.json")
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
    root_agent.add_argument("--memory")
    root_agent.add_argument("--acknowledge-full-recompute-fallback", action="store_true")
    root_agent.set_defaults(func=command_root_agent)
    merge = sub.add_parser("merge-records")
    merge.add_argument("--input", action="append", required=True)
    merge.add_argument("--expected-workers", type=int, required=True)
    merge.add_argument("--output", required=True); merge.set_defaults(func=command_merge_records)
    validate = sub.add_parser("validate-conditions")
    validate.add_argument("--config", default="configs/dspy_mvp.json")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--random-latent", required=True)
    validate.add_argument("--trained-latent", required=True)
    validate.add_argument("--peek64", required=True); validate.add_argument("--peek1024", required=True)
    validate.add_argument("--acknowledge-full-recompute-fallback", action="store_true")
    validate.add_argument("--output", required=True); validate.set_defaults(func=command_validate_conditions)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
