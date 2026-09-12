#!/usr/bin/env python3
"""Fresh-process latent reload comparison for the strengthened Qwen smoke."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from latent_study.latent import load_qwen
from latent_study.artifacts import ARTIFACT_SCHEMA_VERSION, provenance
from latent_study.io import canonical_hash
from latent_study.search import TOOL_SCHEMA_HASH
from latent_study.snapshots import resolve_model_snapshot

parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--revision")
parser.add_argument("--checkpoint", required=True); parser.add_argument("--device", default="cuda:0")
parser.add_argument("--output")
parser.add_argument("--tolerance", type=float, default=0.0); args = parser.parse_args()
import torch
reference = torch.load(args.checkpoint + ".reference.pt", map_location="cpu", weights_only=True)
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
snapshot = resolve_model_snapshot(args.model, args.revision or "unresolved", local_files_only=True)
wrapper = load_qwen(snapshot["resolved_path"], length=int(payload["length"]), dtype="bfloat16",
                    revision=args.revision, device=args.device)
wrapper.load(args.checkpoint, expected_corpus_hash="real-smoke-only",
             expected_phase=payload.get("phase", "study_complete_unattested"),
             expected_model_snapshot_sha256=snapshot["model_snapshot_sha256"],
             expected_tokenizer_snapshot_sha256=snapshot["tokenizer_snapshot_sha256"])
actual = wrapper.prefill_ids(reference["input_ids"].to(wrapper.prefix.device), use_cache=False,
                             memory_boundary=reference["memory_boundary"]).next_logits.detach().float().cpu()
error = float((actual - reference["logits"]).abs().max())
result = {"fresh_process": True, "max_abs_error": error, "tolerance": args.tolerance,
          "passed": error <= args.tolerance, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
          "provenance": provenance(artifact_type="latent_reload_report",
                                   resolved_config_hash=canonical_hash(vars(args)),
                                   protocol_config_hash=canonical_hash({
                                       "protocol": "real_qwen_smoke_v1",
                                       "model_revision": args.revision,
                                       "latent_length": int(payload["length"])}),
                                   model_id="Qwen/Qwen3.5-9B",
                                   model_revision=args.revision or "unresolved",
                                   corpus_hash="real-smoke-only", tool_schema_hash=TOOL_SCHEMA_HASH,
                                   command="python scripts/reload_latent_smoke.py",
                                   cli_overrides=vars(args), repository_root=ROOT,
                                   tokenizer_id="Qwen/Qwen3.5-9B",
                                   tokenizer_revision=args.revision or "unresolved",
                                   model_snapshot_sha256=snapshot["model_snapshot_sha256"],
                                   tokenizer_snapshot_sha256=snapshot["tokenizer_snapshot_sha256"])}
print(json.dumps(result, indent=2))
if args.output:
    from latent_study.io import write_json
    write_json(args.output, result)
if not result["passed"]: raise SystemExit(1)
