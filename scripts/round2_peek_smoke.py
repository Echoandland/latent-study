#!/usr/bin/env python3
"""Bounded paired real-Qwen PEEK smoke; never a full study experiment."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3.5-9B")
    parser.add_argument("--revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--internal-max-new-tokens", type=int, default=1024)
    parser.add_argument("--retry-max-new-tokens", type=int, default=1536)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    from latent_study.corpus import build_manifest
    from latent_study.peek_baseline import QwenPeekClient, study_offline_peek
    from latent_study.records import generate_coverage_records
    from latent_study.search import CodingTools
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with tempfile.TemporaryDirectory(prefix="latent-study-round2-peek-") as temporary:
        root = Path(temporary); corpus = root / "corpus"; corpus.mkdir()
        (corpus / "facts.py").write_text(
            "def lunar_checksum(value):\n"
            "    \"\"\"Return the stable lunar checksum for a corpus value.\"\"\"\n"
            "    return value + 7\n\n"
            "def solar_checksum(value):\n"
            "    \"\"\"Return the stable solar checksum for a corpus value.\"\"\"\n"
            "    return value + 9\n", encoding="utf-8")
        manifest = build_manifest(corpus)
        tools = CodingTools(corpus, manifest["units"])
        records = generate_coverage_records(manifest["units"], manifest["corpus_hash"],
                                            n_actions=2, limit=2, search=tools)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, device_map=args.device, dtype="bfloat16",
            local_files_only=True)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval()
        token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        reports = []
        for budget in (64, 1024):
            output = root / f"peek_{budget}.json"
            client = QwenPeekClient(
                model, tokenizer, internal_max_new_tokens=args.internal_max_new_tokens,
                retry_max_new_tokens=args.retry_max_new_tokens, retries=args.retries)
            try:
                payload = study_offline_peek(
                    records, output, token_budget=budget, client=client,
                    replay_fraction=0.5, seed=17, batch_size=2, updates_per_source=1,
                    token_counter=token_counter, counter_name=f"{args.model}@{args.revision}")
                reports.append({"budget": budget, "status": "success",
                                "phase": payload.get("phase"),
                                "map_text_tokens": payload.get("map_text_tokens"),
                                "map_text_bytes": payload.get("map_text_bytes"),
                                "map_excerpt": payload.get("map_text", "")[:700],
                                "client_statistics": payload.get("client_statistics")})
            except Exception as exc:
                diagnostic = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
                reports.append({"budget": budget, "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                                "phase": diagnostic.get("phase"),
                                "map_text_tokens": diagnostic.get("map_text_tokens"),
                                "map_text_bytes": diagnostic.get("map_text_bytes"),
                                "client_statistics": diagnostic.get("client_statistics")})
        result = {"records": len(records), "corpus_hash": manifest["corpus_hash"],
                  "budgets": reports, "scope": "bounded paired smoke; not a frozen baseline"}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if any(item["status"] != "success" for item in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
