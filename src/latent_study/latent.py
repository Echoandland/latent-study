from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrefixState:
    past_key_values: object
    attention_mask: object
    next_logits: object
    total_length: int
    prefix_insertions: int = 1
    input_ids: object | None = None
    cache_mode: str = "native"
    memory_boundary: int = 0


class SoftPrefixLM:
    """One trainable global input-embedding prefix around a completely frozen LM."""

    def __init__(self, model, tokenizer, length: int = 64, *, init_std: float = 0.02):
        import torch
        if length <= 0:
            raise ValueError("prefix length must be positive")
        self.model, self.tokenizer, self.length = model, tokenizer, length
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        embedding = model.get_input_embeddings().weight
        hidden = int(getattr(model.config, "hidden_size", embedding.shape[-1]))
        if hidden != embedding.shape[-1]:
            raise ValueError("model config hidden size differs from token embedding width")
        self.hidden_size = hidden
        # Keep the only optimized state in fp32; cast differentiably for bf16/fp16 LM forward.
        self.prefix = torch.nn.Parameter(torch.empty(length, hidden, device=embedding.device,
                                                      dtype=torch.float32).normal_(0.0, init_std))
        self.model.eval()

    def trainable_parameters(self):
        return [self.prefix]

    def assert_frozen(self) -> None:
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("base LM parameter unexpectedly trainable")

    def chat_ids(self, messages, *, add_generation_prompt: bool = True):
        from .agent import serialize_chat_ids
        ids = serialize_chat_ids(self.tokenizer, messages, add_generation_prompt=add_generation_prompt)
        boundary = getattr(ids, "_memory_slot_start", 0)
        ids = ids.to(self.prefix.device)
        ids._memory_slot_start = boundary
        ids._memory_slot_end = boundary
        return ids

    def _supported(self, kwargs: dict) -> dict:
        signature = inspect.signature(self.model.forward)
        accepts_any = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
        return kwargs if accepts_any else {k: v for k, v in kwargs.items() if k in signature.parameters}

    def prefill_ids(self, input_ids, *, use_cache: bool = True,
                    memory_boundary: int | None = None) -> PrefixState:
        import torch
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("MVP prefix integration supports one unpadded sequence per root session")
        if memory_boundary is None:
            memory_boundary = int(getattr(input_ids, "_memory_slot_start", 0))
        if not 0 <= memory_boundary <= input_ids.shape[1]:
            raise ValueError("memory boundary falls outside serialized conversation")
        token_embeds = self.model.get_input_embeddings()(input_ids)
        prefix = self.prefix.unsqueeze(0).to(token_embeds.dtype)
        inputs_embeds = torch.cat((token_embeds[:, :memory_boundary], prefix,
                                  token_embeds[:, memory_boundary:]), dim=1)
        total = inputs_embeds.shape[1]
        attention = torch.ones((1, total), dtype=torch.long, device=inputs_embeds.device)
        positions = torch.arange(total, device=inputs_embeds.device).unsqueeze(0)
        kwargs = self._supported({"inputs_embeds": inputs_embeds, "attention_mask": attention,
                                  "position_ids": positions, "cache_position": positions[0],
                                  "use_cache": use_cache, "return_dict": True,
                                  "logits_to_keep": 1})
        output = self.model(**kwargs)
        mode = ("full_recompute_fallback" if getattr(self.model.config, "model_type", "").startswith("qwen3_5")
                else "native")
        return PrefixState(output.past_key_values if use_cache and mode == "native" else None, attention,
                           output.logits[:, -1, :], total, 1, input_ids.detach(), mode,
                           memory_boundary)

    def append_ids(self, state: PrefixState, input_ids) -> PrefixState:
        # Transformers 5.3's torch fallback for Qwen3.5 DeltaNet produces
        # materially different logits for recurrent-cache vs chunk recompute.
        # Until an equivalent kernel is available, retain exact token history
        # and recompute. This is slower but prevents silently invalid results.
        if state.cache_mode == "full_recompute_fallback":
            if state.input_ids is None: raise RuntimeError("Qwen recompute state lost token history")
            return self.prefill_ids(__import__("torch").cat((state.input_ids, input_ids), dim=1),
                                    use_cache=False, memory_boundary=state.memory_boundary)
        return self._append_block(state, input_ids)

    def _append_block(self, state: PrefixState, input_ids) -> PrefixState:
        import torch
        if state.prefix_insertions != 1:
            raise RuntimeError("prefix duplication detected")
        count = input_ids.shape[1]
        attention = torch.cat((state.attention_mask,
                               torch.ones((1, count), dtype=state.attention_mask.dtype,
                                          device=state.attention_mask.device)), dim=1)
        positions = torch.arange(state.total_length, state.total_length + count,
                                 device=input_ids.device).unsqueeze(0)
        kwargs = self._supported({"input_ids": input_ids, "attention_mask": attention,
                                  "position_ids": positions, "cache_position": positions[0],
                                  "past_key_values": state.past_key_values, "use_cache": True,
                                  "return_dict": True, "logits_to_keep": 1})
        output = self.model(**kwargs)
        history = (__import__("torch").cat((state.input_ids, input_ids), dim=1)
                   if state.input_ids is not None else None)
        return PrefixState(output.past_key_values, attention, output.logits[:, -1, :],
                           state.total_length + count, 1, history, state.cache_mode,
                           state.memory_boundary)

    def generate_from_state(self, state: PrefixState, *, max_new_tokens: int,
                            temperature: float = 0.0, generator=None):
        """Decode while extending one prefix-bearing cache; returns token IDs and final state."""
        import torch
        generated = []
        for _ in range(max_new_tokens):
            if temperature > 0:
                probs = torch.softmax(state.next_logits.float() / temperature, dim=-1)
                token = torch.multinomial(probs, 1, generator=generator)
            else:
                token = state.next_logits.argmax(dim=-1, keepdim=True)
            generated.append(token)
            state = self.append_ids(state, token)
            if self.tokenizer.eos_token_id is not None and int(token.item()) == self.tokenizer.eos_token_id:
                break
        return torch.cat(generated, dim=1) if generated else torch.empty(
            (1, 0), dtype=torch.long, device=self.prefix.device), state

    def generate_chat(self, messages, *, max_new_tokens: int, temperature: float = 0.0,
                      generator=None):
        state = self.prefill_ids(self.chat_ids(messages, add_generation_prompt=True), use_cache=True)
        return self.generate_from_state(state, max_new_tokens=max_new_tokens,
                                        temperature=temperature, generator=generator)

    def mean_action_logprob(self, context_ids, action_ids):
        import torch
        import torch.nn.functional as F
        if context_ids.shape[0] != 1 or action_ids.shape[0] != 1 or action_ids.shape[1] < 1:
            raise ValueError("single nonempty action sequence required")
        memory_boundary = int(getattr(context_ids, "_memory_slot_start", 0))
        ids = torch.cat((context_ids, action_ids), dim=1)
        token_embeds = self.model.get_input_embeddings()(ids)
        embeds = torch.cat((token_embeds[:, :memory_boundary],
                            self.prefix.unsqueeze(0).to(token_embeds.dtype),
                            token_embeds[:, memory_boundary:]), dim=1)
        total = embeds.shape[1]
        mask = torch.ones((1, total), dtype=torch.long, device=embeds.device)
        positions = torch.arange(total, device=embeds.device).unsqueeze(0)
        output = self.model(**self._supported({"inputs_embeds": embeds, "attention_mask": mask,
                                               "position_ids": positions, "cache_position": positions[0],
                                               "use_cache": False, "return_dict": True}))
        start = self.length + context_ids.shape[1] - 1
        logits = output.logits[:, start : start + action_ids.shape[1], :]
        selected = F.log_softmax(logits.float(), dim=-1).gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
        return selected.mean()  # only action-token targets; context is never a target

    def label_score(self, prompt_ids, relevant_id: int, irrelevant_id: int):
        state = self.prefill_ids(prompt_ids, use_cache=False)
        return state.next_logits[0, relevant_id] - state.next_logits[0, irrelevant_id]

    def validate_labels(self, relevant: str, irrelevant: str) -> tuple[int, int]:
        ids = []
        for label in (relevant, irrelevant):
            encoded = self.tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"ranking label {label!r} is not one token under actual tokenizer")
            ids.append(encoded[0])
        if ids[0] == ids[1]:
            raise ValueError("ranking labels tokenize to the same token")
        return ids[0], ids[1]

    def save(self, path: str | Path, *, corpus_hash: str, model_id: str,
             model_revision: str = "", seed: int | None = None,
             provenance: dict | None = None) -> None:
        import torch
        from .artifacts import artifact_payload_hash
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(provenance or {})
        # Direct library fixtures may omit provenance, but production CLI
        # callers always provide it.  An unattested production checkpoint is
        # deliberately marked ineligible by the isolation gate.
        attestation = metadata.get("contamination_audit") or {}
        attested = (attestation.get("status") == "pass"
                    and isinstance(attestation.get("artifact_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", attestation["artifact_sha256"])
                    and isinstance(attestation.get("evaluation_dataset_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", attestation["evaluation_dataset_sha256"]))
        from .artifacts import ARTIFACT_SCHEMA_VERSION
        phase = "frozen_before_evaluation" if (provenance is None or attested) else "study_complete_unattested"
        payload = {"prefix": self.prefix.detach().cpu(), "length": self.length,
                   "hidden_size": self.hidden_size, "corpus_hash": corpus_hash,
                   "model_id": model_id, "model_revision": model_revision,
                   "seed": seed, "phase": phase,
                   "evaluation_inputs_seen": False, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                   "provenance": metadata}
        if metadata:
            metadata["artifact_sha256"] = None
            payload["provenance"] = metadata
            metadata["artifact_sha256"] = artifact_payload_hash(payload)
        torch.save(payload, path)

    def load(self, path: str | Path, *, expected_corpus_hash: str | None = None,
             expected_model_id: str | None = None,
             expected_model_revision: str | None = None,
             expected_tokenizer_id: str | None = None,
             expected_tokenizer_revision: str | None = None,
             expected_model_snapshot_sha256: str | None = None,
             expected_tokenizer_snapshot_sha256: str | None = None,
             expected_protocol_config_hash: str | None = None,
             expected_phase: str = "frozen_before_evaluation") -> dict:
        import torch
        payload = torch.load(path, map_location=self.prefix.device, weights_only=True)
        from .artifacts import ARTIFACT_SCHEMA_VERSION
        if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("latent checkpoint uses an incompatible artifact schema")
        if payload["prefix"].shape != self.prefix.shape:
            raise ValueError("saved prefix shape does not match model")
        if expected_corpus_hash and payload["corpus_hash"] != expected_corpus_hash:
            raise ValueError("latent belongs to a different corpus snapshot")
        if expected_model_id and payload.get("model_id") != expected_model_id:
            raise ValueError("latent belongs to a different model")
        if expected_model_revision and payload.get("model_revision") != expected_model_revision:
            raise ValueError("latent belongs to a different model revision")
        if payload.get("phase") != expected_phase or payload.get("evaluation_inputs_seen") is not False:
            raise ValueError("latent is not a clean pre-evaluation frozen artifact")
        if payload.get("length") != self.length or payload.get("hidden_size") != self.hidden_size:
            raise ValueError("latent length/hidden size is incompatible with the deployment model")
        metadata = payload.get("provenance")
        if isinstance(metadata, dict) and isinstance(metadata.get("artifact_sha256"), str):
            from .artifacts import validate_provenance
            validate_provenance(metadata, artifact_path=path,
                                artifact_type=metadata.get("artifact_type"),
                                model_id=expected_model_id, model_revision=expected_model_revision,
                                corpus_hash=expected_corpus_hash,
                                protocol_config_hash=expected_protocol_config_hash,
                                tokenizer_id=expected_tokenizer_id,
                                tokenizer_revision=expected_tokenizer_revision,
                                model_snapshot_sha256=expected_model_snapshot_sha256,
                                tokenizer_snapshot_sha256=expected_tokenizer_snapshot_sha256)
        with torch.no_grad():
            self.prefix.copy_(payload["prefix"].to(dtype=self.prefix.dtype))
        return {k: v for k, v in payload.items() if k != "prefix"}


def load_qwen(model_path: str, *, length: int = 64, dtype: str = "bfloat16",
              revision: str | None = None, device: str | None = None,
              init_std: float = 0.02) -> SoftPrefixLM:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_value = getattr(torch, dtype)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype_value,
                                                 device_map=device, trust_remote_code=False,
                                                 revision=revision)
    if model.config.model_type.lower() not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError(f"MVP requires qwen3_5, got {model.config.model_type}")
    return SoftPrefixLM(model, tokenizer, length, init_std=init_std)
