from __future__ import annotations

import inspect
import json
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
        return ids.to(self.prefix.device)

    def _supported(self, kwargs: dict) -> dict:
        signature = inspect.signature(self.model.forward)
        accepts_any = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
        return kwargs if accepts_any else {k: v for k, v in kwargs.items() if k in signature.parameters}

    def prefill_ids(self, input_ids, *, use_cache: bool = True) -> PrefixState:
        import torch
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("MVP prefix integration supports one unpadded sequence per root session")
        token_embeds = self.model.get_input_embeddings()(input_ids)
        prefix = self.prefix.unsqueeze(0).to(token_embeds.dtype)
        inputs_embeds = torch.cat((prefix, token_embeds), dim=1)
        total = inputs_embeds.shape[1]
        attention = torch.ones((1, total), dtype=torch.long, device=inputs_embeds.device)
        positions = torch.arange(total, device=inputs_embeds.device).unsqueeze(0)
        kwargs = self._supported({"inputs_embeds": inputs_embeds, "attention_mask": attention,
                                  "position_ids": positions, "cache_position": positions[0],
                                  "use_cache": use_cache, "return_dict": True,
                                  "logits_to_keep": 1})
        output = self.model(**kwargs)
        mode = ("full_recompute" if getattr(self.model.config, "model_type", "").startswith("qwen3_5")
                else "native")
        return PrefixState(output.past_key_values if use_cache and mode == "native" else None, attention,
                           output.logits[:, -1, :], total, 1, input_ids.detach(), mode)

    def append_ids(self, state: PrefixState, input_ids) -> PrefixState:
        # Transformers 5.3's torch fallback for Qwen3.5 DeltaNet produces
        # materially different logits for recurrent-cache vs chunk recompute.
        # Until an equivalent kernel is available, retain exact token history
        # and recompute. This is slower but prevents silently invalid results.
        if state.cache_mode == "full_recompute":
            if state.input_ids is None: raise RuntimeError("Qwen recompute state lost token history")
            return self.prefill_ids(__import__("torch").cat((state.input_ids, input_ids), dim=1), use_cache=False)
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
                           state.total_length + count, 1, history, state.cache_mode)

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
        ids = torch.cat((context_ids, action_ids), dim=1)
        token_embeds = self.model.get_input_embeddings()(ids)
        embeds = torch.cat((self.prefix.unsqueeze(0).to(token_embeds.dtype), token_embeds), dim=1)
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

    def save(self, path: str | Path, *, corpus_hash: str, model_id: str) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"prefix": self.prefix.detach().cpu(), "length": self.length,
                    "hidden_size": self.hidden_size, "corpus_hash": corpus_hash,
                    "model_id": model_id, "phase": "frozen_before_evaluation",
                    "evaluation_inputs_seen": False}, path)

    def load(self, path: str | Path, *, expected_corpus_hash: str | None = None) -> dict:
        import torch
        payload = torch.load(path, map_location=self.prefix.device, weights_only=True)
        if payload["prefix"].shape != self.prefix.shape:
            raise ValueError("saved prefix shape does not match model")
        if expected_corpus_hash and payload["corpus_hash"] != expected_corpus_hash:
            raise ValueError("latent belongs to a different corpus snapshot")
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
