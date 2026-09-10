from types import SimpleNamespace

import pytest
import torch

from latent_study.agent import PrefixAgentSession
from latent_study.latent import SoftPrefixLM
from latent_study.objectives import query_pairwise_loss, ranking_loss


class ToyTokenizer:
    eos_token_id = None
    def encode(self, text, add_special_tokens=False):
        if text == "A": return [3]
        if text == "B": return [4]
        return [5 + (ord(c) % 10) for c in text] or [1]
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors=None):
        ids = [1]
        for message in messages: ids += self.encode(message["content"])
        if add_generation_prompt: ids.append(2)
        return torch.tensor([ids])
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([self.encode(text)]))


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, model_type="toy_causal_lm")
        self.embedding = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32, bias=False)
        self.calls = []
    def get_input_embeddings(self): return self.embedding
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None,
                cache_position=None, past_key_values=None, use_cache=False, return_dict=True):
        x = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        past_sum, past_len = (past_key_values if past_key_values is not None else
                              (torch.zeros_like(x[:, :1, :]), 0))
        states = torch.cumsum(x, dim=1) + past_sum
        logits = self.head(states)
        new_cache = (states[:, -1:, :].detach(), past_len + x.shape[1]) if use_cache else None
        self.calls.append({"input": x.shape[1], "mask": attention_mask.shape[1],
                           "positions": position_ids.detach().cpu().tolist(),
                           "cache_position": cache_position.detach().cpu().tolist()})
        return SimpleNamespace(logits=logits, past_key_values=new_cache)


def test_frozen_only_prefix_gradient_action_mask_and_ranking():
    lm = ToyLM(); wrapper = SoftPrefixLM(lm, ToyTokenizer(), length=3)
    assert not any(p.requires_grad for p in lm.parameters())
    context, action = torch.tensor([[1, 2]]), torch.tensor([[6, 7]])
    pos = wrapper.mean_action_logprob(context, action)
    neg = wrapper.mean_action_logprob(context, torch.tensor([[8]]))
    loss = query_pairwise_loss(pos, neg, 1.5)
    loss.backward()
    assert wrapper.prefix.grad is not None and torch.count_nonzero(wrapper.prefix.grad)
    assert all(p.grad is None for p in lm.parameters())
    # Every required evidence group contributes one loss term.
    rank = ranking_loss([([torch.tensor(2.0)], [torch.tensor(0.0)]),
                         ([torch.tensor(1.0), torch.tensor(.5)], [torch.tensor(-1.0)])])
    assert rank.ndim == 0 and rank > 0


def test_labels_prefix_masks_positions_cache_no_duplication_and_save_load(tmp_path):
    wrapper = SoftPrefixLM(ToyLM(), ToyTokenizer(), length=3)
    assert wrapper.validate_labels("A", "B") == (3, 4)
    with pytest.raises(ValueError): wrapper.validate_labels("AA", "B")
    ids = wrapper.chat_ids([{"role": "user", "content": "x"}])
    session = PrefixAgentSession(wrapper, ids)
    first = wrapper.model.calls[-1]
    assert first["input"] == ids.shape[1] + 3
    assert first["mask"] == first["input"]
    assert first["positions"] == [list(range(first["input"]))]
    session.append_tool_turn(torch.tensor([[9, 10]]))
    second = wrapper.model.calls[-1]
    assert second["input"] == 2
    assert second["mask"] == first["input"] + 2
    assert session.state.prefix_insertions == 1
    generated, final_state = wrapper.generate_from_state(session.state, max_new_tokens=2)
    assert generated.shape == (1, 2) and final_state.prefix_insertions == 1
    before = wrapper.prefill_ids(ids, use_cache=False).next_logits.detach().clone()
    path = tmp_path / "latent.pt"
    wrapper.save(path, corpus_hash="abc", model_id="toy")
    with torch.no_grad(): wrapper.prefix.add_(1)
    wrapper.load(path, expected_corpus_hash="abc")
    after = wrapper.prefill_ids(ids, use_cache=False).next_logits.detach()
    assert torch.equal(before, after)


def test_qwen_hybrid_state_uses_exact_recompute_fallback():
    lm = ToyLM(); lm.config.model_type = "qwen3_5"
    wrapper = SoftPrefixLM(lm, ToyTokenizer(), length=2)
    initial = torch.tensor([[1, 2, 3]])
    state = wrapper.prefill_ids(initial)
    state = wrapper.append_ids(state, torch.tensor([[4, 5, 6]]))
    full = wrapper.prefill_ids(torch.tensor([[1, 2, 3, 4, 5, 6]]), use_cache=False)
    assert state.cache_mode == "full_recompute"
    assert torch.equal(state.next_logits, full.next_logits)
    assert state.prefix_insertions == 1
