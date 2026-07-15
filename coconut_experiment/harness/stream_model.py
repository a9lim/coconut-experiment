"""Continuous-carrier ("stream") model — the covertness-extreme third arm.

Generalizes the Coconut feedback to *every* post-prompt position: instead of
discrete tokens, the final-layer hidden state `h_t` is fed straight back as the
input embedding for position `t+1` (the sole carrier of state), while the
unembedded snapshot `argmax(LM_head(h_t))` is emitted as a **cosmetic readout**
token that is *never* re-embedded. This is Coconut's "direct hidden-state
feedback without projection" (cf. `coconut_model.py`) taken to the limit — the
maximally pure version CALM (2510.27688) abandoned for open-domain LMs because
the model "struggles to unpack the semantic information from such a compact
representation." On CheatChain the carrier only has to transport a handful of
mod-p integers, so the regime that's impossible in general is plausible here.

Two forward paths:

- `forward_professor` — **parallel** professor-forced pass for distillation.
  Post-prompt inputs are a *reference* model's hidden states (shifted by one),
  so there is no sequential dependency and the whole sequence trains in one
  forward. MPS-friendly: this is the path that avoids the multi-segment loop
  that pins the latent curriculum to the 4090.
- `generate_stream` — **sequential** self-feeding rollout (bs=1), the real
  deployment path. The model feeds its *own* hidden states forward; the gap
  between this and the professor-forced fit is the exposure-bias diagnostic.

The feedback path carries an optional thin `feedback_proj` (init to identity).
Default `none` is the pure extreme; `linear` is the obvious fallback if pure
collapses (it nudges the arm toward Projected Autoregression, 2601.04854).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, PreTrainedTokenizerBase


class ContinuousStreamModel(nn.Module):
    """All-position continuous-carrier transformer over a GPT-2 backbone."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        base_model: str = "gpt2",
        init_path: Optional[str] = None,
        proj: str = "none",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.proj_kind = proj

        # Init from a trained reference (recommended) or a base checkpoint.
        self.base_causallm = GPT2LMHeadModel.from_pretrained(init_path or base_model)
        if len(tokenizer) != self.base_causallm.config.vocab_size:
            self.base_causallm.resize_token_embeddings(len(tokenizer))

        self.embedding = self.base_causallm.transformer.wte
        self.eos_token_id = tokenizer.eos_token_id

        d = self.base_causallm.config.n_embd
        if proj == "linear":
            self.feedback_proj: nn.Module = nn.Linear(d, d)
            nn.init.eye_(self.feedback_proj.weight)  # start as identity → pure
            nn.init.zeros_(self.feedback_proj.bias)
        elif proj == "none":
            self.feedback_proj = nn.Identity()
        else:
            raise ValueError(f"proj must be 'none' or 'linear', got {proj!r}")

    # --- training: parallel professor forcing -------------------------------

    def forward_professor(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: torch.Tensor,
        ref_hidden: torch.Tensor,
        self_hidden: Optional[torch.Tensor] = None,
        ss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One parallel forward with reference hidden states fed at carried positions.

        Args:
            input_ids:      [B, T] token ids (prompt + reference continuation).
            attention_mask: [B, T] 1 for real tokens, 0 for right-padding.
            prompt_len:     [B]    number of prompt tokens (carrier begins here).
            ref_hidden:     [B, T, D] reference final-layer (post-ln_f) hidden states.
            self_hidden:    [B, T, D] optional student hidden states (scheduled
                            sampling source). Should be detached.
            ss_mask:        [B, T] bool — source positions where `self_hidden`
                            replaces `ref_hidden` before the shift. Requires
                            `self_hidden`.

        Returns:
            (logits [B, T, V], hidden [B, T, D]) from the student.

        At carried position t (t >= prompt_len) the input is the chosen source
        hidden at t-1; at prompt positions it is the token embedding. With
        reference sources only, all positions are independent → fully parallel.
        Scheduled sampling mixes in the student's own (stale, detached) states at
        the source level so the carrier learns to correct back toward the
        reference trajectory from its own drift; the targets stay the reference's.
        """
        _, T = input_ids.shape
        base = self.embedding(input_ids)                       # [B,T,D]

        src = ref_hidden
        if self_hidden is not None and ss_mask is not None:
            src = torch.where(ss_mask.unsqueeze(-1), self_hidden, ref_hidden)

        shifted = torch.zeros_like(ref_hidden)                 # carried input[t] = src[t-1]
        shifted[:, 1:] = src[:, :-1]

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)   # [1,T]
        carried = (pos >= prompt_len.unsqueeze(1)) & attention_mask.bool()
        inputs_embeds = torch.where(
            carried.unsqueeze(-1), self.feedback_proj(shifted), base
        )

        out = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return out.logits, out.hidden_states[-1]

    # --- inference: sequential self-feeding ---------------------------------

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 48,
    ) -> Tuple[List[int], torch.Tensor]:
        """Self-feeding rollout from a prompt. bs=1 (mirrors Coconut's generate).

        The carrier (hidden state) is fed forward; the readout snapshot
        (argmax of the unembedded carrier) is recorded but NOT re-embedded.
        Re-feeds the full embedding sequence each step (O(n^2), but sequences are
        short and this is the pattern proven on MPS in coconut_model.py).

        Returns:
            (readout_token_ids, carrier_trajectory [n_steps, D]).
        """
        assert input_ids.shape[0] == 1, "generate_stream is batch_size=1 only"

        inputs_embeds = self.embedding(input_ids)
        out = self.base_causallm(
            inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False
        )

        readouts: List[int] = []
        traj: List[torch.Tensor] = []
        for _ in range(max_new_tokens):
            h = out.hidden_states[-1][:, -1:, :]               # [1,1,D] carrier
            tok = int(out.logits[0, -1].argmax())              # cosmetic readout
            readouts.append(tok)
            traj.append(h[0, 0].detach())
            if tok == self.eos_token_id:
                break
            inputs_embeds = torch.cat([inputs_embeds, self.feedback_proj(h)], dim=1)
            out = self.base_causallm(
                inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False
            )

        carrier = torch.stack(traj) if traj else torch.empty(0, self.base_causallm.config.n_embd)
        return readouts, carrier

    def selffeed_logits(
        self,
        input_ids: torch.Tensor,
        n_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Differentiable self-feeding rollout (bs=1) — the RL training path.

        Like `generate_stream` but keeps the autograd graph through the BPTT
        carrier chain, so policy gradients reach the whole network (not just the
        readout head). Runs a fixed `n_steps` (no eos break — the carrier is
        deterministic given the prompt, so length is fixed by the caller).

        Returns:
            logits   [1, n_steps, V] — per-position readout logits (grad),
            carrier  [n_steps, D]     — per-position hidden states (grad),
            greedy   list[int]        — argmax readout ids (for locating the answer).
        """
        assert input_ids.shape[0] == 1, "selffeed_logits is batch_size=1 only"
        inputs_embeds = self.embedding(input_ids)
        out = self.base_causallm(
            inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False
        )
        logits_steps: List[torch.Tensor] = []
        hid_steps: List[torch.Tensor] = []
        greedy: List[int] = []
        for _ in range(n_steps):
            last_logits = out.logits[:, -1, :]                 # [1,V]
            h = out.hidden_states[-1][:, -1:, :]               # [1,1,D]
            logits_steps.append(last_logits)
            hid_steps.append(h[:, 0, :])
            greedy.append(int(last_logits.argmax()))
            inputs_embeds = torch.cat([inputs_embeds, self.feedback_proj(h)], dim=1)
            out = self.base_causallm(
                inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False
            )
        return torch.stack(logits_steps, dim=1), torch.stack(hid_steps), greedy

    # --- persistence --------------------------------------------------------

    def save_pretrained(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        self.base_causallm.save_pretrained(save_path)
        if isinstance(self.feedback_proj, nn.Linear):
            torch.save(self.feedback_proj.state_dict(), os.path.join(save_path, "feedback_proj.pt"))
        with open(os.path.join(save_path, "stream_config.json"), "w") as f:
            json.dump({"base_model": self.base_model, "proj": self.proj_kind}, f, indent=2)

    @classmethod
    def from_pretrained(
        cls, load_path: str, tokenizer: PreTrainedTokenizerBase
    ) -> "ContinuousStreamModel":
        with open(os.path.join(load_path, "stream_config.json")) as f:
            cfg = json.load(f)
        model = cls(tokenizer, base_model=cfg["base_model"], init_path=load_path, proj=cfg["proj"])
        proj_path = os.path.join(load_path, "feedback_proj.pt")
        if isinstance(model.feedback_proj, nn.Linear) and os.path.exists(proj_path):
            model.feedback_proj.load_state_dict(
                torch.load(proj_path, map_location="cpu", weights_only=True)
            )
        return model
