"""Task environment: prompts -> conditioning latents; decoded ids -> verifier reward.
Sequence layout follows ELF conditional generation: [cond tokens][generated tokens], padded
with EOS to max_length; cond positions hold T5-encoded latents and are held fixed."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

import hcfrl.elf_path  # noqa: F401
from utils.encoder_utils import build_self_attn_cond_masks, encode_text  # type: ignore
from utils.generation_utils import mask_after_eos, shift_left  # type: ignore
from hcfrl.tasks import arith


@dataclass
class Prompt:
    text: str
    nums: list[int] | None = None
    target: int | None = None
    task: str = "arith"          # "arith" (Countdown/Game24 chain verifier) or "kk"


def load_prompts(path: str) -> list[Prompt]:
    out = []
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r.get("task") == "kk":
                out.append(Prompt(r["input"], task="kk"))
            else:
                out.append(Prompt(r["input"], [int(x) for x in r["nums"]], int(r["target"])))
    return out


def prompt_from_train_row(r: dict) -> Prompt:
    """Train JSONL rows have {input, output[, task]}; derive the verifier fields from the prompt text."""
    if r.get("task") == "kk":
        return Prompt(r["input"], task="kk")
    t, rest = r["input"].split(":")
    return Prompt(r["input"], [int(x) for x in rest.split()], int(t))


class TaskEnv:
    def __init__(self, tokenizer, encoder, config, device):
        self.tok, self.encoder, self.cfg, self.dev = tokenizer, encoder, config, device
        self.L, self.Lc = config.max_length, config.max_input_length
        self.eos = tokenizer.eos_token_id

    @torch.no_grad()
    def encode(self, prompts: list[Prompt]):
        """Returns cond_seq (B, L, D), cond_mask (B, L), gen_mask (B, L), cond_lens (B,)."""
        B = len(prompts)
        ids = np.full((B, self.L), self.eos, dtype=np.int64); cl = np.zeros(B, dtype=np.int64)
        for i, p in enumerate(prompts):
            c = self.tok(p.text, add_special_tokens=False)["input_ids"][: self.Lc]
            ids[i, : len(c)] = c; cl[i] = len(c)
        pos = np.arange(self.L)[None]
        is_cond = pos < cl[:, None]
        enc_mask, _, cond_mask = build_self_attn_cond_masks(is_cond, np.ones_like(is_cond))
        cond_seq = encode_text(torch.from_numpy(ids).to(self.dev), torch.from_numpy(enc_mask).to(self.dev),
                               self.encoder, self.cfg.latent_mean, self.cfg.latent_std,
                               use_bf16=False).float()
        cond_mask = torch.from_numpy(cond_mask).to(self.dev)
        return cond_seq, cond_mask, 1.0 - cond_mask, torch.from_numpy(cl).to(self.dev)

    def decode_text(self, token_ids: torch.Tensor, cond_lens: torch.Tensor) -> list[str]:
        gen = shift_left(token_ids, cond_lens, 0)[:, : self.L - self.Lc]
        gen = mask_after_eos(gen, eos_token_id=self.eos, pad_token_id=self.eos)
        return [self.tok.decode(g.tolist(), skip_special_tokens=True) for g in gen.cpu()]

    @staticmethod
    def reward(prompts: list[Prompt], texts: list[str]) -> torch.Tensor:
        out = []
        for p, t in zip(prompts, texts):
            if p.task == "kk":
                from hcfrl.tasks.kk import KKTask
                try:
                    out.append(float(KKTask.verify(KKTask.parse_prompt(p.text), t)))
                except ValueError:
                    out.append(0.0)
            else:
                out.append(float(arith.verify_chain(t, p.nums, p.target)))
        return torch.tensor(out)
