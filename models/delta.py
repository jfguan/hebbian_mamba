"""All-delta model: delta memory → conv + MLP at every layer.

First layer gets an extra conv + MLP before memory.
Supports 1-4 state matrices per block via delta_num_matrices config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import CausalConv, GatedMLP, DeltaBlock
from train.configs import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class DeltaLayer(nn.Module):
    """Delta memory → conv + MLP. Optionally prepend extra conv + MLP."""

    def __init__(self, cfg: ModelConfig, extra_block: bool = False):
        super().__init__()
        d_inner = cfg.expand * cfg.d_model

        self.extra_block = None
        if extra_block:
            self.extra_norm = RMSNorm(cfg.d_model)
            self.extra_mlp = GatedMLP(cfg.d_model, expand=cfg.expand)
            self.extra_conv = CausalConv(d_inner, d_conv=cfg.d_conv)
            self.extra_block = True

        self.norm_mem = RMSNorm(cfg.d_model)
        self.memory = DeltaBlock(
            d_model=cfg.d_model,
            num_heads=cfg.delta_num_heads or 8,
            chunk_size=cfg.chunk_size,
            num_matrices=cfg.delta_num_matrices,
        )

        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, expand=cfg.expand)
        self.conv = CausalConv(d_inner, d_conv=cfg.d_conv)

    def forward(self, x):
        if self.extra_block is not None:
            normed = self.extra_norm(x)
            val = self.extra_conv(self.extra_mlp.project_up(normed))
            x = x + self.extra_mlp(normed, val)

        normed = self.norm_mem(x)
        x = x + (self.memory(normed) - normed)

        normed = self.norm_mlp(x)
        val = self.conv(self.mlp.project_up(normed))
        return x + self.mlp(normed, val)

    def step(self, x, state=None):
        conv_st = state["conv"] if state else None
        mem_st = state["memory"] if state else None

        if self.extra_block is not None:
            normed = self.extra_norm(x)
            # extra conv step not implemented for inference yet
            x = x + self.extra_mlp(normed, self.extra_mlp.project_up(normed))

        normed = self.norm_mem(x)
        mem_out, mem_st = self.memory.step(normed, mem_st)
        x = x + (mem_out - normed)

        normed = self.norm_mlp(x)
        val, conv_st = self.conv.step(self.mlp.project_up(normed), conv_st)
        return x + self.mlp(normed, val), {"conv": conv_st, "memory": mem_st}


class Delta(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            DeltaLayer(cfg, extra_block=(i == 0))
            for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def step(self, token, states=None):
        x = self.embedding(token).squeeze(1)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer.step(x, state=states[i] if states else None)
            new_states.append(s)
        return self.lm_head(self.norm(x)), new_states
