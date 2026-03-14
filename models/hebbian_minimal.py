"""Conv + Hebbian associative memory.

Depthwise conv for local feature extraction (kernel=d_conv tokens),
Hebbian outer-product memory for long-range associative binding.

Memory: W_t = γW_{t-1} + v_t⊗k_t, read_t = W_{t-1}·rk_t
Training uses O(TC·D + T·D²) chunkwise parallel form; inference uses O(D²) recurrent form.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hebbian_block import HebbianBlock


@dataclass
class Config:
    vocab_size: int = 384
    d_model: int = 512
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 8
    memory_alpha: float = 0.03
    chunk_size: int = 64


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class HebbianLayer(nn.Module):
    """Conv local mixing + Hebbian associative memory + residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        E = cfg.expand
        d_inner = E * D
        self.d_model = D
        self.d_inner = d_inner
        self.d_conv = cfg.d_conv

        # Conv block: project up, depthwise conv, gate, project down
        self.norm = RMSNorm(D)
        self.proj = nn.Linear(D, d_inner, bias=False)
        self.gate = nn.Linear(D, d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, cfg.d_conv,
            bias=True, groups=d_inner, padding=cfg.d_conv - 1,
        )
        self.out_proj = nn.Linear(d_inner, D, bias=False)

        # Hebbian memory
        self.memory = HebbianBlock(
            d_model=D,
            chunk_size=cfg.chunk_size,
            memory_alpha=cfg.memory_alpha,
            learned_alpha=True,
        )

    def _conv(self, x):
        """Parallel form: (B, L, D) -> (B, L, D)."""
        B, L, D = x.shape
        val = self.proj(x).transpose(1, 2)
        val = F.silu(self.conv1d(val)[:, :, :L].transpose(1, 2))
        return self.out_proj(val * F.silu(self.gate(x)))

    def _conv_step(self, x, conv_state):
        """Recurrent form: (B, D), (B, d_inner, d_conv-1) -> (B, D), state."""
        val = self.proj(x)
        conv_input = torch.cat([conv_state, val.unsqueeze(-1)], dim=-1)
        conv_state = conv_input[:, :, 1:]
        assert self.conv1d.bias is not None
        val = F.silu((conv_input * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias)
        return self.out_proj(val * F.silu(self.gate(x))), conv_state

    def forward(self, x):
        out = self._conv(self.norm(x))
        out = self.memory(out)
        return x + out

    def step(self, x, state=None):
        """Recurrent form: W ← γW + v⊗k, read = W·q."""
        B, D = x.shape
        if state is None:
            conv_st = x.new_zeros(B, self.d_inner, self.d_conv - 1)
            mem_state = None
        else:
            conv_st = state["conv"]
            mem_state = state["memory"]

        out, conv_st = self._conv_step(self.norm(x), conv_st)
        out, mem_state = self.memory.step(out, mem_state)

        return x + out, {"conv": conv_st, "memory": mem_state}


class HebbianConv(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([HebbianLayer(cfg) for _ in range(cfg.n_layers)])
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
