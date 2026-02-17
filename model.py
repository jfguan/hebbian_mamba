"""Mamba + Hebbian associative memory.

Memory: W_t = γW_{t-1} + v_t⊗k_t, read_t = W_t·k_t
Training uses O(T²D) parallel form; inference uses O(D²) recurrent form.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import MambaBlock, MambaConfig as MambaCfg, RMSNorm


@dataclass
class Config:
    vocab_size: int = 384
    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 8
    use_memory: bool = True


class HebbianMambaLayer(nn.Module):
    def __init__(self, cfg: Config, mcfg: MambaCfg):
        super().__init__()
        D = cfg.d_model
        self.d_model = D
        self.d_inner = mcfg.d_inner
        self.d_conv = mcfg.d_conv
        self.use_memory = cfg.use_memory

        self.norm = RMSNorm(D)
        self.mamba = MambaBlock(mcfg)

        if self.use_memory:
            self.proj_write = nn.Linear(D, D, bias=False)
            self.proj_read = nn.Linear(D, D, bias=False)
            self.decay = nn.Parameter(torch.tensor(4.6))   # σ(4.6) ≈ 0.99

    def _memory_attend(self, out, W):
        """Parallel form: reads = (M ⊙ KKᵀ)V where M is causal decay mask."""
        B, T, D = out.shape
        device = out.device
        log_gamma = torch.sigmoid(self.decay).log()

        v = self.proj_write(out)                        # write values
        wk = F.pad(out[:, :-1], (0, 0, 1, 0))          # write keys (shifted)
        rk = out                                         # read keys (current)

        # Causal decay: M[t,s] = γ^(t-1-s) · 𝟙[s<t]
        pos = torch.arange(T, device=device)
        diffs = (pos[:, None] - 1 - pos[None, :]).clamp(min=0)
        M = torch.exp(diffs * log_gamma) * (pos[:, None] > pos[None, :])

        # Read: query with current output, match against write keys
        reads = torch.bmm(torch.bmm(rk, wk.transpose(-1, -2)) * M, v)

        # Carried-over W contribution
        if W is not None:
            carry = torch.einsum("bij,btj->bti", W, rk)
            reads = reads + carry * torch.exp(pos * log_gamma)[None, :, None]

        out = out + 0.03 * self.proj_read(reads)

        # W for next chunk: W_T = Σ_t γ^(T-1-t) v_t ⊗ wk_t + γ^T W_0
        w = torch.exp(torch.arange(T - 1, -1, -1, device=device) * log_gamma)
        W_new = torch.einsum("t,btd,bte->bde", w, v, wk)
        if W is not None:
            W_new = W_new + torch.exp(T * log_gamma) * W

        return out, W_new

    def forward(self, x, memory=None):
        residual = x
        out = self.mamba(self.norm(x))
        if self.use_memory:
            out, memory = self._memory_attend(out, memory)
        return residual + out, memory

    def step(self, x, state=None):
        """Recurrent form: W ← γW + v⊗wk, read = W·out."""
        B = x.shape[0]
        residual = x

        if state is None:
            cache = (None, x.new_zeros(B, self.d_inner, self.d_conv - 1))
            W = x.new_zeros(B, self.d_model, self.d_model) if self.use_memory else None
            r_prev = x.new_zeros(B, self.d_model) if self.use_memory else None
        else:
            cache, W, r_prev = state["cache"], state["memory"], state["r_prev"]

        out, cache = self.mamba.step(self.norm(x), cache)

        if self.use_memory:
            gamma = torch.sigmoid(self.decay)
            read = torch.einsum("bij,bj->bi", W, out)
            W = gamma * W + torch.einsum("bi,bj->bij", self.proj_write(out), r_prev)
            r_prev = out
            out = out + 0.03 * self.proj_read(read)

        return residual + out, {"cache": cache, "memory": W, "r_prev": r_prev}


class HebbianMamba(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        mcfg = MambaCfg(
            d_model=cfg.d_model, n_layers=cfg.n_layers,
            d_state=cfg.d_state, d_conv=cfg.d_conv, expand_factor=cfg.expand,
        )
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [HebbianMambaLayer(cfg, mcfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

    def forward(self, input_ids, targets=None, memories=None):
        x = self.embedding(input_ids)
        new_mems = []
        for i, layer in enumerate(self.layers):
            x, mem = layer(x, memory=memories[i] if memories else None)
            new_mems.append(mem)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_mems

    def step(self, token, states=None):
        x = self.embedding(token)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer.step(x, state=states[i] if states else None)
            new_states.append(s)
        return self.lm_head(self.norm(x)), new_states
