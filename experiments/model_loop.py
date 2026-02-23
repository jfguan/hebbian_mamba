"""LoopedMamba: middle layers run Mamba twice with shared weights.

Architecture per looped layer:
    out1 = mamba(norm1(x))          # pass 1
    x1   = x + out1                 # residual
    out2 = mamba(norm2(x1))         # pass 2 — tied weights
    out2 = memory_attend(out2)      # Hebbian once, on final output
    x2   = x1 + out2                # residual

Two mamba passes for depth, one Hebbian at the end. Same memory structure
as HebbianMambaLayer, just double the effective SSM depth per layer.
Boundary layers are standard HebbianMambaLayers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import MambaBlock, MambaConfig as MambaCfg, RMSNorm

from model import Config, HebbianMambaLayer


def _tie_params(src, dst):
    """Make dst's parameters point to src's parameter tensors."""
    for (n1, p1), (n2, p2) in zip(
        src.named_parameters(), dst.named_parameters()
    ):
        parts = n2.split(".")
        mod = dst
        for part in parts[:-1]:
            mod = getattr(mod, part)
        setattr(mod, parts[-1], p1)


class LoopedMambaLayer(nn.Module):
    """Mamba run twice (tied weights) + Hebbian on the final output."""

    def __init__(self, cfg: Config, mcfg: MambaCfg):
        super().__init__()
        D = cfg.d_model
        self.d_model = D
        self.d_inner = mcfg.d_inner
        self.d_conv = mcfg.d_conv
        self.memory_alpha = cfg.memory_alpha
        self.chunk_size = cfg.chunk_size

        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.mamba1 = MambaBlock(mcfg)
        self.mamba2 = MambaBlock(mcfg)
        _tie_params(self.mamba1, self.mamba2)

        self.proj_write = nn.Linear(D, D, bias=False)
        self.proj_read = nn.Linear(D, D, bias=False)
        self.decay = nn.Parameter(torch.tensor(4.6))

    def _memory_attend(self, out):
        B, T, D = out.shape
        C = self.chunk_size
        out32 = out.float()

        gamma = torch.sigmoid(self.decay)
        log_gamma = gamma.log()

        v  = self.proj_write(out32)
        wk = F.pad(out32[:, :-1], (0, 0, 1, 0))
        rk = out32

        W = out32.new_zeros(B, D, D)
        reads_list = []

        for start in range(0, T, C):
            end = min(start + C, T)
            Ci = end - start
            p = torch.arange(Ci, device=out.device)

            rk_c = rk[:, start:end]
            wk_c = wk[:, start:end]
            v_c  =  v[:, start:end]

            inter = torch.matmul(W, rk_c.transpose(1, 2)).transpose(1, 2)
            inter = inter * (gamma ** p)[None, :, None]

            S = torch.bmm(rk_c, wk_c.transpose(1, 2))
            diffs_c = (p[:, None] - 1 - p[None, :]).clamp(min=0)
            causal_c = p[:, None] > p[None, :]
            M_c = torch.exp(diffs_c * log_gamma) * causal_c
            intra = torch.bmm(S * M_c, v_c)

            reads_list.append(inter + intra)

            gw = (gamma ** (Ci - 1 - p))[None, :, None]
            W = gamma ** Ci * W + torch.bmm((v_c * gw).transpose(1, 2), wk_c)

        all_reads = torch.cat(reads_list, dim=1)
        return out + self.memory_alpha * self.proj_read(all_reads).to(out.dtype)

    def step(self, x, state=None):
        """Recurrent form: two mamba steps + W ← γW + v⊗wk, read = W·out."""
        B = x.shape[0]

        if state is None:
            cache1 = (None, x.new_zeros(B, self.d_inner, self.d_conv - 1))
            cache2 = (None, x.new_zeros(B, self.d_inner, self.d_conv - 1))
        else:
            cache1, cache2 = state["cache1"], state["cache2"]

        # Pass 1
        out1, cache1 = self.mamba1.step(self.norm1(x), cache1)
        x1 = x + out1

        # Pass 2
        out2, cache2 = self.mamba2.step(self.norm2(x1), cache2)

        # Recurrent Hebbian memory
        W = state["memory"] if state else x.new_zeros(B, self.d_model, self.d_model)
        r_prev = state["r_prev"] if state else x.new_zeros(B, self.d_model)

        gamma = torch.sigmoid(self.decay)
        raw_out = out2
        write = torch.einsum("bi,bj->bij", self.proj_write(out2), r_prev)
        read = torch.einsum("bij,bj->bi", W, out2)
        W = gamma * W + write

        out2 = out2 + self.memory_alpha * self.proj_read(read)

        new_state = {"cache1": cache1, "cache2": cache2, "memory": W, "r_prev": raw_out}
        return x1 + out2, new_state

    def forward(self, x):
        out1 = self.mamba1(self.norm1(x))
        x1 = x + out1

        out2 = self.mamba2(self.norm2(x1))
        out2 = self._memory_attend(out2)

        return x1 + out2


class HebbianMambaLoop(nn.Module):
    """HebbianMamba with LoopedMambaLayers starting from layer 2."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        mcfg = MambaCfg(
            d_model=cfg.d_model, n_layers=cfg.n_layers,
            d_state=cfg.d_state, d_conv=cfg.d_conv, expand_factor=cfg.expand,
        )
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        n = cfg.n_layers
        loop_start = 2
        loop_end = min(loop_start + 4, n)
        layers = []
        for i in range(n):
            if loop_start <= i < loop_end:
                layers.append(LoopedMambaLayer(cfg, mcfg))
            else:
                layers.append(HebbianMambaLayer(cfg, mcfg))
        self.layers = nn.ModuleList(layers)

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
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def step(self, token, states=None):
        x = self.embedding(token)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer.step(x, state=states[i] if states else None)
            new_states.append(s)
        return self.lm_head(self.norm(x)), new_states

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
