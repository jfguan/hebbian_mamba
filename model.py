"""Mamba + Hebbian associative memory.

Memory: W_t = γW_{t-1} + v_t⊗k_t, read_t = W_{t-1}·rk_t
Training uses O(TC·D + T·D²) chunkwise parallel form; inference uses O(D²) recurrent form.
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
    dual_memory: bool = False  # second W matrix with slower decay
    memory_alpha: float = 0.03  # residual injection scale
    chunk_size: int = 64        # chunkwise parallel form; set >= T for full quadratic


class HebbianMambaLayer(nn.Module):
    def __init__(self, cfg: Config, mcfg: MambaCfg):
        super().__init__()
        D = cfg.d_model
        self.d_model = D
        self.d_inner = mcfg.d_inner
        self.d_conv = mcfg.d_conv
        self.use_memory = cfg.use_memory
        self.memory_alpha = cfg.memory_alpha

        self.norm = RMSNorm(D)
        self.mamba = MambaBlock(mcfg)

        self.dual_memory = cfg.dual_memory
        self.chunk_size = cfg.chunk_size
        if self.use_memory:
            self.proj_write = nn.Linear(D, D, bias=False)
            self.proj_read = nn.Linear(D, D, bias=False)
            self.decay = nn.Parameter(torch.tensor(4.6))   # σ(4.6) ≈ 0.99
            if self.dual_memory:
                self.decay_slow = nn.Parameter(torch.tensor(6.9))  # σ(6.9) ≈ 0.999

    def _memory_attend(self, out):
        """Chunkwise parallel form: O(TC·D + T·D²) time, O(C² + D²) memory.

        Splits the sequence into chunks of size C. Within each chunk the intra-chunk
        contribution is the local quadratic form. Across chunks, the running W matrix
        is passed forward and applied as γ^l * (W_prev @ rk). Reduces to the full
        quadratic form when chunk_size >= T.
        """
        B, T, D = out.shape
        C = self.chunk_size
        out32 = out.float()  # upcast: bf16 backward overflows without this

        gamma = torch.sigmoid(self.decay)
        log_gamma = gamma.log()
        if self.dual_memory:
            gamma_slow = torch.sigmoid(self.decay_slow)
            log_gamma_slow = gamma_slow.log()

        v  = self.proj_write(out32)               # (B, T, D) write values
        wk = F.pad(out32[:, :-1], (0, 0, 1, 0))  # (B, T, D) write keys (shifted)
        rk = out32                                 # (B, T, D) read keys

        W = out32.new_zeros(B, D, D)
        W_slow = out32.new_zeros(B, D, D) if self.dual_memory else None
        reads_list = []

        for start in range(0, T, C):
            end = min(start + C, T)
            Ci = end - start
            p = torch.arange(Ci, device=out.device)  # local positions [0..Ci-1]

            rk_c = rk[:, start:end]   # (B, Ci, D)
            wk_c = wk[:, start:end]   # (B, Ci, D)
            v_c  =  v[:, start:end]   # (B, Ci, D)

            # Inter-chunk: γ^l * (W_prev @ rk_c[l]) for each local position l
            inter = torch.matmul(W, rk_c.transpose(1, 2)).transpose(1, 2)  # (B, Ci, D)
            inter = inter * (gamma ** p)[None, :, None]

            # Intra-chunk: (M_local ⊙ S_local) @ v_c
            S = torch.bmm(rk_c, wk_c.transpose(1, 2))           # (B, Ci, Ci)
            diffs_c = (p[:, None] - 1 - p[None, :]).clamp(min=0)  # (Ci, Ci)
            causal_c = p[:, None] > p[None, :]
            M_c = torch.exp(diffs_c * log_gamma) * causal_c      # (Ci, Ci)
            intra = torch.bmm(S * M_c, v_c)                      # (B, Ci, D)

            chunk_reads = inter + intra

            if self.dual_memory:
                inter_s = torch.matmul(W_slow, rk_c.transpose(1, 2)).transpose(1, 2)
                inter_s = inter_s * (gamma_slow ** p)[None, :, None]
                M_c_slow = torch.exp(diffs_c * log_gamma_slow) * causal_c
                intra_s = torch.bmm(S * M_c_slow, v_c)
                chunk_reads = chunk_reads + inter_s + intra_s

            reads_list.append(chunk_reads)

            # Advance W: γ^Ci * W + Σ_l γ^(Ci-1-l) * v_c[l] ⊗ wk_c[l]
            gw = (gamma ** (Ci - 1 - p))[None, :, None]          # (1, Ci, 1)
            W = gamma ** Ci * W + torch.bmm((v_c * gw).transpose(1, 2), wk_c)
            if self.dual_memory:
                gw_s = (gamma_slow ** (Ci - 1 - p))[None, :, None]
                W_slow = gamma_slow ** Ci * W_slow + torch.bmm((v_c * gw_s).transpose(1, 2), wk_c)

        all_reads = torch.cat(reads_list, dim=1)  # (B, T, D)
        alpha = self.memory_alpha / 2 if self.dual_memory else self.memory_alpha
        return out + alpha * self.proj_read(all_reads).to(out.dtype)

    def forward(self, x):
        residual = x
        out = self.mamba(self.norm(x))
        if self.use_memory:
            out = self._memory_attend(out)
        return residual + out

    def step(self, x, state=None):
        """Recurrent form: W ← γW + v⊗wk, read = W·out."""
        B = x.shape[0]
        residual = x

        if state is None:
            cache = (None, x.new_zeros(B, self.d_inner, self.d_conv - 1))
        else:
            cache = state["cache"]

        out, cache = self.mamba.step(self.norm(x), cache)

        if not self.use_memory:
            return residual + out, {"cache": cache}

        W = state["memory"] if state else x.new_zeros(B, self.d_model, self.d_model)
        r_prev = state["r_prev"] if state else x.new_zeros(B, self.d_model)

        gamma = torch.sigmoid(self.decay)
        raw_out = out
        write = torch.einsum("bi,bj->bij", self.proj_write(out), r_prev)
        read = torch.einsum("bij,bj->bi", W, out)
        W = gamma * W + write

        if self.dual_memory:
            W_slow = state["memory_slow"] if state else x.new_zeros(B, self.d_model, self.d_model)
            gamma_slow = torch.sigmoid(self.decay_slow)
            read = read + torch.einsum("bij,bj->bi", W_slow, out)
            W_slow = gamma_slow * W_slow + write

        alpha = self.memory_alpha / 2 if self.dual_memory else self.memory_alpha
        out = out + alpha * self.proj_read(read)

        new_state = {"cache": cache, "memory": W, "r_prev": raw_out}
        if self.dual_memory:
            new_state["memory_slow"] = W_slow
        return residual + out, new_state


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
