"""Hebbian associative memory block.

Memory: W_t = γW_{t-1} + v_t⊗k_{t-1}, read_t = W_{t-1}·q_t
Training uses O(TC·D + T·D²) chunkwise parallel form; inference uses O(D²) recurrent form.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HebbianBlock(nn.Module):
    """Hebbian outer-product associative memory.

    Maintains a D×D matrix of associations updated at each token.
    Supports both chunkwise parallel (training) and recurrent (inference) forms.
    """

    def __init__(self, d_model: int, chunk_size: int = 64, memory_alpha: float = 0.03,
                 learned_alpha: bool = True):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

        self.proj_write = nn.Linear(d_model, d_model, bias=False)
        self.proj_read = nn.Linear(d_model, d_model, bias=False)
        self.decay = nn.Parameter(torch.tensor(4.6))  # σ(4.6) ≈ 0.99

        if learned_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(memory_alpha).log())
        else:
            self.register_buffer("log_alpha", torch.tensor(memory_alpha).log())

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, out):
        """Chunkwise parallel Hebbian memory. O(TC·D + T·D²) time.

        Args:
            out: (B, T, D) hidden states from backbone.

        Returns:
            (B, T, D) hidden states augmented with memory reads.
        """
        B, T, D = out.shape
        C = self.chunk_size
        out32 = out.float()

        gamma = torch.sigmoid(self.decay)
        log_gamma = gamma.log()

        v = self.proj_write(out32)
        wk = F.pad(out32[:, :-1], (0, 0, 1, 0))
        rk = out32

        W = out32.new_zeros(B, D, D)
        reads_list = []

        for start in range(0, T, C):
            end = min(start + C, T)
            Ci = end - start
            p = torch.arange(Ci, device=out.device)

            rk_c, wk_c, v_c = rk[:, start:end], wk[:, start:end], v[:, start:end]

            # Inter-chunk: γ^l * (W_prev @ rk_c[l])
            inter = torch.matmul(W, rk_c.transpose(1, 2)).transpose(1, 2)
            inter = inter * (gamma ** p)[None, :, None]

            # Intra-chunk: (M ⊙ S) @ v
            S = torch.bmm(rk_c, wk_c.transpose(1, 2))
            diffs = (p[:, None] - 1 - p[None, :]).clamp(min=0)
            causal = p[:, None] > p[None, :]
            M = torch.exp(diffs * log_gamma) * causal
            intra = torch.bmm(S * M, v_c)

            reads_list.append(inter + intra)

            # Advance W: γ^Ci · W + Σ_l γ^(Ci-1-l) · v[l] ⊗ wk[l]
            gw = (gamma ** (Ci - 1 - p))[None, :, None]
            W = gamma ** Ci * W + torch.bmm((v_c * gw).transpose(1, 2), wk_c)

        reads = torch.cat(reads_list, dim=1)
        return out + self.alpha * self.proj_read(reads).to(out.dtype)

    def step(self, out, state=None):
        """Recurrent form: W ← γW + v⊗k, read = W·q. O(D²) per step.

        Args:
            out: (B, D) hidden state from backbone.
            state: dict with 'W' and 'r_prev', or None.

        Returns:
            (B, D) augmented hidden state, new state dict.
        """
        B, D = out.shape

        if state is None:
            W = out.new_zeros(B, D, D)
            r_prev = out.new_zeros(B, D)
        else:
            W = state["W"]
            r_prev = state["r_prev"]

        gamma = torch.sigmoid(self.decay)
        write = torch.einsum("bi,bj->bij", self.proj_write(out), r_prev)
        read = torch.einsum("bij,bj->bi", W, out)
        W = gamma * W + write

        augmented = out + self.alpha * self.proj_read(read)

        new_state = {"W": W, "r_prev": out}
        return augmented, new_state
