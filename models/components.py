"""Reusable components for delta rule models.

- CausalConv: causal depthwise conv1d for local token mixing
- GatedMLP: SwiGLU gated projections for channel mixing
- SlidingWindowAttention: local multi-head attention with RoPE
- DeltaBlock: multi-head delta rule memory with 1-4 state matrices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv(nn.Module):
    """Causal depthwise conv1d with parallel and recurrent forms."""

    def __init__(self, d: int, d_conv: int = 4):
        super().__init__()
        self.d = d
        self.d_conv = d_conv
        self.conv1d = nn.Conv1d(d, d, d_conv, bias=True, groups=d, padding=d_conv - 1)

    def forward(self, x):
        """(B, L, D) -> (B, L, D)."""
        return self.conv1d(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)

    def step(self, x, state=None):
        """(B, D) -> (B, D), state."""
        if state is None:
            state = x.new_zeros(x.shape[0], self.d, self.d_conv - 1)
        conv_input = torch.cat([state, x.unsqueeze(-1)], dim=-1)
        state = conv_input[:, :, 1:]
        assert self.conv1d.bias is not None
        out = (conv_input * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias
        return out, state


class GatedMLP(nn.Module):
    """SwiGLU gated projections: up-project, gate, down-project."""

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.proj = nn.Linear(d_model, d_inner, bias=False)
        self.gate = nn.Linear(d_model, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def project_up(self, x):
        """(B, *, d_model) -> (B, *, d_inner)."""
        return self.proj(x)

    def forward(self, x, val):
        """Gate and project down. x: original input for gate, val: transformed value in d_inner space."""
        return self.out_proj(F.silu(val) * F.silu(self.gate(x)))


class SlidingWindowAttention(nn.Module):
    """Local multi-head attention with RoPE and output gate.

    Standard Q/K/V projections with rotary position embeddings.
    Token shift: K uses previous position's projection.
    """

    def __init__(self, d_model: int, num_heads: int = 8, window_size: int = 256, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, num_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # precompute RoPE frequencies
        d = self.head_dim
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
        t = torch.arange(max_seq_len)
        angles = torch.outer(t, freqs)  # (T, d/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def _rope(self, x, offset=0):
        """Apply rotary embeddings. x: (B, H, T, d)."""
        T = x.shape[2]
        cos = self.cos[offset:offset + T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d/2)
        sin = self.sin[offset:offset + T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, x, attn_mask=None):
        """x: (B, T, D). attn_mask: optional (T, T) bool mask. returns: (B, T, D)."""
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)   # (B, H, T, d)
        k = self.k_proj(F.pad(x[:, :-1], (0, 0, 1, 0))).view(B, T, H, d).transpose(1, 2)
        v = self.proj_v(x).view(B, T, H, d).transpose(1, 2)
        gate = self.gate_proj(x).sigmoid().view(B, T, H, 1)

        q = self._rope(q)
        k = self._rope(k)

        if attn_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2) * gate
        return self.out_proj(out.reshape(B, T, D))

    def step(self, x, state=None):
        """x: (B, D). returns: (B, D), new state."""
        B, D = x.shape
        H, hd, W = self.num_heads, self.head_dim, self.window_size

        if state is None:
            state = {"keys": x.new_zeros(B, 0, H, hd), "vals": x.new_zeros(B, 0, H, hd), "pos": 0}

        pos = state["pos"]
        k_cur = self.k_proj(x).view(B, H, hd)

        if state["keys"].shape[1] > 0:
            keys = state["keys"][:, -W:]
            vals = state["vals"][:, -W:]
        else:
            keys = state["keys"]
            vals = state["vals"]

        gate = self.gate_proj(x).sigmoid().view(B, H, 1)
        if keys.shape[1] > 0:
            q = self.q_proj(x).view(B, H, hd)
            q = self._rope(q.unsqueeze(2), offset=pos).squeeze(2)

            K = keys.shape[1]
            k_roped = self._rope(keys.transpose(1, 2), offset=max(0, pos - K))

            scale = hd ** -0.5
            scores = (q.unsqueeze(2) @ k_roped.transpose(-1, -2)).squeeze(2) * scale
            attn = F.softmax(scores, dim=-1)
            v_mh = vals.transpose(1, 2)
            out = (attn.unsqueeze(-1) * v_mh).sum(2) * gate
            out = self.out_proj(out.reshape(B, D))
        else:
            out = x.new_zeros(B, D)

        new_keys = torch.cat([state["keys"], k_cur.unsqueeze(1)], dim=1)[:, -W:]
        new_vals = torch.cat([state["vals"], self.proj_v(x).view(B, H, hd).unsqueeze(1)], dim=1)[:, -W:]

        return out, {"keys": new_keys, "vals": new_vals, "pos": pos + 1}


class DeltaBlock(nn.Module):
    """Multi-matrix delta rule memory with data-dependent decay.

    Supports 1-4 state matrices with shared token-shifted keys.
    When num_matrices > 1, separate WY corrections per matrix and
    softmax blend gate for read weighting.

    delta rule: S_t = e^g · S_{t-1} + β(v - S·wk)·wk^T
    token shift: write key = previous position, read key = current.
    """

    def __init__(self, d_model: int, num_heads: int = 8, chunk_size: int = 64, num_matrices: int = 1):
        super().__init__()
        assert d_model % num_heads == 0
        assert 1 <= num_matrices <= 4
        self.d_model = d_model
        self.n_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        self.num_matrices = num_matrices
        H, M = num_heads, num_matrices

        self.proj_write = nn.Linear(d_model, M * d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, H, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if M > 1:
            self.blend_proj = nn.Linear(d_model, H * M, bias=False)

        # per-matrix write gates and decay
        self.beta_projs = nn.ModuleList([nn.Linear(d_model, H, bias=False) for _ in range(M)])
        self.alpha_projs = nn.ModuleList([nn.Linear(d_model, H, bias=False) for _ in range(M)])
        self.dt_biases = nn.ParameterList()
        self.A_logs = nn.ParameterList()
        for _ in range(M):
            dt = torch.empty(H).uniform_(0, 1) * (torch.tensor(0.1).log() - torch.tensor(0.001).log()) + torch.tensor(0.001).log()
            dt = dt.exp()
            dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
            dt_bias._no_weight_decay = True
            self.dt_biases.append(dt_bias)
            A_log = nn.Parameter(torch.empty(H).uniform_(0, 4).log())
            A_log._no_weight_decay = True
            self.A_logs.append(A_log)

        C = chunk_size
        self.register_buffer("causal_mask", torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0), persistent=False)
        self.register_buffer("eye_C", torch.eye(C), persistent=False)

    def forward(self, out):
        """out: (B, T, D). returns: (B, T, D)."""
        B, T, D = out.shape
        H, d, C = self.n_heads, self.head_dim, self.chunk_size
        M = self.num_matrices
        x = out.float()

        # -- projections --
        vals = self.proj_write(x).view(B, T, M, H, d)
        vs = [vals[:, :, m] for m in range(M)]
        gate = self.gate_proj(x).sigmoid().view(B, T, H, 1)

        betas = [self.beta_projs[m](x).sigmoid() for m in range(M)]
        decays = [-self.A_logs[m].exp().view(1, 1, H) * F.softplus(self.alpha_projs[m](x) + self.dt_biases[m]) for m in range(M)]

        if M > 1:
            blend = self.blend_proj(x).view(B, T, H, M).softmax(dim=-1)

        # normalized keys with token shift
        rk = F.normalize(x.view(B, T, H, d), dim=-1)
        wk = F.pad(rk[:, :-1], (0, 0, 0, 0, 1, 0))

        # transpose to (B, H, T, ...)
        rk, wk = rk.transpose(1, 2).float(), wk.transpose(1, 2).float()
        vs = [v.transpose(1, 2).float() for v in vs]
        if M > 1:
            blend = blend.permute(0, 2, 1, 3).float()

        # scale by beta
        vs = [vs[m] * betas[m].transpose(1, 2).unsqueeze(-1) for m in range(M)]
        wk_betas = [wk * betas[m].transpose(1, 2).unsqueeze(-1) for m in range(M)]

        # -- pad to chunk boundary --
        pad = (C - (T % C)) % C
        if pad > 0:
            rk = F.pad(rk, (0, 0, 0, pad))
            wk = F.pad(wk, (0, 0, 0, pad))
            for m in range(M):
                vs[m] = F.pad(vs[m], (0, 0, 0, pad))
                wk_betas[m] = F.pad(wk_betas[m], (0, 0, 0, pad))
                decays[m] = F.pad(decays[m], (0, 0, 0, pad))
            if M > 1:
                blend = F.pad(blend, (0, 0, 0, pad))

        T_pad = rk.shape[2]
        N = T_pad // C

        # -- reshape into chunks --
        rk = rk.view(B, H, N, C, d)
        wk = wk.view(B, H, N, C, d)
        for m in range(M):
            vs[m] = vs[m].view(B, H, N, C, d)
            wk_betas[m] = wk_betas[m].view(B, H, N, C, d)
            decays[m] = decays[m].transpose(1, 2).view(B, H, N, C)
        if M > 1:
            blend = blend.view(B, H, N, C, M)

        # -- per-matrix decay masks and WY corrections --
        cums, decay_exps, Ls = [], [], []
        v_corrs, wk_cumdecays, intras = [], [], []
        rk_wk = rk @ wk.transpose(-1, -2)
        causal_upper = torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=1)

        for m in range(M):
            cum = decays[m].cumsum(-1)
            cums.append(cum)
            decay_exps.append(cum.unsqueeze(-1).exp())
            L = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).tril().exp().tril()
            Ls.append(L)

            A = -(wk_betas[m] @ wk.transpose(-1, -2) * L).masked_fill(self.causal_mask, 0)
            A = A.clone()
            for i in range(1, C):
                A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :i].clone().unsqueeze(-1) * A[..., :i, :i].clone()).sum(-2)
            A = A + self.eye_C

            v_corrs.append(A @ vs[m])
            wk_cumdecays.append(A @ (wk_betas[m] * decay_exps[m]))
            intras.append((rk_wk * L).masked_fill(causal_upper, 0))

        # -- chunk-by-chunk propagation --
        Ss = [x.new_zeros(B, H, d, d) for _ in range(M)]
        o = x.new_zeros(B, H, N, C, d)

        for i in range(N):
            rk_i, wk_i = rk[:, :, i], wk[:, :, i]
            os = []

            for m in range(M):
                v_new = v_corrs[m][:, :, i] - wk_cumdecays[m][:, :, i] @ Ss[m]
                o_m = (rk_i * decay_exps[m][:, :, i]).unsqueeze(-2) @ Ss[m].unsqueeze(-3)
                o_m = o_m.squeeze(-2) + intras[m][:, :, i] @ v_new
                dw = (cums[m][:, :, i, -1, None] - cums[m][:, :, i]).exp().unsqueeze(-1)
                Ss[m] = Ss[m] * cums[m][:, :, i, -1, None, None].exp() + (wk_i * dw).transpose(-1, -2) @ v_new
                S_norm = Ss[m].norm(dim=(-2, -1), keepdim=True)
                Ss[m] = Ss[m] * (S_norm.clamp(max=100.0) / S_norm.clamp(min=1e-6))
                os.append(o_m)

            if M == 1:
                o[:, :, i] = os[0]
            else:
                o[:, :, i] = sum(blend[:, :, i, :, m].unsqueeze(-1) * os[m] for m in range(M))

        # -- output --
        o = o.view(B, H, T_pad, d).transpose(1, 2)[:, :T]
        o = o * gate
        return out + self.out_proj(o.reshape(B, T, D)).to(out.dtype)

    def step(self, out, state=None):
        """Sequential recurrence. out: (B, D). returns: (B, D), state."""
        B, D = out.shape
        H, d = self.n_heads, self.head_dim
        M = self.num_matrices

        vals = self.proj_write(out).view(B, H, M, d)
        vs = [vals[:, :, m] for m in range(M)]
        gate = self.gate_proj(out).sigmoid().view(B, H, 1)
        betas = [self.beta_projs[m](out).sigmoid().unsqueeze(-1) for m in range(M)]
        decays = [(-self.A_logs[m].exp() * F.softplus(self.alpha_projs[m](out) + self.dt_biases[m])).exp().view(B, H, 1, 1) for m in range(M)]

        rk = F.normalize(out.view(B, H, d), dim=-1)

        if state is not None:
            Ws = [state[f"W{m}"] for m in range(M)]
            wk = state["wk"]
        else:
            Ws = [out.new_zeros(B, H, d, d) for _ in range(M)]
            wk = out.new_zeros(B, H, d)

        reads = []
        for m in range(M):
            Ws[m] = Ws[m] * decays[m]
            err = (vs[m] - (Ws[m] * wk.unsqueeze(-1)).sum(-2)) * betas[m]
            Ws[m] = Ws[m] + wk.unsqueeze(-1) * err.unsqueeze(-2)
            reads.append((Ws[m] * rk.unsqueeze(-1)).sum(-2))

        if M == 1:
            read = reads[0]
        else:
            blend = self.blend_proj(out).view(B, H, M).softmax(dim=-1)
            read = sum(blend[:, :, m:m+1] * reads[m] for m in range(M))

        read = read * gate
        read = self.out_proj(read.reshape(B, D))

        new_state = {f"W{m}": Ws[m] for m in range(M)}
        new_state["wk"] = rk
        return out + read, new_state
