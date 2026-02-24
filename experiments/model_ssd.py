"""SSD (Mamba-2) model with optional full-stack looping.

Stack looping reruns the entire layer stack K times with tied weights:
    x = embed(tokens)
    for k in range(stack_loops):
        for layer in layers:
            x = layer(x)
    logits = lm_head(norm(x))

Full gradient flows through all passes (like Universal Transformers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import RMSNorm

from model import Config


class SSDBlock(nn.Module):
    """SSD (Mamba-2 dual form) block — pure PyTorch, chunked.

    Projects input to value, gate, B, C, dt. Builds per-chunk causal
    attention matrix M and applies to values, carrying SSM state across chunks.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        E = cfg.expand
        self.d_model = D
        self.d_inner = E * D
        self.d_state = cfg.d_state
        self.d_conv = cfg.d_conv
        self.n_heads = getattr(cfg, "n_heads", self.d_inner // 128)
        self.head_dim = self.d_inner // self.n_heads
        self.chunk_size = getattr(cfg, "ssd_chunk_size", 256)
        H = self.n_heads
        N = self.d_state

        # Single projection: value(ED) + gate(ED) + B(H*N) + C(H*N) + dt(H)
        self.in_proj = nn.Linear(D, 2 * self.d_inner + 2 * H * N + H, bias=False)

        # Depthwise conv on value branch
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, cfg.d_conv,
            bias=True, groups=self.d_inner, padding=cfg.d_conv - 1,
        )

        # A parameter (per-head log decay)
        self.A_log = nn.Parameter(torch.log(torch.ones(H) * 2.0))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, D, bias=False)

        # dt bias
        self.dt_bias = nn.Parameter(torch.zeros(H))

    def forward(self, x):
        """x: (B, L, D) -> (B, L, D)"""
        B, L, D = x.shape
        H = self.n_heads
        N = self.d_state
        HD = self.head_dim
        CS = self.chunk_size

        # Project input
        proj = self.in_proj(x)
        val, gate, B_proj, C_proj, dt = proj.split(
            [self.d_inner, self.d_inner, H * N, H * N, H], dim=-1
        )

        # Conv1d on value branch
        val = val.transpose(1, 2)
        val = self.conv1d(val)[:, :, :L]
        val = val.transpose(1, 2)
        val = F.silu(val)

        # Reshape for multi-head
        val = val.view(B, L, H, HD)
        B_proj = B_proj.view(B, L, H, N)
        C_proj = C_proj.view(B, L, H, N)
        dt = F.softplus(dt + self.dt_bias)

        # Per-step log decay
        A = -torch.exp(self.A_log)
        log_decay = dt * A

        # Chunked SSD in float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            val32 = val.float()
            B32 = B_proj.float()
            C32 = C_proj.float()
            ld32 = log_decay.float()

            h = x.new_zeros(B, H, HD, N, dtype=torch.float32)
            Y_chunks = []

            for start in range(0, L, CS):
                end = min(start + CS, L)
                Ci = end - start

                val_c = val32[:, start:end]
                B_c = B32[:, start:end]
                C_c = C32[:, start:end]
                ld_c = ld32[:, start:end]

                # Intra-chunk: build Ci×Ci causal M and apply
                cum_c = torch.cumsum(ld_c, dim=1)
                cum_ch = cum_c.permute(0, 2, 1)
                decay_diff = cum_ch.unsqueeze(-1) - cum_ch.unsqueeze(-2)
                causal_mask = torch.ones(Ci, Ci, device=x.device, dtype=torch.bool).tril()
                decay_diff = decay_diff.masked_fill(~causal_mask, float("-inf"))
                CB = torch.einsum("blhn,bmhn->bhlm", C_c, B_c)
                M = CB * torch.exp(decay_diff)

                val_ch = val_c.permute(0, 2, 1, 3)
                Y_intra = torch.matmul(M, val_ch)

                # Inter-chunk: contribution from carried state h
                decay_to_pos = torch.exp(cum_c)
                Y_inter = torch.einsum("blhn,bhdn,blh->bhld",
                                       C_c, h, decay_to_pos)

                Y_chunk = Y_intra + Y_inter
                Y_chunk = Y_chunk + self.D.view(1, H, 1, HD) * val_ch
                Y_chunks.append(Y_chunk)

                # Update state h for next chunk
                total_decay = torch.exp(cum_c[:, -1, :])
                h = total_decay.unsqueeze(-1).unsqueeze(-1) * h
                remaining = torch.exp(cum_c[:, -1:, :] - cum_c)
                h = h + torch.einsum("blhd,blhn,blh->bhdn",
                                     val_c, B_c, remaining)

            Y = torch.cat(Y_chunks, dim=2)

        Y = Y.permute(0, 2, 1, 3).contiguous().view(B, L, self.d_inner)
        Y = Y * F.silu(gate)
        return self.out_proj(Y)

    def step(self, x, state=None):
        """Recurrent form: x (B, D) -> y (B, D), new_state."""
        B = x.shape[0]
        H = self.n_heads
        N = self.d_state
        HD = self.head_dim

        if state is None:
            conv_state = x.new_zeros(B, self.d_inner, self.d_conv - 1)
            h = x.new_zeros(B, H, HD, N)
        else:
            conv_state = state["conv_state"]
            h = state["h"]

        proj = self.in_proj(x)
        val, gate, B_proj, C_proj, dt = proj.split(
            [self.d_inner, self.d_inner, H * N, H * N, H], dim=-1
        )

        # Conv1d step
        conv_input = torch.cat([conv_state, val.unsqueeze(-1)], dim=-1)
        conv_state = conv_input[:, :, 1:]
        val = (conv_input * self.conv1d.weight.squeeze(1)).sum(dim=-1) + self.conv1d.bias
        val = F.silu(val)

        val = val.view(B, H, HD)
        B_proj = B_proj.view(B, H, N)
        C_proj = C_proj.view(B, H, N)
        dt = F.softplus(dt + self.dt_bias)

        A = -torch.exp(self.A_log)
        dA = torch.exp(dt * A)

        # SSM recurrence
        h = dA.unsqueeze(-1).unsqueeze(-1) * h + torch.einsum("bhd,bhn->bhdn", val, B_proj)
        y = torch.einsum("bhn,bhdn->bhd", C_proj, h)

        # Skip connection with D
        y = y + self.D.view(1, H, HD) * val

        y = y.view(B, self.d_inner)
        y = y * F.silu(gate)
        y = self.out_proj(y)

        return y, {"conv_state": conv_state, "h": h}


class LinearBlock(nn.Module):
    """Trivial backbone: linear projection + SiLU. No sequence modeling."""

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        E = cfg.expand
        self.proj = nn.Linear(D, E * D, bias=False)
        self.gate = nn.Linear(D, E * D, bias=False)
        self.out_proj = nn.Linear(E * D, D, bias=False)

    def forward(self, x):
        return self.out_proj(F.silu(self.gate(x)) * self.proj(x))

    def step(self, x, state=None):
        return self.forward(x.unsqueeze(1)).squeeze(1), {}


class ConvBlock(nn.Module):
    """Linear backbone + depthwise conv. Local mixing, no SSM recurrence."""

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        E = cfg.expand
        d_inner = E * D
        self.d_inner = d_inner
        self.d_conv = cfg.d_conv

        self.proj = nn.Linear(D, d_inner, bias=False)
        self.gate = nn.Linear(D, d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, cfg.d_conv,
            bias=True, groups=d_inner, padding=cfg.d_conv - 1,
        )
        self.out_proj = nn.Linear(d_inner, D, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        val = self.proj(x).transpose(1, 2)
        val = self.conv1d(val)[:, :, :L].transpose(1, 2)
        val = F.silu(val)
        return self.out_proj(val * F.silu(self.gate(x)))

    def step(self, x, state=None):
        B = x.shape[0]
        if state is None:
            conv_state = x.new_zeros(B, self.d_inner, self.d_conv - 1)
        else:
            conv_state = state["conv_state"]

        val = self.proj(x)
        conv_input = torch.cat([conv_state, val.unsqueeze(-1)], dim=-1)
        conv_state = conv_input[:, :, 1:]
        val = (conv_input * self.conv1d.weight.squeeze(1)).sum(dim=-1) + self.conv1d.bias
        val = F.silu(val)

        gate = F.silu(self.gate(x))
        y = self.out_proj(val * gate)
        return y, {"conv_state": conv_state}


def _memory_attend(out, proj_write, proj_read, decay, chunk_size, memory_alpha, write_alpha=None):
    """Chunkwise parallel Hebbian memory."""
    B, T, D = out.shape
    C = chunk_size
    w_alpha = write_alpha if write_alpha is not None else memory_alpha
    if w_alpha == 0 and memory_alpha == 0:
        return out
    out32 = out.float()

    gamma = torch.sigmoid(decay)
    log_gamma = gamma.log()

    v = proj_write(out32)
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
        v_c = v[:, start:end]

        inter = torch.matmul(W, rk_c.transpose(1, 2)).transpose(1, 2)
        inter = inter * (gamma ** p)[None, :, None]

        S = torch.bmm(rk_c, wk_c.transpose(1, 2))
        diffs_c = (p[:, None] - 1 - p[None, :]).clamp(min=0)
        causal_c = p[:, None] > p[None, :]
        M_c = torch.exp(diffs_c * log_gamma) * causal_c
        intra = torch.bmm(S * M_c, v_c)

        reads_list.append(inter + intra)

        if w_alpha > 0:
            gw = (gamma ** (Ci - 1 - p))[None, :, None]
            W = gamma ** Ci * W + torch.bmm((v_c * gw).transpose(1, 2), wk_c)

    all_reads = torch.cat(reads_list, dim=1)
    return out + memory_alpha * proj_read(all_reads).to(out.dtype)


def _memory_step(out, r_prev, W, proj_write, proj_read, decay, memory_alpha, write_alpha=None):
    """Recurrent Hebbian memory step."""
    w_alpha = write_alpha if write_alpha is not None else memory_alpha
    gamma = torch.sigmoid(decay)
    read = torch.einsum("bij,bj->bi", W, out)
    if w_alpha > 0:
        write = torch.einsum("bi,bj->bij", proj_write(out), r_prev)
        W = gamma * W + write
    else:
        W = gamma * W
    out = out + memory_alpha * proj_read(read)
    return out, W


class SSDLayer(nn.Module):
    """Single SSD pass + Hebbian memory."""

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.d_model = D
        self.memory_alpha = cfg.memory_alpha
        self.chunk_size = cfg.chunk_size

        n_norms = getattr(cfg, "stack_loops", 1)
        self.norms = nn.ModuleList([RMSNorm(D) for _ in range(n_norms)])
        backbone = getattr(cfg, "backbone", "ssd")
        if backbone == "linear":
            self.ssd = LinearBlock(cfg)
        elif backbone == "conv":
            self.ssd = ConvBlock(cfg)
        else:
            self.ssd = SSDBlock(cfg)

        self.proj_write = nn.Linear(D, D, bias=False)
        self.proj_read = nn.Linear(D, D, bias=False)
        self.decay = nn.Parameter(torch.tensor(4.6))

    def forward(self, x, memory_alpha=None, write_alpha=None, loop_idx=0):
        residual = x
        alpha = memory_alpha if memory_alpha is not None else self.memory_alpha
        norm = self.norms[min(loop_idx, len(self.norms) - 1)]
        out = self.ssd(norm(x))
        out = _memory_attend(out, self.proj_write, self.proj_read,
                             self.decay, self.chunk_size, alpha, write_alpha=write_alpha)
        return residual + out

    def step(self, x, state=None, memory_alpha=None, write_alpha=None, loop_idx=0):
        B = x.shape[0]
        residual = x
        alpha = memory_alpha if memory_alpha is not None else self.memory_alpha
        norm = self.norms[min(loop_idx, len(self.norms) - 1)]

        ssd_state = state["ssd"] if state else None
        out, ssd_state = self.ssd.step(norm(x), ssd_state)

        W = state["memory"] if state else x.new_zeros(B, self.d_model, self.d_model)
        r_prev = state["r_prev"] if state else x.new_zeros(B, self.d_model)

        raw_out = out
        out, W = _memory_step(out, r_prev, W, self.proj_write, self.proj_read,
                              self.decay, alpha, write_alpha=write_alpha)

        return residual + out, {"ssd": ssd_state, "memory": W, "r_prev": raw_out}


class HebbianMambaLoopSSD(nn.Module):
    """HebbianMamba with SSD blocks. Optional partial-stack looping.

    stack_loops=2 reruns layers[loop_start:loop_end] twice with a learned
    gate (init near zero). Boundary layers run once. Full gradient through
    all passes (Universal Transformer style).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.stack_loops = getattr(cfg, "stack_loops", 1)
        n = cfg.n_layers
        self.loop_start = getattr(cfg, "loop_start", 0)
        self.loop_end = getattr(cfg, "loop_end", n)

        self.memory_alpha = cfg.memory_alpha
        self.loop_alpha = getattr(cfg, "loop_alpha", cfg.memory_alpha)

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([SSDLayer(cfg) for _ in range(n)])

        if self.stack_loops > 1:
            gate_init = getattr(cfg, "gate_init", -5.0)
            self.loop_gate = nn.Parameter(torch.tensor(gate_init))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        # Pre-loop boundary layers
        for layer in self.layers[:self.loop_start]:
            x = layer(x, loop_idx=0)
        # Looped middle layers
        for k in range(self.stack_loops):
            alpha = self.memory_alpha if k == 0 else self.loop_alpha
            w_alpha = None if k == 0 else 0.0  # read-only on pass 2+
            x_before = x
            for layer in self.layers[self.loop_start:self.loop_end]:
                x = layer(x, memory_alpha=alpha, write_alpha=w_alpha, loop_idx=k)
            if k > 0:
                gate = torch.sigmoid(self.loop_gate)
                x = x_before + gate * (x - x_before)
        # Post-loop boundary layers
        for layer in self.layers[self.loop_end:]:
            x = layer(x, loop_idx=0)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def step(self, token, states=None):
        x = self.embedding(token)
        new_states = []
        si = 0  # state index

        # Pre-loop boundary layers
        for layer in self.layers[:self.loop_start]:
            x, s = layer.step(x, state=states[si] if states else None, loop_idx=0)
            new_states.append(s)
            si += 1

        # Looped middle layers
        for k in range(self.stack_loops):
            alpha = self.memory_alpha if k == 0 else self.loop_alpha
            w_alpha = None if k == 0 else 0.0  # read-only on pass 2+
            x_before = x
            for layer in self.layers[self.loop_start:self.loop_end]:
                x, s = layer.step(x, state=states[si] if states else None, memory_alpha=alpha, write_alpha=w_alpha, loop_idx=k)
                new_states.append(s)
                si += 1
            if k > 0:
                gate = torch.sigmoid(self.loop_gate)
                x = x_before + gate * (x - x_before)

        # Post-loop boundary layers
        for layer in self.layers[self.loop_end:]:
            x, s = layer.step(x, state=states[si] if states else None, loop_idx=0)
            new_states.append(s)
            si += 1

        return self.lm_head(self.norm(x)), new_states

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
