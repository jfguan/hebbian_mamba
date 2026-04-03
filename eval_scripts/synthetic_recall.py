"""Delta memory capacity benchmark.

Train on KV pairs, then measure recall accuracy at increasing pair counts.
Compares 1 matrix vs 2 matrices (shared keys) to test if dual state
increases effective capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.components import CausalConv, GatedMLP, DeltaBlock

# -- config --
N_KEYS = 256
N_VALS = 16
VOCAB = N_KEYS + N_VALS
TRAIN_PAIRS = 32
D_MODEL = 128
N_LAYERS = 4
SWEEP_PAIRS = [4, 8, 16, 32, 64, 96, 128, 160, 192, 256]


def main():
    torch.manual_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    results = {}
    for label, num_matrices, rotate_keys in [
        # ("1 matrix", 1, False),
        # ("2 matrices", 2, False),
        ("2 matrices (rotated keys)", 2, True),
    ]:
        print(f"\n=== {label} ===")
        model = Model(D_MODEL, N_LAYERS, num_matrices=num_matrices, rotate_keys=rotate_keys).to(device)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"params={params:.2f}M  train_pairs={TRAIN_PAIRS}")

        train(model, device)
        results[label] = sweep(model, device)

    plot(results, "eval_results/synthetic_recall.png")


def make_batch(batch_size, num_pairs, device):
    """Generate a KV recall task.

    Store phase: [k1, v1, k2, v2, ...].
    Query phase: [k?, _, k?, _, ...] (reshuffled).
    returns: (input_ids, targets, mask) where mask marks value positions in query.
    """
    B, P = batch_size, num_pairs

    keys = torch.stack([torch.randperm(N_KEYS, device=device)[:P] for _ in range(B)])
    values = torch.randint(N_KEYS, VOCAB, (B, P), device=device)

    store_seq = torch.stack([keys, values], dim=-1).view(B, 2 * P)

    shuffle = torch.stack([torch.randperm(P, device=device) for _ in range(B)])
    query_seq = torch.stack([
        keys.gather(1, shuffle),
        values.gather(1, shuffle),
    ], dim=-1).view(B, 2 * P)

    input_ids = torch.cat([store_seq, query_seq], dim=1)
    targets = input_ids[:, 1:]
    mask = torch.zeros_like(targets, dtype=torch.float)
    mask[:, 2 * P::2] = 1.0
    return input_ids, targets, mask


def train(model, device, steps=1500):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        x, targets, mask = make_batch(64, TRAIN_PAIRS, device)
        logits = model(x)[:, :-1]
        per_token = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none")
        loss = (per_token.view_as(mask) * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                preds = logits.argmax(-1)
                acc = ((preds == targets) * mask).sum() / mask.sum()
            print(f"  step {step:4d}  loss={loss.item():.4f}  recall={acc.item():.1%}")


def sweep(model, device):
    print("  capacity sweep:")
    model.eval()
    results = {}
    with torch.no_grad():
        for n in SWEEP_PAIRS:
            accs = []
            for _ in range(20):
                x, targets, mask = make_batch(64, n, device)
                preds = model(x)[:, :-1].argmax(-1)
                accs.append(((preds == targets) * mask).sum() / mask.sum())
            results[n] = torch.stack(accs).mean().item()
            print(f"    {n:3d} pairs: {results[n]:.1%}")
    return results


def plot(results, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"1 matrix": "#ff7f0e", "2 matrices": "#2ca02c", "2 matrices (rotated keys)": "#1f77b4"}

    for label, res in results.items():
        pairs = list(res.keys())
        acc = [res[p] * 100 for p in pairs]
        ax.plot(pairs, acc, "o-", label=label, color=colors[label], linewidth=2, markersize=6)

    ax.axhline(y=100 / N_VALS, color="r", ls="--", alpha=0.5, label=f"chance ({100/N_VALS:.1f}%)")
    ax.axvline(x=D_MODEL // 4, color="gray", ls="--", alpha=0.5, label=f"head_dim={D_MODEL // 4}")
    ax.set(xlabel="Number of KV pairs", ylabel="Recall accuracy (%)",
           title=f"Delta memory capacity (d_model={D_MODEL})", ylim=(-5, 105))
    ax.set_xticks(SWEEP_PAIRS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# -- model --

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


class Model(nn.Module):
    def __init__(self, d_model, n_layers, d_conv=4, num_matrices=1, rotate_keys=False):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, d_model)
        self.layers = nn.ModuleList([Layer(d_model, d_conv, num_matrices, rotate_keys) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)
        self.embedding.weight = self.head.weight

    def forward(self, ids):
        x = self.embedding(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class Layer(nn.Module):
    def __init__(self, d_model, d_conv=4, num_matrices=1, rotate_keys=False, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.norm = nn.RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, expand=expand)
        self.conv = CausalConv(d_inner, d_conv=d_conv)
        self.memory = DeltaBlock(d_model, num_heads=4, num_matrices=num_matrices)
        self.rotate_keys = rotate_keys and num_matrices > 1

    def forward(self, x):
        normed = self.norm(x)
        out = self.mlp(normed, self.conv(self.mlp.project_up(normed)))
        if self.rotate_keys:
            out = self._forward_rotated(out)
        else:
            out = self.memory(out)
        return x + out

    def _forward_rotated(self, out):
        """Delta with rotated keys per matrix — sequential fallback."""
        mem = self.memory
        B, T, D = out.shape
        H, d, M = mem.n_heads, mem.head_dim, mem.num_matrices
        x = out.float()

        vals = mem.proj_write(x).view(B, T, M, H, d)
        gate = mem.gate_proj(x).sigmoid().view(B, T, H, 1)
        betas = [mem.beta_projs[m](x).sigmoid().view(B, T, H, 1) for m in range(M)]
        decays = [(-mem.A_logs[m].exp().view(1, 1, H) * F.softplus(mem.alpha_projs[m](x) + mem.dt_biases[m])).exp().view(B, T, H, 1, 1) for m in range(M)]
        blend = mem.blend_proj(x).view(B, T, H, M).softmax(dim=-1)

        rk_base = F.normalize(x.view(B, T, H, d), dim=-1)
        wk_base = F.pad(rk_base[:, :-1], (0, 0, 0, 0, 1, 0))

        Ss = [x.new_zeros(B, H, d, d) for _ in range(M)]
        o = x.new_zeros(B, T, H, d)

        for t in range(T):
            reads = []
            for m in range(M):
                wk_t = wk_base[:, t]
                rk_t = rk_base[:, t]
                for _ in range(m):
                    wk_t = rotate_half(wk_t)
                    rk_t = rotate_half(rk_t)

                Ss[m] = Ss[m] * decays[m][:, t]
                err = (vals[:, t, m] - (Ss[m] * wk_t.unsqueeze(-1)).sum(-2)) * betas[m][:, t]
                Ss[m] = Ss[m] + wk_t.unsqueeze(-1) * err.unsqueeze(-2)
                reads.append((Ss[m] * rk_t.unsqueeze(-1)).sum(-2))

            o[:, t] = sum(blend[:, t, :, m].unsqueeze(-1) * reads[m] for m in range(M)) * gate[:, t]

        return out + mem.out_proj(o.reshape(B, T, D)).to(out.dtype)


if __name__ == "__main__":
    main()
