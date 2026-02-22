"""Plot learned decay γ and effective memory window per layer for 18M and 100M code models."""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from model import HebbianMamba


def get_gammas(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = HebbianMamba(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    gammas = []
    for layer in model.layers:
        if layer.use_memory:
            g = torch.sigmoid(layer.decay).item()
        else:
            g = None
        gammas.append(g)
    return gammas


device = "mps" if torch.backends.mps.is_available() else "cpu"

g18  = get_gammas("checkpoints/model_code_memory.pt",   device)
g100 = get_gammas("checkpoints/model_code100M_memory.pt", device)

fig, ax = plt.subplots(figsize=(7, 4))

for gammas, label, color, marker in [
    (g18,  "18M (d=512, 8L)",   "#1f77b4", "o"),
    (g100, "100M (d=1024, 12L)", "#d62728", "s"),
]:
    n = len(gammas)
    # Normalise layer index to [0, 1] so both models share the same x-axis
    xs      = [i / (n - 1) for i in range(n)]
    windows = [1 / (1 - g) for g in gammas]
    ax.plot(xs, windows, marker=marker, markersize=6, linewidth=2,
            color=color, label=label)

ax.set_xlabel("Relative layer depth (0 = first, 1 = last)")
ax.set_ylabel("Effective window (steps)")
ax.set_title("Learned memory window per layer")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/decay_gamma.pdf", dpi=150)
fig.savefig("figures/decay_gamma.png", dpi=150)
print("Saved figures/decay_gamma.pdf and .png")
