"""Generate 100M long-context per-segment chart for paper."""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "eval_long_context/16k_code_100M.txt"
SEG_TOKENS = 512   # each segment covers 512 tokens
N_SEGS = 32        # 32 × 512 = 16384

# Parse the overall section from the log
memory_vals, deep_vals = [], []
in_overall = False
for line in open(LOG):
    if "=== Overall" in line:
        in_overall = True
        continue
    if in_overall and "OVERALL" in line:
        break
    if in_overall and re.match(r"\s+\d+K-\d+K", line):
        parts = line.split()
        memory_vals.append(float(parts[1]))
        deep_vals.append(float(parts[2]))

assert len(memory_vals) == N_SEGS, f"Expected {N_SEGS} segments, got {len(memory_vals)}"

# x = midpoint of each 512-token segment, in tokens
x = [(i + 0.5) * SEG_TOKENS for i in range(N_SEGS)]
x_k = [xi / 1000 for xi in x]  # in K tokens

gap = [d - m for m, d in zip(memory_vals, deep_vals)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})

ax1.plot(x_k, memory_vals, "-", color="#1f77b4", label="Memory (d=1024, 12L)",
         linewidth=1.8, marker="o", markersize=3)
ax1.plot(x_k, deep_vals,   "--", color="#2ca02c", label="Deep baseline (d=1024, 16L)",
         linewidth=1.8, marker="s", markersize=3)
ax1.set_ylabel("Val loss (nats)")
ax1.set_title("100M model long-context evaluation — 8 × 16K code windows")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axvline(2.048, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Train length (2K)")

ax2.fill_between(x_k, gap, alpha=0.3, color="#d62728")
ax2.plot(x_k, gap, "-", color="#d62728", linewidth=1.5)
ax2.axhline(0, color="gray", linewidth=0.8)
ax2.axvline(2.048, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax2.set_xlabel("Context position (K tokens)")
ax2.set_ylabel("Gap (nats)")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("figures/long_context_100M.pdf", dpi=150)
fig.savefig("figures/long_context_100M.png", dpi=150)
print("Saved figures/long_context_100M.pdf and .png")
