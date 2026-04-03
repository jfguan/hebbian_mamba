# Token Shift vs QK Projections in Delta Rule Memory

## Key Finding

QK projections are redundant in delta rule architectures because the delta rule has three other selectivity mechanisms (decay, write gating, error correction) that together cover the same ground. Token-shifted raw hidden states outperform learned QK projections for the delta rule, but fail for softmax attention where QK is the only selectivity mechanism.

## Experimental Evidence

### 18M scale, 30M tokens (The Stack)

| Model | Val Loss | Params |
|---|---|---|
| GDN token-shift (no QK) | **1.455** | 14.7M |
| GDN (QK + SiLU + conv) | 1.763 | 17.9M |
| GDN no-SiLU on QK | 1.866 | 17.9M |
| Transformer (QK projections) | 1.763 | 21.5M |
| Transformer token-shift (no QK) | 2.347 | 17.3M |

### 100M scale, power law projections (The Stack)

| Model | Projected @ 200M tok | Projected @ 500M tok | Params |
|---|---|---|---|
| GDN token-shift (no QK) | **1.091** | **0.968** | 86M |
| GDN (QK + SiLU + conv) | 1.118 | 0.973 | 105M |

Token shift is ahead on the fitted curve at both projections, with 19M fewer params.

## Why QK Is Redundant for Delta Rule

Softmax attention has one selectivity mechanism: Q-K similarity determines which past positions to attend to. If addressing is blunt (token shift without projections), the output is blunt. Nothing compensates.

The delta rule has **four** selectivity mechanisms:

1. **Q-K addressing**: determines where to read/write in the state matrix
2. **Decay**: old associations fade automatically — temporal selectivity without key selectivity
3. **Beta (write gate)**: controls what gets written — content selectivity without key selectivity
4. **Error correction** (`v - S@k`): fixes interference when keys collide — cleans up blunt addressing

Mechanisms 2-4 together cover the selectivity that QK projections provide. Adding QK projections on top is a fourth mechanism for a problem already solved by three. The marginal benefit is small, and the cost is real:
- 2×D² params per layer (~17% of model) spent on addressing optimization
- Gradient bottleneck (gradients flow through proj_k instead of directly into residual stream)
- Non-stationary key space (every update to proj_k reorganizes where associations are stored)

## Why Token Shift Works

Token shift encodes "store current value under previous context's address" — a natural prior for autoregressive prediction. The association pattern `x_{t-1} → v_t` means "what follows this context." When a similar context appears later, the memory retrieves the associated continuation.

The token shift keys are the raw hidden states — the same representations flowing through the residual stream. No translation between key space and residual stream space. Direct gradient flow from memory loss into the hidden states.

## Why Token Shift Fails for Softmax Attention

Softmax attention selects from **individual past positions** — it's a search problem. QK projections let Q encode "what I'm looking for" and K encode "what I contain" in separate learned subspaces, enabling selective patterns like "closing bracket seeks opening bracket."

Without QK projections, softmax attention scores are raw cosine similarity between hidden states. The model can't learn selective retrieval patterns. Decay, write gating, and error correction don't exist in softmax attention — there's nothing to compensate for blunt addressing.

Token shift "sort of works" for softmax (2.35, well above random 6.93) because the basic pattern is useful. But it lacks the learned selectivity that QK provides, so it's much worse than QK (2.35 vs 1.76).

## SiLU on Q/K Is Not the Culprit

Removing SiLU from GDN's Q/K path made results worse (1.866 vs 1.763), not better. Despite SiLU compressing the key space (keys biased toward positive orthant after activation), the GDN model adapts its projections to produce well-spread keys within that constraint. SiLU serves a structural purpose: it makes the short conv on Q/K meaningful (without activation, conv + projection collapses to a single linear map).

Key diversity metrics (cosine similarity, effective rank, condition number) are misleading — they measure geometric properties of the key space but don't predict which model performs better.

## Implications

- For delta rule architectures: drop QK projections, use token shift. Save ~17% params, get equal or better quality.
- For softmax attention: QK projections remain essential.
- Future delta rule research should optimize decay, write gating, and correction — not addressing.
- The delta rule's advantage over softmax isn't just efficiency — it's that redundant selectivity mechanisms make the architecture robust to simpler components.

## Associative Recall Benchmark

Synthetic task: store 4 KV pairs, insert variable-length PAD distractors, query one pair. Train on distances 0-8, evaluate 0-1536+.

### Results

| Model | Distance 0 | Distance 64 | Distance 256 | Distance 1024 |
|---|---|---|---|---|
| GDN (delta + QK + conv) | 100% | 100% | 100% | 100% |
| Transformer (softmax + QK + RoPE) | ~100% | degrades | degrades further | chance |

- **GDN**: perfect recall at ALL distances. Delta state preserves associations indefinitely. PAD tokens write to one slot via error correction convergence — no interference with KV associations.
- **Transformer**: degrades with distance. Softmax must distribute weight across all positions including PADs. More PADs → more dilution → lower weight on KV positions.

### Key Findings from the Benchmark

1. **GDN without conv (QK only, no short conv on K) fails** — stuck at ~30% even at distance 0. The QK projections at the value position address by the value token's identity, not the key token's. The conv on K mixes adjacent positions, giving the value position access to the key's identity.

2. **Token shift solves it trivially** — structural alignment between write key (previous position = key token) and read key (current position = query token). No learning required for the association mechanism.

3. **The conv on GDN's K path serves the same role as token shift** — it provides cross-position information for key construction. Without either conv or token shift, the delta rule can't form cross-position associations through QK alone.

4. **Delta state is distance-invariant**: no softmax, no normalization. PAD writes converge (same token → same key → correction makes subsequent writes near-zero). KV associations occupy separate slots and persist indefinitely.

5. **Transformer distance degradation is structural**: softmax weight on target ≈ e^s / (e^s + N_pad). As N_pad grows, the model needs exponentially larger scores to maintain peaked attention. RoPE doesn't generalize to unseen distances.

## Open Questions

- Does the token shift advantage hold at billion-param scale?
- At what scale (if ever) do QK projections overtake token shift for delta rule?
- Can the three non-QK selectivity mechanisms be further improved to widen the gap?
- Can the transformer's distance degradation be mitigated with attention sinks or other tricks?
