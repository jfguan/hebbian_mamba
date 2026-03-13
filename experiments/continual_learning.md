# Hebbian Associative Memory — Theory Notes

## Core Architecture

The dominant mechanism is a **Hebbian outer-product memory** (W), not the sequence model backbone.

Each layer maintains a D x D matrix W that accumulates associations during inference:
```
W_t = γ * W_{t-1} + v_t ⊗ k_t
read_t = W_{t-1} · query_t
```

The backbone (conv, SSM, or even linear projections) does **local feature extraction** only. Hebbian handles all long-range dependencies.

### Evidence

On codeparrot (4L, matched hyperparams):
| Backbone | Params | Val Loss |
|----------|--------|----------|
| Linear (no mixing) + Heb | ~8.7M | 3.18 |
| Conv (4-token window) + Heb | 8.7M | **1.94** |
| SSD (full SSM recurrence) + Heb | 9.2M | 2.01 |

- Conv **beats** SSD with fewer params. SSM recurrence is redundant.
- Linear-only converges at 3.18 — Hebbian alone does significant work.
- Conv closes 80% of the remaining gap with just 4-token local mixing.

## Biological Analogy

| Biology | Model |
|---------|-------|
| Evolution | Architecture design |
| Development / lifetime learning | Gradient training (slow, offline) |
| Cortex (local feature detectors) | Conv / linear projections (learned weights) |
| Hippocampus (associative binding) | W matrix (fast, online, wiped per sequence) |
| Waking | Inference — W accumulates associations |
| Sleep (non-REM) | Consolidation — distill W knowledge into weights |
| Dreaming (REM) | Generalization — distill under corrupted/recombined inputs |

## Dreaming / Sleep Consolidation Theory

### The Problem
During inference, W captures useful associations for the current context. But W is wiped after each sequence — the model never learns from what W discovered. The projection weights (which control *how* to use W) only update during training.

### Sleep = Self-Distillation

1. **Waking (teacher)**: Run model with W active (alpha > 0) on a sequence. W builds up, predictions improve as context grows. Save logits.

2. **Sleep (student)**: Re-run same model on same sequence with W disabled (alpha = 0). Predictions are worse, especially late in context.

3. **Consolidate**: Backprop the KL divergence between teacher and student outputs. This pushes what W was contributing into the projection weights.

```python
# Teacher — W active
with torch.no_grad():
    teacher_logits = model(tokens, alpha=0.03)

# Student — W disabled
student_logits = model(tokens, alpha=0.0)
loss = kl_div(student_logits, teacher_logits)
loss.backward()  # updates projections to internalize W's knowledge
```

Over many cycles: projections learn shortcuts that W used to provide. W is freed for novel associations.

### Dreaming = Generalized Consolidation

Regular sleep replays exact experiences. Dreaming mixes and corrupts them:

```python
# Dream — corrupted replay
dream_tokens = shuffle_chunks(real_tokens)
teacher_logits = model(dream_tokens, alpha=0.03)  # W tries to bind novel combinations
student_logits = model(dream_tokens, alpha=0.0)
loss = kl_div(student_logits, teacher_logits)
```

This forces generalization: W forms associations in contexts never seen during training. Distilling these into weights teaches the model to generalize beyond its training distribution.

**Nightmares** = adversarial/hard examples where W barely copes. These produce the strongest gradients during consolidation.

### Training Schedule

```
for epoch:
    # Waking — normal training
    for batch in data:
        logits, loss = model(batch, alpha=0.03)
        loss.backward()

    # Sleep — consolidation (every N epochs)
    for batch in replay_buffer:
        with torch.no_grad():
            teacher = model(batch, alpha=0.03)
        student = model(batch, alpha=0.0)
        kl_loss = kl_div(student, teacher)
        kl_loss.backward()

    # Dreaming — generalization (every M epochs)
    for batch in replay_buffer:
        dream_batch = corrupt(batch)
        with torch.no_grad():
            teacher = model(dream_batch, alpha=0.03)
        student = model(dream_batch, alpha=0.0)
        kl_loss = kl_div(student, teacher)
        kl_loss.backward()
```

### Predictions

1. After sleep cycles, the model should perform better with alpha=0 (knowledge internalized).
2. After dream cycles, the model should generalize better to OOD sequences.
3. The gap between alpha=0 and alpha>0 should shrink over consolidation cycles for common patterns but remain for novel ones.
4. Alpha could be dynamically adjusted — lower for familiar patterns (already in weights), higher for novel ones (need W).
