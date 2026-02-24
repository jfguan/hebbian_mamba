"""Train HebbianMambaLoopSSD on codeparrot.

Usage:
    uv run experiments/train_ssd.py
    uv run experiments/train_ssd.py --tag ssd_test --steps 500
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import Config
from experiments.model_ssd import HebbianMambaLoopSSD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import cosine_lr, configure_optimizers, evaluate


def _plot_losses(history, tag, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    use_tokens = "tokens" in history[0]
    xs = [h["tokens"] / 1e6 for h in history] if use_tokens else [h["step"] for h in history]
    xlabel = "Training tokens (M)" if use_tokens else "Step"
    ax.plot(xs, [h["train_loss"] for h in history], label="train", alpha=0.7)
    val = [(h["tokens"] / 1e6 if use_tokens else h["step"], h["val_loss"])
           for h in history if "val_loss" in h]
    if val:
        ax.plot(*zip(*val), "o-", label="val", markersize=4)
    ax.set(xlabel=xlabel, ylabel="loss (nats)", title=f"Training — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"loss_{tag}.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1465)
    p.add_argument("--schedule-steps", type=int, default=1465)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--ckpt-interval", type=int, default=500)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=None,
                   help="Number of SSD heads (default: d_inner // 128)")
    p.add_argument("--stack-loops", type=int, default=1,
                   help="Number of times to run the looped layers (default: 1)")
    p.add_argument("--loop-start", type=int, default=None,
                   help="First looped layer index (default: 0)")
    p.add_argument("--loop-end", type=int, default=None,
                   help="One past last looped layer (default: n_layers)")
    p.add_argument("--gate-init", type=float, default=-5.0,
                   help="Initial value for loop gate (sigmoid applied)")
    p.add_argument("--memory-alpha", type=float, default=0.03)
    p.add_argument("--loop-alpha", type=float, default=None,
                   help="Memory alpha for loop passes > 0 (default: same as memory-alpha)")
    p.add_argument("--dataset", type=str, default="code", choices=["code", "stack"])
    p.add_argument("--backbone", type=str, default="ssd", choices=["ssd", "linear"])
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--tag", type=str, default="code_ssd")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    device_type = device.split(":")[0]
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.dataset == "stack":
        from data_stack import load_dataset, DataLoader
    else:
        from data_code import load_dataset, DataLoader
    ds = load_dataset()
    train_loader = DataLoader(ds["train"], args.batch_size, args.seq_len)
    val_loader = DataLoader(ds["val"], args.batch_size, args.seq_len)

    cfg = Config(
        vocab_size=ds["vocab_size"],
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        memory_alpha=args.memory_alpha,
        use_memory=True,
    )
    # Attach SSD-specific fields to config (read via getattr in model)
    if args.n_heads is not None:
        cfg.n_heads = args.n_heads
    cfg.stack_loops = args.stack_loops
    cfg.gate_init = args.gate_init
    cfg.loop_alpha = args.loop_alpha if args.loop_alpha is not None else args.memory_alpha
    cfg.backbone = args.backbone
    cfg.loop_start = args.loop_start if args.loop_start is not None else 0
    cfg.loop_end = args.loop_end if args.loop_end is not None else args.n_layers

    model = HebbianMambaLoopSSD(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    n_heads_actual = getattr(model.layers[0].ssd, "n_heads", 0)
    if args.stack_loops > 1:
        loop_desc = f"loop layers {cfg.loop_start}-{cfg.loop_end-1} x{args.stack_loops}"
    else:
        loop_desc = "no loops"
    print(f"{n_params/1e6:.2f}M params | {loop_desc} | {n_heads_actual} heads | {device}")

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = configure_optimizers(model, args.lr, weight_decay=0.1)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  New params: {missing}")
        if unexpected:
            print(f"  Dropped: {unexpected}")
        if not missing and not unexpected and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    total_steps = start_step + args.steps
    min_lr = args.lr * 0.1
    grad_accum = args.grad_accum
    print(f"Steps {start_step} -> {total_steps} | B={args.batch_size}x{grad_accum} T={args.seq_len} lr={args.lr}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"history_{args.tag}.jsonl")
    log_file = open(log_path, "a" if args.resume else "w")

    import time
    entry = {}
    try:
        for step in range(start_step, total_steps):
            t0 = time.time()
            lr = cosine_lr(step, args.warmup, args.schedule_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(grad_accum):
                x, y = train_loader.batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / grad_accum
                loss.backward()
                loss_accum += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dt = time.time() - t0
            tokens_seen = (step + 1) * args.batch_size * args.seq_len * grad_accum
            entry = {"step": step, "train_loss": loss_accum, "tokens": tokens_seen}
            print(
                f"step {step:5d} | loss {loss_accum:.4f} | ppl {math.exp(loss_accum):8.2f}"
                f" | lr {lr:.2e} | {dt * 1000:.0f}ms",
                flush=True,
            )

            if step > 0 and step % args.eval_interval == 0:
                vl = evaluate(model, val_loader, device)
                entry["val_loss"] = vl
                print(f"  val loss {vl:.4f} | val ppl {math.exp(vl):.2f}", flush=True)

            if step > 0 and step % args.ckpt_interval == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                ckpt_path = os.path.join(out_dir, f"ckpt_{args.tag}_step{step}.pt")
                torch.save(
                    {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": cfg, "step": step, "model_class": "HebbianMambaLoopSSD"},
                    ckpt_path,
                )
                print(f"  -> {ckpt_path}", flush=True)

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

    except KeyboardInterrupt:
        print(f"\nStopped at step {step}.")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    vl = evaluate(model, val_loader, device)
    log_file.write(json.dumps({"step": step, "train_loss": entry["train_loss"], "val_loss": vl, "tokens": entry.get("tokens", 0)}) + "\n")
    log_file.close()
    print(f"\nFinal val loss: {vl:.4f} | ppl {math.exp(vl):.2f}")

    final_path = os.path.join(out_dir, f"model_{args.tag}.pt")
    torch.save(
        {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
         "config": cfg, "step": step + 1, "model_class": "HebbianMambaLoopSSD"},
        final_path,
    )
    history = [json.loads(line) for line in open(log_path)]
    _plot_losses(history, args.tag, out_dir)
    print(f"Saved {final_path} + {log_path}")


if __name__ == "__main__":
    main()
