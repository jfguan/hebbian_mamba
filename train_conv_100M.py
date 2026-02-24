"""Train a ~100M param HebbianConv on The Stack.

Architecture: 12 layers, d_model=1024, ~102M params.
Conv(4) + Hebbian associative memory.

Usage:
    uv run train_conv_100M.py
    uv run train_conv_100M.py --resume checkpoints/ckpt_stack100M_conv_step2000.pt
"""

import argparse
import json
import math
import os
import time

import torch

from model_conv import Config, HebbianConv
from train import cosine_lr, configure_optimizers, evaluate, plot_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--total-steps", type=int, default=64_000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--memory-alpha", type=float, default=0.03)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--ckpt-interval", type=int, default=2000)
    p.add_argument("--tag", type=str, default="stack100M_conv")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    from data_stack import load_dataset, DataLoader

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

    ds = load_dataset()
    train_loader = DataLoader(ds["train"], args.batch_size, args.seq_len)
    val_loader = DataLoader(ds["val"], args.batch_size, args.seq_len)

    cfg = Config(
        vocab_size=ds["vocab_size"],
        d_model=args.d_model,
        n_layers=args.n_layers,
        memory_alpha=args.memory_alpha,
    )
    model = HebbianConv(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params/1e6:.1f}M params | {args.n_layers}L d={args.d_model} | {device}")

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
        if not missing and not unexpected and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    end_step = min(start_step + args.steps, args.total_steps)
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    min_lr = args.lr * 0.1

    print(f"Steps {start_step} -> {end_step} / {args.total_steps} | B={args.batch_size}x{args.grad_accum} T={args.seq_len} lr={args.lr}")
    print(f"Tokens/step: {tokens_per_step:,} | ~{tokens_per_step * (end_step - start_step) / 1e6:.1f}M tokens this run")

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"checkpoints/history_{args.tag}.jsonl"
    log_file = open(log_path, "a" if args.resume else "w")

    step = start_step
    entry = {}
    tokens_seen = start_step * tokens_per_step
    try:
        for step in range(start_step, end_step):
            t0 = time.time()
            lr = cosine_lr(step, args.warmup, args.total_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(args.grad_accum):
                x, y = train_loader.batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / args.grad_accum
                loss.backward()
                loss_accum += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tokens_seen = (step + 1) * tokens_per_step
            dt = time.time() - t0
            entry = {"step": step, "train_loss": loss_accum, "tokens": tokens_seen}
            print(
                f"step {step:5d} | loss {loss_accum:.4f} | tokens {tokens_seen/1e6:.1f}M"
                f" | lr {lr:.2e} | {dt*1000:.0f}ms",
                flush=True,
            )

            if step > 0 and step % args.eval_interval == 0:
                vl = evaluate(model, val_loader, device)
                entry["val_loss"] = vl
                print(f"  val loss {vl:.4f} | val ppl {math.exp(vl):.2f}", flush=True)

            if step > 0 and step % args.ckpt_interval == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                ckpt_path = f"checkpoints/ckpt_{args.tag}_step{step}.pt"
                torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "config": cfg, "step": step, "model_class": "HebbianConv"}, ckpt_path)
                print(f"  -> {ckpt_path}", flush=True)

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

    except KeyboardInterrupt:
        print(f"\nStopped at step {step}.")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    vl = evaluate(model, val_loader, device)
    log_file.write(json.dumps({"step": step, "train_loss": entry.get("train_loss", 0),
                               "val_loss": vl, "tokens": tokens_seen}) + "\n")
    log_file.close()
    print(f"\nFinal val loss: {vl:.4f} | ppl {math.exp(vl):.2f}")

    final_path = f"checkpoints/model_{args.tag}.pt"
    torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg, "step": step + 1, "model_class": "HebbianConv"}, final_path)
    history = [json.loads(line) for line in open(log_path)]
    plot_losses(history, args.tag)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
