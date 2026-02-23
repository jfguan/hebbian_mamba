"""Train a ~100M param HebbianMambaLoop on The Stack.

Architecture: d_model=1024, n_layers=11 (middle 4 looped) ~97M unique params.
Matches stack100M_memory param count with 4 looped layers for extra SSM depth.

Usage:
    uv run experiments/train_loop_stack100M.py
    uv run experiments/train_loop_stack100M.py --resume experiments/checkpoints/ckpt_stack100M_loop_step2000.pt
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import Config
from experiments.model_loop import HebbianMambaLoop
from train import cosine_lr, configure_optimizers, evaluate, sample, plot_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--total-steps", type=int, default=64_000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--memory-alpha", type=float, default=0.03)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--ckpt-interval", type=int, default=2000)
    p.add_argument("--tag", type=str, default="stack100M_loop")
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
        d_model=1024,
        n_layers=11,
        d_state=16,
        use_memory=True,
        memory_alpha=args.memory_alpha,
    )
    model = HebbianMambaLoop(cfg).to(device)
    n_params = sum(p.numel() for p in set(model.parameters()))

    n_loop = sum(1 for l in model.layers if l.__class__.__name__ == "LoopedMambaLayer")
    print(f"{n_params/1e6:.1f}M params | {n_loop}/{cfg.n_layers} looped layers | {device}")

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
    print(f"Full schedule: {args.total_steps * tokens_per_step / 1e6:.0f}M tokens total")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"history_{args.tag}.jsonl")
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
                ckpt_path = os.path.join(out_dir, f"ckpt_{args.tag}_step{step}.pt")
                torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "config": cfg, "step": step, "model_class": "HebbianMambaLoop"}, ckpt_path)
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

    sample_prompt = 'def fizzbuzz(n):\n    """Print 1 to n; Fizz for multiples of 3, Buzz for 5, FizzBuzz for both."""\n'
    print(f"Sample:\n{sample(raw_model, ds['encode'], ds['decode'], device, prompt=sample_prompt, n=300)}")

    final_path = os.path.join(out_dir, f"model_{args.tag}.pt")
    torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg, "step": step + 1, "model_class": "HebbianMambaLoop"}, final_path)
    history = [json.loads(line) for line in open(log_path)]
    plot_losses(history, args.tag)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
