"""Minimal GPT-2 SFT for the CoT arm.

Primary purpose right now: settle the **validity gate** — can a vanilla GPT-2
learn the clean CheatChain task (mod-p variable chains) to >90% exact-match?
If not, "used the cue" downstream is a capability cop-out, not a hack.

    python -m coconut_experiment.harness.train_cot --data data/gate --out runs/gate-cot --epochs 3

Smoke test on the Mac (no GPU): add `--limit 2000 --max-steps 200`.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from .serialize import CoTCollator, CoTDataset, load_records, setup_tokenizer


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    tokenizer = setup_tokenizer(args.base_model)
    records = load_records(Path(args.data) / "train.json")
    if args.limit:
        records = records[: args.limit]
    print(f"train records: {len(records)}")

    dataset = CoTDataset(records, tokenizer)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=CoTCollator(tokenizer), drop_last=True,
    )

    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(args.base_model)
    model.to(device)  # type: ignore[arg-type]  # torch stub overload gap
    pad_id = tokenizer.pad_token_id
    assert isinstance(pad_id, int)
    model.config.pad_token_id = pad_id

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = min(len(loader) * args.epochs, args.max_steps or 10**9)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps,
    )

    # bf16 autocast on CUDA (Ada/4090) needs no GradScaler; fp32 elsewhere.
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )

    model.train()
    step, recent = 0, []
    done = False
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp_ctx:
                loss = model(**batch).loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            recent = (recent + [loss.item()])[-100:]
            pbar.set_postfix(loss=f"{sum(recent) / len(recent):.4f}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved -> {out}  (steps={step}, final loss≈{sum(recent) / len(recent):.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/gate")
    ap.add_argument("--out", type=str, default="runs/gate-cot")
    ap.add_argument("--base-model", type=str, default="gpt2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="subset train records (0 = all)")
    ap.add_argument("--max-steps", type=int, default=0, help="cap total steps (0 = no cap)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
