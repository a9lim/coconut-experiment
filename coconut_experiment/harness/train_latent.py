"""Curriculum SFT for the latent (Coconut) arm.

Mirrors `train_cot.py`, but trains the forked `ContinuousThoughtModel` through
the multi-stage curriculum: at stage k the first k reasoning steps are replaced
by k*c latent tokens, with optimizer reset and stage-mixing between stages
(Coconut's anti-forgetting recipe). Same CheatChain data as the CoT arm.

    python -m coconut_experiment.harness.train_latent --data data/pcue100 --out runs/pcue100-latent

Smoke (CPU/MPS): `--limit 512 --epochs-per-stage 1 --max-steps-per-stage 20`.

NOTE: the multi-segment forward is several× a normal forward; the full curriculum
(6 stages) is the 4090's job. Size epochs/train-n to the device.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .latent_data import CheatChainLatentDataset, setup
from .serialize import load_records
from .train_cot import pick_device

_TENSOR_KEYS = ("input_ids", "attention_mask", "labels", "position_ids")


def train(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    tokenizer, model, collator = setup(args.base_model)
    model.to(device)

    records = load_records(Path(args.data) / "train.json")
    if args.limit:
        records = records[: args.limit]
    print(f"train records: {len(records)}  | curriculum: {args.num_stages + 1} stages "
          f"x {args.epochs_per_stage} epochs")

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )

    global_step = 0
    for stage in range(args.num_stages + 1):
        dataset = CheatChainLatentDataset(
            records, tokenizer, scheduled_stage=stage, c_thought=args.c_thought,
            max_latent_stage=args.num_stages, uniform_prob=args.uniform_prob,
            seed=args.seed + stage,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collator, drop_last=True,
        )
        # Reset optimizer between stages (Coconut).
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total = len(loader) * args.epochs_per_stage
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total * args.warmup_ratio), total)

        model.train()
        recent, stage_step = [], 0
        for epoch in range(args.epochs_per_stage):
            pbar = tqdm(loader, desc=f"stage {stage}/{args.num_stages} ep {epoch + 1}/{args.epochs_per_stage}")
            for batch in pbar:
                tb = {k: batch[k].to(device) for k in _TENSOR_KEYS}
                with amp_ctx:
                    loss = model(**tb).loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                recent = (recent + [loss.item()])[-100:]
                pbar.set_postfix(loss=f"{sum(recent) / len(recent):.4f}")
                global_step += 1
                stage_step += 1
                if args.max_steps_per_stage and stage_step >= args.max_steps_per_stage:
                    break
            if args.max_steps_per_stage and stage_step >= args.max_steps_per_stage:
                break
        print(f"stage {stage} done: loss≈{sum(recent) / len(recent):.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"saved -> {out}  (total steps={global_step})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/pcue100")
    ap.add_argument("--out", type=str, default="runs/pcue100-latent")
    ap.add_argument("--base-model", type=str, default="gpt2")
    ap.add_argument("--num-stages", type=int, default=5, help="K (chain length); stages 0..K")
    ap.add_argument("--epochs-per-stage", type=int, default=2)
    ap.add_argument("--c-thought", type=int, default=2, help="latent tokens per replaced step")
    ap.add_argument("--uniform-prob", type=float, default=0.3, help="stage-mixing prob")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps-per-stage", type=int, default=0, help="cap steps/stage (smoke)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
