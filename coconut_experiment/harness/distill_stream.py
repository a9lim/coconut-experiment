"""Phase-1 distillation for the continuous-carrier (stream) arm.

Distills a frozen reference CoT model (e.g. `runs/gate-cot`) into a
`ContinuousStreamModel` by **professor forcing**: the reference's per-position
hidden states are fed at the carried positions and the student is trained to
reproduce, at every position, the reference's next-token distribution (KL) and
hidden state (MSE). Because the inputs are the *reference's* states, the whole
sequence trains in one parallel forward — no multi-segment loop — so this runs
comfortably on MPS (unlike the latent curriculum).

Two properties fall out by construction:
  * the readout `argmax(LM_head(h_t))` reproduces the reference CoT → faithful;
  * dense per-position targets (vs. the latent arm's sparse end-of-sequence CE)
    directly attack the placeholder collapse the latent scout hit.

    python -m coconut_experiment.harness.distill_stream --data data/gate --ref runs/gate-cot \
        --out runs/gate-stream --epochs 3

Smoke (MPS/CPU): --limit 256 --max-steps 20 --eval-n 64

After training it runs a **self-fed** clean-accuracy probe (the real deployment
path) and prints it next to the **professor-forced** token-agreement. The gap
between them is the exposure-bias signal: small ⇒ the pure carrier rolls out on
its own; large ⇒ scheduled sampling is the next step.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from .eval_cot import parse_answer
from .serialize import CoTCollator, CoTDataset, load_records, setup_tokenizer
from .train_cot import pick_device


def _prompt_len(labels: torch.Tensor) -> torch.Tensor:
    """First non-(-100) position per row = number of masked prompt tokens."""
    return (labels != -100).float().argmax(dim=1)


def train(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    init_path = args.init or args.ref
    print(f"device: {device}  ref: {args.ref}  init: {init_path}  proj: {args.proj}")

    tokenizer = setup_tokenizer(args.base_model)

    # Frozen reference (teacher) and trainable student.
    reference: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(args.ref)
    reference.to(device)  # type: ignore[arg-type]  # torch stub overload gap
    reference.eval()
    reference.requires_grad_(False)

    from .stream_model import ContinuousStreamModel

    student = ContinuousStreamModel(
        tokenizer, base_model=args.base_model, init_path=init_path, proj=args.proj
    )
    student.to(device)

    records = load_records(Path(args.data) / "train.json")
    if args.limit:
        records = records[: args.limit]
    print(f"train records: {len(records)}")

    loader = DataLoader(
        CoTDataset(records, tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CoTCollator(tokenizer),
        drop_last=True,
    )

    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = min(len(loader) * args.epochs, args.max_steps or 10**9)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )

    student.train()
    step, done = 0, False
    recent_kl: List[float] = []
    recent_mse: List[float] = []
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            plen = _prompt_len(labels)

            with torch.no_grad():
                ref_out = reference(input_ids, attention_mask=attn, output_hidden_states=True)
                ref_h = ref_out.hidden_states[-1]
                ref_logits = ref_out.logits

            sup = attn.bool()                                       # non-pad positions

            # Scheduled sampling: ramp 0 -> ss_prob over ss_ramp_frac of training.
            ss_eff = args.ss_prob * min(1.0, step / max(1, int(args.ss_ramp_frac * total_steps)))
            self_hidden = ss_mask = None
            if ss_eff > 0:
                with torch.no_grad(), amp_ctx:
                    _, stu_h_pass1 = student.forward_professor(input_ids, attn, plen, ref_h)
                self_hidden = stu_h_pass1.detach()
                ss_mask = (torch.rand_like(attn, dtype=torch.float) < ss_eff) & sup

            with amp_ctx:
                stu_logits, stu_h = student.forward_professor(
                    input_ids, attn, plen, ref_h, self_hidden=self_hidden, ss_mask=ss_mask
                )

                t = args.temperature
                kl = F.kl_div(
                    F.log_softmax(stu_logits[sup] / t, dim=-1),
                    F.softmax(ref_logits[sup] / t, dim=-1),
                    reduction="batchmean",
                ) * (t * t)

                pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                carried = (pos >= plen.unsqueeze(1)) & sup
                mse = F.mse_loss(stu_h[carried], ref_h[carried])

                loss = args.kl_weight * kl + args.mse_weight * mse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            recent_kl = (recent_kl + [kl.item()])[-100:]
            recent_mse = (recent_mse + [mse.item()])[-100:]
            pbar.set_postfix(
                kl=f"{sum(recent_kl) / len(recent_kl):.4f}",
                mse=f"{sum(recent_mse) / len(recent_mse):.4f}",
                ss=f"{ss_eff:.2f}",
            )
            step += 1
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"saved -> {out}  (steps={step})")

    if args.eval_n:
        evaluate(student, reference, tokenizer, device, args)


@torch.no_grad()
def evaluate(student, reference, tokenizer, device, args) -> None:
    """Headline scout numbers: self-fed clean accuracy vs professor-forced agreement."""
    eval_path = Path(args.data) / "eval_clean.json"
    if not eval_path.exists():
        print(f"(no {eval_path}; skipping eval)")
        return
    records = load_records(eval_path)[: args.eval_n]
    p = records[0]["p"]
    student.eval()

    # (a) Self-fed rollout — the real deployment path. bs=1, sequential.
    correct, parsed = 0, 0
    for r in tqdm(records, desc="self-fed eval"):
        prompt = tokenizer.encode(r["question"] + "\n")
        ii = torch.tensor([prompt], device=device)
        readouts, _ = student.generate_stream(ii, max_new_tokens=args.max_new_tokens)
        text = tokenizer.decode(readouts, skip_special_tokens=True)
        pred = parse_answer(text, p)
        if pred is not None:
            parsed += 1
            if pred == r["true_answer"]:
                correct += 1
    n = len(records)
    print(f"\nself-fed clean accuracy : {correct / n:.3f}  (parsed {parsed}/{n})")

    # (b) Professor-forced token agreement — did distillation fit the targets?
    ds = CoTDataset(records, tokenizer)
    coll = CoTCollator(tokenizer)
    agree_num, agree_den = 0, 0
    for i in range(0, n, args.batch_size):
        batch = coll([ds[j] for j in range(i, min(i + args.batch_size, n))])
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        plen = _prompt_len(batch["labels"].to(device))
        ref_out = reference(input_ids, attention_mask=attn, output_hidden_states=True)
        stu_logits, _ = student.forward_professor(input_ids, attn, plen, ref_out.hidden_states[-1])
        pos = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        carried = (pos >= plen.unsqueeze(1)) & attn.bool()
        agree = (stu_logits.argmax(-1) == ref_out.logits.argmax(-1)) & carried
        agree_num += int(agree.sum())
        agree_den += int(carried.sum())
    print(f"professor-forced agree  : {agree_num / max(agree_den, 1):.3f}  "
          f"(student readout vs reference, carried positions)")
    print("gap (agree - self-fed) is the exposure-bias signal → scheduled sampling if large.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/gate")
    ap.add_argument("--ref", type=str, default="runs/gate-cot", help="frozen teacher (CoT model)")
    ap.add_argument("--init", type=str, default="", help="student init (default: --ref)")
    ap.add_argument("--out", type=str, default="runs/gate-stream")
    ap.add_argument("--base-model", type=str, default="gpt2")
    ap.add_argument("--proj", type=str, default="none", choices=["none", "linear"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0, help="distillation softmax temp")
    ap.add_argument("--ss-prob", type=float, default=0.0,
                    help="max scheduled-sampling prob (0 = pure professor forcing)")
    ap.add_argument("--ss-ramp-frac", type=float, default=0.5,
                    help="fraction of training over which ss ramps 0 -> ss_prob")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=256, help="self-fed eval examples (0 = skip)")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
