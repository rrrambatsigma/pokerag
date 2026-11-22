"""QLoRA fine-tuning for the B5 adapter.

One seed, one adapter, one output directory. The training set is the chat-format
``train.jsonl`` from :mod:`pokerag.sft.dataset`, so the prompt seen during
training is byte-identical to the prompt used by B4 at inference.

Loss is computed on the assistant turn only: the system and user turns are
masked with ``-100``. Training on the prompt would teach the model to generate
evidence, which is the opposite of grounding.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

from pokerag.core.hashing import sha256_file
from pokerag.core.jsonl import read_jsonl, write_json

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def mask_labels(prefix_ids: Sequence[int], full_ids: Sequence[int]) -> list[int]:
    """``-100`` for the prompt, the token ids for the assistant completion."""
    if list(full_ids[: len(prefix_ids)]) != list(prefix_ids):
        raise ValueError("chat template prefix is not a prefix of the full ids")
    return [-100] * len(prefix_ids) + list(full_ids[len(prefix_ids) :])


def encode_example(tokenizer, messages: Sequence[dict], max_seq: int) -> tuple[list[int], list[int]]:
    """Tokenise one chat example and mask everything before the assistant turn."""
    prompt = tokenizer.apply_chat_template(
        list(messages[:-1]), tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=False
    )
    prefix_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_seq:
        overflow = len(full_ids) - max_seq
        if overflow >= len(prefix_ids):
            raise ValueError("truncation removed the prompt; the assistant turn cannot be located")
        full_ids = full_ids[overflow:]
        prefix_ids = prefix_ids[overflow:]

    labels = mask_labels(prefix_ids, full_ids)
    if all(label == -100 for label in labels):
        raise ValueError("truncation removed the whole assistant turn")
    return full_ids, labels


def collate(batch: Sequence[tuple[list[int], list[int]]], pad_id: int) -> dict:
    import torch

    width = max(len(ids) for ids, _ in batch)
    input_ids, attention, labels = [], [], []
    for ids, targets in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        labels.append(targets + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _batch_loss(model, batch, device) -> tuple[float, int]:
    import torch

    moved = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        out = model(**moved)
    tokens = int((moved["labels"] != -100).sum())
    return float(out.loss) * tokens, tokens


def run(args) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = time.time()
    torch.manual_seed(args.seed)
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            revision=args.base_revision,
            quantization_config=quant,
            device_map={"": device},
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            revision=args.base_revision,
            torch_dtype=torch.float16,
        ).to(device)

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=list(TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    train_rows = list(read_jsonl(Path(args.train)))
    val_rows = list(read_jsonl(Path(args.val)))
    train_data = [
        encode_example(tokenizer, row["messages"], args.max_seq) for row in train_rows
    ]
    val_data = [encode_example(tokenizer, row["messages"], args.max_seq) for row in val_rows]

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in trainable)

    order = list(range(len(train_data)))
    optimizer_steps = 0
    history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_tokens = 0.0, 0
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        for position, index in enumerate(order):
            ids, labels = train_data[index]
            batch = collate([(ids, labels)], tokenizer.pad_token_id)
            moved = {key: value.to(device) for key, value in batch.items()}
            out = model(**moved)
            tokens = int((moved["labels"] != -100).sum())
            loss = out.loss
            (loss / args.grad_accum).backward()
            epoch_loss += float(loss) * tokens
            epoch_tokens += tokens

            if (position + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_loss, val_tokens = 0.0, 0
        for ids, labels in val_data:
            loss_sum, tokens = _batch_loss(
                model, collate([(ids, labels)], tokenizer.pad_token_id), device
            )
            val_loss += loss_sum
            val_tokens += tokens
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(epoch_loss / max(epoch_tokens, 1), 5),
                "val_loss": round(val_loss / max(val_tokens, 1), 5),
                "optimizer_steps": optimizer_steps,
                "elapsed_s": round(time.time() - started, 1),
            }
        )
        print(json.dumps(history[-1]), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    summary = {
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "seed": args.seed,
        "epochs": args.epochs,
        "load_in_4bit": args.load_in_4bit,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": list(TARGET_MODULES),
        },
        "optimizer": {"name": "adamw", "lr": args.lr, "grad_accum": args.grad_accum},
        "max_seq": args.max_seq,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": round(trainable_params / total_params, 6),
        "history": history,
        "peak_vram_gb": round(peak, 2),
        "elapsed_s": round(time.time() - started, 1),
        "dataset": {
            "train": str(args.train),
            "train_sha256": sha256_file(Path(args.train)),
            "val": str(args.val),
            "val_sha256": sha256_file(Path(args.val)),
        },
    }
    write_json(out_dir / "training.json", summary)
    return summary


def build_parser(parser) -> None:
    parser.add_argument("--train", default="data/sft/train.jsonl")
    parser.add_argument("--val", default="data/sft/val.jsonl")
    parser.add_argument("--out", default="artifacts/adapters/B5-r{seed}")
    parser.add_argument(
        "--base-model", default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--base-revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="QLoRA NF4 (default) atau fp16",
    )
