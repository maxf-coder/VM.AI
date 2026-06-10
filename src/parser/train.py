"""
VM-AI - Parser Training Script
Trains the parser model with EXP/PRD tag format.
Run: python src/parser/train.py

Written by: Vanea
"""

import argparse
import os
import time

import numpy as np
import torch
from cfg import Config
from data_generator import DataGenerator
from datasets import Dataset
from huggingface_hub import snapshot_download
from schemas import parse_pipe_simple
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)
from vars import TRACKED_FIELDS
from yaml_parser import VMAI_RealDataParser, VMAI_YamlParser

if os.name != "nt":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _parse_pipe_with_tags(text: str) -> dict:
    """Parse pipe-format with EXP/PRD tags into simple dict (ignores tags for metrics)."""
    result = {}
    for part in text.split("|"):
        part = part.strip()
        if "=" not in part:
            continue
        k, _, rest = part.partition("=")
        k = k.strip().lower()

        if "[" in rest and rest.endswith("]"):
            v = rest[:-1].split("[", 1)[0].strip()
        else:
            v = rest.strip()

        if v.lower() in ("null", ""):
            v = None
        result[k] = v
    return result


def compute_metrics(eval_preds: EvalPrediction, tokenizer):
    predictions, label_ids = eval_preds

    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
    label_ids = np.where(label_ids < 0, tokenizer.pad_token_id, label_ids)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    correct = {f: 0 for f in TRACKED_FIELDS}
    present = {f: 0 for f in TRACKED_FIELDS}

    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        pred_dict = _parse_pipe_with_tags(pred_str)
        label_dict = _parse_pipe_with_tags(label_str)
        for field in TRACKED_FIELDS:
            if field not in label_dict:
                continue
            present[field] += 1
            if pred_dict.get(field) == label_dict[field]:
                correct[field] += 1

    metrics = {}
    total_correct = 0
    total_present = 0
    for field in TRACKED_FIELDS:
        n = present[field]
        c = correct[field]
        acc = round(c / n, 4) if n > 0 else 0.0
        metrics[f"acc_{field}"] = acc
        total_correct += c
        total_present += n

    metrics["acc_overall"] = (
        round(total_correct / total_present, 4) if total_present > 0 else 0.0
    )
    return metrics


def download_base_model(cfg):
    os.makedirs(cfg.model_cache, exist_ok=True)
    if not os.listdir(cfg.model_cache):
        print("Downloading t5-base...")
        snapshot_download(repo_id="google-t5/t5-base", local_dir=cfg.model_cache)
    else:
        print(f"Base model found at {cfg.model_cache}")


def load_model(cfg, device):
    if os.path.exists(cfg.output_dir) and os.listdir(cfg.output_dir):
        print("Resuming from checkpoint...")
        return T5ForConditionalGeneration.from_pretrained(cfg.output_dir).to(device)
    print("Starting from base model...")
    return T5ForConditionalGeneration.from_pretrained(cfg.model_cache).to(device)


def add_special_tokens(tokenizer, model):
    """Add [EXP] and [PRD] as special tokens to prevent subword fragmentation."""
    special_tokens = ["[EXP]", "[PRD]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added special tokens: {special_tokens} (vocab size: {len(tokenizer)})")
    return tokenizer, model


def save_model(model, tokenizer, cfg):
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Model saved to {cfg.output_dir}")


def build_dataset(cfg, mode):
    yp = VMAI_YamlParser(cfg.data_path)
    yp.load_yaml()
    training_data = yp.parse()

    real_examples = []
    if mode != "synthetic":
        if os.path.exists(cfg.real_data_path):
            try:
                rp = VMAI_RealDataParser(cfg.real_data_path)
                rp.load_yaml()
                real_examples = rp.parse()
                print(f"Real examples loaded: {len(real_examples)}")
            except Exception as e:
                print(f"Real data skipped — failed to load: {e}")
        else:
            print("No real data file found — skipping")

    specific_examples = []
    if os.path.exists(cfg.specific_data_path):
        try:
            sp = VMAI_RealDataParser(cfg.specific_data_path)
            sp.load_yaml()
            specific_examples = sp.parse()
            print(f"Specific examples loaded: {len(specific_examples)}")
        except Exception as e:
            print(f"Specific data skipped — failed to load: {e}")
    else:
        print("No specific data file found — skipping")

    gen = DataGenerator(training_data, real_examples, specific_examples)

    if mode == "modify_only":
        data = {"input_text": [], "target_text": []}
        for _ in range(cfg.max_limit):
            inp, tgt = gen._generate_modify()
            data["input_text"].append(inp)
            data["target_text"].append(tgt)
        for example in real_examples + specific_examples:
            if not isinstance(example.get("input"), str):
                continue
            if example["input"].startswith("modify:"):
                inp, tgt = gen._convert_real_modify(example)
                for _ in range(3):
                    data["input_text"].append(inp)
                    data["target_text"].append(tgt)
        dataset = Dataset.from_dict(data)
    elif mode == "specific":
        data = {"input_text": [], "target_text": []}
        for example in specific_examples:
            inp, tgt = gen._convert_real(example)
            data["input_text"].append(inp)
            data["target_text"].append(tgt)
        import random

        random.seed(42)
        for _ in range(min(500, cfg.max_limit)):
            inp, tgt = gen._generate_modify()
            data["input_text"].append(inp)
            data["target_text"].append(tgt)
        dataset = Dataset.from_dict(data)
    elif mode == "real":
        data = {"input_text": [], "target_text": []}
        for example in real_examples:
            inp, tgt = gen._convert_real(example)
            data["input_text"].append(inp)
            data["target_text"].append(tgt)
        dataset = Dataset.from_dict(data)
    else:
        dataset = gen.generate(cfg.max_limit)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])}  |  Test: {len(split['test'])}")
    return split["train"], split["test"]


def tokenize(train_ds, test_ds, tokenizer):
    def tokenize_fn(examples):
        inputs = tokenizer(
            examples["input_text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        targets = tokenizer(
            examples["target_text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        labels = np.array(
            [
                [(t if t != tokenizer.pad_token_id else -100) for t in label]
                for label in targets["input_ids"]
            ],
            dtype=np.int64,
        )
        inputs["labels"] = labels
        return inputs

    cols = ["input_ids", "attention_mask", "labels"]
    tok_train = train_ds.map(tokenize_fn, batched=True).with_format(
        "torch", columns=cols
    )
    tok_test = test_ds.map(tokenize_fn, batched=True).with_format("torch", columns=cols)
    return tok_train, tok_test


def find_latest_checkpoint(output_dir):
    import re

    if not os.path.exists(output_dir):
        return None

    checkpoints = []
    for d in os.listdir(output_dir):
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d)):
            match = re.match(r"checkpoint-(\d+)", d)
            if match:
                checkpoints.append((int(match.group(1)), os.path.join(output_dir, d)))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def train(model, tokenizer, cfg, tok_train, tok_test, lr, resume_from=None):
    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=128,
        push_to_hub=False,
        remove_unused_columns=False,
        optim="adafactor",
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=cfg.fp16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        logging_steps=cfg.logging_steps,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_test,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    print("Starting training...")
    if resume_from:
        print(f"  -> Resuming from: {resume_from}")
    else:
        print("  -> Starting fresh (no checkpoints found)")

    trainer.train(resume_from_checkpoint=resume_from)


def parse_args():
    parser = argparse.ArgumentParser(description="VM.AI Parser Trainer")
    parser.add_argument(
        "--mode",
        choices=["both", "synthetic", "real", "specific", "modify_only"],
        default="both",
        help="Training mode selection",
    )
    return parser.parse_args()


def main():
    start = time.time()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(args.mode)

    if args.mode == "modify_only":
        if not os.path.exists(cfg.output_dir) or not os.listdir(cfg.output_dir):
            print("ERROR: modify_only requires an existing trained checkpoint.")
            print(f"       Nothing found at: {cfg.output_dir}")
            return

    print(f"Device : {device}")
    print(f"Mode   : {args.mode}")
    print(f"Epochs : {cfg.num_train_epochs}")

    is_resume = os.path.exists(cfg.output_dir) and os.listdir(cfg.output_dir)
    lr = cfg.learning_rate_resume if is_resume else cfg.learning_rate_fresh
    print(f"Resume : {is_resume}  LR: {lr}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    download_base_model(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_cache)
    model = load_model(cfg, device)
    tokenizer, model = add_special_tokens(tokenizer, model)

    train_ds, test_ds = build_dataset(cfg, args.mode)
    tok_train, tok_test = tokenize(train_ds, test_ds, tokenizer)

    resume_path = None

    if args.mode == "specific":
        resume_path = None

    train(model, tokenizer, cfg, tok_train, tok_test, lr, resume_from=resume_path)
    save_model(model, tokenizer, cfg)

    elapsed = int(time.time() - start)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"\nDone in {h:02d}h {m:02d}m {s:02d}s")


if __name__ == "__main__":
    main()
