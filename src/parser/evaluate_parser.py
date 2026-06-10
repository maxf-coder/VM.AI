"""
VM-AI - Parser Evaluation Script
Evaluates trained model on test data.
Computes per-field F1, precision, recall, accuracy, and regression metrics.

Run: python src/parser/evaluate_parser.py [--mode both] [--seed 42] [--test_size 0.1]

Written by: Vanea
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfg import Config
from schemas import parse_pipe_simple
from train import build_dataset, compute_metrics, tokenize
from vars import TRACKED_FIELDS as _TRACKED_FIELDS

# Exclude deprecated recurrence fields
DEPRECATED_FIELDS = {"recurrent", "recurrence_days"}
TRACKED_FIELDS = [f for f in _TRACKED_FIELDS if f not in DEPRECATED_FIELDS]

CLASSIFICATION_FIELDS = {"category", "location", "deadline"}
BINARY_FIELDS = {"fixed_time", "recurrent"}
REGRESSION_FIELDS = {"difficulty", "importance"}
STRING_FIELDS = {"name", "duration", "fixed_start", "recurrence_days", "start"}


def _try_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _is_bool_str(val):
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "t", "yes", "1"):
        return True
    if s in ("false", "f", "no", "0"):
        return False
    return None


def _gather_field_values(decoded_preds, decoded_labels):
    pred_values = {f: [] for f in TRACKED_FIELDS}
    label_values = {f: [] for f in TRACKED_FIELDS}
    valid_mask = {f: [] for f in TRACKED_FIELDS}

    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        pred_dict = parse_pipe_simple(pred_str)
        label_dict = parse_pipe_simple(label_str)
        for field in TRACKED_FIELDS:
            pv = pred_dict.get(field)
            lv = label_dict.get(field)
            if lv is None or str(lv).lower() == "null":
                valid_mask[field].append(False)
                pred_values[field].append(None)
                label_values[field].append(None)
            else:
                valid_mask[field].append(True)
                if field in REGRESSION_FIELDS:
                    pred_values[field].append(_try_float(pv))
                    label_values[field].append(_try_float(lv))
                elif field in BINARY_FIELDS:
                    pred_values[field].append(_is_bool_str(pv))
                    label_values[field].append(_is_bool_str(lv))
                else:
                    pred_values[field].append(
                        None if pv is None else str(pv).strip().lower()
                    )
                    label_values[field].append(str(lv).strip().lower())

    return pred_values, label_values, valid_mask


def _compute_classification_metrics(y_true, y_pred):
    filtered_true = []
    filtered_pred = []
    for t, p in zip(y_true, y_pred):
        if t is not None and p is not None:
            filtered_true.append(t)
            filtered_pred.append(p)
    if len(filtered_true) == 0:
        return {"support": 0}
    y_true, y_pred = filtered_true, filtered_pred
    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) < 2:
        acc = accuracy_score(y_true, y_pred)
        return {
            "accuracy": round(acc, 4),
            "f1_macro": round(acc, 4),
            "f1_weighted": round(acc, 4),
            "support": len(y_true),
            "n_classes": len(labels),
        }
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "precision_macro": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "support": len(y_true),
        "n_classes": len(labels),
    }


def _compute_binary_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {}
    filtered_true = []
    filtered_pred = []
    for t, p in zip(y_true, y_pred):
        if t is not None and p is not None:
            filtered_true.append(t)
            filtered_pred.append(p)
    if len(filtered_true) == 0:
        return {"support": 0}
    if len(set(filtered_true)) < 2:
        acc = accuracy_score(filtered_true, filtered_pred)
        return {
            "accuracy": round(acc, 4),
            "f1": round(acc, 4),
            "precision": round(acc, 4),
            "recall": round(acc, 4),
            "support": len(filtered_true),
        }
    return {
        "accuracy": round(accuracy_score(filtered_true, filtered_pred), 4),
        "f1": round(f1_score(filtered_true, filtered_pred, zero_division=0), 4),
        "precision": round(
            precision_score(filtered_true, filtered_pred, zero_division=0), 4
        ),
        "recall": round(recall_score(filtered_true, filtered_pred, zero_division=0), 4),
        "support": len(filtered_true),
    }


def _compute_regression_metrics(y_true, y_pred):
    filtered_true = []
    filtered_pred = []
    for t, p in zip(y_true, y_pred):
        if t is not None and p is not None and not np.isnan(t) and not np.isnan(p):
            filtered_true.append(t)
            filtered_pred.append(p)
    if len(filtered_true) < 2:
        return {"support": len(filtered_true)}
    mse = mean_squared_error(filtered_true, filtered_pred)
    mae = mean_absolute_error(filtered_true, filtered_pred)
    r2 = r2_score(filtered_true, filtered_pred)
    return {
        "mse": round(mse, 4),
        "rmse": round(np.sqrt(mse), 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "support": len(filtered_true),
    }


def _compute_string_metrics(y_true, y_pred):
    exact_matches = sum(
        1 for t, p in zip(y_true, y_pred) if t is not None and p is not None and t == p
    )
    total_valid = sum(
        1 for t, p in zip(y_true, y_pred) if t is not None and p is not None
    )
    return {
        "accuracy": round(exact_matches / total_valid, 4) if total_valid > 0 else 0.0,
        "correct": exact_matches,
        "total": total_valid,
    }


def evaluate():
    parser = argparse.ArgumentParser(description="VM.AI Parser Evaluation")
    parser.add_argument(
        "--mode",
        choices=["both", "synthetic", "real", "specific", "modify_only"],
        default="both",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data split"
    )
    parser.add_argument("--test_size", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Evaluation batch size"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Max generation length"
    )
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width")
    parser.add_argument("--output_json", default=None, help="Save metrics to JSON file")
    parser.add_argument(
        "--output_csv", default=None, help="Save per-instance predictions to CSV"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Limit test samples (for debugging)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(args.mode)

    evals_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals")
    os.makedirs(evals_dir, exist_ok=True)
    if not args.output_json:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        args.output_json = os.path.join(evals_dir, f"eval_{ts}.json")

    checkpoint = args.checkpoint or cfg.output_dir
    print(f"Loading model from: {checkpoint}")
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found at {checkpoint}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_cache)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    special_tokens = ["[EXP]", "[PRD]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    print(f"Building dataset (mode={args.mode})...")
    train_ds, test_ds = build_dataset(cfg, args.mode)
    if args.max_test_samples and args.max_test_samples < len(test_ds):
        test_ds = test_ds.select(range(args.max_test_samples))
    print(f"Test samples: {len(test_ds)}")
    _, tok_test = tokenize(train_ds, test_ds, tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    print(f"Running inference ({len(tok_test)} samples, beam={args.num_beams})...")
    all_preds = []
    all_labels = []
    batch = []
    t0 = time.time()

    for i, example in enumerate(tok_test):
        batch.append(example)
        if len(batch) == args.batch_size or i == len(tok_test) - 1:
            inputs = collator(batch)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_length,
                    num_beams=args.num_beams,
                    early_stopping=True,
                )

            all_preds.extend(generated_ids.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            batch = []

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i + 1}/{len(tok_test)}] {rate:.1f} samples/sec")

    total_time = time.time() - t0
    print(
        f"Inference done: {len(all_preds)} samples in {total_time:.1f}s ({len(all_preds) / total_time:.1f} samples/sec)"
    )

    # Pad variable-length sequences to uniform shape
    max_pred_len = max(arr.shape[0] for arr in all_preds)
    padded_preds = np.full(
        (len(all_preds), max_pred_len), tokenizer.pad_token_id, dtype=np.int64
    )
    for i, arr in enumerate(all_preds):
        padded_preds[i, : len(arr)] = arr
    predictions = padded_preds

    max_label_len = max(arr.shape[0] for arr in all_labels)
    padded_labels = np.full(
        (len(all_labels), max_label_len), tokenizer.pad_token_id, dtype=np.int64
    )
    for i, arr in enumerate(all_labels):
        padded_labels[i, : len(arr)] = arr
    label_ids = padded_labels

    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
    label_ids = np.where(label_ids < 0, tokenizer.pad_token_id, label_ids)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    print()
    print("-" * 60)
    print("Accuracy per field (training eval)")
    print("-" * 60)
    orig_metrics = compute_metrics((predictions, label_ids), tokenizer)
    for key, val in orig_metrics.items():
        print(f"  {key}: {val}")

    print()
    print("-" * 60)
    print("Detailed metrics")
    print("-" * 60)

    pred_values, label_values, valid_mask = _gather_field_values(
        decoded_preds, decoded_labels
    )
    all_metrics = {}

    for field in TRACKED_FIELDS:
        y_true = [
            label_values[field][i]
            for i in range(len(label_values[field]))
            if valid_mask[field][i]
        ]
        y_pred = [
            pred_values[field][i]
            for i in range(len(pred_values[field]))
            if valid_mask[field][i]
        ]

        if len(y_true) == 0:
            all_metrics[field] = {"samples": 0}
            continue

        print(f"\n  {field}")
        if field in REGRESSION_FIELDS:
            metrics = _compute_regression_metrics(y_true, y_pred)
            all_metrics[field] = {"type": "regression", **metrics}
            print(
                f"    mae={metrics.get('mae', '?')}  rmse={metrics.get('rmse', '?')}  r2={metrics.get('r2', '?')}  n={metrics.get('support', 0)}"
            )

            diffs = [
                abs(t - p)
                for t, p in zip(y_true, y_pred)
                if t is not None and p is not None
            ]
            if diffs:
                within_strs = []
                for thresh in [0.05, 0.1, 0.15, 0.2]:
                    within = sum(1 for d in diffs if d <= thresh)
                    within_strs.append(
                        f"within {thresh}: {within}/{len(diffs)} ({round(100 * within / len(diffs), 1)}%)"
                    )
                print(f"    {', '.join(within_strs)}")

        elif field in BINARY_FIELDS:
            metrics = _compute_binary_metrics(y_true, y_pred)
            all_metrics[field] = {"type": "binary", **metrics}
            print(
                f"    f1={metrics.get('f1', '?')}  prec={metrics.get('precision', '?')}  rec={metrics.get('recall', '?')}  acc={metrics.get('accuracy', '?')}  n={metrics.get('support', 0)}"
            )

            y_true_bool = [t for t in y_true if t is not None]
            y_pred_bool = [p for p in y_pred if p is not None]
            if len(y_true_bool) > 0:
                tp = sum(1 for t, p in zip(y_true_bool, y_pred_bool) if t and p)
                fp = sum(1 for t, p in zip(y_true_bool, y_pred_bool) if not t and p)
                fn = sum(1 for t, p in zip(y_true_bool, y_pred_bool) if t and not p)
                tn = sum(1 for t, p in zip(y_true_bool, y_pred_bool) if not t and not p)
                print(f"    tp={tp} fp={fp} fn={fn} tn={tn}")

        elif field in CLASSIFICATION_FIELDS:
            n_classes = len(set(y_true))
            if n_classes < 2:
                acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
                all_metrics[field] = {
                    "type": "classification",
                    "accuracy": round(acc, 4),
                    "n_classes": 1,
                    "support": len(y_true),
                }
                print(f"    acc={round(acc, 4)}  (single class, n={len(y_true)})")
            else:
                metrics = _compute_classification_metrics(y_true, y_pred)
                all_metrics[field] = {"type": "classification", **metrics}
                print(
                    f"    f1-macro={metrics.get('f1_macro', '?')}  f1-weighted={metrics.get('f1_weighted', '?')}  acc={metrics.get('accuracy', '?')}  n={metrics.get('support', 0)}"
                )
                try:
                    report = classification_report(
                        y_true, y_pred, zero_division=0, digits=4
                    )
                    for line in report.split("\n"):
                        if line.strip():
                            print(f"      {line}")
                except Exception:
                    pass

        else:
            metrics = _compute_string_metrics(y_true, y_pred)
            all_metrics[field] = {"type": "string_match", **metrics}
            print(
                f"    acc={metrics['accuracy']}  ({metrics['correct']}/{metrics['total']})"
            )

    print()
    print("-" * 60)
    print("Summary")
    print("-" * 60)
    print(f"  {'Field':<20} {'Metric':<15} {'Value':<10} {'N':<6}")
    print(f"  {'-' * 18}  {'-' * 13}  {'-' * 8}  {'-' * 4}")
    for field in TRACKED_FIELDS:
        m = all_metrics.get(field, {})
        ftype = m.get("type", "?")
        if m.get("samples") == 0:
            print(f"  {field:<20} {'(no data)':<15}")
            continue
        if ftype == "regression":
            metric = "mae"
            val = m.get("mae", "?")
            support = m.get("support", 0)
        elif ftype == "binary":
            metric = "f1"
            val = m.get("f1", "?")
            support = m.get("support", 0)
        elif ftype == "classification":
            metric = "f1-macro"
            val = m.get("f1_macro", m.get("accuracy", "?"))
            support = m.get("support", 0)
        else:
            metric = "acc"
            val = m.get("accuracy", "?")
            support = m.get("total", m.get("support", 0))
        print(f"  {field:<20} {metric:<15} {str(val):<10} {str(support):<6}")

    acc_overall = orig_metrics.get("acc_overall", 0)
    print(f"  {'-' * 18}  {'-' * 13}  {'-' * 8}  {'-' * 4}")
    print(f"  {'overall':<20} {'acc':<15} {str(acc_overall):<10}")

    output = {
        "config": {
            "mode": args.mode,
            "seed": args.seed,
            "test_size": args.test_size,
            "checkpoint": checkpoint,
            "num_beams": args.num_beams,
        },
        "accuracy": orig_metrics,
        "detailed": all_metrics,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved to {args.output_json}")

    if args.output_csv:
        import csv

        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "prediction", "ground_truth"] + TRACKED_FIELDS)
            for i in range(min(len(decoded_preds), len(decoded_labels))):
                pred_dict = parse_pipe_simple(decoded_preds[i])
                label_dict = parse_pipe_simple(decoded_labels[i])
                input_text = (
                    test_ds[i].get("input_text", "") if i < len(test_ds) else ""
                )
                row = [input_text, decoded_preds[i], decoded_labels[i]]
                for field in TRACKED_FIELDS:
                    row.append(
                        pred_dict.get(field, "") + "|" + label_dict.get(field, "")
                    )
                writer.writerow(row)
        print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    evaluate()
