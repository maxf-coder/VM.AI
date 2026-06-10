"""
VM-AI - Data Normalizer with EXP/PRD Tag Support
Normalizes VMAI_REAL_Data.yaml and VMAI_SPECIFIC_Data.yaml to consistent formats with tags.
Run: python src/parser/normalize_data.py

Written by: Vanea
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from schemas import (
    clamp_category,
    detect_explicit_fields,
    normalize_deadline,
    normalize_duration,
    normalize_time,
)


def normalize_example(ex):
    """Normalize a single example's output dict, preserving or adding EXP/PRD tags."""
    out = ex.get("output", {})
    inp = ex.get("input", "")

    explicit_fields = detect_explicit_fields(inp)

    dur = out.get("duration")
    if dur is not None:
        dur = normalize_duration(dur)
        if dur is not None:
            out["duration"] = int(dur)
        else:
            del out["duration"]

    diff = out.get("difficulty")
    if diff is not None:
        try:
            out["difficulty"] = round(float(diff), 2)
        except (ValueError, TypeError):
            del out["difficulty"]

    imp = out.get("importance")
    if imp is not None:
        try:
            out["importance"] = round(float(imp), 2)
        except (ValueError, TypeError):
            del out["importance"]

    cat = out.get("category")
    if cat is not None:
        out["category"] = clamp_category(cat)

    start = out.get("start")
    if start is not None:
        start = normalize_deadline(start)
        out["start"] = start if start else None

    dl = out.get("deadline")
    if dl is not None:
        dl = normalize_deadline(dl)
        out["deadline"] = dl if dl else None

    fs = out.get("fixed_start")
    if fs is not None:
        fs = normalize_time(str(fs))
        out["fixed_start"] = fs if fs else None

    for key in ("fixed_time", "recurrent"):
        if key in out:
            out[key] = bool(out[key])

    rd = out.get("recurrence_days")
    if rd is not None:
        if isinstance(rd, str):
            rd = [d.strip() for d in rd.split(",")]
        rd = [
            d
            for d in rd
            if d
            in [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        ]
        out["recurrence_days"] = rd if rd else None

    ex["output"] = {k: v for k, v in out.items() if v is not None}
    return ex


def normalize_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    examples = data.get("examples", [])
    fixed = 0
    for i, ex in enumerate(examples):
        original = yaml.dump(ex, allow_unicode=True, sort_keys=True)
        examples[i] = normalize_example(ex)
        after = yaml.dump(examples[i], allow_unicode=True, sort_keys=True)
        if original != after:
            fixed += 1

    data["examples"] = examples

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"{path}: {fixed}/{len(examples)} examples normalized")
    return fixed, len(examples)


# TODO:
if __name__ == "__main__":
    files = [
        "D:/Users/user/Desktop/VM.AI/data/VMAI_REAL_Data.yaml",
    ]

    total_fixed = 0
    total_examples = 0
    for f in files:
        if os.path.exists(f):
            fixed, count = normalize_file(f)
            total_fixed += fixed
            total_examples += count

    print(f"\nTotal: {total_fixed}/{total_examples} examples normalized")
