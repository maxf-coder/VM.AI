"""
VM.AI - Dataset Text Report Generator
Outputs concise text-based statistics for all datasets.
Run: python scripts/report.py
     python scripts/report.py --dataset real
"""

import argparse
import os
import re
import sys
from collections import Counter

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "real": os.path.join(ROOT, "data", "VMAI_REAL_Data.yaml"),
    "specific": os.path.join(ROOT, "data", "VMAI_SPECIFIC_Data.yaml"),
    "synthetic": os.path.join(ROOT, "data", "VMAI_SYNTHETIC_Data.yaml"),
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify(ex):
    out = ex.get("output", {})
    return "modify" if "name" not in out or len(out) <= 3 else "add"


def report_real_or_specific(name, data):
    examples = data.get("examples", [])
    if not examples:
        print(f"{name}: no examples")
        return

    records = [classify(ex) for ex in examples]
    adds = [ex for ex, r in zip(examples, records) if r == "add"]
    mods = [ex for ex, r in zip(examples, records) if r == "modify"]

    cats = [str(ex.get("output", {}).get("category", "?")).lower() for ex in examples]
    diffs = [
        float(ex["output"]["difficulty"])
        for ex in examples
        if ex.get("output", {}).get("difficulty") is not None
    ]
    imps = [
        float(ex["output"]["importance"])
        for ex in examples
        if ex.get("output", {}).get("importance") is not None
    ]
    durs = [
        float(ex["output"]["duration"])
        for ex in examples
        if ex.get("output", {}).get("duration") is not None
    ]
    fixed_time = sum(1 for ex in examples if ex.get("output", {}).get("fixed_time"))
    recurrent = sum(1 for ex in examples if ex.get("output", {}).get("recurrent"))
    deadlines = [
        ex["output"]["deadline"]
        for ex in examples
        if ex.get("output", {}).get("deadline")
    ]
    starts = [
        ex["output"]["start"] for ex in examples if ex.get("output", {}).get("start")
    ]
    locations = [
        str(ex["output"]["location"]).lower()
        for ex in examples
        if ex.get("output", {}).get("location")
    ]
    fixed_starts = [
        ex["output"]["fixed_start"]
        for ex in examples
        if ex.get("output", {}).get("fixed_start")
    ]
    rec_days = [
        ex["output"]["recurrence_days"]
        for ex in examples
        if ex.get("output", {}).get("recurrence_days")
    ]

    cat_counts = Counter(cats)
    dl_counts = Counter(str(d) for d in deadlines)
    start_counts = Counter(str(s) for s in starts)
    loc_counts = Counter(locations)
    fs_counts = Counter(str(f) for f in fixed_starts)
    rd_counts = Counter(str(r) for r in rec_days)

    seen = {}
    dups = []
    for i, ex in enumerate(examples):
        inp = ex.get("input", "")
        if inp in seen:
            dups.append((i, seen[inp]))
        else:
            seen[inp] = i

    mod_fields = Counter()
    for ex in mods:
        for k in ex.get("output", {}):
            if k != "name":
                mod_fields[k] += 1

    print(f"--- {name} ---")
    print(f"total: {len(examples)}  add: {len(adds)}  modify: {len(mods)}")
    print(f"duplicates: {len(dups)}")
    for dup_idx, orig_idx in dups:
        print(f"  idx={dup_idx} dup of idx={orig_idx}")

    print(f"categories: {len(cat_counts)}")
    for c, n in cat_counts.most_common():
        print(f"  {c}: {n}")

    if diffs:
        print(
            f"difficulty: min={min(diffs):.2f}  max={max(diffs):.2f}  avg={sum(diffs) / len(diffs):.2f}"
        )
    if imps:
        print(
            f"importance:  min={min(imps):.2f}  max={max(imps):.2f}  avg={sum(imps) / len(imps):.2f}"
        )
    if durs:
        print(
            f"duration:    min={min(durs):.0f}  max={max(durs):.0f}  avg={sum(durs) / len(durs):.0f}"
        )

    print(f"fixed_time: {fixed_time}  recurrent: {recurrent}")

    if loc_counts:
        print(f"locations: {len(loc_counts)}")
        for l, n in loc_counts.most_common():
            print(f"  {l}: {n}")

    if dl_counts:
        print(f"deadlines ({len(dl_counts)} unique):")
        for d, n in dl_counts.most_common():
            print(f"  {d}: {n}")

    if start_counts:
        print(f"starts ({len(start_counts)} unique):")
        for s, n in start_counts.most_common():
            print(f"  {s}: {n}")

    if fs_counts:
        print(f"fixed_start:")
        for f, n in fs_counts.most_common():
            print(f"  {f}: {n}")

    if rd_counts:
        print(f"recurrence_days:")
        for r, n in rd_counts.most_common():
            print(f"  {r}: {n}")

    if mod_fields:
        print(f"modify fields changed:")
        for f, n in mod_fields.most_common():
            print(f"  {f}: {n}")

    print()


def classify_template_section(template):
    t = template.lower()
    if any(
        k in t
        for k in [
            "[difficult",
            "hard one",
            "easy one",
            "moderate",
            "pretty ",
            "seems ",
            "looks ",
            "somewhat ",
            "quite ",
            "rather ",
        ]
    ):
        return "difficulty"
    if any(
        k in t
        for k in [
            "important",
            "urgent",
            "priority",
            "critical",
            "can wait",
            "optional",
            "mandatory",
            "essential",
            "nice to have",
            "top priority",
            "not a priority",
            "mildly",
            "absolutely",
        ]
    ):
        return "importance"
    if "[category]" in t:
        return "category"
    if any(k in t for k in ["every ", "daily", "weekday", "repeat"]):
        return "recurrence"
    if any(
        k in t
        for k in [
            "start ",
            "begin",
            "kick off",
            "commenc",
            "starting",
            "begins",
            "from [date",
        ]
    ):
        return "start"
    if any(k in t for k in ["due ", "deadline", "before ", "by [deadline"]):
        return "deadline+time"
    if any(
        k in t
        for k in [
            "morning",
            "afternoon",
            "evening",
            "noon",
            "midnight",
            "early ",
            "late ",
        ]
    ):
        return "time-of-day"
    if any(
        k in t
        for k in [
            "todo:",
            "task:",
            "reminder:",
            "don't let",
            "flag this",
            "heads up:",
            "note:",
            "action item",
        ]
    ):
        return "casual"
    return "basic"


def report_synthetic(name, data):
    templates = data.get("templates", [])
    if not templates:
        print(f"{name}: no templates")
        return

    all_ph = [ph for t in templates for ph in re.findall(r"\[([A-Z_]+)\]", t)]
    ph_counts = Counter(all_ph)
    fields_per = [len(re.findall(r"\[([A-Z_]+)\]", t)) for t in templates]
    section_counts = Counter(classify_template_section(t) for t in templates)

    print(f"--- {name} ---")
    print(f"templates: {len(templates)}")
    print(f"tasks: {len(data.get('tasks', []))}")

    print(
        f"fields/template: min={min(fields_per)}  max={max(fields_per)}  avg={sum(fields_per) / len(fields_per):.1f}"
    )
    fields_hist = Counter(fields_per)
    for fc in sorted(fields_hist.keys()):
        print(f"  {fc} fields: {fields_hist[fc]} templates")

    print(f"template sections:")
    for s, n in section_counts.most_common():
        print(f"  {s}: {n}")

    print(f"placeholders:")
    for ph, n in ph_counts.most_common():
        print(f"  {ph}: {n}")

    print(f"pools:")
    for key in [
        "deadlines",
        "durations",
        "dates",
        "times",
        "locations",
        "categories",
        "priorities",
        "difficulties",
        "recurrence_days",
    ]:
        print(f"  {key}: {len(data.get(key, []))}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["real", "specific", "synthetic", "all"], default="all"
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in targets:
        path = DATASETS[name]
        if not os.path.exists(path):
            print(f"{name}: file not found")
            continue
        data = load_yaml(path)
        if name == "synthetic":
            report_synthetic(name, data)
        else:
            report_real_or_specific(name, data)


if __name__ == "__main__":
    main()
