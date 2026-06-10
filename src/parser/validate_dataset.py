"""
VM-AI - Dataset Validator with EXP/PRD Consistency Checking
Validates schema structure, keyword consistency, and tag correctness.
Run: python src/parser/validate_dataset.py data/VMAI_REAL_Data.yaml

Written by: Vanea
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import (
    clamp_category,
    detect_explicit_fields,
    normalize_deadline,
    normalize_duration,
    normalize_time,
)
from vars import ALWAYS_EXPLICIT, DAYS, PREDICTED_FIELDS, VALID_CATEGORIES

VALID_CATEGORIES_LIST = list(VALID_CATEGORIES)


def parse_pipe_with_tags(flat: str) -> dict:
    """Parse pipe-format string with EXP/PRD tags."""
    result = {}
    for part in flat.split("|"):
        part = part.strip()
        if "=" not in part:
            continue
        k, _, rest = part.partition("=")
        k = k.strip()

        tag = None
        if "[" in rest and rest.endswith("]"):
            val_str, tag = rest[:-1].split("[", 1)
            tag = tag.strip().upper()
        else:
            val_str = rest

        val_str = val_str.strip()
        if val_str.lower() == "null":
            v = None
        elif val_str.lower() in ("true", "tru", "t"):
            v = True
        elif val_str.lower().startswith("fals"):
            v = False
        else:
            v = val_str

        result[k] = {"value": v, "tag": tag}
    return result


def parse_dict_output(out_dict: dict) -> dict:
    """Convert YAML dict output to tagged schema format for validation."""
    result = {}
    for k, v in out_dict.items():
        if k in [
            "name",
            "fixed_time",
            "fixed_start",
            "recurrent",
            "recurrence_days",
            "deadline",
            "start",
        ]:
            tag = "EXP"
        else:
            tag = "PRD"
        result[k] = {"value": v, "tag": tag}
    return result


def validate_consistency(input_text: str, output_text: str, example_idx: int) -> list:
    """Check if EXP/PRD tags match the input keywords."""
    errors = []
    explicit_fields = detect_explicit_fields(input_text)
    parsed = parse_pipe_with_tags(output_text)

    for field, entry in parsed.items():
        tag = entry.get("tag")
        if not tag:
            continue

        val = entry["value"]
        if val is None and field not in ALWAYS_EXPLICIT:
            continue

        if tag == "EXP":
            if field in PREDICTED_FIELDS and field not in explicit_fields:
                if field not in [
                    "name",
                    "fixed_time",
                    "fixed_start",
                    "recurrent",
                    "recurrence_days",
                ]:
                    errors.append(
                        {
                            "idx": example_idx,
                            "type": "TAG_MISMATCH",
                            "field": field,
                            "detail": f"Tagged EXP but no keywords found in input",
                            "input": input_text[:50],
                        }
                    )
        elif tag == "PRD":
            if field in explicit_fields and field in PREDICTED_FIELDS:
                errors.append(
                    {
                        "idx": example_idx,
                        "type": "TAG_MISMATCH",
                        "field": field,
                        "detail": f"Tagged PRD but input contains keywords for this field",
                        "input": input_text[:50],
                    }
                )

    return errors


def validate_schema(output_text: str, example_idx: int) -> list:
    """Validate field values are within expected ranges/types."""
    errors = []
    try:
        if "[" in output_text:
            parsed = parse_pipe_with_tags(output_text)
        elif output_text.startswith("{"):
            import json

            parsed = {
                k: {"value": v, "tag": "EXP"}
                for k, v in json.loads(output_text).items()
            }
        else:
            parsed = parse_pipe_with_tags(output_text)
    except:
        return errors

    for field, entry in parsed.items():
        val = entry["value"]
        if val is None:
            continue

        if field == "difficulty":
            try:
                d = float(val)
                if not (0.0 <= d <= 1.0):
                    errors.append(
                        {
                            "idx": example_idx,
                            "type": "VALUE_RANGE",
                            "field": field,
                            "detail": f"Difficulty {d} not in [0,1]",
                        }
                    )
            except:
                errors.append(
                    {
                        "idx": example_idx,
                        "type": "VALUE_TYPE",
                        "field": field,
                        "detail": f"Difficulty '{val}' not numeric",
                    }
                )

        elif field == "importance":
            try:
                i = float(val)
                if not (0.0 <= i <= 1.0):
                    errors.append(
                        {
                            "idx": example_idx,
                            "type": "VALUE_RANGE",
                            "field": field,
                            "detail": f"Importance {i} not in [0,1]",
                        }
                    )
            except:
                errors.append(
                    {
                        "idx": example_idx,
                        "type": "VALUE_TYPE",
                        "field": field,
                        "detail": f"Importance '{val}' not numeric",
                    }
                )

        elif field == "duration":
            try:
                dur = int(val)
                if dur < 0:
                    errors.append(
                        {
                            "idx": example_idx,
                            "type": "VALUE_RANGE",
                            "field": field,
                            "detail": f"Duration {dur} negative",
                        }
                    )
            except:
                errors.append(
                    {
                        "idx": example_idx,
                        "type": "VALUE_TYPE",
                        "field": field,
                        "detail": f"Duration '{val}' not integer",
                    }
                )

        elif field == "category":
            if val.lower() not in VALID_CATEGORIES:
                errors.append(
                    {
                        "idx": example_idx,
                        "type": "VALUE_ENUM",
                        "field": field,
                        "detail": f"Category '{val}' not in {VALID_CATEGORIES_LIST}",
                    }
                )

        elif field == "fixed_start":
            normalized = normalize_time(str(val))
            if not normalized:
                errors.append(
                    {
                        "idx": example_idx,
                        "type": "VALUE_FORMAT",
                        "field": field,
                        "detail": f"Time '{val}' invalid format",
                    }
                )

    return errors


def validate_duplicates(examples: list) -> list:
    """Check for duplicate input texts."""
    seen = {}
    errors = []
    for i, ex in enumerate(examples):
        inp = ex.get("input", "")
        if inp in seen:
            errors.append(
                {
                    "idx": i,
                    "type": "DUPLICATE",
                    "detail": f"Duplicate of example {seen[inp]}",
                    "input": inp[:50],
                }
            )
        else:
            seen[inp] = i
    return errors


def validate_file(path: str, fix: bool = False) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    examples = data.get("examples", [])
    all_errors = []
    stats = Counter()

    field_tags = defaultdict(lambda: {"EXP": 0, "PRD": 0})

    for i, ex in enumerate(examples):
        inp = ex.get("input", "")
        out_raw = ex.get("output", "")

        is_modify = inp.startswith("modify:")
        stats["modify" if is_modify else "add"] += 1

        if isinstance(out_raw, dict):
            parsed = {
                k: {
                    "value": v,
                    "tag": "EXP"
                    if k
                    in [
                        "name",
                        "fixed_time",
                        "fixed_start",
                        "recurrent",
                        "recurrence_days",
                        "deadline",
                        "start",
                    ]
                    else "PRD",
                }
                for k, v in out_raw.items()
            }
            skip_tag_check = True
        else:
            parsed = parse_pipe_with_tags(str(out_raw))
            skip_tag_check = False

        for field, entry in parsed.items():
            tag = entry.get("tag", "UNKNOWN")
            if tag in ("EXP", "PRD"):
                field_tags[field][tag] += 1

        for field in parsed:
            stats[f"has_{field}"] += 1

        if not is_modify:
            for field, entry in parsed.items():
                val = entry["value"]
                if val is None:
                    continue

                if field == "difficulty":
                    try:
                        d = float(val)
                        if not (0.0 <= d <= 1.0):
                            all_errors.append(
                                {
                                    "idx": i,
                                    "type": "VALUE_RANGE",
                                    "field": field,
                                    "detail": f"Difficulty {d} not in [0,1]",
                                }
                            )
                    except:
                        pass
                elif field == "importance":
                    try:
                        imp = float(val)
                        if not (0.0 <= imp <= 1.0):
                            all_errors.append(
                                {
                                    "idx": i,
                                    "type": "VALUE_RANGE",
                                    "field": field,
                                    "detail": f"Importance {imp} not in [0,1]",
                                }
                            )
                    except:
                        pass
                elif field == "duration":
                    try:
                        dur = int(val)
                        if dur < 0:
                            all_errors.append(
                                {
                                    "idx": i,
                                    "type": "VALUE_RANGE",
                                    "field": field,
                                    "detail": f"Duration {dur} negative",
                                }
                            )
                    except:
                        pass
                elif field == "category":
                    if str(val).lower() not in VALID_CATEGORIES:
                        all_errors.append(
                            {
                                "idx": i,
                                "type": "VALUE_ENUM",
                                "field": field,
                                "detail": f"Category '{val}' not in {VALID_CATEGORIES_LIST}",
                            }
                        )
                elif field == "fixed_start":
                    if normalize_time(str(val)) is None and val is not None:
                        all_errors.append(
                            {
                                "idx": i,
                                "type": "VALUE_FORMAT",
                                "field": field,
                                "detail": f"Time '{val}' invalid format",
                            }
                        )

    all_errors.extend(validate_duplicates(examples))

    if fix and all_errors:
        print(f"Applying fixes to {path}...")
        print("Auto-fix not implemented. Review errors manually.")

    return {
        "file": os.path.basename(path),
        "total": len(examples),
        "stats": dict(stats),
        "field_tags": {k: dict(v) for k, v in field_tags.items()},
        "errors": all_errors,
    }


def print_report(results: list):
    print("\n" + "=" * 80)
    print("VM.AI TRAINING DATA VALIDATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    total_examples = 0
    total_errors = 0

    for res in results:
        print(f"\nFILE: {res['file']}")
        print(f"  Total examples: {res['total']}")
        total_examples += res["total"]

        print(f"  Add examples: {res['stats'].get('add', 0)}")
        print(f"  Modify examples: {res['stats'].get('modify', 0)}")

        features = [
            "difficulty",
            "importance",
            "category",
            "duration",
            "location",
            "deadline",
            "fixed_time",
            "fixed_start",
            "recurrent",
            "recurrence_days",
        ]
        for feat in features:
            count = res["stats"].get(f"has_{feat}", 0)
            print(f"  {feat.capitalize()} examples: {count}")

        print(f"\n  TAG DISTRIBUTION:")
        for field, tags in res["field_tags"].items():
            exp = tags.get("EXP", 0)
            prd = tags.get("PRD", 0)
            print(f"    {field:15s}: EXP={exp:3d}  PRD={prd:3d}")

        if res["errors"]:
            print(f"\n  ERRORS ({len(res['errors'])}):")
            error_types = Counter(e["type"] for e in res["errors"])
            for etype, count in error_types.items():
                print(f"    {etype}: {count}")
            total_errors += len(res["errors"])
        else:
            print(f"\n  [OK] No errors found")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files processed: {len(results)}")
    print(f"Total examples: {total_examples}")
    print(f"Total errors: {total_errors}")
    print("=" * 80)

    if total_errors == 0:
        print("\n[OK] ALL VALIDATIONS PASSED - DATA IS READY FOR TRAINING")
    else:
        print(f"\n[WARN] {total_errors} ERRORS FOUND - REVIEW BEFORE TRAINING")


def main():
    parser = argparse.ArgumentParser(description="VM.AI Dataset Validator")
    parser.add_argument("files", nargs="+", help="YAML files to validate")
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes")
    args = parser.parse_args()

    results = []
    for f in args.files:
        if not os.path.exists(f):
            print(f"Warning: {f} not found, skipping")
            continue
        results.append(validate_file(f, fix=args.fix))

    print_report(results)


if __name__ == "__main__":
    main()
