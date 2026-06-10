"""
VM-AI - Schema and Parsing Utilities with EXP/PRD Support
Centralizes all parsing, normalization, and tag logic.

Format: field=value[TAG] | field2=value2[TAG2]
Tags: EXP = Explicit (user stated), PRD = Predicted (model inferred)

Written by: Vanea
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vars import ALL_FIELDS, ALWAYS_EXPLICIT, DAYS, PREDICTED_FIELDS, VALID_CATEGORIES

DIFFICULTY_KEYWORDS = {
    "hard",
    "difficult",
    "challenging",
    "complex",
    "intense",
    "heavy",
    "tough",
    "easy",
    "simple",
    "light",
    "quick",
    "moderate",
    "medium",
    "urgent",
}

IMPORTANCE_KEYWORDS = {
    "urgent",
    "critical",
    "asap",
    "emergency",
    "important",
    "priority",
    "must",
    "low priority",
    "not urgent",
    "minor",
    "can wait",
    "whenever",
    "optional",
}

CATEGORY_KEYWORDS = {
    "work",
    "fitness",
    "health",
    "finance",
    "study",
    "home",
    "shopping",
    "travel",
    "creative",
    "learning",
    "admin",
    "errands",
    "social",
    "family",
    "personal",
}

DURATION_KEYWORDS = {"minute", "minutes", "min", "hour", "hours", "hr"}

TIME_KEYWORDS = {"am", "pm", "morning", "afternoon", "evening", "noon", "midnight"}

RECURRENCE_KEYWORDS = {"every", "daily", "each", "weekday", "weekly", "repeat"}


def detect_explicit_fields(input_text: str) -> set:
    """Returns a set of field names that should be marked EXP based on input keywords."""
    s = input_text.lower()
    explicit = set()

    explicit.add("name")

    if any(kw in s for kw in DIFFICULTY_KEYWORDS):
        explicit.add("difficulty")

    if any(kw in s for kw in IMPORTANCE_KEYWORDS):
        explicit.add("importance")

    if any(kw in s for kw in CATEGORY_KEYWORDS):
        explicit.add("category")

    if any(kw in s for kw in DURATION_KEYWORDS) or re.search(r"\d+\s*(min|hour|hr)", s):
        explicit.add("duration")

    if any(kw in s for kw in TIME_KEYWORDS) or re.search(
        r"\d{1,2}(:\d{2})?\s*(am|pm)", s
    ):
        explicit.add("fixed_time")
        explicit.add("fixed_start")

    if any(kw in s for kw in RECURRENCE_KEYWORDS):
        explicit.add("recurrent")
        explicit.add("recurrence_days")

    if any(
        kw in s
        for kw in [
            "tomorrow",
            "next week",
            "due",
            "deadline",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
    ):
        explicit.add("deadline")
        explicit.add("start")

    if re.search(
        r"\bby\s+(tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|today|tonight|the weekend|eod)",
        s,
    ):
        explicit.add("deadline")
        explicit.add("start")

    return explicit


def normalize_time(time_str: str) -> str | None:
    """Convert various time formats to HH:MM."""
    if not time_str:
        return None
    time_str = str(time_str).strip().lower()

    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        ampm = match.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    if "morning" in time_str:
        return "08:00"
    if "afternoon" in time_str:
        return "13:00"
    if "evening" in time_str:
        return "18:00"
    if "noon" in time_str:
        return "12:00"
    if "midnight" in time_str:
        return "00:00"

    if re.match(r"^\d{1,2}:\d{2}$", time_str):
        parts = time_str.split(":")
        if 0 <= int(parts[0]) <= 23:
            return time_str

    return None


def normalize_duration(val) -> str | None:
    """Convert any duration value to integer minutes string."""
    if val is None:
        return None
    val = str(val).lower().strip()
    if val.isdigit():
        return val

    match = re.search(r"(\d+(?:\.\d+)?)\s*hours?", val)
    if match:
        return str(int(float(match.group(1)) * 60))

    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|min)", val)
    if match:
        return str(int(float(match.group(1))))

    if "half" in val and "day" in val:
        return "720"
    if "all day" in val:
        return "960"

    match = re.search(r"(\d+(?:\.\d+)?)", val)
    if match:
        num = float(match.group(1))
        return str(int(num * 60)) if num <= 24 else str(int(num))
    return None


def normalize_deadline(val) -> str | None:
    """Normalize deadline/start to a small fixed vocabulary."""
    if val is None:
        return None
    s = str(val).lower().strip()
    valid = {
        "today",
        "tomorrow",
        "tonight",
        "this weekend",
        "next week",
        "this week",
        "next month",
    }
    if s in valid:
        return s
    for d in DAYS:
        if s == d.lower() or s == d:
            return d
    for d in DAYS:
        if f"next {d.lower()}" in s:
            return f"next {d}"
    if "next week" in s:
        return "next week"
    if "tomorrow" in s:
        return "tomorrow"
    if "today" in s:
        return "today"
    if "tonight" in s:
        return "tonight"
    if "weekend" in s:
        return "this weekend"
    if "eod" in s or "end of day" in s:
        return "today"
    if "end of week" in s:
        return "this weekend"
    if "end of month" in s:
        return "next week"
    for m in [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]:
        if m in s:
            return "next week"
    if "q1" in s or "q2" in s or "q3" in s or "q4" in s:
        return "next week"
    if "asap" in s or "soon" in s:
        return "tomorrow"
    for d in DAYS:
        if d.lower() in s:
            return d
    return s


def clamp_category(cat):
    if cat is None:
        return None
    cat = str(cat).lower().strip()
    return cat if cat in VALID_CATEGORIES else "personal"


def pipe_to_schema(flat: str, input_text: str = "") -> dict:
    """
    Parse pipe-format string with EXP/PRD tags into schema dict.
    Format: name=gym[EXP] | difficulty=0.6[PRD] | category=fitness[PRD]
    If no tags present, auto-detect based on input_text.
    Handles T5 sentinel tokens (<extra_id_*>) — strips them entirely.
    """
    flat = re.sub(r"<extra_id_\d+>", "", flat)
    flat = re.sub(r"\s*\|\s*<extra_id_\d+>\s*\|\s*", " | ", flat)
    flat = re.sub(r"\s*\|\s*\|\s*", " | ", flat)

    raw = {}
    for part in flat.split("|"):
        part = part.strip()
        if "=" not in part:
            continue
        k, _, rest = part.partition("=")
        k = k.replace("-", "_")
        k = k.strip()

        tag = None
        if "[" in rest and rest.endswith("]"):
            val_str, tag = rest[:-1].split("[", 1)
            tag = tag.strip()
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

        raw[k] = {"value": v, "tag": tag}

    schema = {}
    for field, default in ALL_FIELDS.items():
        if field in raw:
            entry = raw[field]
            val = entry["value"]
            tag = entry.get("tag")

            if not tag:
                explicit_fields = detect_explicit_fields(input_text)
                tag = "EXP" if field in explicit_fields else "PRD"

            schema[field] = {"value": val, "predicted": tag == "PRD"}
        else:
            explicit_fields = detect_explicit_fields(input_text)
            tag = "EXP" if field in explicit_fields else "PRD"
            schema[field] = {"value": default, "predicted": tag == "PRD"}

    return schema


def schema_to_pipe(schema: dict) -> str:
    """
    Convert schema dict to pipe-format string with EXP/PRD tags.
    Format: name=gym[EXP] | difficulty=0.6[PRD]
    """
    parts = []
    for field, entry in schema.items():
        val = entry["value"]
        tag = "PRD" if entry.get("predicted", True) else "EXP"

        if val is None:
            continue
        if field == "duration":
            val = normalize_duration(val)
            if val is None:
                continue
        if field == "fixed_start":
            val = normalize_time(str(val))
            if val is None:
                continue
        if field == "deadline" or field == "start":
            val = normalize_deadline(val)
            if val is None:
                continue

        if isinstance(val, bool):
            parts.append(f"{field}={'true' if val else 'false'}[{tag}]")
        elif isinstance(val, list):
            parts.append(f"{field}={','.join(val)}[{tag}]")
        else:
            parts.append(f"{field}={val}[{tag}]")
    return " | ".join(parts)


def changed_to_pipe(changed: dict) -> str:
    """Convert changed fields dict to pipe-format string with tags."""
    parts = []
    for field, entry in changed.items():
        val = entry["value"]
        tag = "PRD" if entry.get("predicted", True) else "EXP"

        if val is None:
            continue
        if field == "duration":
            val = normalize_duration(val)
            if val is None:
                continue
        if field == "fixed_start":
            val = normalize_time(str(val))
            if val is None:
                continue

        if isinstance(val, bool):
            parts.append(f"{field}={'true' if val else 'false'}[{tag}]")
        else:
            parts.append(f"{field}={val}[{tag}]")
    return " | ".join(parts)


def parse_pipe_simple(flat: str) -> dict:
    """Parse pipe-format string WITHOUT tags into simple dict."""
    result = {}
    for part in flat.split("|"):
        part = part.strip()
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        k, v = k.strip().lower(), v.strip()
        if v in ("null", ""):
            v = None
        result[k] = v
    return result
