"""
VM-AI - Schema Function Tests
Tests normalize_time, normalize_duration, normalize_deadline, clamp_category, detect_explicit_fields.
No model required.
Run: python tests/test_schemas.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "parser"))
from schemas import (
    clamp_category,
    detect_explicit_fields,
    normalize_deadline,
    normalize_duration,
    normalize_time,
)

passed = 0
failed = 0


def c(n, ok, d=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS | {n}")
    else:
        failed += 1
        print(f"  FAIL | {n} | {d}")


print("=" * 80)
print("  SCHEMA FUNCTION TESTS")
print("=" * 80)

print("\n[1] NORMALIZE TIME")
c("12h AM", normalize_time("12:00am") == "00:00")
c("12h PM", normalize_time("12:00pm") == "12:00")
c("6am", normalize_time("6am") == "06:00")
c("3pm", normalize_time("3pm") == "15:00")
c("9:30pm", normalize_time("9:30pm") == "21:30")
c("morning", normalize_time("morning") == "08:00")
c("afternoon", normalize_time("afternoon") == "13:00")
c("evening", normalize_time("evening") == "18:00")
c("noon", normalize_time("noon") == "12:00")
c("midnight", normalize_time("midnight") == "00:00")
c("24h format", normalize_time("14:00") == "14:00")
c("invalid", normalize_time("abc") is None)
c("empty", normalize_time("") is None)
c("none", normalize_time(None) is None)

print("\n[2] NORMALIZE DURATION")
c("plain number", normalize_duration("45") == "45")
c("30 minutes", normalize_duration("30 minutes") == "30")
c("30 min", normalize_duration("30 min") == "30")
c("2 hours", normalize_duration("2 hours") == "120")
c("1 hour", normalize_duration("1 hour") == "60")
c("1.5 hours", normalize_duration("1.5 hours") == "90")
c("half day", normalize_duration("half day") == "720")
c("all day", normalize_duration("all day") == "960")
c("2.5 (hours)", normalize_duration("2.5") == "150")
c("none", normalize_duration(None) is None)
c("empty", normalize_duration("") is None)

print("\n[3] NORMALIZE DEADLINE")
c("today", normalize_deadline("today") == "today")
c("tomorrow", normalize_deadline("tomorrow") == "tomorrow")
c("tonight", normalize_deadline("tonight") == "tonight")
c("this weekend", normalize_deadline("this weekend") == "this weekend")
c("next week", normalize_deadline("next week") == "next week")
c("Monday", normalize_deadline("monday") == "Monday")
c("Monday capitalized", normalize_deadline("Monday") == "Monday")
c("next Monday", normalize_deadline("next monday") == "next Monday")
c("eod", normalize_deadline("eod") == "today")
c("end of day", normalize_deadline("end of day") == "today")
c("end of week", normalize_deadline("end of week") == "this weekend")
c("asap", normalize_deadline("asap") == "tomorrow")
c("weekend in text", normalize_deadline("by the weekend") == "this weekend")
c("none", normalize_deadline(None) is None)

print("\n[4] CLAMP CATEGORY")
c("valid work", clamp_category("work") == "work")
c("valid fitness", clamp_category("fitness") == "fitness")
c("uppercase", clamp_category("WORK") == "work")
c("invalid", clamp_category("sports") == "personal")
c("none", clamp_category(None) is None)
c("empty string", clamp_category("") == "personal")

print("\n[5] DETECT EXPLICIT FIELDS")
r = detect_explicit_fields("gym at 6am tomorrow")
c("has name", "name" in r)
c("has fixed_time", "fixed_time" in r)
c("has fixed_start", "fixed_start" in r)
c("has deadline", "deadline" in r)
c("has start", "start" in r)
r = detect_explicit_fields("hard urgent workout")
c("has difficulty", "difficulty" in r)
c("has importance", "importance" in r)
r = detect_explicit_fields("30 minute gym session")
c("has duration", "duration" in r)
r = detect_explicit_fields("meditate every morning")
c("has recurrent", "recurrent" in r)
c("has recurrence_days", "recurrence_days" in r)
r = detect_explicit_fields("just a simple task")
c("no deadline", "deadline" not in r)
c("no fixed_time", "fixed_time" not in r)
c("no difficulty", "difficulty" not in r)

print(f"\n{'=' * 80}")
print(
    f"  RESULTS: {passed}/{passed + failed} passed ({100 * passed // (passed + failed) if passed + failed else 0}%)"
)
print(f"{'=' * 80}")
sys.exit(0 if failed == 0 else 1)
