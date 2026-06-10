"""
VM-AI - Modify Mode Tests
Tests modify mode with direct instruction format.
Model receives: just the change instruction (e.g. "push deadline to wednesday")
Model outputs: changed fields only (e.g. "deadline=Wednesday[EXP]")
Run: python tests/test_modify.py

Requires: finetuned_parser model in models/
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "parser"))
from chat import TaskPlannerPredictor


def g(r, f):
    """Get value from schema dict."""
    e = r.get(f, {})
    return e.get("value") if isinstance(e, dict) else e


p = TaskPlannerPredictor()
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
print("  MODIFY MODE TESTS")
print("=" * 80)

tests = [
    # (existing_task, change_instruction, field, expected, comparison_op)
    # ── Fixed Time ────────────────────────────────────────────────────────
    (
        {
            "name": "gym",
            "fixed_time": True,
            "fixed_start": "06:00",
            "duration": 45,
            "category": "fitness",
            "difficulty": 0.35,
            "importance": 0.51,
            "recurrent": False,
        },
        "move to 7am",
        "fixed_start",
        "07:00",
    ),
    (
        {
            "name": "meeting",
            "fixed_time": True,
            "fixed_start": "14:00",
            "duration": 30,
            "category": "work",
            "difficulty": 0.2,
            "importance": 0.6,
            "recurrent": False,
        },
        "move to 3pm",
        "fixed_start",
        "15:00",
    ),
    (
        {
            "name": "dentist",
            "fixed_time": True,
            "fixed_start": "10:00",
            "duration": 60,
            "category": "health",
            "difficulty": 0.15,
            "importance": 0.8,
            "recurrent": False,
        },
        "reschedule to 2pm",
        "fixed_start",
        "14:00",
    ),
    (
        {
            "name": "dinner",
            "fixed_time": True,
            "fixed_start": "18:00",
            "duration": 45,
            "category": "personal",
            "difficulty": 0.2,
            "importance": 0.5,
            "recurrent": False,
        },
        "change to 7pm",
        "fixed_start",
        "19:00",
    ),
    (
        {
            "name": "dentist",
            "fixed_time": True,
            "fixed_start": "14:00",
            "duration": 60,
            "category": "health",
            "difficulty": 0.15,
            "importance": 0.8,
            "recurrent": False,
        },
        "cancel fixed time",
        "fixed_time",
        False,
    ),
    (
        {
            "name": "meeting",
            "fixed_time": True,
            "fixed_start": "15:00",
            "duration": 60,
            "category": "work",
            "difficulty": 0.2,
            "importance": 0.6,
            "recurrent": False,
        },
        "make it flexible",
        "fixed_time",
        False,
    ),
    # ── Duration ──────────────────────────────────────────────────────────
    (
        {
            "name": "workout",
            "duration": 45,
            "category": "fitness",
            "difficulty": 0.6,
            "importance": 0.5,
            "fixed_time": False,
            "recurrent": False,
        },
        "make it 90 minutes",
        "duration",
        90,
    ),
    (
        {
            "name": "meeting",
            "duration": 30,
            "category": "work",
            "difficulty": 0.2,
            "importance": 0.6,
            "fixed_time": False,
            "recurrent": False,
        },
        "make it 1 hour",
        "duration",
        60,
    ),
    (
        {
            "name": "call",
            "duration": 30,
            "category": "work",
            "difficulty": 0.1,
            "importance": 0.4,
            "fixed_time": False,
            "recurrent": False,
        },
        "make it 15 minutes",
        "duration",
        15,
    ),
    # ── Difficulty ────────────────────────────────────────────────────────
    (
        {
            "name": "workout",
            "duration": 45,
            "category": "fitness",
            "difficulty": 0.35,
            "importance": 0.5,
            "fixed_time": False,
            "recurrent": False,
        },
        "make it harder",
        "difficulty",
        0.6,
        ">",
    ),
    (
        {
            "name": "coding task",
            "duration": 90,
            "category": "work",
            "difficulty": 0.5,
            "importance": 0.6,
            "fixed_time": False,
            "recurrent": False,
        },
        "make it easy",
        "difficulty",
        0.4,
        "<",
    ),
    (
        {
            "name": "study session",
            "duration": 60,
            "category": "study",
            "difficulty": 0.3,
            "importance": 0.5,
            "fixed_time": False,
            "recurrent": False,
        },
        "make it challenging",
        "difficulty",
        0.6,
        ">",
    ),
    # ── Importance ────────────────────────────────────────────────────────
    (
        {
            "name": "client call",
            "duration": 30,
            "category": "work",
            "difficulty": 0.3,
            "importance": 0.5,
            "fixed_time": False,
            "recurrent": False,
        },
        "make it urgent",
        "importance",
        0.7,
        ">",
    ),
    (
        {
            "name": "presentation",
            "duration": 60,
            "category": "work",
            "difficulty": 0.6,
            "importance": 0.5,
            "fixed_time": False,
            "recurrent": False,
        },
        "make it critical",
        "importance",
        0.8,
        ">",
    ),
    (
        {
            "name": "cleanup",
            "duration": 30,
            "category": "home",
            "difficulty": 0.2,
            "importance": 0.5,
            "fixed_time": False,
            "recurrent": False,
        },
        "make it low priority",
        "importance",
        0.4,
        "<",
    ),
    # ── Category ──────────────────────────────────────────────────────────
    (
        {
            "name": "yoga",
            "duration": 60,
            "difficulty": 0.35,
            "importance": 0.5,
            "category": "work",
            "fixed_time": False,
            "recurrent": False,
        },
        "categorize it as fitness",
        "category",
        "fitness",
    ),
    (
        {
            "name": "meditation",
            "duration": 15,
            "difficulty": 0.1,
            "importance": 0.5,
            "category": "work",
            "fixed_time": False,
            "recurrent": False,
        },
        "make it a health task",
        "category",
        "health",
    ),
    (
        {
            "name": "pay rent",
            "duration": 10,
            "difficulty": 0.1,
            "importance": 0.9,
            "category": "home",
            "fixed_time": False,
            "recurrent": False,
        },
        "actually its finance",
        "category",
        "finance",
    ),
    # ── Deadline ──────────────────────────────────────────────────────────
    (
        {
            "name": "report",
            "deadline": "Friday",
            "duration": 120,
            "category": "work",
            "difficulty": 0.5,
            "importance": 0.6,
            "fixed_time": False,
            "recurrent": False,
        },
        "push deadline to next Monday",
        "deadline",
        "next Monday",
    ),
    (
        {
            "name": "assignment",
            "deadline": "tomorrow",
            "duration": 90,
            "category": "study",
            "difficulty": 0.6,
            "importance": 0.7,
            "fixed_time": False,
            "recurrent": False,
        },
        "extend to next week",
        "deadline",
        "next week",
    ),
]

for t in tests:
    task, change, field, exp = t[0], t[1], t[2], t[3]
    op = t[4] if len(t) > 4 else "=="

    # Convert plain dict to schema dict
    schema = {k: {"value": v, "predicted": False} for k, v in task.items()}
    result = p.predict_modify(schema, change)

    if "error" in result:
        ok = False
        actual = result["error"]
    else:
        actual = g(result, field)
        if op == ">":
            try:
                ok = actual is not None and float(actual) > exp
            except:
                ok = False
        elif op == "<":
            try:
                ok = actual is not None and float(actual) < exp
            except:
                ok = False
        elif isinstance(exp, bool):
            ok = actual == exp
        else:
            ok = actual is not None and exp.lower() in str(actual).lower()

    c(f"mod: '{change[:35]}' -> {field}={exp}", ok, f"got={actual}")

print(f"\n{'=' * 80}")
print(
    f"  RESULTS: {passed}/{passed + failed} passed ({100 * passed // (passed + failed) if passed + failed else 0}%)"
)
print(f"{'=' * 80}")
sys.exit(0 if failed == 0 else 1)
