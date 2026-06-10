"""
VM-AI - Explicit Field Detection Tests
Verifies detect_explicit_fields() doesn't over-trigger on common words like "at", "by".
No model required.
Run: python tests/test_explicit_fields.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "parser"))
from schemas import detect_explicit_fields

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
print("  EXPLICIT FIELD DETECTION TESTS")
print("=" * 80)

print("\n[1] NO OVER-TRIGGER ON COMMON WORDS")
r = detect_explicit_fields("look at the board")
c("'at' no deadline", "deadline" not in r, f"got: {r}")
c("'at' no start", "start" not in r, f"got: {r}")
r = detect_explicit_fields("stand by the door")
c("'by' no deadline", "deadline" not in r, f"got: {r}")
c("'by' no start", "start" not in r, f"got: {r}")
r = detect_explicit_fields("buy a new book")
c("'buy' no deadline", "deadline" not in r, f"got: {r}")

print("\n[2] CORRECT TRIGGERS")
r = detect_explicit_fields("finish by tomorrow")
c("'by tomorrow' -> deadline", "deadline" in r)
c("'by tomorrow' -> start", "start" in r)
r = detect_explicit_fields("due next week")
c("'due' -> deadline", "deadline" in r)
r = detect_explicit_fields("meeting at 3pm")
c("'3pm' -> fixed_time", "fixed_time" in r)
c("'3pm' -> fixed_start", "fixed_start" in r)
r = detect_explicit_fields("hard task")
c("'hard' -> difficulty", "difficulty" in r)
r = detect_explicit_fields("urgent email")
c("'urgent' -> importance", "importance" in r)
r = detect_explicit_fields("30 minute run")
c("'minute' -> duration", "duration" in r)
r = detect_explicit_fields("run every morning")
c("'every' -> recurrent", "recurrent" in r)
c("'every' -> recurrence_days", "recurrence_days" in r)
r = detect_explicit_fields("workout at the gym")
c("'gym' -> category", "category" in r)

print("\n[3] EDGE CASES")
r = detect_explicit_fields("")
c("empty input has name", "name" in r)
c("empty input no deadline", "deadline" not in r)
r = detect_explicit_fields("just a task")
c("generic has name", "name" in r)
c("generic no other fields", len(r) == 1, f"got: {r}")
r = detect_explicit_fields("by the way, do something")
c("'by the way' no trigger", "deadline" not in r, f"got: {r}")

print(f"\n{'=' * 80}")
print(
    f"  RESULTS: {passed}/{passed + failed} passed ({100 * passed // (passed + failed) if passed + failed else 0}%)"
)
print(f"{'=' * 80}")
sys.exit(0 if failed == 0 else 1)
