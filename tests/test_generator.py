"""
VM-AI - Data Generator Tests
Tests keyword inference functions without requiring the model.
Run: python tests/test_generator.py

No model required.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "parser"))
from data_generator import DataGenerator

# ── Stub training data for generator init ────────────────────────────────────


class StubTrainingData:
    templates = ["[TASK] by [DEADLINE]."]
    tasks = ["test task"]
    durations = ["30"]
    deadlines = ["tomorrow"]
    locations = ["home"]
    dates = ["Monday"]
    times = ["09:00"]
    priorities = ["medium"]
    difficulties = ["0.5"]
    categories = ["work"]
    fixed_starts = ["09:00"]
    recurrence_days = ["Monday"]

    def get_placeholder_map(self):
        return {
            "TASK": self.tasks,
            "DURATION": self.durations,
            "DEADLINE": self.deadlines,
            "LOCATION": self.locations,
            "DATE": self.dates,
            "TIME": self.times,
            "PRIORITY": self.priorities,
            "DIFFICULTY": self.difficulties,
            "CATEGORY": self.categories,
            "FIXED_START": self.fixed_starts,
            "RECURRENCE_DAY": self.recurrence_days,
        }


gen = DataGenerator(StubTrainingData())
passed = 0
failed = 0


def c(n, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS | {n}")
    else:
        failed += 1
        print(f"  FAIL | {n} | {detail}")


print("=" * 80)
print("  DATA GENERATOR TESTS")
print("=" * 80)

# ── Category Inference ──────────────────────────────────────────────────────
print("\n[1] CATEGORY INFERENCE")
cats = [
    ("urgent client call", "work"),
    ("pay the rent", "finance"),
    ("file the taxes", "finance"),
    ("meditate every morning", "health"),
    ("go to the gym", "fitness"),
    ("study for the exam", "study"),
    ("grocery shopping", "shopping"),
    ("write blog post", "creative"),
    ("book flight to Paris", "travel"),
    ("call mom", "personal"),
    ("pick up kids from school", "family"),
    ("practice guitar", "creative"),
    ("do the laundry", "home"),
    ("study at the library", "study"),
    ("team meeting", "work"),
    ("yoga every Tuesday", "fitness"),
    ("send the package", "errands"),
    ("update the config", "admin"),
    ("learn Spanish", "learning"),
    ("buy a birthday gift", "shopping"),
]
for s, e in cats:
    c(f"'{s}' -> {e}", gen._infer_category(s) == e, f"got={gen._infer_category(s)}")

# ── Difficulty Inference ────────────────────────────────────────────────────
print("\n[2] DIFFICULTY INFERENCE")
diffs = [
    ("hard workout session", 0.7, 1.0),
    ("easy 15 minute stretch", 0.0, 0.3),
    ("urgent client call", 0.6, 1.0),
    ("critical system crash fix", 0.6, 1.0),
    ("low priority cleanup", 0.0, 0.55),
    ("moderate difficulty task", 0.35, 0.65),
    ("pay the rent", 0.0, 0.6),
    ("file the taxes", 0.5, 1.0),
    ("quick easy task", 0.0, 0.3),
    ("challenging coding problem", 0.65, 1.0),
    ("simple email reply", 0.0, 0.35),
]
for s, lo, hi in diffs:
    v = float(gen._infer_difficulty(s))
    c(f"'{s}' -> {v}", lo <= v <= hi, f"expect {lo}-{hi}")

# ── Importance Inference ────────────────────────────────────────────────────
print("\n[3] IMPORTANCE INFERENCE")
imps = [
    ("urgent client call", 0.7, 1.0),
    ("critical system crash fix", 0.85, 1.0),
    ("low priority cleanup", 0.0, 0.4),
    ("very important presentation", 0.6, 1.0),
    ("pay the rent", 0.7, 1.0),
    ("file the taxes", 0.7, 1.0),
    ("optional task", 0.0, 0.3),
    ("asap fix needed", 0.7, 1.0),
]
for s, lo, hi in imps:
    v = float(gen._infer_importance(s))
    c(f"'{s}' -> {v}", lo <= v <= hi, f"expect {lo}-{hi}")

# ── Duration Inference ──────────────────────────────────────────────────────
print("\n[4] DURATION INFERENCE")
durs = [
    ("quick 5 minute stretch", 5, 5),
    ("15 minute meditation", 15, 15),
    ("2 hour study session", 120, 120),
    ("30 minute meeting", 30, 30),
    ("pay the rent", 5, 30),
    ("team meeting", 15, 90),
    ("go to the doctor", 30, 90),
    ("critical system crash fix", 60, 150),
    ("workout session", 30, 90),
    ("go for a run", 15, 45),
    ("cook dinner", 30, 60),
]
for s, lo, hi in durs:
    v = int(gen._infer_duration(s))
    c(f"'{s}' -> {v}", lo <= v <= hi, f"expect {lo}-{hi}")

# ── Location Inference ──────────────────────────────────────────────────────
print("\n[5] LOCATION INFERENCE")
locs = [
    ("study at the library", "library"),
    ("meet at the coffee shop", "coffee shop"),
    ("work from home", "home"),
    ("go to the gym at the gym", "gym"),
    ("buy groceries at the supermarket", "supermarket"),
    ("team meeting at the office", "office"),
    ("just a regular meeting", None),
]
for s, e in locs:
    c(f"'{s}' -> {e}", gen._infer_location(s) == e, f"got={gen._infer_location(s)}")

# ── Numeric Extraction from Templates ───────────────────────────────────────
# Note: _infer_difficulty uses keyword matching, not number extraction.
# These tests verify that keywords override random values.
print("\n[6] KEYWORD OVERRIDE")
nums = [
    ("hard task", 0.6, 1.0),
    ("easy task", 0.0, 0.4),
]
for s, lo, hi in nums:
    v = float(gen._infer_difficulty(s))
    c(f"'{s}' -> {v}", lo <= v <= hi, f"expect {lo}-{hi}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(
    f"  RESULTS: {passed}/{passed + failed} passed ({100 * passed // (passed + failed) if passed + failed else 0}%)"
)
print(f"{'=' * 80}")
sys.exit(0 if failed == 0 else 1)
