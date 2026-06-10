"""
VM-AI - Core Parser Tests
Tests pipe format parsing, schema conversion, and diff logic.
No model required.
Run: python tests/test_core.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "parser"))
from schemas import changed_to_pipe, pipe_to_schema, schema_to_pipe

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
print("  CORE COMPONENT TESTS")
print("=" * 80)

# ── pipe_to_schema: basic parsing ───────────────────────────────────────────
print("\n[1] PIPE TO SCHEMA — BASIC PARSING")
r = pipe_to_schema(
    "name=gym[EXP] | difficulty=0.55[PRD] | duration=60[PRD] | category=fitness[PRD] | importance=0.41[PRD] | fixed_time=true[EXP] | fixed_start=06:30[EXP] | recurrent=false[EXP]"
)
c("name=gym", r["name"]["value"] == "gym")
c("diff=0.55", r["difficulty"]["value"] == "0.55")
c("dur=60", r["duration"]["value"] == "60")
c("cat=fitness", r["category"]["value"] == "fitness")
c("imp=0.41", r["importance"]["value"] == "0.41")
c("ft=true", r["fixed_time"]["value"] is True)
c("fs=06:30", r["fixed_start"]["value"] == "06:30")
c("rec=false", r["recurrent"]["value"] is False)

# ── pipe_to_schema: EXP/PRD tags ───────────────────────────────────────────
print("\n[2] PIPE TO SCHEMA — EXP/PRD TAGS")
c("name is EXP", r["name"]["predicted"] is False)
c("diff is PRD", r["difficulty"]["predicted"] is True)
c("dur is PRD", r["duration"]["predicted"] is True)
c("cat is PRD", r["category"]["predicted"] is True)
c("loc is PRD", r["location"]["predicted"] is True)
c("imp is PRD", r["importance"]["predicted"] is True)
c("ft is EXP", r["fixed_time"]["predicted"] is False)
c("fs is EXP", r["fixed_start"]["predicted"] is False)
c("rec is EXP", r["recurrent"]["predicted"] is False)
c("rec_days is PRD", r["recurrence_days"]["predicted"] is True)
c("start is PRD", r["start"]["predicted"] is True)
c("deadline is PRD", r["deadline"]["predicted"] is True)

# ── pipe_to_schema: null values ────────────────────────────────────────────
print("\n[3] PIPE TO SCHEMA — NULL VALUES")
r = pipe_to_schema("name=s[EXP] | location=null[PRD] | fixed_start=null[PRD]")
c("null loc", r["location"]["value"] is None)
c("null fs", r["fixed_start"]["value"] is None)

# ── pipe_to_schema: bool edge cases ────────────────────────────────────────
print("\n[4] PIPE TO SCHEMA — BOOL PARSING")
r = pipe_to_schema("fixed_time=true[EXP] | recurrent=true[EXP]")
c("true", r["fixed_time"]["value"] is True)
c("true rec", r["recurrent"]["value"] is True)

r = pipe_to_schema("fixed_time=false[EXP] | recurrent=false[EXP]")
c("false", r["fixed_time"]["value"] is False)
c("false rec", r["recurrent"]["value"] is False)

# ── pipe_to_schema: missing fields get defaults ────────────────────────────
print("\n[5] PIPE TO SCHEMA — MISSING FIELDS")
r = pipe_to_schema("name=x[EXP]")
c("missing diff=None", r["difficulty"]["value"] is None)
c("missing ft=False", r["fixed_time"]["value"] is False)
c("missing name=x", r["name"]["value"] == "x")

# ── schema_to_pipe: roundtrip ──────────────────────────────────────────────
print("\n[6] SCHEMA TO PIPE — ROUNDTRIP")
schema = {
    "name": {"value": "gym session", "predicted": False},
    "start": {"value": None, "predicted": True},
    "deadline": {"value": "tomorrow", "predicted": False},
    "difficulty": {"value": "0.7", "predicted": True},
    "duration": {"value": "60", "predicted": True},
    "category": {"value": "fitness", "predicted": False},
    "location": {"value": "gym", "predicted": True},
    "importance": {"value": "0.9", "predicted": False},
    "fixed_time": {"value": True, "predicted": False},
    "fixed_start": {"value": "09:00", "predicted": False},
    "recurrent": {"value": False, "predicted": False},
    "recurrence_days": {"value": None, "predicted": True},
}
pipe = schema_to_pipe(schema)
parsed = pipe_to_schema(pipe, input_text="gym session")
c("roundtrip name", parsed["name"]["value"] == "gym session")
c("roundtrip deadline", parsed["deadline"]["value"] == "tomorrow")
c("roundtrip difficulty", parsed["difficulty"]["value"] == "0.7")
c("roundtrip category", parsed["category"]["value"] == "fitness")
c("roundtrip fixed_time", parsed["fixed_time"]["value"] is True)
c("roundtrip fixed_start", parsed["fixed_start"]["value"] == "09:00")
c("roundtrip recurrent", parsed["recurrent"]["value"] is False)
c("pipe has tags", "[EXP]" in pipe and "[PRD]" in pipe)

# ── changed_to_pipe ────────────────────────────────────────────────────────
print("\n[7] CHANGED TO PIPE")
changed = {
    "fixed_start": {"value": "07:00", "predicted": False},
    "duration": {"value": "90", "predicted": True},
}
pipe = changed_to_pipe(changed)
r = pipe_to_schema(pipe, input_text="")
c("ch fs", r["fixed_start"]["value"] == "07:00")
c("ch dur", r["duration"]["value"] == "90")
c("ch: has tags", "[EXP]" in pipe and "[PRD]" in pipe)

# ── Diff logic ─────────────────────────────────────────────────────────────
print("\n[8] DIFF SCHEMA — CHANGE DETECTION")
old = {
    "name": {"value": "gym", "predicted": False},
    "difficulty": {"value": "0.35", "predicted": True},
    "fixed_start": {"value": "06:00", "predicted": False},
}
new = {
    "name": {"value": "gym", "predicted": False},
    "difficulty": {"value": "0.8", "predicted": True},
    "fixed_start": {"value": "07:00", "predicted": False},
}
from chat import TaskPlannerPredictor

ch = TaskPlannerPredictor._diff_schemas(old, new)
c("d: diff changed", "difficulty" in ch)
c("d: diff val", ch.get("difficulty", {}).get("value") == "0.8")
c("d: fs changed", "fixed_start" in ch)
c("d: fs val", ch.get("fixed_start", {}).get("value") == "07:00")
c("d: name skipped", "name" not in ch)

# None values skipped
old = {
    "name": {"value": "gym", "predicted": False},
    "difficulty": {"value": "0.35", "predicted": True},
}
new = {
    "name": {"value": "gym", "predicted": False},
    "difficulty": {"value": None, "predicted": True},
}
c("d: none skip", len(TaskPlannerPredictor._diff_schemas(old, new)) == 0)

# Bool changes
old = {"fixed_time": {"value": True, "predicted": False}}
new = {"fixed_time": {"value": False, "predicted": False}}
c("d: bool change", "fixed_time" in TaskPlannerPredictor._diff_schemas(old, new))

# Case insensitive
old = {
    "name": {"value": "Gym", "predicted": False},
    "category": {"value": "FITNESS", "predicted": True},
}
new = {
    "name": {"value": "gym", "predicted": False},
    "category": {"value": "fitness", "predicted": True},
}
c("d: case skip", len(TaskPlannerPredictor._diff_schemas(old, new)) == 0)

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(
    f"  RESULTS: {passed}/{passed + failed} passed ({100 * passed // (passed + failed) if passed + failed else 0}%)"
)
print(f"{'=' * 80}")
sys.exit(0 if failed == 0 else 1)
