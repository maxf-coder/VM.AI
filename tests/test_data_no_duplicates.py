"""
VM-AI - Duplicate Input Checker
Verifies no duplicate input texts exist in REAL and SPECIFIC datasets.
No model required.
Run: python tests/test_data_no_duplicates.py
"""

import os
import sys

import yaml

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
print("  DUPLICATE INPUT CHECKER")
print("=" * 80)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
FILES = [
    os.path.join(data_dir, "VMAI_REAL_Data.yaml"),
    os.path.join(data_dir, "VMAI_SPECIFIC_Data.yaml"),
]

for f in FILES:
    fname = os.path.basename(f)
    print(f"\n--- {fname} ---")
    if not os.path.exists(f):
        c(f"{fname} exists", False, "file not found")
        continue
    with open(f, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    examples = data.get("examples", [])
    c(f"{fname} loaded", len(examples) > 0, f"examples={len(examples)}")
    seen = {}
    dups = []
    for i, ex in enumerate(examples):
        inp = ex.get("input", "")
        if inp in seen:
            dups.append((i, seen[inp], inp[:60]))
        else:
            seen[inp] = i
    c(f"{fname} no duplicates", len(dups) == 0, f"duplicates={len(dups)}")
    if dups:
        for idx, orig, snippet in dups[:5]:
            print(f"    dup idx={idx} of idx={orig}: '{snippet}'")

print(f"\n{'=' * 80}")
print(
    f"  RESULTS: {passed}/{passed + failed} passed ({100 * passed // (passed + failed) if passed + failed else 0}%)"
)
print(f"{'=' * 80}")
sys.exit(0 if failed == 0 else 1)
