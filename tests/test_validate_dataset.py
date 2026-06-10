"""
VM-AI - Dataset Validation Test
Runs validate_dataset.py on all 3 YAML files and checks for errors.
No model required.
Run: python tests/test_validate_dataset.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "parser"))
from validate_dataset import validate_file

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
print("  DATASET VALIDATION TESTS")
print("=" * 80)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
FILES = [
    os.path.join(data_dir, "VMAI_SYNTHETIC_Data.yaml"),
    os.path.join(data_dir, "VMAI_REAL_Data.yaml"),
    os.path.join(data_dir, "VMAI_SPECIFIC_Data.yaml"),
]

for f in FILES:
    fname = os.path.basename(f)
    print(f"\n--- {fname} ---")
    if not os.path.exists(f):
        c(f"{fname} exists", False, "file not found")
        continue
    c(f"{fname} exists", True)
    res = validate_file(f)
    c(f"{fname} has examples", res["total"] > 0, f"total={res['total']}")
    c(f"{fname} no errors", len(res["errors"]) == 0, f"errors={len(res['errors'])}")
    if res["errors"]:
        from collections import Counter

        error_types = Counter(e["type"] for e in res["errors"])
        for etype, count in error_types.items():
            print(f"    {etype}: {count}")
    stats = res.get("stats", {})
    print(f"    add={stats.get('add', 0)}, modify={stats.get('modify', 0)}")

print(f"\n{'=' * 80}")
print(
    f"  RESULTS: {passed}/{passed + failed} passed ({100 * passed // (passed + failed) if passed + failed else 0}%)"
)
print(f"{'=' * 80}")
sys.exit(0 if failed == 0 else 1)
