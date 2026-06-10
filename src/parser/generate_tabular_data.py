import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CATEGORIES = ["work", "health", "personal", "errand", "social", "study"]
CATEGORY_WEIGHTS = [0.25, 0.12, 0.22, 0.15, 0.10, 0.16]

LOCATIONS = {
    "work": ["office", "home", "cafe"],
    "health": ["gym", "outdoor", "home"],
    "personal": ["home", "store", "cafe"],
    "errand": ["store", "home", "outdoor"],
    "social": ["cafe", "outdoor", "home"],
    "study": ["home", "cafe", "office"],
}

DURATION_RANGES = {
    "work": (30, 120),
    "health": (20, 60),
    "personal": (15, 90),
    "errand": (10, 45),
    "social": (60, 180),
    "study": (30, 120),
}

TRAVEL_OVERHEAD = {
    "home": 0,
    "office": 15,
    "gym": 20,
    "store": 10,
    "cafe": 10,
    "outdoor": 25,
}

CATEGORY_MULTIPLIERS = {
    "work": 1.08,
    "health": 1.15,
    "personal": 0.88,
    "errand": 0.92,
    "social": 1.02,
    "study": 1.05,
}

CORRELATIONS = {
    "work": {"diff_mean": 0.55, "imp_mean": 0.60},
    "health": {"diff_mean": 0.60, "imp_mean": 0.70},
    "personal": {"diff_mean": 0.35, "imp_mean": 0.40},
    "errand": {"diff_mean": 0.30, "imp_mean": 0.45},
    "social": {"diff_mean": 0.40, "imp_mean": 0.35},
    "study": {"diff_mean": 0.50, "imp_mean": 0.55},
}


def _clamp_gauss(mean, std, lo=0.0, hi=1.0):
    v = random.gauss(mean, std)
    return round(max(lo, min(hi, v)), 3)


def _diff_mod(difficulty):
    if difficulty > 0.6:
        return 1.0 + (difficulty - 0.6) * 1.0
    if difficulty < 0.4:
        return 1.0 - (0.4 - difficulty) * 0.6
    return 1.0


def _imp_mod(importance):
    if importance > 0.7:
        return 1.0 + (importance - 0.7) * 0.7
    if importance < 0.3:
        return 1.0 - (0.3 - importance) * 0.5
    return 1.0


def _deadline_undoable(time_diff, scheduled):
    if time_diff == -1:
        return False
    time_minutes = time_diff * 60
    if time_minutes <= 0:
        return True
    if time_minutes < scheduled * 0.4:
        return True
    return False


def _deadline_rush(time_diff, scheduled):
    if time_diff == -1:
        return False
    time_minutes = time_diff * 60
    if time_minutes < scheduled * 1.0:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="data/VMAI_DURATION_Data.csv")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_path = project_root / args.output

    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    random.seed(seed)

    entries = []

    for _ in range(args.count):
        category = random.choices(CATEGORIES, weights=CATEGORY_WEIGHTS)[0]
        location = random.choice(LOCATIONS[category])

        lo, hi = DURATION_RANGES[category]
        scheduled = random.randint(lo, hi)

        corr = CORRELATIONS[category]
        difficulty = _clamp_gauss(corr["diff_mean"], 0.20)
        importance = _clamp_gauss(corr["imp_mean"], 0.20)

        if random.random() < 0.50:
            current_hour = random.randint(6, 22)
            fixed_decimal = random.randint(max(8, current_hour), 23) + random.choice(
                [0.0, 0.25, 0.30, 0.5, 0.75]
            )
            if fixed_decimal > 23.75:
                fixed_decimal = 23.75
            time_diff = round(fixed_decimal - current_hour, 2)
            fh = int(fixed_decimal)
            fm = int(round((fixed_decimal - fh) * 60))
            fixed_time = f"{fh:02d}:{fm:02d}"
        else:
            fixed_time = ""
            time_diff = -1

        base = scheduled * random.uniform(0.95, 1.05)
        base *= _diff_mod(difficulty)
        base *= _imp_mod(importance)
        base *= CATEGORY_MULTIPLIERS[category]
        base += TRAVEL_OVERHEAD[location]

        if _deadline_undoable(time_diff, scheduled):
            real_duration = 0.0
        else:
            if _deadline_rush(time_diff, scheduled):
                time_minutes = time_diff * 60
                base *= 0.75 if time_minutes < scheduled * 0.6 else 0.88
            noise = random.gauss(0, max(3, scheduled * 0.04))
            raw = max(0, base + noise)
            real_duration = round(min(raw, scheduled * 3.5), 1)
            if real_duration < 1.0:
                real_duration = round(random.uniform(1.0, 3.0), 1)

        entries.append(
            [
                difficulty,
                importance,
                scheduled,
                category,
                location,
                fixed_time,
                time_diff,
                real_duration,
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "difficulty",
                "importance",
                "scheduled_duration",
                "category",
                "location",
                "fixed_time",
                "time_difference",
                "real_duration",
            ]
        )
        w.writerows(entries)

    durations = [e[7] for e in entries]
    undoable = sum(1 for d in durations if d == 0)
    nonzero = [d for d in durations if d > 0]

    print(f"Generated {len(entries)} rows (seed={seed})")
    print(f"Output: {output_path}")
    print(f"  Undoable (real=0): {undoable} ({100 * undoable // len(entries)}%)")
    print(
        f"  Real dur nonzero: min={min(nonzero):.1f} max={max(nonzero):.1f} mean={sum(nonzero) / len(nonzero):.1f}"
    )
    print(
        f"  Deadline: {sum(1 for e in entries if e[6] != -1)} ({100 * sum(1 for e in entries if e[6] != -1) // len(entries)}%)"
    )
    print()
    print("  Samples:")
    for e in random.sample(entries, min(10, len(entries))):
        td = "no" if e[6] == -1 else f"{e[6]:+.1f}h"
        ft = e[5] if e[5] else "  -  "
        print(
            f"    diff={e[0]:.2f} imp={e[1]:.2f} dur={e[2]:3d} cat={e[3]:8s} loc={e[4]:7s} td={td:6s}  real={e[7]:.1f} | ft={ft}"
        )


if __name__ == "__main__":
    main()
