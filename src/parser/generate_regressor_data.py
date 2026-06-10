"""
VM-AI - Regressor Training Data Generator (v2)
Generates natural, human-sounding training data for diff/imp regression.
Uses sentence frames with task slots, not keyword Mad Libs.

Run from project root: python src/parser/generate_regressor_data.py [--count 600]

Output: data/VMAI_REGR_Data.csv

Written by: Vanea
"""

import argparse
import csv
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

VERB_MAP = {
    "default": ["do", "take care of", "handle", "finish", "get done"],
    "health": ["go to", "do", "have", "get"],
    "code": [
        "fix",
        "write",
        "finish",
        "complete",
        "debug",
        "refactor",
        "review",
        "work on",
        "push",
        "ship",
    ],
    "meeting": ["attend", "have", "go to", "join", "lead", "run", "prepare for"],
    "fitness": ["do", "go for"],
    "study": ["study for", "do", "finish", "prepare for", "review", "go through"],
    "errands": ["do", "take care of", "handle"],
    "home": ["do", "get", "take care of", "finish"],
    "admin": ["do", "fill out", "submit", "file", "complete"],
}

TASK_VERB_CATEGORY = {
    "email": "code",
    "report": "code",
    "code review": "code",
    "deployment": "code",
    "documentation": "code",
    "unit testing": "code",
    "performance review": "code",
    "meeting": "meeting",
    "standup": "meeting",
    "team standup": "meeting",
    "sprint planning": "meeting",
    "retrospective": "meeting",
    "call with client": "meeting",
    "interview": "meeting",
    "networking event": "meeting",
    "presentation": "meeting",
    "slides": "meeting",
    "gym session": "fitness",
    "workout": "fitness",
    "run": "fitness",
    "yoga": "fitness",
    "meditation": "fitness",
    "stretch": "fitness",
    "walk": "fitness",
    "homework": "study",
    "assignment": "study",
    "study session": "study",
    "exam prep": "study",
    "reading": "study",
    "groceries": "errands",
    "shopping": "errands",
    "laundry": "home",
    "cleaning": "home",
    "cooking": "home",
    "dishes": "home",
    "vacuuming": "home",
    "dusting": "home",
    "budget": "admin",
    "invoice": "admin",
    "taxes": "admin",
    "timesheet": "admin",
    "expense report": "admin",
    "doctor appointment": "health",
    "dentist": "health",
    "checkup": "health",
    "medication": "health",
}

TASKS = list(TASK_VERB_CATEGORY.keys())

FRAMES = [
    ("{v} {t} quick and easy", 0.0, 0.1, 0.05, 0.18, set()),
    ("{v} {t} easy no effort", 0.0, 0.08, 0.05, 0.15, set()),
    ("{v} {t} takes no time", 0.0, 0.1, 0.05, 0.18, set()),
    ("{v} {t} simple pointless", 0.02, 0.1, 0.02, 0.12, set()),
    ("{v} {t} quick nothing important", 0.02, 0.1, 0.05, 0.15, set()),
    ("{v} {t} barely anything", 0.0, 0.08, 0.05, 0.15, set()),
    ("{v} {t} quick doesn't matter", 0.02, 0.1, 0.02, 0.12, set()),
    ("{v} {t} easy simple", 0.08, 0.2, 0.08, 0.2, set()),
    ("{v} {t} easy no big deal", 0.08, 0.18, 0.08, 0.18, set()),
    ("{v} {t} straightforward", 0.1, 0.22, 0.12, 0.25, set()),
    ("{v} {t} simple enough", 0.08, 0.18, 0.1, 0.22, set()),
    ("{v} {t} easy nothing to stress", 0.05, 0.15, 0.05, 0.15, set()),
    ("{v} {t} light and simple", 0.05, 0.15, 0.08, 0.2, set()),
    ("{v} {t} quick and simple", 0.05, 0.15, 0.08, 0.2, set()),
    ("easy {t} session", 0.08, 0.18, 0.08, 0.2, set()),
    ("{v} {t} easy but critical", 0.05, 0.15, 0.8, 0.95, set()),
    ("{v} {t} quick but important", 0.05, 0.15, 0.75, 0.9, set()),
    ("{v} {t} easy very important", 0.08, 0.18, 0.8, 0.95, set()),
    ("{v} {t} simple urgent need", 0.08, 0.18, 0.8, 0.95, set()),
    ("{v} {t} quick urgent", 0.08, 0.15, 0.85, 0.95, set()),
    ("{v} {t} easy must do", 0.08, 0.18, 0.8, 0.95, set()),
    ("{v} {t} simple critical", 0.08, 0.18, 0.85, 0.98, set()),
    ("{v} {t} quick can't miss", 0.05, 0.15, 0.85, 0.95, set()),
    ("{v} {t} trivial but high stakes", 0.05, 0.15, 0.8, 0.95, set()),
    ("{v} {t} one click very important", 0.02, 0.1, 0.85, 0.98, set()),
    ("{v} {t} tiny task major impact", 0.05, 0.15, 0.8, 0.95, set()),
    ("{v} {t} thirty seconds can't skip", 0.03, 0.12, 0.85, 0.98, set()),
    ("{v} {t} takes no time top priority", 0.03, 0.12, 0.8, 0.95, set()),
    ("{v} {t} barely any effort extremely important", 0.05, 0.15, 0.85, 0.98, set()),
    ("{v} {t} moderate effort", 0.3, 0.5, 0.35, 0.5, set()),
    ("{v} {t} this week", 0.3, 0.5, 0.35, 0.55, set()),
    ("{v} {t} decent amount of work", 0.35, 0.55, 0.3, 0.5, set()),
    ("{v} {t} moderate nothing special", 0.3, 0.5, 0.25, 0.45, set()),
    ("{v} {t} medium effort", 0.3, 0.5, 0.3, 0.5, set()),
    ("{v} {t} regular task", 0.3, 0.5, 0.3, 0.5, set()),
    ("{v} {t} standard routine", 0.3, 0.5, 0.3, 0.5, set()),
    ("{v} {t} should take some time", 0.35, 0.55, 0.3, 0.5, set()),
    ("{v} {t} important moderate effort", 0.3, 0.5, 0.6, 0.8, set()),
    ("{v} {t} important needs focus", 0.35, 0.55, 0.6, 0.8, set()),
    ("{v} {t} decent effort important", 0.3, 0.5, 0.55, 0.75, set()),
    ("{v} {t} medium priority", 0.3, 0.5, 0.5, 0.7, set()),
    ("{v} {t} worthwhile effort", 0.3, 0.5, 0.5, 0.7, set()),
    ("{v} {t} solid mid level task", 0.35, 0.55, 0.4, 0.6, set()),
    ("{v} {t} fair amount of work", 0.35, 0.55, 0.35, 0.55, set()),
    ("{v} {t} needs reasonable effort", 0.3, 0.5, 0.35, 0.55, set()),
    ("{v} {t} moderately important", 0.25, 0.45, 0.5, 0.7, set()),
    ("{v} {t} somewhat challenging", 0.4, 0.6, 0.35, 0.55, set()),
    ("{v} {t} decent priority", 0.3, 0.5, 0.5, 0.7, set()),
    ("{v} {t} takes some effort", 0.35, 0.55, 0.3, 0.5, set()),
    ("{v} {t} worth the time", 0.3, 0.5, 0.55, 0.75, set()),
    ("{v} {t} should prioritize", 0.3, 0.5, 0.55, 0.75, set()),
    ("{v} {t} requires focus", 0.35, 0.55, 0.5, 0.7, set()),
    ("{v} {t} needs attention", 0.3, 0.5, 0.55, 0.75, set()),
    ("{v} {t} considerable effort", 0.45, 0.6, 0.35, 0.55, set()),
    ("{v} {t} quite demanding", 0.5, 0.7, 0.4, 0.6, set()),
    ("{v} {t} fairly important", 0.25, 0.45, 0.5, 0.7, set()),
    ("{v} {t} non trivial", 0.45, 0.6, 0.3, 0.5, set()),
    ("{v} {t} average task average priority", 0.4, 0.6, 0.4, 0.6, set()),
    ("{v} {t} neither easy nor hard", 0.4, 0.6, 0.35, 0.55, set()),
    ("{v} {t} middle of the road", 0.4, 0.6, 0.4, 0.6, set()),
    ("{v} {t} not easy not important", 0.4, 0.6, 0.3, 0.5, set()),
    ("{v} {t} kinda hard kinda matters", 0.45, 0.6, 0.45, 0.6, set()),
    ("{v} {t} decent work reasonable priority", 0.4, 0.55, 0.4, 0.6, set()),
    ("{v} {t} medium difficulty medium stakes", 0.4, 0.6, 0.4, 0.6, set()),
    ("{v} {t} hard hard", 0.55, 0.75, 0.5, 0.7, set()),
    ("{v} {t} hard needs focus", 0.55, 0.75, 0.55, 0.75, set()),
    ("{v} {t} hard demanding", 0.6, 0.8, 0.55, 0.75, set()),
    ("{v} {t} tough challenging", 0.55, 0.75, 0.5, 0.7, set()),
    ("{v} {t} hard quite important", 0.55, 0.75, 0.6, 0.8, set()),
    ("{v} {t} difficult needs doing", 0.55, 0.75, 0.5, 0.75, set()),
    ("hard {t} session", 0.55, 0.75, 0.5, 0.7, set()),
    ("{v} {t} hard critical", 0.55, 0.75, 0.85, 0.95, set()),
    ("{v} {t} hard very important", 0.55, 0.75, 0.8, 0.95, set()),
    ("{v} {t} difficult urgent", 0.55, 0.75, 0.8, 0.95, set()),
    ("{v} {t} tough urgent must do", 0.6, 0.8, 0.85, 0.95, set()),
    ("{v} {t} hard cannot miss", 0.6, 0.8, 0.85, 0.95, set()),
    ("{v} {t} demanding critical", 0.6, 0.8, 0.85, 0.95, set()),
    ("{v} {t} complex very important", 0.6, 0.8, 0.8, 0.95, set()),
    ("{v} {t} extremely hard life or death", 0.8, 0.95, 0.9, 1.0, set()),
    ("{v} {t} brutal intense", 0.8, 0.95, 0.8, 0.95, set()),
    ("{v} {t} extremely complex critical", 0.75, 0.9, 0.85, 0.95, set()),
    ("{v} {t} grueling most important", 0.85, 0.98, 0.85, 0.95, set()),
    ("{v} {t} hardest thing ever very important", 0.85, 0.98, 0.9, 1.0, set()),
    ("{v} {t} extreme critical", 0.8, 0.95, 0.85, 0.95, set()),
    ("{v} {t} hard nobody cares", 0.6, 0.8, 0.05, 0.18, set()),
    ("{v} {t} difficult pointless", 0.6, 0.8, 0.05, 0.18, set()),
    ("{v} {t} hard doesn't matter", 0.55, 0.75, 0.05, 0.18, set()),
    ("{v} {t} complex nobody asked for", 0.55, 0.75, 0.05, 0.15, set()),
    ("{v} {t} hard waste of time", 0.6, 0.8, 0.05, 0.15, set()),
    ("{v} {t} tough no benefit", 0.55, 0.75, 0.05, 0.15, set()),
    ("{v} {t} hard zero payoff", 0.6, 0.8, 0.05, 0.15, set()),
    ("{v} {t} draining but irrelevant", 0.65, 0.8, 0.05, 0.15, set()),
    ("{v} {t} takes forever but skip it", 0.65, 0.85, 0.05, 0.15, set()),
    ("{v} {t} exhausting low stakes", 0.6, 0.78, 0.08, 0.2, set()),
    ("{v} {t} technically hard completely optional", 0.6, 0.8, 0.08, 0.2, set()),
    ("{v} {t} massive effort basically useless", 0.7, 0.88, 0.05, 0.15, set()),
    ("{v} {t} moderate doesn't matter", 0.3, 0.5, 0.1, 0.28, set()),
    ("{v} {t} no rush", 0.25, 0.5, 0.15, 0.32, set()),
    ("{v} {t} not urgent", 0.25, 0.5, 0.15, 0.32, set()),
    ("{v} {t} low priority", 0.25, 0.5, 0.15, 0.3, set()),
    ("{v} {t} whenever", 0.25, 0.5, 0.15, 0.3, set()),
    ("{v} {t} someday task", 0.2, 0.45, 0.18, 0.35, set()),
    ("{v} {t} background task", 0.2, 0.45, 0.2, 0.38, set()),
    ("{v} {t} can wait", 0.2, 0.45, 0.2, 0.38, set()),
    ("{v} {t} nice to have", 0.2, 0.4, 0.2, 0.38, set()),
    ("{v} {t} if time allows", 0.25, 0.5, 0.2, 0.35, set()),
    ("{v} {t} easy should do", 0.08, 0.2, 0.45, 0.65, set()),
    ("{v} {t} simple worth doing", 0.08, 0.2, 0.45, 0.65, set()),
    ("{v} {t} quick but matters", 0.08, 0.18, 0.5, 0.7, set()),
    ("{v} {t} easy moderate important", 0.08, 0.2, 0.5, 0.7, set()),
    ("{v} {t} not hard", 0.08, 0.25, 0.2, 0.4, set()),
    ("{v} {t} not difficult", 0.08, 0.25, 0.2, 0.4, set()),
    ("{v} {t} not important", 0.2, 0.4, 0.1, 0.28, set()),
    ("{v} {t} not urgent", 0.25, 0.5, 0.15, 0.32, set()),
    ("{v} {t} not a priority", 0.2, 0.4, 0.15, 0.28, set()),
    ("{v} {t} not critical should do", 0.25, 0.5, 0.3, 0.5, set()),
    ("{v} {t} not hard but important", 0.1, 0.25, 0.6, 0.8, set()),
    ("{v} {t} not important but hard", 0.6, 0.8, 0.15, 0.3, set()),
    ("{v} {t} minor task", 0.15, 0.35, 0.2, 0.38, set()),
    ("{v} {t} small but worth doing", 0.1, 0.3, 0.2, 0.4, set()),
    ("{v} {t} easy low importance", 0.08, 0.2, 0.2, 0.38, set()),
    ("{v} {t} routine low stakes", 0.2, 0.4, 0.2, 0.38, set()),
    ("{v} {t} quick low value", 0.08, 0.2, 0.2, 0.35, set()),
    ("easy {t}", 0.05, 0.2, 0.1, 0.25, set()),
    ("hard {t}", 0.55, 0.8, 0.4, 0.7, set()),
    ("quick {t}", 0.05, 0.15, 0.15, 0.32, set()),
    ("simple {t}", 0.08, 0.2, 0.15, 0.32, set()),
    ("tough {t}", 0.55, 0.75, 0.4, 0.6, set()),
    ("urgent {t}", 0.3, 0.5, 0.8, 0.95, set()),
    ("important {t}", 0.3, 0.5, 0.7, 0.9, set()),
    ("critical {t}", 0.3, 0.5, 0.85, 0.98, set()),
    ("moderate {t}", 0.3, 0.5, 0.3, 0.5, set()),
    ("intense {t}", 0.7, 0.85, 0.6, 0.8, set()),
    ("extreme {t}", 0.8, 0.95, 0.7, 0.9, set()),
    ("pointless {t}", 0.2, 0.4, 0.08, 0.2, set()),
    ("optional {t}", 0.2, 0.4, 0.15, 0.32, set()),
    ("mandatory {t}", 0.3, 0.5, 0.85, 0.98, set()),
    ("{v} {t} tomorrow", 0.2, 0.4, 0.2, 0.4, set()),
    ("{v} {t} next week", 0.2, 0.4, 0.15, 0.35, set()),
    ("{v} {t} on Monday", 0.2, 0.4, 0.2, 0.4, set()),
    ("{v} {t} at 6am", 0.2, 0.4, 0.2, 0.38, set()),
    ("{v} {t} this evening", 0.2, 0.4, 0.15, 0.35, set()),
    ("{v} {t} tonight", 0.2, 0.4, 0.2, 0.4, set()),
    ("{v} {t} by noon", 0.25, 0.45, 0.25, 0.45, set()),
    ("{v} {t} sometime this week", 0.2, 0.4, 0.15, 0.3, set()),
    ("{v} {t} in the morning", 0.2, 0.4, 0.2, 0.38, set()),
    ("{v} {t} before lunch", 0.2, 0.4, 0.2, 0.4, set()),
]

EDGE_CASES = [
    ("absolutely impossible", 0.99, 0.95),
    ("literally takes 5 seconds", 0.01, 0.15),
    ("world ends if I don't do this", 0.30, 1.0),
    ("beyond trivial", 0.02, 0.08),
    ("one click and it's done", 0.01, 0.18),
    ("life or death situation", 0.70, 1.0),
    ("single most important thing in my life", 0.60, 0.99),
    ("could not possibly care less", 0.10, 0.08),
    ("hardest thing i've ever done and it means everything", 0.95, 0.98),
    ("pointless busywork that somehow takes forever", 0.75, 0.08),
    ("takes two seconds but absolutely critical", 0.05, 0.95),
    ("super easy and nobody cares", 0.10, 0.08),
    ("extremely complex problem nobody will ever look at", 0.85, 0.12),
    ("flip the switch to deploy the fix", 0.05, 0.90),
    ("just need to press enter to save the company", 0.02, 0.98),
    ("the entire company depends on this one thing", 0.40, 0.95),
    ("soul crushing grind for something that doesn't matter", 0.90, 0.08),
    ("quick confirmation text", 0.02, 0.60),
    ("five minute fix saves the whole project", 0.15, 0.90),
    ("building an entire dashboard nobody asked for", 0.80, 0.12),
    ("this one email determines if we get the deal", 0.10, 0.95),
    ("barely registers as a task", 0.01, 0.08),
    ("i automated a process that ran fine manually", 0.70, 0.12),
    ("debugging a race condition that never happens", 0.75, 0.12),
    ("this is the easiest thing i will do all day", 0.02, 0.18),
]


def get_verb(task_name):
    cat = TASK_VERB_CATEGORY.get(task_name, "default")
    return random.choice(VERB_MAP[cat])


def main():
    parser = argparse.ArgumentParser(
        description="Generate natural regressor training data"
    )
    parser.add_argument(
        "--count", type=int, default=600, help="Number of entries to generate"
    )
    parser.add_argument(
        "--output",
        default="data/VMAI_REGR_Data.csv",
        help="Output CSV path (relative to project root)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (random if omitted)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_path = project_root / args.output

    print(f"Generating {args.count} entries...")
    print(f"Output: {output_path}")
    print()

    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    print(f"Seed: {seed}")
    random.seed(seed)
    used = set()
    entries = []
    attempts = 0

    quadrant_frames = []
    buckets = [
        (0.0, 0.45, 0.0, 0.45, "lo/lo"),
        (0.0, 0.45, 0.55, 1.0, "lo/hi"),
        (0.55, 1.0, 0.0, 0.45, "hi/lo"),
        (0.55, 1.0, 0.55, 1.0, "hi/hi"),
    ]

    def frame_quadrant(dl, dh, il, ih):
        cx = (dl + dh) / 2
        cy = (il + ih) / 2
        best = 0
        best_dist = float("inf")
        for qi, (ql, qh, ilq, ihq, _) in enumerate(buckets):
            qx = (ql + qh) / 2
            qy = (ilq + ihq) / 2
            dist = (cx - qx) ** 2 + (cy - qy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = qi
        return best

    frame_quads = [frame_quadrant(*f[1:5]) for f in FRAMES]

    quad_frames = [[] for _ in range(4)]
    for i, q in enumerate(frame_quads):
        quad_frames[q].append(i)

    target_per_quad = args.count // 4
    quad_counts = [0, 0, 0, 0]
    cycle = 0

    while len(entries) < args.count and attempts < args.count * 8:
        attempts += 1

        if len(entries) < args.count * 0.8:
            qi = cycle % 4
            cycle += 1
        else:
            qi = min(range(4), key=lambda i: quad_counts[i])

        if quad_counts[qi] >= target_per_quad + 10:
            qi = random.randint(0, 3)

        frame_pool = quad_frames[qi]
        if not frame_pool:
            continue

        idx = random.choice(frame_pool)
        template, dl, dh, il, ih, flags = FRAMES[idx]
        diff = round(random.uniform(dl, dh), 3)
        imp = round(random.uniform(il, ih), 3)

        task = random.choice(TASKS)
        verb = get_verb(task)
        sentence = template.format(v=verb, t=task)
        sentence = re.sub(r"\s+", " ", sentence).strip().rstrip(",")
        sentence = sentence.replace("'", "'")

        key = sentence.lower()
        if key in used:
            continue
        if len(sentence.split()) < 3:
            continue

        s_lower = sentence.lower()
        urgent = {"urgent", "asap", "critical", "can't wait", "emergency"}
        low_imp = {
            "doesn't matter",
            "pointless",
            "no big deal",
            "skip",
            "ignore",
            "not important",
        }
        if any(w in s_lower for w in urgent) and any(w in s_lower for w in low_imp):
            continue

        used.add(key)
        entries.append((sentence, diff, imp))
        quad_counts[qi] += 1

    for text, diff, imp in EDGE_CASES:
        key = text.lower()
        if key not in used:
            used.add(key)
            entries.append((text, diff, imp))
            qi = min(3, max(0, int(diff > 0.5) * 2 + int(imp > 0.5)))
            quad_counts[qi] += 1

    random.shuffle(entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "difficulty", "importance"])
        writer.writerows(entries)

    diffs = [e[1] for e in entries]
    imps = [e[2] for e in entries]
    n = len(entries)

    print(
        f"Generated {n} unique entries ({attempts} attempts, {attempts - n} rejected)"
    )
    print(
        f"  Difficulty: min={min(diffs):.2f} max={max(diffs):.2f} "
        f"mean={sum(diffs) / n:.3f}"
    )
    print(
        f"  Importance: min={min(imps):.2f} max={max(imps):.2f} "
        f"mean={sum(imps) / n:.3f}"
    )
    print()

    qnames = ["lo/lo", "lo/hi", "hi/lo", "hi/hi"]
    print("  Quadrant breakdown:")
    for qi, name in enumerate(qnames):
        cnt = sum(
            1
            for d, i in zip(diffs, imps)
            if (qi == 0 and d < 0.5 and i < 0.5)
            or (qi == 1 and d < 0.5 and i >= 0.5)
            or (qi == 2 and d >= 0.5 and i < 0.5)
            or (qi == 3 and d >= 0.5 and i >= 0.5)
        )
        print(f"    {name}: {cnt} ({100 * cnt / n:.0f}%)")

    print()
    print("  Samples:")
    samples = random.sample(entries, min(15, n))
    for text, diff, imp in sorted(samples, key=lambda x: x[2], reverse=True):
        print(f"    {diff:.2f}/{imp:.2f}  {text}")


if __name__ == "__main__":
    main()
