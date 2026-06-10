"""
VM.AI - Dataset Plot Generator
Generates all plots for a given dataset into a dedicated output folder.
Run: python scripts/plot_dataset.py --dataset real
     python scripts/plot_dataset.py --dataset specific
"""

import argparse
import os
import sys
from collections import Counter

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
except ImportError:
    print("Install: pip install matplotlib")
    sys.exit(1)


DATASET_CONFIG = {
    "real": {
        "path": os.path.join(ROOT, "data", "VMAI_REAL_Data.yaml"),
        "folder": "real",
        "accent": "#58a6ff",
        "title_suffix": "Real Data",
    },
    "specific": {
        "path": os.path.join(ROOT, "data", "VMAI_SPECIFIC_Data.yaml"),
        "folder": "specific",
        "accent": "#3fb950",
        "title_suffix": "Specific Data",
    },
    "synthetic": {
        "path": os.path.join(ROOT, "data", "VMAI_SYNTHETIC_Data.yaml"),
        "folder": "synthetic",
        "accent": "#bc8cff",
        "title_suffix": "Synthetic Data",
    },
}


def setup_style():
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#0d1117",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "text.color": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#21262d",
        }
    )


def load_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("examples", [])


def classify(ex):
    output = ex.get("output", {})
    is_modify = "name" not in output or len(output) <= 3
    return {
        "type": "modify" if is_modify else "add",
        "category": str(output.get("category", "unknown")).lower(),
        "difficulty": float(output.get("difficulty", 0))
        if output.get("difficulty") is not None
        else None,
        "importance": float(output.get("importance", 0))
        if output.get("importance") is not None
        else None,
        "duration": float(output.get("duration", 0))
        if output.get("duration") is not None
        else None,
        "fixed_time": output.get("fixed_time", False),
        "fixed_start": output.get("fixed_start", None),
        "recurrent": output.get("recurrent", False),
        "deadline": output.get("deadline", None),
        "start": output.get("start", None),
    }


def save(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png")


def plot_overview(records, out_dir, cfg):
    adds = [r for r in records if r["type"] == "add"]
    modifies = [r for r in records if r["type"] == "modify"]
    cat_counts = Counter(r["category"] for r in records)
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle(
        f"VM.AI Dataset Overview — {cfg['title_suffix']}",
        color="#f0f6fc",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    labels, sizes = zip(*cat_counts.most_common())
    colors = plt.cm.tab10(range(len(labels)))
    ax1.pie(
        sizes,
        labels=labels,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
        textprops={"color": "#c9d1d9", "fontsize": 9},
    )
    ax1.set_title("Category Distribution", color="#f0f6fc", fontweight="bold")
    ax2 = fig.add_subplot(gs[0, 1])
    if adds:
        diffs = [r["difficulty"] for r in adds if r["difficulty"] is not None]
        imps = [r["importance"] for r in adds if r["importance"] is not None]
        cats = [r["category"] for r in adds if r["difficulty"] is not None]
        for cat in sorted(set(cats)):
            mask = [c == cat for c in cats]
            ax2.scatter(
                [d for d, m in zip(diffs, mask) if m],
                [i for i, m in zip(imps, mask) if m],
                label=cat,
                s=60,
                alpha=0.7,
                edgecolors="#30363d",
            )
    ax2.set_title("Difficulty vs Importance", color="#f0f6fc", fontweight="bold")
    ax2.set_xlabel("Difficulty")
    ax2.set_ylabel("Importance")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, framealpha=0.3)
    ax2.grid(True, alpha=0.2)
    ax3 = fig.add_subplot(gs[0, 2])
    durs = [
        r["duration"]
        for r in records
        if r["duration"] is not None and r["duration"] > 0
    ]
    if durs:
        ax3.hist(durs, bins=20, color=cfg["accent"], edgecolor="#30363d", alpha=0.8)
    ax3.set_title("Duration Distribution", color="#f0f6fc", fontweight="bold")
    ax3.set_xlabel("Minutes")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.2, axis="y")
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    stats_text = f"TOTAL: {len(records)}\nADD: {len(adds)}\nMODIFY: {len(modifies)}\nRECURRENCE: {sum(1 for r in records if r['recurrent'])}\nFIXED TIME: {sum(1 for r in records if r['fixed_time'])}\nCATEGORIES: {len(cat_counts)}"
    ax4.text(
        0.05,
        0.5,
        stats_text,
        fontsize=12,
        family="monospace",
        color="#c9d1d9",
        va="center",
    )
    ax4.set_title("Statistics", color="#f0f6fc", fontweight="bold", y=1.1)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(
        ["Add", "Modify"],
        [len(adds), len(modifies)],
        color=[cfg["accent"], "#bc8cff"],
        edgecolor="#30363d",
    )
    ax5.set_title("Add vs Modify", color="#f0f6fc", fontweight="bold")
    ax5.set_ylabel("Count")
    mx = max(len(adds), len(modifies), 1)
    ax5.set_ylim(0, mx * 1.2)
    for i, v in enumerate([len(adds), len(modifies)]):
        ax5.text(
            i, v + mx * 0.05, str(v), color="#c9d1d9", ha="center", fontweight="bold"
        )
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    deadlines = Counter(r["deadline"] for r in records if r["deadline"])
    dl_text = (
        "\n".join(f"  {k}: {v}" for k, v in deadlines.most_common(8))
        if deadlines
        else "  (none)"
    )
    ax6.text(
        0.05,
        0.5,
        f"DEADLINES\n{dl_text}",
        fontsize=11,
        family="monospace",
        color="#c9d1d9",
        va="center",
    )
    ax6.set_title("Deadline Breakdown", color="#f0f6fc", fontweight="bold", y=1.1)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, out_dir, "overview")


def plot_categories(records, out_dir, cfg):
    counts = Counter(r["category"] for r in records)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, values = zip(*counts.most_common())
    ax.barh(labels, values, color=cfg["accent"], edgecolor="#30363d")
    ax.set_title(
        f"Category Distribution — {cfg['title_suffix']}",
        color="#f0f6fc",
        fontweight="bold",
    )
    ax.set_xlabel("Count")
    ax.grid(True, alpha=0.15, axis="x")
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), color="#c9d1d9", va="center")
    plt.tight_layout()
    save(fig, out_dir, "categories")


def plot_scatter(records, out_dir, cfg):
    adds = [
        r
        for r in records
        if r["type"] == "add"
        and r["difficulty"] is not None
        and r["importance"] is not None
    ]
    if not adds:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    diffs = [r["difficulty"] for r in adds]
    imps = [r["importance"] for r in adds]
    cats = [r["category"] for r in adds]
    for cat in sorted(set(cats)):
        mask = [c == cat for c in cats]
        ax.scatter(
            [d for d, m in zip(diffs, mask) if m],
            [i for i, m in zip(imps, mask) if m],
            label=cat,
            s=60,
            alpha=0.7,
            edgecolors="#30363d",
        )
    ax.set_title(
        f"Difficulty vs Importance — {cfg['title_suffix']}",
        color="#f0f6fc",
        fontweight="bold",
    )
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Importance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save(fig, out_dir, "scatter")


def plot_difficulty(records, out_dir, cfg):
    vals = [r["difficulty"] for r in records if r["difficulty"] is not None]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=20, color=cfg["accent"], edgecolor="#30363d", alpha=0.8)
    ax.set_title(
        f"Difficulty Distribution — {cfg['title_suffix']}",
        color="#f0f6fc",
        fontweight="bold",
    )
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save(fig, out_dir, "difficulty")


def plot_importance(records, out_dir, cfg):
    vals = [r["importance"] for r in records if r["importance"] is not None]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=20, color="#bc8cff", edgecolor="#30363d", alpha=0.8)
    ax.set_title(
        f"Importance Distribution — {cfg['title_suffix']}",
        color="#f0f6fc",
        fontweight="bold",
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save(fig, out_dir, "importance")


def plot_duration(records, out_dir, cfg):
    vals = [
        r["duration"]
        for r in records
        if r["duration"] is not None and r["duration"] > 0
    ]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=20, color="#f78166", edgecolor="#30363d", alpha=0.8)
    ax.set_title(
        f"Duration Distribution (min) — {cfg['title_suffix']}",
        color="#f0f6fc",
        fontweight="bold",
    )
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save(fig, out_dir, "duration")


def plot_features(records, out_dir, cfg):
    fixed_time = sum(1 for r in records if r["fixed_time"])
    recurrent = sum(1 for r in records if r["recurrent"])
    has_deadline = sum(1 for r in records if r["deadline"])
    has_start = sum(1 for r in records if r["start"])
    has_location = sum(1 for r in records if r.get("fixed_start"))
    labels = ["Fixed Time", "Recurrent", "Deadline", "Start", "Location"]
    values = [fixed_time, recurrent, has_deadline, has_start, has_location]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        labels,
        values,
        color=[cfg["accent"], "#bc8cff", "#f78166", "#58a6ff", "#3fb950"],
        edgecolor="#30363d",
    )
    ax.set_title(
        f"Feature Coverage — {cfg['title_suffix']}", color="#f0f6fc", fontweight="bold"
    )
    ax.set_xlabel("Count")
    ax.grid(True, alpha=0.15, axis="x")
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), color="#c9d1d9", va="center")
    plt.tight_layout()
    save(fig, out_dir, "features")


def main():
    parser = argparse.ArgumentParser(description="VM.AI Dataset Plot Generator")
    parser.add_argument(
        "--dataset",
        choices=["real", "specific", "synthetic"],
        required=True,
        help="Dataset to plot",
    )
    args = parser.parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    path = cfg["path"]
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        sys.exit(1)
    out_dir = os.path.join(ROOT, "scripts", "output", cfg["folder"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating plots for {args.dataset} dataset...")
    print(f"Output: {out_dir}/\n")
    setup_style()
    examples = load_examples(path)
    if not examples:
        print(
            f"No examples found in {path} (synthetic data uses templates, not examples)"
        )
        sys.exit(0)
    records = [classify(ex) for ex in examples]
    plot_overview(records, out_dir, cfg)
    plot_categories(records, out_dir, cfg)
    plot_scatter(records, out_dir, cfg)
    plot_difficulty(records, out_dir, cfg)
    plot_importance(records, out_dir, cfg)
    plot_duration(records, out_dir, cfg)
    plot_features(records, out_dir, cfg)
    print(f"\nDone. All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
