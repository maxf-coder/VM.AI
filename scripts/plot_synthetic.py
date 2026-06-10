"""
VM.AI - Synthetic Dataset Plot Generator
Visualizes template statistics, field coverage, and value distributions.
Run: python scripts/plot_synthetic.py
"""

import os
import re
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


def save(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png")


def load_data():
    path = os.path.join(ROOT, "data", "VMAI_SYNTHETIC_Data.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_placeholders(template):
    return re.findall(r"\[([A-Z_]+)\]", template)


def plot_overview(data, out_dir):
    templates = data.get("templates", [])
    tasks = data.get("tasks", [])
    deadlines = data.get("deadlines", [])
    durations = data.get("durations", [])
    dates = data.get("dates", [])
    times = data.get("times", [])
    locations = data.get("locations", [])
    categories = data.get("categories", [])
    priorities = data.get("priorities", [])
    difficulties = data.get("difficulties", [])
    recurrence = data.get("recurrence_days", [])

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle(
        "VM.AI Synthetic Dataset Overview",
        color="#f0f6fc",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    items = [
        ("templates", len(templates)),
        ("tasks", len(tasks)),
        ("deadlines", len(deadlines)),
        ("durations", len(durations)),
        ("dates", len(dates)),
        ("times", len(times)),
        ("locations", len(locations)),
        ("categories", len(categories)),
        ("priorities", len(priorities)),
        ("difficulties", len(difficulties)),
        ("recurrence_days", len(recurrence)),
    ]
    labels, values = zip(*items)
    ax1.barh(labels, values, color="#bc8cff", edgecolor="#30363d")
    ax1.set_title("Value Pool Sizes", color="#f0f6fc", fontweight="bold")
    ax1.set_xlabel("Count")
    ax1.grid(True, alpha=0.15, axis="x")
    for i, v in enumerate(values):
        ax1.text(v + max(values) * 0.01, i, str(v), color="#c9d1d9", va="center")

    ax2 = fig.add_subplot(gs[0, 1])
    all_ph = [ph for t in templates for ph in extract_placeholders(t)]
    ph_counts = Counter(all_ph)
    ph_labels, ph_values = zip(*ph_counts.most_common())
    ax2.barh(ph_labels, ph_values, color="#58a6ff", edgecolor="#30363d")
    ax2.set_title(
        "Placeholder Usage Across Templates", color="#f0f6fc", fontweight="bold"
    )
    ax2.set_xlabel("Count")
    ax2.grid(True, alpha=0.15, axis="x")
    for i, v in enumerate(ph_values):
        ax2.text(v + max(ph_values) * 0.01, i, str(v), color="#c9d1d9", va="center")

    ax3 = fig.add_subplot(gs[0, 2])
    cats = [c for c in categories]
    ax3.pie(
        [1] * len(cats),
        labels=cats,
        startangle=90,
        textprops={"color": "#c9d1d9", "fontsize": 9},
        colors=plt.cm.tab10(range(len(cats))),
    )
    ax3.set_title("Available Categories", color="#f0f6fc", fontweight="bold")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    stats_text = f"TEMPLATES: {len(templates)}\nTASKS: {len(tasks)}\nDEADLINES: {len(deadlines)}\nDURATIONS: {len(durations)}\nTIMES: {len(times)}\nLOCATIONS: {len(locations)}\nCATEGORIES: {len(categories)}\nPRIORITIES: {len(priorities)}\nDIFFICULTIES: {len(difficulties)}"
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
    field_counts_per_template = [len(extract_placeholders(t)) for t in templates]
    ax5.hist(
        field_counts_per_template,
        bins=range(1, max(field_counts_per_template) + 2),
        color="#f78166",
        edgecolor="#30363d",
        alpha=0.8,
    )
    ax5.set_title("Fields Per Template", color="#f0f6fc", fontweight="bold")
    ax5.set_xlabel("Number of Placeholders")
    ax5.set_ylabel("Count")
    ax5.grid(True, alpha=0.2, axis="y")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ph_text = "\n".join(f"  [{ph}]: {c}" for ph, c in ph_counts.most_common())
    ax6.text(
        0.05,
        0.5,
        f"PLACEHOLDER COVERAGE\n{ph_text}",
        fontsize=11,
        family="monospace",
        color="#c9d1d9",
        va="center",
    )
    ax6.set_title("Placeholder Breakdown", color="#f0f6fc", fontweight="bold", y=1.1)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, out_dir, "overview")


def plot_template_complexity(data, out_dir):
    templates = data.get("templates", [])
    field_counts = [len(extract_placeholders(t)) for t in templates]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        field_counts,
        bins=range(1, max(field_counts) + 2),
        color="#bc8cff",
        edgecolor="#30363d",
        alpha=0.8,
    )
    ax.set_title(
        "Template Complexity (Fields Per Template)", color="#f0f6fc", fontweight="bold"
    )
    ax.set_xlabel("Number of Placeholders")
    ax.set_ylabel("Template Count")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save(fig, out_dir, "template_complexity")


def plot_placeholder_usage(data, out_dir):
    templates = data.get("templates", [])
    all_ph = [ph for t in templates for ph in extract_placeholders(t)]
    ph_counts = Counter(all_ph)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, values = zip(*ph_counts.most_common())
    ax.barh(labels, values, color="#58a6ff", edgecolor="#30363d")
    ax.set_title(
        "Placeholder Frequency Across Templates", color="#f0f6fc", fontweight="bold"
    )
    ax.set_xlabel("Usage Count")
    ax.grid(True, alpha=0.15, axis="x")
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), color="#c9d1d9", va="center")
    plt.tight_layout()
    save(fig, out_dir, "placeholder_usage")


def plot_categories(data, out_dir):
    categories = data.get("categories", [])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(categories, [1] * len(categories), color="#3fb950", edgecolor="#30363d")
    ax.set_title("Available Categories", color="#f0f6fc", fontweight="bold")
    ax.set_xlabel("Present")
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.15, axis="x")
    plt.tight_layout()
    save(fig, out_dir, "categories")


def plot_difficulty_distribution(data, out_dir):
    difficulties = [float(d) for d in data.get("difficulties", [])]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(difficulties, bins=15, color="#58a6ff", edgecolor="#30363d", alpha=0.8)
    ax.set_title("Difficulty Value Distribution", color="#f0f6fc", fontweight="bold")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save(fig, out_dir, "difficulty")


def plot_importance_distribution(data, out_dir):
    priorities = [float(p) for p in data.get("priorities", [])]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(priorities, bins=15, color="#bc8cff", edgecolor="#30363d", alpha=0.8)
    ax.set_title(
        "Priority/Importance Value Distribution", color="#f0f6fc", fontweight="bold"
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save(fig, out_dir, "importance")


def plot_duration_distribution(data, out_dir):
    durations = data.get("durations", [])
    mins = []
    for d in durations:
        d = str(d).lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minute|min)", d)
        if m:
            mins.append(float(m.group(1)))
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hr)", d)
            if m:
                mins.append(float(m.group(1)) * 60)
            elif "half" in d and "day" in d:
                mins.append(720)
            elif "all day" in d:
                mins.append(960)
    if not mins:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mins, bins=15, color="#f78166", edgecolor="#30363d", alpha=0.8)
    ax.set_title(
        "Duration Value Distribution (minutes)", color="#f0f6fc", fontweight="bold"
    )
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    save(fig, out_dir, "duration")


def plot_value_pools(data, out_dir):
    pools = {
        "tasks": len(data.get("tasks", [])),
        "deadlines": len(data.get("deadlines", [])),
        "durations": len(data.get("durations", [])),
        "dates": len(data.get("dates", [])),
        "times": len(data.get("times", [])),
        "locations": len(data.get("locations", [])),
        "categories": len(data.get("categories", [])),
        "priorities": len(data.get("priorities", [])),
        "difficulties": len(data.get("difficulties", [])),
        "recurrence_days": len(data.get("recurrence_days", [])),
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, values = zip(*sorted(pools.items(), key=lambda x: x[1]))
    colors = plt.cm.tab10(range(len(labels)))
    ax.barh(labels, values, color=colors, edgecolor="#30363d")
    ax.set_title("Value Pool Sizes", color="#f0f6fc", fontweight="bold")
    ax.set_xlabel("Count")
    ax.grid(True, alpha=0.15, axis="x")
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), color="#c9d1d9", va="center")
    plt.tight_layout()
    save(fig, out_dir, "value_pools")


def main():
    out_dir = os.path.join(ROOT, "scripts", "output", "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating plots for synthetic dataset...")
    print(f"Output: {out_dir}/\n")
    setup_style()
    data = load_data()
    plot_overview(data, out_dir)
    plot_template_complexity(data, out_dir)
    plot_placeholder_usage(data, out_dir)
    plot_categories(data, out_dir)
    plot_difficulty_distribution(data, out_dir)
    plot_importance_distribution(data, out_dir)
    plot_duration_distribution(data, out_dir)
    plot_value_pools(data, out_dir)
    print(f"\nDone. All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
