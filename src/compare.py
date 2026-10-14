# --------------------------------------------------------------------------------------------------------------------------------------
# First- vs Second-Order Comparison
# --------------------------------------------------------------------------------------------------------------------------------------
# Reads output/first_order/summary.json and output/second_order/summary.json (written by main.py) and puts the
# key numbers side by side. Both orders were produced by the same code, so any difference comes from the model.
#
# Usage: python src/compare.py   (run automatically by: python src/main.py --order both)


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "output")
COMPARISON_DIR = os.path.join(OUTPUT_ROOT, "comparison")

METHOD_TITLES = {"fixed": "Fixed length", "fitted": "Fitted length", "terminating": "Terminating state"}

# (key, description, which is better: "lower" / "higher" / None)
METHOD_METRICS = [
    ("mae", "MAE observed vs predicted matrix", "lower"),
    ("hellinger_lengths", "Hellinger distance: care period lengths", "lower"),
    ("hellinger_visits", "Hellinger distance: category contacts (average)", "lower"),
    ("levenshtein_mean", "Levenshtein distance (mean)", "lower"),
    ("levenshtein_norm_mean", "Levenshtein distance, normalised (mean)", "lower"),
    ("nw_mean", "Needleman-Wunsch score (mean)", "higher"),
    ("nw_norm_mean", "Needleman-Wunsch score, normalised (mean)", "higher"),
    ("sw_mean", "Smith-Waterman score (mean)", "higher"),
    ("sw_norm_mean", "Smith-Waterman score, normalised (mean)", "higher"),
    ("avg_chain_length", "Average simulated chain length", None),
    ("fallback_steps", "Steps from unseen histories (random fallback)", "lower"),
]

MODEL_METRICS = [
    ("train_test_mae_main", "Train vs test MAE (main categories)", "lower"),
    ("train_test_mae_all", "Train vs test MAE (all categories)", "lower"),
    ("train_test_chi2_p_main", "Train vs test chi-squared p-value (main)", "higher"),
    ("sensitivity_mae_5pct_main", "Perturbation MAE at 5% noise (main)", "lower"),
    ("sensitivity_mae_5pct_all", "Perturbation MAE at 5% noise (all)", "lower"),
    ("bootstrap_avg_std_main", "Bootstrap average std (main)", "lower"),
    ("bootstrap_avg_std_all", "Bootstrap average std (all)", "lower"),
    ("capped_terminating_chains", "Terminating chains that hit the maximum length", "lower"),
]


def load_summary(order_name):
    path = os.path.join(OUTPUT_ROOT, f"{order_name}_order", "summary.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def better(first, second, direction):
    if direction is None or first == second:
        return "-"
    if direction == "lower":
        return "First" if first < second else "Second"
    return "First" if first > second else "Second"


def format_number(value):
    if isinstance(value, int):
        return f"{value}"
    return f"{value:.4f}"


def main():
    first = load_summary("first")
    second = load_summary("second")
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    rows = []
    for kind, title in METHOD_TITLES.items():
        for key, description, direction in METHOD_METRICS:
            a = first["methods"][kind][key]
            b = second["methods"][kind][key]
            rows.append([title, description, a, b, better(a, b, direction)])
    for key, description, direction in MODEL_METRICS:
        rows.append(["Model", description, first[key], second[key], better(first[key], second[key], direction)])

    # ----------------------------------
    # Text table
    # ----------------------------------
    lines = []
    lines.append("=" * 118)
    lines.append("FIRST- vs SECOND-ORDER MARKOV CHAIN COMPARISON")
    lines.append("=" * 118)
    lines.append(f"Random seed: first-order = {first['seed']}, second-order = {second['seed']}")
    lines.append(f"Matrices: first-order {first['main_label']} / {first['all_label']}, "
                 f"second-order {second['main_label']} / {second['all_label']}")
    lines.append("")
    lines.append("Data coverage (histories = the states the next state is conditioned on):")
    for summary, name in ((first, "First-order"), (second, "Second-order")):
        lines.append(f"  {name}: main {summary['observed_histories_main']}/{summary['possible_histories_main']} "
                     f"histories observed, {summary['missing_main']}/{summary['possible_main']} transitions missing; "
                     f"all {summary['observed_histories_all']}/{summary['possible_histories_all']} histories observed, "
                     f"{summary['missing_all']}/{summary['possible_all']} transitions missing")
    lines.append("")
    lines.append(f"{'Method':<18} | {'Metric':<50} | {'First-order':>12} | {'Second-order':>12} | Better")
    lines.append("-" * 118)
    previous_group = None
    for group, description, a, b, winner in rows:
        if previous_group is not None and group != previous_group:
            lines.append("-" * 118)
        previous_group = group
        lines.append(f"{group:<18} | {description:<50} | {format_number(a):>12} | {format_number(b):>12} | {winner}")
    lines.append("")
    lines.append("Convergence:")
    lines.append(f"  First-order  main: {first['convergence_main']}")
    lines.append(f"  First-order  all:  {first['convergence_all']}")
    lines.append(f"  Second-order main: {second['convergence_main']}")
    lines.append(f"  Second-order all:  {second['convergence_all']}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - MAE is only comparable in direction, not size: the second-order matrices have more rows, and")
    lines.append("    each row is estimated from fewer transitions.")
    lines.append("  - 'Better' is a simple comparison of the two numbers, not a significance test.")

    text = "\n".join(lines)
    with open(os.path.join(COMPARISON_DIR, "comparison_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

    # ----------------------------------
    # CSV (for tables in the paper)
    # ----------------------------------
    with open(os.path.join(COMPARISON_DIR, "comparison_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "first_order", "second_order", "better"])
        writer.writerows(rows)

    # ----------------------------------
    # Figure: normalised sequence-similarity and distribution metrics, by method
    # ----------------------------------
    plot_metrics = [
        ("mae", "MAE (lower is better)"),
        ("hellinger_lengths", "Hellinger: lengths (lower is better)"),
        ("hellinger_visits", "Hellinger: contacts (lower is better)"),
        ("levenshtein_norm_mean", "Normalised Levenshtein (lower is better)"),
        ("nw_norm_mean", "Normalised Needleman-Wunsch (higher is better)"),
        ("sw_norm_mean", "Normalised Smith-Waterman (higher is better)"),
    ]
    kinds = list(METHOD_TITLES.keys())
    x = np.arange(len(kinds))
    width = 0.38

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, title) in zip(axes.flatten(), plot_metrics):
        first_values = [first["methods"][k][key] for k in kinds]
        second_values = [second["methods"][k][key] for k in kinds]
        ax.bar(x - width / 2, first_values, width, label="First-order", color="steelblue")
        ax.bar(x + width / 2, second_values, width, label="Second-order", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_TITLES[k] for k in kinds], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0, 0].legend()
    plt.suptitle("First- vs Second-Order Markov Chains", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "comparison_metrics.png"), dpi=300)
    plt.close(fig)

    print(f"\nComparison saved to {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
