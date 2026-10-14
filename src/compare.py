# --------------------------------------------------------------------------------------------------------------------------------------
# First- vs Second-Order Comparison
# --------------------------------------------------------------------------------------------------------------------------------------
# Reads output/<data>/first_order/summary.json and output/<data>/second_order/summary.json (written by main.py) and puts the
# key numbers side by side. Both orders were produced by the same code, so any difference comes from the model.
#
# Usage: python src/compare.py --data combined   (run automatically by: python src/main.py --order both)


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "output")

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


def load_cv_episodes(dataset, order_name):
    path = os.path.join(OUTPUT_ROOT, dataset, f"{order_name}_order", "cross_validation_episodes.csv")
    return pd.read_csv(path)


def lower_is_better(key):
    return key.startswith("lev_norm")


def paired_comparison(first_episodes, second_episodes, keys):
    """
    Second-order minus first-order score for each held-out episode, with mean, 95% CI and Wilcoxon signed-rank test.
    Both orders were scored on the same held-out episodes and the same transitions, so the comparison is paired.
    """
    merged = first_episodes.merge(second_episodes, on="ActivityID", suffixes=("_first", "_second"))
    results = {}
    for key in keys:
        diff = (merged[f"{key}_second"] - merged[f"{key}_first"]).dropna().values
        n = len(diff)
        mean = diff.mean()
        half_width = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
        p_value = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0

        # Which order is better, only if the confidence interval excludes zero
        ci_low, ci_high = mean - half_width, mean + half_width
        if ci_low > 0 or ci_high < 0:
            second_better = (mean < 0) if lower_is_better(key) else (mean > 0)
            verdict = "Second" if second_better else "First"
        else:
            verdict = "No clear difference"
        results[key] = {"mean_diff": mean, "ci_low": ci_low, "ci_high": ci_high, "wilcoxon_p": p_value, "n": n,
                        "better": verdict}
    return results


def load_summary(dataset, order_name):
    path = os.path.join(OUTPUT_ROOT, dataset, f"{order_name}_order", "summary.json")
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


def main(dataset="combined"):
    first = load_summary(dataset, "first")
    second = load_summary(dataset, "second")
    comparison_dir = os.path.join(OUTPUT_ROOT, dataset, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # ----------------------------------
    # Leave-one-out cross-validation: the validation on unseen episodes
    # ----------------------------------
    cv_keys = list(first["cv_descriptions"].keys())
    paired = paired_comparison(load_cv_episodes(dataset, "first"), load_cv_episodes(dataset, "second"), cv_keys)

    cv_lines = []
    cv_lines.append("VALIDATION ON UNSEEN EPISODES: LEAVE-ONE-OUT CROSS-VALIDATION")
    cv_lines.append("Each episode held out in turn, model built from all the others. Mean [95% CI] over held-out episodes.")
    cv_lines.append("Difference = second-order minus first-order on the same episode (paired). 'Better' is only given when")
    cv_lines.append("the 95% CI of the difference excludes zero. Log-likelihood, accuracy, Needleman-Wunsch and")
    cv_lines.append("Smith-Waterman: higher is better. Levenshtein: lower is better.")
    cv_lines.append("")
    cv_lines.append(f"{'Score':<62} | {'First-order':>26} | {'Second-order':>26} | "
                    f"{'Difference (2nd - 1st)':>28} | {'Wilcoxon p':>10} | Better")
    cv_lines.append("-" * 182)
    cv_rows = []
    for key in cv_keys:
        a, b, d = first["cv"][key], second["cv"][key], paired[key]
        a_text = f"{a['mean']:.4f} [{a['ci_low']:.4f}, {a['ci_high']:.4f}]"
        b_text = f"{b['mean']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}]"
        d_text = f"{d['mean_diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
        p_text = "< 0.001" if d["wilcoxon_p"] < 0.001 else f"{d['wilcoxon_p']:.3f}"
        cv_lines.append(f"{first['cv_descriptions'][key]:<62} | {a_text:>26} | {b_text:>26} | {d_text:>28} | "
                        f"{p_text:>10} | {d['better']}")
        cv_rows.append([first["cv_descriptions"][key], a["mean"], a["ci_low"], a["ci_high"], b["mean"], b["ci_low"],
                        b["ci_high"], d["mean_diff"], d["ci_low"], d["ci_high"], d["wilcoxon_p"], d["n"], d["better"]])
    cv_lines.append("")
    cv_lines.append(f"Perplexity (main / all categories): first-order {first['cv_perplexity']['main']:.3f} / "
                    f"{first['cv_perplexity']['all']:.3f}, second-order {second['cv_perplexity']['main']:.3f} / "
                    f"{second['cv_perplexity']['all']:.3f} (lower is better)")
    cv_lines.append("")

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
    lines.append(f"Dataset: {dataset} ({first['n_train']} training episodes)")
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
    lines += cv_lines
    lines.append("=" * 118)
    lines.append("IN-SAMPLE CHECKS: simulated chains vs the training data they were generated from")
    lines.append("(these show whether each model reproduces its training data; they are not a validation)")
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
    lines.append("  - In the in-sample table, 'Better' is a simple comparison of the two numbers, not a significance test.")

    text = "\n".join(lines)
    with open(os.path.join(comparison_dir, "comparison_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

    # ----------------------------------
    # CSV (for tables in the paper)
    # ----------------------------------
    with open(os.path.join(comparison_dir, "comparison_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "first_order", "second_order", "better"])
        writer.writerows(rows)

    with open(os.path.join(comparison_dir, "comparison_cross_validation.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "first_mean", "first_ci_low", "first_ci_high", "second_mean", "second_ci_low",
                         "second_ci_high", "difference_mean", "difference_ci_low", "difference_ci_high",
                         "wilcoxon_p", "n_episodes", "better"])
        writer.writerows(cv_rows)

    # ----------------------------------
    # Figure: cross-validation differences (positive = second-order better)
    # ----------------------------------
    # The scores are on different scales, so each group gets its own panel and x-axis
    panels = [
        ("Predicting the next contact", [k for k in cv_keys if k.startswith(("loglik", "accuracy"))]),
        ("Sequence similarity: Levenshtein and Smith-Waterman",
         [k for k in cv_keys if k.startswith(("lev_norm", "sw_norm"))]),
        ("Sequence similarity: Needleman-Wunsch", [k for k in cv_keys if k.startswith("nw_norm")]),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 0.45 * len(cv_keys) + 4),
                             gridspec_kw={"height_ratios": [len(keys) for _, keys in panels]})

    for ax, (panel_title, keys) in zip(axes, panels):
        y = np.arange(len(keys))[::-1]
        for yi, key in zip(y, keys):
            d = paired[key]
            sign = -1 if lower_is_better(key) else 1
            mean = sign * d["mean_diff"]
            low, high = sorted((sign * d["ci_low"], sign * d["ci_high"]))
            colour = "green" if low > 0 else "red" if high < 0 else "grey"
            ax.errorbar(mean, yi, xerr=[[mean - low], [high - mean]], fmt="o", color=colour, capsize=4)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([first["cv_descriptions"][k].replace(" (lower is better)", "") for k in keys], fontsize=9)
        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)

    axes[-1].set_xlabel("Improvement of second-order over first-order (mean and 95% CI over held-out episodes)")
    plt.suptitle(f"Leave-One-Out Cross-Validation: Second- vs First-Order ({dataset})\n"
                 "green = second-order better, red = first-order better, grey = no clear difference", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "comparison_cross_validation.png"), dpi=300)
    plt.close(fig)

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
    plt.savefig(os.path.join(comparison_dir, "comparison_metrics.png"), dpi=300)
    plt.close(fig)

    print(f"\nComparison saved to {comparison_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="combined")
    main(parser.parse_args().data)
