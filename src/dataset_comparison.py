# --------------------------------------------------------------------------------------------------------------------------------------
# Dataset Comparison: Mock vs Actual Care, and Single-Bed vs Multi-Bed Rooms
# --------------------------------------------------------------------------------------------------------------------------------------
# Before mock and actual episodes (or single- and multi-bed rooms) are pooled into one training set, this checks how
# similar they are. Three comparisons are made:
#   1. Mock vs Actual (single-bed): like-for-like, as all mock episodes took place in single-bed rooms
#   2. Mock vs Actual (all rooms)
#   3. Actual single-bed vs Actual multi-bed
#
# For each comparison:
#   - Episode length (mean, median, Mann-Whitney U test, Hellinger distance)
#   - Share of contacts in each category (total variation distance, permutation test)
#   - Contacts per episode in each category (Mann-Whitney U test, Hellinger distance)
#   - First-order transition matrices (MAE between the two groups' matrices, permutation test)
#   - How episodes start and end (Fisher's exact test)
#   - Sequence similarity (normalised Levenshtein distance within and between the groups)
#
# Permutation tests shuffle whole episodes between the two groups, so contacts from the same episode stay together.
#
# Usage: python src/dataset_comparison.py   -> output/data_comparison/


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Levenshtein import ratio
from scipy import stats

from config import OUTPUT_ROOT, SEED, all_categories, main_categories
from data_loader import load_data
from distances import hellinger_distance_histograms


OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "data_comparison")
N_PERMUTATIONS = 2000

COMPARISONS = {
    "mock_vs_actual_sb": {
        "title": "Mock vs Actual (single-bed rooms)",
        "a": ("Mock", {"Source": ["Mock"], "RoomType": ["SB"]}),
        "b": ("Actual SB", {"Source": ["Actual"], "RoomType": ["SB"]}),
    },
    "mock_vs_actual_all": {
        "title": "Mock vs Actual (all rooms)",
        "a": ("Mock", {"Source": ["Mock"], "RoomType": ["SB", "MB"]}),
        "b": ("Actual", {"Source": ["Actual"], "RoomType": ["SB", "MB"]}),
    },
    "actual_sb_vs_mb": {
        "title": "Actual care: single-bed vs multi-bed rooms",
        "a": ("Actual SB", {"Source": ["Actual"], "RoomType": ["SB"]}),
        "b": ("Actual MB", {"Source": ["Actual"], "RoomType": ["MB"]}),
    },
}

category_index = {category: i for i, category in enumerate(all_categories)}
main_index = [category_index[category] for category in main_categories]
letters = {category: category[0] for category in all_categories}  # In = I, Equipment = E, ... (all unique)


# --------------------------------------------------------------------------------------------------------------------------------------
# Per-Episode Summaries
# --------------------------------------------------------------------------------------------------------------------------------------

def summarise_episodes(data):
    """
    One entry per care episode: its length, contacts per category, first-order transition counts, how it starts and
    ends, and the sequence as a string of letters.
    """
    episodes = []
    for activity_id, episode in data.groupby("ActivityID", sort=False):
        states = episode["SurfaceCategories"].tolist()
        codes = [category_index[state] for state in states]

        transitions = np.zeros((len(all_categories), len(all_categories)))
        for current_state, next_state in zip(codes[:-1], codes[1:]):
            transitions[current_state, next_state] += 1

        episodes.append({
            "ActivityID": activity_id,
            "Source": episode["Source"].iloc[0],
            "RoomType": episode["RoomType"].iloc[0],
            "length": len(states),
            "category_counts": np.bincount(codes, minlength=len(all_categories)),
            "transitions": transitions,
            "starts_in": states[0] == "In",
            "ends_out": states[-1] == "Out",
            # Left the room and came back during the episode (an "Out" that is not the last contact)
            "re_entry": "Out" in states[:-1],
            "string": "".join(letters[state] for state in states),
        })
    return pd.DataFrame(episodes)


def select(episodes, condition):
    keep = np.ones(len(episodes), dtype=bool)
    for column, values in condition.items():
        keep &= episodes[column].isin(values).values
    return episodes[keep].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------------------------
# Statistics Comparing Two Groups of Episodes
# --------------------------------------------------------------------------------------------------------------------------------------

def row_normalise(counts):
    sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, sums, out=np.zeros_like(counts), where=sums > 0)


def matrix_mae(counts_a, counts_b, index=None):
    """
    MAE between two groups' transition probability matrices.
    Only rows (current states) observed in both groups are compared.
    """
    if index is not None:
        counts_a = counts_a[np.ix_(index, index)]
        counts_b = counts_b[np.ix_(index, index)]
    probs_a, probs_b = row_normalise(counts_a), row_normalise(counts_b)
    rows = (counts_a.sum(axis=1) > 0) & (counts_b.sum(axis=1) > 0)
    return np.abs(probs_a[rows] - probs_b[rows]).mean()


def total_variation(counts_a, counts_b):
    """Total variation distance between two sets of category proportions (0 = identical, 1 = no overlap)."""
    return 0.5 * np.abs(counts_a / counts_a.sum() - counts_b / counts_b.sum()).sum()


def permutation_test(stacked_a, stacked_b, statistic, rng):
    """
    How often a statistic at least as large as the observed one appears when whole episodes are shuffled between the
    two groups at random. A small p-value means the difference is unlikely to be chance.
    """
    observed = statistic(stacked_a.sum(axis=0), stacked_b.sum(axis=0))
    pooled = np.concatenate([stacked_a, stacked_b])
    n_a = len(stacked_a)
    total = pooled.sum(axis=0)

    count = 0
    for _ in range(N_PERMUTATIONS):
        in_a = np.zeros(len(pooled), dtype=bool)
        in_a[rng.choice(len(pooled), size=n_a, replace=False)] = True
        sum_a = pooled[in_a].sum(axis=0)
        if statistic(sum_a, total - sum_a) >= observed:
            count += 1

    return observed, (count + 1) / (N_PERMUTATIONS + 1)


def mean_normalised_levenshtein(strings_a, strings_b=None):
    """Mean normalised Levenshtein distance between all pairs (within one group if strings_b is None)."""
    if strings_b is None:
        distances = [1 - ratio(strings_a[i], strings_a[j])
                     for i in range(len(strings_a)) for j in range(i + 1, len(strings_a))]
    else:
        distances = [1 - ratio(x, y) for x in strings_a for y in strings_b]
    return np.mean(distances)


def compare_groups(name_a, episodes_a, name_b, episodes_b, rng):
    """All statistics for one comparison."""
    result = {"name_a": name_a, "name_b": name_b, "n_a": len(episodes_a), "n_b": len(episodes_b)}

    # Episode length
    lengths_a, lengths_b = episodes_a["length"].values, episodes_b["length"].values
    result["length"] = {
        "mean_a": lengths_a.mean(), "mean_b": lengths_b.mean(),
        "median_a": np.median(lengths_a), "median_b": np.median(lengths_b),
        "sd_a": lengths_a.std(ddof=1), "sd_b": lengths_b.std(ddof=1),
        "mannwhitney_p": stats.mannwhitneyu(lengths_a, lengths_b).pvalue,
        "hellinger": hellinger_distance_histograms(lengths_a, lengths_b),
    }

    # Share of contacts in each category
    counts_a = np.stack(episodes_a["category_counts"].values)
    counts_b = np.stack(episodes_b["category_counts"].values)
    tvd, tvd_p = permutation_test(counts_a, counts_b, total_variation, rng)
    result["category_share"] = {
        "share_a": counts_a.sum(axis=0) / counts_a.sum(),
        "share_b": counts_b.sum(axis=0) / counts_b.sum(),
        "tvd": tvd, "permutation_p": tvd_p,
    }

    # Contacts per episode in each category
    result["contacts_per_episode"] = {}
    for category, i in category_index.items():
        result["contacts_per_episode"][category] = {
            "mean_a": counts_a[:, i].mean(), "mean_b": counts_b[:, i].mean(),
            "mannwhitney_p": stats.mannwhitneyu(counts_a[:, i], counts_b[:, i]).pvalue,
            "hellinger": hellinger_distance_histograms(counts_a[:, i], counts_b[:, i]),
        }

    # First-order transition matrices
    transitions_a = np.stack(episodes_a["transitions"].values)
    transitions_b = np.stack(episodes_b["transitions"].values)
    mae_all, mae_all_p = permutation_test(transitions_a, transitions_b, matrix_mae, rng)
    mae_main, mae_main_p = permutation_test(
        transitions_a, transitions_b, lambda a, b: matrix_mae(a, b, main_index), rng)
    result["transitions"] = {
        "mae_all": mae_all, "mae_all_p": mae_all_p,
        "mae_main": mae_main, "mae_main_p": mae_main_p,
        "matrix_a": row_normalise(transitions_a.sum(axis=0)),
        "matrix_b": row_normalise(transitions_b.sum(axis=0)),
    }

    # How episodes start and end
    result["start_end"] = {}
    for column in ("starts_in", "ends_out", "re_entry"):
        yes_a, yes_b = episodes_a[column].sum(), episodes_b[column].sum()
        table = [[yes_a, len(episodes_a) - yes_a], [yes_b, len(episodes_b) - yes_b]]
        result["start_end"][column] = {
            "share_a": yes_a / len(episodes_a), "share_b": yes_b / len(episodes_b),
            "fisher_p": stats.fisher_exact(table).pvalue,
        }

    # Sequence similarity: are episodes from different groups less alike than episodes from the same group?
    strings_a, strings_b = episodes_a["string"].tolist(), episodes_b["string"].tolist()
    result["levenshtein"] = {
        "within_a": mean_normalised_levenshtein(strings_a),
        "within_b": mean_normalised_levenshtein(strings_b),
        "between": mean_normalised_levenshtein(strings_a, strings_b),
    }

    return result


# --------------------------------------------------------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------------------------------------------------------

def plot_comparison(key, title, result, episodes_a, episodes_b):
    folder = os.path.join(OUTPUT_DIR, key)
    os.makedirs(folder, exist_ok=True)
    name_a, name_b = result["name_a"], result["name_b"]

    # Episode lengths
    lengths_a, lengths_b = episodes_a["length"].values, episodes_b["length"].values
    bins = np.arange(min(lengths_a.min(), lengths_b.min()), max(lengths_a.max(), lengths_b.max()) + 5, 4)
    plt.figure(figsize=(12, 7))
    sns.histplot(lengths_a, bins=bins, alpha=0.4, color="blue", label=f"{name_a} (n = {len(lengths_a)})",
                 stat="probability")
    sns.histplot(lengths_b, bins=bins, alpha=0.4, color="orange", label=f"{name_b} (n = {len(lengths_b)})",
                 stat="probability")
    plt.xlabel("Care Period Length (contacts)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.title(f"Care Period Length: {title}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.text(0.95, 0.80, f"Hellinger Distance: {result['length']['hellinger']:.3f}\n"
                         f"Mann-Whitney p = {result['length']['mannwhitney_p']:.4f}",
             transform=plt.gca().transAxes, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "care_period_lengths.png"), dpi=300)
    plt.close()

    # Share of contacts in each category
    x = np.arange(len(all_categories))
    width = 0.38
    plt.figure(figsize=(12, 7))
    plt.bar(x - width / 2, result["category_share"]["share_a"], width, label=name_a, color="steelblue")
    plt.bar(x + width / 2, result["category_share"]["share_b"], width, label=name_b, color="coral")
    plt.xticks(x, all_categories)
    plt.ylabel("Share of all contacts", fontsize=14)
    plt.title(f"Contacts by Category: {title}\n(total variation distance = {result['category_share']['tvd']:.3f}, "
              f"permutation p = {result['category_share']['permutation_p']:.4f})", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "category_shares.png"), dpi=300)
    plt.close()

    # Transition matrices side by side, and their difference
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    matrices = [(result["transitions"]["matrix_a"], name_a, "Blues", 0, 1),
                (result["transitions"]["matrix_b"], name_b, "Blues", 0, 1),
                (result["transitions"]["matrix_a"] - result["transitions"]["matrix_b"],
                 f"{name_a} minus {name_b}", "RdBu_r", -0.5, 0.5)]
    for ax, (matrix, label, cmap, vmin, vmax) in zip(axes, matrices):
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
                    xticklabels=all_categories, yticklabels=all_categories, cbar=False)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Next")
        ax.set_ylabel("Current")
    plt.suptitle(f"First-Order Transition Probabilities: {title}\n(MAE = {result['transitions']['mae_all']:.3f}, "
                 f"permutation p = {result['transitions']['mae_all_p']:.4f})", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "transition_matrices.png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------------------------

def p_text(p):
    if np.isnan(p):
        return "n/a (identical)"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def report(key, title, result):
    a, b = result["name_a"], result["name_b"]
    lines = ["=" * 100, title.upper(), "=" * 100,
             f"{a}: {result['n_a']} episodes    {b}: {result['n_b']} episodes", ""]

    r = result["length"]
    lines += ["--- Episode length (contacts) ---",
              f"{'':<14}{a:>14}{b:>14}",
              f"{'Mean':<14}{r['mean_a']:>14.1f}{r['mean_b']:>14.1f}",
              f"{'Median':<14}{r['median_a']:>14.1f}{r['median_b']:>14.1f}",
              f"{'Std':<14}{r['sd_a']:>14.1f}{r['sd_b']:>14.1f}",
              f"Mann-Whitney U test: {p_text(r['mannwhitney_p'])}",
              f"Hellinger distance between length distributions: {r['hellinger']:.3f}", ""]

    r = result["category_share"]
    lines += ["--- Share of all contacts in each category ---",
              f"{'Category':<14}{a:>14}{b:>14}"]
    for i, category in enumerate(all_categories):
        lines.append(f"{category:<14}{r['share_a'][i]:>14.3f}{r['share_b'][i]:>14.3f}")
    lines += [f"Total variation distance: {r['tvd']:.3f} (permutation test {p_text(r['permutation_p'])})", ""]

    lines += ["--- Contacts per episode in each category ---",
              f"{'Category':<14}{'Mean ' + a:>16}{'Mean ' + b:>16}{'Mann-Whitney':>16}{'Hellinger':>12}"]
    for category, r in result["contacts_per_episode"].items():
        lines.append(f"{category:<14}{r['mean_a']:>16.2f}{r['mean_b']:>16.2f}"
                     f"{p_text(r['mannwhitney_p']):>16}{r['hellinger']:>12.3f}")
    lines.append("")

    r = result["transitions"]
    lines += ["--- First-order transition matrices ---",
              f"MAE between the two groups' 7x7 matrices: {r['mae_all']:.4f} (permutation test {p_text(r['mae_all_p'])})",
              f"MAE between the two groups' 5x5 matrices: {r['mae_main']:.4f} (permutation test {p_text(r['mae_main_p'])})",
              f"{a}:",
              pd.DataFrame(r["matrix_a"], index=all_categories, columns=all_categories).round(3).to_string(),
              f"{b}:",
              pd.DataFrame(r["matrix_b"], index=all_categories, columns=all_categories).round(3).to_string(), ""]

    lines += ["--- How episodes start and end ---",
              f"{'':<34}{a:>14}{b:>14}{'Fisher exact':>16}"]
    labels = {"starts_in": "Starts with In", "ends_out": "Ends with Out",
              "re_entry": "Leaves and re-enters (Out mid-way)"}
    for column, r in result["start_end"].items():
        lines.append(f"{labels[column]:<34}{r['share_a']:>14.1%}{r['share_b']:>14.1%}{p_text(r['fisher_p']):>16}")
    lines.append("")

    r = result["levenshtein"]
    lines += ["--- Sequence similarity (mean normalised Levenshtein distance, 0 = identical) ---",
              f"Within {a}: {r['within_a']:.3f}",
              f"Within {b}: {r['within_b']:.3f}",
              f"Between {a} and {b}: {r['between']:.3f}",
              "If 'between' is close to the 'within' values, episodes from the two groups are about as alike as",
              "episodes from the same group.", ""]

    return "\n".join(lines)


def summary_row(key, title, result):
    return {
        "comparison": title,
        "n_a": result["n_a"], "n_b": result["n_b"],
        "mean_length_a": result["length"]["mean_a"], "mean_length_b": result["length"]["mean_b"],
        "length_mannwhitney_p": result["length"]["mannwhitney_p"],
        "length_hellinger": result["length"]["hellinger"],
        "category_tvd": result["category_share"]["tvd"],
        "category_tvd_p": result["category_share"]["permutation_p"],
        "transition_mae_7x7": result["transitions"]["mae_all"],
        "transition_mae_7x7_p": result["transitions"]["mae_all_p"],
        "transition_mae_5x5": result["transitions"]["mae_main"],
        "transition_mae_5x5_p": result["transitions"]["mae_main_p"],
        "levenshtein_within_a": result["levenshtein"]["within_a"],
        "levenshtein_within_b": result["levenshtein"]["within_b"],
        "levenshtein_between": result["levenshtein"]["between"],
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    episodes = summarise_episodes(load_data())

    sections = [
        "DATASET COMPARISON",
        f"Permutation tests: {N_PERMUTATIONS} random reallocations of whole episodes between the two groups (seed {SEED}).",
        "p-values are not corrected for multiple testing; read them alongside the size of each difference.",
        "",
    ]
    rows = []
    for key, comparison in COMPARISONS.items():
        name_a, condition_a = comparison["a"]
        name_b, condition_b = comparison["b"]
        episodes_a, episodes_b = select(episodes, condition_a), select(episodes, condition_b)

        result = compare_groups(name_a, episodes_a, name_b, episodes_b, rng)
        sections.append(report(key, comparison["title"], result))
        rows.append(summary_row(key, comparison["title"], result))
        plot_comparison(key, comparison["title"], result, episodes_a, episodes_b)

    text = "\n".join(sections)
    with open(os.path.join(OUTPUT_DIR, "data_comparison_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "data_comparison_summary.csv"), index=False)
    print(text)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
