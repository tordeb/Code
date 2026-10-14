# --------------------------------------------------------------------------------------------------------------------------------------
# Plotting of Findings
# --------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # Save figures to file without needing a display (e.g. inside Docker)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from config import ORDER_NAME, OUTPUT_DIR

from transition_matrices import convergence_main, convergence_all, target_main, target_all, wilson_ci, assess_stability
from metrics import hellinger_results, levenshtein_results, nw_results, sw_results


ORDER_TITLE = f"{ORDER_NAME.capitalize()}-Order"


# --------------------------------------------------------------------------------------------------------------------------------------
# Probability against Care Period Lengths
# --------------------------------------------------------------------------------------------------------------------------------------

# Loop through each method and create plots
for method_name, res in hellinger_results.items():
    real_lengths = res['real_lengths']
    simulated_lengths = res['simulated_lengths']

    # Calculate bin edges with width of 1
    bin_min = int(min(min(real_lengths), min(simulated_lengths)))
    bin_max = int(max(max(real_lengths), max(simulated_lengths))) + 1
    bins_length = np.arange(bin_min, bin_max + 1, 1)

    # Plot
    plt.figure(figsize=(12, 8))
    sns.histplot(real_lengths, bins=bins_length, alpha=0.4, color="blue", label="Observed",
                 stat="probability", kde=False)
    sns.histplot(simulated_lengths, bins=bins_length, alpha=0.4, color="orange", label="Predicted",
                 stat="probability", kde=False)

    # Labels
    plt.xlabel("Care Period Length", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.title(f"Probability Against Care Period Length ({ORDER_TITLE}, {method_name})", fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)

    # Add Hellinger Distance to plot
    plt.text(0.95, 0.95, f"Hellinger Distance: {res['lengths']:.4f}",
             transform=plt.gca().transAxes, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/care_period_length_distribution_{method_name}.png", dpi=300)
    plt.close()


# --------------------------------------------------------------------------------------------------------------------------------------
# Probability against Number of Contacts to Each Category
# --------------------------------------------------------------------------------------------------------------------------------------

# Loop through each method
for method_name, res in hellinger_results.items():
    all_real_visits = res['real_visits']
    all_sim_visits = res['sim_visits']
    categories_to_use = list(all_real_visits.keys())

    # Find global min/max for x-axis and y-axis
    global_min_x = min(min(min(v) for v in all_real_visits.values()), min(min(v) for v in all_sim_visits.values()))
    global_max_x = max(max(max(v) for v in all_real_visits.values()), max(max(v) for v in all_sim_visits.values()))
    bins_global = np.arange(global_min_x, global_max_x + 2, 1)

    # Calculate max probability for y-axis
    global_max_y = 0
    for category in categories_to_use:
        hist_real, _ = np.histogram(all_real_visits[category], bins=bins_global)
        hist_sim, _ = np.histogram(all_sim_visits[category], bins=bins_global)
        max_prob_real = hist_real.max() / len(all_real_visits[category])
        max_prob_sim = hist_sim.max() / len(all_sim_visits[category])
        global_max_y = max(global_max_y, max_prob_real, max_prob_sim)

    # Create the plots
    fig = plt.figure(figsize=(15, 10))

    # Adjust grid based on number of categories
    if len(categories_to_use) <= 5:
        gs = gridspec.GridSpec(2, 6, figure=fig)
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 1:3]),
            fig.add_subplot(gs[1, 3:5])
        ]
    else:
        gs = gridspec.GridSpec(3, 6, figure=fig)
        axes = [
            fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6]),
            fig.add_subplot(gs[2, 2:4])
        ]

    for ax, category in zip(axes, categories_to_use):
        # Plot
        sns.histplot(all_real_visits[category], bins=bins_global, alpha=0.4, color="blue", label="Observed",
                     stat="probability", kde=False, ax=ax)
        sns.histplot(all_sim_visits[category], bins=bins_global, alpha=0.4, color="orange", label="Predicted",
                     stat="probability", kde=False, ax=ax)

        # Set uniform axes for all subplots
        ax.set_xlim(global_min_x - 0.5, global_max_x + 1.5)
        ax.set_ylim(0, global_max_y * 1.1)  # Add 10% padding

        ax.set_xlabel("Number of Contacts")
        ax.set_ylabel("Probability")
        ax.set_title(f"{category}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add Hellinger Distance
        ax.text(0.95, 0.95, f"H = {res['visits'][category]:.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.suptitle(f"Probability Against Contacts to Each Category ({ORDER_TITLE}, {method_name})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/probability_against_category_visits_{method_name}.png", dpi=300)
    plt.close()


# --------------------------------------------------------------------------------------------------------------------------------------
# Plotting of Findings from DNA investigation
# --------------------------------------------------------------------------------------------------------------------------------------
# One figure per simulation method (absolute and normalised side by side), for each of:
# Levenshtein distance, Needleman-Wunsch global alignment and Smith-Waterman local alignment.

def skew_interpretation(skew):
    if abs(skew) < 0.5:
        return "Approximately symmetric"
    elif skew > 0:
        return "Right-skewed (tail to the right)"
    return "Left-skewed (tail to the left)"


def plot_score_panel(ax, values, mean, median, mode, mu, sigma, skew, colour, xlabel, title, decimals,
                     legend_loc, stats_box_x):
    """One histogram panel with mean / median / mode lines, a normal fit and a skew box."""
    sns.histplot(values, bins=50, alpha=0.6, color=colour, stat="probability", kde=False, ax=ax)

    # Add vertical lines for mean, median, mode
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.{decimals}f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median = {median:.{decimals}f}')
    ax.axvline(mode, color='orange', linestyle='--', linewidth=2, label=f'Mode = {mode:.{decimals}f}')

    # Normal fit overlay - scale to probability (not density) using the bin width
    x_range = np.linspace(values.min(), values.max(), 200)
    bin_width = (values.max() - values.min()) / 50
    if sigma > 0 and bin_width > 0:
        normal_curve = stats.norm.pdf(x_range, mu, sigma) * bin_width
        ax.plot(x_range, normal_curve, 'black', linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'Theoretical Normal (μ={mu:.{decimals}f}, σ={sigma:.{decimals}f})')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"{title}\n(All Observed vs All Simulated)", fontsize=13)
    ax.legend(fontsize=9, loc=legend_loc)
    ax.grid(True, alpha=0.3)

    # Stats legend
    stats_text = f"n = {len(values):,}\n{skew_interpretation(skew)}\nSkew = {skew:.3f}"
    ax.text(stats_box_x, 0.50, stats_text, transform=ax.transAxes, ha='right', va='center',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))


def plot_score_distribution(results, metric_title, xlabel, file_stem, legend_loc, stats_box_x):
    for method_name, res in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        plot_score_panel(axes[0], res['scores'], res['mean_abs'], res['median_abs'], res['mode_abs'],
                         res['mu_abs'], res['sigma_abs'], res['skew_abs'], "steelblue",
                         f"Absolute {xlabel}", f"Distribution of Absolute {metric_title}", 1,
                         legend_loc, stats_box_x)
        plot_score_panel(axes[1], res['normalised'], res['mean_norm'], res['median_norm'], res['mode_norm'],
                         res['mu_norm'], res['sigma_norm'], res['skew_norm'], "coral",
                         f"Normalised {xlabel}", f"Distribution of Normalised {metric_title}", 3,
                         legend_loc, stats_box_x)

        plt.suptitle(f"{ORDER_TITLE}, {method_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{file_stem}_{method_name}.png", dpi=300)
        plt.close(fig)


# Levenshtein Distance Plot
plot_score_distribution(levenshtein_results, "Levenshtein Distances", "Levenshtein Distance",
                        "levenshtein_distance_distribution", "upper right", 0.98)

# Needleman-Wunsch Plot
plot_score_distribution(nw_results, "NW Alignment Scores", "Needleman-Wunsch Score",
                        "needleman_wunsch_distribution", "upper left", 0.45)

# Smith-Waterman Plot
plot_score_distribution(sw_results, "SW Local Alignment Scores", "Smith-Waterman Score",
                        "smith_waterman_distribution", "upper right", 0.98)


# --------------------------------------------------------------------------------------------------------------------------------------
# Convergence Plots
# --------------------------------------------------------------------------------------------------------------------------------------

def plot_convergence(convergence_results, target_transition, save_path=None):
    """
    Plot how a transition probability changes as more training episodes are added.
    Uses the same confidence interval and stability rule as the printed assessment.
    """
    sample_sizes = [r['sample_size'] for r in convergence_results]
    probabilities = [r['probability'] for r in convergence_results]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, probabilities, 'b-o', linewidth=2, markersize=6)

    # 95% confidence interval at each point (Wilson method)
    # Uses from_count: how many times the starting state / state pair occurred
    lower_ci = []
    upper_ci = []
    for result in convergence_results:
        lo, hi = wilson_ci(result['target_count'], result['from_count'])
        lower_ci.append(lo)
        upper_ci.append(hi)

    plt.fill_between(sample_sizes, lower_ci, upper_ci, alpha=0.2, color='blue',
                     label='95% Confidence Interval')

    # Final estimate line
    final_prob = probabilities[-1]
    plt.axhline(y=final_prob, color='red', linestyle='--', alpha=0.7,
                label=f'Final Estimate: {final_prob:.4f}')

    history, next_state = target_transition[:-1], target_transition[-1]
    plt.xlabel('Number of Training Episodes', fontsize=12)
    plt.ylabel(f'P({next_state} | {", ".join(history)})', fontsize=12)
    history_text = history[0] if len(history) == 1 else f"({', '.join(history)})"
    plt.title(f'Convergence of Transition Probability ({ORDER_TITLE})\n{history_text} → {next_state}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Stability badge: uses the SAME function as the printout
    stability_text = assess_stability(convergence_results)

    if stability_text.startswith("STABLE"):
        text_color = 'green'
    elif stability_text.startswith("MODERATE"):
        text_color = 'orange'
    elif stability_text.startswith("UNSTABLE"):
        text_color = 'red'
    else:  # INSUFFICIENT DATA
        text_color = 'grey'

    plt.text(0.02, 0.98, f'Stability: {stability_text}',
             transform=plt.gca().transAxes, fontsize=10, color=text_color,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=text_color))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.close()

# Plot main categories convergence
plot_convergence(convergence_main, target_main, f"{OUTPUT_DIR}/convergence_main_categories.png")

# Plot all categories convergence
plot_convergence(convergence_all, target_all, f"{OUTPUT_DIR}/convergence_all_categories.png")
