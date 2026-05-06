# --------------------------------------------------------------------------------------------------------------------------------------
# Plotting of Findings
# --------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from config import main_categories, all_categories, n_simulations, OUTPUT_DIR

from data_loader import df_train, train_ids
from transition_matrices import *
from simulations import *
from metrics import *


# --------------------------------------------------------------------------------------------------------------------------------------
# Probability against Care Period Lengths
# --------------------------------------------------------------------------------------------------------------------------------------

# Define the three simulation methods to loop through
simulation_methods = {
    "25x5_fixed": fixed_chains,
    "25x5_fitted": fitted_chains,
    "49x7_terminating": terminating_chains
}

# Loop through each method and create plots
for method_name, method_chains in simulation_methods.items():

    # Get lengths from real data
    real_lengths = df_train.groupby("ActivityID").size().values

    # Get lengths from simulated data
    simulated_lengths = [len(chain) for chain in method_chains]

    # Calculate Hellinger Distance
    h_dist_lengths = hellinger_distance_histograms(real_lengths, simulated_lengths)

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
    plt.title(f"Probability Against Care Period Length ({method_name})", fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)

    # Add Hellinger Distance to plot
    plt.text(0.95, 0.95, f"Hellinger Distance: {h_dist_lengths:.4f}", 
            transform=plt.gca().transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/care_period_length_distribution_{method_name}.png", dpi=300)
    plt.show()


# --------------------------------------------------------------------------------------------------------------------------------------
# Probability against Number of Visits to Each Category
# --------------------------------------------------------------------------------------------------------------------------------------

# Loop through each method
for method_name, method_chains in simulation_methods.items():

    hellinger_distance_visits = {}

    # Collect all data to determine global min/max
    all_real_visits = {}
    all_sim_visits = {}

    # Use the correct categories for each method
    if method_name == "49x7_terminating":
        categories_to_use = all_categories
    else:
        categories_to_use = main_categories

    for category in categories_to_use:
        # Count visits in real data
        real_visits = []
        for activity_id in train_ids:
            activity_data = df_train[df_train["ActivityID"] == activity_id]
            count = (activity_data["SurfaceCategories"] == category).sum()
            real_visits.append(count)
        all_real_visits[category] = real_visits
        
        # Count visits in simulated data
        sim_visits = []
        for chain in method_chains:
            count = chain.count(category)
            sim_visits.append(count)
        
        all_sim_visits[category] = sim_visits

    # Find global min/max for x-axis and y-axis
    if all_real_visits and all_sim_visits:
        global_min_x = min(min(min(visits) for visits in all_real_visits.values()),
                    min(min(visits) for visits in all_sim_visits.values()))
        global_max_x = max(max(max(visits) for visits in all_real_visits.values()),
                    max(max(visits) for visits in all_sim_visits.values()))
    else:
        global_min_x, global_max_x = 0, 1 

    bins_global = np.arange(global_min_x, global_max_x + 2, 1)

    # Calculate max probability for y-axis
    global_max_y = 0
    for category in categories_to_use:
        if category in all_real_visits and category in all_sim_visits:
            hist_real, _ = np.histogram(all_real_visits[category], bins=bins_global)
            hist_sim, _ = np.histogram(all_sim_visits[category], bins=bins_global)
            max_prob_real = hist_real.max() / len(all_real_visits[category])
            max_prob_sim = hist_sim.max() / len(all_sim_visits[category])
            global_max_y = max(global_max_y, max_prob_real, max_prob_sim)

    # Create the plots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 6, figure=fig)

    # Adjust grid based on number of categories
    num_categories = len(categories_to_use)
    if num_categories <= 5:
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

    for idx, category in enumerate(categories_to_use):
        if idx >= len(axes):
            break

        real_visits = all_real_visits[category]
        sim_visits = all_sim_visits[category]
        
        # Calculate Hellinger Distance
        h_dist = hellinger_distance_histograms(real_visits, sim_visits)
        hellinger_distance_visits[category] = h_dist
        
        # Plot
        ax = axes[idx]
        sns.histplot(real_visits, bins=bins_global, alpha=0.4, color="blue", label="Observed", 
                    stat="probability", kde=False, ax=ax)
        sns.histplot(sim_visits, bins=bins_global, alpha=0.4, color="orange", label="Predicted", 
                    stat="probability", kde=False, ax=ax)
        
        # Set uniform axes for all subplots
        ax.set_xlim(global_min_x - 0.5, global_max_x + 1.5)
        ax.set_ylim(0, global_max_y * 1.1)  # Add 10% padding
        
        ax.set_xlabel("Number of Visits")
        ax.set_ylabel("Probability")
        ax.set_title(f"{category}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add Hellinger Distance
        ax.text(0.95, 0.95, f"H = {h_dist:.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    plt.suptitle(f"Probability Against Visits to Each Category ({method_name})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/probability_against_category_visits_{method_name}.png", dpi=300)


# --------------------------------------------------------------------------------------------------------------------------------------
# Plotting of Findings from DNA investigation
# --------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------
# Levenshtein Distance Plot
# ----------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Absolute Levenshtein Distances
ax1 = axes[0]
sns.histplot(all_levenshtein_distances, bins=50, alpha=0.6, color="steelblue", 
             stat="probability", kde=False, ax=ax1)

# Add vertical lines for mean, median, mode
ax1.axvline(mean_abs, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_abs:.1f}')
ax1.axvline(median_abs, color='green', linestyle='--', linewidth=2, label=f'Median = {median_abs:.1f}')
ax1.axvline(mode_abs, color='orange', linestyle='--', linewidth=2, label=f'Mode = {mode_abs:.1f}')
        
# Normal fit overlay - scale to probability (not density)
x_range = np.linspace(all_levenshtein_distances.min(), all_levenshtein_distances.max(), 200)

# Get bin width to scale the normal curve to match probability bars
bin_width = (all_levenshtein_distances.max() - all_levenshtein_distances.min()) / 50
normal_curve = stats.norm.pdf(x_range, mu_abs, sigma_abs) * bin_width
ax1.plot(x_range, normal_curve, 'r-', linewidth=2, 
         label=f'Theoretical Normal (μ={mu_abs:.1f}, σ={sigma_abs:.1f})')

ax1.set_xlabel("Absolute Levenshtein Distance", fontsize=12)
ax1.set_ylabel("Probability", fontsize=12)
ax1.set_title("Distribution of Absolute Levenshtein Distances\n(All Observed vs All Simulated)", fontsize=13)
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3)

# Skew
if abs(skew_abs) < 0.5:
    skew_interpretation = "Approximately symmetric"
elif skew_abs > 0:
    skew_interpretation = "Right-skewed (tail to the right)"
else:
    skew_interpretation = "Left-skewed (tail to the left)"

# Stats legend
stats_text = f"n = {len(all_levenshtein_distances):,}\n{skew_interpretation}\nSkew = {skew_abs:.3f}"
ax1.text(0.98, 0.50, stats_text, transform=ax1.transAxes, ha='right', va='center', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

# Plot 2: Normalised Levenshtein Distances
ax2 = axes[1]
sns.histplot(all_levenshtein_normalised, bins=50, alpha=0.6, color="coral", 
             stat="probability", kde=False, ax=ax2)

# Add vertical lines
ax2.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_norm:.3f}')
ax2.axvline(median_norm, color='green', linestyle='--', linewidth=2, label=f'Median = {median_norm:.3f}')
ax2.axvline(mode_norm, color='orange', linestyle='--', linewidth=2, label=f'Mode = {mode_norm:.3f}')
        
# Normal fit overlay - scale to probability
x_range_norm = np.linspace(all_levenshtein_normalised.min(), all_levenshtein_normalised.max(), 200)
bin_width_norm = (all_levenshtein_normalised.max() - all_levenshtein_normalised.min()) / 50
normal_curve_norm = stats.norm.pdf(x_range_norm, mu_norm, sigma_norm) * bin_width_norm
ax2.plot(x_range_norm, normal_curve_norm, 'r-', linewidth=2, 
         label=f'Theoretical Normal (μ={mu_norm:.3f}, σ={sigma_norm:.3f})')

ax2.set_xlabel("Normalised Levenshtein Distance", fontsize=12)
ax2.set_ylabel("Probability", fontsize=12)
ax2.set_title("Distribution of Normalised Levenshtein Distances\n(All Observed vs All Simulated)", fontsize=13)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.3)

# Skew
if abs(skew_norm) < 0.5:
    skew_interpretation_norm = "Approximately symmetric"
elif skew_norm > 0:
    skew_interpretation_norm = "Right-skewed (tail to the right)"
else:
    skew_interpretation_norm = "Left-skewed (tail to the left)"

# Stats legend
stats_text_norm = f"n = {len(all_levenshtein_normalised):,}\n{skew_interpretation_norm}\nSkew = {skew_norm:.3f}"
ax2.text(0.98, 0.50, stats_text_norm, transform=ax2.transAxes, ha='right', va='center', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/levenshtein_distance_distribution.png", dpi=300)
plt.show()


# ----------------------------------
# Needleman-Wunsch Plot
# ----------------------------------
plt.close('all')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Absolute NW Scores
ax1 = axes[0]
sns.histplot(all_nw_scores, bins=50, alpha=0.6, color="steelblue", 
             stat="probability", kde=False, ax=ax1)

# Add vertical lines for mean, median, mode
ax1.axvline(mean_nw, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_nw:.1f}')
ax1.axvline(median_nw, color='green', linestyle='--', linewidth=2, label=f'Median = {median_nw:.1f}')
ax1.axvline(mode_nw, color='orange', linestyle='--', linewidth=2, label=f'Mode = {mode_nw:.1f}')

# Normal fit overlay
x_range = np.linspace(all_nw_scores.min(), all_nw_scores.max(), 200)
bin_width = (all_nw_scores.max() - all_nw_scores.min()) / 50
normal_curve = stats.norm.pdf(x_range, mu_nw, sigma_nw) * bin_width
ax1.plot(x_range, normal_curve, 'black', linewidth=1.5, linestyle='--', alpha=0.7,
         label=f'Theoretical Normal (μ={mu_nw:.1f}, σ={sigma_nw:.1f})')

ax1.set_xlabel("Needleman-Wunsch Score", fontsize=12)
ax1.set_ylabel("Probability", fontsize=12)
ax1.set_title("Distribution of NW Alignment Scores\n(All Observed vs All Simulated)", fontsize=13)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.3)

# Skew interpretation
if abs(skew_nw) < 0.5:
    skew_interpretation_nw = "Approximately symmetric"
elif skew_nw > 0:
    skew_interpretation_nw = "Right-skewed"
else:
    skew_interpretation_nw = "Left-skewed"

# Stats text box
stats_text_nw = f"n = {len(all_nw_scores):,}\n{skew_interpretation_nw}\nSkew = {skew_nw:.3f}"
ax1.text(0.20, 0.50, stats_text_nw, transform=ax1.transAxes, ha='right', va='center', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

# Plot 2: Normalised NW Scores
ax2 = axes[1]
sns.histplot(all_nw_normalised, bins=50, alpha=0.6, color="coral", 
             stat="probability", kde=False, ax=ax2)

# Add vertical lines
ax2.axvline(mean_nw_norm, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_nw_norm:.3f}')
ax2.axvline(median_nw_norm, color='green', linestyle='--', linewidth=2, label=f'Median = {median_nw_norm:.3f}')
ax2.axvline(mode_nw_norm, color='orange', linestyle='--', linewidth=2, label=f'Mode = {mode_nw_norm:.3f}')

# Normal fit overlay
x_range_norm = np.linspace(all_nw_normalised.min(), all_nw_normalised.max(), 200)
bin_width_norm = (all_nw_normalised.max() - all_nw_normalised.min()) / 50
normal_curve_norm = stats.norm.pdf(x_range_norm, mu_nw_norm, sigma_nw_norm) * bin_width_norm
ax2.plot(x_range_norm, normal_curve_norm, 'black', linewidth=1.5, linestyle='--', alpha=0.7,
         label=f'Theoretical Normal (μ={mu_nw_norm:.3f}, σ={sigma_nw_norm:.3f})')

ax2.set_xlabel("Normalised NW Score", fontsize=12)
ax2.set_ylabel("Probability", fontsize=12)
ax2.set_title("Distribution of Normalised NW Scores\n(All Observed vs All Simulated)", fontsize=13)
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, alpha=0.3)

# Skew interpretation
if abs(skew_nw_norm) < 0.5:
    skew_interpretation_nw_norm = "Approximately symmetric"
elif skew_nw_norm > 0:
    skew_interpretation_nw_norm = "Right-skewed"
else:
    skew_interpretation_nw_norm = "Left-skewed"

# Stats text box
stats_text_nw_norm = f"n = {len(all_nw_normalised):,}\n{skew_interpretation_nw_norm}\nSkew = {skew_nw_norm:.3f}"
ax2.text(0.20, 0.50, stats_text_nw_norm, transform=ax2.transAxes, ha='right', va='center', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/needleman_wunsch_distribution.png", dpi=300)
plt.show()


# ----------------------------------
# Smith-Waterman Plot
# ----------------------------------
plt.close('all')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Absolute SW Scores
ax1 = axes[0]
sns.histplot(all_sw_scores, bins=50, alpha=0.6, color="steelblue", 
             stat="probability", kde=False, ax=ax1)

# Add vertical lines for mean, median, mode
ax1.axvline(mean_sw, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_sw:.1f}')
ax1.axvline(median_sw, color='green', linestyle='--', linewidth=2, label=f'Median = {median_sw:.1f}')
ax1.axvline(mode_sw, color='orange', linestyle='--', linewidth=2, label=f'Mode = {mode_sw:.1f}')

# Normal fit overlay
x_range = np.linspace(all_sw_scores.min(), all_sw_scores.max(), 200)
bin_width = (all_sw_scores.max() - all_sw_scores.min()) / 50
normal_curve = stats.norm.pdf(x_range, mu_sw, sigma_sw) * bin_width
ax1.plot(x_range, normal_curve, 'black', linewidth=1.5, linestyle='--', alpha=0.7,
         label=f'Theoretical Normal (μ={mu_sw:.1f}, σ={sigma_sw:.1f})')

ax1.set_xlabel("Smith-Waterman Score", fontsize=12)
ax1.set_ylabel("Probability", fontsize=12)
ax1.set_title("Distribution of SW Local Alignment Scores\n(All Observed vs All Simulated)", fontsize=13)
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3)

# Plot 2: Normalised SW Scores  
ax2 = axes[1]
sns.histplot(all_sw_normalised, bins=50, alpha=0.6, color="coral", 
             stat="probability", kde=False, ax=ax2)

ax2.axvline(mean_sw_norm, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_sw_norm:.3f}')
ax2.axvline(median_sw_norm, color='green', linestyle='--', linewidth=2, label=f'Median = {median_sw_norm:.3f}')
ax2.axvline(mode_sw_norm, color='orange', linestyle='--', linewidth=2, label=f'Mode = {mode_sw_norm:.3f}')

x_range_norm = np.linspace(all_sw_normalised.min(), all_sw_normalised.max(), 200)
bin_width_norm = (all_sw_normalised.max() - all_sw_normalised.min()) / 50
normal_curve_norm = stats.norm.pdf(x_range_norm, mu_sw_norm, sigma_sw_norm) * bin_width_norm
ax2.plot(x_range_norm, normal_curve_norm, 'black', linewidth=1.5, linestyle='--', alpha=0.7,
         label=f'Theoretical Normal (μ={mu_sw_norm:.3f}, σ={sigma_sw_norm:.3f})')

ax2.set_xlabel("Normalised SW Score", fontsize=12)
ax2.set_ylabel("Probability", fontsize=12)
ax2.set_title("Distribution of Normalised SW Scores\n(All Observed vs All Simulated)", fontsize=13)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/smith_waterman_distribution.png", dpi=300)
plt.show()


# --------------------------------------------------------------------------------------------------------------------------------------
# Convergence Plots
# --------------------------------------------------------------------------------------------------------------------------------------
def plot_convergence(convergence_results, target_transition, save_path=None):
    """
    Plot convergence of transition probability
    """
    sample_sizes = [r['sample_size'] for r in convergence_results]
    probabilities = [r['probability'] for r in convergence_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, probabilities, 'b-o', linewidth=2, markersize=6)
    
    # Add confidence intervals (approximate)
    # For binomial proportion: CI ≈ p ± 1.96*sqrt(p(1-p)/n)
    lower_ci = []
    upper_ci = []
    
    for result in convergence_results:
        p = result['probability']
        n = result['total_transitions']
        if n > 0 and p > 0:
            se = np.sqrt(p * (1 - p) / n)  # Standard error
            margin = 1.96 * se  # 95% confidence interval
            lower_ci.append(max(0, p - margin))
            upper_ci.append(min(1, p + margin))
        else:
            lower_ci.append(0)
            upper_ci.append(0)
    
    # Plot confidence band
    plt.fill_between(sample_sizes, lower_ci, upper_ci, alpha=0.2, color='blue', 
                     label='95% Confidence Interval')
    
    # Final value line (what we converge to)
    final_prob = probabilities[-1]
    plt.axhline(y=final_prob, color='red', linestyle='--', alpha=0.7, 
                label=f'Final Estimate: {final_prob:.4f}')
    
    plt.xlabel('Number of Training Episodes', fontsize=12)
    prev_state, curr_state, next_state = target_transition
    plt.ylabel(f'P({next_state} | {prev_state}, {curr_state})', fontsize=12)
    plt.title(f'Convergence of Transition Probability\n({prev_state}, {curr_state}) → {next_state}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotation about stability
    if len(probabilities) >= 3:
        last_three = probabilities[-3:]
        std_last_three = np.std(last_three)
        if std_last_three < 0.01:
            stability_text = "STABLE (σ < 0.01)"
            text_color = 'green'
        elif std_last_three < 0.05:
            stability_text = "MODERATE (σ < 0.05)"
            text_color = 'orange'
        else:
            stability_text = "UNSTABLE (σ ≥ 0.05)"
            text_color = 'red'
        
        plt.text(0.02, 0.98, f'Stability: {stability_text}', 
                transform=plt.gca().transAxes, fontsize=10, color=text_color,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=text_color))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

# Plot main categories convergence
plot_convergence(
    convergence_main, 
    target_main, 
    f"{OUTPUT_DIR}/convergence_main_categories.png"
)

# Plot all categories convergence  
plot_convergence(
    convergence_all,
    target_all,
    f"{OUTPUT_DIR}/convergence_all_categories.png"
)