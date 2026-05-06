# --------------------------------------------------------------------------------------------------------------------------------------
# Metrics for Comparison of Observed and Predicted
# --------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import time

from Levenshtein import distance
from Levenshtein import ratio
from scipy import stats
from scipy.stats import chisquare

from config import n_simulations
from data_loader import df_train, train_ids
from transition_matrices import *
from simulations import *


# --------------------------------------------------------------------------------------
# MAE with Proper Alignment
# --------------------------------------------------------------------------------------
# Only evaluates on transitions that exist in observed data
# Predicted values for missing transitions default to 0

def calculate_mae_safe(observed, predicted):
    pred_aligned = predicted.reindex(
        index=observed.index,
        columns=observed.columns,
        fill_value=0
    )
    return np.abs(observed - pred_aligned).mean().mean()


# --------------------------------------------------------------------------------------
# Compare Observed vs Predicted
# --------------------------------------------------------------------------------------

# Difference in observed and predicted transition matrices
pred_aligned = predicted_fixed_25x5.reindex(
    index=observed_main.index,
    columns=observed_main.columns,
    fill_value=0
)

difference_fixed_25x5 = observed_main - pred_aligned

pred_fitted_aligned = predicted_fitted_25x5.reindex(
    index=observed_main.index,
    columns=observed_main.columns,
    fill_value=0
)

difference_fitted_25x5 = observed_main - pred_fitted_aligned


pred_term_aligned = predicted_terminating_49x7.reindex(
    index=observed_all.index,
    columns=observed_all.columns,
    fill_value=0
)

difference_terminating_49x7 = observed_all - pred_term_aligned

# Overall error
mae_main = calculate_mae_safe(observed_main, predicted_fixed_25x5)
mae_all = calculate_mae_safe(observed_all, predicted_terminating_49x7)
mae_fitted = calculate_mae_safe(observed_main, predicted_fitted_25x5)


# --------------------------------------------------------------------------------------
# Test Against Final 20% of Data
# ----=---------------------------------------------------------------------------------

#  Compare observed matrix with test matrix
train_test_difference_main = observed_main - test_matrix_main
train_test_difference_all = observed_all - test_matrix_all

# Overall error
train_test_mae_main = calculate_mae_safe(observed_main, test_matrix_main)
train_test_mae_all = calculate_mae_safe(observed_all, test_matrix_all)


# --------------------------------------------------------------------------------------------------------------------------------------
# Hellinger Distance Function
# --------------------------------------------------------------------------------------------------------------------------------------
# Measures the similarity between two probability distributions

def hellinger_distance_histograms(data1, data2, bins=20):
    """
    Calculate Hellinger distance between two datasets
    """
    # Create histograms with same bins
    min_val = min(min(data1), min(data2))
    max_val = max(max(data1), max(data2))

    hist1, bin_edges = np.histogram(data1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(data2, bins=bins, range=(min_val, max_val), density=True)

    # Normalise to make probability distributions
    hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
    hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

    # Hellinger distance
    return np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2))


# --------------------------------------------------------------------------------------------------------------------------------------
# Levenshtein Distance
# --------------------------------------------------------------------------------------------------------------------------------------

# Map from Surface Categories to single letters
category_to_letter = {
    "In":           "I",
    "Equipment":    "E",
    "Patient":      "P",
    "NearPatient":  "N",
    "FarPatient":   "F",
    "HygieneArea":  "H",
    "Out":          "O"
}

# Function to convert a chain to a string of letters 
def chain_to_string(chain):
    return "".join(category_to_letter[state] for state in chain)

# Call all observed care periods as strings
observed_chains = []
for activity_id in train_ids:
    chain = df_train[df_train["ActivityID"] == activity_id]["SurfaceCategories"].tolist()
    observed_chains.append(chain_to_string(chain))

# Create a dictionary to store the stats for all 3 methods
levenshtein_results = {}

# Loop through each method and calculate Levenshtein distances
for method_name, method_data in simulation_methods.items():
    method_chains = method_data["chains"]

    # Convert simulated chains to strings
    simulated_chains = []
    start_time_sim = time.time()
    
    for chain in method_chains:
        simulated_chains.append(chain_to_string(chain))
    
    time_sim = time.time() - start_time_sim
    
    # Calculate all pairwise Levenshtein distances
    start_time_calc = time.time()
    
    all_levenshtein_distances = []
    all_levenshtein_normalised = []
    
    for obs_str in observed_chains:
        for sim_str in simulated_chains:
            # Absolute distance
            dist_abs = distance(obs_str, sim_str)
            all_levenshtein_distances.append(dist_abs)
            
            # Normalised distance (ratio gives similarity, so 1 - ratio = normalised distance)
            dist_norm = 1 - ratio(obs_str, sim_str)
            all_levenshtein_normalised.append(dist_norm)
    
    time_calc = time.time() - start_time_calc
    
    # Convert to numpy arrays
    all_levenshtein_distances = np.array(all_levenshtein_distances)
    all_levenshtein_normalised = np.array(all_levenshtein_normalised)
    
    # Calculate statistics for summary and plotting
    # For absolute distances
    mean_abs = np.mean(all_levenshtein_distances)
    median_abs = np.median(all_levenshtein_distances)
    mode_abs = stats.mode(all_levenshtein_distances, keepdims=False)[0]
    skew_abs = stats.skew(all_levenshtein_distances)
    mu_abs, sigma_abs = stats.norm.fit(all_levenshtein_distances)
    
    # For normalised distances  
    mean_norm = np.mean(all_levenshtein_normalised)
    median_norm = np.median(all_levenshtein_normalised)
    mode_norm = stats.mode(all_levenshtein_normalised, keepdims=False)[0]
    skew_norm = stats.skew(all_levenshtein_normalised)
    mu_norm, sigma_norm = stats.norm.fit(all_levenshtein_normalised)

    # Store results in the dictionary
    levenshtein_results[method_name] = {
        'observed_count': len(observed_chains),
        'simulated_count': len(simulated_chains),
        'total_comparisons': len(all_levenshtein_distances),
        'time_sim': time_sim,
        'time_calc': time_calc,
        'mean_abs': mean_abs,
        'median_abs': median_abs,
        'mode_abs': mode_abs,
        'sigma_abs': sigma_abs,
        'min_abs': all_levenshtein_distances.min(),
        'max_abs': all_levenshtein_distances.max(),
        'skew_abs': skew_abs,
        'mu_abs': mu_abs,
        'mean_norm': mean_norm,
        'median_norm': median_norm,
        'mode_norm': mode_norm,
        'sigma_norm': sigma_norm,
        'min_norm': all_levenshtein_normalised.min(),
        'max_norm': all_levenshtein_normalised.max(),
        'skew_norm': skew_norm,
        'mu_norm': mu_norm
    }   


# --------------------------------------------------------------------------------------------------------------------------------------
# Needleman-Wunsch Algorithm (Global Alignment)
# --------------------------------------------------------------------------------------------------------------------------------------
def needleman_wunsch(x, y, match=1, mismatch=1, gap=1):
    """
    Aligns two sequences using Needleman-Wunsch global alignment.
    
    Parameters:
    x, y: sequences to align
    match: reward for matching character
    mismatch: penalty for mismatching characters
    gap: penalty for inserting a gap 
    """
    nx = len(x)
    ny = len(y)

    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    F[0,:] = np.linspace(0, -ny * gap, ny + 1)

    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4

    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap

            tmax = np.max(t)
            F[i+1,j+1] = tmax

            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4

    i = nx
    j = ny
    rx = []
    ry = []
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            rx.append(x[i-1])
            ry.append('-')
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(y[j-1])
            j -= 1

    aligned_x = ''.join(rx)[::-1]
    aligned_y = ''.join(ry)[::-1]
    score = F[nx, ny]
    
    return aligned_x, aligned_y, score


# Create a dictionary to store the stats for all 3 methods
nw_results = {}

# Loop through all three simulation methods
for method_name, method_data in simulation_methods.items():
    method_chains = method_data["chains"]

    # Convert simulated chains to strings
    simulated_chains = []
    start_time_sim = time.time()
    
    for chain in method_chains:
        simulated_chains.append(chain_to_string(chain))
    
    time_sim = time.time() - start_time_sim
        
    # Calculate all pairwise NW scores
    start_time_nw = time.time()

    all_nw_scores = []
    all_nw_normalised = []

    for obs_str in observed_chains:
        for sim_str in simulated_chains:
            # Get alignment score
            _, _, score = needleman_wunsch(obs_str, sim_str)
            all_nw_scores.append(score)
            
            # Normalised score (1.0 = perfect match)
            max_possible = min(len(obs_str), len(sim_str)) * 1  # match=1
            norm_score = score / max_possible if max_possible > 0 else 0
            all_nw_normalised.append(norm_score)

    time_nw = time.time() - start_time_nw

    # Convert to numpy arrays
    all_nw_scores = np.array(all_nw_scores)
    all_nw_normalised = np.array(all_nw_normalised)

    # Calculate statistics for summary and plotting
    # For absolute scores
    mean_nw = np.mean(all_nw_scores)
    median_nw = np.median(all_nw_scores)
    mode_nw = stats.mode(all_nw_scores, keepdims=False)[0]
    skew_nw = stats.skew(all_nw_scores)
    mu_nw, sigma_nw = stats.norm.fit(all_nw_scores)

    # For normalised scores
    mean_nw_norm = np.mean(all_nw_normalised)
    median_nw_norm = np.median(all_nw_normalised)
    mode_nw_norm = stats.mode(all_nw_normalised, keepdims=False)[0]
    skew_nw_norm = stats.skew(all_nw_normalised)
    mu_nw_norm, sigma_nw_norm = stats.norm.fit(all_nw_normalised)

    # Store results in the dictionary
    nw_results[method_name] = {
        'observed_count': len(observed_chains),
        'simulated_count': len(simulated_chains),
        'total_comparisons': len(all_nw_scores),
        'time_sim': time_sim,
        'time_calc': time_nw,
        'mean_abs': mean_nw,  
        'median_abs': median_nw,
        'mode_abs': mode_nw,
        'sigma_abs': sigma_nw,
        'min_abs': all_nw_scores.min(),
        'max_abs': all_nw_scores.max(),
        'skew_abs': skew_nw,
        'mu_abs': mu_nw,
        'mean_norm': mean_nw_norm,
        'median_norm': median_nw_norm,
        'mode_norm': mode_nw_norm,
        'sigma_norm': sigma_nw_norm,
        'min_norm': all_nw_normalised.min(),
        'max_norm': all_nw_normalised.max(),
        'skew_norm': skew_nw_norm,
        'mu_norm': mu_nw_norm
    } 


# --------------------------------------------------------------------------------------------------------------------------------------
# Smith-Waterman Algorithm (Local Alignment)
# --------------------------------------------------------------------------------------------------------------------------------------
def smith_waterman(x, y, match=2, mismatch=-1, gap=-1):
    """
    Aligns two sequences locally using Smith-Waterman algorithm.
    
    Identifies best subsequence alignment,  
    ignoring poorly matching regions.
    
    Parameters:
    x, y: sequences to align
    match: reward for matching characters
    mismatch: penalty for mismatching characters
    gap: penalty for inserting a gap
    
    Returns:
    aligned_x, aligned_y, score, start_pos_x, start_pos_y
    """
    nx, ny = len(x), len(y)
    
    # Initialise scoring matrix (all zeros for local alignment)
    F = np.zeros((nx + 1, ny + 1))
    
    # Track traceback directions
    P = np.zeros((nx + 1, ny + 1))
    
    max_score = 0
    max_i, max_j = 0, 0
    
    # Fill scoring matrix
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            # Calculate scores for three possible moves
            diagonal = F[i-1, j-1] + (match if x[i-1] == y[j-1] else mismatch)
            up = F[i-1, j] + gap
            left = F[i, j-1] + gap
            
            # Take maximum, but never go below 0 (different to global)
            F[i, j] = max(0, diagonal, up, left)
            
            # Track direction for traceback
            if F[i, j] == diagonal and diagonal > 0:
                P[i, j] = 1  # Diagonal
            elif F[i, j] == up and up > 0:
                P[i, j] = 2  # Up
            elif F[i, j] == left and left > 0:
                P[i, j] = 3  # Left
            
            # Track maximum score position
            if F[i, j] > max_score:
                max_score = F[i, j]
                max_i, max_j = i, j
    
    # Traceback from maximum score position
    aligned_x, aligned_y = [], []
    i, j = max_i, max_j
    
    while i > 0 and j > 0 and F[i, j] > 0:
        if P[i, j] == 1:  # Diagonal
            aligned_x.append(x[i-1])
            aligned_y.append(y[j-1])
            i, j = i-1, j-1
        elif P[i, j] == 2:  # Up
            aligned_x.append(x[i-1])
            aligned_y.append('-')
            i = i-1
        elif P[i, j] == 3:  # Left
            aligned_x.append('-')
            aligned_y.append(y[j-1])
            j = j-1
        else:
            break
    
    aligned_x_str = ''.join(reversed(aligned_x))
    aligned_y_str = ''.join(reversed(aligned_y))
    
    return aligned_x_str, aligned_y_str, max_score


# Create a dictionary to store the stats for all 3 methods
sw_results = {}

# Loop through all three simulation methods
for method_name, method_data in simulation_methods.items():
    method_chains = method_data["chains"]

    # Convert simulated chains to strings
    simulated_chains = []
    start_time_sim = time.time()
    
    for chain in method_chains:
        simulated_chains.append(chain_to_string(chain))
    
    time_sim = time.time() - start_time_sim
    
    # Calculate all pairwise SW scores
    start_time_sw = time.time()

    all_sw_scores = []
    all_sw_normalised = []

    for obs_str in observed_chains:
        for sim_str in simulated_chains:
            # Get alignment score
            _, _, score = smith_waterman(obs_str, sim_str)
            all_sw_scores.append(score)
            
            # Normalised score
            max_possible_local = min(len(obs_str), len(sim_str)) * 2  # match=2
            norm_score = score / max_possible_local if max_possible_local > 0 else 0
            all_sw_normalised.append(norm_score)

    time_sw = time.time() - start_time_sw

    # Convert to numpy arrays
    all_sw_scores = np.array(all_sw_scores)
    all_sw_normalised = np.array(all_sw_normalised)

    # Calculate statistics for summary and plotting
    # For absolute scores
    mean_sw = np.mean(all_sw_scores)
    median_sw = np.median(all_sw_scores)
    mode_sw = stats.mode(all_sw_scores, keepdims=False)[0]
    skew_sw = stats.skew(all_sw_scores)
    mu_sw, sigma_sw = stats.norm.fit(all_sw_scores)

    # For normalised scores
    mean_sw_norm = np.mean(all_sw_normalised)
    median_sw_norm = np.median(all_sw_normalised)
    mode_sw_norm = stats.mode(all_sw_normalised, keepdims=False)[0]
    skew_sw_norm = stats.skew(all_sw_normalised)
    mu_sw_norm, sigma_sw_norm = stats.norm.fit(all_sw_normalised)

    # Store results in the dictionary
    sw_results[method_name] = {
        'observed_count': len(observed_chains),
        'simulated_count': len(simulated_chains),
        'total_comparisons': len(all_sw_scores),
        'time_sim': time_sim,
        'time_calc': time_sw,
        'mean_abs': mean_sw,  
        'median_abs': median_sw,
        'mode_abs': mode_sw,
        'sigma_abs': sigma_sw,
        'min_abs': all_sw_scores.min(),
        'max_abs': all_sw_scores.max(),
        'skew_abs': skew_sw,
        'mu_abs': mu_sw,
        'mean_norm': mean_sw_norm,
        'median_norm': median_sw_norm,
        'mode_norm': mode_sw_norm,
        'sigma_norm': sigma_sw_norm,
        'min_norm': all_sw_normalised.min(),
        'max_norm': all_sw_normalised.max(),
        'skew_norm': skew_sw_norm,
        'mu_norm': mu_sw_norm
    } 


# --------------------------------------------------------------------------------------------------------------------------------------
# Sensitivity Analysis
# --------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------
# Matrix Perturbation
# --------------------------------------------------------------------------------------------------------------------------------------
# Test if small errors in the transition matrix completely change results.
# If the model is robust, small changes should not matter.

def perturb_matrix(matrix, noise_level=0.01):
    """
    Add small random noise to transition matrix to test sensitivity.
    
    Parameters:
    matrix: Original Transition Matrix
    noise_level: Maximum amount of noise to add
    
    Returns:
    Perturbed matrix with rows normalised to sum to 1
    """
    # Convert to numpy for easier manipulation
    matrix_values = matrix.values.copy()
    
    # Add random noise between -noise_level and +noise_level
    noise = np.random.uniform(-noise_level, noise_level, matrix_values.shape)
    perturbed_values = matrix_values + noise
    
    # Ensure no negative probabilities
    perturbed_values = np.maximum(perturbed_values, 0.001)  # Minimum probability
    
    # Normalise rows to sum to 1
    row_sums = perturbed_values.sum(axis=1, keepdims=True)
    perturbed_values = perturbed_values / row_sums
    
    # Convert back to DataFrame with same index/columns
    perturbed_matrix = pd.DataFrame(
        perturbed_values, 
        index=matrix.index, 
        columns=matrix.columns
    )
    
    return perturbed_matrix

# Test multiple levels of perturbation
noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1%, 0.5%, 1%, 2%, 5%
n_tests_per_level = 50


# ----------------------------------
# 25x5 Perturbation
# ----------------------------------

sensitivity_results_25x5 = []

for noise_level in noise_levels:
    mae_values_25x5 = []
    
    for test in range(n_tests_per_level):
        # Perturb the observed 25x5 matrix
        perturbed_main = perturb_matrix(observed_main, noise_level)
        
        # Run simulations with perturbed matrix
        perturbed_transitions_25x5 = []
        for _ in range(100):
            chain = simulate_25x5(perturbed_main, train_start_probs_main, main_categories, avg_length)
            for i in range(len(chain) - 2):
                perturbed_transitions_25x5.append({"Previous": chain[i], "Current": chain[i + 1], "Next": chain[i + 2]})
        
        # Calculate predicted matrix
        df_perturbed_25x5 = pd.DataFrame(perturbed_transitions_25x5)
        predicted_perturbed_25x5 = calculate_transition_matrix(
            df_perturbed_25x5, "Previous", "Current", "Next", main_categories
        )
        
        # Compare to original predicted matrix
        mae_25x5 = calculate_mae_safe(predicted_fixed_25x5, predicted_perturbed_25x5)
        mae_values_25x5.append(mae_25x5)
    
    # Store results
    mean_mae_25x5 = np.mean(mae_values_25x5)
    std_mae_25x5 = np.std(mae_values_25x5)
    sensitivity_results_25x5.append({
        'noise_level': noise_level,
        'mean_mae': mean_mae_25x5,
        'std_mae': std_mae_25x5
    })
    

# ----------------------------------
# 49x7 Perturbation
# ----------------------------------

sensitivity_results_49x7 = []

for noise_level in noise_levels:
    mae_values_49x7 = []
    
    for test in range(n_tests_per_level):
        # Perturb the observed 49x7 matrix
        perturbed_all = perturb_matrix(observed_all, noise_level)
        
        # Run simulations with perturbed matrix
        perturbed_transitions_49x7 = []
        for _ in range(100):
            chain = simulate_49x7(perturbed_all, train_start_probs_all, all_categories)
            for i in range(len(chain) - 2):
                perturbed_transitions_49x7.append({"Previous": chain[i], "Current": chain[i + 1], "Next": chain[i + 2]})
        
        # Calculate predicted matrix
        df_perturbed_49x7 = pd.DataFrame(perturbed_transitions_49x7)
        predicted_perturbed_49x7 = calculate_transition_matrix(
            df_perturbed_49x7, "Previous", "Current", "Next", all_categories
        )
        
        # Compare to original predicted matrix
        mae_49x7 = calculate_mae_safe(predicted_terminating_49x7, predicted_perturbed_49x7)
        mae_values_49x7.append(mae_49x7)
    
    # Store results
    mean_mae_49x7 = np.mean(mae_values_49x7)
    std_mae_49x7 = np.std(mae_values_49x7)
    sensitivity_results_49x7.append({
        'noise_level': noise_level,
        'mean_mae': mean_mae_49x7,
        'std_mae': std_mae_49x7
    })


# ----------------------------------
# Comparison of 25x5 and 49x7 Sensitivity
# ----------------------------------
# Which model is more robust.

# Store comparison results
comparison_results = []

for i, noise_level in enumerate(noise_levels):
    mae_25x5 = sensitivity_results_25x5[i]['mean_mae']
    mae_49x7 = sensitivity_results_49x7[i]['mean_mae']
    
    more_robust = "25x5" if mae_25x5 < mae_49x7 else "49x7"
    if abs(mae_25x5 - mae_49x7) < 0.001:
        more_robust = "Similar"
    
    print(f"{noise_level*100:8.1f}%   | {mae_25x5:8.4f}   | {mae_49x7:8.4f}   | {more_robust}")

# Overall robustness assessment
final_mae_25x5 = sensitivity_results_25x5[-1]['mean_mae']
final_mae_49x7 = sensitivity_results_49x7[-1]['mean_mae']

if abs(final_mae_25x5 - final_mae_49x7) < 0.01:
    conclusion = "Both models show similar sensitivity to noise"
elif final_mae_25x5 < final_mae_49x7:
    conclusion = "25x5 model is more robust (less sensitive to noise)"
else:
    conclusion = "49x7 model is more robust (less sensitive to noise)"


# --------------------------------------------------------------------------------------------------------------------------------------
# Bootstrap Analysis
# --------------------------------------------------------------------------------------------------------------------------------------
# Randomly sample training data multiple times to see how much the transition matrix would change if different data was collected
# Also known as 'Data Resampling'

def bootstrap_transition_matrix(df_train, sample_fraction=0.8, n_bootstrap=100):
    """
    Create multiple second-order transition matrices from random samples.
    """
    activity_ids = df_train["ActivityID"].unique()
    matrices_25x5 = []
    matrices_49x7 = []
    
    for i in range(n_bootstrap):
        # Random sample of care episodes
        sample_size = int(len(activity_ids) * sample_fraction)
        sample_ids = np.random.choice(activity_ids, size=sample_size, replace=False)
        
        # Create sample dataset
        sample_data = df_train[df_train["ActivityID"].isin(sample_ids)].copy()
        
        # Prepare sample data
        sample_data["Current_Category"] = sample_data.groupby("ActivityID")["SurfaceCategories"].shift(-1)
        sample_data["Next_Category"] = sample_data.groupby("ActivityID")["SurfaceCategories"].shift(-2)
        sample_clean = sample_data.dropna(subset=["Current_Category", "Next_Category"])
        
        # 25x5 matrix
        sample_main = sample_clean[
            (sample_clean["SurfaceCategories"].isin(main_categories)) &
            (sample_clean["Current_Category"].isin(main_categories)) &
            (sample_clean["Next_Category"].isin(main_categories))
        ]
        
        if len(sample_main) > 0: 
            matrix_25x5 = calculate_transition_matrix(
                sample_main,
                "SurfaceCategories",    # Previous
                "Current_Category",     # Current
                "Next_Category",        # Next
                main_categories
            )

            # Use column names that match what calculate_transition_matrix produces
            full_index_25x5 = pd.MultiIndex.from_product(
                [main_categories, main_categories],
                names=["SurfaceCategories", "Current_Category"]  # <-- fixed
            )

            matrix_25x5 = matrix_25x5.reindex(
                index=full_index_25x5,
                columns=main_categories,
                fill_value=0
            )

            # Only append if shape is correct
            if matrix_25x5.shape == (25, 5):
                matrices_25x5.append(matrix_25x5.values)
        
        # 49x7 matrix
        sample_all = sample_clean[
            (sample_clean["SurfaceCategories"].isin(all_categories)) &
            (sample_clean["Current_Category"].isin(all_categories)) &
            (sample_clean["Next_Category"].isin(all_categories))
        ]

        if len(sample_all) > 0:
            matrix_49x7 = calculate_transition_matrix(
                sample_all,
                "SurfaceCategories",    # Previous
                "Current_Category",     # Current
                "Next_Category",        # Next
                all_categories
            )

            # Use column names that match what calculate_transition_matrix produces
            full_index_49x7 = pd.MultiIndex.from_product(
                [all_categories, all_categories],
                names=["SurfaceCategories", "Current_Category"]  # <-- fixed
            )

            matrix_49x7 = matrix_49x7.reindex(
                index=full_index_49x7,
                columns=all_categories,
                fill_value=0
            )

            # Only append if shape is correct
            if matrix_49x7.shape == (49, 7):
                matrices_49x7.append(matrix_49x7.values)
    
    return np.stack(matrices_25x5), np.stack(matrices_49x7)


# Run bootstrap analysis
bootstrap_25x5, bootstrap_49x7 = bootstrap_transition_matrix(df_train)

# Calculate statistics
mean_matrix_25x5 = np.mean(bootstrap_25x5, axis=0)
std_matrix_25x5 = np.std(bootstrap_25x5, axis=0)

mean_matrix_49x7 = np.mean(bootstrap_49x7, axis=0)
std_matrix_49x7 = np.std(bootstrap_49x7, axis=0)

# Compare to original - reindex to full grid to match bootstrap mean shape
original_25x5 = observed_main.reindex(
    index=pd.MultiIndex.from_product(
        [main_categories, main_categories],
        names=["SurfaceCategories", "Current_Category"]
    ),
    columns=main_categories,
    fill_value=0
).values

diff_from_mean_25x5 = np.abs(original_25x5 - mean_matrix_25x5)


# --------------------------------------------------------------------------------------------------------------------------------------
# Chi-Squared Test
# --------------------------------------------------------------------------------------------------------------------------------------
# Tests if the differences between observed and predicted matrices are statistically significant or random variation

def chi_squared_test_matrices(observed_matrix, predicted_matrix, matrix_name=""):
    """
    Tests if observed and predicted matrices are significantly different.
    
    H0 (null hypothesis): The matrices are the same
    H1 (alternative): The matrices are different
    
    If p-value < 0.05, we reject H0 (matrices are significantly different)
    """
    # Align both matrices to the same full index before flattening
    all_index = predicted_matrix.index.union(observed_matrix.index)
    all_columns = predicted_matrix.columns.union(observed_matrix.columns)

    observed_matrix = observed_matrix.reindex(index=all_index, columns=all_columns, fill_value=0)
    predicted_matrix = predicted_matrix.reindex(index=all_index, columns=all_columns, fill_value=0)

    # Flatten matrices to 1D arrays
    observed_flat = observed_matrix.values.flatten()
    predicted_flat = predicted_matrix.values.flatten()
    
    # Convert probabilities to expected counts
    total_transitions = 10000  # Arbitrary scaling factor
    
    observed_counts = observed_flat * total_transitions
    predicted_counts = predicted_flat * total_transitions
    
    # Remove entries where expected count is zero
    mask = predicted_counts > 0
    observed_clean = observed_counts[mask]
    predicted_clean = predicted_counts[mask]
    
    # Normalise to same total 
    observed_total = observed_clean.sum()
    predicted_total = predicted_clean.sum()
    predicted_clean = predicted_clean * (observed_total / predicted_total)

    # Chi-squared test
    chi2_statistic, p_value = chisquare(observed_clean, predicted_clean)
    
    # Degrees of freedom
    df = len(observed_clean) - 1
    
    return chi2_statistic, p_value, df, len(observed_clean)

print("=== Chi-Squared Tests ===")

# Test 25x5 matrices
chi2_25x5, p_val_25x5, df_25x5, n_cells_25x5 = chi_squared_test_matrices(
    observed_main, predicted_fixed_25x5, "25x5"
)

if p_val_25x5 < 0.001:
    significance_25x5 = "highly significant (p < 0.001)"
elif p_val_25x5 < 0.01:
    significance_25x5 = "very significant (p < 0.01)"
elif p_val_25x5 < 0.05:
    significance_25x5 = "significant (p < 0.05)"
else:
    significance_25x5 = "not significant (p ≥ 0.05)"


# Test 49x7 matrices
chi2_49x7, p_val_49x7, df_49x7, n_cells_49x7 = chi_squared_test_matrices(
    observed_all, predicted_terminating_49x7, "49x7"
)


if p_val_49x7 < 0.001:
    significance_49x7 = "highly significant (p < 0.001)"
elif p_val_49x7 < 0.01:
    significance_49x7 = "very significant (p < 0.01)"
elif p_val_49x7 < 0.05:
    significance_49x7 = "significant (p < 0.05)"
else:
    significance_49x7 = "not significant (p ≥ 0.05)"


# Test train vs test matrices for validation
chi2_test_25x5, p_val_test_25x5, df_test_25x5, n_cells_test_25x5 = chi_squared_test_matrices(
    observed_main, test_matrix_main, "Train vs Test 25x5"
)


# --------------------------------------------------------------------------------------------------------------------------------------
# Missing Transitions
# --------------------------------------------------------------------------------------------------------------------------------------
# Finding transitions that never occurred in the training data but might be possible in real life

def analyse_missing_transitions(observed_matrix, matrix_name=""):
    """
    Find transitions that never occurred in training data.
    
    These might be:
    1. Impossible transitions (e.g., Patient → Patient without leaving)
    2. Very rare but possible transitions
    3. Very plausible but missing due to insufficient data
    """
    zero_transitions = []

    # For second-order: index is (prev, curr) state pairs
    for state_pair in observed_matrix.index:
        for next_state in observed_matrix.columns:
            if observed_matrix.loc[state_pair, next_state] == 0:
                zero_transitions.append((state_pair, next_state))

    return zero_transitions


# 25x5 matrix
missing_main = analyse_missing_transitions(observed_main, "25x5")
n_possible_main = len(observed_main.index) * len(observed_main.columns)

# Categorise missing transitions
impossible_transitions = []
rare_transitions = []

for state_pair, next_state in missing_main:
    prev_state, curr_state = state_pair
    # Define what seems impossible vs just rare
    if curr_state == next_state == "Patient" and prev_state == "Patient":
        impossible_transitions.append((state_pair, next_state))
    elif curr_state == "HygieneArea" and next_state == "Patient":
        rare_transitions.append((state_pair, next_state))  # Should wash hands first!
    else:
        rare_transitions.append((state_pair, next_state))


# Analyse 49x7 matrix
missing_all = analyse_missing_transitions(observed_all, "49x7")
n_possible_all = len(observed_all.index) * len(observed_all.columns)

# Check if simulations predict any of these missing events
predicted_events = []

for state_pair, next_state in missing_main:
    try:
        prob = predicted_fixed_25x5.loc[state_pair, next_state]
        if prob > 0:
            predicted_events.append((state_pair, next_state, prob))
    except KeyError:
        pass  # State pair doesn't exist in predicted matrix either

# Check 49x7 predicted events
predicted_events_49x7 = []
for state_pair, next_state in missing_all:
    try:
        prob = predicted_terminating_49x7.loc[state_pair, next_state]
        if prob > 0:
            predicted_events_49x7.append((state_pair, next_state, prob))
    except KeyError:
        pass  # State pair doesn't exist in predicted matrix either

# --------------------------------------------------------------------------------------------------------------------------------------
# Hellinger Distances
# --------------------------------------------------------------------------------------------------------------------------------------

# Dictionary to store Hellinger results
hellinger_results = {}

for method_name, method_data in simulation_methods.items():
    method_chains = method_data["chains"]
    
    # Get lengths from real data
    real_lengths = df_train.groupby("ActivityID").size().values

    # Get lengths from simulated data
    simulated_lengths = [len(chain) for chain in method_chains]

    # Calculate Hellinger Distance for lengths
    h_dist_lengths = hellinger_distance_histograms(real_lengths, simulated_lengths)
    
    print(f"Method: {method_name}")
    print(f"  Care Period Lengths: {h_dist_lengths:.4f}")
    
    # Calculate Hellinger Distance for category visits
    if method_name == "49x7_terminating":
        categories_to_use = all_categories
    else:
        categories_to_use = main_categories
    
    hellinger_distance_visits = {}
    all_real_visits = {}
    all_sim_visits = {}

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
        
        # Calculate Hellinger Distance
        h_dist = hellinger_distance_histograms(real_visits, sim_visits)
        hellinger_distance_visits[category] = h_dist
        print(f"  {category}: {h_dist:.4f}")
    
    # Average Hellinger Distance
    avg_h_dist = np.mean(list(hellinger_distance_visits.values()))
    print(f"  Average Hellinger Distance (Category Visits): {avg_h_dist:.4f}")
    print()

    # Store results in the dictionary
    hellinger_results[method_name] = {
        'lengths': h_dist_lengths,
        'visits': hellinger_distance_visits,
        'average_visits': avg_h_dist
    }


# --------------------------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------------------------

# Compare Observed vs Predicted
print("=== Difference between Observed and Predicted Matrices ===")
print("For 25x5 matrix")
print(difference_fixed_25x5.round(3).to_string())
print()
print("For 49x7 matrix")
print(difference_terminating_49x7.round(3).to_string())
print()
print("For 25x5 fitted-length matrix")
print(difference_fitted_25x5.round(3).to_string())
print()
print(f"Mean Absolute Error for 25x5 (fixed length): {mae_main:.4f}")
print(f"Mean Absolute Error for 49x7: {mae_all:.4f}")
print(f"Mean Absolute Error for 25x5 (fitted length): {mae_fitted:.4f}")
print()

# Test Against Final 20% of Data
print("==== Difference between Observed and Test Matrices ===")
print("For 25x5 matrix")
print(train_test_difference_main.round(3).to_string())
print()
print("For 49x7 matrix")
print(train_test_difference_all.round(3).to_string())
print()
print(f"Mean Absolute Error (Train vs Test) for 25x5: {train_test_mae_main:.4f}")
print(f"Mean Absolute Error (Train vs Test) for 49x7: {train_test_mae_all:.4f}")
print()

# Levenshtein Distance
print("=== Levenshtein Distance ===")
for method_name, res in levenshtein_results.items():
    print(f"=== Method: ({method_name}) ===")
    print(f"Observed chains:  {res['observed_count']}")
    print(f"Simulated chains: {res['simulated_count']}")
    print(f"Total comparisons: {res['observed_count']} x {res['simulated_count']} = {res['total_comparisons']:,}")
    print(f"Simulation time:  {res['time_sim']:.2f} seconds")
    print(f"Calculation time: {res['time_calc']:.2f} seconds")
    print(f"Absolute Distance:")
    print(f"  Mean:   {res['mean_abs']:.2f}")
    print(f"  Median: {res['median_abs']:.2f}")
    print(f"  Mode:   {res['mode_abs']:.2f}")
    print(f"  Std:    {res['sigma_abs']:.2f}")
    print(f"  Min:    {res['min_abs']}")
    print(f"  Max:    {res['max_abs']}")
    print(f"  Skew:   {res['skew_abs']:.3f}")
    print(f"Normalised Distance:")
    print(f"  Mean:   {res['mean_norm']:.4f}")
    print(f"  Median: {res['median_norm']:.4f}")
    print(f"  Mode:   {res['mode_norm']:.4f}")
    print(f"  Std:    {res['sigma_norm']:.4f}")
    print(f"  Min:    {res['min_norm']:.4f}")
    print(f"  Max:    {res['max_norm']:.4f}")
    print(f"  Skew:   {res['skew_norm']:.3f}")
    print(f"Normal Fit (Absolute):    μ = {res['mu_abs']:.2f}, σ = {res['sigma_abs']:.2f}")
    print(f"Normal Fit (Normalised):  μ = {res['mu_norm']:.4f}, σ = {res['sigma_norm']:.4f}")
    print()

# Needleman-Wunsch Algorithm (Global Alignment)
print("=== Needleman-Wunsch Summary ===")
for method_name, res in nw_results.items():
    print(f"=== Method: ({method_name}) ===")
    print(f"Observed chains:  {res['observed_count']}")
    print(f"Simulated chains: {res['simulated_count']}")
    print(f"Total comparisons: {res['observed_count']} x {res['simulated_count']} = {res['total_comparisons']:,}")
    print(f"Calculation time: {res['time_calc']:.2f} seconds")
    print(f"Absolute Score (Higher is better/more similar):")
    print(f"  Mean:   {res['mean_abs']:.2f}")
    print(f"  Median: {res['median_abs']:.2f}")
    print(f"  Mode:   {res['mode_abs']:.2f}")
    print(f"  Std:    {res['sigma_abs']:.2f}")
    print(f"  Min:    {res['min_abs']:.2f}")
    print(f"  Max:    {res['max_abs']:.2f}")
    print(f"  Skew:   {res['skew_abs']:.3f}")
    print(f"Normalised Score:")
    print(f"  Mean:   {res['mean_norm']:.4f}")
    print(f"  Median: {res['median_norm']:.4f}")
    print(f"  Mode:   {res['mode_norm']:.4f}")
    print(f"  Std:    {res['sigma_norm']:.4f}")
    print(f"  Min:    {res['min_norm']:.4f}")
    print(f"  Max:    {res['max_norm']:.4f}")
    print(f"  Skew:   {res['skew_norm']:.3f}")
    print(f"Normal Fit (Absolute):    μ = {res['mu_abs']:.2f}, σ = {res['sigma_abs']:.2f}")
    print(f"Normal Fit (Normalised):  μ = {res['mu_norm']:.4f}, σ = {res['sigma_norm']:.4f}")
    print()

# Smith-Waterman Algorithm (Local Alignment)
print("=== Smith-Waterman Summary ===")
for method_name, res in sw_results.items():
    print(f"=== Method ({method_name}) ===")
    print(f"Observed chains:  {res['observed_count']}")
    print(f"Simulated chains: {res['simulated_count']}")
    print(f"Total comparisons: {res['observed_count']} x {res['simulated_count']} = {res['total_comparisons']:,}")
    print(f"Calculation time: {res['time_calc']:.2f} seconds")
    print(f"Absolute Score (Higher is better/more similar):")
    print(f"  Mean:   {res['mean_abs']:.2f}")
    print(f"  Median: {res['median_abs']:.2f}")
    print(f"  Mode:   {res['mode_abs']:.2f}")
    print(f"  Std:    {res['sigma_abs']:.2f}")
    print(f"  Min:    {res['min_abs']:.2f}")
    print(f"  Max:    {res['max_abs']:.2f}")
    print(f"  Skew:   {res['skew_abs']:.3f}")
    print(f"Normalised Score:")
    print(f"  Mean:   {res['mean_norm']:.4f}")
    print(f"  Median: {res['median_norm']:.4f}")
    print(f"  Mode:   {res['mode_norm']:.4f}")
    print(f"  Std:    {res['sigma_norm']:.4f}")
    print(f"  Min:    {res['min_norm']:.4f}")
    print(f"  Max:    {res['max_norm']:.4f}")
    print(f"  Skew:   {res['skew_norm']:.3f}")
    print(f"Normal Fit (Absolute):    μ = {res['mu_abs']:.2f}, σ = {res['sigma_abs']:.2f}")
    print(f"Normal Fit (Normalised):  μ = {res['mu_norm']:.4f}, σ = {res['sigma_norm']:.4f}")
    print()

# Matrix Perturbation
print("=== Sensitivity Analysis ===")
print("=== 25x5 Matrix Sensitivity ===")
for res in sensitivity_results_25x5:
    print(f"Noise Level {res['noise_level']*100:.1f}%: MAE = {res['mean_mae']:.4f} ± {res['std_mae']:.4f}")
print(f"5x5 Model sensitivity: {'HIGH' if sensitivity_results_25x5[-1]['mean_mae'] > 0.1 else 'LOW'}")
print()
print("=== 49x7 Matrix Sensitivity ===")
for res in sensitivity_results_49x7:
    print(f"Noise Level {res['noise_level']*100:.1f}%: MAE = {res['mean_mae']:.4f} ± {res['std_mae']:.4f}")
print(f" 49x7 Model sensitivity: {'HIGH' if sensitivity_results_49x7[-1]['mean_mae'] > 0.1 else 'LOW'}")
print()
print("=== Sensitivity Comparison ===")
print("Noise Level | 5x5 MAE    | 7x7 MAE    | More Robust")
print("-" * 50)
for res in comparison_results:
    print(f"{res['noise_level']*100:8.1f}%   | {res['mae_5x5']:8.4f}   | {res['mae_7x7']:8.4f}   | {res['more_robust']}")
print()
print(f"Overall Assessment:")
print(f"  25x5 matrix: {'Robust' if final_mae_25x5 < 0.05 else 'Sensitive'} (MAE = {final_mae_25x5:.4f})")
print(f"  49x7 matrix: {'Robust' if final_mae_49x7 < 0.05 else 'Sensitive'} (MAE = {final_mae_49x7:.4f})")
print(f"  Conclusion: {conclusion}")
print()

# Bootstrap Analysis
print("=== Bootstrap Analysis ===")
print(f"Number of samples: {len(bootstrap_25x5)}")
print(f"25x5 Matrix Variability:")
print(f"  Mean absolute difference from bootstrap mean: {diff_from_mean_25x5.mean():.4f}")
print(f"  Average standard deviation across entries: {std_matrix_25x5.mean():.4f}")
print(f"  Most variable transition: {np.unravel_index(std_matrix_25x5.argmax(), std_matrix_25x5.shape)}")
print(f"  Max standard deviation: {std_matrix_25x5.max():.4f}")
print(f"49x7 Matrix Variability:")
print(f"  Average standard deviation across entries: {std_matrix_49x7.mean():.4f}")
print(f"  Max standard deviation: {std_matrix_49x7.max():.4f}")
print()

# Chi-Squared Test
print("=== Chi-Squared Tests ===")
print(f"25x5 Matrix (Observed vs Predicted):")
print(f"  Chi-squared statistic: {chi2_25x5:.4f}")
print(f"  Degrees of freedom: {df_25x5}")
print(f"  P-value: {p_val_25x5:.6f}")
print(f"  Number of cells tested: {n_cells_25x5}")
print(f"  Result: Difference is {significance_25x5}")
print()
print(f"49x7 Matrix (Observed vs Predicted):")
print(f"  Chi-squared statistic: {chi2_49x7:.4f}")
print(f"  Degrees of freedom: {df_49x7}")
print(f"  P-value: {p_val_49x7:.6f}")
print(f"  Number of cells tested: {n_cells_49x7}")
print(f"  Result: Difference is {significance_49x7}")
print()
print(f"\nTrain vs Test 25x5 Matrix:")
print(f"  Interpretation: {'Overfitting likely' if p_val_test_25x5 < 0.05 else 'Model generalizes well'}")
print()

# Missing Transitions
print("=== Missing Events Analysis ===")
print()
print(f"25x5 Matrix - Transitions that never occurred in training data:")
print(f"Total missing: {len(missing_main)} out of {n_possible_main} possible")
for state_pair, next_state in missing_main:
    print(f"  ({state_pair[0]}, {state_pair[1]}) → {next_state}")
print()
print(f"Categorisation:")
print(f"  Probably impossible: {len(impossible_transitions)}")
for state_pair, next_state in impossible_transitions:
    print(f"    ({state_pair[0]}, {state_pair[1]}) → {next_state}")
print(f"  Rare but possible: {len(rare_transitions)}")
for state_pair, next_state in rare_transitions[:5]:  # Show first 5
    print(f"    ({state_pair[0]}, {state_pair[1]}) → {next_state}")
if len(rare_transitions) > 5:
    print(f"    ... and {len(rare_transitions)-5} more")    
print()
print(f"\n49x7 Matrix - Missing transitions: {len(missing_all)} out of {n_possible_all}")
print()
print(f"\n=== Do Simulations Predict Missing Events? ===")
print(f"Events that were missing in training but predicted by simulations: {len(predicted_events)}")
for state_pair, next_state, prob in predicted_events:
    print(f"  ({state_pair[0]}, {state_pair[1]}) → {next_state}: {prob:.4f}")
if len(predicted_events) == 0:
    print("  None - simulations only predict events that were observed in training")
print()
print(f"Events that were missing in training but predicted by 49x7 simulations: {len(predicted_events_49x7)}")
for state_pair, next_state, prob in predicted_events_49x7[:10]:  # Show first 10
    print(f"  ({state_pair[0]}, {state_pair[1]}) → {next_state}: {prob:.4f}")
if len(predicted_events_49x7) > 10:
    print(f"  ... and {len(predicted_events_49x7)-10} more")
if len(predicted_events_49x7) == 0:
    print("  None - simulations only predict events observed in training")
print()

# Hellinger Distances
print("=== Summary of Hellinger Distances ===")
print()
for method_name, res in hellinger_results.items():
    print(f"Method: {method_name}")
    print(f"  Care Period Lengths: {res['lengths']:.4f}")
    
    # Print individual category distances
    for category, h_dist in res['visits'].items():
        print(f"  {category}: {h_dist:.4f}")
    
    # Print average distance
    print(f"  Average Hellinger Distance (Category Visits): {res['average_visits']:.4f}")
    print()