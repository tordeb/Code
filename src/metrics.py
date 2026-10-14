# --------------------------------------------------------------------------------------------------------------------------------------
# Metrics for Comparison of Observed and Predicted
# --------------------------------------------------------------------------------------------------------------------------------------
# Every metric below is applied in exactly the same way to the first- and second-order models.


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

from config import ORDER, MAIN_LABEL, ALL_LABEL, main_categories, all_categories
from data_loader import df_train, train_ids
from transition_matrices import *
from simulations import *
from distances import hellinger_distance_histograms


# --------------------------------------------------------------------------------------------------------------------------------------
# Compare Observed vs Predicted
# --------------------------------------------------------------------------------------------------------------------------------------
# MAE uses calculate_mae_safe (transition_matrices.py): only histories observed in the training data are evaluated,
# and predicted values for missing transitions default to 0.

# Difference in observed and predicted transition matrices
difference_fixed_main = observed_main - align_to(observed_main, predicted_fixed_main)
difference_fitted_main = observed_main - align_to(observed_main, predicted_fitted_main)
difference_terminating_all = observed_all - align_to(observed_all, predicted_terminating_all)

# Overall error
mae_main = mae_fixed
mae_all = mae_terminating


# --------------------------------------------------------------------------------------------------------------------------------------
# Test Against Final 20% of Data
# --------------------------------------------------------------------------------------------------------------------------------------

# Compare observed matrix with test matrix
train_test_difference_main = observed_main - align_to(observed_main, test_matrix_main)
train_test_difference_all = observed_all - align_to(observed_all, test_matrix_all)

# Overall error
train_test_mae_main = calculate_mae_safe(observed_main, test_matrix_main)
train_test_mae_all = calculate_mae_safe(observed_all, test_matrix_all)


# --------------------------------------------------------------------------------------------------------------------------------------
# DNA Sequence Comparisons
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


def summarise_scores(scores, normalised, time_sim, time_calc, n_simulated):
    """Summary statistics shared by the Levenshtein, Needleman-Wunsch and Smith-Waterman results."""
    mu_abs, sigma_abs = stats.norm.fit(scores)
    mu_norm, sigma_norm = stats.norm.fit(normalised)

    return {
        'observed_count': len(observed_chains),
        'simulated_count': n_simulated,
        'total_comparisons': len(scores),
        'time_sim': time_sim,
        'time_calc': time_calc,
        'mean_abs': np.mean(scores),
        'median_abs': np.median(scores),
        'mode_abs': stats.mode(scores, keepdims=False)[0],
        'sigma_abs': sigma_abs,
        'min_abs': scores.min(),
        'max_abs': scores.max(),
        'skew_abs': stats.skew(scores),
        'mu_abs': mu_abs,
        'mean_norm': np.mean(normalised),
        'median_norm': np.median(normalised),
        'mode_norm': stats.mode(normalised, keepdims=False)[0],
        'sigma_norm': sigma_norm,
        'min_norm': normalised.min(),
        'max_norm': normalised.max(),
        'skew_norm': stats.skew(normalised),
        'mu_norm': mu_norm,
        # Raw values, used for plotting
        'scores': scores,
        'normalised': normalised
    }


def compare_all_pairs(method_chains, score_function):
    """
    Compare every observed chain with every simulated chain.

    score_function(obs_str, sim_str) returns (absolute score, normalised score).
    """
    start_time_sim = time.time()
    simulated_chains = [chain_to_string(chain) for chain in method_chains]
    time_sim = time.time() - start_time_sim

    start_time_calc = time.time()
    scores = []
    normalised = []
    for obs_str in observed_chains:
        for sim_str in simulated_chains:
            score, norm_score = score_function(obs_str, sim_str)
            scores.append(score)
            normalised.append(norm_score)
    time_calc = time.time() - start_time_calc

    return summarise_scores(np.array(scores), np.array(normalised), time_sim, time_calc, len(simulated_chains))


# --------------------------------------------------------------------------------------------------------------------------------------
# Levenshtein Distance
# --------------------------------------------------------------------------------------------------------------------------------------
# Absolute and normalised edit distance
# Measures how many 'edits' are needed to move from one sequence to the next

def levenshtein_scores(obs_str, sim_str):
    # Normalised distance (ratio gives similarity, so 1 - ratio = normalised distance)
    return distance(obs_str, sim_str), 1 - ratio(obs_str, sim_str)


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


def needleman_wunsch_scores(obs_str, sim_str):
    _, _, score = needleman_wunsch(obs_str, sim_str)
    # Normalised score (1.0 = perfect match)
    max_possible = min(len(obs_str), len(sim_str)) * 1  # match=1
    return score, (score / max_possible if max_possible > 0 else 0)


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
    aligned_x, aligned_y, score
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


def smith_waterman_scores(obs_str, sim_str):
    _, _, score = smith_waterman(obs_str, sim_str)
    # Normalised score
    max_possible_local = min(len(obs_str), len(sim_str)) * 2  # match=2
    return score, (score / max_possible_local if max_possible_local > 0 else 0)


# ----------------------------------
# Run all three comparisons for all three simulation methods
# ----------------------------------

levenshtein_results = {}
nw_results = {}
sw_results = {}

for method_name, method_data in simulation_methods.items():
    levenshtein_results[method_name] = compare_all_pairs(method_data["chains"], levenshtein_scores)
    nw_results[method_name] = compare_all_pairs(method_data["chains"], needleman_wunsch_scores)
    sw_results[method_name] = compare_all_pairs(method_data["chains"], smith_waterman_scores)


# --------------------------------------------------------------------------------------------------------------------------------------
# Sensitivity Analysis
# --------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------
# Matrix Perturbation
# --------------------------------------------------------------------------------------------------------------------------------------
# Test if small errors in the transition matrix completely change results
# If the model is robust, small changes should not matter

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
n_chains_per_test = 100


def sensitivity_analysis(observed_matrix, predicted_matrix, start_probs, categories, simulation_kwargs):
    """
    Perturb the observed matrix, re-run simulations, and compare the new predicted matrix with the original one.
    Uses calculate_mae_safe, the same MAE as everywhere else.
    """
    sensitivity_results = []

    for noise_level in noise_levels:
        mae_values = []

        for test in range(n_tests_per_level):
            # Perturb the observed matrix
            perturbed_lookup = build_lookup(perturb_matrix(observed_matrix, noise_level))

            # Run simulations with perturbed matrix
            chains = [
                simulate(perturbed_lookup, start_probs, categories, **simulation_kwargs)
                for _ in range(n_chains_per_test)
            ]

            # Calculate predicted matrix
            predicted_perturbed = calculate_transition_matrix(
                chains_to_transitions(chains), HISTORY_COLS, NEXT_COL, categories
            )

            # Compare to original predicted matrix
            mae_values.append(calculate_mae_safe(predicted_matrix, predicted_perturbed))

        # Store results
        sensitivity_results.append({
            'noise_level': noise_level,
            'mean_mae': np.mean(mae_values),
            'std_mae': np.std(mae_values)
        })

    return sensitivity_results


# Main categories (fixed length) and all categories (terminating state)
sensitivity_results_main = sensitivity_analysis(
    observed_main, predicted_fixed_main, train_start_probs_main, main_categories, {"length": avg_length}
)
sensitivity_results_all = sensitivity_analysis(
    observed_all, predicted_terminating_all, train_start_probs_all, all_categories, {"stop_state": "Out"}
)


# ----------------------------------
# Comparison of Main and All Sensitivity
# ----------------------------------
# Which model is more robust.

comparison_results = []

for i, noise_level in enumerate(noise_levels):
    mae_main_noise = sensitivity_results_main[i]['mean_mae']
    mae_all_noise = sensitivity_results_all[i]['mean_mae']

    more_robust = MAIN_LABEL if mae_main_noise < mae_all_noise else ALL_LABEL
    if abs(mae_main_noise - mae_all_noise) < 0.001:
        more_robust = "Similar"

    comparison_results.append({
        'noise_level': noise_level,
        'mae_main': mae_main_noise,
        'mae_all': mae_all_noise,
        'more_robust': more_robust
    })

# Overall robustness assessment
final_mae_main = sensitivity_results_main[-1]['mean_mae']
final_mae_all = sensitivity_results_all[-1]['mean_mae']

if abs(final_mae_main - final_mae_all) < 0.01:
    conclusion = "Both models show similar sensitivity to noise"
elif final_mae_main < final_mae_all:
    conclusion = f"{MAIN_LABEL} model is more robust (less sensitive to noise)"
else:
    conclusion = f"{ALL_LABEL} model is more robust (less sensitive to noise)"


# --------------------------------------------------------------------------------------------------------------------------------------
# Bootstrap Analysis
# --------------------------------------------------------------------------------------------------------------------------------------
# Randomly sample training data multiple times to see how much the transition matrix would change if different data was collected
# Also known as 'Data Resampling'
#
# Every bootstrap matrix is placed on the full grid of histories (5 / 7 rows for first-order, 25 / 49 rows for
# second-order) so that matrices from different samples line up.

def bootstrap_transition_matrix(df_train, sample_fraction=0.8, n_bootstrap=100):
    """
    Create multiple transition matrices from random samples of training data.

    This simulates 'what if we had different observations?'
    """
    activity_ids = df_train["ActivityID"].unique()
    matrices_main = []
    matrices_all = []

    for i in range(n_bootstrap):
        # Random sample of care episodes
        sample_size = int(len(activity_ids) * sample_fraction)
        sample_ids = np.random.choice(activity_ids, size=sample_size, replace=False)

        # Create sample dataset
        sample_data = df_train[df_train["ActivityID"].isin(sample_ids)]

        # Main categories matrix
        sample_main = build_transitions(sample_data, main_categories)
        if len(sample_main) > 0:
            matrix_main = calculate_transition_matrix(sample_main, HISTORY_COLS, NEXT_COL, main_categories)
            matrices_main.append(
                matrix_main.reindex(index=full_history_index(main_categories), fill_value=0).values
            )

        # All categories matrix
        sample_all = build_transitions(sample_data)
        if len(sample_all) > 0:
            matrix_all = calculate_transition_matrix(sample_all, HISTORY_COLS, NEXT_COL, all_categories)
            matrices_all.append(
                matrix_all.reindex(index=full_history_index(all_categories), fill_value=0).values
            )

    return np.stack(matrices_main), np.stack(matrices_all)


# Run bootstrap analysis
bootstrap_main, bootstrap_all = bootstrap_transition_matrix(df_train)

# Calculate statistics
mean_matrix_main = np.mean(bootstrap_main, axis=0)
std_matrix_main = np.std(bootstrap_main, axis=0)

mean_matrix_all = np.mean(bootstrap_all, axis=0)
std_matrix_all = np.std(bootstrap_all, axis=0)

# Compare to original - reindex to full grid to match bootstrap mean shape
original_main = observed_main.reindex(index=full_history_index(main_categories), fill_value=0).values
diff_from_mean_main = np.abs(original_main - mean_matrix_main)

original_all = observed_all.reindex(index=full_history_index(all_categories), fill_value=0).values
diff_from_mean_all = np.abs(original_all - mean_matrix_all)


def most_variable_transition(std_matrix, categories):
    """Label of the matrix entry with the largest bootstrap standard deviation, e.g. 'Patient → Equipment'."""
    row, col = np.unravel_index(std_matrix.argmax(), std_matrix.shape)
    history = full_history_index(categories)[row]
    return f"{format_history(history)} → {categories[col]}"


most_variable_main = most_variable_transition(std_matrix_main, main_categories)
most_variable_all = most_variable_transition(std_matrix_all, all_categories)


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


def significance_label(p_value):
    if p_value < 0.001:
        return "highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "very significant (p < 0.01)"
    elif p_value < 0.05:
        return "significant (p < 0.05)"
    return "not significant (p ≥ 0.05)"


def chi_squared_result(observed_matrix, predicted_matrix, matrix_name):
    chi2, p_val, dof, n_cells = chi_squared_test_matrices(observed_matrix, predicted_matrix, matrix_name)
    return {'chi2': chi2, 'p_value': p_val, 'df': dof, 'n_cells': n_cells, 'significance': significance_label(p_val)}


# Observed vs predicted
chi2_main = chi_squared_result(observed_main, predicted_fixed_main, MAIN_LABEL)
chi2_all = chi_squared_result(observed_all, predicted_terminating_all, ALL_LABEL)

# Test train vs test matrices for validation
chi2_train_test_main = chi_squared_result(observed_main, test_matrix_main, f"Train vs Test {MAIN_LABEL}")


# --------------------------------------------------------------------------------------------------------------------------------------
# Missing Transitions
# --------------------------------------------------------------------------------------------------------------------------------------
# Finding transitions that never occurred in the training data but might be possible in real life.
# Checked over every possible history (5 x 5 = 25 cells first-order, 25 x 5 = 125 cells second-order), so histories
# that were never observed at all are counted as missing too.

def analyse_missing_transitions(observed_matrix, categories, matrix_name=""):
    """
    Find transitions that never occurred in training data.

    These might be:
    1. Impossible transitions (e.g., Patient → Patient without leaving)
    2. Very rare but possible transitions
    3. Very plausible but missing due to insufficient data

    Returns:
    - a list of missing (history, next state) transitions
    - the number of possible transitions
    - the number of histories never observed at all
    """
    full_matrix = observed_matrix.reindex(index=full_history_index(categories), columns=categories, fill_value=0)

    zero_transitions = []
    for history, row in zip(full_matrix.index, full_matrix.values):
        for next_state, probability in zip(full_matrix.columns, row):
            if probability == 0:
                zero_transitions.append((history, next_state))

    unobserved_histories = int((full_matrix.sum(axis=1) == 0).sum())

    return zero_transitions, full_matrix.size, unobserved_histories


def predicted_missing_events(missing, predicted_matrix):
    """Transitions missing from the training data that the simulations still produced."""
    events = []
    for history, next_state in missing:
        if history in predicted_matrix.index:
            prob = predicted_matrix.loc[[history], next_state].iloc[0]
            if prob > 0:
                events.append((history, next_state, prob))
    return events


# Main categories matrix
missing_main, n_possible_main, unobserved_histories_main = analyse_missing_transitions(observed_main, main_categories)

# Categorise missing transitions
impossible_transitions = []
rare_transitions = []

for history, next_state in missing_main:
    states = (history if isinstance(history, tuple) else (history,)) + (next_state,)
    # Define what seems impossible vs. just rare
    if all(state == "Patient" for state in states):
        impossible_transitions.append((history, next_state))
    else:
        # If it's not in the impossible list, it's rare but possible
        # (e.g. HygieneArea → Patient: should wash hands first)
        rare_transitions.append((history, next_state))

# All categories matrix
missing_all, n_possible_all, unobserved_histories_all = analyse_missing_transitions(observed_all, all_categories)

# Check if simulations predict any of these missing events
predicted_events = predicted_missing_events(missing_main, predicted_fixed_main)
predicted_events_all = predicted_missing_events(missing_all, predicted_terminating_all)


# --------------------------------------------------------------------------------------------------------------------------------------
# Hellinger Distances
# --------------------------------------------------------------------------------------------------------------------------------------

# Get lengths from real data
real_lengths = df_train.groupby("ActivityID").size().values

# Dictionary to store Hellinger results
hellinger_results = {}

for method_name, method_data in simulation_methods.items():
    method_chains = method_data["chains"]

    # Get lengths from simulated data
    simulated_lengths = [len(chain) for chain in method_chains]

    # Calculate Hellinger Distance for lengths
    h_dist_lengths = hellinger_distance_histograms(real_lengths, simulated_lengths)

    # Calculate Hellinger Distance for category visits
    hellinger_distance_visits = {}
    all_real_visits = {}
    all_sim_visits = {}

    for category in method_data["categories"]:
        # Count visits in real data
        real_visits = []
        for activity_id in train_ids:
            activity_data = df_train[df_train["ActivityID"] == activity_id]
            count = (activity_data["SurfaceCategories"] == category).sum()
            real_visits.append(count)
        all_real_visits[category] = real_visits

        # Count visits in simulated data
        sim_visits = [chain.count(category) for chain in method_chains]
        all_sim_visits[category] = sim_visits

        # Calculate Hellinger Distance
        hellinger_distance_visits[category] = hellinger_distance_histograms(real_visits, sim_visits)

    # Average Hellinger Distance
    avg_h_dist = np.mean(list(hellinger_distance_visits.values()))

    # Store results in the dictionary
    hellinger_results[method_name] = {
        'lengths': h_dist_lengths,
        'visits': hellinger_distance_visits,
        'average_visits': avg_h_dist,
        # Raw values, used for plotting
        'real_lengths': real_lengths,
        'simulated_lengths': simulated_lengths,
        'real_visits': all_real_visits,
        'sim_visits': all_sim_visits
    }
