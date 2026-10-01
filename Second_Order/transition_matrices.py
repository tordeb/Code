# --------------------------------------------------------------------------------------------------------------------------------------
# Observed Transition Probability Matrices (from training data)
# --------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from config import main_categories, all_categories
from data_loader import df_train, df_test, train_ids, test_ids


# --------------------------------------------------------------------------------------------------------------------------------------
# Transition Matrices
# --------------------------------------------------------------------------------------------------------------------------------------
# Formula P(x_k | x_i, x_j) = x_ijk / Σ(m=1 to M) x_ijm
# 
# Where:
# P(x_k | x_i, x_j) = Probability of transitioning to state k, given previous state i and current state j
# x_ijk = Number of observed transitions from i to j to k
# Σ(m=1 to M) x_ijm = Total transitions from state pair (i, j) to any state m

def calculate_transition_matrix(data, prev_col, curr_col, next_col, categories):
    """
    Calculate second-order transition probability matrices using:
    P(x_k | x_i, x_j) = x_ijk / Σ(m=1 to M) x_ijm
     
    Parameters:
    data: DataFrame with transition data
    prev_col: Column name for previous state (i)
    curr_col: Column name for current state (j)
    next_col: Column name for next state (k)
    categories: List of category names

    Returns: 
    - Second-order transition probability matrix
    """
    # Create state pairs (i, j)
    data = data.copy()

    # Count transitions from (i, j) to k 
    counts = data.groupby([prev_col, curr_col, next_col]).size().unstack(fill_value=0)

    # Sum each row (Σ(m=1 to M) x_ijm) 
    sum_row = counts.sum(axis=1)

    # Divide counts by row totals (P(x_k | x_i, x_j)) 
    probabilities = counts.div(sum_row, axis=0)

    # Reorder rows and columns 
    probabilities = probabilities.reindex(columns=categories, fill_value=0)

    # Fill missing values with 0 
    probabilities = probabilities.fillna(0)  
     
    return probabilities


# ----------------------------------
# Prepare Data for Second-Order
# ----------------------------------
# Need three columns: previous state, current state, next state
# Group by ActivityID for individual chains

# Add previous category column
df_train["Previous_Category"] = df_train.groupby("ActivityID")["SurfaceCategories"].shift(1)
# Add next category column
df_train["Next_Category"] = df_train.groupby("ActivityID")["SurfaceCategories"].shift(-1) 

# Clean training data to remove rows where there's no next category (from "Out" to "nothing") (NaN values)
df_train_clean = df_train.dropna(subset=["Previous_Category", "Next_Category"])

# Filter for 25x5 matrix (i.e. remove 'in' and 'out' states in training data)
df_train_main = df_train_clean[
    (df_train_clean["Previous_Category"].isin(main_categories)) &
    (df_train_clean["SurfaceCategories"].isin(main_categories)) &
    (df_train_clean["Next_Category"].isin(main_categories))
]   


# ----------------------------------
# Calculate the Matrices
# ----------------------------------
observed_main = calculate_transition_matrix(
    df_train_main, "Previous_Category", "SurfaceCategories", "Next_Category", main_categories
)

# Observed 49x7 matrix
observed_all = calculate_transition_matrix(
    df_train_clean, "Previous_Category", "SurfaceCategories", "Next_Category", all_categories
)


# --------------------------------------------------------------------------------------------------------------------------------------
# Starting State Pair Probabilities (from training data)
# --------------------------------------------------------------------------------------------------------------------------------------
# Select starting state pair using weighted random choice

# Get First Two Rows of Each ActivityID
def get_starting_pairs(df, activity_ids, categories=None):
    """
    Get the first two states of each care period as a starting pair
    """
    starting_pairs = []

    for activity_id in activity_ids:
        activity_data = df[df["ActivityID"] == activity_id]["SurfaceCategories"].values

        if len(activity_data) >= 2:
            first_state = activity_data[0]
            second_state = activity_data[1]

            # Filter by categories
            if categories is None:
                starting_pairs.append((first_state, second_state))
            elif first_state in categories and second_state in categories:
                starting_pairs.append((first_state, second_state))
    
    return starting_pairs

# Count how often each pair is first, filtering for main and all categories
train_first_main = get_starting_pairs(df_train, train_ids, main_categories)
train_first_all = get_starting_pairs(df_train, train_ids)

# Count and convert to probabilities
def pairs_to_probabilities(pairs):
    """
    Convert list of pairs to probability distribution
    """
    pair_counts = pd.Series(pairs).value_counts(normalize=True)
    return pair_counts

train_start_probs_main = pairs_to_probabilities(train_first_main)
train_start_probs_all = pairs_to_probabilities(train_first_all)


# --------------------------------------------------------------------------------------------------------------------------------------
# 'Test' Transition Matrices
# --------------------------------------------------------------------------------------------------------------------------------------
# Transition matrices formed of just the test data (final 20%) for comparison.
# Follows same process as for the first 80%.

# Calculate matrix from test data
# Need three columns: previous state, current state, next state
df_test["Previous_Category"] = df_test.groupby("ActivityID")["SurfaceCategories"].shift(1)
df_test["Next_Category"] = df_test.groupby("ActivityID")["SurfaceCategories"].shift(-1)

# Clean test data to remove rows where there's no previous or next category (NaN values)
df_test_clean = df_test.dropna(subset=["Previous_Category", "Next_Category"])

# Filter for 25x5 matrix
df_test_main = df_test_clean[
    (df_test_clean["Previous_Category"].isin(main_categories)) &
    (df_test_clean["SurfaceCategories"].isin(main_categories)) &
    (df_test_clean["Next_Category"].isin(main_categories))
]

# For 25x5 matrix
test_matrix_main = calculate_transition_matrix(
    df_test_main, "Previous_Category", "SurfaceCategories", "Next_Category", main_categories
)

# For 49x7 matrix
test_matrix_all = calculate_transition_matrix(
    df_test_clean, "Previous_Category", "SurfaceCategories", "Next_Category", all_categories
)


# --------------------------------------------------------------------------------------------------------------------------------------
# Convergence Analysis
# --------------------------------------------------------------------------------------------------------------------------------------
# Shows how transition probabilities change with more training data.
# Helps to determine amount of training data required for stable estimates.

def convergence_analysis(df_train, train_ids, categories, target_transition=None):
    """
    Analyse how transition probabilities converge as training data increases.

    For second-order: tracks P(next | previous, current)
    Shows: P(target_transition) vs Number of Training Episodes

    Parameters:
    df_train: DataFrame containing the training data
    train_ids: list of training episode IDs
    categories: list of surface categories to include
    target_transition: optional transition to track, for example ("Equipment", "Patient")

    Returns:
    - convergence_results: a list containing the estimated probability at different training sample sizes
    - target_transition: the transition that was analysed
    """
    # Sample sizes to test (tailored to current data)
    max_episodes = len(train_ids)
    step = max(1, max_episodes // 5)
    sample_sizes = list(range(step, max_episodes, step))
    if max_episodes not in sample_sizes:
        sample_sizes.append(max_episodes)
    
    # If no specific transition specified, pick an important one
    # For second-order: (previous, current, next)
    if target_transition is None:
        if "Equipment" in categories and "Patient" in categories and "HygieneArea" in categories:
            target_transition = ("HygieneArea", "Equipment", "Patient")
        else:
            target_transition = (categories[0], categories[1], categories[0])
    
    convergence_results = []
    
    for size in sample_sizes:
        # Use first 'size' episodes (chronological order)
        subset_ids = train_ids[:size]
        subset_data = df_train[df_train["ActivityID"].isin(subset_ids)].copy()
        
        # Prepare data (same as in main analysis)
        subset_data["Previous_Category"] = subset_data.groupby("ActivityID")["SurfaceCategories"].shift(1)
        subset_data["Next_Category"] = subset_data.groupby("ActivityID")["SurfaceCategories"].shift(-1)
        subset_clean = subset_data.dropna(subset=["Previous_Category", "Next_Category"])
        
        # Filter for categories of interest (three columns)
        subset_filtered = subset_clean[
            (subset_clean["Previous_Category"].isin(categories)) &
            (subset_clean["SurfaceCategories"].isin(categories)) &
            (subset_clean["Next_Category"].isin(categories))
        ]
        
        if len(subset_filtered) > 0:
            # Calculate transition matrix
            subset_matrix = calculate_transition_matrix(
                subset_filtered, "Previous_Category", "SurfaceCategories", "Next_Category", categories
            )
            
            # Extract target probability using state pair
            state_pair = (target_transition[0], target_transition[1])
            next_state = target_transition[2]
            
            if state_pair in subset_matrix.index and next_state in subset_matrix.columns:
                target_prob = subset_matrix.loc[state_pair].get(next_state, 0) \
                if state_pair in subset_matrix.index else 0
            else:
                target_prob = 0
            
            # Count total transitions from this state pair
            total_transitions = len(subset_filtered[
                (subset_filtered["Previous_Category"] == target_transition[0]) &
                (subset_filtered["SurfaceCategories"] == target_transition[1])
            ])
            
            # Count target transition occurrences (all three states matching)
            target_count = len(subset_filtered[
                (subset_filtered["Previous_Category"] == target_transition[0]) &
                (subset_filtered["SurfaceCategories"] == target_transition[1]) &
                (subset_filtered["Next_Category"] == target_transition[2])
            ])
        else:
            target_prob = 0
            total_transitions = 0
            target_count = 0
        
        convergence_results.append({
            'sample_size': size,
            'probability': target_prob,
            'total_transitions': total_transitions,
            'target_count': target_count,
            'from_count': total_transitions
        })
    
    return convergence_results, target_transition

# Run convergence analysis for main categories
convergence_main, target_main = convergence_analysis(
    df_train, train_ids, main_categories
)

# Run convergence analysis for all categories
convergence_all, target_all = convergence_analysis(
    df_train, train_ids, all_categories, ("In", "FarPatient", "Equipment")
)

# Confidence interval
def wilson_ci(k, n, z=1.96):
    """95% confidence interval for k successes out of n (same method as Table 5)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half

# Assess stability
def assess_stability(results, max_rel_width=0.5, max_drift=0.10, min_count=5):
    """Decides stable/moderate/unstable/insufficient data. Looks at the final estimate (all training episodes)."""
    if len(results) < 2:
        return "INSUFFICIENT DATA"

    final = results[-1]
    p = final['probability']       # the estimate
    k = final['target_count']      # times the transition happened
    n = final['from_count']        # times the starting pair occurred

    # Check 1: seen often enough?
    if p == 0 or k < min_count:
        return f"INSUFFICIENT DATA (observed {k} times)"

    # Check 2: is the uncertainty small compared to the estimate?
    lo, hi = wilson_ci(k, n)
    rel_width = (hi - lo) / p

    # Check 3: did the estimate stop moving at the end?
    drift = abs(p - results[-2]['probability']) / p

    detail = f"rel. CI width = {rel_width:.2f}, drift = {drift:.0%}"
    if rel_width < max_rel_width and drift < max_drift:
        return f"STABLE ({detail})"
    elif rel_width < 2 * max_rel_width:
        return f"MODERATE ({detail})"
    else:
        return f"UNSTABLE ({detail})"

# Estimate minimum required episodes
def estimate_minimum_episodes(results):
    """First episode count where the estimate becomes STABLE. None if it never does."""
    for i in range(2, len(results) + 1):
        if assess_stability(results[:i]).startswith("STABLE"):
            return results[i - 1]['sample_size']
    return None

stability_main = assess_stability(convergence_main)
stability_all = assess_stability(convergence_all)
min_episodes_main = estimate_minimum_episodes(convergence_main)
min_episodes_all = estimate_minimum_episodes(convergence_all)

# Is the data sufficient? Only if it actually became stable
sufficient_main = "YES" if min_episodes_main is not None else "NO"
sufficient_all = "YES" if min_episodes_all is not None else "NO"


# --------------------------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------------------------

# Transition Matrices
print("=== Transition Matrices ===")
print("=== Function Explanation ===")
print(calculate_transition_matrix.__doc__)
print()
print("=== Observed 25x5 Matrix (from training data) ===")
print(observed_main.round(3).to_string())
print()
print("=== Observed 49x7 Matrix (from training data) ===")
print(observed_all.round(3).to_string())
print()

# Starting State Pair Probabilities (from training data)
print("=== Starting Pair Probabilities (from training data) ===")
print("25x5:")
print(train_start_probs_main.round(3))
print()
print("49x7:")
print(train_start_probs_all.round(3))
print()

# 'Test' Transition Matrices
print("=== Test 25x5 Matrix using unseen data ===")
print(test_matrix_main.round(3).to_string())
print()
print("=== Test 49x7 Matrix using unseen data ===")
print(test_matrix_all.round(3).to_string())
print()

# Convergence Analysis
print("=== Convergence Analysis ===")
print()
print(f"Main Categories Convergence:")
print(f"Tracking transition (Main): {target_main[0]} → {target_main[1]} → {target_main[2]}")
for result in convergence_main:
    print(
        f"  {result['sample_size']:4d} episodes: "
        f"P = {result['probability']:.4f} "
        f"({result['target_count']:3d}/{result['total_transitions']:4d} transitions)")
print()
print(f"All Categories Convergence:")
print(f"Tracking transition (All)): {target_all[0]} → {target_all[1]} → {target_all[2]}")
for result in convergence_all:
    print(
        f"  {result['sample_size']:4d} episodes: "
        f"P = {result['probability']:.4f} "
        f"({result['target_count']:3d}/{result['from_count']:4d} transitions)")
print()
print(f"\n=== Convergence Assessment ===")
print(f"Main categories ({target_main[0]} → {target_main[1]} → {target_main[2]}): {stability_main}")
print(f"All categories ({target_all[0]} → {target_all[1]} → {target_all[2]}): {stability_all}")
print()
print(f"\nMinimum episodes for stability:")
print(f"  Main categories: {min_episodes_main if min_episodes_main else 'not reached'}")
print(f"  All categories: {min_episodes_all if min_episodes_all else 'not reached'}")
print(f"  Current dataset: {len(train_ids)} episodes")
print()
print(f"\nIs current data sufficient?")
print(f"  Main categories: {sufficient_main}")
print(f"  All categories: {sufficient_all}")
print()