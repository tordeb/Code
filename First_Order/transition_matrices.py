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
# Formula P(x_j | x_i) = x_ij / Σ(m=1 to M) x_im
# 
# Where:
# P(x_j | x_i) = Probability of transitioning to state j, given current state i
# x_ij = Number of observed transitions from i to j
# Σ(m=1 to M) x_im = Total transitions from state i to any state m

def calculate_transition_matrix(data, from_col, to_col, categories):
    """
    Calculate first-order transition probability matrices using:
    P(x_j | x_i) = x_ij / Σ(m=1 to M) x_im
     
    Parameters:
    data: DataFrame with transition data
    from_col: Column name for current state
    to_col: Column name for next state
    categories: List of category names

    Returns: 
    - First-order transition probability matrix
    """
    # Count transitions (x_ij)
    counts = pd.crosstab(data[from_col], data[to_col])

    # Sum each row (Σ(m=1 to M) x_im)
    sum_row = counts.sum(axis=1)

    # Divide counts by row totals (P(x_ij))
    probabilities = counts.div(sum_row, axis=0)

    # Reorder rows and columns
    probabilities = probabilities.reindex(index=categories, columns=categories)

    # Fill missing values with 0
    probabilities = probabilities.fillna(0)

    return probabilities


# ----------------------------------
# Prepare Data for First-Order
# ----------------------------------
# Group by ActivityID for individual chains

# Add next category column
# shift(-1) = look at following row to see what the following state is
df_train["Next_Category"] = df_train.groupby("ActivityID")["SurfaceCategories"].shift(-1)

# Clean training data to remove rows where there's no next category (from "Out" to "nothing") (NaN values)
df_train_clean = df_train.dropna(subset=["Next_Category"])

# Filter for 5x5 matrix (i.e. remove 'in' and 'out' states in training data)
df_train_main = df_train_clean[
    (df_train_clean["SurfaceCategories"].isin(main_categories)) &
    (df_train_clean["Next_Category"].isin(main_categories))
]  


# ----------------------------------
# Calculate the Matrices
# ----------------------------------
# Observed 5x5 matrix
observed_main = calculate_transition_matrix(
    df_train_main, "SurfaceCategories", "Next_Category", main_categories
)

# Observed 7x7 matrix
observed_all = calculate_transition_matrix(
    df_train_clean, "SurfaceCategories", "Next_Category", all_categories
)


# --------------------------------------------------------------------------------------------------------------------------------------
# Starting State Probabilities (from training data)
# --------------------------------------------------------------------------------------------------------------------------------------
# Select starting state using weighted random choice

# Count how often each category is the first state
train_first_rows = df_train.groupby("ActivityID").first()

# For 5x5 filter to main categories
train_first_main = train_first_rows[train_first_rows["SurfaceCategories"].isin(main_categories)]

# Count and convert to probabilities
train_start_probs_main = train_first_main["SurfaceCategories"].value_counts(normalize=True)

# For 7x7, all categories
train_start_probs_all = train_first_rows["SurfaceCategories"].value_counts(normalize=True)


# --------------------------------------------------------------------------------------------------------------------------------------
# 'Test' Transition Matrices
# --------------------------------------------------------------------------------------------------------------------------------------
# Transition matrices formed of just the test data (final 20%) for comparison.
# Follows same process as for the first 80%.

# Calculate matrix from test data
df_test["Next_Category"] = df_test.groupby("ActivityID")["SurfaceCategories"].shift(-1)
df_test_clean = df_test.dropna(subset=["Next_Category"])

# Filter for 25x5 matrix
df_test_main = df_test_clean[
    (df_test_clean["SurfaceCategories"].isin(main_categories)) &
    (df_test_clean["Next_Category"].isin(main_categories))
]

# For 5x5 matrix
test_matrix_main = calculate_transition_matrix(
    df_test_main, "SurfaceCategories", "Next_Category", main_categories
)

# For 7x7 matrix
test_matrix_all = calculate_transition_matrix(
    df_test_clean, "SurfaceCategories", "Next_Category", all_categories
)



# --------------------------------------------------------------------------------------------------------------------------------------
# Convergence Analysis
# --------------------------------------------------------------------------------------------------------------------------------------
# Shows how transition probabilities change with more training data
# Helps to determine amount of training data required for stable estimates

def convergence_analysis(df_train, train_ids, categories, target_transition=None):
    """
    Analyses how transition probabilities converge as training data increases.
    Shows: P(target_transition) vs Number of Training Episodes.

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
    if target_transition is None:
        if "Equipment" in categories and "Patient" in categories:
            target_transition = ("Equipment", "Patient")
        else:
            target_transition = (categories[0], categories[1])
    
    convergence_results = []
    
    for size in sample_sizes:
        # Use first 'size' episodes (chronological order)
        subset_ids = train_ids[:size]
        subset_data = df_train[df_train["ActivityID"].isin(subset_ids)].copy()
        
        # Prepare data (samE as in main analysis)
        subset_data["Next_Category"] = subset_data.groupby("ActivityID")["SurfaceCategories"].shift(-1)
        subset_clean = subset_data.dropna(subset=["Next_Category"])
        
        # Filter for categories of interest
        subset_filtered = subset_clean[
            (subset_clean["SurfaceCategories"].isin(categories)) &
            (subset_clean["Next_Category"].isin(categories))
        ]
        
        if len(subset_filtered) > 0:
            # Calculate transition matrix
            subset_matrix = calculate_transition_matrix(
                subset_filtered, "SurfaceCategories", "Next_Category", categories
            )
            
            # Extract target probability
            target_prob = subset_matrix.loc[target_transition[0], target_transition[1]]
            
            # Count total transitions
            total_transitions = len(subset_filtered)
            
            # Count target transition occurrences
            target_count = len(subset_filtered[
                (subset_filtered["SurfaceCategories"] == target_transition[0]) &
                (subset_filtered["Next_Category"] == target_transition[1])
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
            'episodes': len(subset_ids)
        })
                
    return convergence_results, target_transition

# Run convergence analysis for main categories
convergence_main, target_main = convergence_analysis(
    df_train, train_ids, main_categories
)

# Run convergence analysis for all categories
convergence_all, target_all = convergence_analysis(
    df_train, train_ids, all_categories, ("In", "Equipment")
)

# Assess stability
def assess_stability(results, threshold=0.01):
    """Check if probability has stabilized."""
    if len(results) < 3:
        return "Insufficient data"
    
    last_three_probs = [r['probability'] for r in results[-3:]]
    std_dev = np.std(last_three_probs)
    
    if std_dev < threshold:
        return f"STABLE (σ = {std_dev:.4f})"
    else:
        return f"UNSTABLE (σ = {std_dev:.4f})"

# Estimate minimum required episodes
def estimate_minimum_episodes(results, stability_threshold=0.01):
    """Estimate minimum episodes needed for stable estimates."""
    for i in range(2, len(results)):
        last_three_probs = [r['probability'] for r in results[i-2:i+1]]
        if np.std(last_three_probs) < stability_threshold:
            return results[i]['sample_size']
    return results[-1]['sample_size']  # Needs more data

stability_main = assess_stability(convergence_main)
stability_all = assess_stability(convergence_all)
min_episodes_main = estimate_minimum_episodes(convergence_main)
min_episodes_all = estimate_minimum_episodes(convergence_all)

# Is current data sufficient
sufficient_main = "YES" if len(train_ids) >= min_episodes_main else "NO"
sufficient_all = "YES" if len(train_ids) >= min_episodes_all else "NO"


# --------------------------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------------------------

# Transition Matrices
print("=== Transition Matrices ===")
print("=== Function Explanation ===")
print(calculate_transition_matrix.__doc__)
print()
print("=== Observed 5x5 Matrix (from training data) ===")
print(observed_main.round(3).to_string())
print()
print("=== Observed 7x7 Matrix (from training data) ===")
print(observed_all.round(3).to_string())
print()

# Starting State Probabilities (from training data)
print("=== Starting Probabilities (from training data) ===")
print("5x5:")
print(train_start_probs_main.round(3))
print()
print("7x7:")
print(train_start_probs_all.round(3))
print()

# 'Test' Transition Matrices
print("=== Test 5x5 Matrix using unseen data ===")
print(test_matrix_main.round(3).to_string())
print()
print("=== Test 7x7 Matrix using unseen data ===")
print(test_matrix_all.round(3).to_string())
print()

# Convergence Analysis
print(f"=== Convergence Analysis ===")
print(f"Tracking transition (Main): {target_main[0]} → {target_main[1]}")
print(f"Tracking transition (All): {target_all[0]} → {target_all[1]}")
print()
print(f"Main Categories Convergence:")
for result in convergence_main:
    print(
        f"  {result['episodes']:4d} episodes: "
        f"P = {result['probability']:.4f} "
        f"({result['target_count']:3d}/{result['total_transitions']:4d} transitions)"
    )
print()
print(f"All Categories Convergence:")
for result in convergence_all:
    print(
        f"  {result['episodes']:4d} episodes: "
        f"P = {result['probability']:.4f} "
        f"({result['target_count']:3d}/{result['total_transitions']:4d} transitions)"
    )
print()
print(f"=== Convergence Assessment ===")
print(f"Main categories ({target_main[0]} → {target_main[1]}): {stability_main}")
print(f"All categories ({target_all[0]} → {target_all[1]}): {stability_all}")
print()
print(f"Minimum episodes for stability:")
print(f"  Main categories: ~{min_episodes_main} episodes")
print(f"  All categories: ~{min_episodes_all} episodes")
print(f"  Current dataset: {len(train_ids)} episodes")
print()
print(f"\nIs current data sufficient?")
print(f"  Main categories: {sufficient_main}")
print(f"  All categories: {sufficient_all}")
print()