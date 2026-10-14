# --------------------------------------------------------------------------------------------------------------------------------------
# Observed Transition Probability Matrices (from training data)
# --------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from config import ORDER, main_categories, all_categories, convergence_targets
from data_loader import df_train, df_test, train_ids, test_ids


# --------------------------------------------------------------------------------------------------------------------------------------
# Transition Matrices
# --------------------------------------------------------------------------------------------------------------------------------------
# First-order:  P(x_j | x_i) = x_ij / Σ(m=1 to M) x_im
#   P(x_j | x_i) = Probability of transitioning to state j, given current state i
#   x_ij = Number of observed transitions from i to j
#   Σ(m=1 to M) x_im = Total transitions from state i to any state m
#
# Second-order: P(x_k | x_i, x_j) = x_ijk / Σ(m=1 to M) x_ijm
#   P(x_k | x_i, x_j) = Probability of transitioning to state k, given previous state i and current state j
#   x_ijk = Number of observed transitions from i to j to k
#   Σ(m=1 to M) x_ijm = Total transitions from state pair (i, j) to any state m
#
# Both are the same calculation: count how often each 'history' (the last ORDER states) is followed by each next state,
# then divide by the row total. For first-order the history is one state, for second-order it is a pair of states.

# Columns holding the history, oldest first
if ORDER == 1:
    HISTORY_COLS = ["SurfaceCategories"]
else:
    HISTORY_COLS = ["Previous_Category", "SurfaceCategories"]
NEXT_COL = "Next_Category"


def calculate_transition_matrix(data, history_cols, next_col, categories):
    """
    Calculate transition probability matrices of any order using:
    First-order:  P(x_j | x_i) = x_ij / Σ(m=1 to M) x_im
    Second-order: P(x_k | x_i, x_j) = x_ijk / Σ(m=1 to M) x_ijm

    Parameters:
    data: DataFrame with transition data
    history_cols: Column names for the history (["SurfaceCategories"] for first-order,
                  ["Previous_Category", "SurfaceCategories"] for second-order)
    next_col: Column name for next state
    categories: List of category names

    Returns:
    - Transition probability matrix (rows = histories, columns = next state)
    """
    # Count transitions from each history to each next state
    counts = data.groupby(history_cols + [next_col]).size().unstack(fill_value=0)

    # Sum each row
    sum_row = counts.sum(axis=1)

    # Divide counts by row totals
    probabilities = counts.div(sum_row, axis=0)

    # Reorder rows and columns
    # First-order keeps every category as a row. Second-order keeps only the state pairs that were observed.
    if len(history_cols) == 1:
        probabilities = probabilities.reindex(index=categories, columns=categories)
    else:
        probabilities = probabilities.reindex(columns=categories)

    # Fill missing values with 0
    probabilities = probabilities.fillna(0)

    return probabilities


def history_key(states):
    """Row label for a history: a single state for first-order, a tuple of states for second-order."""
    states = tuple(states)
    return states[0] if len(states) == 1 else states


def full_history_index(categories):
    """Every possible history (all 5 / 7 states for first-order, all 25 / 49 pairs for second-order)."""
    if ORDER == 1:
        return pd.Index(categories, name=HISTORY_COLS[0])
    return pd.MultiIndex.from_product([categories] * ORDER, names=HISTORY_COLS)


# ----------------------------------
# Prepare Data
# ----------------------------------
# Group by ActivityID for individual chains

def build_transitions(df, categories=None):
    """
    Add the history and next-state columns to a care-period DataFrame.

    Parameters:
    df: DataFrame of care periods (one row per surface contact)
    categories: optional list of categories; every state in the transition must be in it

    Returns:
    - DataFrame with one row per (history, next state) transition
    """
    data = df.copy()
    grouped = data.groupby("ActivityID")["SurfaceCategories"]

    # shift(1) = look at the preceding row to see what the previous state was
    if ORDER == 2:
        data["Previous_Category"] = grouped.shift(1)

    # shift(-1) = look at following row to see what the following state is
    data[NEXT_COL] = grouped.shift(-1)

    # Remove rows where there's no previous or next category (e.g. from "Out" to "nothing") (NaN values)
    data = data.dropna(subset=HISTORY_COLS + [NEXT_COL])

    # Filter for categories (e.g. remove 'In' and 'Out' states for the main-category matrix)
    if categories is not None:
        keep = np.ones(len(data), dtype=bool)
        for col in HISTORY_COLS + [NEXT_COL]:
            keep &= data[col].isin(categories).values
        data = data[keep]

    return data


def chains_to_transitions(chains):
    """Convert simulated chains into the same transition format as build_transitions."""
    rows = []
    for chain in chains:
        for i in range(len(chain) - ORDER):
            row = dict(zip(HISTORY_COLS, chain[i:i + ORDER]))
            row[NEXT_COL] = chain[i + ORDER]
            rows.append(row)
    return pd.DataFrame(rows, columns=HISTORY_COLS + [NEXT_COL])


df_train_clean = build_transitions(df_train)
df_train_main = build_transitions(df_train, main_categories)


# ----------------------------------
# Calculate the Matrices
# ----------------------------------

observed_main = calculate_transition_matrix(df_train_main, HISTORY_COLS, NEXT_COL, main_categories)
observed_all = calculate_transition_matrix(df_train_clean, HISTORY_COLS, NEXT_COL, all_categories)


# --------------------------------------------------------------------------------------------------------------------------------------
# Starting State Probabilities (from training data)
# --------------------------------------------------------------------------------------------------------------------------------------
# Select starting state (first-order) or starting state pair (second-order) using weighted random choice

def get_starting_states(df, activity_ids, categories=None):
    """
    Get the first ORDER states of each care period (a single state for first-order, a pair for second-order).

    Parameters:
    df: DataFrame containing the care-period data
    activity_ids: list of ActivityID values to include
    categories: optional list of categories to filter by

    Returns:
    - A list of starting histories
    """
    starting_states = []

    for activity_id in activity_ids:
        activity_data = df[df["ActivityID"] == activity_id]["SurfaceCategories"].values

        if len(activity_data) >= ORDER:
            start = activity_data[:ORDER]

            # Filter by categories
            if categories is None or all(state in categories for state in start):
                starting_states.append(history_key(start))

    return starting_states


def states_to_probabilities(states):
    """Convert list of starting states to probability distribution."""
    return pd.Series(states).value_counts(normalize=True)


train_start_probs_main = states_to_probabilities(get_starting_states(df_train, train_ids, main_categories))
train_start_probs_all = states_to_probabilities(get_starting_states(df_train, train_ids))


# --------------------------------------------------------------------------------------------------------------------------------------
# 'Test' Transition Matrices
# --------------------------------------------------------------------------------------------------------------------------------------
# Transition matrices formed of just the test data (final 20%) for comparison.
# Follows same process as for the first 80%.

df_test_clean = build_transitions(df_test)
df_test_main = build_transitions(df_test, main_categories)

test_matrix_main = calculate_transition_matrix(df_test_main, HISTORY_COLS, NEXT_COL, main_categories)
test_matrix_all = calculate_transition_matrix(df_test_clean, HISTORY_COLS, NEXT_COL, all_categories)


# --------------------------------------------------------------------------------------------------------------------------------------
# MAE with Proper Alignment
# --------------------------------------------------------------------------------------------------------------------------------------
# Mean Absolute Error measures how far predicted numbers are from real numbers.
# Used for every MAE in the analysis (observed vs predicted, train vs test, perturbation), for both orders.

def align_to(reference, other):
    """Reindex 'other' to the rows and columns of 'reference'. Missing entries become 0."""
    return other.reindex(index=reference.index, columns=reference.columns, fill_value=0)


def calculate_mae_safe(observed, predicted):
    """
    Calculate MAE with proper alignment.
    Only evaluates on histories that were observed (rows of the observed matrix that sum to more than 0).
    Predicted values for missing transitions default to 0, which penalises the model for missing them.
    """
    observed_rows = observed[observed.sum(axis=1) > 0]
    pred_aligned = align_to(observed_rows, predicted)
    return np.abs(observed_rows - pred_aligned).mean().mean()


# --------------------------------------------------------------------------------------------------------------------------------------
# Convergence Analysis
# --------------------------------------------------------------------------------------------------------------------------------------
# Shows how transition probabilities change with more training data
# Helps to determine amount of training data required for stable estimates

def convergence_analysis(df_train, train_ids, categories, target_transition):
    """
    Analyses how transition probabilities converge as training data increases.
    Shows: P(target_transition) vs Number of Training Episodes.

    Parameters:
    df_train: DataFrame containing the training data
    train_ids: list of training episode IDs
    categories: list of surface categories to include
    target_transition: transition to track, (history..., next), for example ("Equipment", "Patient")
                       for first-order or ("HygieneArea", "Equipment", "Patient") for second-order

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

    target_history = history_key(target_transition[:-1])
    target_next = target_transition[-1]

    convergence_results = []

    for size in sample_sizes:
        # Use first 'size' episodes (chronological order)
        subset_ids = train_ids[:size]
        subset_data = df_train[df_train["ActivityID"].isin(subset_ids)]

        # Prepare data (same as in main analysis) and filter for categories of interest
        subset_filtered = build_transitions(subset_data, categories)

        if len(subset_filtered) > 0:
            # Calculate transition matrix
            subset_matrix = calculate_transition_matrix(subset_filtered, HISTORY_COLS, NEXT_COL, categories)

            # Extract target probability
            if target_history in subset_matrix.index and target_next in subset_matrix.columns:
                target_prob = subset_matrix.loc[[target_history], target_next].iloc[0]
            else:
                target_prob = 0

            # Rows where the history matches the target history
            from_rows = np.ones(len(subset_filtered), dtype=bool)
            for col, state in zip(HISTORY_COLS, target_transition[:-1]):
                from_rows &= (subset_filtered[col] == state).values

            # Count total transitions
            total_transitions = len(subset_filtered)

            # Count times we were in the starting history, and how often it was followed by the target
            from_count = int(from_rows.sum())
            target_count = int((from_rows & (subset_filtered[NEXT_COL] == target_next).values).sum())

        else:
            target_prob = 0
            total_transitions = 0
            target_count = 0
            from_count = 0

        convergence_results.append({
            'sample_size': size,
            'probability': target_prob,
            'total_transitions': total_transitions,
            'target_count': target_count,
            'from_count': from_count,
            'episodes': len(subset_ids)
        })

    return convergence_results, target_transition


# Run convergence analysis for main categories
convergence_main, target_main = convergence_analysis(
    df_train, train_ids, main_categories, convergence_targets["main"]
)

# Run convergence analysis for all categories
convergence_all, target_all = convergence_analysis(
    df_train, train_ids, all_categories, convergence_targets["all"]
)


# Confidence interval
def wilson_ci(k, n, z=1.96):
    """95% confidence interval for k successes out of n."""
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
    n = final['from_count']        # times we were in the starting history

    # Check 1: seen often enough?
    if p == 0 or k < min_count:
        return f"INSUFFICIENT DATA (observed {k} time{'s' if k != 1 else ''})"

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


def format_transition(transition):
    """('A', 'B', 'C') -> 'A → B → C'"""
    return " → ".join(transition)


def format_history(key):
    """Row label as text: 'A' for first-order, '(A, B)' for second-order."""
    return f"({', '.join(key)})" if isinstance(key, tuple) else str(key)
