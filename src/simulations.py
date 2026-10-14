# --------------------------------------------------------------------------------------------------------------------------------------
# Simulation and Predicted Transition Probability Matrices
# --------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import time

from scipy import stats

from config import (
    ORDER, MAIN_LABEL, ALL_LABEL, main_categories, all_categories, n_simulations, MAX_TERMINATING_LENGTH
)
from data_loader import df_train, train_ids
from transition_matrices import *


# --------------------------------------------------------------------------------------------------------------------------------------
# Define Functions for Predicted Transition Matrices
# --------------------------------------------------------------------------------------------------------------------------------------

def build_lookup(mc):
    """Convert a transition matrix into {history: probabilities} for fast sampling."""
    return {"columns": mc.columns.to_numpy(), "rows": dict(zip(mc.index, mc.to_numpy()))}


def simulate(lookup, start_probs, categories, length=None, stop_state=None, counters=None):
    """
    Simulates one care period, for either order.
    Starts with a weighted random starting state (first-order) or state pair (second-order).
    Picks each next state using the last ORDER states.

    Fixed / fitted length: pass length, the chain runs for that many states.
    Terminating state: pass stop_state="Out", the chain ends when "Out" is reached (or at MAX_TERMINATING_LENGTH).

    If a history was never seen in the training data, the next state is picked uniformly at random from
    categories. This can only happen for second-order, where not every state pair has been observed.

    Parameters:
    lookup: the transition matrix, from build_lookup
    start_probs: the starting probabilities for each starting state / state pair
    categories: the categories to use if a history has not been observed
    length: the required length of the simulated sequence (fixed / fitted length)
    stop_state: the terminating state (terminating state)
    counters: optional dictionary that counts 'fallback_steps' and 'capped_chains'

    Returns:
    - A simulated sequence of surface categories.
    """
    # Pick a starting state / pair using weighted probabilities
    start = start_probs.index[np.random.choice(len(start_probs), p=start_probs.values)]
    chain = list(start) if isinstance(start, tuple) else [start]

    max_length = length if stop_state is None else MAX_TERMINATING_LENGTH

    while len(chain) < max_length and (stop_state is None or chain[-1] != stop_state):
        probs = lookup["rows"].get(history_key(chain[-ORDER:]))

        if probs is not None and probs.sum() > 0:
            # Normalise for rounding errors
            next_state = np.random.choice(lookup["columns"], p=probs / probs.sum())
        else:
            # If history not in training data, pick randomly
            next_state = np.random.choice(categories)
            if counters is not None:
                counters["fallback_steps"] += 1

        chain.append(next_state)

    if counters is not None and stop_state is not None and chain[-1] != stop_state:
        counters["capped_chains"] += 1

    return chain


# --------------------------------------------------------------------------------------------------------------------------------------
# Sequence Length Distribution Fitting
# --------------------------------------------------------------------------------------------------------------------------------------

def fit_length_distribution(real_lengths):
    """
    Fit and compare distributions for care episode lengths.
    Tests Poisson and Negative Binomial distributions.
    """
    # Calculate basic statistics
    mean_len = np.mean(real_lengths)
    var_len = np.var(real_lengths)

    # Store statistics for later printing
    length_stats = {
        'min': real_lengths.min(),
        'max': real_lengths.max(),
        'mean': mean_len,
        'variance': var_len,
        'var_mean_ratio': var_len/mean_len
    }

    distributions = {}

    # Poisson Distribution
    # Good when variance = mean
    lambda_param = mean_len
    poisson_ll = np.sum(stats.poisson.logpmf(real_lengths, lambda_param))
    distributions['poisson'] = {
        'params': (lambda_param,),
        'loglik': poisson_ll,
        'aic': 2 * 1 - 2 * poisson_ll,  # 2k - 2ln(L), k=1 parameter
        'name': f'Poisson(λ={lambda_param:.2f})'
    }

    # Negative Binomial Distribution
    # Good when variance > mean (over-dispersed)
    warnings = []
    if var_len > mean_len:
        # Method of moments estimation
        p_nb = mean_len / var_len
        n_nb = (mean_len ** 2) / (var_len - mean_len)

        if n_nb > 0 and 0 < p_nb < 1:
            try:
                nbinom_ll = np.sum(stats.nbinom.logpmf(real_lengths, n_nb, p_nb))
                distributions['nbinom'] = {
                    'params': (n_nb, p_nb),
                    'loglik': nbinom_ll,
                    'aic': 2 * 2 - 2 * nbinom_ll,  # k=2 parameters
                    'name': f'NegBinom(n={n_nb:.2f}, p={p_nb:.3f})'
                }
            except Exception:
                warnings.append("Could not fit Negative Binomial")

    # Find best distribution (lowest AIC)
    if distributions:
        best_dist = min(distributions.keys(), key=lambda x: distributions[x]['aic'])
    else:
        best_dist = 'poisson'  # Fallback

    return distributions, best_dist, length_stats, warnings


def calculate_goodness_of_fit(real_lengths, distribution, params):
    """
    Calculate Kolmogorov-Smirnov test for goodness of fit.
    """
    if distribution == 'poisson':
        D_stat, p_value = stats.kstest(real_lengths, lambda x: stats.poisson.cdf(x, params[0]))
    elif distribution == 'nbinom':
        D_stat, p_value = stats.kstest(real_lengths, lambda x: stats.nbinom.cdf(x, params[0], params[1]))
    else:
        return None, None

    return D_stat, p_value


def sample_fitted_lengths(distributions, best_dist, n_samples, fallback_mean):
    """
    Sample lengths from the best-fitting distribution.
    """
    if best_dist not in distributions:
        warning = f"Warning: {best_dist} not available, using Poisson"
        return stats.poisson.rvs(fallback_mean, size=n_samples), warning

    params = distributions[best_dist]['params']

    if best_dist == 'poisson':
        return stats.poisson.rvs(params[0], size=n_samples), None
    elif best_dist == 'nbinom':
        return stats.nbinom.rvs(params[0], params[1], size=n_samples), None


# ----------------------------------
# Fit Distribution
# ----------------------------------

# Analyse real care episode lengths
real_lengths = df_train.groupby("ActivityID").size().values

# Fit distributions
distributions, best_dist, length_stats, fitting_warnings = fit_length_distribution(real_lengths)

# Goodness of fit test
goodness_of_fit = None
if best_dist in distributions:
    D_stat, p_value = calculate_goodness_of_fit(
        real_lengths,
        best_dist,
        distributions[best_dist]['params']
    )

    if D_stat is not None:
        goodness_of_fit = {
            'D_stat': D_stat,
            'p_value': p_value,
            'good_fit': p_value > 0.05
        }

# Generate fitted lengths for simulations
fitted_lengths, sampling_warning = sample_fitted_lengths(
    distributions, best_dist, n_simulations, length_stats['mean']
)

# Store all fitted length statistics
fitted_stats = {
    'min': fitted_lengths.min(),
    'max': fitted_lengths.max(),
    'mean': fitted_lengths.mean()
}


# --------------------------------------------------------------------------------------------------------------------------------------
# Run Simulations
# --------------------------------------------------------------------------------------------------------------------------------------

avg_length = int(df_train.groupby("ActivityID").size().mean())

# A chain must be at least long enough to hold its starting state(s) plus one transition
min_length = ORDER + 1

lookup_main = build_lookup(observed_main)
lookup_all = build_lookup(observed_all)

simulation_counters = {}


# Fixed Length (main categories)
start_time = time.time()
simulation_counters["fixed"] = {"fallback_steps": 0, "capped_chains": 0}
fixed_chains = [
    simulate(lookup_main, train_start_probs_main, main_categories, length=avg_length,
             counters=simulation_counters["fixed"])
    for _ in range(n_simulations)
]
time_fixed = time.time() - start_time


# Terminating State (all categories)
start_time = time.time()
simulation_counters["terminating"] = {"fallback_steps": 0, "capped_chains": 0}
terminating_chains = [
    simulate(lookup_all, train_start_probs_all, all_categories, stop_state="Out",
             counters=simulation_counters["terminating"])
    for _ in range(n_simulations)
]
time_terminating = time.time() - start_time


# Fitted Length (main categories)
start_time = time.time()
simulation_counters["fitted"] = {"fallback_steps": 0, "capped_chains": 0}
fitted_chains = [
    simulate(lookup_main, train_start_probs_main, main_categories, length=max(min_length, int(length)),
             counters=simulation_counters["fitted"])
    for length in fitted_lengths
]
time_fitted = time.time() - start_time


# --------------------------------------------------------------------------------------------------------------------------------------
# Simulated Lengths Outside the Observed Range
# --------------------------------------------------------------------------------------------------------------------------------------
# A mismatch in length can be two-sided: too many very short chains as well as a long tail. The length cap only
# shortens the tail, so both ends are reported.

def length_extremes(chains):
    lengths = np.array([len(chain) for chain in chains])
    return {
        "shorter_than_observed": np.mean(lengths < real_lengths.min()),
        "longer_than_observed": np.mean(lengths > real_lengths.max()),
        "at_most_5": np.mean(lengths <= 5),
        "percentile_5": np.percentile(lengths, 5),
        "percentile_95": np.percentile(lengths, 95),
    }

observed_length_extremes = {
    "min": real_lengths.min(), "max": real_lengths.max(), "at_most_5": np.mean(real_lengths <= 5),
    "percentile_5": np.percentile(real_lengths, 5), "percentile_95": np.percentile(real_lengths, 95),
}


# --------------------------------------------------------------------------------------------------------------------------------------
# Calculate Predicted Matrices
# --------------------------------------------------------------------------------------------------------------------------------------
# Note: for second-order only state pairs that occur in the simulations are printed in the matrices.

predicted_fixed_main = calculate_transition_matrix(
    chains_to_transitions(fixed_chains), HISTORY_COLS, NEXT_COL, main_categories
)
predicted_terminating_all = calculate_transition_matrix(
    chains_to_transitions(terminating_chains), HISTORY_COLS, NEXT_COL, all_categories
)
predicted_fitted_main = calculate_transition_matrix(
    chains_to_transitions(fitted_chains), HISTORY_COLS, NEXT_COL, main_categories
)


# --------------------------------------------------------------------------------------------------------------------------------------
# Example Chains
# --------------------------------------------------------------------------------------------------------------------------------------

main_sim = simulate(lookup_main, train_start_probs_main, main_categories, length=avg_length)

all_sim = simulate(lookup_all, train_start_probs_all, all_categories, stop_state="Out")

fitted_sim = simulate(lookup_main, train_start_probs_main, main_categories,
                      length=max(min_length, int(fitted_lengths[0])))


# --------------------------------------------------------------------------------------------------------------------------------------
# Calculate MAE with Proper Alignment
# --------------------------------------------------------------------------------------------------------------------------------------

mae_fixed = calculate_mae_safe(observed_main, predicted_fixed_main)
mae_terminating = calculate_mae_safe(observed_all, predicted_terminating_all)
mae_fitted = calculate_mae_safe(observed_main, predicted_fitted_main)


# ----------------------------------
# Dictionary for the 3 methods
# ----------------------------------
# Keys match the original output file names, e.g. "5x5_fixed" (first-order) and "25x5_fixed" (second-order)

simulation_methods = {
    f"{MAIN_LABEL}_fixed": {
        "kind": "fixed",
        "name": f"{MAIN_LABEL} fixed length",
        "chains": fixed_chains,
        "predicted_matrix": predicted_fixed_main,
        "observed_matrix": observed_main,
        "categories": main_categories,
        "mae": mae_fixed,
        "time": time_fixed
    },
    f"{MAIN_LABEL}_fitted": {
        "kind": "fitted",
        "name": f"{MAIN_LABEL} fitted length",
        "chains": fitted_chains,
        "predicted_matrix": predicted_fitted_main,
        "observed_matrix": observed_main,
        "categories": main_categories,
        "mae": mae_fitted,
        "time": time_fitted
    },
    f"{ALL_LABEL}_terminating": {
        "kind": "terminating",
        "name": f"{ALL_LABEL} terminating state",
        "chains": terminating_chains,
        "predicted_matrix": predicted_terminating_all,
        "observed_matrix": observed_all,
        "categories": all_categories,
        "mae": mae_terminating,
        "time": time_terminating
    }
}

for method in simulation_methods.values():
    method["length_extremes"] = length_extremes(method["chains"])
