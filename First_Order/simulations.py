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

from config import main_categories, all_categories, n_simulations
from data_loader import df_train, train_ids
from transition_matrices import *


# mc = - Stores the transition matrix (DataFrame) as mc (Markov Chain)
mc_main = observed_main
mc_all = observed_all


# --------------------------------------------------------------------------------------------------------------------------------------
# Define Functions for Predicted Transition Matrices
# --------------------------------------------------------------------------------------------------------------------------------------

def simulate_5x5(mc, start_probs, length):
    """
    Simulates one care period for the 5 main categories.
    Starts with weighted random state.
    Runs for fixed length.
    """
    current_state = np.random.choice(start_probs.index, p=start_probs.values)
    chain = [current_state]

    while len(chain) < length:
        probs = mc.loc[current_state]
        next_state = np.random.choice(mc.columns, p=probs)
        chain.append(next_state)
        current_state = next_state
    return chain
    
def simulate_7x7(mc, start_probs):
    """
    Simulates one care period for all 7 categories.
    Starts with weighted random state.
    Ends when terminating state "Out" is reached.
    Length is determined by probabilities.
    """
    # Pick starting state using weighted probabilities
    current_state = np.random.choice(start_probs.index, p=start_probs.values)

    # Start the chain
    chain = [current_state]

    # Continue until "Out"
    while current_state != "Out":

        # Use transition probabilities for current state
        probs = mc.loc[current_state]

        # Pick next state based on probabilities
        next_state = np.random.choice(mc.columns, p=probs)

        # Add to chain
        chain.append(next_state)

        # Update current state
        current_state = next_state

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
            except:
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
    
    dist_info = distributions[best_dist]
    params = dist_info['params']
    
    if best_dist == 'poisson':
        return stats.poisson.rvs(params[0], size=n_samples), None
    elif best_dist == 'nbinom':
        return stats.nbinom.rvs(params[0], params[1], size=n_samples), None
    elif best_dist == 'geom':
        return stats.geom.rvs(params[0], size=n_samples), None


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


# 5x5 Fixed Length

start_time = time.time()

# Collect chains and transitions from 5x5 fixed-length simulations
fixed_chains = []
transitions_5x5 = []

for _ in range(n_simulations):
    chain = simulate_5x5(observed_main, train_start_probs_main, avg_length)

    # Save full chain for later sequence analysis
    fixed_chains.append(chain)

    # Save transitions for predicted matrix
    for i in range(len(chain) - 1):
        transitions_5x5.append({"Current": chain[i], "Next": chain[i + 1]})

time_5x5 = time.time() - start_time


# 7x7 Terminating State

start_time = time.time()

# Collect chains and transitions from 7x7 terminating-state simulations
terminating_chains = []
transitions_7x7 = []

for _ in range(n_simulations):
    chain = simulate_7x7(observed_all, train_start_probs_all)

    # Save full chain for later sequence analysis
    terminating_chains.append(chain)

    # Save transitions for predicted matrix
    for i in range(len(chain) - 1):
        transitions_7x7.append({"Current": chain[i], "Next": chain[i + 1]})

time_7x7 = time.time() - start_time

# 5x5 Fitted Length

start_time = time.time()

fitted_chains = []
transitions_fitted_5x5 = []

for i, length in enumerate(fitted_lengths):
    chain = simulate_5x5(observed_main, train_start_probs_main, max(2, int(length)))
    fitted_chains.append(chain)
    for j in range(len(chain) - 1):
        transitions_fitted_5x5.append({"Current": chain[j], "Next": chain[j + 1]})

time_fitted = time.time() - start_time

# --------------------------------------------------------------------------------------------------------------------------------------
# Calculate Predicted Matrices amd Simulations
# --------------------------------------------------------------------------------------------------------------------------------------

# Convert to DataFrames
df_sim_5x5 = pd.DataFrame(transitions_5x5)
df_sim_7x7 = pd.DataFrame(transitions_7x7)
df_sim_fitted = pd.DataFrame(transitions_fitted_5x5)

# Calculate predicted 5x5 matrix
predicted_fixed_5x5 = calculate_transition_matrix(
    df_sim_5x5, "Current", "Next", main_categories
)
# Calculate predicted 7x7 matrix
predicted_terminating_7x7 = calculate_transition_matrix(
    df_sim_7x7, "Current", "Next", all_categories
)
# Calculate predicted fitted 5x5 matrix
predicted_fitted_5x5 = calculate_transition_matrix(
    df_sim_fitted, "Current", "Next", main_categories
)


# --------------------------------------------------------------------------------------------------------------------------------------
# Example Chains
# --------------------------------------------------------------------------------------------------------------------------------------

main_sim = simulate_5x5(mc_main, train_start_probs_main, avg_length)

all_sim = simulate_7x7(mc_all, train_start_probs_all)

fitted_sim = simulate_5x5(observed_main, train_start_probs_main, max(2, int(fitted_lengths[0])))


# --------------------------------------------------------------------------------------------------------------------------------------
# Calculate MAE with Proper Alignment
# --------------------------------------------------------------------------------------------------------------------------------------
# Mean Absolute Error measures how far predicted numbers are from real numbers
# Proper alignment penalises model for missing transitions that occurred in the real data

def calculate_mae_safe(observed, predicted):
    """
    Calculate MAE with proper alignment.
    Only evaluates on transitions that exist in observed data.
    Predicted values for missing pairs default to 0.
    """
    pred_aligned = predicted.reindex(
        index=observed.index, 
        columns=observed.columns, 
        fill_value=0
    )
    mae = np.abs(observed - pred_aligned).mean().mean()
    return mae

# Calculate MAE for all three approaches
mae_fixed = calculate_mae_safe(observed_main, predicted_fixed_5x5)
mae_fitted = calculate_mae_safe(observed_main, predicted_fitted_5x5)
mae_terminating = calculate_mae_safe(observed_all, predicted_terminating_7x7)


# ----------------------------------
# Dictionary for the 3 methods
# ----------------------------------

simulation_methods = {
    "5x5_fixed": {
        "name": "5x5 fixed length",
        "chains": fixed_chains,
        "predicted_matrix": predicted_fixed_5x5,
        "observed_matrix": observed_main,
        "categories": main_categories,
        "mae": mae_fixed
    },
    "5x5_fitted": {
        "name": "5x5 fitted length",
        "chains": fitted_chains,
        "predicted_matrix": predicted_fitted_5x5,
        "observed_matrix": observed_main,
        "categories": main_categories,
        "mae": mae_fitted
    },
    "7x7_terminating": {
        "name": "7x7 terminating state",
        "chains": terminating_chains,
        "predicted_matrix": predicted_terminating_7x7,
        "observed_matrix": observed_all,
        "categories": all_categories,
        "mae": mae_terminating
    }
}


# --------------------------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------------------------

# Run Simulations
print("=== Simulations ====")
print(f"Running {n_simulations} simulations for each method...")
print(f"Average Care Period Length: {avg_length}")
print()
print(f"5x5 Fixed Length Simulations: {time_5x5:.2f} seconds")
print(f"7x7 Terminating State Simulations: {time_7x7:.2f} seconds")
print(f"5x5 Fitted Length Simulations: {time_fitted:.2f} seconds")
print()

# Sequence Length Distribution Fitting
print("=== Sequence Length Distribution Fitting ===")
print(f"Observed lengths: min={length_stats['min']}, max={length_stats['max']}")
print(f"Mean={length_stats['mean']:.2f}, Variance={length_stats['variance']:.2f}")
print(f"Variance/Mean ratio: {length_stats['var_mean_ratio']:.2f} (>1 suggests over-dispersion)")
print()
for warning in fitting_warnings:
    print(f"  Warning: {warning}")
print()
print("Distribution comparison (AIC = Akaike Information Criterion, lower is better):")
for name, info in distributions.items():
    marker = " <-- BEST FIT" if name == best_dist else ""
    print(f"  {info['name']}: AIC = {info['aic']:.2f}{marker}")
print()
if goodness_of_fit:
    print(f"Goodness of fit test (Kolmogorov-Smirnov):")
    print(f"  D-statistic: {goodness_of_fit['D_stat']:.4f}")
    print(f"  P-value: {goodness_of_fit['p_value']:.4f}")
    print(f"  Result: {'Good fit' if goodness_of_fit['good_fit'] else 'Poor fit'} (α=0.05)")
print()
print(f"=== Simulations with Fitted Length Distribution ===")
print(f"Using: {distributions[best_dist]['name']}")
print(f"Sampled lengths: min={fitted_stats['min']}, max={fitted_stats['max']}, mean={fitted_stats['mean']:.2f}")
print()
if sampling_warning:
    print(sampling_warning)
print()
print(f"Fitted-length simulations completed in {time_fitted:.2f} seconds")
print()

# Calculate Predicted Matrices
print("=== Predicted 5x5 Matrix (from fixed-length simulations) ===")
print(predicted_fixed_5x5.round(3).to_string())
print()
print("=== Predicted 7x7 Matrix (from terminating-state simulations) ===")
print(predicted_terminating_7x7.round(3).to_string())
print()
print("=== Predicted 5x5 Matrix (from fitted-length simulations) ===")
print(predicted_fitted_5x5.round(3).to_string())
print()

# Example Chains
print("5x5 Simulation (Fixed Length)")
print(main_sim)
print(f"Length: {len(main_sim)}")
print(f"Started with: {main_sim[0]}")
print(f"Ended with: {main_sim[-1]}")
print()
print("7x7 Simulation")
print(all_sim)
print(f"Length: {len(all_sim)}")
print(f"Started with: {all_sim[0]}")
print(f"Ended with: {all_sim[-1]}")
print()
print("5x5 Simulation (Fitted Distribution)")
print(fitted_sim)
print(f"Length: {len(fitted_sim)}")
print(f"Started with: {fitted_sim[0]}")
print(f"Ended with: {fitted_sim[-1]}")
print()

# Calculate MAE with Proper Alignment
print(f"=== Comparison of All Three Simulation Approaches ===")
print(f"  (a) Fixed length MAE (5x5):        {mae_fixed:.4f}")
print(f"  (b) Terminating state MAE (7x7):   {mae_terminating:.4f}")
print(f"  (c) Fitted length MAE (5x5):       {mae_fitted:.4f}")
print()
print(f"Note: Comparing 5x5 vs 7x7 is not strictly like-for-like")
print(f"  because 7x7 has more parameters ({7*7} vs {5*5}) and includes")
print(f"  terminating states (In/Out). However, the 7x7 MAE of {mae_terminating:.4f}")
print(f"  shows how well the terminating model performs overall.")
print()

# Dictionary for the 3 methods
print("\n=== Dictionary Inventory ===")
print(f"{'Method':<25} | {'Chains':<10} | {'Avg. Chain Len':<15} | {'MAE':<10}")
print("-" * 65)
for method_key, data in simulation_methods.items():
    # Calculate average chain length for this specific method
    avg_len = np.mean([len(c) for c in data["chains"]])
    n_chains = len(data["chains"])
    
    print(f"{data['name']:<25} | {n_chains:<10} | {avg_len:<15.2f} | {data['mae']:<10.4f}")
print()
