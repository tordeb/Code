# --------------------------------------------------------------------------------------------------------------------------------------
# Goals: train data, test data, observed matrix, predicted matrix, markov chain
# --------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# Main Execution Script
# --------------------------------------------------------------------------------------------------------------------------------------

# Import all modules to execute the code
import config
import data_loader
import transition_matrices
import simulations
import metrics
import plotting
import sys

# Import specific variables needed for summary and saving
from config import n_simulations, OUTPUT_DIR
from data_loader import n_activities, train_ids, test_ids
from simulations import (
    simulation_methods, fixed_chains, fitted_chains, terminating_chains,
    avg_length, time_5x5, time_7x7, time_fitted,
    distributions, best_dist, length_stats, fitted_stats,
    goodness_of_fit, fitting_warnings, sampling_warning,
    mae_fixed, mae_fitted, mae_terminating,
    main_sim, all_sim, fitted_sim,
    predicted_fixed_5x5, predicted_terminating_7x7, predicted_fitted_5x5
)
from transition_matrices import (
    observed_main, observed_all, calculate_transition_matrix,
    train_start_probs_main, train_start_probs_all,
    test_matrix_main, test_matrix_all
)
from metrics import (
    difference_fixed_5x5, difference_terminating_7x7,
    mae_main, mae_all, mae_fitted,
    train_test_difference_main, train_test_difference_all,
    train_test_mae_main, train_test_mae_all,
    levenshtein_results, nw_results, sw_results,
    sensitivity_results_5x5, sensitivity_results_7x7,
    comparison_results, final_mae_5x5, final_mae_7x7, conclusion,
    bootstrap_5x5, bootstrap_7x7, std_matrix_5x5, std_matrix_7x7,
    diff_from_mean_5x5, mean_matrix_5x5,
    chi2_5x5, p_val_5x5, df_5x5, n_cells_5x5, significance_5x5,
    chi2_7x7, p_val_7x7, df_7x7, n_cells_7x7, significance_7x7,
    p_val_test_5x5, missing_main, missing_all,
    impossible_transitions, rare_transitions,
    predicted_events, predicted_events_7x7,
    hellinger_results
)
from plotting import *


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
# Save All Results to Text File
# --------------------------------------------------------------------------------------------------------------------------------------

# Create output file
output_file = f"{OUTPUT_DIR}/first_order_results.txt"

# Save original stdout
original_stdout = sys.stdout

# Open file for writing ad redirect print statements to it
with open(output_file, "w") as f:
    try:
        sys.stdout = f

        # Header
        print("=" * 90)
        print("FIRST-ORDER MARKOV CHAIN")
        print("=" * 90)
        print()
        print()

        # Simulation Parameters
        print(f"Simulations = {n_simulations}")

        # Split Data in Train (80%) and Test (20%)
        print("=== Train/Test Split ===")
        print(f"Total ActivityIDs: {n_activities}")
        print(f"Train ActivityIDs: {len(train_ids)} ({len(train_ids)/n_activities*100:.0f}%)")
        print(f"Test ActivityIDs: {len(test_ids)} ({len(test_ids)/n_activities*100:.0f}%)")
        print()

        # Transition Matrices
        print("=== Transition Matrices ===")
        print("=== Function, Formula Explanation ===")
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
        for name, chains, mae in [
            ("5x5 fixed length",      fixed_chains,      mae_fixed),
            ("5x5 fitted length",     fitted_chains,     mae_fitted),
            ("7x7 terminating state", terminating_chains, mae_terminating),
        ]:
            avg_len = np.mean([len(c) for c in chains])
            n_chains = len(chains)
            print(f"{name:<25} | {n_chains:<10} | {avg_len:<15.2f} | {mae:<10.4f}")
        print()

        # Compare Observed vs Predicted
        print("=== Difference between Observed and Predicted Matrices ===")
        print("For 5x5 matrix")
        print(difference_fixed_5x5.round(3).to_string())
        print()
        print("For 7x7 matrix")
        print(difference_terminating_7x7.round(3).to_string())
        print()
        print("For 5x5 fitted-length matrix")
        print(difference_fitted_5x5.round(3).to_string())
        print()
        print(f"Mean Absolute Error for 5x5 (fixed length): {mae_main:.4f}")
        print(f"Mean Absolute Error for 7x7: {mae_all:.4f}")
        print(f"Mean Absolute Error for 5x5 (fitted length): {mae_fitted:.4f}")
        print()

        # Test Against Final 20% of Data
        print("==== Difference between Observed and Test Matrices ===")
        print("For 5x5 matrix")
        print(train_test_difference_main.round(3).to_string())
        print()
        print("For 7x7 matrix")
        print(train_test_difference_all.round(3).to_string())
        print()
        print(f"Mean Absolute Error (Train vs Test) for 5x5: {train_test_mae_main:.4f}")
        print(f"Mean Absolute Error (Train vs Test) for 7x7: {train_test_mae_all:.4f}")
        print()

        # Levenshtein Distance
        print("=== Levenshtein Distance Summary ===")
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
        print("=== 5x5 Matrix Sensitivity ===")
        for res in sensitivity_results_5x5:
            print(f"Noise Level {res['noise_level']*100:.1f}%: MAE = {res['mean_mae']:.4f} ± {res['std_mae']:.4f}")
        print(f"5x5 Model sensitivity: {'HIGH' if sensitivity_results_5x5[-1]['mean_mae'] > 0.1 else 'LOW'}")
        print()
        print("=== 7x7 Matrix Sensitivity ===")
        for res in sensitivity_results_7x7:
            print(f"Noise Level {res['noise_level']*100:.1f}%: MAE = {res['mean_mae']:.4f} ± {res['std_mae']:.4f}")
        print(f" 7x7 Model sensitivity: {'HIGH' if sensitivity_results_7x7[-1]['mean_mae'] > 0.1 else 'LOW'}")
        print()
        print("=== Sensitivity Comparison ===")
        print("Noise Level | 5x5 MAE    | 7x7 MAE    | More Robust")
        print("-" * 50)
        for res in comparison_results:
            print(f"{res['noise_level']*100:8.1f}%   | {res['mae_5x5']:8.4f}   | {res['mae_7x7']:8.4f}   | {res['more_robust']}")
        print()
        print(f"Overall Assessment:")
        print(f"  5x5 matrix: {'Robust' if final_mae_5x5 < 0.05 else 'Sensitive'} (MAE = {final_mae_5x5:.4f})")
        print(f"  7x7 matrix: {'Robust' if final_mae_7x7 < 0.05 else 'Sensitive'} (MAE = {final_mae_7x7:.4f})")
        print(f"  Conclusion: {conclusion}")
        print()

        # Bootstrap Analysis
        print("=== Bootstrap Analysis ===")
        print(f"Number of samples: {len(bootstrap_5x5)}")
        print(f"5x5 Matrix Variability:")
        print(f"  Mean absolute difference from bootstrap mean: {diff_from_mean_5x5.mean():.4f}")
        print(f"  Average standard deviation across entries: {std_matrix_5x5.mean():.4f}")
        print(f"  Most variable transition: {np.unravel_index(std_matrix_5x5.argmax(), std_matrix_5x5.shape)}")
        print(f"  Max standard deviation: {std_matrix_5x5.max():.4f}")
        print(f"7x7 Matrix Variability:")
        print(f"  Average standard deviation across entries: {std_matrix_7x7.mean():.4f}")
        print(f"  Max standard deviation: {std_matrix_7x7.max():.4f}")
        print()

        # Chi-Squared Test
        print("=== Chi-Squared Tests ===")
        print(f"5x5 Matrix (Observed vs Predicted):")
        print(f"  Chi-squared statistic: {chi2_5x5:.4f}")
        print(f"  Degrees of freedom: {df_5x5}")
        print(f"  P-value: {p_val_5x5:.6f}")
        print(f"  Number of cells tested: {n_cells_5x5}")
        print(f"  Result: Difference is {significance_5x5}")
        print()
        print(f"7x7 Matrix (Observed vs Predicted):")
        print(f"  Chi-squared statistic: {chi2_7x7:.4f}")
        print(f"  Degrees of freedom: {df_7x7}")
        print(f"  P-value: {p_val_7x7:.6f}")
        print(f"  Number of cells tested: {n_cells_7x7}")
        print(f"  Result: Difference is {significance_7x7}")
        print()
        print(f"\nTrain vs Test 5x5 Matrix:")
        print(f"  Interpretation: {'Overfitting likely' if p_val_test_5x5 < 0.05 else 'Model generalizes well'}")
        print()

        # Missing Transitions
        print("=== Missing Events Analysis ===")
        print()
        print(f"5x5 Matrix - Transitions that never occurred in training data:")
        print(f"Total missing: {len(missing_main)}")
        for from_state, to_state in missing_main:
            print(f"  {from_state} → {to_state}")
        print()
        print(f"Categorisation:")
        print(f"  Probably impossible: {len(impossible_transitions)}")
        for trans in impossible_transitions:
            print(f"    {trans[0]} → {trans[1]}")
        print(f"  Rare but possible: {len(rare_transitions)}")
        for trans in rare_transitions[:5]:  # Show first 5
            print(f"    {trans[0]} → {trans[1]}")
        if len(rare_transitions) > 5:
            print(f"    ... and {len(rare_transitions)-5} more")
        print()    
        print(f"7x7 Matrix - Missing transitions: {len(missing_all)}")
        print()
        print(f"=== Do Simulations Predict Missing Events? ===")
        print(f"Events that were missing in training but predicted by 5x5 simulations: {len(predicted_events)}")
        for from_state, to_state, prob in predicted_events:
            print(f"  {from_state} → {to_state}: {prob:.4f}")
        if len(predicted_events) == 0:
            print("  None - simulations only predict events that were observed in training")
        print()
        print(f"Events that were missing in training but predicted by 7x7 simulations: {len(predicted_events_7x7)}")
        for from_state, to_state, prob in predicted_events_7x7[:10]:  # Show first 10
            print(f"  {from_state} → {to_state}: {prob:.4f}")
        if len(predicted_events_7x7) > 10:
            print(f"  ... and {len(predicted_events_7x7)-10} more")
        if len(predicted_events_7x7) == 0:
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

    finally:
        sys.stdout = original_stdout

print("Results saved to first_order_results.txt")
