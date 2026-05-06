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
    avg_length, time_25x5, time_49x7, time_fitted,
    distributions, best_dist, length_stats, fitted_stats,
    goodness_of_fit, fitting_warnings, sampling_warning,
    mae_fixed, mae_fitted, mae_terminating,
    main_sim, all_sim, fitted_sim,
    predicted_fixed_25x5, predicted_terminating_49x7, predicted_fitted_25x5
)
from transition_matrices import (
    observed_main, observed_all, calculate_transition_matrix,
    train_start_probs_main, train_start_probs_all,
    test_matrix_main, test_matrix_all
)
from metrics import (
    difference_fixed_25x5, difference_terminating_49x7,
    mae_main, mae_all, mae_fitted,
    train_test_difference_main, train_test_difference_all,
    train_test_mae_main, train_test_mae_all,
    levenshtein_results, nw_results, sw_results,
    sensitivity_results_25x5, sensitivity_results_49x7,
    comparison_results, final_mae_25x5, final_mae_49x7, conclusion,
    bootstrap_25x5, bootstrap_49x7, std_matrix_25x5, std_matrix_49x7,
    diff_from_mean_25x5, mean_matrix_25x5,
    chi2_25x5, p_val_25x5, df_25x5, n_cells_25x5, significance_25x5,
    chi2_49x7, p_val_49x7, df_49x7, n_cells_49x7, significance_49x7,
    p_val_test_25x5, missing_main, missing_all,
    impossible_transitions, rare_transitions,
    predicted_events, predicted_events_49x7,
    hellinger_results
)
from plotting import *


simulation_methods = {
    "25x5_fixed": {
        "name": "25x5 fixed length",
        "chains": fixed_chains,
        "predicted_matrix": predicted_fixed_25x5,
        "observed_matrix": observed_main,
        "categories": main_categories,
        "mae": mae_fixed
    },
    "25x5_fitted": {
        "name": "25x5 fitted length",
        "chains": fitted_chains,
        "predicted_matrix": predicted_fitted_25x5,
        "observed_matrix": observed_main,
        "categories": main_categories,
        "mae": mae_fitted
    },
    "49x7_terminating": {
        "name": "49x7 terminating state",
        "chains": terminating_chains,
        "predicted_matrix": predicted_terminating_49x7,
        "observed_matrix": observed_all,
        "categories": all_categories,
        "mae": mae_terminating
    }
}


# --------------------------------------------------------------------------------------------------------------------------------------
# Save All Results to Text File
# --------------------------------------------------------------------------------------------------------------------------------------

# Create output file
output_file = f"{OUTPUT_DIR}/second_order_results.txt"

# Save original stdout
original_stdout = sys.stdout

# Open file for writing ad redirect print statements to it
with open(output_file, "w") as f:
    try:
        sys.stdout = f

        # Header
        print("=" * 90)
        print("SECOND-ORDER MARKOV CHAIN")
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
                f"({result['target_count']:3d}/{result['total_transitions']:4d} transitions)")
        print()
        print(f"\n=== Convergence Assessment ===")
        print(f"Main categories ({target_main[0]} → {target_main[1]} → {target_main[2]}): {stability_main}")
        print(f"All categories ({target_all[0]} → {target_all[1]} → {target_all[2]}): {stability_all}")
        print()
        print(f"\nMinimum episodes for stability:")
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
        print(f"25x5 Fixed length Simulations:{time_25x5:.2f} seconds")
        print(f"49x7 Terminating State Simulations: {time_49x7:.2f} seconds")
        print(f"49x7 Fitted Length Simulations: {time_fitted:.2f} seconds")
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

        # Calculate Predicted Matrices amd Simulations
        print("=== Predicted 25x5 Matrix (from fixed-length simulations) ===")
        print(predicted_fixed_25x5.round(3).to_string())
        print()
        print("=== Predicted 49x7 Matrix (from terminatnig-state simulations) ===")
        print(predicted_terminating_49x7.round(3).to_string())
        print()
        print("=== Predicted 25x5 Matrix (from fitted-length simulations) ===")
        print(predicted_fitted_25x5.round(3).to_string())
        print()

        # Example Chains
        print("25x5 Simulation")
        print(main_sim)
        print(f"Length: {len(main_sim)}")
        print(f"Length: {len(main_sim)}")
        print(f"Started with: {main_sim[0]}")
        print(f"Ended with: {main_sim[-1]}")
        print()
        print("49x7 Simulation")
        print(all)
        print(all_sim)
        print(f"Length: {len(all_sim)}")
        print(f"Started with: {all_sim[0]}")
        print(f"Ended with: {all_sim[-1]}")
        print()
        print("25x5 Simulation (Fitted Distribution)")
        print(fitted_sim)
        print(f"Length: {len(fitted_sim)}")
        print(f"Started with: {fitted_sim[0]}")
        print(f"Ended with: {fitted_sim[-1]}")
        print()

        # Calculate MAE with Proper Alignment
        print(f"=== Comparison of All Three Simulation Approaches ===")
        print(f"  (a) Fixed length MAE (25x5):        {mae_fixed:.4f}")
        print(f"  (b) Terminating state MAE (49x7):   {mae_terminating:.4f}")
        print(f"  (c) Fitted length MAE (25x5):       {mae_fitted:.4f}")
        print()
        print(f"Note: Comparing 25x5 vs 49x7 is not strictly like-for-like")
        print(f"  because 49x7 has more parameters ({49*7} vs {25*5}) and includes")
        print(f"  terminating states (In/Out). However, the 49x7 MAE of {mae_terminating:.4f}")
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
        print("Noise Level | 25x5 MAE    | 49x7 MAE    | More Robust")
        print("-" * 50)
        for res in comparison_results:
            print(f"{res['noise_level']*100:8.1f}%   | {res['mae_25x5']:8.4f}   | {res['mae_49x7']:8.4f}   | {res['more_robust']}")
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
            
    finally:
        sys.stdout = original_stdout

print("Results saved to second_order_results.txt")

