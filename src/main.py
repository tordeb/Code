# --------------------------------------------------------------------------------------------------------------------------------------
# Goals: train data, test data, observed matrix, predicted matrix, markov chain
# --------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------
# Main Execution Script
# --------------------------------------------------------------------------------------------------------------------------------------
# Usage:
#   python src/main.py --order 1      first-order model only   -> output/<data>/first_order/
#   python src/main.py --order 2      second-order model only  -> output/<data>/second_order/
#   python src/main.py --order both   both models, then the side-by-side comparison -> output/<data>/comparison/
#
#   --data chooses the care episodes: mock, actual, actual_sb, actual_mb, combined_sb or combined (default)
#
# Each order is run in its own Python process so the two runs cannot share any state; they share only the code.

import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="First- vs second-order Markov chain analysis")
    parser.add_argument("--order", choices=["1", "2", "both"], default="both",
                        help="Markov chain order to run (default: both, followed by the comparison)")
    parser.add_argument("--data", choices=["mock", "actual", "actual_sb", "actual_mb", "combined_sb", "combined"],
                        default="combined", help="Care episodes to use (default: combined, see config.py)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


class Tee:
    """Write everything printed to both the console and the results file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def to_python(value):
    """Convert numpy numbers to plain Python numbers so they can be saved as JSON."""
    if hasattr(value, "item"):
        return value.item()
    return value


def run_single_order():
    # Import all modules to execute the code (config reads MARKOV_ORDER / MARKOV_SEED from the environment)
    import numpy as np

    from config import (
        ORDER, ORDER_NAME, MAIN_LABEL, ALL_LABEL, SEED, n_simulations, MAX_TERMINATING_LENGTH, OUTPUT_DIR,
        DATASET, DATASETS
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from data_loader import n_activities, train_ids, test_ids, split_summary
    from transition_matrices import (
        calculate_transition_matrix, observed_main, observed_all, train_start_probs_main, train_start_probs_all,
        test_matrix_main, test_matrix_all, convergence_main, convergence_all, target_main, target_all,
        stability_main, stability_all, min_episodes_main, min_episodes_all, sufficient_main, sufficient_all,
        format_transition, format_history
    )
    from simulations import (
        simulation_methods, simulation_counters, avg_length, distributions, best_dist, length_stats, fitted_stats,
        goodness_of_fit, fitting_warnings, sampling_warning, time_fitted,
        main_sim, all_sim, fitted_sim, predicted_fixed_main, predicted_terminating_all, predicted_fitted_main
    )
    from metrics import (
        difference_fixed_main, difference_fitted_main, difference_terminating_all, mae_main, mae_all, mae_fitted,
        train_test_difference_main, train_test_difference_all, train_test_mae_main, train_test_mae_all,
        levenshtein_results, nw_results, sw_results,
        sensitivity_results_main, sensitivity_results_all, comparison_results,
        final_mae_main, final_mae_all, conclusion,
        bootstrap_main, std_matrix_main, std_matrix_all, diff_from_mean_main, diff_from_mean_all,
        most_variable_main, most_variable_all,
        chi2_main, chi2_all, chi2_train_test_main,
        missing_main, n_possible_main, unobserved_histories_main, missing_all, n_possible_all,
        unobserved_histories_all, impossible_transitions, rare_transitions, predicted_events, predicted_events_all,
        hellinger_results
    )
    import plotting  # Saves all figures to OUTPUT_DIR
    from cross_validation import (
        cv_results, cv_summary, cv_perplexity, cv_by_group, cv_hellinger_lengths, CV_SCORES, SMOOTHING,
        N_CV_SIMULATIONS
    )

    unit = "State" if ORDER == 1 else "State Pair"

    # ----------------------------------------------------------------------------------------------------------------------------------
    # Save All Results to Text File
    # ----------------------------------------------------------------------------------------------------------------------------------

    output_file = f"{OUTPUT_DIR}/{ORDER_NAME}_order_results.txt"
    original_stdout = sys.stdout

    with open(output_file, "w", encoding="utf-8") as f:
        try:
            sys.stdout = Tee(original_stdout, f)

            # Header
            print("=" * 90)
            print(f"{ORDER_NAME.upper()}-ORDER MARKOV CHAIN")
            print("=" * 90)
            print()
            print()

            # Dataset and Simulation Parameters
            print(f"Dataset = {DATASET}: {DATASETS[DATASET]['description']}")
            print(f"Simulations = {n_simulations}")
            print(f"Random seed = {SEED}")
            print(f"Maximum terminating-state chain length = {MAX_TERMINATING_LENGTH}")
            print()

            # Split Data in Train (80%) and Test (20%)
            print("=== Train/Test Split ===")
            print(f"Total ActivityIDs: {n_activities}")
            print(f"Train ActivityIDs: {len(train_ids)} ({len(train_ids)/n_activities*100:.0f}%)")
            print(f"Test ActivityIDs: {len(test_ids)} ({len(test_ids)/n_activities*100:.0f}%)")
            print()
            print("Episodes by group (Source, RoomType):")
            print(split_summary.to_string())
            print()

            # Transition Matrices
            print("=== Transition Matrices ===")
            print("=== Function, Formula Explanation ===")
            print(calculate_transition_matrix.__doc__)
            print()
            print(f"=== Observed {MAIN_LABEL} Matrix (from training data) ===")
            print(observed_main.round(3).to_string())
            print()
            print(f"=== Observed {ALL_LABEL} Matrix (from training data) ===")
            print(observed_all.round(3).to_string())
            print()

            # Starting State Probabilities (from training data)
            print(f"=== Starting {unit} Probabilities (from training data) ===")
            print(f"{MAIN_LABEL}:")
            print(train_start_probs_main.round(3))
            print()
            print(f"{ALL_LABEL}:")
            print(train_start_probs_all.round(3))
            print()

            # 'Test' Transition Matrices
            print(f"=== Test {MAIN_LABEL} Matrix using unseen data ===")
            print(test_matrix_main.round(3).to_string())
            print()
            print(f"=== Test {ALL_LABEL} Matrix using unseen data ===")
            print(test_matrix_all.round(3).to_string())
            print()

            # Convergence Analysis
            print("=== Convergence Analysis ===")
            print(f"Tracking transition (Main): {format_transition(target_main)}")
            print(f"Tracking transition (All): {format_transition(target_all)}")
            print()
            for label, results in (("Main", convergence_main), ("All", convergence_all)):
                print(f"{label} Categories Convergence:")
                for result in results:
                    print(
                        f"  {result['episodes']:4d} episodes: "
                        f"P = {result['probability']:.4f} "
                        f"({result['target_count']:3d}/{result['from_count']:4d} transitions)"
                    )
                print()
            print("=== Convergence Assessment ===")
            print(f"Main categories ({format_transition(target_main)}): {stability_main}")
            print(f"All categories ({format_transition(target_all)}): {stability_all}")
            print()
            print("Minimum episodes for stability:")
            print(f"  Main categories: {min_episodes_main if min_episodes_main else 'not reached'}")
            print(f"  All categories: {min_episodes_all if min_episodes_all else 'not reached'}")
            print(f"  Current dataset: {len(train_ids)} episodes")
            print()
            print("Is current data sufficient?")
            print(f"  Main categories: {sufficient_main}")
            print(f"  All categories: {sufficient_all}")
            print()

            # Run Simulations
            print("=== Simulations ====")
            print(f"Running {n_simulations} simulations for each method...")
            print(f"Average Care Period Length: {avg_length}")
            print()
            for method in simulation_methods.values():
                print(f"{method['name']} simulations: {method['time']:.2f} seconds")
            print()
            print("Steps where the history was never seen in training (next state picked at random):")
            for method in simulation_methods.values():
                print(f"  {method['name']}: {simulation_counters[method['kind']]['fallback_steps']}")
            print(f"Terminating-state chains stopped at the maximum length before reaching 'Out': "
                  f"{simulation_counters['terminating']['capped_chains']} of {n_simulations}")
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
            print(f"=== Predicted {MAIN_LABEL} Matrix (from fixed-length simulations) ===")
            print(predicted_fixed_main.round(3).to_string())
            print()
            print(f"=== Predicted {ALL_LABEL} Matrix (from terminating-state simulations) ===")
            print(predicted_terminating_all.round(3).to_string())
            print()
            print(f"=== Predicted {MAIN_LABEL} Matrix (from fitted-length simulations) ===")
            print(predicted_fitted_main.round(3).to_string())
            print()

            # Example Chains
            for title, chain in ((f"{MAIN_LABEL} Simulation (Fixed Length)", main_sim),
                                 (f"{ALL_LABEL} Simulation (Terminating State)", all_sim),
                                 (f"{MAIN_LABEL} Simulation (Fitted Distribution)", fitted_sim)):
                print(title)
                print(chain)
                print(f"Length: {len(chain)}")
                print(f"Started with: {chain[0]}")
                print(f"Ended with: {chain[-1]}")
                print()

            # Calculate MAE with Proper Alignment
            print(f"=== Comparison of All Three Simulation Approaches (in-sample simulation check) ===")
            print("Note: these compare simulated chains with the TRAINING data they were generated from, so they")
            print("  check that the simulations reproduce the model; they shrink towards 0 as more chains are")
            print("  simulated. The validation of the model is the leave-one-out cross-validation at the end.")
            print(f"  (a) Fixed length MAE ({MAIN_LABEL}):        {mae_main:.4f}")
            print(f"  (b) Terminating state MAE ({ALL_LABEL}):   {mae_all:.4f}")
            print(f"  (c) Fitted length MAE ({MAIN_LABEL}):       {mae_fitted:.4f}")
            print()
            print(f"Note: Comparing {MAIN_LABEL} vs {ALL_LABEL} is not strictly like-for-like")
            print(f"  because {ALL_LABEL} has more parameters and includes")
            print(f"  terminating states (In/Out). However, the {ALL_LABEL} MAE of {mae_all:.4f}")
            print(f"  shows how well the terminating model performs overall.")
            print()

            # Dictionary for the 3 methods
            print("\n=== Dictionary Inventory ===")
            print(f"{'Method':<25} | {'Chains':<10} | {'Avg. Chain Len':<15} | {'MAE':<10}")
            print("-" * 65)
            for method in simulation_methods.values():
                avg_len = np.mean([len(c) for c in method["chains"]])
                print(f"{method['name']:<25} | {len(method['chains']):<10} | {avg_len:<15.2f} | {method['mae']:<10.4f}")
            print()

            # Compare Observed vs Predicted
            print("=== Difference between Observed and Predicted Matrices ===")
            print(f"For {MAIN_LABEL} matrix")
            print(difference_fixed_main.round(3).to_string())
            print()
            print(f"For {ALL_LABEL} matrix")
            print(difference_terminating_all.round(3).to_string())
            print()
            print(f"For {MAIN_LABEL} fitted-length matrix")
            print(difference_fitted_main.round(3).to_string())
            print()
            print(f"Mean Absolute Error for {MAIN_LABEL} (fixed length): {mae_main:.4f}")
            print(f"Mean Absolute Error for {ALL_LABEL}: {mae_all:.4f}")
            print(f"Mean Absolute Error for {MAIN_LABEL} (fitted length): {mae_fitted:.4f}")
            print()

            # Test Against Final 20% of Data
            print("==== Difference between Observed and Test Matrices ===")
            print(f"For {MAIN_LABEL} matrix")
            print(train_test_difference_main.round(3).to_string())
            print()
            print(f"For {ALL_LABEL} matrix")
            print(train_test_difference_all.round(3).to_string())
            print()
            print(f"Mean Absolute Error (Train vs Test) for {MAIN_LABEL}: {train_test_mae_main:.4f}")
            print(f"Mean Absolute Error (Train vs Test) for {ALL_LABEL}: {train_test_mae_all:.4f}")
            print()

            # Levenshtein Distance, Needleman-Wunsch (Global Alignment), Smith-Waterman (Local Alignment)
            for title, value_name, all_results in (("Levenshtein Distance Summary", "Distance", levenshtein_results),
                                                   ("Needleman-Wunsch Summary", "Score", nw_results),
                                                   ("Smith-Waterman Summary", "Score", sw_results)):
                print(f"=== {title} (simulated chains vs training episodes, in-sample) ===")
                for method_name, res in all_results.items():
                    print(f"=== Method: ({method_name}) ===")
                    print(f"Observed chains:  {res['observed_count']}")
                    print(f"Simulated chains: {res['simulated_count']}")
                    print(f"Total comparisons: {res['observed_count']} x {res['simulated_count']} = {res['total_comparisons']:,}")
                    print(f"Calculation time: {res['time_calc']:.2f} seconds")
                    if value_name == "Distance":
                        print(f"Absolute Distance (Lower is better/more similar):")
                    else:
                        print(f"Absolute Score (Higher is better/more similar):")
                    print(f"  Mean:   {res['mean_abs']:.2f}")
                    print(f"  Median: {res['median_abs']:.2f}")
                    print(f"  Mode:   {res['mode_abs']:.2f}")
                    print(f"  Std:    {res['sigma_abs']:.2f}")
                    print(f"  Min:    {res['min_abs']:.2f}")
                    print(f"  Max:    {res['max_abs']:.2f}")
                    print(f"  Skew:   {res['skew_abs']:.3f}")
                    print(f"Normalised {value_name}:")
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
            for label, results in ((MAIN_LABEL, sensitivity_results_main), (ALL_LABEL, sensitivity_results_all)):
                print(f"=== {label} Matrix Sensitivity ===")
                for res in results:
                    print(f"Noise Level {res['noise_level']*100:.1f}%: MAE = {res['mean_mae']:.4f} ± {res['std_mae']:.4f}")
                print(f"{label} Model sensitivity: {'HIGH' if results[-1]['mean_mae'] > 0.1 else 'LOW'}")
                print()
            print("=== Sensitivity Comparison ===")
            print(f"Noise Level | {MAIN_LABEL + ' MAE':<10} | {ALL_LABEL + ' MAE':<10} | More Robust")
            print("-" * 50)
            for res in comparison_results:
                print(f"{res['noise_level']*100:8.1f}%   | {res['mae_main']:10.4f} | {res['mae_all']:10.4f} | {res['more_robust']}")
            print()
            print(f"Overall Assessment:")
            print(f"  {MAIN_LABEL} matrix: {'Robust' if final_mae_main < 0.05 else 'Sensitive'} (MAE = {final_mae_main:.4f})")
            print(f"  {ALL_LABEL} matrix: {'Robust' if final_mae_all < 0.05 else 'Sensitive'} (MAE = {final_mae_all:.4f})")
            print(f"  Conclusion: {conclusion}")
            print()

            # Bootstrap Analysis
            print("=== Bootstrap Analysis ===")
            print(f"Number of samples: {len(bootstrap_main)}")
            for label, std_matrix, diff_matrix, most_variable in (
                    (MAIN_LABEL, std_matrix_main, diff_from_mean_main, most_variable_main),
                    (ALL_LABEL, std_matrix_all, diff_from_mean_all, most_variable_all)):
                print(f"{label} Matrix Variability:")
                print(f"  Mean absolute difference from bootstrap mean: {diff_matrix.mean():.4f}")
                print(f"  Average standard deviation across entries: {std_matrix.mean():.4f}")
                print(f"  Most variable transition: {most_variable}")
                print(f"  Max standard deviation: {std_matrix.max():.4f}")
            print()

            # Chi-Squared Test
            print("=== Chi-Squared Tests ===")
            for title, res in ((f"{MAIN_LABEL} Matrix (Observed vs Predicted)", chi2_main),
                               (f"{ALL_LABEL} Matrix (Observed vs Predicted)", chi2_all),
                               (f"Train vs Test {MAIN_LABEL} Matrix", chi2_train_test_main)):
                print(f"{title}:")
                print(f"  Chi-squared statistic: {res['chi2']:.4f}")
                print(f"  Degrees of freedom: {res['df']}")
                print(f"  P-value: {res['p_value']:.6f}")
                print(f"  Number of cells tested: {res['n_cells']}")
                print(f"  Result: Difference is {res['significance']}")
                print()
            print(f"Train vs Test {MAIN_LABEL} Interpretation: "
                  f"{'Overfitting likely' if chi2_train_test_main['p_value'] < 0.05 else 'Model generalises well'}")
            print()

            # Missing Transitions
            print("=== Missing Events Analysis ===")
            print()
            print(f"{MAIN_LABEL} Matrix - Transitions that never occurred in training data:")
            print(f"Total missing: {len(missing_main)} out of {n_possible_main} possible")
            print(f"Histories never observed at all: {unobserved_histories_main} of {len(observed_main.columns) ** ORDER}")
            for history, next_state in missing_main:
                print(f"  {format_history(history)} → {next_state}")
            print()
            print(f"Categorisation:")
            print(f"  Probably impossible: {len(impossible_transitions)}")
            for history, next_state in impossible_transitions:
                print(f"    {format_history(history)} → {next_state}")
            print(f"  Rare but possible: {len(rare_transitions)}")
            for history, next_state in rare_transitions[:5]:  # Show first 5
                print(f"    {format_history(history)} → {next_state}")
            if len(rare_transitions) > 5:
                print(f"    ... and {len(rare_transitions)-5} more")
            print()
            print(f"{ALL_LABEL} Matrix - Missing transitions: {len(missing_all)} out of {n_possible_all}")
            print(f"Histories never observed at all: {unobserved_histories_all} of {len(observed_all.columns) ** ORDER}")
            print()
            print(f"=== Do Simulations Predict Missing Events? ===")
            for label, events in ((MAIN_LABEL, predicted_events), (ALL_LABEL, predicted_events_all)):
                print(f"Events that were missing in training but predicted by {label} simulations: {len(events)}")
                for history, next_state, prob in events[:10]:  # Show first 10
                    print(f"  {format_history(history)} → {next_state}: {prob:.4f}")
                if len(events) > 10:
                    print(f"  ... and {len(events)-10} more")
                if len(events) == 0:
                    print("  None - simulations only predict events that were observed in training")
                print()

            # Hellinger Distances
            print("=== Summary of Hellinger Distances (simulated chains vs training episodes, in-sample) ===")
            print()
            for method_name, res in hellinger_results.items():
                print(f"Method: {method_name}")
                print(f"  Care Period Lengths: {res['lengths']:.4f}")
                for category, h_dist in res['visits'].items():
                    print(f"  {category}: {h_dist:.4f}")
                print(f"  Average Hellinger Distance (Category Visits): {res['average_visits']:.4f}")
                print()

            # Leave-One-Out Cross-Validation
            print("=" * 90)
            print("=== Leave-One-Out Cross-Validation (validation on unseen episodes) ===")
            print("=" * 90)
            print(f"Each of the {len(cv_results)} episodes was held out in turn; the model was built from the other")
            print(f"{len(cv_results) - 1} and scored on the held-out episode. Mean and 95% confidence interval over episodes.")
            print(f"Log-likelihood and accuracy use add-{SMOOTHING} smoothing. Only transitions with at least two earlier")
            print("contacts are scored, so first- and second-order are scored on exactly the same transitions.")
            print(f"Sequence similarity: {N_CV_SIMULATIONS} chains simulated per held-out episode.")
            print()
            print(f"{'Score':<62} | {'Mean':>8} | {'95% CI':>19} | {'n':>4}")
            print("-" * 102)
            for key, description in CV_SCORES.items():
                res = cv_summary[key]
                ci = f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]"
                print(f"{description:<62} | {res['mean']:>8.4f} | {ci:>19} | {res['n']:>4}")
            print()
            print(f"Perplexity (main categories): {cv_perplexity['main']:.3f}")
            print(f"Perplexity (all categories):  {cv_perplexity['all']:.3f}")
            print("(Perplexity = exp(-log-likelihood): the effective number of equally likely next contacts. Lower is better.)")
            print()
            print("Hellinger distance, held-out episode lengths vs cross-validation simulated chain lengths:")
            for method, h_dist in cv_hellinger_lengths.items():
                print(f"  {method}: {h_dist:.4f}")
            print()
            if len(cv_by_group) > 1:
                print("Mean scores by group of the held-out episode:")
                print(cv_by_group.round(4).to_string())
                print()
            print(f"Per-episode scores saved to {OUTPUT_DIR}/cross_validation_episodes.csv")
            print()

        finally:
            sys.stdout = original_stdout

    print(f"Results saved to {output_file}")

    # ----------------------------------------------------------------------------------------------------------------------------------
    # Save Key Numbers for the First- vs Second-Order Comparison
    # ----------------------------------------------------------------------------------------------------------------------------------

    def observed_histories(matrix):
        return int((matrix.sum(axis=1) > 0).sum())

    summary = {
        "order": ORDER,
        "dataset": DATASET,
        "seed": SEED,
        "n_train": len(train_ids),
        "main_label": MAIN_LABEL,
        "all_label": ALL_LABEL,
        "observed_histories_main": observed_histories(observed_main),
        "possible_histories_main": len(observed_main.columns) ** ORDER,
        "observed_histories_all": observed_histories(observed_all),
        "possible_histories_all": len(observed_all.columns) ** ORDER,
        "train_test_mae_main": train_test_mae_main,
        "train_test_mae_all": train_test_mae_all,
        "train_test_chi2_p_main": chi2_train_test_main["p_value"],
        "chi2_p_main": chi2_main["p_value"],
        "chi2_p_all": chi2_all["p_value"],
        "sensitivity_mae_5pct_main": final_mae_main,
        "sensitivity_mae_5pct_all": final_mae_all,
        "bootstrap_avg_std_main": std_matrix_main.mean(),
        "bootstrap_avg_std_all": std_matrix_all.mean(),
        "missing_main": len(missing_main),
        "possible_main": n_possible_main,
        "missing_all": len(missing_all),
        "possible_all": n_possible_all,
        "convergence_main": f"{format_transition(target_main)}: {stability_main}",
        "convergence_all": f"{format_transition(target_all)}: {stability_all}",
        "capped_terminating_chains": simulation_counters["terminating"]["capped_chains"],
        "cv": cv_summary,
        "cv_descriptions": CV_SCORES,
        "cv_perplexity": cv_perplexity,
        "cv_hellinger_lengths": cv_hellinger_lengths,
        "methods": {}
    }

    for method_name, method in simulation_methods.items():
        summary["methods"][method["kind"]] = {
            "name": method_name,
            "mae": method["mae"],
            "avg_chain_length": np.mean([len(c) for c in method["chains"]]),
            "fallback_steps": simulation_counters[method["kind"]]["fallback_steps"],
            "hellinger_lengths": hellinger_results[method_name]["lengths"],
            "hellinger_visits": hellinger_results[method_name]["average_visits"],
            "levenshtein_mean": levenshtein_results[method_name]["mean_abs"],
            "levenshtein_norm_mean": levenshtein_results[method_name]["mean_norm"],
            "nw_mean": nw_results[method_name]["mean_abs"],
            "nw_norm_mean": nw_results[method_name]["mean_norm"],
            "sw_mean": sw_results[method_name]["mean_abs"],
            "sw_norm_mean": sw_results[method_name]["mean_norm"],
        }

    summary = json.loads(json.dumps(summary, default=to_python))
    with open(f"{OUTPUT_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    args = parse_args()

    if args.order == "both":
        # Run each order in a separate process, then compare
        for order in ("1", "2"):
            print(f"\n{'#' * 30} Running order {order} {'#' * 30}\n", flush=True)
            subprocess.run([sys.executable, os.path.abspath(__file__), "--order", order, "--data", args.data,
                            "--seed", str(args.seed)], check=True)

        import compare
        compare.main(args.data)
    else:
        os.environ["MARKOV_ORDER"] = args.order
        os.environ["MARKOV_DATA"] = args.data
        os.environ["MARKOV_SEED"] = str(args.seed)
        run_single_order()
