# --------------------------------------------------------------------------------------------------------------------------------------
# Leave-One-Out Cross-Validation
# --------------------------------------------------------------------------------------------------------------------------------------
# The main validation of the model: how well does it predict care episodes it has NOT been trained on?
#
# For every care episode in the dataset:
#   1. Hold the episode out,
#   2. Build the model from all the other episodes,
#   3. Score how well the model predicts the held-out episode,
# and repeat until every episode has been held out once. This gives one score per episode, so each score can be
# reported as a mean with a 95% confidence interval. First- and second-order models are scored on the same held-out
# episodes, so they can be compared episode by episode (see compare.py).
#
# Scores:
#   - Log-likelihood per transition: the average log-probability the model gives each real next contact
#     (higher = better). Perplexity = exp(-log-likelihood) is the same information as "the number of equally likely
#     next contacts the model is effectively choosing between" (lower = better).
#   - Next-contact accuracy: how often the model's most likely next contact is the one that actually happened.
#   - Sequence similarity: chains simulated from the model, compared with the held-out episode using normalised
#     Levenshtein distance, Needleman-Wunsch and Smith-Waterman (same scoring as metrics.py).
#
# Add-one (Laplace) smoothing is used for the log-likelihood and accuracy: every transition is counted one extra time,
# so no transition has probability 0 (which would make the log-likelihood infinitely bad). A history never seen in
# training becomes equally likely to go to any category, which matches what simulate() does. The log-likelihood is
# also reported with lighter smoothing (0.1 and 0.5), to show whether conclusions depend on this choice.
#
# Only transitions with at least two earlier states are scored (in both orders), because a second-order model cannot
# score the first transition of an episode. Both orders are therefore scored on exactly the same transitions.


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy import stats

from config import ORDER, OUTPUT_DIR, main_categories, all_categories
from data_loader import df
from transition_matrices import (
    HISTORY_COLS, NEXT_COL, build_transitions, calculate_transition_matrix, history_key, full_history_index,
    get_starting_states, states_to_probabilities
)
from simulations import build_lookup, simulate, fit_length_distribution, sample_fitted_lengths
from metrics import chain_to_string, levenshtein_scores, needleman_wunsch_scores, smith_waterman_scores
from distances import hellinger_distance_histograms


SMOOTHING = 1                      # Add-one (Laplace) smoothing, used for the main scores
SMOOTHING_SENSITIVITY = [0.1, 0.5]  # Lighter smoothing, log-likelihood only
N_CV_SIMULATIONS = 100  # Chains simulated per held-out episode for the sequence-similarity scores


# --------------------------------------------------------------------------------------------------------------------------------------
# Scoring Transitions
# --------------------------------------------------------------------------------------------------------------------------------------

def scored_transitions(data, categories):
    """
    Transitions to score: every contact with at least two earlier contacts in the same episode, where the two
    earlier contacts and the next contact are all in the category list. Identical for both orders.
    """
    data = data.copy()
    grouped = data.groupby("ActivityID")["SurfaceCategories"]
    data["Previous_Category"] = grouped.shift(1)
    data[NEXT_COL] = grouped.shift(-1)
    data = data.dropna(subset=["Previous_Category", "SurfaceCategories", NEXT_COL])
    keep = (data["Previous_Category"].isin(categories) & data["SurfaceCategories"].isin(categories)
            & data[NEXT_COL].isin(categories))
    return data[keep]


def smoothed_matrix(train_data, categories, smoothing=SMOOTHING):
    """
    Transition probabilities from the training episodes, with add-one smoothing, on the full grid of histories.
    Training uses every available transition (same as the main analysis), not only the scored ones.
    """
    transitions = build_transitions(train_data, categories)

    counts = transitions.groupby(HISTORY_COLS + [NEXT_COL]).size().unstack(fill_value=0)
    counts = counts.reindex(index=full_history_index(categories), columns=categories, fill_value=0)
    smoothed = counts + smoothing
    return smoothed.div(smoothed.sum(axis=1), axis=0)


def score_episode(probabilities, episode, categories):
    """
    Log-likelihood per transition and next-contact accuracy for one held-out episode.
    Returns (log-likelihood, accuracy, number of transitions scored), or NaNs if nothing could be scored.
    """
    transitions = scored_transitions(episode, categories)
    if len(transitions) == 0:
        return np.nan, np.nan, 0

    column = {category: i for i, category in enumerate(probabilities.columns)}
    rows = dict(zip(probabilities.index, probabilities.to_numpy()))

    log_likelihoods = []
    correct = []
    for _, transition in transitions.iterrows():
        row = rows[history_key([transition[col] for col in HISTORY_COLS])]
        probability = row[column[transition[NEXT_COL]]]
        log_likelihoods.append(np.log(probability))

        # Accuracy: 1 if the true next contact is the single most likely one; shared between ties
        best = np.flatnonzero(row == row.max())
        correct.append(1 / len(best) if column[transition[NEXT_COL]] in best else 0)

    return np.mean(log_likelihoods), np.mean(correct), len(transitions)


# --------------------------------------------------------------------------------------------------------------------------------------
# Sequence Similarity with the Held-Out Episode
# --------------------------------------------------------------------------------------------------------------------------------------

def simulate_fold(train_data, train_ids):
    """
    Simulate N_CV_SIMULATIONS chains for each of the three methods, using a model built from the training episodes
    of this fold (unsmoothed, exactly as in the main analysis).
    """
    observed_main = calculate_transition_matrix(build_transitions(train_data, main_categories), HISTORY_COLS,
                                                NEXT_COL, main_categories)
    observed_all = calculate_transition_matrix(build_transitions(train_data), HISTORY_COLS, NEXT_COL, all_categories)
    start_main = states_to_probabilities(get_starting_states(train_data, train_ids, main_categories))
    start_all = states_to_probabilities(get_starting_states(train_data, train_ids))
    lookup_main, lookup_all = build_lookup(observed_main), build_lookup(observed_all)

    lengths = train_data.groupby("ActivityID").size().values
    avg_length = int(lengths.mean())
    distributions, best_dist, length_stats, _ = fit_length_distribution(lengths)
    fitted_lengths, _ = sample_fitted_lengths(distributions, best_dist, N_CV_SIMULATIONS, length_stats["mean"])

    chains = {}
    if len(start_main) > 0:
        chains["fixed"] = [simulate(lookup_main, start_main, main_categories, length=avg_length)
                           for _ in range(N_CV_SIMULATIONS)]
        chains["fitted"] = [simulate(lookup_main, start_main, main_categories, length=max(ORDER + 1, int(length)))
                            for length in fitted_lengths]
    chains["terminating"] = [simulate(lookup_all, start_all, all_categories, stop_state="Out")
                             for _ in range(N_CV_SIMULATIONS)]
    return chains


def sequence_scores(chains, episode):
    """Mean normalised Levenshtein, Needleman-Wunsch and Smith-Waterman scores against the held-out episode."""
    held_out = chain_to_string(episode["SurfaceCategories"].tolist())
    scores = {}
    for method, method_chains in chains.items():
        strings = [chain_to_string(chain) for chain in method_chains]
        scores[f"lev_norm_{method}"] = np.mean([levenshtein_scores(held_out, s)[1] for s in strings])
        scores[f"nw_norm_{method}"] = np.mean([needleman_wunsch_scores(held_out, s)[1] for s in strings])
        scores[f"sw_norm_{method}"] = np.mean([smith_waterman_scores(held_out, s)[1] for s in strings])
    return scores


# --------------------------------------------------------------------------------------------------------------------------------------
# Run Leave-One-Out
# --------------------------------------------------------------------------------------------------------------------------------------

episode_ids = df["ActivityID"].unique()
episode_info = df.groupby("ActivityID", sort=False)[["Source", "RoomType"]].first()

cv_rows = []
cv_simulated_lengths = {"fixed": [], "fitted": [], "terminating": []}

for held_out_id in episode_ids:
    train_data = df[df["ActivityID"] != held_out_id]
    train_ids = [i for i in episode_ids if i != held_out_id]
    episode = df[df["ActivityID"] == held_out_id]

    row = {"ActivityID": held_out_id, "Source": episode_info.loc[held_out_id, "Source"],
           "RoomType": episode_info.loc[held_out_id, "RoomType"], "length": len(episode)}

    for label, categories in (("main", main_categories), ("all", all_categories)):
        probabilities = smoothed_matrix(train_data, categories)
        log_likelihood, accuracy, n_scored = score_episode(probabilities, episode, categories)
        row[f"loglik_{label}"] = log_likelihood
        row[f"accuracy_{label}"] = accuracy
        row[f"n_scored_{label}"] = n_scored
        for smoothing in SMOOTHING_SENSITIVITY:
            row[f"loglik_{label}_smoothing_{smoothing}"] = score_episode(
                smoothed_matrix(train_data, categories, smoothing), episode, categories)[0]

    chains = simulate_fold(train_data, train_ids)
    row.update(sequence_scores(chains, episode))
    for method, method_chains in chains.items():
        cv_simulated_lengths[method].extend(len(chain) for chain in method_chains)

    cv_rows.append(row)

cv_results = pd.DataFrame(cv_rows)
cv_results.to_csv(f"{OUTPUT_DIR}/cross_validation_episodes.csv", index=False)


# --------------------------------------------------------------------------------------------------------------------------------------
# Summaries: Mean and 95% Confidence Interval over Held-Out Episodes
# --------------------------------------------------------------------------------------------------------------------------------------

def mean_ci(values):
    """Mean and 95% confidence interval (t-distribution) of the per-episode scores, ignoring episodes not scored."""
    values = pd.Series(values).dropna().values
    n = len(values)
    mean = values.mean()
    half_width = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"mean": mean, "ci_low": mean - half_width, "ci_high": mean + half_width, "n": n}


CV_SCORES = {
    "loglik_main": "Log-likelihood per transition (main categories)",
    "loglik_all": "Log-likelihood per transition (all categories)",
    "accuracy_main": "Next-contact accuracy (main categories)",
    "accuracy_all": "Next-contact accuracy (all categories)",
}
for method in ("fixed", "fitted", "terminating"):
    CV_SCORES[f"lev_norm_{method}"] = f"Normalised Levenshtein, {method} (lower is better)"
    CV_SCORES[f"nw_norm_{method}"] = f"Normalised Needleman-Wunsch, {method}"
    CV_SCORES[f"sw_norm_{method}"] = f"Normalised Smith-Waterman, {method}"

for label in ("main", "all"):
    for smoothing in SMOOTHING_SENSITIVITY:
        CV_SCORES[f"loglik_{label}_smoothing_{smoothing}"] = (f"Log-likelihood per transition ({label} categories), "
                                                             f"smoothing {smoothing}")

cv_summary = {key: mean_ci(cv_results[key]) for key in CV_SCORES if key in cv_results}

# Perplexity from the mean log-likelihood
cv_perplexity = {label: np.exp(-cv_summary[f"loglik_{label}"]["mean"]) for label in ("main", "all")}

# Results broken down by group (e.g. mock vs actual), when the dataset contains more than one group
cv_by_group = cv_results.groupby(["Source", "RoomType"])[
    ["loglik_main", "loglik_all", "accuracy_main", "accuracy_all"]].mean()

# Held-out lengths vs lengths of the chains simulated in cross-validation (pooled over all folds)
held_out_lengths = cv_results["length"].values
cv_hellinger_lengths = {method: hellinger_distance_histograms(held_out_lengths, lengths)
                        for method, lengths in cv_simulated_lengths.items() if lengths}
