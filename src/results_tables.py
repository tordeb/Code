# --------------------------------------------------------------------------------------------------------------------------------------
# LaTeX Tables for the Paper: Model Results (Results 3.3 onwards)
# --------------------------------------------------------------------------------------------------------------------------------------
# Reads the summary.json written by main.py for each order and writes ready-to-paste LaTeX tables to
# output/<dataset>/latex/. Run after `python src/main.py --order both --data <dataset>`.
#
#   table_hellinger.tex   Hellinger distances between simulated chains and training episodes: episode length and
#                         contacts per category, for each simulation method and order (in-sample)
#   table_matrix_mae.tex  MAE between transition matrices: observed vs predicted (in-sample), predicted vs test, and
#                         train vs test
#
# Usage: python src/results_tables.py [--data combined]


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import argparse
import json
import os

import numpy as np

from config import OUTPUT_ROOT, main_categories
from latex_tables import NAMES, full_width, note


METHODS = [("fixed", "Fixed"), ("fitted", "Fitted"), ("terminating", "Term.")]
ORDERS = [("first", "First-order"), ("second", "Second-order")]


def load_summaries(dataset):
    summaries = {}
    for order, _ in ORDERS:
        with open(os.path.join(OUTPUT_ROOT, dataset, f"{order}_order", "summary.json"), encoding="utf-8") as f:
            summaries[order] = json.load(f)
    return summaries


def save(dataset, name, lines):
    folder = os.path.join(OUTPUT_ROOT, dataset, "latex")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
        f.write("\n".join(full_width(line) for line in lines) + "\n")
    print(f"Saved {os.path.join(folder, name)}")


# --------------------------------------------------------------------------------------------------------------------------------------
# Hellinger Distances (Results 3.3)
# --------------------------------------------------------------------------------------------------------------------------------------

def hellinger_table(summaries):
    """
    One column per order and method. The average is over the five main categories for every method, so the
    terminating-state method (which also has In and Out) is compared like-for-like; In and Out are shown separately.
    """
    columns = [(order, method) for order, _ in ORDERS for method, _ in METHODS]

    def value(order, method, category=None):
        result = summaries[order]["methods"][method]
        if category is None:
            return result["hellinger_lengths"]
        return result["hellinger_visits_by_category"].get(category, np.nan)

    def best_row(values):
        """Bold the lowest value within each order."""
        cells = []
        for o, _ in enumerate(ORDERS):
            block = values[o * len(METHODS):(o + 1) * len(METHODS)]
            finite = [v for v in block if not np.isnan(v)]
            lowest = min(finite) if finite else None
            for v in block:
                if np.isnan(v):
                    cells.append("--")
                elif v == lowest:
                    cells.append(f"\\textbf{{{v:.3f}}}")
                else:
                    cells.append(f"{v:.3f}")
        return " & ".join(cells)

    fixed_length = summaries["first"]["fixed_length"]
    capped = {order: summaries[order]["capped_terminating_chains"] for order, _ in ORDERS}
    n_train = summaries["first"]["n_train"]

    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Hellinger distance between the simulated chains and the training episodes, for episode length "
             "and for the number of contacts with each category per episode (in-sample; 0 = identical "
             "distributions, 1 = no overlap). The lowest value for each order is in bold.}",
             "\\label{tab:hellinger}",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{\\textbf{First-order}} & \\multicolumn{3}{c}{\\textbf{Second-order}} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             " & " + " & ".join(f"\\textbf{{{label}}}" for _ in ORDERS for _, label in METHODS) + " \\\\",
             "\\midrule"]

    lines.append("Episode length & " + best_row([value(o, m) for o, m in columns]) + " \\\\")
    lines.append("\\addlinespace")
    lines.append("\\multicolumn{7}{l}{\\textit{Contacts per episode}} \\\\")
    for category in main_categories:
        lines.append(f"{NAMES[category]} & " + best_row([value(o, m, category) for o, m in columns]) + " \\\\")
    averages = [np.mean([value(o, m, c) for c in main_categories]) for o, m in columns]
    lines.append("\\addlinespace")
    lines.append("Average (5 categories) & " + best_row(averages) + " \\\\")
    lines.append("\\addlinespace")
    for category in ("In", "Out"):
        values = [value(o, m, category) if m == "terminating" else np.nan for o, m in columns]
        cells = ["--" if np.isnan(v) else f"{v:.3f}" for v in values]
        lines.append(f"{NAMES[category]} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note(f"Fixed: every chain has {fixed_length} contacts (the mean training episode length, rounded down). "
                  f"Fitted: lengths drawn from the negative binomial fitted to the training episode lengths. "
                  f"Term.\\ (terminating-state): seven-category chains that stop at \\textit{{Out}}, capped at 100 "
                  f"contacts ({capped['first']} first-order and {capped['second']} second-order chains of 1,000 reached "
                  f"the cap). Distances compare 1,000 simulated chains with the {n_train} training episodes they were "
                  f"estimated from, so they check that the simulations reproduce the training data, not how well the "
                  f"model predicts new episodes. The average is over the five main categories for every method; "
                  f"\\textit{{In}} and \\textit{{Out}} exist only in the terminating-state chains.")
    lines += ["\\end{table}"]
    return lines


# --------------------------------------------------------------------------------------------------------------------------------------
# Transition Matrix MAE (Results 3.4)
# --------------------------------------------------------------------------------------------------------------------------------------

def matrix_mae_table(summaries):
    """
    Three comparisons, each the MAE between two transition probability matrices:
      observed (training) vs predicted (simulated)   in-sample: do the simulations reproduce the training matrix?
      predicted vs test                               simulated matrix vs the matrix of the held-out test episodes
      train vs test                                   training matrix vs test matrix (no simulation involved)
    The terminating-state (7-category) values leave out histories ending in Out; the full values are in the note.
    """
    def method_value(order, method, which):
        result = summaries[order]["methods"][method]
        if method != "terminating":
            return result["mae"] if which == "observed" else result["predicted_test_mae"]
        key = "mae_terminating_without_out" if which == "observed" else "predicted_test_mae_terminating_without_out"
        return summaries[order][key]

    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Mean absolute error (MAE) between transition probability matrices. Observed vs predicted is "
             "in-sample: the matrix estimated from the simulated chains against the training matrix they were "
             "simulated from. Predicted vs test and train vs test compare with the matrix of the held-out test "
             "episodes.}",
             "\\label{tab:matrix_mae}",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{\\textbf{First-order}} & \\multicolumn{3}{c}{\\textbf{Second-order}} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             " & " + " & ".join(f"\\textbf{{{label}}}" for _ in ORDERS for _, label in METHODS) + " \\\\",
             "\\midrule"]

    for which, label in (("observed", "Observed vs predicted"), ("test", "Predicted vs test")):
        cells = [f"{method_value(order, method, which):.4f}" for order, _ in ORDERS for method, _ in METHODS]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")

    cells = []
    for order, _ in ORDERS:
        cells.append(f"\\multicolumn{{2}}{{c}}{{{summaries[order]['train_test_mae_main']:.4f}}}")
        cells.append(f"{summaries[order]['train_test_mae_all_without_out']:.4f}")
    lines.append("Train vs test & " + " & ".join(cells) + " \\\\")

    first, second = summaries["first"], summaries["second"]
    def size(label):
        return "$" + label.replace("x", "\\times") + "$"

    n_test_note = (f"Fixed and fitted use the five-category matrices ({size(first['main_label'])} and "
                   f"{size(second['main_label'])}), terminating-state the seven-category matrices "
                   f"({size(first['all_label'])} and {size(second['all_label'])}), so the train vs test value is "
                   f"shared by fixed and fitted. ")
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note(
        n_test_note
        + "Only histories observed in the first matrix of each pair are compared. Seven-category values leave out "
          "histories ending in \\textit{Out}: simulated chains stop at \\textit{Out}, so those rows are always empty, "
          "whereas the observed episodes continue after \\textit{Out}. Including them, observed vs predicted is "
          f"{first['methods']['terminating']['mae']:.4f} (first-order) and "
          f"{second['methods']['terminating']['mae']:.4f} (second-order), predicted vs test "
          f"{first['methods']['terminating']['predicted_test_mae']:.4f} and "
          f"{second['methods']['terminating']['predicted_test_mae']:.4f}, and train vs test "
          f"{first['train_test_mae_all']:.4f} and {second['train_test_mae_all']:.4f}. MAE is comparable between "
          "orders in direction only: second-order matrices have more rows, each estimated from fewer transitions.")
    lines += ["\\end{table}"]
    return lines


# --------------------------------------------------------------------------------------------------------------------------------------
# Sequence Similarity with the Scrambled Baseline (Results 3.5)
# --------------------------------------------------------------------------------------------------------------------------------------

SIMILARITY_METRICS = [("levenshtein_norm_mean", "levenshtein_norm_scrambled", "Levenshtein distance", True),
                      ("nw_norm_mean", "nw_norm_scrambled", "Needleman-Wunsch score", False),
                      ("sw_norm_mean", "sw_norm_scrambled", "Smith-Waterman score", False)]
METHOD_NAMES = {"fixed": "Fixed", "fitted": "Fitted", "terminating": "Terminating"}


def similarity_table(summaries):
    """Mean normalised similarity of simulated chains to the training episodes, next to the scrambled chains."""
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Mean normalised sequence similarity between the simulated chains and the training episodes "
             "(in-sample), for each simulation method and for the same chains with their contacts shuffled "
             "(scrambled baseline). Values in bold beat the scrambled baseline.}",
             "\\label{tab:similarity}",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{\\textbf{First-order}} & \\multicolumn{2}{c}{\\textbf{Second-order}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             " & \\textbf{Model} & \\textbf{Scrambled} & \\textbf{Model} & \\textbf{Scrambled} \\\\",
             "\\midrule"]

    for i, (model_key, scrambled_key, label, lower_is_better) in enumerate(SIMILARITY_METRICS):
        arrow = "($\\downarrow$)" if lower_is_better else "($\\uparrow$)"
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{label} {arrow}}}}} \\\\")
        for method, method_label in METHOD_NAMES.items():
            cells = []
            for order, _ in ORDERS:
                result = summaries[order]["methods"][method]
                model, scrambled = result[model_key], result[scrambled_key]
                better = model < scrambled if lower_is_better else model > scrambled
                cells += [f"\\textbf{{{model:.3f}}}" if better else f"{model:.3f}", f"{scrambled:.3f}"]
            lines.append(f"\\quad {method_label} & " + " & ".join(cells) + " \\\\")
        if i < len(SIMILARITY_METRICS) - 1:
            lines.append("\\addlinespace")

    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note("Each of the 1,000 simulated chains is compared with each of the 130 training episodes. Levenshtein: "
                  "1 $-$ normalised similarity ratio, lower is better. Needleman-Wunsch: global alignment score divided "
                  "by the shorter sequence length (match 1, mismatch $-1$, gap $-1$). Smith-Waterman: local alignment "
                  "score divided by twice the shorter length (match 2, mismatch $-1$, gap $-1$). Scrambled: the first "
                  "200 chains of each method with their contacts shuffled, keeping their length and category mix.")
    lines += ["\\end{table}"]
    return lines


# --------------------------------------------------------------------------------------------------------------------------------------
# Leave-One-Out Validation (Results, validation on unseen episodes)
# --------------------------------------------------------------------------------------------------------------------------------------

def load_cv_comparison(dataset):
    import pandas as pd
    table = pd.read_csv(os.path.join(OUTPUT_ROOT, dataset, "comparison", "comparison_cross_validation.csv"))
    return table.set_index("score")


def loo_table(summaries, cv_comparison):
    first, second = summaries["first"], summaries["second"]

    def mean_ci(cv, key):
        return f"{cv[key]['mean']:.3f}"

    def difference(score):
        r = cv_comparison.loc[score]
        return f"{r['difference_mean']:+.3f} [{r['difference_ci_low']:+.3f}, {r['difference_ci_high']:+.3f}]"

    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Leave-one-out validation on unseen episodes. Each episode was held out in turn and the model "
             "built from all the others. Means over held-out episodes; the difference is second-order minus first-order "
             "on the same episode, with its 95\\% CI.}",
             "\\label{tab:loo}",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{\\textbf{Order}} & \\\\",
             "\\cmidrule(lr){2-4}",
             "\\textbf{Score} & \\textbf{0} & \\textbf{1} & \\textbf{2} & "
             "\\textbf{Difference (2 $-$ 1) [95\\% CI]} \\\\",
             "\\midrule",
             "\\multicolumn{5}{l}{\\textit{Predicting the next contact}} \\\\"]

    for label_key, n_cat in (("main", 5), ("all", 7)):
        name = "main categories" if label_key == "main" else "all categories"
        zero = first["cv"][f"loglik_{label_key}_zero"]["mean"]
        lines.append(f"Log-likelihood ({n_cat} cat.) & {zero:.3f} & {mean_ci(first['cv'], f'loglik_{label_key}')} & "
                     f"{mean_ci(second['cv'], f'loglik_{label_key}')} & "
                     f"{difference(f'Log-likelihood per transition ({name})')} \\\\")
        lines.append(f"Perplexity ({n_cat} cat.) & {np.exp(-zero):.2f} & {first['cv_perplexity'][label_key]:.2f} & "
                     f"{second['cv_perplexity'][label_key]:.2f} & \\\\")
    for label_key, n_cat in (("main", 5), ("all", 7)):
        name = "main categories" if label_key == "main" else "all categories"
        zero = first["cv"][f"accuracy_{label_key}_zero"]["mean"]
        lines.append(f"Accuracy ({n_cat} cat.) & {zero:.3f} & {mean_ci(first['cv'], f'accuracy_{label_key}')} & "
                     f"{mean_ci(second['cv'], f'accuracy_{label_key}')} & "
                     f"{difference(f'Next-contact accuracy ({name})')} \\\\")

    lines += ["\\addlinespace",
              " & \\textbf{Scrambled} & & & \\\\",
              "\\multicolumn{5}{l}{\\textit{Similarity of simulated chains to the held-out episode}} \\\\"]
    for method in ("fixed", "fitted", "terminating"):
        for key, label, score in (("lev", "Lev.\\ ($\\downarrow$)", "Normalised Levenshtein, {m} (lower is better)"),
                                  ("sw", "S-W ($\\uparrow$)", "Normalised Smith-Waterman, {m}")):
            lines.append(f"{METHOD_NAMES[method]}, {label} & "
                         f"{first['cv'][f'{key}_norm_{method}_scrambled']['mean']:.3f} & "
                         f"{mean_ci(first['cv'], f'{key}_norm_{method}')} & {mean_ci(second['cv'], f'{key}_norm_{method}')} & "
                         f"{difference(score.format(m=method))} \\\\")

    n_main, n_all = first["cv"]["loglik_main"]["n"], first["cv"]["loglik_all"]["n"]
    smooth = cv_comparison.loc["Log-likelihood per transition (main categories), smoothing 0.1"]
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note(
        f"Log-likelihood is per transition (higher is better) and perplexity $= \\exp(-$log-likelihood$)$ (lower is "
        f"better). Accuracy is the share of next contacts predicted correctly, with shared credit for ties. Only "
        f"transitions with at least two earlier contacts are scored, so both orders are scored on the same transitions; "
        f"{n_main} and {n_all} episodes had at least one such transition with five and seven categories. Transition "
        f"probabilities use add-one smoothing; with smoothing 0.1 the five-category log-likelihood difference is "
        f"{smooth['difference_mean']:+.3f} [{smooth['difference_ci_low']:+.3f}, {smooth['difference_ci_high']:+.3f}]. "
        f"Zero-order predicts from category frequencies only. For the similarity rows the first column is the "
        f"scrambled baseline (the first-order chains with their contacts shuffled) instead of the zero-order model. "
        f"Similarity uses 100 simulated chains per held-out episode; Lev.\\ = normalised Levenshtein distance (lower is "
        f"better), S-W = normalised Smith-Waterman score (higher is better).")
    lines += ["\\end{table}"]
    return lines


# --------------------------------------------------------------------------------------------------------------------------------------
# Sensitivity and Robustness (Results 3.6)
# --------------------------------------------------------------------------------------------------------------------------------------

def transition_label(text):
    """'(HygieneArea, NearPatient) → Equipment' -> paper names with a LaTeX arrow."""
    for key in sorted(NAMES, key=len, reverse=True):
        text = text.replace(key, NAMES[key])
    return text.replace("→", "$\\rightarrow$")


def perturbation_table(summaries):
    first, second = summaries["first"], summaries["second"]
    columns = [(first, "sensitivity_main"), (first, "sensitivity_all"), (second, "sensitivity_main"),
               (second, "sensitivity_all")]
    size = lambda label: "$" + label.replace("x", "\\times") + "$"

    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Matrix perturbation: MAE between the predicted matrix from the original simulations and the "
             "predicted matrix from 100 chains simulated from a perturbed transition matrix (mean $\\pm$ SD over 50 "
             "repetitions). The 0\\% row repeats the test without any perturbation.}",
             "\\label{tab:perturbation}",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{\\textbf{First-order}} & \\multicolumn{2}{c}{\\textbf{Second-order}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             f"\\textbf{{Noise}} & {size(first['main_label'])} fixed & {size(first['all_label'])} term. & "
             f"{size(second['main_label'])} fixed & {size(second['all_label'])} term. \\\\",
             "\\midrule"]
    for i in range(len(first["sensitivity_main"])):
        level = first["sensitivity_main"][i]["noise_level"] * 100
        cells = [f"{s[key][i]['mean_mae']:.4f} $\\pm$ {s[key][i]['std_mae']:.4f}" for s, key in columns]
        label = "0\\% (none)" if level == 0 else f"{level:.1f}\\%"
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if level == 0:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note("Noise is uniform on $[-x, +x]$, added to every transition probability; probabilities are then "
                  "floored at 0.001 and each row renormalised. Fixed: fixed-length simulation (five categories); term.: "
                  "terminating-state simulation (seven categories). Values close to the 0\\% row are within the "
                  "variation from simulating only 100 chains.")
    lines += ["\\end{table}"]
    return lines


def bootstrap_table(summaries):
    first, second = summaries["first"], summaries["second"]
    columns = [(first, "main"), (first, "all"), (second, "main"), (second, "all")]
    size = lambda label: "$" + label.replace("x", "\\times") + "$"

    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Resampling: variability of the transition probabilities across 100 matrices, each estimated "
             "from a random 80\\% of the training episodes.}",
             "\\label{tab:bootstrap}",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{\\textbf{First-order}} & \\multicolumn{2}{c}{\\textbf{Second-order}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             f" & {size(first['main_label'])} & {size(first['all_label'])} & {size(second['main_label'])} & "
             f"{size(second['all_label'])} \\\\",
             "\\midrule"]
    for key, label in (("bootstrap_mean_abs_diff", "Mean abs.\\ difference from mean"),
                       ("bootstrap_avg_std", "Average SD"), ("bootstrap_max_std", "Maximum SD")):
        lines.append(f"{label} & " + " & ".join(f"{s[f'{key}_{c}']:.4f}" for s, c in columns) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    most_variable = "; ".join(f"{size(s[f'{c}_label'])}: {transition_label(s[f'bootstrap_most_variable_{c}'])}"
                              for s, c in columns)
    lines += note("SD is the standard deviation of each transition probability across the 100 resampled matrices, "
                  "averaged over every cell of the full matrix (including state pairs never observed, which are 0 in "
                  "every sample). Sampling 80\\% of episodes without replacement understates the variation expected "
                  f"between independent datasets of the same size. Most variable transition: {most_variable}.")
    lines += ["\\end{table}"]
    return lines


# --------------------------------------------------------------------------------------------------------------------------------------
# Key Findings (Discussion)
# --------------------------------------------------------------------------------------------------------------------------------------

def key_findings_table(summaries, cv_comparison, order_selection):
    """
    Panel A: model-level results (one value per order). Panel B: results per simulation method.
    order_selection: DataFrame from output/order_selection/order_selection.csv (pooled rows are used).
    """
    first, second = summaries["first"], summaries["second"]
    pooled = order_selection[order_selection["data"] == "Pooled"].set_index("states")

    def stability(text):
        return text.split(": ")[-1].split(" (")[0].capitalize()

    def pair(a, b, fmt):
        return f"{fmt.format(a)} & {fmt.format(b)}"

    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Key findings for the first- and second-order models, combined data. Panel A: results for each "
             "model; panel B: results for each simulation method. Arrows show the direction that is better; the "
             "better order is in bold where the comparison is a test.}",
             "\\label{tab:key_findings}",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lrr}", "\\toprule",
             "\\multicolumn{3}{l}{\\textbf{A. Model}} \\\\",
             "\\midrule",
             " & \\textbf{First-order} & \\textbf{Second-order} \\\\",
             "\\midrule"]

    def bold_lower(a, b, fmt):
        return (f"\\textbf{{{fmt.format(a)}}} & {fmt.format(b)}" if a < b else f"{fmt.format(a)} & \\textbf{{{fmt.format(b)}}}")

    def bold_higher(a, b, fmt):
        return (f"\\textbf{{{fmt.format(a)}}} & {fmt.format(b)}" if a > b else f"{fmt.format(a)} & \\textbf{{{fmt.format(b)}}}")

    lines.append("\\textit{Order selection (all 164 episodes)} & & \\\\")
    for states in (5, 7):
        lines.append(f"\\quad BIC, {states} categories ($\\downarrow$) & "
                     + bold_lower(pooled.loc[states, "bic_1"], pooled.loc[states, "bic_2"], "{:,.1f}") + " \\\\")
    lines.append("\\textit{Leave-one-out, unseen episodes} & & \\\\")
    for key, states in (("main", 5), ("all", 7)):
        diff = cv_comparison.loc[f"Log-likelihood per transition ({'main' if key == 'main' else 'all'} categories)"]
        cells = (bold_lower(first["cv_perplexity"][key], second["cv_perplexity"][key], "{:.2f}")
                 if diff["better"] != "No clear difference" else pair(first["cv_perplexity"][key],
                                                                      second["cv_perplexity"][key], "{:.2f}"))
        lines.append(f"\\quad Perplexity, {states} categories ($\\downarrow$) & {cells} \\\\")
    for key, states in (("main", 5), ("all", 7)):
        diff = cv_comparison.loc[f"Next-contact accuracy ({'main' if key == 'main' else 'all'} categories)"]
        a, b = first["cv"][f"accuracy_{key}"]["mean"], second["cv"][f"accuracy_{key}"]["mean"]
        cells = bold_higher(a, b, "{:.3f}") if diff["better"] != "No clear difference" else pair(a, b, "{:.3f}")
        lines.append(f"\\quad Accuracy, {states} categories ($\\uparrow$) & {cells} \\\\")
    lines.append("\\textit{Estimation (130 training episodes)} & & \\\\")
    lines.append("\\quad Unobserved transitions, 5 / 7 cat. & "
                 f"{first['missing_main']} / {first['missing_all']} & {second['missing_main']} / {second['missing_all']} \\\\")
    lines.append("\\quad Convergence, 5 / 7 cat. & "
                 f"{stability(first['convergence_main'])} / {stability(first['convergence_all'])} & "
                 f"{stability(second['convergence_main'])} / {stability(second['convergence_all'])} \\\\")
    lines.append("\\quad Resampling max.\\ SD, 5 / 7 cat. ($\\downarrow$) & "
                 f"{first['bootstrap_max_std_main']:.3f} / {first['bootstrap_max_std_all']:.3f} & "
                 f"{second['bootstrap_max_std_main']:.3f} / {second['bootstrap_max_std_all']:.3f} \\\\")
    lines.append("\\quad Train vs test MAE, 5 / 7 cat. ($\\downarrow$) & "
                 f"{first['train_test_mae_main']:.4f} / {first['train_test_mae_all_without_out']:.4f} & "
                 f"{second['train_test_mae_main']:.4f} / {second['train_test_mae_all_without_out']:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{1em}", ""]

    # Panel B
    lines += ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
              "\\multicolumn{7}{l}{\\textbf{B. Simulation method}} \\\\",
              "\\midrule",
              " & \\multicolumn{3}{c}{\\textbf{In-sample}} & \\multicolumn{3}{c}{\\textbf{Unseen episodes}} \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
              "\\textbf{Method} & \\shortstack{\\textbf{MAE}\\\\obs.\\ vs pred.} & "
              "\\shortstack{\\textbf{Hellinger}\\\\length} & \\shortstack{\\textbf{Hellinger}\\\\contacts} & "
              "\\shortstack{\\textbf{MAE}\\\\pred.\\ vs test} & \\shortstack{\\textbf{Lev.}\\\\($\\downarrow$)} & "
              "\\shortstack{\\textbf{S-W}\\\\($\\uparrow$)} \\\\",
              "\\midrule"]
    for o, (order, order_label) in enumerate(ORDERS):
        s = summaries[order]
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{order_label}}}}} \\\\")
        for i, (method, _) in enumerate(METHODS):
            m = s["methods"][method]
            if method == "terminating":
                mae_obs, mae_test = s["mae_terminating_without_out"], s["predicted_test_mae_terminating_without_out"]
            else:
                mae_obs, mae_test = m["mae"], m["predicted_test_mae"]
            contacts = np.mean([m["hellinger_visits_by_category"][c] for c in main_categories])
            lev, lev_scr = s["cv"][f"lev_norm_{method}"]["mean"], s["cv"][f"lev_norm_{method}_scrambled"]["mean"]
            sw, sw_scr = s["cv"][f"sw_norm_{method}"]["mean"], s["cv"][f"sw_norm_{method}_scrambled"]["mean"]
            lev_cell = f"{lev:.3f}" + ("$^*$" if lev < lev_scr else "")
            sw_cell = f"{sw:.3f}" + ("$^*$" if sw > sw_scr else "")
            lines.append(f"\\quad {METHOD_NAMES[method]} & {mae_obs:.4f} & {m['hellinger_lengths']:.3f} & "
                         f"{contacts:.3f} & {mae_test:.4f} & {lev_cell} & {sw_cell} \\\\")
        if o < len(ORDERS) - 1:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note(
        "A: BIC from the order-selection test on all episodes (Table~\\ref{tab:order_selection}); perplexity and "
        "accuracy from leave-one-out validation (Table~\\ref{tab:loo}), in bold where the 95\\% CI of the difference "
        "between orders excludes zero (with add-one smoothing; the log-likelihood difference is not significant with "
        "smoothing 0.1). Convergence is for the two tracked transitions (Section~3.2). Resampling SD is the largest "
        "standard deviation of any transition probability (Table~\\ref{tab:bootstrap}). "
        "B: MAE and Hellinger distances are in-sample checks against the 130 training episodes (Tables~"
        "\\ref{tab:hellinger} and~\\ref{tab:matrix_mae}); Hellinger contacts is the average over the five main "
        "categories; terminating-state MAEs leave out histories ending in \\textit{Out}. Lev.\\ and S-W are the "
        "normalised Levenshtein distance and Smith-Waterman score between simulated chains and held-out episodes in "
        "leave-one-out validation; $^*$ better than the scrambled baseline.")
    lines += ["\\end{table}"]
    return lines


def main():
    parser = argparse.ArgumentParser(description="LaTeX tables for the model results")
    parser.add_argument("--data", default="combined", help="Dataset folder in output/ (default: combined)")
    args = parser.parse_args()

    summaries = load_summaries(args.data)
    save(args.data, "table_hellinger.tex", hellinger_table(summaries))
    save(args.data, "table_matrix_mae.tex", matrix_mae_table(summaries))
    save(args.data, "table_similarity.tex", similarity_table(summaries))
    save(args.data, "table_loo.tex", loo_table(summaries, load_cv_comparison(args.data)))
    save(args.data, "table_perturbation.tex", perturbation_table(summaries))
    save(args.data, "table_bootstrap.tex", bootstrap_table(summaries))

    order_selection_file = os.path.join(OUTPUT_ROOT, "order_selection", "order_selection.csv")
    if os.path.exists(order_selection_file):
        import pandas as pd
        save(args.data, "table_key_findings.tex",
             key_findings_table(summaries, load_cv_comparison(args.data), pd.read_csv(order_selection_file)))
    else:
        print("Skipped table_key_findings.tex: run src/order_selection.py first")


if __name__ == "__main__":
    main()
