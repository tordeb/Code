# --------------------------------------------------------------------------------------------------------------------------------------
# Markov Chain Order Selection: Likelihood-Ratio (G²) Tests and BIC
# --------------------------------------------------------------------------------------------------------------------------------------
# Tests whether each extra step of history improves the fit: order 0 vs 1, and order 1 vs 2 (Anderson & Goodman, 1957),
# and which order BIC selects. Done for mock, actual and pooled (mock + actual) episodes, with 5 and 7 categories.
#
# All three orders are fitted to the SAME transitions, so their log-likelihoods are comparable: every run of three
# consecutive contacts in an episode that all fall in the category list (x_{t-2}, x_{t-1}, x_t), with x_t the contact
# being predicted. This is the set the leave-one-out validation scores (cross_validation.scored_transitions), and the
# same way the 5-category data is built elsewhere: transitions to or from In and Out are dropped, not joined across.
#
#   Order 0: P(x_t)                     log L0 = Σ n_k   ln(n_k   / N)
#   Order 1: P(x_t | x_{t-1})           log L1 = Σ n_jk  ln(n_jk  / n_j.)
#   Order 2: P(x_t | x_{t-2}, x_{t-1})  log L2 = Σ n_ijk ln(n_ijk / n_ij.)
#
#   G² (order m vs m+1) = 2 (log L_{m+1} - log L_m), compared with chi-squared on df = k_{m+1} - k_m.
#   k_m = free parameters = Σ over observed histories of (next states observed from that history - 1), so transitions
#         that never occur (e.g. In -> In) are not counted as parameters.
#   BIC = -2 log L + k ln N. The BIC penalty for the higher order is (k_{m+1} - k_m) ln N, on the same scale as G²:
#         BIC prefers the higher order only if G² is larger than the penalty.
#
# All episodes are used (not the training set): the table describes the data, like the mock vs actual tables.
#
# Usage: python src/order_selection.py   -> output/order_selection/ and output/data_comparison/latex/table_order_selection.tex


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import os

import numpy as np
import pandas as pd
from scipy import stats

from config import OUTPUT_ROOT, all_categories, main_categories
from data_loader import load_data
from latex_tables import note, p_latex, save, transition_counts


OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "order_selection")

GROUPS = {"Mock": ["Mock"], "Actual": ["Actual"], "Pooled": ["Mock", "Actual"]}
STATE_SETS = {5: main_categories, 7: all_categories}


# --------------------------------------------------------------------------------------------------------------------------------------
# Log-Likelihood and Parameters of Each Order
# --------------------------------------------------------------------------------------------------------------------------------------

def fit(triples, order):
    """
    Maximum log-likelihood and number of free parameters of an order-`order` chain, fitted to the triples.

    triples: {(x_{t-2}, x_{t-1}, x_t): count}
    """
    counts = {}
    for triple, n in triples.items():
        key = triple[3 - order - 1:]  # (x_t), (x_{t-1}, x_t) or (x_{t-2}, x_{t-1}, x_t)
        counts[key] = counts.get(key, 0) + n

    history_totals, next_states = {}, {}
    for key, n in counts.items():
        history = key[:-1]
        history_totals[history] = history_totals.get(history, 0) + n
        next_states[history] = next_states.get(history, 0) + 1

    loglik = sum(n * np.log(n / history_totals[key[:-1]]) for key, n in counts.items())
    parameters = sum(k - 1 for k in next_states.values())
    return loglik, parameters


def order_selection(episodes, categories):
    triples = transition_counts(episodes, categories, 2)
    n = sum(triples.values())
    fits = {order: fit(triples, order) for order in (0, 1, 2)}

    result = {"transitions": n}
    for low, high in ((0, 1), (1, 2)):
        g2 = 2 * (fits[high][0] - fits[low][0])
        df = fits[high][1] - fits[low][1]
        result[f"g2_{low}{high}"] = g2
        result[f"df_{low}{high}"] = df
        result[f"p_{low}{high}"] = stats.chi2.sf(g2, df)
        result[f"bic_penalty_{low}{high}"] = df * np.log(n)

    bic = {order: -2 * loglik + k * np.log(n) for order, (loglik, k) in fits.items()}
    for order, (loglik, k) in fits.items():
        result[f"loglik_{order}"] = loglik
        result[f"parameters_{order}"] = k
        result[f"bic_{order}"] = bic[order]
    result["bic_order"] = min(bic, key=bic.get)
    return result


# --------------------------------------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------------------------------------

def episodes_by_source(data):
    episodes = (data.groupby("ActivityID", sort=False)
                .agg(Source=("Source", "first"), states=("SurfaceCategories", list))
                .reset_index())
    return episodes


def latex_table(rows):
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Markov chain order selection. $G^2$ is the likelihood-ratio statistic for adding one more "
             "step of history (Anderson and Goodman, 1957); the BIC penalty is the extra parameters times $\\ln N$, "
             "on the same scale as $G^2$, so BIC prefers the higher order only when $G^2$ exceeds it.}",
             "\\label{tab:order_selection}",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{lrrrrrrrc}", "\\toprule",
             " & & & \\multicolumn{2}{c}{\\textbf{Order 0 vs 1}} & \\multicolumn{3}{c}{\\textbf{Order 1 vs 2}} & \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}",
             "\\textbf{Data} & \\textbf{States} & $N$ & $G^2$ (df) & $p$ & $G^2$ (df) & $p$ & \\shortstack{\\textbf{BIC}\\\\"
             "\\textbf{penalty}} & \\shortstack{\\textbf{BIC}\\\\\\textbf{order}} \\\\",
             "\\midrule"]

    for g, group in enumerate(GROUPS):
        group_rows = [r for r in rows if r["data"] == group]
        for i, r in enumerate(group_rows):
            label = group if i == 0 else ""
            lines.append(f"{label} & {r['states']} & {r['transitions']:,} & "
                         f"{r['g2_01']:.1f} ({r['df_01']}) & {p_latex(r['p_01'])} & "
                         f"{r['g2_12']:.1f} ({r['df_12']}) & {p_latex(r['p_12'])} & {r['bic_penalty_12']:.1f} & {r['bic_order']} \\\\")
        if g < len(GROUPS) - 1:
            lines.append("\\addlinespace")

    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note(f"$N$ is the number of transitions with at least two earlier contacts in the same episode, all in the "
                  f"state set; all three orders are fitted to these same transitions, using every episode. Degrees of "
                  f"freedom count only transitions observed in the data. $p$-values use the chi-squared approximation, which treats "
                  f"transitions as independent and is unreliable for sparse second-order cells; they are not "
                  f"corrected for multiple testing.")
    lines += ["\\end{table}"]
    return lines


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    episodes = episodes_by_source(load_data())

    rows = []
    for group, sources in GROUPS.items():
        group_episodes = episodes[episodes["Source"].isin(sources)]
        for states, categories in STATE_SETS.items():
            rows.append({"data": group, "episodes": len(group_episodes), "states": states,
                         **order_selection(group_episodes, categories)})

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUTPUT_DIR, "order_selection.csv"), index=False)

    lines = ["MARKOV CHAIN ORDER SELECTION (all episodes; all orders fitted to the same transitions)", ""]
    for r in rows:
        lines += [f"--- {r['data']} ({r['episodes']} episodes), {r['states']} states: N = {r['transitions']:,} transitions ---",
                  f"Log-likelihood (parameters): order 0 {r['loglik_0']:.1f} ({r['parameters_0']}), "
                  f"order 1 {r['loglik_1']:.1f} ({r['parameters_1']}), order 2 {r['loglik_2']:.1f} ({r['parameters_2']})",
                  f"Order 0 vs 1: G² = {r['g2_01']:.1f}, df = {r['df_01']}, p = {r['p_01']:.3g}, "
                  f"BIC penalty = {r['bic_penalty_01']:.1f}",
                  f"Order 1 vs 2: G² = {r['g2_12']:.1f}, df = {r['df_12']}, p = {r['p_12']:.3g}, "
                  f"BIC penalty = {r['bic_penalty_12']:.1f}",
                  f"BIC: order 0 {r['bic_0']:.1f}, order 1 {r['bic_1']:.1f}, order 2 {r['bic_2']:.1f} "
                  f"-> selects order {r['bic_order']}", ""]
    text = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "order_selection.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

    save("table_order_selection.tex", latex_table(rows))
    print(f"Saved to {OUTPUT_DIR} and output/data_comparison/latex/table_order_selection.tex")


if __name__ == "__main__":
    main()
