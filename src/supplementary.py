# --------------------------------------------------------------------------------------------------------------------------------------
# Supplementary Material: Data Requirements for Second-Order Estimation (S1) and Surface Mapping (S2)
# --------------------------------------------------------------------------------------------------------------------------------------
# S1. How many care episodes a second-order model would need for stable transition estimates.
#
#     A second-order transition probability P(next | previous, current) is estimated from the n times its preceding
#     state pair (previous, current) occurred. With a binomial standard error SE = sqrt(p(1 - p) / n), reaching a
#     target SE for a representative probability p = 0.10 needs n = p(1 - p) / SE² observations of the state pair:
#     SE 0.10 -> 9, SE 0.05 -> 36, SE 0.03 -> 100.
#
#     n is counted per observed state pair (a row of the matrix), not per matrix cell. Rows are very uneven, so the
#     average hides the rare state pairs that limit the model. For each target this reports:
#       - the share of observed state pairs that already reach it,
#       - the episodes needed for the average state pair to reach it (episodes x k / mean n),
#       - the episodes needed for 90% and for all observed state pairs to reach it (episodes x k / n of the state pair
#         at the 10th percentile, or of the rarest one).
#     Extra episodes are assumed to have the same mix of contacts as the current ones. State pairs never observed
#     cannot be scaled at all, so the figures are lower bounds.
#
#     Uses the training episodes (130 for the combined data), the data the models and the convergence and bootstrap
#     analyses use.
#
# S2. The complete mapping of every surface label in the data to its category, with contact counts.
#
# Usage: python src/supplementary.py [--data combined]
#        -> output/<data>/supplementary/sample_size.txt and output/<data>/latex/table_s1_*.tex, table_s2_mapping.tex


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import argparse
import math
import os
import sys

import numpy as np


TARGETS = [(0.10, "broad"), (0.05, "intermediate"), (0.03, "precise")]
P = 0.10


def required_observations(se, p=P):
    return math.ceil(round(p * (1 - p) / se ** 2, 6))


# --------------------------------------------------------------------------------------------------------------------------------------
# S1: Data Requirements
# --------------------------------------------------------------------------------------------------------------------------------------

def data_requirements(episodes, categories, order):
    from latex_tables import transition_counts
    runs = transition_counts(episodes, categories, order)
    counts = {}
    for run, n_run in runs.items():
        counts[run[:-1]] = counts.get(run[:-1], 0) + n_run

    k_states = len(categories)
    n = np.array(sorted(counts.values()))
    n_episodes = len(episodes)
    total = int(n.sum())

    result = {
        "states": k_states, "order": order, "episodes": n_episodes, "transitions": total,
        "per_episode": total / n_episodes,
        "possible_histories": k_states ** order, "observed_histories": len(n),
        "possible_cells": k_states ** (order + 1), "unobserved_cells": k_states ** (order + 1) - len(runs),
        "mean_n": n.mean(), "median_n": float(np.median(n)), "min_n": int(n.min()),
        "p10_n": float(np.percentile(n, 10)),
        "targets": [],
    }
    for se, label in TARGETS:
        k = required_observations(se)
        result["targets"].append({
            "se": se, "label": label, "k": k,
            "share_met": float(np.mean(n >= k)),
            "episodes_mean": math.ceil(n_episodes * k / n.mean()),
            "episodes_90": math.ceil(n_episodes * k / result["p10_n"]),
            "episodes_all": math.ceil(n_episodes * k / n.min()),
        })
    return result


def s1_targets_table():
    from latex_tables import note
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             f"\\caption{{Observations of a preceding state pair needed for a target standard error, for a "
             f"representative transition probability $p = {P:.2f}$.}}",
             "\\label{tab:s1_targets}",
             "\\begin{tabular}{lr}", "\\toprule",
             "\\textbf{Target SE} & \\textbf{Required observations ($k$)} \\\\", "\\midrule"]
    for se, label in TARGETS:
        lines.append(f"{se:.2f} ({label}) & {required_observations(se)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note("$k = p(1-p)/\\mathrm{SE}^2$. Transitions within an episode are not independent, so these are "
                  "approximate minimum requirements.")
    lines += ["\\end{table}"]
    return lines


def s1_data_table(results):
    from latex_tables import note
    columns = results  # list of result dicts: first-order 5 and 7 categories, then second-order 5 and 7

    def row(name, values):
        return f"{name} & " + " & ".join(values) + " \\\\"

    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             f"\\caption{{Observations per preceding state (first-order) or state pair (second-order) in the "
             f"{results[0]['episodes']} training episodes.}}",
             "\\label{tab:s1_data}",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{l" + "r" * len(columns) + "}", "\\toprule",
             " & \\multicolumn{2}{c}{\\textbf{First-order}} & \\multicolumn{2}{c}{\\textbf{Second-order}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             " & " + " & ".join(f"{r['states']} cat." for r in columns) + " \\\\",
             "\\midrule",
             row("Transitions", [f"{r['transitions']:,}" for r in columns]),
             row("Transitions per episode ($u$)", [f"{r['per_episode']:.1f}" for r in columns]),
             row("Histories observed / possible", [f"{r['observed_histories']} / {r['possible_histories']}" for r in columns]),
             row("Transitions unobserved / possible",
                 [f"{r['unobserved_cells']} / {r['possible_cells']}" for r in columns]),
             "\\addlinespace",
             "\\multicolumn{" + str(len(columns) + 1) + "}{l}{\\textit{Observations per observed history ($n$)}} \\\\",
             row("\\quad Mean", [f"{r['mean_n']:.1f}" for r in columns]),
             row("\\quad Median", [f"{r['median_n']:.0f}" for r in columns]),
             row("\\quad 10th percentile", [f"{r['p10_n']:.1f}" for r in columns]),
             row("\\quad Minimum", [f"{r['min_n']}" for r in columns]),
             "\\addlinespace",
             "\\multicolumn{" + str(len(columns) + 1) + "}{l}{\\textit{Share of observed histories with $n \\geq k$}} \\\\"]
    for i, (se, label_text) in enumerate(TARGETS):
        k = required_observations(se)
        lines.append(row(f"\\quad $k = {k}$ (SE {se:.2f})", [f"{r['targets'][i]['share_met']:.0%}".replace("%", "\\%")
                                                           for r in columns]))
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note("A history is the state (first-order) or state pair (second-order) a transition is conditioned on; "
                  "$n$ is the number of transitions observed from it. Five-category counts use only runs of contacts "
                  "that all fall in the five categories. Unobserved transitions are counted over the full grid "
                  "(5$^3$ = 125 and 7$^3$ = 343 for second-order), including histories never observed.")
    lines += ["\\end{table}"]
    return lines


def s1_episodes_table(results):
    from latex_tables import note
    second = [r for r in results if r["order"] == 2]
    n_episodes = results[0]["episodes"]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Care episodes required for stable second-order estimates, assuming additional episodes have "
             f"the same mix of contacts as the current {n_episodes} training episodes.}}",
             "\\label{tab:s1_episodes}",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{llrrrr}", "\\toprule",
             " & & & \\multicolumn{3}{c}{\\textbf{Episodes for the target to be met by}} \\\\",
             "\\cmidrule(lr){4-6}",
             "\\textbf{Model} & \\textbf{Target SE} & $k$ & \\shortstack{\\textbf{Average}\\\\\\textbf{state pair}} & "
             "\\shortstack{\\textbf{90\\% of}\\\\\\textbf{state pairs}} & \\shortstack{\\textbf{All observed}\\\\"
             "\\textbf{state pairs}} \\\\",
             "\\midrule"]
    for g, r in enumerate(second):
        for i, t in enumerate(r["targets"]):
            model = f"{r['states']} categories" if i == 0 else ""
            lines.append(f"{model} & {t['se']:.2f} ({t['label']}) & {t['k']} & {t['episodes_mean']:,} & "
                         f"{t['episodes_90']:,} & {t['episodes_all']:,} \\\\")
        if g < len(second) - 1:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note(f"Episodes required $= {n_episodes} \\times k / n$, where $n$ is the mean, 10th-percentile or minimum "
                  f"number of observations per observed state pair (Table~\\ref{{tab:s1_data}}). State pairs never "
                  f"observed are not included, so all values are lower bounds. The all-observed column depends on the "
                  f"single rarest state pair and is therefore very uncertain. Using all 164 episodes instead of the "
                  f"130 training episodes would raise every $n$ by about a quarter, but would not change the number "
                  f"of episodes required, which depends only on how often each state pair occurs per episode.")
    lines += ["\\end{table}"]
    return lines


# --------------------------------------------------------------------------------------------------------------------------------------
# S2: Surface Mapping
# --------------------------------------------------------------------------------------------------------------------------------------

def s2_mapping_table(data):
    from config import surface_categories
    from latex_tables import NAMES, note
    counts = data.groupby(["Surface", "Source"]).size().unstack(fill_value=0)
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\caption{Mapping of every surface label in the data to its surface category, with the number of "
             "contacts in mock and actual care.}",
             "\\label{tab:s2_mapping}",
             "\\begin{tabular}{llrr}", "\\toprule",
             "\\textbf{Category} & \\textbf{Surface label} & \\textbf{Mock} & \\textbf{Actual} \\\\", "\\midrule"]
    categories = list(surface_categories)
    n_labels = 0
    for c, category in enumerate(categories):
        surfaces = [s for s in surface_categories[category] if s in counts.index]
        for i, surface in enumerate(surfaces):
            mock = int(counts.loc[surface].get("Mock", 0))
            actual = int(counts.loc[surface].get("Actual", 0))
            marker = "$^\\dagger$" if mock == 0 else ""
            name = NAMES[category] if i == 0 else ""
            lines.append(f"{name} & {surface}{marker} & {mock:,} & {actual:,} \\\\")
            n_labels += 1
        if c < len(categories) - 1:
            lines.append("\\addlinespace")
    unused = sorted(set(s for v in surface_categories.values() for s in v) - set(counts.index))
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note(f"{n_labels} surface labels. $^\\dagger$ Recorded only in actual care; mapped to the category of the "
                  f"closest mock surface (e.g.\\ GownOn as ApronOn, IVDrip as IV). Curtain and Notes were assigned to "
                  f"Far Patient. In and Out mark the health care worker crossing the room boundary."
                  + (f" Labels in the mapping but not in the data: {', '.join(unused)}." if unused else ""))
    lines += ["\\end{table}"]
    return lines


# --------------------------------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Supplementary material tables")
    parser.add_argument("--data", default="combined", help="Dataset (default: combined)")
    args = parser.parse_args()
    os.environ["MARKOV_DATA"] = args.data

    from config import OUTPUT_ROOT, all_categories, main_categories
    from data_loader import all_data, df_train
    from latex_tables import full_width

    episodes = (df_train.groupby("ActivityID", sort=False)
                .agg(Source=("Source", "first"), states=("SurfaceCategories", list)).reset_index())

    results = [data_requirements(episodes, categories, order)
               for order in (1, 2) for categories in (main_categories, all_categories)]

    out_dir = os.path.join(OUTPUT_ROOT, args.data, "supplementary")
    latex_dir = os.path.join(OUTPUT_ROOT, args.data, "latex")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)

    lines = [f"DATA REQUIREMENTS FOR SECOND-ORDER ESTIMATION ({args.data}, {len(episodes)} training episodes)", ""]
    for r in results:
        lines += [f"--- Order {r['order']}, {r['states']} categories ---",
                  f"Transitions: {r['transitions']:,} ({r['per_episode']:.1f} per episode)",
                  f"Histories observed: {r['observed_histories']} of {r['possible_histories']}; "
                  f"transitions unobserved: {r['unobserved_cells']} of {r['possible_cells']}",
                  f"Observations per observed history: mean {r['mean_n']:.1f}, median {r['median_n']:.0f}, "
                  f"10th percentile {r['p10_n']:.1f}, minimum {r['min_n']}"]
        for t in r["targets"]:
            lines.append(f"  SE {t['se']:.2f} (k = {t['k']}): {t['share_met']:.0%} of histories met; episodes needed: "
                         f"average {t['episodes_mean']:,}, 90% {t['episodes_90']:,}, all {t['episodes_all']:,}")
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "sample_size.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)

    for name, table in (("table_s1_targets.tex", s1_targets_table()), ("table_s1_data.tex", s1_data_table(results)),
                        ("table_s1_episodes.tex", s1_episodes_table(results)),
                        ("table_s2_mapping.tex", s2_mapping_table(all_data))):
        with open(os.path.join(latex_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(full_width(line) for line in table) + "\n")
        print(f"Saved {os.path.join(latex_dir, name)}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
