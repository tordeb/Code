# --------------------------------------------------------------------------------------------------------------------------------------
# LaTeX Tables for the Paper: Mock vs Actual Care (Results 3.1)
# --------------------------------------------------------------------------------------------------------------------------------------
# Writes ready-to-paste LaTeX tables to output/data_comparison/latex/. Called at the end of dataset_comparison.py.
#
#   table_mva_descriptive.tex   Episodes, lengths, category shares and how episodes start / end, per group
#   table_mva_tests.tex         Tests comparing the groups (length, category mix, transition matrices, start / end)
#   table_mva_first_5x5.tex     First-order 5-category matrices, mock and actual (paper Table 3)
#   table_mva_first_7x7.tex     First-order 7-category matrices, mock and actual (paper Table 4)
#   table_mva_second_mock.tex   Second-order 5-category matrix, mock (paper p.8)
#   table_mva_second_actual.tex Second-order 5-category matrix, actual (paper p.9)
#   table_mva_wilson.tex        Selected transitions with 95% Wilson score intervals (paper Table 5)
#
# All tables use every episode (not the training set): they describe the data rather than a fitted model.
# The 5-category counts keep only transitions where every state is one of the 5 main categories, the same way the
# models build their 5-category data (transitions to or from In and Out are dropped, not joined across).
#
# LaTeX packages needed: booktabs, multirow


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import os

import numpy as np

from config import OUTPUT_ROOT, all_categories, main_categories


LATEX_DIR = os.path.join(OUTPUT_ROOT, "data_comparison", "latex")

# Names used in the paper
NAMES = {"In": "In", "Equipment": "Equipment", "Patient": "Patient", "NearPatient": "Near Patient",
         "FarPatient": "Far Patient", "HygieneArea": "Hygiene Areas", "Out": "Out"}
SHORT_NAMES = {"In": "In", "Equipment": "Equip.", "Patient": "Patient", "NearPatient": "Near Pt.",
               "FarPatient": "Far Pt.", "HygieneArea": "Hygiene", "Out": "Out"}
TRANSITION_NAMES = {"Equipment": "Equipment", "Patient": "Patient", "NearPatient": "Near Patient",
                    "FarPatient": "Far Patient", "HygieneArea": "Hygiene"}

GROUPS = {
    "Mock": {"Source": ["Mock"], "RoomType": ["SB", "MB"]},
    "Actual SB": {"Source": ["Actual"], "RoomType": ["SB"]},
    "Actual MB": {"Source": ["Actual"], "RoomType": ["MB"]},
    "Actual": {"Source": ["Actual"], "RoomType": ["SB", "MB"]},
}

# Transitions discussed in the text (paper Table 5): (history..., next state)
WILSON_TRANSITIONS = [
    ("Patient", "Equipment"),
    ("Patient", "HygieneArea"),
    ("Equipment", "Patient"),
    ("Patient", "Patient"),
    ("Equipment", "Patient", "Equipment"),
    ("HygieneArea", "Patient", "Equipment"),
    ("HygieneArea", "Equipment", "Patient"),
    ("Equipment", "Equipment", "Patient"),
]


# --------------------------------------------------------------------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------------------------------------------------------------------

def select(episodes, condition):
    keep = np.ones(len(episodes), dtype=bool)
    for column, values in condition.items():
        keep &= episodes[column].isin(values).values
    return episodes[keep]


def transition_counts(episodes, categories, order):
    """
    Counts of each (history, next state), using only runs of order + 1 consecutive contacts that all fall in
    `categories`. Returns a dict {(history..., next): count}.
    """
    counts = {}
    for states in episodes["states"]:
        for i in range(len(states) - order):
            run = tuple(states[i:i + order + 1])
            if all(state in categories for state in run):
                counts[run] = counts.get(run, 0) + 1
    return counts


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z / (1 + z ** 2 / n) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return centre - half, centre + half


# --------------------------------------------------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------------------------------------------------

def p_latex(p):
    if np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def cell(k, n):
    """'0.576 (208)': probability with the count in parentheses."""
    return f"{k / n:.3f} ({k})"


def note(text):
    return ["\\par\\smallskip", "\\begin{minipage}{\\linewidth}\\footnotesize", text, "\\end{minipage}"]


def stacked(text):
    """Bold column heading split over two lines at its first space (keeps wide tables within the text width)."""
    if " " not in text:
        return f"\\textbf{{{text}}}"
    top, bottom = text.split(" ", 1)
    return f"\\shortstack{{\\textbf{{{top}}}\\\\\\textbf{{{bottom}}}}}"


def full_width(line):
    """Stretch a tabular to the full text width (extra space goes between columns; the font size is unchanged)."""
    if line.startswith("\\begin{tabular}{"):
        return "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}" + line[len("\\begin{tabular}{"):]
    return "\\end{tabular*}" if line == "\\end{tabular}" else line


def save(name, lines):
    os.makedirs(LATEX_DIR, exist_ok=True)
    with open(os.path.join(LATEX_DIR, name), "w", encoding="utf-8") as f:
        f.write("\n".join(full_width(line) for line in lines) + "\n")


# --------------------------------------------------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------------------------------------------------

def first_order_table(groups, categories, short, caption, label, footnote):
    """First-order matrices for each group, stacked, with probability (count) per cell and row totals n."""
    names = SHORT_NAMES if short else NAMES
    k = len(categories)
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small" if not short else "\\footnotesize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{3pt}" if short else "\\setlength{\\tabcolsep}{4pt}",
             f"\\begin{{tabular}}{{ll{'r' * k}r}}", "\\toprule",
             f" & & \\multicolumn{{{k}}}{{c}}{{\\textbf{{To}}}} & \\\\",
             f"\\cmidrule(lr){{3-{k + 2}}}",
             "\\textbf{Care} & \\textbf{From} & "
             + " & ".join(stacked(names[c]) for c in categories) + " & $n$ \\\\", "\\midrule"]

    for g, (group_name, episodes) in enumerate(groups):
        counts = transition_counts(episodes, categories, 1)
        rows = []
        for current in categories:
            row = [counts.get((current, nxt), 0) for nxt in categories]
            n = sum(row)
            if n == 0:
                continue  # a state never left (e.g. Out in mock care) has no row
            rows.append(f"{names[current]} & " + " & ".join(cell(x, n) for x in row) + f" & {n:,} \\\\")
        rows[0] = f"\\multirow{{{len(rows)}}}{{*}}{{{group_name}}} & " + rows[0]
        rows[1:] = [" & " + row for row in rows[1:]]
        lines += rows
        lines.append("\\midrule" if g < len(groups) - 1 else "\\bottomrule")

    lines += ["\\end{tabular}"] + note(footnote) + ["\\end{table}"]
    return lines


def second_order_table(group_name, episodes, label):
    """Second-order 5-category matrix for one group: every state pair, '--' for pairs never observed."""
    counts = transition_counts(episodes, main_categories, 2)
    k = len(main_categories)
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             f"\\caption{{Second-order transition probability matrix for the five-category model, {group_name.lower()} "
             f"care episodes. Cells give the maximum-likelihood estimate with the underlying transition count in "
             f"parentheses.}}",
             f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{4pt}",
             f"\\begin{{tabular}}{{ll{'r' * k}r}}", "\\toprule",
             f"\\multicolumn{{2}}{{c}}{{\\textbf{{Preceding pair}}}} & \\multicolumn{{{k}}}{{c}}{{\\textbf{{To}}}} & \\\\",
             f"\\cmidrule(lr){{1-2}}\\cmidrule(lr){{3-{k + 2}}}",
             "$x_{j-1}$ & $x_j$ & " + " & ".join(stacked(NAMES[c]) for c in main_categories)
             + " & $n$ \\\\", "\\midrule"]

    unobserved = 0
    for p, previous in enumerate(main_categories):
        for current in main_categories:
            row = [counts.get((previous, current, nxt), 0) for nxt in main_categories]
            n = sum(row)
            if n == 0:
                unobserved += 1
                cells = " & ".join(["--"] * k)
            else:
                cells = " & ".join(cell(x, n) for x in row)
            lines.append(f"{NAMES[previous]} & {NAMES[current]} & {cells} & {n:,} \\\\")
        if p < len(main_categories) - 1:
            lines.append("\\addlinespace")

    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note(f"Rows sum to 1. $n$ is the number of transitions observed from that state pair, using all "
                  f"{len(episodes)} {group_name.lower()} episodes. {unobserved} of {k * k} state pairs were "
                  f"unobserved (--). Estimates from state pairs with small $n$ are unstable; see "
                  f"Table~\\ref{{tab:mva_wilson}}.")
    lines += ["\\end{table}"]
    return lines


def wilson_table(groups):
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Point estimates, transition counts and 95\\% Wilson score intervals for the transitions "
             "discussed in the text, five-category model.}",
             "\\label{tab:mva_wilson}",
             "\\begin{tabular}{llrrr}", "\\toprule",
             "\\textbf{Order} & \\textbf{Transition} & \\textbf{Count} & \\textbf{Estimate} & \\textbf{95\\% CI} \\\\",
             "\\midrule"]

    for g, (group_name, episodes) in enumerate(groups):
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{group_name} care}}}} \\\\")
        counts = {1: transition_counts(episodes, main_categories, 1), 2: transition_counts(episodes, main_categories, 2)}
        for transition in WILSON_TRANSITIONS:
            order = len(transition) - 1
            history, nxt = transition[:-1], transition[-1]
            k = counts[order].get(transition, 0)
            n = sum(counts[order].get(history + (c,), 0) for c in main_categories)
            low, high = wilson_interval(k, n)
            if order == 1:
                label = f"{TRANSITION_NAMES[history[0]]} $\\rightarrow$ {TRANSITION_NAMES[nxt]}"
                if history[0] == nxt:
                    label += " (self)"
            else:
                label = (f"({TRANSITION_NAMES[history[0]]}, {TRANSITION_NAMES[history[1]]}) "
                         f"$\\rightarrow$ {TRANSITION_NAMES[nxt]}")
            order_name = "First" if order == 1 else "Second"
            lines.append(f"{order_name} & {label} & {k}/{n:,} & {k / n:.3f} & [{low:.3f}, {high:.3f}] \\\\")
        if g < len(groups) - 1:
            lines.append("\\addlinespace")

    lines += ["\\bottomrule", "\\end{tabular}"]
    lines += note("Intervals are Wilson score intervals for a binomial proportion and do not account for "
                  "within-episode dependence; they are therefore lower bounds on the true uncertainty.")
    lines += ["\\end{table}"]
    return lines


def descriptive_table(episodes, results):
    """One column per group: size, lengths, category shares, start / end behaviour and within-group similarity."""
    groups = {name: select(episodes, condition) for name, condition in GROUPS.items()}
    within = {
        "Mock": results["mock_vs_actual_sb"]["levenshtein"]["within_a"],
        "Actual SB": results["mock_vs_actual_sb"]["levenshtein"]["within_b"],
        "Actual MB": results["actual_sb_vs_mb"]["levenshtein"]["within_b"],
        "Actual": results["mock_vs_actual_all"]["levenshtein"]["within_b"],
    }
    headers = ["Mock", "Actual (SB)", "Actual (MB)", "Actual (all)"]

    def row(label, values):
        return f"{label} & " + " & ".join(values) + " \\\\"

    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Mock and actual care episodes compared. SB = single-bed room, MB = multi-bed room.}",
             "\\label{tab:mva_descriptive}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             " & " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\", "\\midrule",
             row("Episodes", [f"{len(g)}" for g in groups.values()]),
             row("Contacts", [f"{g['length'].sum():,}" for g in groups.values()]),
             row("Episode length, mean (SD)",
                 [f"{g['length'].mean():.1f} ({g['length'].std(ddof=1):.1f})" for g in groups.values()]),
             row("Episode length, median", [f"{g['length'].median():.1f}" for g in groups.values()]),
             "\\addlinespace",
             f"\\multicolumn{{5}}{{l}}{{\\textit{{Share of contacts (\\%)}}}} \\\\"]

    for i, category in enumerate(all_categories):
        shares = []
        for g in groups.values():
            counts = np.stack(g["category_counts"].values).sum(axis=0)
            shares.append(f"{100 * counts[i] / counts.sum():.1f}")
        lines.append(row(f"\\quad {NAMES[category]}", shares))

    lines += ["\\addlinespace", f"\\multicolumn{{5}}{{l}}{{\\textit{{Episodes (\\%)}}}} \\\\",
              row("\\quad Start with \\textit{In}", [f"{100 * g['starts_in'].mean():.1f}" for g in groups.values()]),
              row("\\quad End with \\textit{Out}", [f"{100 * g['ends_out'].mean():.1f}" for g in groups.values()]),
              row("\\quad Leave and re-enter", [f"{100 * g['re_entry'].mean():.1f}" for g in groups.values()]),
              "\\addlinespace",
              row("Within-group Levenshtein", [f"{within[name]:.3f}" for name in groups]),
              "\\bottomrule", "\\end{tabular}"]
    lines += note("Within-group Levenshtein is the mean normalised Levenshtein distance between all pairs of "
                  "episodes in the group (0 = identical, 1 = completely different).")
    lines += ["\\end{table}"]
    return lines


def tests_table(results):
    """One column per comparison: each test's statistic with its p-value in parentheses."""
    keys = ["mock_vs_actual_sb", "mock_vs_actual_all", "actual_sb_vs_mb"]
    headers = ["Mock vs actual (SB)", "Mock vs actual (all)", "Actual SB vs MB"]

    def row(label, values):
        return f"{label} & " + " & ".join(values) + " \\\\"

    def stat_p(value, p, fmt):
        return f"{value:{fmt}} ({p_latex(p)})"

    r = {key: results[key] for key in keys}
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Tests comparing groups of care episodes. Each cell gives the statistic with its $p$-value in "
             "parentheses.}",
             "\\label{tab:mva_tests}",
             "\\begin{tabular}{lrrr}", "\\toprule",
             " & " + " & ".join(stacked(h) for h in headers) + " \\\\",
             " & " + " & ".join(f"({r[k]['n_a']} vs {r[k]['n_b']})" for k in keys) + " \\\\", "\\midrule",
             f"\\multicolumn{{4}}{{l}}{{\\textit{{Episode length}}}} \\\\",
             row("\\quad Mann-Whitney $p$", [p_latex(r[k]["length"]["mannwhitney_p"]) for k in keys]),
             row("\\quad Hellinger distance", [f"{r[k]['length']['hellinger']:.3f}" for k in keys]),
             "\\addlinespace",
             row("Category mix: TVD", [stat_p(r[k]["category_share"]["tvd"], r[k]["category_share"]["permutation_p"],
                                              ".3f") for k in keys]),
             "\\addlinespace",
             f"\\multicolumn{{4}}{{l}}{{\\textit{{First-order transition matrices}}}} \\\\",
             row("\\quad MAE, 5 categories", [stat_p(r[k]["transitions"]["mae_main"], r[k]["transitions"]["mae_main_p"],
                                                     ".4f") for k in keys]),
             row("\\quad MAE, 7 categories", [stat_p(r[k]["transitions"]["mae_all"], r[k]["transitions"]["mae_all_p"],
                                                     ".4f") for k in keys])]
    for name, label in (("g2_main", "5 categories"), ("g2_all", "7 categories")):
        lines.append(row(f"\\quad $G^2$ (df), {label}",
                         [f"{r[k]['transitions'][name]['g2']:.1f} ({r[k]['transitions'][name]['df']}), "
                          f"{p_latex(r[k]['transitions'][name]['permutation_p'])}" for k in keys]))

    labels = {"starts_in": "Start with \\textit{In}", "ends_out": "End with \\textit{Out}",
              "re_entry": "Leave and re-enter"}
    lines += ["\\addlinespace", f"\\multicolumn{{4}}{{l}}{{\\textit{{Episodes, Fisher's exact $p$}}}} \\\\"]
    for column, label in labels.items():
        lines.append(row(f"\\quad {label}", [p_latex(r[k]["start_end"][column]["fisher_p"]) for k in keys]))

    lines += ["\\addlinespace",
              row("Between-group Levenshtein", [f"{r[k]['levenshtein']['between']:.3f}" for k in keys]),
              "\\bottomrule", "\\end{tabular}"]
    lines += note("TVD = total variation distance between the two groups' shares of contacts in each category; MAE = "
                  "mean absolute error between the two groups' transition probability matrices. $p$-values for TVD, "
                  "MAE and $G^2$ come from permutation tests (2,000 random reallocations of whole episodes between "
                  "the groups), which allow for transitions within an episode being related. $p$-values are not "
                  "corrected for multiple testing. Between-group Levenshtein is the mean normalised distance between "
                  "episodes from different groups; compare it with the within-group values in "
                  "Table~\\ref{tab:mva_descriptive}.")
    lines += ["\\end{table}"]
    return lines


# --------------------------------------------------------------------------------------------------------------------------------------
# Write All Tables
# --------------------------------------------------------------------------------------------------------------------------------------

def out_of_room_counts(episodes):
    """Episodes with contacts after their first Out, and episodes that never reach Out."""
    after_out = sum("Out" in states[:-1] for states in episodes["states"])
    never_out = sum("Out" not in states for states in episodes["states"])
    return after_out, never_out


def write_mock_vs_actual_tables(episodes, results):
    """
    episodes: DataFrame from dataset_comparison.summarise_episodes
    results: {comparison key: result of dataset_comparison.compare_groups}
    """
    mock = select(episodes, GROUPS["Mock"])
    actual = select(episodes, GROUPS["Actual"])
    groups = [("Mock", mock), ("Actual", actual)]
    g2 = results["mock_vs_actual_all"]["transitions"]["g2_main"]

    save("table_mva_descriptive.tex", descriptive_table(episodes, results))
    save("table_mva_tests.tex", tests_table(results))

    save("table_mva_first_5x5.tex", first_order_table(
        groups, main_categories, short=False,
        caption="First-order transition probability matrices for the five-category model, estimated separately for "
                "mock and actual care episodes. Cells give the maximum-likelihood estimate with the underlying "
                "transition count in parentheses.",
        label="tab:mva_first_5x5",
        footnote=f"Rows sum to 1. $n$ is the total number of transitions observed from that state; transitions to or "
                 f"from \\textit{{In}} and \\textit{{Out}} are excluded. Mock: {len(mock)} episodes; actual: "
                 f"{len(actual)} episodes. Transition matrices are not homogeneous between care types "
                 f"($G^2 = {g2['g2']:.1f}$, df $= {g2['df']}$, permutation $p$ {p_latex(g2['permutation_p'])})."))

    mock_after, mock_never = out_of_room_counts(mock)
    actual_after, actual_never = out_of_room_counts(actual)
    save("table_mva_first_7x7.tex", first_order_table(
        groups, all_categories, short=True,
        caption="First-order transition probability matrices for the seven-category model, which adds the "
                "\\textit{In} and \\textit{Out} room-crossing states used by the terminating-state simulation.",
        label="tab:mva_first_7x7",
        footnote=f"\\textit{{Out}} is treated as absorbing in simulation, but is not absorbing in the observed data: "
                 f"{actual_after} of {len(actual)} actual and {mock_after} of {len(mock)} mock episodes record further "
                 f"contacts after leaving the room, and {actual_never} actual and {mock_never} mock episodes never "
                 f"reach \\textit{{Out}} and are therefore right-censored. Transitions out of \\textit{{Out}} and into "
                 f"\\textit{{In}} correspond to these post-exit contacts and to room re-entry within a single recorded "
                 f"activity. A state with no row was never left."))

    save("table_mva_second_mock.tex", second_order_table("Mock", mock, "tab:mva_second_mock"))
    save("table_mva_second_actual.tex", second_order_table("Actual", actual, "tab:mva_second_actual"))
    save("table_mva_wilson.tex", wilson_table(groups))

    print(f"LaTeX tables saved to {LATEX_DIR}")
