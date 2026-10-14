# Code Notes: First- and Second-Order Markov Chain Analysis

These notes explain what each file in `src/` does and how it works, for both the first-order and the second-order model.

Both models are run by the **same code**. The only setting that changes between them is `ORDER` in `config.py`:

- **First-order (`ORDER = 1`):** the next state depends only on the current state, `P(next | current)`.
- **Second-order (`ORDER = 2`):** the next state depends on the previous and current state, `P(next | previous, current)`.

The states the next state depends on are called the **history**: one state for first-order, a pair of states for second-order. Wherever the two models differ, these notes say so.


# File Structure

There are 9 files in `src/`. You can also run everything through Docker, which saves installing the Python packages locally (`dockerfile`, `requirements.txt`).

1. The Data: `data/movsdf.rbind_orientationcorrected.csv`
   - 62 observed care periods, with each surface contact by the HCW recorded,
   - Other columns include: care type, HCW type, orientation, date, time, and surface category.

2. Configuration and Constants: `config.py`
   - The Markov chain order (1 or 2), set by `main.py`,
   - The surface categories, split into "main" (5) and "all" (7),
   - The matrix labels: 5x5 and 7x7 (first-order), or 25x5 and 49x7 (second-order),
   - The transitions tracked in the convergence analysis,
   - File paths, the number of simulations, the maximum terminating-chain length, and the random seed.

3. Loading the Data: `data_loader.py`
   - Loads the data,
   - Splits it into 80% train data and 20% test data.

4. Generating the Transition Matrices: `transition_matrices.py`
   - The transition matrix function (used for both orders),
   - Starting-state (first-order) or starting-pair (second-order) probabilities,
   - 'Test' matrices from the test data,
   - The Mean Absolute Error (MAE) function used everywhere in the analysis,
   - A convergence analysis showing how the transition probabilities change as more training data is added.

5. Running Simulations: `simulations.py`
   - The simulation function (used for both orders),
   - Fits a distribution to the observed care-period lengths,
   - Runs the three simulation methods (fixed length, fitted length, terminating state),
   - Calculates the predicted transition matrices.

6. Comparison Metrics: `metrics.py`
   - Compares the observed transition matrices with the predicted ones,
   - Tests against the 'test' matrices,
   - Hellinger distance,
   - Sequence comparison metrics from DNA analysis:
      - Levenshtein Distance (absolute and normalised edit distance),
      - Needleman-Wunsch Algorithm (global sequence alignment),
      - Smith-Waterman Algorithm (local sequence alignment).
   - Sensitivity analysis (matrix perturbation) and bootstrap analysis,
   - Chi-squared tests,
   - Transitions that never occurred in the training data.

7. Plotting of Graphs: `plotting.py`
   - Probability against care period length,
   - Probability against number of contacts to each category,
   - Levenshtein Distance, Needleman-Wunsch and Smith-Waterman distributions,
   - Convergence.

8. Main: `main.py`
   - Runs the other files for one order, or for both,
   - Writes the results to a `.txt` file and a `summary.json` file.

9. Comparison: `compare.py`
   - Reads both `summary.json` files and puts the key results for first- and second-order side by side.

`data_exploration.py` is a separate script for looking at the raw data. It is not part of the analysis.


# Project Goals

1. Train data,
2. Test data,
3. Produce transition matrices for the observed data, from predictions, and from predictions using unseen data,
4. Produce 1000+ chains,
5. Compare these matrices and chains,
6. Plot these comparisons,
7. Compare the first-order chain against the second-order chain, using identical methods for both.


---

# 1. Running the Code

```
python src/main.py --order both     # first-order, second-order, then the comparison
python src/main.py --order 1        # first-order only
python src/main.py --order 2        # second-order only
```

`--order both` runs each order in its own Python process, one after the other, and then runs `compare.py`. Running them separately means the two models share the code but nothing else. Results go to:

- `output/first_order/`: `first_order_results.txt`, `summary.json`, and the figures,
- `output/second_order/`: `second_order_results.txt`, `summary.json`, and the figures,
- `output/comparison/`: `comparison_summary.txt`, `comparison_summary.csv`, and `comparison_metrics.png`.

The random seed (`--seed`, default 42) is the same for both orders. Running the code again with the same seed gives exactly the same results.


---

# 2. Configuration and Constants

# Category Definitions

   ## 5 Main Categories

      ### 1. Equipment
      - Equipment (general)
      - IV (Intravenous equipment)
      - ObsTrolley (Observation trolley)
      - Sharps (Sharps container)
      - Stethoscope
      - Tray

      ### 2. Patient
      - Direct contact with patient

      ### 3. NearPatient
      - Bed
      - Chair
      - Table

      ### 4. FarPatient
      - Door
      - Other (miscellaneous surfaces)

      ### 5. HygieneArea
      - Alc (Alcohol gel dispenser)
      - AlcOutside (Alcohol gel outside room)
      - ApronOff (Apron disposal)
      - ApronOn (Apron donning area)
      - GlovesOff (Glove removal)
      - GlovesOn (Glove donning)
      - PaperTowel (Paper towel dispenser)
      - Sink
      - Soap (Soap dispenser)
      - Waste (Waste bin)
      - Wipes (Cleaning wipes)

   ## 7 Categories
   Main categories plus two terminating states:

      ### 6. In
      - Entry point for HCW to enter the room

      ### 7. Out
      - Exit point for HCW to leave the room

   ## Purpose of terminating states:
   - Real care episodes have natural start (In) and end (Out) points, with likely transitions after/before these points,
   - The all-categories matrix captures complete care episodes,
   - Allows simulation of variable-length care periods.

# Matrix Sizes

Each row of a transition matrix is a history, and each column is a possible next state.

| | Main categories | All categories |
|---|---|---|
| First-order (history = 1 state) | 5 rows x 5 columns = **5x5** | 7 x 7 = **7x7** |
| Second-order (history = pair of states) | 5 x 5 = 25 rows x 5 columns = **25x5** | 7 x 7 = 49 rows x 7 columns = **49x7** |

# Other Settings

- `n_simulations = 1000`: chains simulated for each method,
- `MAX_TERMINATING_LENGTH = 100`: a terminating-state chain stops at this length if it never reaches "Out" (same for both orders),
- `SEED = 42`: the random seed, so results can be reproduced.


---

# 3. Loading the Data

# Splitting of Data

The code finds all unique care episodes and multiplies the number of episodes by 0.8 to get a split point. The first 80% of whole episodes are used for training the transition matrices, and the remaining 20% of whole episodes are used for testing them. Each subset is stored in its own DataFrame (a table-like data structure).


---

# 4. Generating the Transition Matrices

# Transition Matrices

   ## The Formulas

   First-order:
            P(x_j | x_i) = x_ij / Σ(m=1 to M) x_im
   Where:
      - P(x_j | x_i) = Probability of transitioning to state j, given current state i,
      - x_ij = Number of observed transitions from i to j,
      - Σ(m=1 to M) x_im = Total transitions from state i to any state m.

   Second-order:
            P(x_k | x_i, x_j) = x_ijk / Σ(m=1 to M) x_ijm
   Where:
      - P(x_k | x_i, x_j) = Probability of transitioning to state k, given previous state i and current state j,
      - x_ijk = Number of observed transitions from i to j to k,
      - Σ(m=1 to M) x_ijm = Total transitions from state pair (i, j) to any state m.

   Both formulas do the same thing: count how often each history is followed by each next state, then divide by the total for that history. This is why a single function handles both orders.

   ## The Function

   calculate_transition_matrix:
      Parameters:
      data: DataFrame with transition data
      history_cols: column(s) holding the history: ["SurfaceCategories"] for first-order, or ["Previous_Category", "SurfaceCategories"] for second-order
      next_col: column name for the next state
      categories: list of category names

      Returns:
      - Transition probability matrix (rows = histories, columns = next state)

   It counts the number of transitions from each history to each next state. For second-order, the history is a state pair, for example "Equipment, Patient" becomes "(Equipment, Patient)". Then, for each history, it divides these counts by the total number of transitions from that history. This gives the probability of moving to each possible next state. Finally, the columns are put in the category order, and any missing values are filled with 0.

   For first-order every category is kept as a row, even if it was never followed by anything (for example "Out"). For second-order only the state pairs that were actually observed are kept as rows, because many of the 25 / 49 possible pairs never occur.

The same process is used for the transition matrices from the test data.

# Preparing of Data

`build_transitions` groups the data by ActivityID so that each care period is handled separately. It adds a "Next_Category" column for the next state, and for second-order also a "Previous_Category" column for the previous state.
Rows with no next state (e.g. from "Out" to "nothing") or, for second-order, no previous state are removed. For the main-category matrix, any transition that includes "In" or "Out" is also removed.

`chains_to_transitions` turns simulated chains into the same table format, so observed and simulated data go through exactly the same matrix calculation.

# Starting State Probabilities

A first-order chain needs one starting state. A second-order chain needs a starting **pair**, because the next state depends on the previous two states.

   get_starting_states:
      Parameters:
      df: DataFrame containing the care-period data
      activity_ids: list of ActivityID values to include
      categories: optional list of categories to filter by

      Returns:
      - A list of starting states (first-order) or starting pairs (second-order)

   For each care period it takes the first state (first-order) or the first two states (second-order). It only keeps them if they are all in the category list. For example, for the main-category matrices, starts that include "In" are filtered out. The counts are then converted to probabilities.

   Note: most care periods start with "In", so the main-category starting probabilities come only from the few periods that don't. In the current data this means every main-category simulation starts in HygieneArea.

# MAE with Proper Alignment

   calculate_mae_safe:
      Calculates the Mean Absolute Error between an observed and a predicted matrix.
      Only evaluates histories that were observed (rows of the observed matrix that sum to more than 0).
      Predicted values for missing transitions default to 0.

      Parameters:
      observed: observed transition probability matrix
      predicted: predicted transition probability matrix

      Returns:
      - mae: mean absolute error between the observed and predicted matrices

   The predicted matrix is lined up with the observed matrix (`align_to`), so the same rows and columns are compared. Missing predicted transitions count as zero, which penalises the model for missing them. A lower MAE means a better match.
   This one function is used for every MAE in the analysis (observed vs predicted, train vs test, and perturbation), for both orders.

# Convergence Analysis

   ## The Function

   convergence_analysis:
      Analyses how transition probabilities converge as training data increases.
      Shows: P(target_transition) vs Number of Training Episodes.

      Parameters:
      df_train: DataFrame containing the training data
      train_ids: list of training episode IDs
      categories: list of surface categories to include
      target_transition: the transition to track, written as (history..., next)

      Returns:
      - convergence_results: the estimated probability at different training sample sizes
      - target_transition: the transition that was analysed

   It checks how the estimated transition probabilities change as more training episodes are added. This shows whether the estimates become more stable with more data, and estimates how much training data is needed.
   The function uses the first 10, 20, 30... training episodes, calculates a transition matrix for each subset, and records the probability of the target transition. It also records how often the history occurred (`from_count`) and how often it was followed by the target next state (`target_count`).

   Transitions tracked (set in `config.py`):

   | | Main categories | All categories |
   |---|---|---|
   | First-order | Equipment → Patient | In → Equipment |
   | Second-order | (HygieneArea, Equipment) → Patient | (In, FarPatient) → Equipment |

   ## Stability

   - `wilson_ci`: 95% confidence interval for the final estimate (Wilson method),
   - `assess_stability`: labels the estimate as STABLE, MODERATE, UNSTABLE or INSUFFICIENT DATA (fewer than 5 occurrences). It uses the width of the confidence interval compared with the estimate, and how much the estimate changed in the last step,
   - `estimate_minimum_episodes`: the first number of episodes at which the estimate became STABLE.


---

# 5. Running Simulations

# The Function

simulate:
   Simulates one care period, for either order.
   Starts with a weighted random starting state (first-order) or starting pair (second-order).
   Picks each next state using the last 1 (first-order) or 2 (second-order) states.

   Parameters:
   lookup: the transition matrix, prepared by `build_lookup` for fast sampling
   start_probs: the starting probabilities
   categories: the categories to choose from if a history has not been observed
   length: the required length of the chain (fixed / fitted length)
   stop_state: "Out" for the terminating-state method
   counters: optional, counts how often the fallback and the length cap are used

   Returns:
   - A simulated sequence of surface categories.

It picks a starting state or pair using the starting probabilities and adds it to the chain. It then uses the transition matrix to pick the next state based on the current history, adds it to the chain, and repeats.
With a length, it stops when the chain reaches that length. With `stop_state="Out"`, it stops when "Out" is reached, or at `MAX_TERMINATING_LENGTH` (100) if "Out" is never reached.

If a second-order chain reaches a state pair that never appeared in the training data, the next state is picked at random from the categories. This cannot happen for first-order, because every state has a row. The results file reports how often this happened (0 times with the current data and seed).

# Simulations

The code runs three sets of simulations:
   1. Fixed Length (main categories, 5x5 / 25x5):
      - Calculates the average care-period length from the training data,
      - Runs simulations with chains of this fixed length.

   2. Terminating State (all categories, 7x7 / 49x7):
      - Runs simulations where the length is variable,
      - The simulation stops when the chain reaches "Out".

   3. Fitted Length (main categories, 5x5 / 25x5):
      - Fits a statistical distribution (Poisson or Negative Binomial) to the observed care-period lengths,
      - Samples a length for each chain from this distribution,
      - Runs simulations using these sampled lengths.

The full chains are kept for the sequence comparisons. They are also converted into transitions to calculate the predicted transition matrices. The time taken is recorded.

# Sequence Length Distribution Fitting

The fixed-length method uses the average care-period length, and the terminating-state method lets "Out" decide the length. In practice care periods vary in length, and terminating-state chains can become very long. So the fitted-length method samples lengths from a statistical distribution fitted to the real lengths.

   ## The Functions

   fit_length_distribution:
      Fits and compares distributions for care episode lengths.
      Tests Poisson and Negative Binomial distributions.

      Parameters:
      real_lengths: observed care-period lengths from the training data

      Returns:
      - distributions: fitted distribution information, including parameters, log-likelihood, and AIC
      - best_dist: the distribution with the lowest AIC
      - length_stats: summary statistics for the observed lengths
      - warnings: any warnings produced during fitting

   It calculates the minimum, maximum, mean, variance, and variance-to-mean ratio of the observed lengths. It fits a Poisson distribution, and also a Negative Binomial distribution if variance > mean. It then calculates the Akaike Information Criterion (AIC: a number for comparing how well different statistical models explain data) for each, and picks the one with the lowest AIC.

   calculate_goodness_of_fit:
      Uses a Kolmogorov-Smirnov test to compare the observed lengths with the fitted distribution.
      Returns the D statistic and p-value. A p-value greater than 0.05 suggests the fitted distribution is an acceptable fit.

   sample_fitted_lengths:
      Samples care-period lengths from the best-fitting distribution, for the fitted-length simulations.

After the simulations, the MAE (`calculate_mae_safe`) compares each predicted matrix with the observed one. The three methods and their chains are stored in the `simulation_methods` dictionary for the later analysis.


---

# 6. Comparison Metrics

This file compares the observed data with the predictions from all three simulation methods. Every metric is applied in exactly the same way for both orders.

   ## Mean Absolute Error

   Subtracts the predicted probabilities from the observed ones (after lining the matrices up) and calculates the MAE. The same method compares the training (observed) matrices with the test matrices.

   For second-order, the matrices have many more rows and each row is estimated from fewer transitions. So when comparing the two orders, the direction of the difference in MAE is meaningful, but its size is not directly comparable.

   ## Hellinger Distance

   hellinger_distance_histograms:
      Calculates the Hellinger distance between two datasets.

      Parameters:
      data1: first dataset
      data2: second dataset
      bins: number of histogram bins to use

      Returns:
      - A number from 0 (the distributions are identical) to 1 (the distributions are very different)

   The Hellinger distance turns both datasets into histograms and measures how different their shapes are. It is calculated for care-period lengths and for the number of contacts with each category.

   ## DNA Comparisons

   Next, it uses metrics commonly used in DNA analysis to compare observed and simulated sequences. Each surface category is converted into a single letter (In = I, Equipment = E, Patient = P, NearPatient = N, FarPatient = F, HygieneArea = H, Out = O). Every observed care period is then compared with every simulated chain (49 x 1000 = 49,000 comparisons per method).
   These metrics work on the full chains, so they are the same for both orders.

      ### Levenshtein Distance (absolute and normalised edit distance)

      Counts the minimum number of edits (insertions, deletions, substitutions) needed to turn one sequence into the other.
      A smaller distance means the sequences are more similar.

      ### Needleman-Wunsch Algorithm (global sequence alignment)

      needleman_wunsch:
         Parameters:
         x, y: sequences to align
         match: reward for matching characters
         mismatch: penalty for mismatching characters
         gap: penalty for inserting a gap

         Returns:
         - aligned_x, aligned_y: the aligned sequences
         - score: final global alignment score

      Compares the full observed and simulated sequences. It rewards matching categories and penalises mismatches and gaps.
      A higher score means the two full sequences are more similar.

      ### Smith-Waterman Algorithm (local sequence alignment)

      smith_waterman:
         Same parameters as Needleman-Wunsch.
         Returns aligned_x, aligned_y, and the score.

      Finds the best-matching section of the two sequences, ignoring poorly matching regions.
      A higher score means the observed and simulated sequences contain more similar sections.

   For all three, `compare_all_pairs` runs the comparisons and `summarise_scores` calculates the mean, median, mode, standard deviation, minimum, maximum, skew, and a normal-distribution fit, for both the absolute and the normalised scores.

   ## Sensitivity Analysis

   The sensitivity analysis estimates how much the results would change if the transition probabilities were slightly different.

   perturb_matrix:
      Adds small random noise to a transition matrix.

      Parameters:
      matrix: original transition matrix
      noise_level: maximum amount of noise to add

      Returns:
      - Perturbed matrix with rows normalised to sum to 1.

   It adds random noise between -noise_level and +noise_level to each value, makes sure no probability is negative, and normalises each row to sum to 1.
   This is repeated at 0.1%, 0.5%, 1%, 2% and 5% noise, 50 times per level. Each time, 100 chains are simulated from the perturbed matrix, and their predicted matrix is compared with the original predicted matrix using `calculate_mae_safe`. This is done for the main-category model (fixed length) and the all-category model (terminating state), and the two are compared to see which is more robust.

   ## Bootstrap Analysis

   The bootstrap analysis randomly samples the training data many times, to see how much the transition matrix would change if a different set of care periods had been observed.

   bootstrap_transition_matrix:
      Parameters:
      df_train: training dataset
      sample_fraction: proportion of care episodes to sample each time (80%)
      n_bootstrap: number of bootstrap samples (100)

      Returns:
      - matrices for the main categories and for all categories

   Each sample picks a random 80% of the training care periods and calculates the transition matrices. Every matrix is placed on the full grid of possible histories (5 / 7 rows for first-order, 25 / 49 rows for second-order) so they all line up. The code then measures how much each entry varies across samples, and reports the transition that varies the most.
   If entries vary widely, the estimates depend strongly on which care periods happened to be observed.

   ## Chi-Squared Test

   The chi-squared test checks whether the differences between two matrices are statistically significant or just random variation.

   chi_squared_test_matrices:
      H0 (null hypothesis): The matrices are the same
      H1 (alternative): The matrices are different
      If p-value < 0.05, we reject H0 (the matrices are significantly different).

      Returns:
      - chi2_statistic: a larger value suggests a greater difference,
      - p_value: used to decide statistical significance,
      - df: degrees of freedom,
      - the number of matrix cells tested.

   The function lines up both matrices, flattens them into lists, and converts the probabilities into approximate counts. It removes cells where the predicted count is zero, rescales the predicted counts to the same total as the observed counts, and runs the test.
   It is used for observed vs predicted (main and all categories) and for train vs test (main categories).

   ## Missing Transitions

   analyse_missing_transitions:
      Finds transitions that never occurred in the training data.

      These might be:
      1. Impossible transitions (e.g., Patient → Patient without leaving),
      2. Very rare but possible transitions,
      3. Very plausible but missing because of insufficient data.

      Returns:
      - the list of missing transitions,
      - the number of possible transitions,
      - the number of histories that were never observed at all.

   Every possible history is checked (25 cells for 5x5, 125 for 25x5, 49 for 7x7, 343 for 49x7), so state pairs that never occurred count as missing too. The code then checks whether the simulations produced any of these missing transitions. If none are found, the simulations only reproduce transitions that were seen in training.
   Some missing second-order transitions are structurally impossible (for example, nothing can follow "Out"). So the count is best reported together with the number of histories observed.


---

# 7. Plotting of Graphs

All figures are saved in `output/first_order/` or `output/second_order/`, with the method name in the file name (for example `..._5x5_fixed.png`, `..._25x5_fixed.png`).

# Probability Against Care Period Lengths

Compares the distribution of real care-period lengths (from the training data) with the simulated chain lengths, and shows the Hellinger distance (0 = identical).

# Probability Against Number of Contacts to Each Category

For each category, counts how many times it appears in each real care period and in each simulated chain. For example, for "Equipment" it counts how many equipment contacts occurred in each care period. Each subplot shows the probability distribution of the number of contacts with that category, with its Hellinger distance.

# Levenshtein, Needleman-Wunsch and Smith-Waterman Plots

One figure per simulation method for each metric. The left panel shows the absolute values and the right panel shows the normalised values. Each panel shows the mean, median, mode, a fitted normal curve, and the skew.

# Convergence Plots

plot_convergence:
   Plots the estimated transition probability against the number of training episodes. It shows the 95% confidence interval, the final estimate, and the stability label (from the same `assess_stability` function as the results file).

   Parameters:
   convergence_results: results from the convergence analysis
   target_transition: the transition being analysed
   save_path: where to save the plot


---

# 8. Main

`main.py` runs the other files for the chosen order. It writes the full results to `first_order_results.txt` or `second_order_results.txt`, and also prints them to the screen. It also saves the key numbers to `summary.json` for the comparison.

With `--order both`, it runs order 1, then order 2, each in its own process, and then runs `compare.py`.


---

# 9. Comparison

`compare.py` reads both `summary.json` files and creates:
- `comparison_summary.txt`: every key metric for first- and second-order side by side, grouped by simulation method, with a column showing which did better,
- `comparison_summary.csv`: the same table, for making tables in the paper,
- `comparison_metrics.png`: bar charts of MAE, Hellinger distances and the normalised sequence-similarity scores.

It also lists data coverage: how many of the possible histories were observed, and how many transitions never occurred, for each model.
