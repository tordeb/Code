# Code Notes: First- and Second-Order Markov Chain Analysis

These notes explain what each file in `src/` does and how it works, for both the first-order and the second-order model.

Both models are run by the same code. Choose the order with `--order` when running `main.py`; it is passed to `ORDER` in `config.py`:

- **First-order (`ORDER = 1`):** the next state depends only on the current state, `P(next | current)`.
- **Second-order (`ORDER = 2`):** the next state depends on the previous and current state, `P(next | previous, current)`.

The **history** is the state(s) the next state depends on: one state for first-order, a pair of states for second-order.

# File Structure

There are 12 files in `src/`. Run everything through Docker to save installing the Python packages locally (`dockerfile`, `requirements.txt`).

1. The Data: `data/MockVsActual_GithubDOI.csv` (from [MockVsActualHospitalCare](https://github.com/MarcoFelipeKing/MockVsActualHospitalCare))
   - 164 care periods, with each surface contact by the HCW recorded:
      - 62 mock care periods, all in single-bed rooms,
      - 102 actual care periods: 44 in single-bed rooms (SB) and 58 in multi-bed rooms (MB).
   - Columns: ActivityID, care type, HCW type, room type, surface, date, time, and type (Mock or Actual).

2. Configuration and Constants: `config.py`
   - The Markov chain order (1 or 2) and the dataset (which episodes to use), both set by `main.py`,
   - The mapping of each surface to a category, and the categories, split into "main" (5) and "all" (7),
   - The matrix labels: 5x5 and 7x7 (first-order), or 25x5 and 49x7 (second-order),
   - The transitions tracked in the convergence analysis,
   - File paths, the number of simulations, the maximum terminating-chain length, and the random seed.

3. Loading the Data: `data_loader.py`
   - Loads the data and adds the category of each surface,
   - Keeps the episodes for the chosen dataset,
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
   - A scrambled baseline for the sequence comparisons,
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

10. Dataset Comparison: `dataset_comparison.py`
   - Compares mock with actual care, and single-bed with multi-bed rooms, to check whether the data can be pooled.

11. Distances: `distances.py`
   - The Hellinger distance function, shared by `metrics.py` and `dataset_comparison.py`.

12. Cross-Validation: `cross_validation.py`
   - Leave-one-out cross-validation: the main validation of each model on care episodes it was not trained on.

`data_exploration.py` is a separate script for looking at the raw data. It is not included in the analysis.

 
--- 

# Project Goals

1. Train data,
2. Test data,
3. Produce transition matrices for the observed data, from predictions, and from predictions using unseen data,
4. Produce 1000+ chains,
5. Compare these matrices and chains,
6. Plot these comparisons,
7. Compare the first-order chain against the second-order chain, using identical methods for both,
8. Compare mock and actual care (and single- and multi-bed rooms) before pooling them for more data.


---

# 1. Running the Code

```
python src/main.py --order both --data combined    # first-order, second-order, then the comparison
python src/main.py --order 1 --data mock           # first-order only, mock episodes only
python src/dataset_comparison.py                   # mock vs actual, single-bed vs multi-bed
```

`--data` chooses which care episodes are used:

| `--data` | Episodes |
|---|---|
| `mock` | 62 mock (single-bed) |
| `actual` | 102 actual (single- and multi-bed) |
| `actual_sb` | 44 actual, single-bed |
| `actual_mb` | 58 actual, multi-bed |
| `combined_sb` | 62 mock + 44 actual single-bed = 106 |
| `combined` (default) | 62 mock + 102 actual = 164 |

`--order both` runs each order in its own Python process, one after the other, and then runs `compare.py`. Running them separately means the two models share the code but nothing else. Results go to a folder for the dataset, e.g. for `--data combined`:

- `output/combined/first_order/`: `first_order_results.txt`, `summary.json`, and the figures,
- `output/combined/second_order/`: `second_order_results.txt`, `summary.json`, and the figures,
- `output/combined/comparison/`: `comparison_summary.txt`, `comparison_summary.csv`, and `comparison_metrics.png`.

`dataset_comparison.py` writes to `output/data_comparison/`.

The random seed (`--seed`, default 42) is the same for both orders. Running the code again with the same seed gives exactly the same results.


---

# 2. Configuration and Constants

# Category Definitions

   Each surface in the data is mapped to a category (`surface_categories` in `config.py`). The mock and actual data were recorded with partly different surface names. The mock mapping is the one from the original categorised mock data (`movsdf.rbind_orientationcorrected.csv`). Each actual-only surface was given the category of the closest mock surface, and Curtain and Notes were assigned to FarPatient.

   ## 5 Main Categories

      ### 1. Equipment
      - Mock: Equipment (general), IV (Intravenous equipment), ObsTrolley (Observation trolley), Sharps (Sharps container), Stethoscope, Tray
      - Actual: BloodObsEq (blood / observation equipment), EqMisc (miscellaneous equipment), EqTray, IVDrip, IVStand, MedsTrolley (medication trolley), Nebuliser, Syringe, TabletComputer, and ObsTrolley, Sharps, Stethoscope, Tray

      ### 2. Patient
      - Direct contact with patient

      ### 3. NearPatient
      - Mock: Bed, Chair, Table
      - Actual: Bedding, OtherNear (other surfaces near the patient), Chair, Table

      ### 4. FarPatient
      - Mock: Door, Other (miscellaneous surfaces)
      - Actual: Curtain, Notes, OtherFar (other surfaces far from the patient), WindowBlind, Door

      ### 5. HygieneArea
      - Mock: Alc (Alcohol gel dispenser), AlcOutside (Alcohol gel outside room), ApronOff (Apron disposal), ApronOn (Apron donning area), GlovesOff (Glove removal), GlovesOn (Glove donning), PaperTowel (Paper towel dispenser), Sink, Soap (Soap dispenser), Waste (Waste bin), Wipes (Cleaning wipes)
      - Actual: AlcInside (Alcohol gel inside room), GownOff, GownOn, Tap, and AlcOutside, GlovesOff, GlovesOn, PaperTowel, Sink, Soap, Waste, Wipes

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

# Loading

`load_data` reads `MockVsActual_GithubDOI.csv` and adds a "SurfaceCategories" column using the mapping in `config.py`. If a surface has no category, it stops with an error rather than silently dropping contacts.
Mock and actual episodes are both numbered from 1, so each ActivityID is given a prefix to keep them apart: "M001"-"M062" for mock and "A001"-"A102" for actual.
The episodes for the chosen dataset (`--data`) are then kept.

# Splitting of Data

The data is split separately within each group (mock, actual single-bed, actual multi-bed). In each group the number of episodes is multiplied by 0.8 to get a split point. The first 80% of whole episodes are used for training the transition matrices, and the remaining 20% of whole episodes are used for testing them. This way the training and test data contain the same mix of groups. Each subset is stored in its own DataFrame (a table-like data structure).

When there is more than one group, the training episodes are shuffled (using the random seed). The convergence analysis adds training episodes in order, so without shuffling it would see all of one group first.

With `--data mock` this gives exactly the same split as the original analysis (49 training and 13 test episodes).


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

   Note: most care periods start with "In", so the main-category starting probabilities come only from the periods that don't. With the mock data alone this means every main-category simulation starts in HygieneArea. With the combined data (where fewer actual episodes start with "In") the first-order starting probabilities are HygieneArea 59%, Equipment 27% and FarPatient 14%.

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
   - `estimate_minimum_episodes`: the number of episodes from which the estimate stays STABLE at every later step, up to and including the final estimate. If the final estimate is not STABLE, the minimum is "not reached" and the data is not sufficient, even if the estimate was briefly STABLE earlier.


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

# Simulated Lengths Outside the Observed Range

A mismatch between simulated and real lengths can be two-sided: too many very short chains as well as a long tail. The length cap (`MAX_TERMINATING_LENGTH`) only shortens the long tail, and does nothing about chains that are too short. So the results file reports, for each method, the share of simulated chains shorter than the shortest observed episode, the share longer than the longest observed episode, the share with 5 or fewer contacts, and the 5th-95th percentile of lengths, next to the same figures for the observed episodes.

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

**These metrics are in-sample checks, not a validation.** The simulated chains are generated from a matrix estimated from the training episodes, and are then compared with those same training episodes. The observed-vs-predicted MAE, for example, shrinks towards 0 as more chains are simulated, however well or badly the model describes real care. These metrics show whether the simulations reproduce the training data. The validation of the model on unseen episodes is the leave-one-out cross-validation (section 11).

   ## Mean Absolute Error

   Subtracts the predicted probabilities from the observed ones (after lining the matrices up) and calculates the MAE. The same method compares:
   - the training (observed) matrix with the test matrix ("train vs test"). This does not depend on the simulation method, so it is the same for fixed and fitted length (both use the main-category matrix),
   - each method's predicted (simulated) matrix with the test matrix ("predicted vs test"), for all three methods. Only histories that occur in the test data are evaluated.

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

   ## Scrambled Baseline

   A similarity score on its own does not say whether it is good. With only 5-7 categories, two unrelated sequences of similar length are already fairly similar. So the same scores are calculated for **scrambled** chains: the first 200 simulated chains of each method, with the order of their contacts shuffled. A scrambled chain keeps the same length and the same mix of categories as the original, but has no sequential structure.
   If the model's chains score better than their scrambled versions, the similarity comes from the order of contacts. If not, it was coming only from length and composition. The scrambling uses its own random generator, so the rest of the analysis is unaffected.

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

   ## Chi-Squared Test (removed)

   Earlier versions compared matrices with a chi-squared test. It was removed for two reasons. First, it turned probabilities into counts by multiplying by an arbitrary 10,000, so its p-value depended on that arbitrary number rather than on the data. Second, two of its three uses compared the training matrix with simulations generated from that same matrix. The questions it addressed are now answered by the leave-one-out cross-validation (model vs unseen data) and by the permutation tests in `dataset_comparison.py` (differences between groups of episodes).

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

`compare.py` reads both `summary.json` files and both `cross_validation_episodes.csv` files and creates:
- `comparison_summary.txt`: first the leave-one-out cross-validation results, then every in-sample metric for first- and second-order side by side, grouped by simulation method,
- `comparison_cross_validation.csv`: the cross-validation results (each order's mean and 95% CI, and the paired difference with its 95% CI and Wilcoxon p-value), for tables in the paper,
- `comparison_cross_validation.png`: the paired differences with their 95% CIs, drawn so that a point to the right of zero means second-order did better,
- `comparison_summary.csv`: the in-sample table,
- `comparison_metrics.png`: bar charts of the in-sample MAE, Hellinger distances and normalised sequence-similarity scores.

For the cross-validation, both orders are scored on the same held-out episodes, so the comparison is **paired**: for each episode, the second-order score minus the first-order score. The table gives the mean of these differences, a 95% confidence interval, and a Wilcoxon signed-rank test (a paired test that does not assume the differences are normally distributed). One order is only called "better" when the 95% CI of the difference excludes zero.

It also lists data coverage: how many of the possible histories were observed, and how many transitions never occurred, for each model.


---

# 10. Dataset Comparison

`dataset_comparison.py` checks how similar the different groups of care episodes are, to decide whether they can be pooled into one training set. It makes three comparisons:

1. Mock vs Actual (single-bed rooms): like-for-like, as all mock episodes took place in single-bed rooms,
2. Mock vs Actual (all rooms),
3. Actual single-bed vs Actual multi-bed.

It works on whole episodes (not the train/test split) and uses first-order transitions, which have enough data to compare reliably.

   ## What is compared

   - Episode length: mean, median and standard deviation, a Mann-Whitney U test, and the Hellinger distance between the length distributions,
   - Share of all contacts in each category: the total variation distance (half the sum of the differences in share: 0 = identical, 1 = no overlap) with a permutation test,
   - Contacts per episode in each category: means, Mann-Whitney U tests and Hellinger distances,
   - First-order transition matrices: the MAE between the two groups' 7x7 and 5x5 matrices (only rows observed in both groups), with a permutation test,
   - How episodes start and end: the share starting with "In", ending with "Out", and leaving and re-entering mid-episode, with Fisher's exact test,
   - Sequence similarity: the mean normalised Levenshtein distance between episodes within each group and between the two groups. If the "between" value is close to the "within" values, episodes from the two groups are about as alike as episodes from the same group.

   ## Permutation tests

   A permutation test asks: if the group labels meant nothing, how often would a difference at least this large appear by chance? It shuffles whole episodes between the two groups 2000 times and recalculates the statistic each time. The p-value is the proportion of shuffles with a difference at least as large as the real one.
   Shuffling whole episodes, rather than single contacts, keeps the contacts from the same episode together. This matters because contacts within an episode are not independent.

   p-values are not corrected for multiple testing, so they should be read alongside the size of each difference.

   ## Outputs

   `output/data_comparison/`:
   - `data_comparison_summary.txt`: every statistic for all three comparisons,
   - `data_comparison_summary.csv`: the key numbers in one table,
   - one folder per comparison with `care_period_lengths.png`, `category_shares.png` and `transition_matrices.png` (both groups' matrices and their difference).


---

# 11. Leave-One-Out Cross-Validation

`cross_validation.py` is the main validation of each model: how well does it predict care episodes it has **not** been trained on?

   ## How it works

   For every care episode in the dataset:
   1. Hold the episode out,
   2. Build the model from all the other episodes,
   3. Score how well the model predicts the held-out episode.

   This is repeated until every episode has been held out once (62 times for the mock data, 164 for the combined data). That gives one score per episode, which is reported as a mean with a 95% confidence interval (t-distribution). It uses every episode, not only the 80/20 split, so it makes the most of a small dataset.

   ## Scores

   - **Log-likelihood per transition:** the average of log(probability the model gave the next contact that actually happened). It is always negative; closer to 0 is better.
   - **Perplexity:** exp(-log-likelihood). The effective number of equally likely next contacts the model is choosing between, so lower is better. It is the same information as the log-likelihood, but easier to explain.
   - **Next-contact accuracy:** how often the model's most likely next contact is the one that actually happened. If several contacts are equally likely, the credit is shared between them.
   - **Sequence similarity:** 100 chains are simulated from the model of each fold (for each of the three simulation methods) and compared with the held-out episode using normalised Levenshtein distance, Needleman-Wunsch and Smith-Waterman (the same scoring as `metrics.py`). The Hellinger distance between the held-out episode lengths and the simulated chain lengths (pooled over all folds) is also reported.

   ## Smoothing

   If a held-out episode contains a transition that never occurred in the training episodes, an unsmoothed model gives it probability 0, and log(0) is minus infinity. To avoid this, add-one (Laplace) smoothing is used: every transition is counted one extra time before calculating probabilities. A history never seen in training then becomes equally likely to go to any category, which matches what `simulate()` does.
   Smoothing affects second-order more than first-order, because its rows contain fewer transitions. The log-likelihood is therefore also reported with lighter smoothing (0.1 and 0.5 extra counts), to check that the conclusion does not depend on this choice.

   The sequence-similarity scores use the unsmoothed model, exactly as in the main simulations.

   ## Baselines

   - **Zero-order model:** predicts every next contact from the overall frequency of each category in the training episodes (add-one smoothed), ignoring the order of contacts completely. It is scored on the same transitions as the first- and second-order models. The difference between first-order and zero-order shows how much the model gains from knowing the current contact; the difference between second- and first-order shows what the previous contact adds on top.
   - **Scrambled chains:** for every held-out episode, the simulated chains are also scrambled (contacts shuffled, same length and mix of categories) and scored in the same way. The difference between model and scrambled shows whether the sequence similarity comes from the order of contacts.

   Both are reported as "model minus baseline" on the same held-out episode, with a mean and 95% confidence interval.

   ## Which transitions are scored

   A second-order model cannot score the first transition of an episode, because there is no previous contact. So that both orders are compared fairly, only transitions with at least two earlier contacts are scored, in **both** orders. For the main categories, the two earlier contacts and the next contact must all be main categories. Both orders are therefore scored on exactly the same transitions of the same episodes.

   ## Outputs

   - `cross_validation_episodes.csv` (in each order's output folder): every score for every held-out episode,
   - a "Leave-One-Out Cross-Validation" section at the end of `first_order_results.txt` / `second_order_results.txt`, including the mean scores for each group (e.g. mock vs actual) when the dataset contains more than one,
   - the paired first- vs second-order comparison in `compare.py` (section 9).
