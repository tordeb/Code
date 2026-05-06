# First-Order Markov Chain Analysis

# File Structure

There are 7 necessary files to run the analysis of the first-order Markov Chain (9 if including the Docker setup used to run the code without downloading Python packages locally):

1. The Data: movsdf.rbind_orientationcorrected.csv
   - 62 observed care periods with each surface contact transition by the HCW noted,
   - Other headings include: the care type, HCW type, orientation, date, time, and surface category.

2. Loading the Data: data_loader.py
   - Loads the data,
   - Splits it into 80% train data and 20% test data.

3. Configuration and Constraints: config.py
   - The analysis compares using a fixed length chain with using terminating states, hence the file defines the surface categories into "main" and "all". 
   - Creates file paths for the output to be saved to.
   - States the number of simulated Markov chain sequences to be generated.

4. Generating the Transition Matrices: transition_matrices.py
   - Defines the function for the transition matrices,
   - Produces probabilities for the most-likely starting state,
   - Produces 'test' matrices using the test data,
   - Produces a convergence analysis to see how transition probabilities change with an increased amount of training data.

5. Running Simulations: simulations.py
   - Defines functions for and calculates the predicted transition matrices,
   - Runs the simulations,
   - Produces a sequence length distribution fitting to analyse the length of predicted sequences,
   - Compares this fitted distribution with using the fixed length main categories and the terminating state all categories methods.

6. Comparison Metrics: metrics.py
   - Compares the observed transition matrices with the predicted,
   - Tests against the 'test' matrices,
   - Produces the Hellinger distance,
   - Compares using common DNA metrics:
      - Levenshtein Distance (absolute and normalised edit distance),
      - Needleman-Wunsch Algorithm (global sequence alignment),
      - Smith-Waterman Algorithm (local sequence alignment).
   - Produces a sensitivity analysis using matrix perturbation and a bootstrap analysis,
   - Calculates if there are any missing transitions in the training data that are possible.

7. Plotting of Graphs: plotting.py
   - Plots graphs for:
      - Probability against care period length,
      - Probability against number of contacts to each category,
      - Levenshtein Distance,
      - Needleman-Wunsch Algorithm,
      - Smith-Waterman Algorithm,
      - Convergence.

8. Main: main.py
   - Runs the other files,
   - Exports the results to a .txt file.      


# Project Goals

1. Train data
2. Test data
3. Produce transition matrices for the observed data, from predictions, and from predictions using unseen data
4. Produce 1000+ chains,
5. Compare these matrices and chains,
6. Plot these comparisons,
7. Ultimately, compare against the second-order chain.


---

# 2. Loading the Data

# Splitting of Data

The code retrieves all unique care episodes and multiplies the number of episodes by 0.8 to create a clear point to split the data into 80% whole episodes for training the transition matrices and 20% whole episodes for testing the transition matrices against. It creates two separate DataFrames (table-like data structures) for retrieval of each data subset. 


---

# 3. Configuration and Constraints

# Category Definitions

   ## 5 Main Categories (5x5 Matrix)

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

      ## 4. FarPatient
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

   ## 7 Categories (7x7 Matrix)
   Main categories plus two terminating states

      ### 6. In
      - Entry point for HCW to enter the room

      ### 7. Out
      - Exit point for HCW to leave the room

   ## Purpose of transition states:
   - Real care episodes have natural start (In) and end (Out) points (with likely transitions after/before these points)
   - 7x7 matrix captures complete care episodes
   - Allows simulation of variable-length care periods


---

# 4. Generating the Transition Matrices

# Transition Matrices 

   ## The Function

   The formula is: 
            P(x_j | x_i) = x_ij / Σ(m=1 to M) x_im
   Where:
      - P(x_j | x_i) = Probability of transitioning to state j, given current state i,
      - x_ij = Number of observed transitions from i to j,
      - Σ(m=1 to M) x_im = Total transitions from state i to any state m. 

   The function:
      Calculates the first-order transition probability matrices using:
      P(x_j | x_i) = x_ij / Σ(m=1 to M) x_im
     
      Parameters:
      data: DataFrame with transition data
      from_col: Column name for current state
      to_col: Column name for next state
      categories: List of category names

      Returns: 
      - First-order transition probability matrix

   It first counts the number of transitions from each current state to each next state. 
   Then, for each current state, it divides these transition counts by the total number of transitions from that state. 
   This gives the probability of moving to each possible next state. 
   The matrix is then reordered using the specified category list, and any missing values are filled with 0.

The same process is used for calculating the transition matrices using the test data. 

# Preparing of Data

The data is grouped by ActivityID to produce individual chains.
Then, a "Next_Category" column is added for the immediately following state.
The data is cleaned for occurrences when there is no next category (e.g. from "Out" to "nothing") and for the 5x5 matrix, "In" and "Out" states are removed.

Starting state probabilities are produced by counting how often each category is the first state and converting to a probability.
The data is filtered for the 5x5 matrix again to only the main categories.

# Convergence Analysis

   ## The Function
   
   The function:
      Analyses how transition probabilities converge as training data increases.
      Shows: P(target_transition) vs Number of Training Episodes.

      Parameters:
      df_train: DataFrame containing the training data
      train_ids: list of training episode IDs
      categories: list of surface categories to include
      target_transition: optional transition to track, for example ("Equipment", "Patient")

      Returns:
      - convergence_results: a list containing the estimated probability at different training sample sizes
      - target_transition: the transition that was analysed
   
   It checks how the estimated transition probabilities change as more training data is added to assess whether the transition probability estimates become more stable when more training episodes are included. 
   This method is used to estimate how much training data is required for stable estimates. 
   The function selects increasing numbers of training episodes and creates transition pairs within each episode using the current category and the next category.
   Then, it filters the data for the chosen categories (removing "In" and "Out" for the 5x5).
   It calculates a transition probability matrix for each training subset and extracts the probability of a target transition.
   Finally, it records the sample size, transition probability, total number of transitions, and number of target transition occurrences. 


---


# 5. Running Simulations

# The Functions

The function (5x5): 
   Simulates one care period for the 5 main categories.
   Starts with weighted random state.
   Runs for fixed length.

   Parameters:
   mc: the 5 × 5 transition probability matrix
   start_probs: the starting probabilities for each category
   length: the required length of the simulated sequence

   Returns:
   - A simulated sequence of surface categories with the specified fixed length.

The function (7x7):
   Simulates one care period for all 7 categories.
   Starts with weighted random state.
   Ends when terminating state "Out" is reached.
   Length is determined by probabilities.

   Parameters: 
   mc: the 7 × 7 transition probability matrix
   start_probs: the starting probabilities for each category

   Returns: 
   - A simulated sequence of surface categories ending in "Out".

Both use random selection to choose a starting category using the starting probabilities.
They add the starting category to the simulated chain and uses the transition matrix to select the next category based on the current category.
Then, adds this category to the chain.
The 5x5 repeats this process until the chain reaches the specified length, while the 7x7 repeats this process until it reaches the category "Out". 

# Simulations

The code runs three sets of simulations:
   1. 5x5 Fixed Length:
      - Calculates the average care-period length from the training data,
      - Runs simulations with chains of this fixed length,
      - Saves the full chains to current-to-next transition pairs.
   
   2. 7x7 Terminating State:
      - Runs simulations where the length is variable,
      - The simulation stops when the current state becomes "Out",
      - Saves the full chains and current-to-next transition pairs. 

   3. 5x5 Fitted Length: 
      - Fits a statistical distribution (Poisson or Negative Binomial) to the observed care-period lengths,
      - Samples new lengths from this distribution,
      - Runs simulations using these sampled lengths,
      - Saves the full chains and current-to-next transition pairs. 

The transitions are converted to DataFrames to calculate the predicted transition matrices.
The time is also recorded. 

# Sequence Length Distribution Fitting

The sequence length distribution fitting section analyses the length of the real care episodes and uses this information to simulate the Markov chains. 
Previously, the 5x5 simulated using a fixed length based on the average care-period length, and the 7x7 used terminating states. 
However, in practice care periods have varied lengths, and the 7x7 produces care periods of 100+ transitions. 
So, the sequence length distribution fitting uses statistical distributions to sample lengths. 

   ## The Functions

   fit_length_distribution:
      Fits and compare distributions for care episode lengths.
      Tests Poisson and Negative Binomial distributions.

      Parameters: 
      real_lengths: observed care-period lengths from the training data

      Returns:
      - distributions: fitted distribution information, including parameters, log-likelihood, and AIC
      - best_dist: the distribution with the lowest AIC
      - length_stats: summary statistics for the observed lengths
      - warnings: any warnings produced during fitting

   It calculates the following summary statistics for the observed care-period lengths: minimum length, maximum length, mean length, variance, variance-to-mean ratio.
   Then it fits a Poisson distribution and a Negative Binomial distribution (if variance > mean).
   It calculates the Akaike Information Criterion (AIC - a number to compare how well different statistical models explain data) for each distribution and selects the distribution with the lowest AIC and best fit. 

   calculate_goodness_of_fit:
      Calculates Kolmogorov-Smirnov test for goodness of fit.

      Parameters:
      real_lengths: observed care-period lengths
      distribution: selected distribution name
      params: parameters of the selected distribution

      Returns:
      - D_stat: Kolmogorov-Smirnov test statistic
      - p_value: p-value from the goodness-of-fit test
   
   Uses a Kolmogorov-Smirnov test to compare the observed care-period lengths with the fitted distribution. 
   A p-value greater than 0.05 suggests the fitted distribution is an acceptable fit to the observed data. 

   sample_fitted_lengths:
      Sample lengths from the best-fitting distribution.

      Parameters:
      distributions: fitted distribution information
      best_dist: selected best-fitting distribution
      n_samples: number of simulated lengths to generate
      fallback_mean: mean length used if the selected distribution is unavailable

      Returns:
      - A list/array of simulated care-period lengths.
      - A warning message if the best distribution could not be used.
   
   Samples care-period lengths from the best-fitting distribution and uses these lengths in the 5x5 simulation instead of a fixed length. 

After fitting the distribution, the code runs new 5x5 simulations, converts each simulated chain into transition pairs, and calculates a predicted transition matrix from the fitted-length simulations.
It uses the MAE (mean absolute error) to compare this fitted distribution transition matrix with the observed 5x5 transition matrix. 

   # The Function

   calculate_mae_safe: 
      Calculate MAE with proper alignment.
      Only evaluates on transitions that exist in observed data.
      Predicted values for missing pairs default to 0.

      Parameters:
      observed: observed transition probability matrix
      predicted: predicted transition probability matrix from simulations

      Returns:
      - mae: mean absolute error between the observed and predicted matrices
   
   The predicted matrix is aligned with the observed matrix, such that the same columns and rows can be compared.
   Missing predicted transitions are treated as zero. 
   Then, it calculates MAE. 
   A lower MAE indicates a better agreement between predicted and observed transition matrices.

Finally, it compares the three simulation approaches:
1. Fixed length 5x5 simulation,
2. Fitted length 5x5 simulation,
3. Terminating state 7x7 simulation.
The code gives the MAE for each approach and identifies the method with the closest match to the observed transition probabilities.
It stores the simulated chains for all three methods for later analysis. 


---


# 6. Comparison Metrics

This file compares the observed transition matrices with the predicted ones for all three simulation methods 

   ## Mean Absolute Error

   Calculates the difference by subtracting the predicted probabilities from the observed ones.
   Then, calculates the MAE. 
   It does the same method when comparing observed with test transition matrices. 

   ## Hellinger Distance 

   It calculates the Hellinger distance using the function:
      Calculates Hellinger distance between two datasets.

      Parameters: 
      0 = the distributions are identical
      1 = the distributions are very different

      Returns:
      - data1: first dataset
      - data2: second dataset
      - bins: number of histogram bins to use

   The Hellinger distance compares two distributions by converting them to histograms and measuring how different their probability shapes are.
   A smaller Hellinger distance means the distributions are more similar. 

   ## DNA Comparisons

   Then it uses metrics commonly used in DNA mutation exploration to compare observed vs. predicted.

      ### Levenshtein Distance (absolute and normalised edit distance)

      First, each surface category is converted into a single letter.
      The Levenshtein distance then compares two strings and counts the minimum number of edits needed to change one sequence into the other.
      A smaller Levenshtein distance means the observed and simulated sequences are more similar.

      ### Needleman-Wunsch Algorithm (global sequence alignment)

      Function: 
      Aligns two sequences using Needleman-Wunsch global alignment.
    
      Parameters:
      x, y: sequences to align
      match: reward for matching character
      mismatch: penalty for mismatching characters
      gap: penalty for inserting a gap 

      Returns:
      - aligned_x: aligned version of the first sequence
      - aligned_y: aligned version of the second sequence
      - score: final global alignment score  

      The NW algorithm compares the full observed and simulated sequences using global alignment.
      It rewards matching categories and penalises mismatches or gaps. 
      A higher score means the two full sequences are more similar. 

      ### Smith-Waterman Algorithm (local sequence alignment)

      Function:
      Aligns two sequences locally using Smith-Waterman algorithm.
      Identifies best subsequence alignment, ignoring poorly matching regions.
    
      Parameters:
      x, y: sequences to align
      match: reward for matching characters
      mismatch: penalty for mismatching characters
      gap: penalty for inserting a gap
    
      Returns:
      - aligned_x, aligned_y, score, start_pos_x, start_pos_y

      The SW algorithm compares subsequences rather than entire sequences.
      A higher score means that the observed and simulated sequences contain more similar subsections. 

   ## Sensitivity Analysis

   Then, the code produces a sensitivity analysis to estimate how much the results would change if a different data set were to be used. 
   The first metric done by perturbing the values of the predicted transition matrix by adding small, random noise to the values.

   The function is:
      Adds small random noise to transition matrix to test sensitivity.
      
      Parameters:
      matrix: Original Transition Matrix
      noise_level: Maximum amount of noise to add
      
      Returns:
      Perturbed matrix with rows normalised to sum to 1.

   It adds random noise between -0.01 and +0.01 to each value, ensuring no negative probabilities, and normalises the rows to sum to 1.
   Then, it repeats the test with 0.1%, 0.5%, 1%, 2%, 5% noise levels at 50 tests per level. 

   Using the perturbed values, the code produces perturbed matrices for the 5x5 and 7x7 and use MAE to compare the predicted transition matrix to the perturbed transition matrices.
   It compares which model is more robust by comparing MAE values. 

   The second sensitivity analysis is the Bootstrap Analysis, which randomly samples training data multiple times to see how much the transition matrix would change if a different dataset was used. 
   It uses the following function:
      Create multiple transition matrices from random samples of training data.
      
      This simulates 'what if we had different observations?'.

      Parameters:
      df_train: training dataset
      sample_fraction: proportion of care episodes to sample each time
      n_bootstrap: number of bootstrap samples to generate

      Returns:
      - matrices_5x5: array of bootstrapped 5 × 5 transition matrices
      - matrices_7x7: array of bootstrapped 7 × 7 transition matrices

   The function randomly selects a subset of ActivityID values, creates transition pairs within those selected care episodes, and calculates a 5x5 and 7x7 transition matrix.
   It repeats this many times using random subsets of the training data and measures how much the matrix entries vary across samples.
   If they vary widely, the estimates may depend strongly on which care episodes were sampled. 

   ## Chi-Squared Test

   The Chi-Squared test analyses if the differences between the observed and predicted matrices are statistically significant or random variation. 

   Function:
      Tests if observed and predicted matrices are significantly different.
    
      H0 (null hypothesis): The matrices are the same
      H1 (alternative): The matrices are different
      
      If p-value < 0.05, we reject H0 (matrices are significantly different)

      Parameters: 
      observed_matrix: Observed Transition Matrix
      predicted_matrix: Predicted Transition Matrix
      matrix_name=: Label for matrix being tested

      Returns:
      - chi2_statistic: The Chi-Squared test statistic. A larger value suggests a greater difference between the observed and predicted matrices.
      - p_value: The probability value used to determine statistical significance.
      - df: Degrees of freedom for the test.
      - len(observed_clean): Number of matrix cells included in the test after removing cells with zero expected counts.
   
   The function flattens the observed and predicted matrices into 1-D arrays.
   It converts the probabilities into approximate counts using a fixed total number of transitions.
   Then, removes cells where the predicted expected count is zero.
   It normalises the predicted counts, so they have the same total as the observed counts and runs the Chi-Squared test.
   A p-value below 0.5 suggests the matrices are significantly different. 

   ## Missing Transitions

   The function finds transitions that never occurred in the training data but could be possible in reality. 
      Finds transitions that never occurred in training data.
    
      These might be:
      1. Impossible transitions (e.g., Patient → Patient without leaving)
      2. Very rare but possible transitions
      3. Very plausible but missing due to insufficient data

      Parameters:
      observed_matrix: transition probability matrix
      matrix_name: optional name of the matrix

      Returns: 
      - a list of missing transitions.

   It checks whether the simulated data includes transitions that were absent from the training data.
   If none are found, the simulations only reproduce transitions that were observed during training.
   If some are found, these may represent rare or newly generated transitions introduced by the simulation process.    


---


# 7. Plotting of Graphs

For all graphs, the code loops through all three simulation methods (5x5 fixed, 5x5 fitted, and 7x7 terminating) to create separate subplots for each method. 

# Probability Against Care Period Lengths

This graph compares distribution of real care-period lengths with the distribution of simulated care-period lengths.
The real care-period lengths are calculated from the training data by counting how many observations belong to each ActivityID.
The simulated care-period lengths are calculated by counting the number of states in each simulated chain.
Then, the Hellinger distance is added to measure how different the two distributions are (0 = identical).

# Probability Against Number of Contacts to Each Category

These graphs compare how often each surface category is visited in the real and simulated care periods.
For each category, the code counts how many times that category appears in each observed care period and each simulated chain.
For example, for the category "Equipment", it counts how many equipment contacts occurred in each care period.
Each subplot shows the probability distribution of the number of visits to a category.

# Levenshtein Distance Plot

This section plots the (Absolute and Normalised) Levenshtein distance results from the DNA-style sequence comparison.
The Levenshtein distance measures the number of edits (insertion, deletion, substitution) needed to change one sequence into another.
A smaller Levenshtein distance means the observed and simulated sequences are more similar.

The absolute graph shows raw edit distance between observed and simulated, while the normalised shows the edit distance on a standardised scale. 
The plots also show mean, median, mode, theoretical normal distribution curve, and skewness interpretation.

# Needleman-Wunsch Plot

This section plots the results from the (Absolute and Normalised) Needleman-Wunsch global alignment analysis.
NW compares the full observed sequence with the full simulated sequence.
It rewards matching categories and penalises mismatches or gaps.
A higher score means the two full sequences are more similar overall.

The absolute graph shows raw global alignment scores while the normalised graph shows scores on a standardised scale. 
The plots also show mean, median, mode, theoretical normal distribution curve, and skewness interpretation.

# Smith-Waterman Plot

This section plots the results from the (Absolute and Normalised) Smith-Waterman local alignment analysis.
Smith-Waterman compares the best matching subsections of two sequences, which is useful if only part of an observed sequence is similar to part of a simulated sequence.
A higher score means that the observed and simulated sequences contain more similar subsections.

The absolute graph shows raw local alignment scores while the normalised graph shows scores on a standardised scale. 
The plots also show mean, median, mode, and theoretical normal distribution curve.

# Convergence Plots

The function plots how a selected transition probability changes as more training episodes are included:
   Plot convergence of transition probability.

   Parameters:
   convergence_results: results from the convergence analysis
   target_transition: transition being analysed, for example ("Equipment", "Patient")
   save_path: optional path where the plot should be saved

   Returns:
   - the graph.

The function is used to check whether the estimated transition probability becomes stable as the amount of training data increases.
The plot shows estimated transition probability at different training sample sizes, approximate 95% confidence interval, final probability estimate, and stability label. 


--- 

# 8. Main

The main file runs the other files and exports the written outputs to a .txt file. 