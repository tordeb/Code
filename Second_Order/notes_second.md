# Second-Order Markov Chain Analysis

The second-order file system follows the same process and goals as the first-order, with a different base formula. 

These notes do not thoroughly annotate the second-order code, but - rather, highlight the differences to the first-order code. 

# File Structure

1. The Data: movsdf.rbind_orientationcorrected.csv

2. Loading the Data: data_loader.py

3. Configuration and Constraints: config.py

4. Generating the Transition Matrices: transition_matrices.py

5. Running Simulations: simulations.py

6. Comparison Metrics: metrics.py

7. Plotting of Graphs: plotting.py

8. Main: main.py


# Project Goals

1. Train data
2. Test data
3. Produce transition matrices for the observed data, from predictions, and from predictions using unseen data
4. Produce 1000+ chains,
5. Compare these matrices and chains,
6. Plot these comparisons,
7. Ultimately, compare against the first-order chain.


--- 


# 4. Generating the Transition Matrices

Since the second-order matrix uses the previous state and current state to predict the next state, the matrix dimensions change:
    - For 5 categories, there are 5 * 5 = 25 previous-current state pairs for a 25x5 matrix.
    - For 7 categories, there are 7 * 7 = 49 previous-current state pairs for a 49*7 matrix.

For the transition matrices, the formula is:
            P(x_k | x_i, x_j) = x_ijk / Σ(m=1 to M) x_ijm
Where:
    - P(x_k | x_i, x_j) = Probability of transitioning to state k, given previous state i and current state j
    - x_ijk = Number of observed transitions from i to j to k
    - Σ(m=1 to M) x_ijm = Total transitions from state pair (i, j) to any state m

The function: 
    Calculates second-order transition probability matrices using:
    P(x_k | x_i, x_j) = x_ijk / Σ(m=1 to M) x_ijm
     
    Parameters:
    data: DataFrame with transition data
    prev_col: Column name for previous state (i)
    curr_col: Column name for current state (j)
    next_col: Column name for next state (k)
    categories: List of category names

    Returns: 
    - Second-order transition probability matrix

Similar to the first-order transition probability function, it counts the number of transitions, however, it first groups transitions into state pairs using the previous and current states.
For example, "Equipment, Patient" becomes "(Equipment, Patient)". 
Then, it counts how many times each state pair transitions to each next (single) state.
It counts the total number of transitions from each state pair and divides each transition count by the row total to give the probability of moving from a state pair to each possible next state.
The matrix is then reordered using the specific category list, and any missing values are filled with 0.

    # Starting State Pair Probabilities

    For the first-order Markov chain, one starting state is needed, however - for the second-order, a starting state pair is needed because the next state depends of the previous two states. 

    The function:
        Get the first two states of each care period as a starting pair

        Parameters:
        df: DataFrame containing the care-period data
        activity_ids: list of ActivityID values to include
        categories: optional list of categories to filter by

        Returns: 
        A list of starting state pairs
    
    The function loops through each care period using the ActivityID and gets the sequence of surface categories for that care period.
    It checks that the care period has at least two states and extracts the first and second. 
    It only keeps the pair if both states are in the category list (i.e. for the 25x5 it filters out pairs including "In") and returns a list of starting pairs. 


---

# 5. Running Simulations

When running the simulations, each transition is recorded as the previous state, current state, and next state (as opposed to just the current state and next state from first-order) because the second-order transition probability depends on two previous states, not just the current state. 


--- 

# 6. Comparison Metrics

The evaluation methods are the same for the second-order, however, what they measure differs.
In the first-order model, the next state depends only on the current state and in the second-order the next state depends on the previous and current state.
Therefore, the matrix comparison use 25x5 and 49x7 matrices, where each row is a pair of states and each column is the next possible state.
The DNA-style sequence comparison methods do not have major changes because they compare the full observed and simulated chains directly. 


--- 

# 7. Plotting of Graphs

As the plots compare the final observed and simulated care-period chains, the plotting code is mostly the same as first-order. 
In the convergence plot, the code now tracks a transition from a previous-current state pair to the next state, rather than from the current state to next state. 


---

# 8. Main

The main file runs the other files and exports the written outputs to a .txt file. 