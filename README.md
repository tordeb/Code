# First vs Second Order Markov Chains

Code and analysis for the dissertation: "A Comparative Study of First- and Second-Order Markov Chains to Predict Transition Sequences in a Healthcare Context."

The dataset matches [MockVsActualHospitalCare](https://github.com/MarcoFelipeKing/MockVsActualHospitalCare). It contains 62 observed care episodes, collected at an NHS respiratory ward, recording the sequence of hand-to-surface contacts made by healthcare workers (HCWs) during patient care.

## Overview

First- and second-order Markov chains are built from the same data and evaluated in exactly the same way:

- First-order: the next contact depends only on the current one, `P(next | current)`,
- Second-order: the next contact depends on the previous and current ones, `P(next | previous, current)`.

Each model is simulated using three methods:

- Fixed-length: sequences run for the mean observed episode length,
- Fitted-length: sequence lengths are sampled from a fitted negative binomial (or Poisson) distribution,
- Terminating-state: sequences end naturally when they reach the *Out* state.

Model performance is evaluated using:

- Transition matrix accuracy: Mean Absolute Error (MAE) against the observed and held-out test matrices, plus chi-squared tests,
- Distributional similarity: Hellinger distance for episode lengths and category contact distributions,
- Sequence similarity: Levenshtein distance, Needleman-Wunsch global alignment, and Smith-Waterman local alignment,
- Robustness and sensitivity: bootstrap resampling, and matrix perturbation across five noise levels,
- Data sufficiency: convergence of transition probabilities as training data grows, and transitions never seen in training.

## A Fair Comparison

Both models are produced by one codebase, where the Markov chain order is a single setting. They therefore use the same train/test split, the same simulation code, the same metric functions and the same random seed. The only thing that differs is how many previous states the model looks at, so any difference in the results comes from the model itself.

A second-order chain works like a first-order chain whose states are pairs. Every matrix calculation, simulation and metric is written once and works for both orders.

## Structure

- `src/`: the analysis code
  - `config.py`: Markov chain order, categories, file paths, number of simulations, random seed
  - `data_loader.py`: loads the data and splits it 80/20 into training and test episodes
  - `transition_matrices.py`: transition matrices, starting probabilities, test matrices, MAE, convergence analysis
  - `simulations.py`: simulation of care episodes and fitting of episode lengths
  - `metrics.py`: all comparison metrics
  - `plotting.py`: all figures
  - `main.py`: runs the analysis for one order or for both
  - `compare.py`: puts the first- and second-order results side by side
  - `data_exploration.py`: a quick look at the raw data (not part of the analysis)
- `data/`: the dataset (`movsdf.rbind_orientationcorrected.csv`)
- `output/first_order/`, `output/second_order/`: full results (`.txt`) and figures for each model
- `output/comparison/`: first- vs second-order comparison table (`.txt` and `.csv`) and figure
- `docs/notes.md`: detailed notes on how each part of the code works

## Running

```
pip install -r requirements.txt
python src/main.py --order both     # first-order, second-order, then the comparison
python src/main.py --order 1        # first-order only
python src/main.py --order 2        # second-order only
```

Or with Docker, which avoids installing Python packages locally:

```
docker build -t markov-chains .
docker run --rm -v "$(pwd)/output:/app/output" markov-chains
```

The random seed defaults to 42 (`--seed` to change it). Running again with the same seed gives exactly the same results.

## Key Findings

The first-order terminating-state model is recommended for datasets of this size.

- First-order performs better on every matrix-level and robustness measure. It has a lower MAE for all three simulation methods, about half the train-vs-test MAE (0.104 vs 0.188 for the main categories), less sensitivity to perturbation, and lower bootstrap variability.
- The second-order terminating-state model produces sequences somewhat closer to the observed ones. It has lower Hellinger distances and Levenshtein distance and a higher Needleman-Wunsch score. For the fixed- and fitted-length methods, the two orders perform almost the same.
- The second-order model is limited by matrix sparsity. Only 27 of the 49 possible state pairs were observed for the all-category matrix, and none of its tracked transition probabilities reached a stable estimate. As a result, it generalises less well to unseen data.

The full side-by-side results are in `output/comparison/comparison_summary.txt`.

## Limitations

- Main-category simulations start from the first state(s) of each episode that are main categories. Most episodes begin with *In*, which is excluded, so in this data every main-category simulation starts in *HygieneArea*.
- Observed episodes include *In*/*Out*, while the main-category simulated sequences do not. This affects both orders equally, but makes the absolute sequence-similarity scores of the fixed- and fitted-length methods pessimistic.
- Terminating-state chains are stopped at 100 states if they never reach *Out*. This applies to 10 of the 1,000 first-order chains and 3 of the 1,000 second-order chains.
- MAE values are comparable in direction but not in size between the two orders, because a second-order row is estimated from far fewer transitions.

## Ways This Research Can Be Extended

1. Evaluate the second-order model with a larger dataset to determine whether its performance improves enough to justify the extra complexity.
2. Compare Markov chain simulated sequences with those generated by large language models (LLMs) using sequence alignment metrics.
3. Extend the research to different ward types, care procedures, or patient room configurations.
4. Incorporate surface and air contamination levels into the transition model to predict pathogen exposure.
5. Analyse hand hygiene patterns from the transition sequences to identify high-risk contact behaviours.
