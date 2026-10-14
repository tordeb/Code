# First vs Second Order Markov Chains

Code and analysis for the dissertation: "A Comparative Study of First- and Second-Order Markov Chains to Predict Transition Sequences in a Healthcare Context."

The data comes from [MockVsActualHospitalCare](https://github.com/MarcoFelipeKing/MockVsActualHospitalCare) (`MockVsActual_GithubDOI.csv`). It records the sequence of hand-to-surface contacts made by healthcare workers (HCWs) during patient care at an NHS respiratory ward, in 164 care episodes:

- 62 mock care episodes, all in single-bed rooms,
- 102 actual care episodes: 44 in single-bed rooms and 58 in multi-bed rooms.

## Overview

First- and second-order Markov chains are built from the same data and evaluated in exactly the same way:

- First-order: the next contact depends only on the current one, `P(next | current)`,
- Second-order: the next contact depends on the previous and current ones, `P(next | previous, current)`.

Each model is simulated using three methods:

- Fixed-length: sequences run for the mean observed episode length,
- Fitted-length: sequence lengths are sampled from a fitted negative binomial (or Poisson) distribution,
- Terminating-state: sequences end naturally when they reach the *Out* state.

Model performance is evaluated using:

- Validation on unseen episodes: leave-one-out cross-validation, scoring each held-out episode by log-likelihood per transition (perplexity), next-contact accuracy and sequence similarity, with 95% confidence intervals and a paired first- vs second-order comparison,
- Baselines: a zero-order model (category frequencies only) for next-contact prediction, and scrambled sequences (same length and category mix, order shuffled) for sequence similarity,
- Transition matrix accuracy: Mean Absolute Error (MAE) of the predicted matrices against the observed and held-out test matrices,
- Distributional similarity: Hellinger distance for episode lengths and category contact distributions,
- Sequence similarity: Levenshtein distance, Needleman-Wunsch global alignment, and Smith-Waterman local alignment,
- Robustness and sensitivity: bootstrap resampling, and matrix perturbation across five noise levels,
- Data sufficiency: convergence of transition probabilities as training data grows, and transitions never seen in training.

Before pooling the data, mock and actual care (and single- and multi-bed rooms) are compared directly.

## A Fair Comparison

Both models are produced by one codebase, where the Markov chain order is a single setting. They therefore use the same data, the same simulation code, the same metric functions and the same random seed. In the cross-validation, both are scored on the same held-out episodes and the same transitions. The only thing that differs is how many previous contacts the model looks at, so any difference in the results comes from the model itself.

A second-order chain works like a first-order chain whose states are pairs. Every matrix calculation, simulation and metric is written once and works for both orders.

## Structure

- `src/`: the analysis code
  - `config.py`: Markov chain order, dataset choice, surface-to-category mapping, file paths, number of simulations, random seed
  - `data_loader.py`: loads the data, adds surface categories, and splits it 80/20 into training and test episodes
  - `transition_matrices.py`: transition matrices, starting probabilities, test matrices, MAE, convergence analysis
  - `simulations.py`: simulation of care episodes and fitting of episode lengths
  - `metrics.py`: in-sample comparison metrics
  - `cross_validation.py`: leave-one-out cross-validation (validation on unseen episodes)
  - `plotting.py`: all figures
  - `main.py`: runs the analysis for one order or for both
  - `compare.py`: puts the first- and second-order results side by side
  - `dataset_comparison.py`: compares mock vs actual care and single- vs multi-bed rooms
  - `distances.py`: Hellinger distance, shared by the metrics and the dataset comparison
  - `data_exploration.py`: a quick look at the raw data (not part of the analysis)
- `data/`: the dataset (`MockVsActual_GithubDOI.csv`)
- `output/<dataset>/first_order/`, `output/<dataset>/second_order/`: full results (`.txt`), per-episode cross-validation scores (`.csv`) and figures for each model
- `output/<dataset>/comparison/`: first- vs second-order comparison tables (`.txt` and `.csv`) and figures
- `output/data_comparison/`: mock vs actual and single- vs multi-bed comparison
- `docs/notes.md`: detailed notes on how each part of the code works

## Running

```
pip install -r requirements.txt
python src/main.py --order both --data combined    # first-order, second-order, then the comparison
python src/main.py --order 1 --data mock           # first-order only, mock episodes only
python src/dataset_comparison.py                   # mock vs actual, single-bed vs multi-bed
```

`--data` can be `mock`, `actual`, `actual_sb`, `actual_mb`, `combined_sb` (mock + actual single-bed) or `combined` (all 164 episodes, the default).

Or with Docker, which avoids installing Python packages locally:

```
docker build -t markov-chains .
docker run --rm -v "$(pwd)/output:/app/output" markov-chains
```

The random seed defaults to 42 (`--seed` to change it). Running again with the same seed gives exactly the same results.

## Surface Categories

Each surface is mapped to one of five categories (Equipment, Patient, NearPatient, FarPatient, HygieneArea), plus *In* and *Out*. The mock data uses the mapping from the original categorised mock dataset. Surfaces recorded only in the actual data were given the category of the closest mock surface (for example GownOn as HygieneArea, like ApronOn, and IVDrip as Equipment, like IV), and Curtain and Notes were assigned to FarPatient. The full mapping is in `config.py`.

## Key Findings

### Mock vs actual care

Mock and actual care differ clearly, so pooling them mixes two different kinds of episode (`output/data_comparison/`):

- Actual single-bed episodes are longer than mock episodes (mean 32.6 vs 18.8 contacts, Mann-Whitney p < 0.001).
- The share of contacts by category differs (total variation distance 0.30, permutation p < 0.001): Equipment makes up 60% of actual contacts but 31% of mock contacts, and FarPatient 5% vs 21%.
- The first-order transition matrices differ (MAE 0.095, permutation p < 0.001). For example, after entering the room, mock episodes almost always go to FarPatient (98%, mostly the door), while actual episodes most often go to Equipment (64%).
- 41% of actual single-bed episodes include leaving and re-entering the room; no mock episode does.

Single- and multi-bed actual episodes are more alike: the category shares do not differ clearly (total variation distance 0.09, p = 0.12), but episode lengths and transition matrices do.

In the leave-one-out cross-validation on the combined data, mock episodes are predicted less well than actual episodes by both models.

### First- vs second-order

On unseen episodes (leave-one-out cross-validation), the second-order model predicts the next contact better than the first-order model, for both the mock data and the combined data:

| Combined data (164 episodes) | First-order | Second-order | Difference (95% CI) |
|---|---|---|---|
| Log-likelihood per transition, main categories | −1.123 | −1.087 | +0.036 [+0.008, +0.064] |
| Log-likelihood per transition, all categories | −1.357 | −1.312 | +0.046 [+0.022, +0.070] |
| Next-contact accuracy, main categories | 59.2% | 63.7% | +4.4 points [+1.4, +7.5] |
| Next-contact accuracy, all categories | 52.0% | 55.6% | +3.7 points [+1.3, +6.0] |

- Both models clearly beat a zero-order baseline that predicts contacts from category frequencies alone (combined data, main categories: accuracy 44.6% for zero-order, +14.6 points for first-order, +19.1 points for second-order). Knowing the current contact matters most; the previous contact adds a smaller further gain.
- This advantage of second-order holds with add-one and add-0.5 smoothing, but not with very light smoothing (0.1), where the difference is no longer clear.
- For simulated sequences, the picture is mixed. First-order sequences are slightly closer to held-out episodes for the fixed- and fitted-length methods. Second-order terminating-state sequences are closer by Levenshtein and Needleman-Wunsch, but not by Smith-Waterman.

### Do the simulated sequences capture the order of contacts?

Each simulated chain was compared with a scrambled version of itself (same length and category mix, contacts shuffled):

- **Terminating-state chains** score clearly better than their scrambled versions on all three sequence metrics, for both orders and both datasets. Their similarity to real episodes comes from the order of contacts.
- **Fixed- and fitted-length chains** do not beat their scrambled versions on Levenshtein or Needleman-Wunsch (on the combined data they are slightly worse), and only beat them on Smith-Waterman. Their overall similarity to real episodes comes from length and category mix rather than order; only short local stretches reflect real sequences.

### Episode lengths

The length mismatch of the terminating-state model is two-sided. On the combined data, 9.1% of first-order terminating chains are shorter than the shortest observed episode and 17.3% have 5 or fewer contacts (3.8% of observed episodes), while 4.9% are longer than the longest observed. The 100-contact cap only affects the long end. The second-order model produces far fewer very short chains (1.0% shorter than the shortest observed).
- Second-order estimates remain less certain. In the combined data, 42 of the 49 possible state pairs are observed, and its tracked transition probabilities do not reach a stable estimate, while the first-order ones do. Its train-vs-test MAE and bootstrap variability are also higher.
- Combining the data reduces second-order sparsity: missing transitions in the 25x5 matrix fall from 38/125 (mock only) to 14/125.

The full results are in `output/combined/comparison/comparison_summary.txt` and `output/mock/comparison/comparison_summary.txt`.

## Limitations

- Mock and actual care differ in episode length, category mix and transitions, and some differences (such as the door contact after entering the room) may reflect how contacts were recorded. Results on the combined data describe a mixture of the two.
- The category mapping for surfaces recorded only in the actual data is a judgement, and affects the category comparisons.
- Main-category simulations start from the first state(s) of each episode that are main categories, because episodes beginning with *In* are excluded from those starting probabilities.
- Observed episodes include *In*/*Out*, while the main-category simulated sequences do not. This affects both orders equally, but makes the absolute sequence-similarity scores of the fixed- and fitted-length methods pessimistic.
- Terminating-state chains stop at the first *Out* (or at 100 contacts), although some actual episodes leave and re-enter the room.
- MAE values are comparable in direction but not in size between the two orders, because a second-order row is estimated from far fewer transitions.

## Ways This Research Can Be Extended

1. Collect more data, to determine whether the second-order model's better next-contact prediction holds as its estimates become more stable.
2. Compare Markov chain simulated sequences with those generated by large language models (LLMs) using sequence alignment metrics.
3. Extend the research to different ward types, care procedures, or patient room configurations.
4. Incorporate surface and air contamination levels into the transition model to predict pathogen exposure.
5. Analyse hand hygiene patterns from the transition sequences to identify high-risk contact behaviours.
