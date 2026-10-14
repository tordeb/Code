# --------------------------------------------------------------------------------------------------------------------------------------
# Configuration and Constants
# --------------------------------------------------------------------------------------------------------------------------------------
# Shared by the first- and second-order models. The only setting that differs between the two runs is ORDER.


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np


# --------------------------------------------------------------------------------------------------------------------------------------
# Markov Chain Order
# --------------------------------------------------------------------------------------------------------------------------------------
# 1 = first-order:  P(next | current)
# 2 = second-order: P(next | previous, current)
#
# Set by main.py (python main.py --order 1|2|both) through the MARKOV_ORDER environment variable.

ORDER = int(os.environ.get("MARKOV_ORDER", "1"))

if ORDER not in (1, 2):
    raise ValueError(f"MARKOV_ORDER must be 1 or 2, got {ORDER}")

ORDER_NAME = {1: "first", 2: "second"}[ORDER]


# --------------------------------------------------------------------------------------------------------------------------------------
# Define Categories
# --------------------------------------------------------------------------------------------------------------------------------------
# 5 main categories and all 7 categories:
#
# 5 main:
# Equipment: Equipment, IV, ObsTrolley, Sharps, Stethoscope, Tray
# Patient: Patient
# Near Patient: Bed, Chair, Table
# Far Patient: Door, Other
# Hygiene Areas: Alc, AlcOutside, ApronOff, ApronOn, GlovesOff, GlovesOn, PaperTowel, Sink, Soap, Waste, Wipes
#
# +2 terminating categories:
# In: In
# Out: Out

main_categories = ["Equipment", "Patient", "NearPatient", "FarPatient", "HygieneArea"]
all_categories = ["In", "Equipment", "Patient", "NearPatient", "FarPatient", "HygieneArea", "Out"]

# Matrix labels: (number of histories) x (number of next states)
# First-order: 5x5 and 7x7. Second-order: 25x5 and 49x7.
MAIN_LABEL = f"{len(main_categories) ** ORDER}x{len(main_categories)}"
ALL_LABEL = f"{len(all_categories) ** ORDER}x{len(all_categories)}"


# --------------------------------------------------------------------------------------------------------------------------------------
# Convergence Analysis Targets
# --------------------------------------------------------------------------------------------------------------------------------------
# The transition tracked in the convergence analysis: (history..., next state)

convergence_targets = {
    1: {
        "main": ("Equipment", "Patient"),
        "all": ("In", "Equipment"),
    },
    2: {
        "main": ("HygieneArea", "Equipment", "Patient"),
        "all": ("In", "FarPatient", "Equipment"),
    },
}[ORDER]


# --------------------------------------------------------------------------------------------------------------------------------------
# File Paths
# --------------------------------------------------------------------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(PROJECT_DIR, "data", "movsdf.rbind_orientationcorrected.csv")
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "output")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"{ORDER_NAME}_order")
COMPARISON_DIR = os.path.join(OUTPUT_ROOT, "comparison")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------------------------------------------------------------------
# Simulation Parameters
# --------------------------------------------------------------------------------------------------------------------------------------

n_simulations = 1000

# Terminating-state simulations stop at "Out", or at this length if "Out" is never reached.
# Applied identically to both orders.
MAX_TERMINATING_LENGTH = 100

# Random seed, so that both orders are run on the same random stream and results can be reproduced
SEED = int(os.environ.get("MARKOV_SEED", "42"))
np.random.seed(SEED)
