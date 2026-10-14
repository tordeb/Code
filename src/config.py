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
# Dataset
# --------------------------------------------------------------------------------------------------------------------------------------
# The data file holds 62 mock care episodes (all in single-bed rooms) and 102 actual care episodes
# (44 in single-bed rooms, SB, and 58 in multi-bed rooms, MB).
#
# Set by main.py (python main.py --data <name>) through the MARKOV_DATA environment variable.

DATASETS = {
    "mock":        {"types": ["Mock"],           "rooms": ["SB", "MB"], "description": "Mock care episodes only (62, single-bed)"},
    "actual":      {"types": ["Actual"],         "rooms": ["SB", "MB"], "description": "Actual care episodes only (102, single- and multi-bed)"},
    "actual_sb":   {"types": ["Actual"],         "rooms": ["SB"],       "description": "Actual care episodes in single-bed rooms (44)"},
    "actual_mb":   {"types": ["Actual"],         "rooms": ["MB"],       "description": "Actual care episodes in multi-bed rooms (58)"},
    "combined_sb": {"types": ["Mock", "Actual"], "rooms": ["SB"],       "description": "Mock + actual single-bed episodes (106)"},
    "combined":    {"types": ["Mock", "Actual"], "rooms": ["SB", "MB"], "description": "Mock + all actual episodes (164)"},
}

DATASET = os.environ.get("MARKOV_DATA", "combined")

if DATASET not in DATASETS:
    raise ValueError(f"MARKOV_DATA must be one of {list(DATASETS)}, got {DATASET}")


# --------------------------------------------------------------------------------------------------------------------------------------
# Define Categories
# --------------------------------------------------------------------------------------------------------------------------------------
# 5 main categories and all 7 categories:
#
# 5 main:
# Equipment: Equipment, IV, ObsTrolley, Sharps, Stethoscope, Tray (mock)
#            BloodObsEq, EqMisc, EqTray, IVDrip, IVStand, MedsTrolley, Nebuliser, Syringe, TabletComputer (actual)
# Patient: Patient
# Near Patient: Bed, Chair, Table (mock)
#               Bedding, OtherNear (actual)
# Far Patient: Door, Other (mock)
#              Curtain, Notes, OtherFar, WindowBlind (actual)
# Hygiene Areas: Alc, AlcOutside, ApronOff, ApronOn, GlovesOff, GlovesOn, PaperTowel, Sink, Soap, Waste, Wipes (mock)
#                AlcInside, GownOff, GownOn, Tap (actual)
#
# +2 terminating categories:
# In: In
# Out: Out
#
# The mock mapping is the one used in movsdf.rbind_orientationcorrected.csv. The actual-only surfaces were mapped to
# the category of the closest mock surface (e.g. GownOn like ApronOn, IVDrip like IV); Curtain and Notes were assigned
# to FarPatient.

surface_categories = {
    "Equipment": [
        "Equipment", "IV", "ObsTrolley", "Sharps", "Stethoscope", "Tray",
        "BloodObsEq", "EqMisc", "EqTray", "IVDrip", "IVStand", "MedsTrolley", "Nebuliser", "Syringe", "TabletComputer",
    ],
    "Patient": ["Patient"],
    "NearPatient": ["Bed", "Chair", "Table", "Bedding", "OtherNear"],
    "FarPatient": ["Door", "Other", "Curtain", "Notes", "OtherFar", "WindowBlind"],
    "HygieneArea": [
        "Alc", "AlcOutside", "ApronOff", "ApronOn", "GlovesOff", "GlovesOn", "PaperTowel", "Sink", "Soap", "Waste",
        "Wipes", "AlcInside", "GownOff", "GownOn", "Tap",
    ],
    "In": ["In"],
    "Out": ["Out"],
}

# Surface -> category lookup, e.g. "IVDrip" -> "Equipment"
SURFACE_TO_CATEGORY = {surface: category for category, surfaces in surface_categories.items() for surface in surfaces}

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

DATA_FILE = os.path.join(PROJECT_DIR, "data", "MockVsActual_GithubDOI.csv")
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "output")
DATASET_DIR = os.path.join(OUTPUT_ROOT, DATASET)
OUTPUT_DIR = os.path.join(DATASET_DIR, f"{ORDER_NAME}_order")
COMPARISON_DIR = os.path.join(DATASET_DIR, "comparison")


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
