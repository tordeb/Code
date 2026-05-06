# --------------------------------------------------------------------------------------------------------------------------------------
# Load Data and Train/Test Split
# --------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
from config import DATA_FILE, main_categories, all_categories


# --------------------------------------------------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------------------------------------------------

# Show all columns and make output wider for print commands
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# --------------------------------------------------------------------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------------------------------------------------------------------
# The Data: movsdf.rbind_orientationcorrected.csv

df = pd.read_csv(DATA_FILE)


# --------------------------------------------------------------------------------------------------------------------------------------
# Split Data in Train (80%) and Test (20%)
# --------------------------------------------------------------------------------------------------------------------------------------

# Get all unique care episodes
activity_ids = df["ActivityID"].unique()
n_activities = len(activity_ids)

# Calculate where to split for training (80%) and split for testing (20%)
split_point = int(n_activities * 0.8)
train_ids = activity_ids[:split_point]
test_ids = activity_ids[split_point:]

# Create separate DataFrames
df_train = df[df["ActivityID"].isin(train_ids)].copy()
df_test = df[df["ActivityID"].isin(test_ids)].copy()


# --------------------------------------------------------------------------------------------------------------------------------------
# Reportint
# --------------------------------------------------------------------------------------------------------------------------------------

# Split Data in Train (80%) and Test (20%)
print("=== Train/Test Split ===")
print(f"Total ActivityIDs: {n_activities}")
print(f"Train ActivityIDs: {len(train_ids)} ({len(train_ids)/n_activities*100:.0f}%)")
print(f"Test ActivityIDs: {len(test_ids)} ({len(test_ids)/n_activities*100:.0f}%)")
print()