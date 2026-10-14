# --------------------------------------------------------------------------------------------------------------------------------------
# Load Data and Train/Test Split
# --------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from config import DATA_FILE, DATASET, DATASETS, SURFACE_TO_CATEGORY, SEED


# --------------------------------------------------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------------------------------------------------

# Show all columns and make output wider for print commands
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# --------------------------------------------------------------------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------------------------------------------------------------------
# The Data: MockVsActual_GithubDOI.csv
# One row per surface contact. "type" says whether the episode was Mock or Actual care, "RoomType" whether it was in a
# single-bed (SB) or multi-bed (MB) room.

def load_data():
    """
    Load every care episode (mock and actual) and add the surface category of each contact.

    Returns:
    - DataFrame with columns including ActivityID, Source (Mock / Actual), RoomType (SB / MB), Surface and
      SurfaceCategories. ActivityID is made unique across sources: "M001"-"M062" (mock), "A001"-"A102" (actual).
    """
    data = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    data = data.rename(columns={"type": "Source"})

    # Map each surface to its category (see config.py)
    data["SurfaceCategories"] = data["Surface"].map(SURFACE_TO_CATEGORY)
    unmapped = sorted(data.loc[data["SurfaceCategories"].isna(), "Surface"].unique())
    if unmapped:
        raise ValueError(f"Surfaces without a category in config.py: {unmapped}")

    # Mock and actual episodes are both numbered from 1, so add a prefix to keep them apart
    data["EpisodeNumber"] = data["ActivityID"]
    data["ActivityID"] = data["Source"].str[0] + data["ActivityID"].astype(str).str.zfill(3)

    return data


all_data = load_data()

# Keep the episodes for the chosen dataset (config.py)
dataset_filter = DATASETS[DATASET]
df = all_data[all_data["Source"].isin(dataset_filter["types"]) & all_data["RoomType"].isin(dataset_filter["rooms"])].copy()


# --------------------------------------------------------------------------------------------------------------------------------------
# Split Data in Train (80%) and Test (20%)
# --------------------------------------------------------------------------------------------------------------------------------------
# The split is made separately within each group (mock, actual single-bed, actual multi-bed), so the training and test
# data contain the same mix of groups. Within a group, episodes are put in order of their episode number and the first
# 80% are used for training. (The data file lists the actual episodes in text order: 1-10, 100-102, 11, ..., so file
# order is not used.)

# Get all unique care episodes
activity_ids = df["ActivityID"].unique()
n_activities = len(activity_ids)

episode_groups = (df.groupby("ActivityID", sort=False)[["Source", "RoomType", "EpisodeNumber"]].first()
                  .sort_values("EpisodeNumber", kind="stable"))

train_ids = []
test_ids = []
for _, group in episode_groups.groupby(["Source", "RoomType"], sort=False):
    # Calculate where to split for training (80%) and split for testing (20%)
    split_point = int(len(group) * 0.8)
    train_ids.extend(group.index[:split_point])
    test_ids.extend(group.index[split_point:])

train_ids = np.array(train_ids)
test_ids = np.array(test_ids)

# With more than one group, shuffle the training episodes (with the fixed seed) so that the convergence analysis, which
# adds training episodes in order, does not see all of one group first
if episode_groups.groupby(["Source", "RoomType"]).ngroups > 1:
    train_ids = np.random.default_rng(SEED).permutation(train_ids)

# Create separate DataFrames
df_train = df[df["ActivityID"].isin(train_ids)].copy()
df_test = df[df["ActivityID"].isin(test_ids)].copy()

# Number of training / test episodes in each group, for the report
group_columns = ["Source", "RoomType"]
split_summary = pd.DataFrame({
    "train": episode_groups.loc[train_ids, group_columns].value_counts(),
    "test": episode_groups.loc[test_ids, group_columns].value_counts()
}).fillna(0).astype(int)
