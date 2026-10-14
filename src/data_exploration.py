# --------------------------------------------------------------------------------------------------------------------------------------
# Data Exploration (not part of the main analysis)
# --------------------------------------------------------------------------------------------------------------------------------------
# Usage: python src/data_exploration.py

import pandas as pd

from config import DATA_FILE

# Load the data file
df = pd.read_csv(DATA_FILE)

# Get unique combinations of Surface and Surface Category
surfaces = df[["SurfaceCategories", "Surface"]].drop_duplicates()

# Sort alphabetically by Surface name
surfaces = surfaces.sort_values(["SurfaceCategories", "Surface"])

# Display the results
print(surfaces.to_string(index=False))


# Show all unique category names
print("All unique SurfaceCategories")
print(df["SurfaceCategories"].unique())
print()
# Count how many times each category appears
print("Count of each category")
print(df["SurfaceCategories"].value_counts())
print()


# Count how often each category is the first state
first_rows = df.groupby("ActivityID").first()
print("First States in Data")
print(first_rows["SurfaceCategories"].value_counts())
print()


# Calculate average ActivityID length
ActivityID_length = df.groupby("ActivityID").size()
print(f"Mean care period length: {ActivityID_length.mean():.1f}")
