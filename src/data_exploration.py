# --------------------------------------------------------------------------------------------------------------------------------------
# Data Exploration (not part of the main analysis)
# --------------------------------------------------------------------------------------------------------------------------------------
# Usage: python src/data_exploration.py

from data_loader import load_data

# Load the data file (mock and actual episodes, with surface categories added)
df = load_data()

# Number of episodes and contacts in each group
episodes = df.groupby("ActivityID")[["Source", "RoomType"]].first()
print("Episodes by Source and RoomType")
print(episodes.value_counts())
print()

# Get unique combinations of Surface and Surface Category, with how often each surface was touched
surfaces = df.groupby(["SurfaceCategories", "Surface", "Source"]).size().unstack(fill_value=0)

# Display the results
print("Surfaces in each category (number of contacts)")
print(surfaces.to_string())
print()

# Count how many times each category appears
print("Count of each category")
print(df.groupby(["SurfaceCategories", "Source"]).size().unstack(fill_value=0))
print()

# Count how often each category is the first state
first_rows = df.groupby("ActivityID").first()
print("First States in Data")
print(first_rows.groupby(["SurfaceCategories", "Source"]).size().unstack(fill_value=0))
print()

# Calculate average episode length
episode_length = df.groupby("ActivityID").size()
print("Mean care period length")
print(episode_length.groupby(episodes["Source"]).mean().round(1))
