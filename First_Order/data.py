import pandas as pd

# Load the data file
df = pd.read_csv('movsdf.rbind_orientationcorrected.csv')

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
# Count how many times each category apprears
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

plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title("Test Graph")
plt.savefig("/app/output/test_graph.png")
plt.show()
print("Graph saved!")