# --------------------------------------------------------------------------------------------------------------------------------------
# Distance Functions (shared by metrics.py and dataset_comparison.py)
# --------------------------------------------------------------------------------------------------------------------------------------

import numpy as np


# --------------------------------------------------------------------------------------------------------------------------------------
# Hellinger Distance Function
# --------------------------------------------------------------------------------------------------------------------------------------
# Measures the similarity between two probability distributions

def hellinger_distance_histograms(data1, data2, bins=20):
    """
    Calculate Hellinger distance between two datasets
    """
    # Determine global min and max values for histogram bins to align
    min_val = min(min(data1), min(data2))
    max_val = max(max(data1), max(data2))

    # Create frequency distributions (histograms) for both data sets
    # Use the same bin width to compare probability
    hist1, bin_edges = np.histogram(data1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(data2, bins=bins, range=(min_val, max_val), density=True)

    # Normalise the histograms to sum to 1.0 to convert raw frequencies into probability distributions
    hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
    hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

    # Apply Hellinger distance formula to measure overlap between the distributions
    # 0 = identical distributions, 1 = significant divergence
    return np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2))
