import numpy as np
from sklearn.metrics import log_loss, brier_score_loss

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Parameters:
    - y_true: Ground truth labels (0 or 1).
    - y_prob: Predicted probabilities for the positive class.
    - n_bins: Number of bins to divide probability scores into.
    
    Returns:
    - ECE value.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)  # Define bin edges
    bin_indices = np.digitize(y_prob, bins) - 1  # Assign each prob to a bin

    bin_true_proportion = np.zeros(n_bins)  # True fraction in each bin
    bin_pred_proportion = np.zeros(n_bins)  # Average confidence in each bin
    bin_counts = np.zeros(n_bins)  # Count of samples in each bin

    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):  # Only compute for non-empty bins
            bin_true_proportion[i] = np.mean(y_true[bin_mask])
            bin_pred_proportion[i] = np.mean(y_prob[bin_mask])
            bin_counts[i] = np.sum(bin_mask)

    # Compute ECE (Weighted absolute difference between accuracy and confidence)
    ece = np.sum((bin_counts / np.sum(bin_counts)) * np.abs(bin_true_proportion - bin_pred_proportion))
    return ece



# Compute Metrics
# logloss = log_loss(y_test, y_prob)
# brier = brier_score_loss(y_test, y_prob)
# ece = expected_calibration_error(y_test, y_prob)
