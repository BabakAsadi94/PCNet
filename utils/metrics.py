# utils/metrics.py

import numpy as np

def calculate_r2(targets, outputs):
    ss_res = np.sum((targets - outputs) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_rmse(targets, outputs):
    return np.sqrt(np.mean((targets - outputs) ** 2))

def calculate_mae(targets, outputs):
    return np.mean(np.abs(targets - outputs))

def calculate_mape(targets, outputs):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        targets (np.ndarray): Ground truth target values.
        outputs (np.ndarray): Predicted values.

    Returns:
        float: MAPE value in percentage.
    """
    epsilon = 1e-10  # Small value to avoid division by zero
    return np.mean(np.abs((targets - outputs) / (targets + epsilon))) * 100
