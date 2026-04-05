import pandas as pd
import numpy as np

def compute_reference(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the point-wise reference (median and IQR) from a set of normal time series.
    Args:
        df : DataFrame where each row is a time series and each column is a time point
    Returns:
        medians : np.ndarray of shape (n_points,) with the point-wise median
        iqrs    : np.ndarray of shape (n_points,) with the point-wise IQR (Q3 - Q1)
    """
    values = df.values
    medians = np.median(values, axis=0)
    q1 = np.percentile(values, 25, axis=0)
    q3 = np.percentile(values, 75, axis=0)
    iqrs = q3 - q1
    return medians, iqrs


def compute_mean_errors(
    df: pd.DataFrame,
    reference_medians: np.ndarray,
    reference_iqrs: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Computes the mean IQR-normalized error of each time series against the reference.
    Args:
        df                : DataFrame where each row is a time series and each column is a time point
        reference_medians : np.ndarray of shape (n_points,) with the point-wise median
        reference_iqrs    : np.ndarray of shape (n_points,) with the point-wise IQR
        epsilon           : small value added to the IQR to avoid division by zero
    Returns:
        mean_errors : np.ndarray of shape (n_series,) with the mean error per series
    """
    values = df.values
    point_errors = np.abs(values - reference_medians) / (reference_iqrs + epsilon)
    mean_errors = point_errors.mean(axis=1)
    return mean_errors
