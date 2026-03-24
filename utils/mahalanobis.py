import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

def extract_features(series: np.ndarray) -> np.ndarray:
    """
    Extract summary statistics from a single time series.
 
    Parameters
    ----------
    series : np.ndarray
        1D array representing one time series.
 
    Returns
    -------
    np.ndarray
        Feature vector.
    """
    return np.array([
        np.mean(series),
        np.std(series),
        np.median(series),
        np.min(series),
        np.max(series),
        np.max(series) - np.min(series),          # amplitude
        np.percentile(series, 25),
        np.percentile(series, 75),
        np.percentile(series, 75) - np.percentile(series, 25),  # IQR
    ])

def extract_features_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    Extract features from all series in a DataFrame.
 
    Parameters
    ----------
    df : pd.DataFrame
        Each row is a time series.
 
    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_series, n_features).
    """
    return np.array([extract_features(row.values) for _, row in df.iterrows()])

class Mahalanobis:
    """
    Anomaly detector based on Mahalanobis distance.
 
    Learns the mean vector and covariance matrix from normal
    training data, then scores new samples by their Mahalanobis
    distance to the learned distribution.
    """
 
    def __init__(self, regularization: float = 1e-6):
        """
        Parameters
        ----------
        regularization : float
            Small value added to the diagonal of the covariance
            matrix to ensure it is invertible.
        """
        self.reg = regularization
        self.mean = None
        self.cov_inv = None
        self.threshold = None
 
    def fit(self, X_train: np.ndarray):
        """
        Learn normal behavior from training features.
 
        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix (n_samples, n_features) — mostly normal.
        """
        self.mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train, rowvar=False)
        # Regularize to avoid singular matrix
        cov += np.eye(cov.shape[0]) * self.reg
        self.cov_inv = np.linalg.inv(cov)
        return self
 
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance for each sample.
 
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
 
        Returns
        -------
        np.ndarray
            Mahalanobis distances.
        """
        distances = np.array([
            mahalanobis(x, self.mean, self.cov_inv) for x in X
        ])
        return distances