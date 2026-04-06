import numpy as np
import pandas as pd

def train_test_split(
    *arrays: np.ndarray | list | pd.DataFrame | pd.Series,
    test_size: float | int = 0.25,
    random_state: int | None = None,
    shuffle: bool = True
) -> list[np.ndarray | pd.DataFrame | pd.Series]:
    """
    Split arrays into random train and test subsets.
    
    Parameters:
    -----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, pandas dataframes/series
    test_size : float or int, default=0.25
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
    random_state : int or None, default=None
        Controls the shuffling applied to the data before splitting.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    
    Returns:
    --------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        Preserves input type (DataFrame returns DataFrame, array returns array)
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    
    # Check all arrays have same length
    n_samples: int = len(arrays[0])
    for array in arrays[1:]:
        if len(array) != n_samples:
            raise ValueError("All arrays must have the same length")
    
    # Store original types
    original_types: list[type] = [type(arr) for arr in arrays]
    
    # Calculate number of test samples
    n_test: int
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = test_size
    
    n_train: int = n_samples - n_test
    
    # Create indices
    indices: np.ndarray = np.arange(n_samples)
    
    # Shuffle if needed
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    
    # Split indices
    train_indices: np.ndarray = indices[:n_train]
    test_indices: np.ndarray = indices[n_train:]
    
    # Split all arrays preserving type
    result: list[np.ndarray | pd.DataFrame | pd.Series] = []
    for arr, orig_type in zip(arrays, original_types):
        if isinstance(arr, pd.DataFrame):
            # For DataFrame, use iloc to preserve structure
            result.append(arr.iloc[train_indices].reset_index(drop=True))
            result.append(arr.iloc[test_indices].reset_index(drop=True))
        elif isinstance(arr, pd.Series):
            # For Series, use iloc to preserve structure
            result.append(arr.iloc[train_indices].reset_index(drop=True))
            result.append(arr.iloc[test_indices].reset_index(drop=True))
        else:
            # For numpy arrays and lists
            arr_np: np.ndarray = np.array(arr) if not isinstance(arr, np.ndarray) else arr
            result.append(arr_np[train_indices])
            result.append(arr_np[test_indices])
    
    return result

def split_data(
    normal_df: pd.DataFrame, 
    anomaly_df: pd.DataFrame, 
    contamination: float = 0.0, 
    random_state: int = 42
) -> dict[str, pd.DataFrame]:
    """
    Split normal and anomaly dataframes into train, hyperparam, threshold and test sets.
    
    Contaminates train and hyperparam sets with the specified proportion of anomalies.
    The remaining anomalies are split 50/50 between threshold and test sets.
    
    Parameters:
        normal_df : Normal samples.
        anomaly_df : Anomalous samples
        contamination : Proportion of anomalies in train and hyperparam sets (0.0 to 1.0).
        random_state : Random seed for reproducibility.
    
    Returns:
        dict with keys: train_df, hyperparam_df, threshold_df, test_df
    """
    if not 0.0 <= contamination < 1.0:
        raise ValueError("contamination must be in [0.0, 1.0)")

    # Split normal_df (70/15/7.5/7.5)
    train_normal, temp_df = train_test_split(
        normal_df, test_size=0.30, random_state=random_state
    )
    hyperparam_normal, temp2_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_state
    )
    threshold_normal, test_normal = train_test_split(
        temp2_df, test_size=0.50, random_state=random_state
    )

    # Compute how many anomalies are needed for train and hyperparam
    # n_anom / (n_normal + n_anom) = c
    # => n_anom = c * n_normal / (1 - c)
    def n_anomalies_needed(n_normal, c):
        if c == 0.0:
            return 0
        return int(round(c * n_normal / (1 - c)))

    n_anom_train = n_anomalies_needed(len(train_normal), contamination)
    n_anom_hyper = n_anomalies_needed(len(hyperparam_normal), contamination)
    total_contamination_anom = n_anom_train + n_anom_hyper

    if total_contamination_anom > len(anomaly_df):
        raise ValueError(
            f"Not enough anomalies: need {total_contamination_anom}, "
            f"got {len(anomaly_df)}."
        )

    # Split anomaly_df
    anomaly_shuffled = anomaly_df.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    anom_train = anomaly_shuffled.iloc[:n_anom_train]
    anom_hyper = anomaly_shuffled.iloc[n_anom_train:total_contamination_anom]
    anom_remaining = anomaly_shuffled.iloc[total_contamination_anom:].reset_index(drop=True)

    # Split remaining anomalies 50/50 between threshold and test
    threshold_anom, test_anom = train_test_split(
        anom_remaining, test_size=0.50, random_state=random_state
    )

    # Concat and shuffle
    train_df = pd.concat([train_normal, anom_train], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    hyperparam_df = pd.concat([hyperparam_normal, anom_hyper], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    threshold_df = pd.concat([threshold_normal, threshold_anom], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    test_df = pd.concat([test_normal, test_anom], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return {
        "train_df": train_df,
        "hyperparam_df": hyperparam_df,
        "threshold_df": threshold_df,
        "test_df": test_df,
    }
