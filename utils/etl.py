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
