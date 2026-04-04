import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd

from autoencoder.datamodule import AutoencoderDataModuleAllData

def generate_predictions(model: torch.nn.Module, dataloader: DataLoader) -> np.ndarray:
    """
    Generate predictions from a model given a dataloader.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): Dataloader providing the input windows.

    Returns:
        np.ndarray: Array of model predictions for each window.
    """
    model.eval()
    predictions = []

    device = next(model.parameters()).device
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 3:
                inputs, _, _ = batch
            else:
                inputs, _ = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

def reconstruct_time_series(predictions: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    Reconstruct the time series from overlapping windows of predictions.

    Args:
        predictions (np.ndarray): Array of predictions with shape (num_windows, num_channels, window_size).
        window_size (int): Size of each window.
        step_size (int): Step size between windows.

    Returns:
        np.ndarray: Reconstructed time series.
    """
    num_windows, num_channels, _ = predictions.shape
    reconstructed_series = np.zeros((num_windows * step_size + window_size - step_size, num_channels))
    window_sum = np.zeros(reconstructed_series.shape)
    
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        reconstructed_series[start:end] += predictions[i].T
        window_sum[start:end] += 1
    
    reconstructed_series /= window_sum  # Normalize by number of overlaps
    return reconstructed_series

def compute_reconstruction_and_errors(df: pd.DataFrame, model, window_size: int, step_size: int, batch_size: int, scaler) -> pd.DataFrame:
    """
    Compute the reconstruction and point-wise reconstruction errors for a given DataFrame and model.
    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        model: Trained autoencoder model.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for the sliding window.
        batch_size (int): Batch size for the DataLoader.
        scaler: Scaler used for normalizing the data (if applicable).
    Returns:
        df (pd.DataFrame): DataFrame containing the original & reconstructed time series, the point_errors and the window_index.
    """
    df = df.copy()
    if scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    data_module = AutoencoderDataModuleAllData(df, window_size, step_size,  batch_size)
    data_module.setup()
    dataloader = data_module.dataloader()
    
    predictions = generate_predictions(model, dataloader)
    
    reconstruction = reconstruct_time_series(predictions, window_size, step_size)
    
    point_errors = (df.values - reconstruction) ** 2 # Considering 1 channel

    if scaler is not None:
        df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
        reconstruction = scaler.inverse_transform(reconstruction)

    df['reconstructed'] = reconstruction
    df['point_error'] = point_errors

    # Add window index for reference
    df['window_index'] = (df.index // step_size).astype(int)

    return df

def compute_window_mean(points: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    Compute the mean of points within sliding windows.
    Args:
        points (np.ndarray): Array of values.
        window_size (int): Size of each window.
        step_size (int): Step size between windows.
    Returns:
        np.ndarray: Array of mean values for each window.
    """
    num_points = len(points)
    num_windows = (num_points - window_size) // step_size + 1
    window_means = np.zeros(num_windows)

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        window_means[i] = points[start:end].mean()
    
    return window_means

def extract_latent_representations(df: pd.DataFrame, model, window_size: int, step_size: int, batch_size: int, scaler) -> np.ndarray:
    """
    Extract latent representations,

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        model: Trained autoencoder model.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for the sliding window.
        batch_size (int): Batch size for the DataLoader.
        scaler: Scaler used for normalizing the data (if applicable).
    Returns:
        Array of latent representations.
    """
    df = df.copy()
    if scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    data_module = AutoencoderDataModuleAllData(df, window_size, step_size,  batch_size)
    data_module.setup()
    dataloader = data_module.dataloader()

    latent_representations = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)
            latent_space = model(inputs, return_latent=True)
            latent_representations.append(latent_space.cpu().numpy())
    
    latent_representations = np.concatenate(latent_representations)
    return latent_representations
