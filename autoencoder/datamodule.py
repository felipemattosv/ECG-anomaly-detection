import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

def create_windows(df, window_size, step_size):
    """
    Create windowed data from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        window_size (int): Size of each window.
        step_size (int): Step size between windows.

    Returns:
        np.ndarray: Array of windowed data.
    """
    # if step_size == window_size:
    #     print("Generating non-overlapping windows.")
    # else:
    #     print("Warning: Windows will overlap.")

    num_windows = (len(df) - window_size) // step_size + 1
    #print(f"Number of Windows: {num_windows}")

    # Generate windows of the data
    windows = np.array([df.iloc[i:i+window_size].values for i in range(0, num_windows * step_size, step_size)])

    # # Print the shape of the windows array
    # print(f"Windows Shape: {windows.shape}")

    return windows

class AutoencoderDataset(Dataset):
    def __init__(self, windows: np.ndarray, additional_info: np.ndarray = None):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        window_tensor = torch.tensor(window, dtype=torch.float32)

        # print(f"[Dataset __getitem__] Window tensor type: {type(window_tensor)}, shape: {window_tensor.shape}")

        return window_tensor, window_tensor

class AutoencoderDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 window_size: int,
                 step_size: int,
                 batch_size: int = 64):

        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        # Ensure all data is numerical. This is important after one-hot encoding.
        self.train_df = self.train_df.apply(pd.to_numeric, errors='coerce')
        self.val_df = self.val_df.apply(pd.to_numeric, errors='coerce')

        # Create windows for the input data to train the model
        train_windows = create_windows(self.train_df, self.window_size, self.step_size)
        val_windows = create_windows(self.val_df, self.window_size, self.step_size)

        print(f"Train windows shape: {train_windows.shape}")
        print(f"Val windows shape: {val_windows.shape}")

        # # Convert windows to PyTorch tensors
        # train_windows = torch.tensor(train_windows, dtype=torch.float32)
        # val_windows = torch.tensor(val_windows, dtype=torch.float32)
        # test_windows = torch.tensor(test_windows, dtype=torch.float32)

        # Create datasets
        self.train_dataset = AutoencoderDataset(train_windows)
        self.val_dataset = AutoencoderDataset(val_windows)

        # print("----------------------")
        # print("Data processing complete.")
        # print(f"Train dataset shape: {train_windows.shape}")
        # print(f"Val dataset shape: {val_windows.shape}")
        # print(f"Test dataset shape: {test_windows.shape}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=5,
                          pin_memory=True,
                          drop_last = False,
                          persistent_workers=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=5,
                          pin_memory=True,
                          drop_last = False,
                          persistent_workers=True,
                          collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        """
        Stacks the individual windows into a single tensor and permutes the dimensions
        to get the desired shape (batch_size, input_channels, window_size).
        """

        # # Print the type and size of the input batch for debugging
        # print("Collate Function Debug:")
        # print(f"Batch Type: {type(batch)}")
        # print(f"Batch Length: {len(batch)}")

        # # Print details for each element in the batch
        # for idx, item in enumerate(batch):
        #     print(f"Item {idx} Type: {type(item)}, Item Shape: {item[0].shape if isinstance(item, (list, tuple)) and isinstance(item[0], torch.Tensor) else 'Not a tensor'}")

        inputs, targets = zip(*batch)  # Separate inputs and targets
        inputs = torch.stack(inputs)   # Stack inputs into a tensor
        targets = torch.stack(targets) # Stack targets into a tensor

        # # Print the shapes of the stacked inputs and targets
        # print(f"Inputs Shape after stacking: {inputs.shape}")
        # print(f"Targets Shape after stacking: {targets.shape}")

        # Permute dimensions to get (batch_size, input_channels, window_size)
        inputs = inputs.permute(0, 2, 1)
        targets = targets.permute(0, 2, 1)

        # print(f"Inputs Shape after permutation: {inputs.shape}")
        # print(f"Targets Shape after permutation: {targets.shape}")

        return inputs, targets

class AutoencoderDataModuleAllData(pl.LightningDataModule):
    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int,
                 step_size: int,
                 batch_size: int = 64):

        super().__init__()
        self.df = df
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        # Ensure all data is numerical. This is important after one-hot encoding.
        self.df = self.df.apply(pd.to_numeric, errors='coerce')

        # Create windows for the input data to train the model
        windows = create_windows(self.df, self.window_size, self.step_size)

        # Create datasets
        self.dataset = AutoencoderDataset(windows)

    def dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True,
                          drop_last = False,
                          persistent_workers=False,
                          collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        """
        Stacks the individual windows into a single tensor and permutes the dimensions
        to get the desired shape (batch_size, input_channels, window_size).
        """
        inputs, targets = zip(*batch)  # Separate inputs and targets
        inputs = torch.stack(inputs)   # Stack inputs into a tensor
        targets = torch.stack(targets) # Stack targets into a tensor

        # Permute dimensions to get (batch_size, input_channels, window_size)
        inputs = inputs.permute(0, 2, 1)
        targets = targets.permute(0, 2, 1)

        return inputs, targets
