import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import hiplot as hip
import math

def plot_series(
    df: pd.DataFrame,
    indices: list,
    time_cols_list: list,
    labels: list = None,
    title: str = "Time Series",
    figsize: tuple = (12, 4),
    alpha: float = 0.8,
):
    """
    Plots multiple time series in a single figure.

    Args:
        df             : DataFrame where each row is a time series
        indices        : list of row indices to plot, e.g. [0, 5, 12]
        time_cols_list : list of column groups, e.g. [['t1',...,'t140'], ['rt1',...,'rt140']]
        labels         : label for each column group, e.g. ['original', 'reconstructed']
                         if None, uses group index
        title          : figure title
        figsize        : figure size
        alpha          : line transparency
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = cm.tab10(np.linspace(0, 1, len(indices) * len(time_cols_list)))
    color_idx = 0

    for idx in indices:
        for g, col_group in enumerate(time_cols_list):
            label_base = labels[g] if labels else f"group {g}"
            label = f"{label_base} (idx={idx})"
            ax.plot(range(len(col_group)), df.loc[idx, col_group].values,
                    label=label, color=colors[color_idx], alpha=alpha, linewidth=1.5)
            color_idx += 1

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_series_grid(
    df: pd.DataFrame,
    indices: list,
    time_cols_list: list,
    labels: list = None,
    titles: list = None,
    main_title: str = "Time Series Grid",
    ncols: int = 2,
    subplot_figsize: tuple = (6, 3),
    alpha: float = 0.8,
    sharey: bool = False,
):
    """
    Plots one series per subplot, with multiple column groups overlaid.

    Args:
        df             : DataFrame where each row is a time series
        indices        : list of row indices, e.g. [0, 5, 10, 20]
        time_cols_list : list of column groups, e.g. [['t1',...,'t140'], ['rt1',...,'rt140']]
        labels         : label for each column group, e.g. ['original', 'reconstructed']
        titles         : title for each subplot; if None uses "idx=N"
        main_title     : overall figure title
        ncols          : number of columns in the grid
        subplot_figsize: size of each grid cell (width, height)
        alpha          : line transparency
        sharey         : share Y axis across subplots
    """
    n_plots = len(indices)
    nrows = int(np.ceil(n_plots / ncols))
    figsize = (subplot_figsize[0] * ncols, subplot_figsize[1] * nrows)
    colors = cm.tab10(np.linspace(0, 1, len(time_cols_list)))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey,
                             squeeze=False)
    fig.suptitle(main_title, fontsize=15, fontweight="bold", y=1.01)

    for plot_i, ax in enumerate(axes.flat):
        if plot_i >= n_plots:
            ax.set_visible(False)
            continue

        idx = indices[plot_i]
        subplot_title = titles[plot_i] if titles else f"idx={idx}"

        for g, (col_group, color) in enumerate(zip(time_cols_list, colors)):
            label = labels[g] if labels else f"group {g}"
            ax.plot(range(len(col_group)), df.loc[idx, col_group].values,
                    label=label, color=color, alpha=alpha, linewidth=1.5)

        ax.set_title(subplot_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_mean_series_with_min_max_envelope(
    df: pd.DataFrame,
    time_cols: list,
    title: str = "Mean Time Series with Min-Max Envelope",
    figsize: tuple = (12, 4),
    mean_color: str = "#FF0000",
    envelope_color: str = "#61B2E9",
    alpha_envelope: float = 0.3,
    alpha_line: float = 0.9,
):
    """
    Plots the mean time series with a min-max envelope showing variability.
    
    Args:
        df              : DataFrame where each row is a time series
        time_cols       : list of time column names, e.g. ['t1', 't2', ..., 't140']
        title           : figure title
        figsize         : figure size
        mean_color      : color for the mean line
        envelope_color  : color for the min-max shaded area
        alpha_envelope  : transparency for the shaded envelope
        alpha_line      : transparency for the mean line
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract time series data
    time_data = df[time_cols].values  # shape: (n_samples, n_timesteps)
    
    # Compute statistics across all samples
    mean_series = np.mean(time_data, axis=0)
    min_series = np.min(time_data, axis=0)
    max_series = np.max(time_data, axis=0)
    
    # Time axis
    time_axis = range(len(time_cols))
    
    # Plot envelope (min-max)
    ax.fill_between(time_axis, min_series, max_series, 
                    color=envelope_color, alpha=alpha_envelope, 
                    label="Min-Max Range")
    
    # Plot mean line
    ax.plot(time_axis, mean_series, color=mean_color, 
            linewidth=2, alpha=alpha_line, label="Mean")
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.show()

def plot_mean_series_with_quartile_envelope(
    df: pd.DataFrame,
    time_cols: list,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    title: str = "Mean Time Series with Quartile Envelope",
    figsize: tuple = (12, 4),
    mean_color: str = "#FF0000",
    envelope_color: str = "#61B2E9",
    alpha_envelope: float = 0.3,
    alpha_line: float = 0.9,
):
    """
    Plots the mean time series with a quartile envelope showing variability.
    
    Args:
        df              : DataFrame where each row is a time series
        time_cols       : list of time column names, e.g. ['t1', 't2', ..., 't140']
        lower_quantile  : lower quantile for envelope (default 0.25 = Q1)
        upper_quantile  : upper quantile for envelope (default 0.75 = Q3)
        title           : figure title
        figsize         : figure size
        mean_color      : color for the mean line
        envelope_color  : color for the quartile shaded area
        alpha_envelope  : transparency for the shaded envelope
        alpha_line      : transparency for the mean line
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract time series data
    time_data = df[time_cols].values  # shape: (n_samples, n_timesteps)
    
    # Compute statistics across all samples
    mean_series = np.mean(time_data, axis=0)
    lower_series = np.quantile(time_data, lower_quantile, axis=0)
    upper_series = np.quantile(time_data, upper_quantile, axis=0)
    
    # Time axis
    time_axis = range(len(time_cols))
    
    # Plot envelope (quartiles)
    label_envelope = f"Q{int(lower_quantile*100)}-Q{int(upper_quantile*100)} Range"
    ax.fill_between(time_axis, lower_series, upper_series, 
                    color=envelope_color, alpha=alpha_envelope, 
                    label=label_envelope)
    
    # Plot mean line
    ax.plot(time_axis, mean_series, color=mean_color, 
            linewidth=2, alpha=alpha_line, label="Mean")
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.show()

def plot_mean_comparison_grid(
    df: pd.DataFrame,
    time_cols: list,
    base_idx: list,
    compare_idx: list,
    titles: list = None,
    base_label: str = "Base",
    compare_labels: list = None,
    main_title: str = "Mean Time Series Comparison",
    ncols: int = 2,
    subplot_figsize: tuple = (6, 3),
    alpha: float = 0.8,
    sharey: bool = True,
):
    """
    Plots a grid comparing the mean of base indices against the mean of different classes.
    
    Args:
        df              : DataFrame where each row is a time series
        time_cols       : list of time column names, e.g. ['t1', 't2', ..., 't140']
        base_idx        : list of row indices for the base/reference class (e.g., normal class)
        compare_idx     : list of lists, each containing indices for a class to compare
                          e.g., [[1,2,3], [10,11,12], [20,21,22]]
        titles          : title for each subplot; if None uses "Comparison {i+1}"
        base_label      : label for the base class line in legend
        compare_labels  : labels for each comparison class; if None uses "Class {i+1}"
        main_title      : overall figure title
        ncols           : number of columns in the grid
        subplot_figsize : size of each grid cell (width, height)
        alpha           : line transparency
        sharey          : share Y axis across subplots
    """
    n_plots = len(compare_idx)
    nrows = int(np.ceil(n_plots / ncols))
    figsize = (subplot_figsize[0] * ncols, subplot_figsize[1] * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey,
                             squeeze=False)
    fig.suptitle(main_title, fontsize=15, fontweight="bold", y=1.01)
    
    # Compute base mean once
    base_data = df.loc[base_idx, time_cols].values
    base_mean = np.mean(base_data, axis=0)
    time_axis = range(len(time_cols))
    
    # Define colors
    base_color = "#2E86AB"
    compare_colors = cm.Set2(np.linspace(0, 1, n_plots))
    
    for plot_i, ax in enumerate(axes.flat):
        if plot_i >= n_plots:
            ax.set_visible(False)
            continue
        
        # Get comparison class indices
        class_idx = compare_idx[plot_i]
        class_data = df.loc[class_idx, time_cols].values
        class_mean = np.mean(class_data, axis=0)
        
        # Plot base mean
        ax.plot(time_axis, base_mean, color=base_color, 
                linewidth=2, alpha=alpha, label=base_label)
        
        # Plot comparison class mean
        compare_label = compare_labels[plot_i] if compare_labels else f"Class {plot_i+1}"
        ax.plot(time_axis, class_mean, color=compare_colors[plot_i], 
                linewidth=2, alpha=alpha, label=compare_label)
        
        # Styling
        subplot_title = titles[plot_i] if titles else f"Comparison {plot_i+1}"
        ax.set_title(subplot_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.show()

def plot_histogram_distributions(
        title: str,
        xlabel: str,
        points: list[np.ndarray], labels: list[str], 
        threshold: float = 5,
        figsize: tuple = (12, 6),
        xlim: list[float] = [0, 10],ylim: list[float] = [0, 500], 
        num_bins: int =100) -> None:
    """
    Plot the distributions using histograms

    Args:
        title: Title of the plot.
        x_label: Label for the x-axis.
        points: Data points.
        labels: List of labels for the datasets.
        threshold: Threshold. Defaults to 5.
        xlim: x-axis limits for the plot. Defaults to [0, 10].
        num_bins: Number of bins for the histogram. Defaults to 50000.
    """
    plt.figure(figsize=figsize)

    for i, (errors, label) in enumerate(zip(points, labels)):
        plt.hist(errors, bins=num_bins, alpha=0.5, label=label, color=f"C{i}")
    
    if threshold is not None:
        plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')

    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(loc = 'upper right')
    plt.ylim(ylim)
    plt.xlim(xlim)

    plt.show()

def plot_feature_space(
        df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
        title: str = 'Feature Space',
        figsize: tuple = (12, 8),
        alpha: float = 0.5,
        s: int = 30) -> None:
    """
    Plot a 2D or 3D scatter of features colored by label.
    Args:
        df: DataFrame with feature columns and a label column.
        feature_columns: List of 2 or 3 column names to use as axes.
        label_column: Column name containing labels.
        title: Title of the plot. Defaults to 'Feature Space'.
        figsize: Figure size. Defaults to (12, 8).
        alpha: Point transparency. Defaults to 0.5.
        s: Point size. Defaults to 30.
    """
    unique_labels = df[label_column].unique()
    fig = plt.figure(figsize=figsize)

    if len(feature_columns) == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(unique_labels):
            mask = df[label_column] == label
            ax.scatter(
                df.loc[mask, feature_columns[0]],
                df.loc[mask, feature_columns[1]],
                df.loc[mask, feature_columns[2]],
                alpha=alpha, label=label, color=f"C{i}", s=s
            )
        ax.set_zlabel(feature_columns[2])

    elif len(feature_columns) == 2:
        ax = fig.add_subplot(111)
        for i, label in enumerate(unique_labels):
            mask = df[label_column] == label
            ax.scatter(
                df.loc[mask, feature_columns[0]],
                df.loc[mask, feature_columns[1]],
                alpha=alpha, label=label, color=f"C{i}", s=s
            )

    else:
        raise ValueError("feature_columns must have 2 or 3 elements.")

    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.set_title(title)
    ax.legend(loc='upper right')
    plt.show()

#TODO improve hiplot using https://facebookresearch.github.io/hiplot/experiment_settings.html#drawing-lines-by-connecting-datapoints 
def visualize_with_hiplot(df: pd.DataFrame) -> None:
    """
    Visualizes a Pandas DataFrame using HiPlot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to visualize.
    
    Returns:
    - Displays an interactive HiPlot visualization.
    """
    # Convert the DataFrame to a HiPlot experiment
    data = df.to_dict(orient='records')
    exp = hip.Experiment.from_iterable(data)
    
    # Display the HiPlot visualization
    exp.display()

def plot_loss_curves(train_losses: list[float], val_losses: list[float]) -> None:
    """
    Plot the training and validation loss curves.

    Args:
        train_losses: List of training loss values.
        val_losses: List of validation loss values.
    """
    plt.figure(figsize=(15, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_weight_distributions(model) -> None:
    """
    Visualizes the weights distribution of the encoder layers.

    Params:
    - model: PyTorch model containing the encoder module.
    """
    encoder_weights = []
    for name, param in model.encoder.named_parameters():
        if 'weight' in name:  # Select the weight parameters of the encoder layers
            encoder_weights.append(param.data.cpu().numpy())
    
    fig, axs = plt.subplots(1, len(encoder_weights), figsize=(6 * len(encoder_weights), 5))

    if len(encoder_weights) == 1:
        axs = [axs]  # Ensure axs is iterable even if there's only one layer

    for i, weights in enumerate(encoder_weights):
        axs[i].hist(weights.flatten(), bins=30, color='skyblue', edgecolor='black')
        axs[i].set_title(f"Layer {i+1}")
        axs[i].set_xlabel("Weight")
        axs[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def plot_windows_original_and_reconstruction(
    df,
    window_indexes,
    signal_col='signal',
    reconstructed_col='reconstructed',
    title="Original vs Reconstructed Series",
    n_cols=2,
    figsize=(14, 10),
):
    """
    Plots signal vs reconstructed series for selected windows in a grid.

    Args:
        df                : DataFrame with signal, reconstructed and window_index columns
        window_indexes    : list of window indices to plot, e.g. [0, 3, 5, 12]
        signal_col        : name of the original signal column
        reconstructed_col : name of the reconstructed signal column
        title             : overall figure title
        n_cols            : number of columns in the grid
        figsize           : overall figure size (width, height)
    """
    n = len(window_indexes)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    axes = axes.flatten()

    for i, idx in enumerate(window_indexes):
        subset = df[df['window_index'] == idx]
        axes[i].plot(subset.index, subset[signal_col], label=signal_col)
        axes[i].plot(subset.index, subset[reconstructed_col], label=reconstructed_col)
        axes[i].set_title(f'Window {idx}')
        axes[i].legend()

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_scatter(
    df,
    x_col,
    y_col,
    label_col=None,
    title='Scatter Plot',
    figsize=(10, 6),
    alpha=0.7,
    s=30,
):
    """
    Plots a scatter from a DataFrame.

    Args:
        df        : DataFrame with the data
        x_col     : column name for the x axis
        y_col     : column name for the y axis
        label_col : optional column for coloring and legend grouping
        title     : figure title
        figsize   : figure size (width, height)
        alpha     : point transparency
        s         : point size
    """
    fig, ax = plt.subplots(figsize=figsize)

    if label_col:
        for label, group in df.groupby(label_col):
            ax.scatter(group[x_col], group[y_col], label=label, alpha=alpha, s=s)
        ax.legend(title=label_col)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=alpha, s=s)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
