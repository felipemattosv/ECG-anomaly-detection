import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


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
