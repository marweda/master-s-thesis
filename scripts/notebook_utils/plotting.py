from typing import Optional, Tuple, List
import os
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns


# TRUE_GPD_LOC_COLOR = "#57a7a8"
# TRUE_GPD_SCALE_COLOR = "#506eaf"
# TRUE_GPD_SHAPE_COLOR = "#b04fa4"
# PRED_GPD_LOC_COLOR = "#00FFED"
# PRED_GPD_SCALE_COLOR = "#0055FF"
# PRED_GPD_SHAPE_COLOR = "#FF00A5"
GPD_SCALE_COLOR = "#2CD3CB"
GPD_LOC_COLOR = "#2196F3"
GPD_SHAPE_COLOR = "#E69F00"
ELBO_COLOR = "#2C3E50"  # "#2C3E50"
EXCESS_COLOR = "#CC6677"  # "#F44336"  009E73
SCATTERPLOT_COLOR = "#4CAF50"  # "#4CAF50"
QUANTILE_COLOR = "#2196F3"
QUANTILE_HDI_COLOR = "#7ec1f7"
GPD_MEAN_COLOR = "#882255"
GPD_MEAN_HDI_COLOR = "#d969a1"
QQ_SCATTER_LINE_COLOR = "#7F8C8D"
QQ_SCATTER_QUANTILES_COLOR = "#2C3E50"  # 688797
VIOLIN_COLOR_PALETTE = [
    "#462fbb",
    "#117733",
    "#44AA99",
    "#88CCEE",
    "#CC6677",
    "#AA4499",
    "#882255",
]


VIOLIN_REFERENCE_COLOR = "#BDAE65"  # "#CC6677"


def plot_wasserstein_violinplot(
    wasserstein_distances: List[jnp.ndarray],
    reference_wassersteindistances: jnp.ndarray,
    x_labels: List[str],
    reference_x_label: str,
    nrows: int,
    ncols: int,
    palette: List[str],
    reference_color: str,
    fig_title: str,
    subplot_titles: List[str],
    y_label: str = "Wasserstein Distance",
    title_fontsize: int = 16,
    label_fontsize: int = 12,
    quartiles_linewidth: float = 2.0,
    quartiles_color: str = "black",
    fig_size: Tuple[int, int] = (20, 25),
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    do_save: bool = True,
):
    """
    Plots violinplots in a grid layout with reference violin at the end of each row.

    Parameters:
        wasserstein_distances: 
            Main distribution data for violinplots
        reference_wassersteindistances: 
            Reference distribution for each row
        x_labels: Labels for violins
        reference_x_label: 
            Label for reference violins
        nrows: 
            Number of rows in grid
        ncols: 
            Number of columns per row (excluding reference)
        palette: 
            Color palette for violins
        reference_color: 
            Color for reference violin
        quartiles_linewidth: float: 
            median linewidth. Default 2.0
        quartiles_color: 
            Color for the internal quartile lines (default is "black").
        fig_title: 
            Overall figure title
        subplot_titles: 
            Titles for each row (length must match nrows)
        y_label: 
            Y-axis label
        title_fontsize: 
            Font size for titles
        label_fontsize: 
            Font size for axis labels
        fig_size: 
            Figure dimensions
    """

    # Convert JAX arrays to NumPy
    main_data = [np.array(w) for w in wasserstein_distances]
    ref_data = np.array(reference_wassersteindistances)

    # Create figure and axes grid (one column, multiple rows)
    fig, axs = plt.subplots(nrows, 1, figsize=fig_size, squeeze=False)
    fig.suptitle(fig_title, fontsize=title_fontsize, y=0.99)

    # Split main data into chunks for each row
    row_chunks = [main_data[i : i + ncols] for i in range(0, len(main_data), ncols)]

    for row_idx, (row_data, ax_row) in enumerate(zip(row_chunks, axs)):
        ax = ax_row[0]

        # Prepare combined data (main row data + reference)
        combined_data = row_data + [ref_data]
        combined_labels = x_labels + [reference_x_label]

        # Build a single DataFrame for seaborn
        df_list = []
        for data, label in zip(combined_data, combined_labels):
            temp_df = pd.DataFrame({"value": data, "label": label})
            df_list.append(temp_df)
        plot_df = pd.concat(df_list)

        # Create color palette mapping
        row_colors = [palette[i % len(palette)] for i in range(len(row_data))] + [
            reference_color
        ]
        color_map = {label: color for label, color in zip(combined_labels, row_colors)}

        # Create violin plot with quartiles
        sns.violinplot(
            x="label",
            y="value",
            hue="label",  # hue as label to map correct palette
            data=plot_df,
            palette=color_map,
            inner="quartile",
            cut=0,
            ax=ax,
            inner_kws={"linewidth": quartiles_linewidth, 
                       "color": quartiles_color},
            legend=False,
        )

        # Remove any default subplot title
        ax.set_title("")

        # Place the subplot title vertically on the right side
        ax.text(
            1.02,  # a bit to the right of the axes
            0.5,  # vertically centered
            subplot_titles[row_idx],
            rotation=270,
            transform=ax.transAxes,
            fontsize=title_fontsize - 2,
            va="center",
            ha="left",
        )

        # Axis labels
        ax.set_xlabel("")
        ax.set_ylabel(y_label, fontsize=label_fontsize)

        # Tick parameters
        ax.tick_params(axis="x", rotation=45, labelsize=label_fontsize - 2)
        ax.tick_params(axis="y", labelsize=label_fontsize - 2)

        # Adjust x-axis label placement
        if row_idx == 0:
            # Move x-axis tick labels to the top for the first row
            ax.xaxis.set_ticks_position("top")
            ax.tick_params(axis="x", labeltop=True, labelbottom=False)
        else:
            # Remove x-axis labels for subsequent rows
            ax.set_xticklabels([])
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
                labeltop=False,
            )

        # Add grid lines
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Saving logic
    if (file_name is None) != (save_dir is None):
        raise ValueError(
            "Both file_name and save_dir must be provided to save the plot."
        )
    elif file_name and save_dir:
        if do_save:
            os.makedirs(save_dir, exist_ok=True)
            base, ext = os.path.splitext(file_name)
            file_name_svg = base + ".svg"
            plt.savefig(
                os.path.join(save_dir, file_name_svg), bbox_inches="tight", format="svg"
            )
            print(f"Plot saved to {os.path.join(save_dir, file_name_svg)}")
        else:
            warnings.warn(
                "The WD Violinplot plot has not been saved. Flag do_save=True for saving.",
                category=UserWarning,
            )

    plt.show()


def plot_elbo(
    num_iterations,
    title: str,
    elbo_values,
    elbo_color,
    initial_percentage,
    final_percentage,
    save_dir: str = None,
    file_name=None,
    apply_ma: tuple = (False, False, False),
    window: int = None,
    tick_label_plain: Optional[List[bool]] = [False, False, False],
    do_save: bool = True,
):
    """
    Plots the ELBO over iterations along with two zoomed-in plots for the first initial_percentage
    and last final_percentage of iterations, with optional moving averages.

    Parameters:
        num_iterations: int
            Total number of optimization iterations.
        elbo_values: array-like
            1D array containing the ELBO values for each iteration.
        elbo_color: str
            Color used for plotting the ELBO curve.
        initial_percentage: float
            Proportion (between 0 and 1) of the total iterations to zoom in on at the beginning.
        final_percentage: float
            Proportion (between 0 and 1) of the total iterations to zoom in on at the end.
        save_dir: Optional[str], optional
            Directory path where the plot will be saved. If None, the plot is not saved.
        file_name: Optional[str], optional
            Name of the file to save the plot (including extension). If provided, saves as SVG.
        apply_ma: tuple of three booleans, optional
            Flags to apply moving average to the main plot, first zoom, and last zoom respectively.
        window: int, optional
            Window size for the moving average. Required if any element in apply_ma is True.
        tick_label_plain: Optional[List[bool]];
            If first element true, then tick label format of first plot will be plain, otherwise automatic.
            Same for the other elements. Default: all false
        do_save: bool
            Flag to prevent saving with do_save=False. Default is True.
    """
    sns.set_theme(style="whitegrid")

    # Validate parameters
    if len(apply_ma) != 3:
        raise ValueError("apply_ma must be a tuple of three booleans.")
    if any(apply_ma) and window is None:
        raise ValueError("If any apply_ma is True, window must be provided.")
    if window is not None and window < 1:
        raise ValueError("window must be a positive integer.")

    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[2, 1])

    # Main plot
    ax_main = fig.add_subplot(gs[0, :])
    sns.lineplot(x=range(num_iterations), y=elbo_values, ax=ax_main, color=elbo_color)
    title_main = title
    if apply_ma[0]:
        ma_main = pd.Series(elbo_values).rolling(window=window).mean()
        sns.lineplot(x=range(num_iterations), y=ma_main, ax=ax_main, color=elbo_color)
        title_main += f" with MA-Window of {window}"
    ax_main.set_title(title_main, fontsize=14)
    ax_main.set_xlabel("Iteration", fontsize=13)
    ax_main.set_ylabel("ELBO", fontsize=13)
    if tick_label_plain[0]:
        ax_main.ticklabel_format(style="plain", axis="y", useOffset=False)

    # Calculate ranges for zoomed plots
    num_first = max(1, int(num_iterations * initial_percentage))
    num_last = max(1, int(num_iterations * final_percentage))
    start_last = num_iterations - num_last

    # First zoomed plot
    ax_first = fig.add_subplot(gs[1, 0])
    data_first = elbo_values[:num_first]
    sns.lineplot(x=range(num_first), y=data_first, ax=ax_first, color=elbo_color)
    title_first = f"First {initial_percentage*100:.1f}% of Iterations"
    if apply_ma[1]:
        ma_first = pd.Series(data_first).rolling(window=window).mean()
        sns.lineplot(x=range(num_first), y=ma_first, ax=ax_first, color=elbo_color)
        title_first += f" with MA-Window of {window}"
    ax_first.set_title(title_first, fontsize=13)
    ax_first.set_xlabel("Iteration", fontsize=12)
    ax_first.set_ylabel("ELBO", fontsize=12)
    if tick_label_plain[1]:
        ax_first.ticklabel_format(style="plain", axis="y", useOffset=False)

    # Second zoomed plot
    ax_last = fig.add_subplot(gs[1, 1])
    data_last = elbo_values[start_last:]
    x_last = range(start_last, num_iterations)
    sns.lineplot(x=x_last, y=data_last, ax=ax_last, color=elbo_color)
    title_last = f"Last {final_percentage*100:.1f}% of Iterations"
    if apply_ma[2]:
        ma_last = pd.Series(data_last).rolling(window=window).mean()
        sns.lineplot(x=x_last, y=ma_last, ax=ax_last, color=elbo_color)
        title_last += f" with MA-Window of {window}"
    ax_last.set_title(title_last, fontsize=13)
    ax_last.set_xlabel("Iteration", fontsize=12)
    ax_last.set_ylabel("ELBO", fontsize=12)

    if tick_label_plain[2]:
        ax_last.ticklabel_format(style="plain", axis="y", useOffset=False)

    plt.tight_layout()

    # Save and show
    if (file_name is None) != (save_dir is None):
        raise ValueError(
            "Both file_name and save_dir must be provided to save the plot."
        )
    elif file_name and save_dir:
        if do_save:
            os.makedirs(save_dir, exist_ok=True)
            base, ext = os.path.splitext(file_name)
            file_name_svg = base + ".svg"
            plt.savefig(
                os.path.join(save_dir, file_name_svg), bbox_inches="tight", format="svg"
            )
            print(f"Plot saved to {os.path.join(save_dir, file_name_svg)}")
        else:
            warnings.warn(
                "The elbo plot has not been saved. Flag do_save=True for saving.",
                category=UserWarning,
            )

    plt.show()


def plot_regression_results(
    scatter_x: jnp.ndarray,
    scatter_y: jnp.ndarray,
    line_xs: List[jnp.ndarray],  # Changed to list of arrays
    regression_lines: List[jnp.ndarray],
    hdi_lower_bounds: List[jnp.ndarray],
    hdi_upper_bounds: List[jnp.ndarray],
    hdi_alphas: List[float],
    regression_line_colors: List[str],
    hdi_colors: List[str],
    hdi_labels: List[str],
    scatter_color: str,
    scatter_label: str,
    regression_line_labels: List[str],
    x_label: str,
    y_label: str,
    title: str,
    fig_size: tuple = (12, 7),
    file_name: Optional[str] = None,
    excess_mask: Optional[jnp.ndarray] = None,
    excess_color: Optional[str] = None,
    pred_param_x: Optional[jnp.ndarray] = None,
    pred_param_lines: Optional[List[jnp.ndarray]] = None,
    pred_param_line_colors: Optional[List[str]] = None,
    pred_param_x_label: Optional[str] = None,
    pred_param_y_label: Optional[str] = None,
    pred_param_line_labels: Optional[List[str]] = None,
    pred_param_titles: Optional[List[str]] = None,
    pred_fig_size: tuple = (12, 4),
    save_dir: Optional[str] = None,
    y_origin: Optional[float] = None,  # New y-axis truncation parameter
    do_save: bool = True,
):
    """Plots regression results with optional subplots for predictive parameters."""
    # Validate matching lengths for regression components
    assert (
        len(line_xs)
        == len(regression_lines)
        == len(hdi_lower_bounds)
        == len(hdi_upper_bounds)
    ), "Regression components must have equal lengths"

    sns.set_style("whitegrid")
    pred_params_provided = any(
        [
            pred_param_x is not None,
            pred_param_lines is not None,
            pred_param_line_colors is not None,
            pred_param_x_label is not None,
            pred_param_y_label is not None,
            pred_param_line_labels is not None,
            pred_param_titles is not None,
        ]
    )

    # Create figure layout
    if pred_params_provided:
        num_pred = len(pred_param_lines) if pred_param_lines else 0
        main_height, sub_height = fig_size[1], pred_fig_size[1]
        total_height = main_height + num_pred * sub_height
        total_width = max(fig_size[0], pred_fig_size[0])
        fig = plt.figure(figsize=(total_width, total_height))
        gs = gridspec.GridSpec(
            num_pred + 1, 1, height_ratios=[fig_size[1]] + [pred_fig_size[1]] * num_pred
        )
        ax_main = fig.add_subplot(gs[0])
        pred_axes = [fig.add_subplot(gs[i + 1]) for i in range(num_pred)]
    else:
        fig, ax_main = plt.subplots(figsize=fig_size)

    # Main plot: Scatter points
    sns.scatterplot(
        x=scatter_x,
        y=scatter_y,
        color=scatter_color,
        label=scatter_label,
        alpha=0.7,
        ax=ax_main,
    )

    # Highlight excess points
    if excess_mask is not None:
        excess_color = excess_color or "#F44336"
        sns.scatterplot(
            x=scatter_x[excess_mask],
            y=scatter_y[excess_mask],
            color=excess_color,
            label="Excess",
            alpha=0.7,
            ax=ax_main,
        )

    # Plot each regression line and HDI with individual x-values
    for (
        line_x,
        line,
        lower,
        upper,
        alpha,
        line_color,
        hdi_color,
        line_label,
        hdi_label,
    ) in zip(
        line_xs,
        regression_lines,
        hdi_lower_bounds,
        hdi_upper_bounds,
        hdi_alphas,
        regression_line_colors,
        hdi_colors,
        regression_line_labels,
        hdi_labels,
    ):
        sns.lineplot(
            x=line_x,
            y=line,
            color=line_color,
            label=line_label,
            linewidth=2.2,
            ax=ax_main,
        )
        ax_main.fill_between(
            line_x, lower, upper, color=hdi_color, alpha=alpha, label=hdi_label
        )
        sns.lineplot(x=line_x, y=lower, color=hdi_color, alpha=alpha, ax=ax_main)
        sns.lineplot(x=line_x, y=upper, color=hdi_color, alpha=alpha, ax=ax_main)

    # Set y-axis origin if specified
    if y_origin is not None:
        ax_main.set_ylim(bottom=y_origin)

    ax_main.legend(fontsize=10, loc="upper left")
    ax_main.set_xlabel(x_label, fontsize=13)
    ax_main.set_ylabel(y_label, fontsize=13)
    ax_main.set_title(title, fontsize=14, fontweight="bold")

    # Plot predictive parameter subplots
    if pred_params_provided and pred_param_lines:
        for ax, line, color, label, sub_title in zip(
            pred_axes,
            pred_param_lines,
            pred_param_line_colors or [None] * len(pred_param_lines),
            pred_param_line_labels or [None] * len(pred_param_lines),
            pred_param_titles or [None] * len(pred_param_lines),
        ):
            sns.lineplot(x=pred_param_x, y=line, color=color, label=label, ax=ax)
            ax.set_xlabel(pred_param_x_label)
            ax.set_ylabel(pred_param_y_label)
            ax.set_title(sub_title)
            ax.legend()

    plt.tight_layout()

    # Save handling
    if file_name and save_dir:
        if do_save:
            full_path = os.path.join(
                save_dir,
                file_name if file_name.endswith(".svg") else f"{file_name}.svg",
            )
            plt.savefig(full_path, format="svg", bbox_inches="tight")
            print(f"Plot saved to {full_path}")
        else:
            warnings.warn(
                "The regression_results plot have not been saved. Flag do_save=True for saving.",
                category=UserWarning,
            )
    elif file_name or save_dir:
        raise ValueError(
            "Both file_name and save_dir must be provided to save the plot."
        )

    plt.show()


def plot_gpd_qq_plot(
    sigma_hat,
    xi_hat,
    Y_excesses,
    qq_line_color,
    quantiles_color,
    file_name,
    save_dir,
    do_save,
):
    # Avoid division by zero and handle cases where 1 + xi_hat * Y_excesses / sigma_hat <= 0
    epsilon = 1e-8  # Small constant for numerical stability
    tilde_Y = (1 / xi_hat) * jnp.log(
        jnp.clip(1 + xi_hat * Y_excesses / sigma_hat, a_min=epsilon, a_max=None)
    )

    # Sort tilde_Y (ensure conversion to a NumPy array if needed)
    sorted_tilde_Y = np.sort(np.array(tilde_Y))

    # Compute theoretical exponential quantiles
    k = len(sorted_tilde_Y)
    theoretical_quantiles = -np.log(1 - (np.arange(1, k + 1) / (k + 1)))

    # Set Seaborn theme
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Create the scatter plot using Seaborn
    sns.scatterplot(
        x=theoretical_quantiles,
        y=sorted_tilde_Y,
        color=quantiles_color,
        label="Empirical Quantiles",
    )

    # Plot the 45° reference line using Seaborn's lineplot
    sns.lineplot(
        x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
        y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
        color=qq_line_color,
        linestyle="--",
        label="45° Line",
    )

    # Set labels and title
    plt.xlabel("Theoretical Quantiles (Exponential)")
    plt.ylabel("Empirical Quantiles")
    plt.title("Residual Quantile Plot (Exponential Scale)")
    plt.legend()

    # Save handling
    if file_name and save_dir:
        if do_save:
            full_path = os.path.join(
                save_dir,
                file_name if file_name.endswith(".svg") else f"{file_name}.svg",
            )
            plt.savefig(full_path, format="svg", bbox_inches="tight")
            print(f"Plot saved to {full_path}")
        else:
            warnings.warn(
                "The gpd_qq plot has not been saved. Flag do_save=True for saving.",
                category=UserWarning,
            )
    elif file_name or save_dir:
        raise ValueError(
            "Both file_name and save_dir must be provided to save the plot."
        )
    plt.show()


def plot_synthetic_data(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    scatterplot_color: str,
    line_palette: List[str],
    lines: list,
    scatter_xlabel: str,
    scatter_ylabel: str,
    scatter_title: str,
    line_xlabel: str,
    line_ylabel: str,
    line_title: str,
    line_labels: list,
    file_name: str = None,
    save_dir: str = None,
    do_save: bool = True,
):
    """
    Plots synthetic data as a scatter plot and compares multiple lines (e.g., true vs predicted).

    Parameters:
        X: jnp.ndarray
            1D array of input values.
        Y: jnp.ndarray
            1D array of target/output values.
        scatterplot_color: str
            Color for the scatter plot points.
        line_palette: List[str]
            list of colors for each line in the line plot.
        lines: list
            list of 1D arrays, each representing a line to plot (e.g., true function, predictions).
        scatter_xlabel: str
            X-axis label for the scatter plot.
        scatter_ylabel: str
            Y-axis label for the scatter plot.
        scatter_title: str
            Title for the scatter plot.
        line_xlabel: str
            X-axis label for the line plot.
        line_ylabel: str
            Y-axis label for the line plot.
        line_title: str
            Title for the line plot.
        line_labels: list
            Legend labels for each line in the line plot.
        file_name: str
            Filename to save the plot (saves as SVG).
        save_dir: Optional[str], optional
            Directory path to save the plot. Uses current directory if None.
        do_save: bool
            Flag to prevent saving with do_save=False. Default is True.
    """
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sns.scatterplot(x=X, y=Y, alpha=0.8, s=15, color=scatterplot_color, ax=axs[0])
    axs[0].set_xlabel(scatter_xlabel)
    axs[0].set_ylabel(scatter_ylabel)
    axs[0].set_title(scatter_title)

    for i, (line, label) in enumerate(zip(lines, line_labels)):
        sns.lineplot(
            x=X,
            y=line,
            label=label,
            color=line_palette[i],
            linewidth=1.5,
            ax=axs[1],
        )
    axs[1].set_xlabel(line_xlabel)
    axs[1].set_ylabel(line_ylabel)
    axs[1].set_title(line_title)
    axs[1].legend()

    plt.tight_layout()

    if (file_name is None) != (save_dir is None):
        # This condition is True if exactly one of the two is provided.
        raise ValueError(
            "For saving, both a file name and a save directory must be provided."
        )
    elif file_name and save_dir:
        # Both file_name and save_dir are provided; proceed with saving.
        if do_save:
            base, _ = os.path.splitext(file_name)
            file_name_svg = base + ".svg"
            full_file_path = os.path.join(save_dir, file_name_svg)
            plt.savefig(full_file_path, bbox_inches="tight", format="svg")
            print(f"Plot saved to {full_file_path}")
        else:
            warnings.warn(
                "The synthetic_data plot has not been saved. Flag do_save=True for saving.",
                category=UserWarning,
            )

    plt.show()


def plot_data(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    scatterplot_color: str,
    scatter_xlabel: str,
    scatter_ylabel: str,
    scatter_title: str,
    figsize=(12, 6),
    file_name: str = None,
    save_dir: str = None,
    do_save: bool = True,
):
    """
    Plots data as a scatter plot.

    Parameters:
        X: jnp.ndarray
            1D array of input values.
        Y: jnp.ndarray
            1D array of target/output values.
        scatterplot_color: str
            Color for the scatter plot points.
        scatter_xlabel: str
            X-axis label for the scatter plot.
        scatter_ylabel: str
            Y-axis label for the scatter plot.
        scatter_title: str
            Title for the scatter plot.
        file_name: str
            Filename to save the plot (saves as SVG).
        save_dir: Optional[str], optional
            Directory path to save the plot. Uses current directory if None.
        do_save: bool
            Flag to prevent saving with do_save=False. Default is True.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(x=X, y=Y, alpha=0.8, s=15, color=scatterplot_color, ax=ax)
    ax.set_xlabel(scatter_xlabel)
    ax.set_ylabel(scatter_ylabel)
    ax.set_title(scatter_title)

    plt.tight_layout()

    # Save the plot as an SVG
    if file_name:
        base, _ = os.path.splitext(file_name)
        file_name_svg = base + ".svg"
        if save_dir is not None:
            full_file_path = os.path.join(save_dir, file_name_svg)
        else:
            full_file_path = file_name_svg
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")

    if (file_name is None) != (save_dir is None):
        # This condition is True if exactly one of the two is provided.
        raise ValueError(
            "For saving, both a file name and a save directory must be provided."
        )
    elif file_name and save_dir:
        # Both file_name and save_dir are provided; proceed with saving.
        if do_save:
            base, _ = os.path.splitext(file_name)
            file_name_svg = base + ".svg"
            full_file_path = os.path.join(save_dir, file_name_svg)
            plt.savefig(full_file_path, bbox_inches="tight", format="svg")
            print(f"Plot saved to {full_file_path}")
        else:
            warnings.warn(
                "The data plot has not been saved. Flag do_save=True for saving.",
                category=UserWarning,
            )
    plt.show()


def plot_true_predicted_comparison(
    X_SYN: jnp.ndarray,
    X_PRED: jnp.ndarray,
    true_parameter_values: List[jnp.ndarray],
    predicted_parameter_values: List[jnp.ndarray],
    true_palette: List[str],
    pred_palette: List[str],
    line_labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    fig_size: Tuple[int, int] = (10, 6),
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    do_save: bool = True,
):
    """
    Plots a comparison of the true generated parameter values per observation i against the predicted ones
    The function works for any number parameters.

    Parameters:
        X: jnp.ndarray
            1D array of x-values of the true data
        X_PRED: jnp.ndarray
            1D array of x-values used for predicting
        true_parameter_values: List[jnp.ndarray]
            list of 1D arrays, each containing the original line's y-values.
        predicted_parameter_values: List[jnp.ndarray]
            list of 1D arrays, each containing the predicted line's y-values.
        true_palette: List[str]
            list of color strings for the true parameter.
        pred_palette: List[str]
            list of color strings for the predicted lines.
        parameter_labels: List[str]
            list of labels for each line (e.g., "Location", "Scale", "Shape"). The predicted
            lines will be labeled as "Predicted <label>".
        title: str
            Title of the plot.
        xlabel: str
            Label for the x-axis.
        ylabel: str
            Label for the y-axis.
        fig_size: Tuple[int, int], optional
            Figure size (default is (10, 6)).
        save_dir: Optional[str], optional
            Directory path where the plot will be saved (if provided).
        file_name: Optional[str], optional
            File name (with extension) for saving the plot. The file will be saved as an SVG.
        do_save: bool
            Flag to prevent saving with do_save=False. Default is True.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=fig_size)

    # Iterate over all lines and plot:
    for i, (true, pred, label) in enumerate(
        zip(true_parameter_values, predicted_parameter_values, line_labels)
    ):
        ax.plot(
            X_SYN,
            true,
            linestyle="--",
            color=true_palette[i],
            label=f"True {label}",
        )
        ax.plot(
            X_PRED,
            pred,
            linestyle="-",
            color=pred_palette[i],
            label=f"Predicted {label}",
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()

    if (file_name is None) != (save_dir is None):
        # This condition is True if exactly one of the two is provided.
        raise ValueError(
            "For saving, both a file name and a save directory must be provided."
        )
    elif file_name and save_dir:
        if do_save:
            # Both file_name and save_dir are provided; proceed with saving.
            base, _ = os.path.splitext(file_name)
            file_name_svg = base + ".svg"
            full_file_path = os.path.join(save_dir, file_name_svg)
            plt.savefig(full_file_path, bbox_inches="tight", format="svg")
            print(f"Plot saved to {full_file_path}")
        else:
            warnings.warn(
                "The true_predicted_comparison plot has not been saved. Flag do_save=True for saving.",
                category=UserWarning,
            )
    plt.show()
