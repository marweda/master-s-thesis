from typing import Optional, Tuple, List, Dict
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
EXCESS_COLOR = "#4b4f4c"  # "#462fbb"  # "#F44336"  009E73
SCATTERPLOT_COLOR = "#4CAF50"  # "#4CAF50"
QUANTILE_COLOR = "#2196F3"
QUANTILE_HDI_COLOR = "#7ec1f7"
GPD_MEAN_COLOR = "#CC6677"
GPD_MEAN_HDI_COLOR = "#d68391"  # d27887
GPD_075Q_COLOR = "#CC6677"
GPD_075Q_HDI_COLOR = "#CC6677"
GPD_025Q_COLOR = "#CC6677"
GPD_025Q_HDI_COLOR = "#CC6677"
QQ_SCATTER_LINE_COLOR = "#7F8C8D"
QQ_SCATTER_QUANTILES_COLOR = "#2C3E50"  # 688797
COLORBLIND_PALETTE = [
    "#462fbb",
    "#15933f",
    "#44AA99",
    "#88CCEE",
    "#CC6677",
    "#AA4499",
    "#882255",
    "#BDAE65",
]
# VIOLIN_REFERENCE_COLOR = "#BDAE65"  # "#CC6677"


def plot_wasserstein_violinplot(
    wasserstein_distances: List[jnp.ndarray],
    reference_wassersteindistances: jnp.ndarray,
    x_labels: List[str],
    reference_x_label: str,
    nrows: int,  # Original number of rows (4 in your case)
    ncols: int,  # Number of columns per original row (7 in your case)
    palette: List[str],
    reference_color: str,
    fig_title: str,
    subplot_titles: List[str],
    y_label: str = "Wasserstein Distance",
    title_fontsize: int = 16,
    label_fontsize: int = 12,
    quartiles_linewidth: float = 2.0,
    quartiles_color: str = "black",
    fig_size: Tuple[int, int] = (11.69, 8.27),  # DIN-A4 landscape (width, height)
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    do_save: bool = True,
):
    """Plots violinplots in a reorganized grid layout."""
    # Convert JAX arrays to NumPy
    main_data = [np.array(w) for w in wasserstein_distances]
    ref_data = np.array(reference_wassersteindistances)

    # Create a grid of subplots (2x2 for 4 original rows)
    grid_rows = 2
    grid_cols = 2
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=fig_size, squeeze=False)
    axs_flat = axs.ravel()  # Flatten for easy iteration
    fig.suptitle(fig_title, fontsize=title_fontsize, y=0.99)

    # Split main data into chunks for each original row
    row_chunks = [main_data[i : i + ncols] for i in range(0, len(main_data), ncols)]

    for row_idx, (row_data, ax) in enumerate(zip(row_chunks, axs_flat)):
        # Prepare combined data (main row + reference)
        combined_data = row_data + [ref_data]
        combined_labels = x_labels + [reference_x_label]

        # Build DataFrame (reset indices to avoid duplicates)
        df_list = []
        for data, label in zip(combined_data, combined_labels):
            temp_df = pd.DataFrame({"value": data, "label": label})
            df_list.append(temp_df)
        plot_df = pd.concat(df_list, ignore_index=True)

        # Create color palette
        row_colors = [palette[i % len(palette)] for i in range(len(row_data))] + [
            reference_color
        ]
        color_map = {label: color for label, color in zip(combined_labels, row_colors)}

        # Create violin plot
        sns.violinplot(
            x="label",
            y="value",
            hue="label",
            data=plot_df,
            palette=color_map,
            inner="quartile",
            cut=0,
            ax=ax,
            inner_kws={"linewidth": quartiles_linewidth, "color": quartiles_color},
            legend=False,
        )

        # Add subplot title (top-center, above the plot)
        ax.text(
            0.5,  # Centered horizontally
            1.02,  # Slightly above the subplot
            subplot_titles[row_idx],
            transform=ax.transAxes,
            fontsize=title_fontsize - 2,
            ha="center",
            va="bottom",
        )

        # Axis labels and ticks
        ax.set_xlabel("")
        ax.set_ylabel(
            y_label if row_idx % grid_cols == 0 else "", fontsize=label_fontsize
        )
        ax.tick_params(axis="x", rotation=45, labelsize=label_fontsize - 2)
        ax.tick_params(axis="y", labelsize=label_fontsize - 2)

        # Show x-axis labels only for bottom row of the grid
        grid_row = row_idx // grid_cols
        if grid_row < grid_rows - 1:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0)

        # Add grid lines
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    if do_save and save_dir and file_name:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    else:
        warnings.warn(
            "The violin plot has not been saved. Flag do_save=True for saving.",
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
    ax_first.set_title(title_first, fontsize=14)
    ax_first.set_xlabel("Iteration", fontsize=13)
    ax_first.set_ylabel("ELBO", fontsize=13)
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
    scatter_size: int,
    line_xs: List[jnp.ndarray],
    line_xs_hdi: List[jnp.ndarray],
    regression_lines: List[jnp.ndarray],
    hdi_lower_bounds: List[jnp.ndarray],
    hdi_upper_bounds: List[jnp.ndarray],
    hdi_alphas: List[float],
    regression_line_colors: List[str],
    regression_line_styles: List[str],
    line_alphas: List[float],
    hdi_colors: List[str],
    hdi_labels: List[str],
    scatter_color: str,
    scatter_label: str,
    regression_line_labels: List[str],
    x_label: str,
    y_label: str,
    title: str,
    fig_size: tuple = (12, 7),
    marker: str = "o",
    file_name: Optional[str] = None,
    excess_mask: Optional[jnp.ndarray] = None,
    excess_color: Optional[str] = None,
    posterior_param_x: Optional[jnp.ndarray] = None,
    posterior_param_lines: Optional[List[jnp.ndarray]] = None,
    posterior_param_line_colors: Optional[List[str]] = None,
    posterior_param_hdis_lower: Optional[List[str]] = None,
    posterior_param_hdis_upper: Optional[List[str]] = None,
    posterior_param_hdi_alphas: Optional[List[str]] = None,
    posterior_param_hdi_labels: Optional[List[str]] = None,
    posterior_param_hdi_colors: Optional[List[str]] = None,
    posterior_param_x_label: Optional[str] = None,
    posterior_param_y_label: Optional[str] = None,
    posterior_param_line_labels: Optional[List[str]] = None,
    posterior_param_titles: Optional[List[str]] = None,
    posterior_fig_size: tuple = (12, 4),
    save_dir: Optional[str] = None,
    y_origin: Optional[float] = None,
    do_save: bool = True,
):
    """Plots regression results with optional subplots for predictive parameters."""
    sns.set_style("whitegrid")
    pred_params_provided = any(
        [
            posterior_param_x is not None,
            posterior_param_lines is not None,
            posterior_param_line_colors is not None,
            posterior_param_x_label is not None,
            posterior_param_y_label is not None,
            posterior_param_line_labels is not None,
            posterior_param_titles is not None,
        ]
    )

    # Create figure layout
    if pred_params_provided:
        num_pred = len(posterior_param_lines) if posterior_param_lines else 0
        main_height, sub_height = fig_size[1], posterior_fig_size[1]
        total_height = main_height + num_pred * sub_height
        total_width = max(fig_size[0], posterior_fig_size[0])
        fig = plt.figure(figsize=(total_width, total_height))
        gs = gridspec.GridSpec(
            num_pred + 1, 1, height_ratios=[fig_size[1]] + [posterior_fig_size[1]] * num_pred
        )
        ax_main = fig.add_subplot(gs[0])
        pred_axes = [fig.add_subplot(gs[i + 1]) for i in range(num_pred)]
    else:
        fig, ax_main = plt.subplots(figsize=fig_size)

    # Main plot: Scatter points (only if scatter_x and scatter_y are provided)
    if scatter_x is not None and scatter_y is not None:
        facecolors="None"
        if marker != "o":
            facecolors=scatter_color
        
        sns.scatterplot(
            x=scatter_x,
            y=scatter_y,
            marker=marker,
            facecolors=facecolors,
            edgecolor=scatter_color,
            label=scatter_label,
            alpha=0.65,
            ax=ax_main,
            s=scatter_size,
        )

        # Highlight excess points
        if excess_mask is not None:
            excess_color = excess_color or "#F44336"
            sns.scatterplot(
                x=scatter_x[excess_mask],
                y=scatter_y[excess_mask],
                marker=marker,
                color=excess_color,
                facecolors=excess_color,
                edgecolor=excess_color,
                label="Excess",
                alpha=0.7,
                ax=ax_main,
                s=scatter_size,
            )

    # Plot each regression line with individual x-values
    for (
        line_x,
        line,
        line_color,
        line_style,
        line_label,
        alpha,
    ) in zip(
        line_xs,
        regression_lines,
        regression_line_colors,
        regression_line_styles,
        regression_line_labels,
        line_alphas,
    ):
        sns.lineplot(
            x=line_x,
            y=line,
            color=line_color,
            linestyle=line_style,
            label=line_label,
            linewidth=2.5,
            ax=ax_main,
            alpha=alpha,
        )

    # Plot each HDI with individual x-values
    for (
        line_x,
        lower,
        upper,
        alpha,
        hdi_color,
        hdi_label,
    ) in zip(
        line_xs_hdi,
        hdi_lower_bounds,
        hdi_upper_bounds,
        hdi_alphas,
        hdi_colors,
        hdi_labels,
    ):
        
        ax_main.fill_between(
            line_x, lower, upper, color=hdi_color, alpha=alpha, label=hdi_label
        )
        sns.lineplot(x=line_x, y=lower, color=hdi_color, alpha=alpha, ax=ax_main)
        sns.lineplot(x=line_x, y=upper, color=hdi_color, alpha=alpha, ax=ax_main)

    # Set y-axis origin if specified
    if y_origin is not None:
        ax_main.set_ylim(bottom=y_origin)

    ax_main.legend(fontsize=12, loc="upper left")
    ax_main.set_xlabel(x_label, fontsize=13)
    ax_main.set_ylabel(y_label, fontsize=13)
    ax_main.set_title(title, fontsize=14)

    # Plot predictive parameter subplots
    if pred_params_provided and posterior_param_lines:
        for ax, line, color, label, sub_title, lower, upper, alpha, hdi_label, hdi_color in zip(
            pred_axes,
            posterior_param_lines,
            posterior_param_line_colors or [None] * len(posterior_param_lines),
            posterior_param_line_labels or [None] * len(posterior_param_lines),
            posterior_param_titles or [None] * len(posterior_param_lines),
            posterior_param_hdis_lower,
            posterior_param_hdis_upper,
            posterior_param_hdi_alphas,
            posterior_param_hdi_labels,
            posterior_param_hdi_colors,
        ):
            ax.fill_between(
                posterior_param_x, lower, upper, color=hdi_color, alpha=alpha, label=hdi_label
            )
            sns.lineplot(x=line_x, y=lower, color=hdi_color, alpha=alpha, ax=ax)
            sns.lineplot(x=line_x, y=upper, color=hdi_color, alpha=alpha, ax=ax)
            sns.lineplot(x=posterior_param_x, y=line, color=color, label=label, ax=ax)
            ax.set_xlabel(posterior_param_x_label)
            ax.set_ylabel(posterior_param_y_label)
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
    Plots synthetic data as a scatter plot and compares multiple lines (e.g., true vs predicted) in separate figures.
    """
    sns.set_theme(style="whitegrid")

    # Create scatter plot figure
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=X, y=Y, alpha=0.8, s=15, color=scatterplot_color, ax=ax1)
    ax1.set_xlabel(scatter_xlabel, fontsize=13)
    ax1.set_ylabel(scatter_ylabel, fontsize=13)
    ax1.set_title(scatter_title, fontsize=14)
    fig1.tight_layout()

    # Create line plot figure
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for i, (line, label) in enumerate(zip(lines, line_labels)):
        sns.lineplot(
            x=X,
            y=line,
            label=label,
            color=line_palette[i],
            linewidth=2.5,
            ax=ax2,
        )
    ax2.set_xlabel(line_xlabel, fontsize=13)
    ax2.set_ylabel(line_ylabel, fontsize=13)
    ax2.set_title(line_title, fontsize=14)
    ax2.legend()
    fig2.tight_layout()

    # Handling saving of both figures
    if (file_name is None) != (save_dir is None):
        raise ValueError(
            "For saving, both a file name and a save directory must be provided."
        )
    elif file_name and save_dir:
        if do_save:
            base, _ = os.path.splitext(file_name)
            scatter_file = base + "_scatter.svg"
            line_file = base + "_line.svg"

            # Save scatter plot
            scatter_path = os.path.join(save_dir, scatter_file)
            fig1.savefig(scatter_path, bbox_inches="tight", format="svg")
            print(f"Scatter plot saved to {scatter_path}")

            # Save line plot
            line_path = os.path.join(save_dir, line_file)
            fig2.savefig(line_path, bbox_inches="tight", format="svg")
            print(f"Line plot saved to {line_path}")
        else:
            warnings.warn(
                "The synthetic_data plots have not been saved. Set do_save=True for saving.",
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
    ax.set_xlabel(scatter_xlabel, fontsize=13)
    ax.set_ylabel(scatter_ylabel, fontsize=13)
    ax.set_title(scatter_title, fontsize=14)

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
    predicted_hdi_lower: List[jnp.ndarray],
    predicted_hdi_upper: List[jnp.ndarray],
    hdi_colors: List[jnp.ndarray],
    hdi_alphas: List[float],
    hdi_label: List[str],
    true_palette: List[str],
    posterior_palette: List[str],
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
        predicted_hdi_lower: List[jnp.ndarray]
            tba
        predicted_hdi_upper: List[jnp.ndarray]
            tba
        hdi_colors: List[jnp.ndarray]
            tba
        hdi_alphas: List[float]
            tba
        hdi_label: List[str]
            tba
        true_palette: List[str]
            list of color strings for the true parameter.
        posterior_palette: List[str]
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
    for i, (
        true,
        pred,
        label,
        hdi_color,
        lower,
        upper,
        alpha,
        hdi_label,
        t_color,
        p_color,
    ) in enumerate(
        zip(
            true_parameter_values,
            predicted_parameter_values,
            line_labels,
            hdi_colors,
            predicted_hdi_lower,
            predicted_hdi_upper,
            hdi_alphas,
            hdi_label,
            true_palette,
            posterior_palette,
        )
    ):
        ax.plot(
            X_SYN,
            true,
            linestyle="--",
            color=t_color,
            label=f"True {label}",
        )
        ax.plot(
            X_PRED,
            pred,
            linestyle="-",
            color=p_color,
            label=f"Predicted {label}",
        )
        ax.fill_between(
            X_PRED, lower, upper, color=hdi_color, alpha=alpha, label=hdi_label
        )
        ax.plot(
            X_PRED,
            lower,
            linestyle="--",
            color=hdi_color,
            alpha=alpha,
        )
        ax.plot(
            X_PRED,
            upper,
            linestyle="--",
            color=hdi_color,
            alpha=alpha,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
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


def prior_predictive_plot(
    line_x: jnp.ndarray,
    y_list: List[jnp.ndarray],
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "Prior Predictive Plot",
    alpha: float = 0.5,
    color: str = "blue",
    fig_size: Tuple[int, int] = (10, 6),
    file_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    do_save: bool = True,
):
    """
    Plots a set of prior predictive lines against a common x-axis.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=fig_size)

    # Ensure that line_x and each y in y_list are numpy arrays
    x_values = np.array(line_x)
    df_list = []
    for idx, y in enumerate(y_list):
        df_temp = pd.DataFrame({"x": x_values, "y": np.array(y)})
        df_temp["line"] = idx  # group identifier for each line
        df_list.append(df_temp)
    plot_df = pd.concat(df_list, ignore_index=True)

    # Plot all lines at once using the long-form DataFrame.
    # `units="line"` and `estimator=None` ensure each group is plotted as its own line.
    sns.lineplot(
        data=plot_df,
        x="x",
        y="y",
        units="line",
        estimator=None,
        color=color,
        alpha=alpha,
        legend=False,
        ax=ax,
    )

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if file_name and save_dir:
        if do_save:
            os.makedirs(save_dir, exist_ok=True)
            full_path = os.path.join(
                save_dir,
                file_name if file_name.endswith(".svg") else f"{file_name}.svg",
            )
            plt.savefig(full_path, bbox_inches="tight", format="svg")
            print(f"Plot saved to {full_path}")
        else:
            warnings.warn(
                "The prior predictive plot has not been saved. Flag do_save=True for saving.",
                category=UserWarning,
            )
    elif (file_name is None) != (save_dir is None):
        raise ValueError(
            "Both file_name and save_dir must be provided to save the plot."
        )

    plt.show()


def plot_support_checks_lineplot(
    support_data_dict: Dict,
    Ns: List[int],
    epochs_max: int,
    colorpalette: List[str],
    save_dir: str = None,
    file_name: str = "support_checks_lineplot.svg",
    do_save: bool = True,
    y_cutoff_lower: Optional[float] = 0.57,
    y_cutoff_upper: float = 1.005,
    y_cutoff_lower_right: Optional[float] = -0.005,
    y_cutoff_upper_right: Optional[float] = 1.01,
    main_title: str = "Responses' Support Validity Progression by Sample Size during Optimization",
    x_label: str = "Epoch",
    y_label: str = "Proportion of responses within support",
    figsize: Tuple[float, float] = (12, 8),
    legend_fontsize: int = 12,
    x_cut: int = 1300,
    x_prolonging: int = 500,
    subplot_width_ratio: List[float] = [3, 1.2],
) -> None:
    # Build DataFrame for main plot
    data = []
    for N in Ns:
        N_str = str(N)
        if N_str not in support_data_dict:
            continue
        runs = support_data_dict[N_str]
        for run_name, run_data in runs.items():
            if "support_means" not in run_data:
                continue
            full_array = np.array(
                [
                    run_data["support_means"].get(f"epoch_{e+1}", np.nan)
                    for e in range(epochs_max)
                ]
            )
            for epoch, mean in enumerate(full_array):
                data.append(
                    {
                        "N": f"N={N}",
                        "Epoch": epoch + 1,
                        "Proportion": mean,
                        "Run": f"{N}_{run_name}",
                    }
                )
    df = pd.DataFrame(data)
    if df.empty:
        warnings.warn("No data found for plotting.", UserWarning)
        return

    # Calculate cut points
    right_start = max(epochs_max - x_prolonging, x_cut + 1)
    left_df = df[df["Epoch"] <= x_cut]
    right_df = df[df["Epoch"] >= right_start]

    # Create figure with separate y-axes
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=subplot_width_ratio, wspace=0.1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Axis styling
    ax1.spines[["right"]].set_visible(False)
    ax2.spines[["left"]].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax2.tick_params(labelright=True)

    # Plot data with independent y-axes
    sns.lineplot(
        data=left_df,
        x="Epoch",
        y="Proportion",
        hue="N",
        units="Run",
        alpha=0.3,
        palette=colorpalette,
        legend=False,
        ax=ax1,
        estimator=None,
        linestyle="-",
        linewidth=2.0,
    )
    sns.lineplot(
        data=right_df,
        x="Epoch",
        y="Proportion",
        hue="N",
        units="Run",
        alpha=0.3,
        palette=colorpalette,
        legend=False,
        ax=ax2,
        estimator=None,
        linestyle="-",
        linewidth=2.0,
    )

    # Set axis limits with independent cutoffs
    ax1.set(
        xlim=(0, x_cut),
        ylim=(y_cutoff_lower if y_cutoff_lower is not None else 0, y_cutoff_upper),
    )
    ax2.set(
        xlim=(right_start, epochs_max),
        ylim=(
            (
                y_cutoff_lower_right
                if y_cutoff_lower_right is not None
                else (y_cutoff_lower or 0)
            ),
            (
                y_cutoff_upper_right
                if y_cutoff_upper_right is not None
                else y_cutoff_upper
            ),
        ),
    )

    # Add break markers
    d = 0.5
    break_kw = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **break_kw)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **break_kw)
    ax1.grid(True)
    ax2.grid(True)

    # Labels and titles
    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel(y_label, fontsize=14)
    ax2.set_xlabel(None, fontsize=14)
    ax2.set_ylabel(None, fontsize=14)
    ax1.xaxis.set_label_coords(0.8, -0.05)
    plt.suptitle(main_title, y=0.93, fontsize=15)

    # Legend on right plot center
    handles = [
        plt.Line2D([0], [0], color=color, label=n, lw=3)
        for n, color in zip(left_df["N"].unique(), colorpalette)
    ]
    legend = ax2.legend(
        handles=handles,
        title="N",
        loc="best",
        #bbox_to_anchor=(0.95, 0.82),
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize + 2,
    )
    for line in legend.get_lines():
        line.set_linewidth(3.0)
        line.set_alpha(1.0)

    plt.tight_layout()

    if do_save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, file_name), bbox_inches="tight", format="svg"
        )

    plt.show()


def plot_support_checks_histogram(
    support_data_dict: Dict,
    Ns: List[int],
    epochs_max: int,
    colorpalette: List[str],
    save_dir: str = None,
    file_name: str = "support_checks_histogram.svg",
    do_save: bool = True,
    x_label: str = "Proportions",
    y_label: str = "Rel. freq. of proportions",
    figsize: Tuple[float, float] = (8, 6),
    legend_fontsize: int = 12,
) -> None:
    # Build DataFrame for last epoch
    title = f"Last Epoch's Support Validity Distribution: Epoch {epochs_max}"
    data = []
    for N in Ns:
        N_str = str(N)
        if N_str not in support_data_dict:
            continue
        runs = support_data_dict[N_str]
        for run_name, run_data in runs.items():
            if "support_means" not in run_data:
                continue
            # Get the last epoch's mean
            last_epoch_key = f"epoch_{epochs_max}"
            mean = run_data["support_means"].get(last_epoch_key, np.nan)
            data.append(
                {
                    "N": f"N={N}",
                    "Epoch": epochs_max,
                    "Proportion": mean,
                    "Run": f"{N}_{run_name}",
                }
            )
    df_last = pd.DataFrame(data)
    if df_last.empty:
        warnings.warn("No data found for plotting the histogram.", UserWarning)
        return
    df_last = df_last.dropna(subset=["Proportion"])
    if df_last.empty:
        warnings.warn(
            "No valid data (excluding NaNs) for plotting the histogram.", UserWarning
        )
        return

    # Process the data into buckets
    epsilon = 1e-6

    def assign_bucket(p):
        if abs(p - 0.0) < epsilon:
            return "0.0"
        if abs(p - 1.0) < epsilon:
            return "1.0"
        return "(0.0, 1.0)"

    df_last["Bucket"] = df_last["Proportion"].apply(assign_bucket)
    bucket_order = ["0.0", "(0.0, 1.0)", "1.0"]

    # Group and calculate proportions
    grouped = (
        df_last.groupby(["N", "Bucket"])
        .size()
        .reset_index(name="Count")
        .assign(
            Proportion=lambda x: x.groupby("N")["Count"].transform(
                lambda s: s / s.sum()
            )
        )
    )
    grouped["Bucket"] = pd.Categorical(
        grouped["Bucket"], categories=bucket_order, ordered=True
    )
    grouped = grouped.sort_values(["Bucket", "N"])

    # Create the plot
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.grid(True)
    sns.barplot(
        data=grouped,
        x="Bucket",
        y="Proportion",
        hue="N",
        palette=colorpalette,
        dodge=True,
        order=bucket_order,
        ax=ax,
    )
    # Set labels and title
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    plt.title(title, fontsize=14)

    # Handle legend with correct color ordering based on original Ns
    present_Ns = df_last["N"].unique().tolist()
    present_Ns_integers = [int(n.split("=")[1]) for n in present_Ns]
    # Sort present_Ns according to their original position in Ns
    present_Ns_sorted = sorted(present_Ns, key=lambda x: Ns.index(int(x.split("=")[1])))
    # Get corresponding color indices from original colorpalette
    color_indices = [Ns.index(int(n.split("=")[1])) for n in present_Ns_sorted]
    colors = [colorpalette[i] for i in color_indices]

    handles = [
        plt.Line2D([0], [0], color=color, label=n, lw=3)
        for n, color in zip(present_Ns_sorted, colors)
    ]
    legend = ax.legend(
        handles=handles,
        title="N",
        loc="best",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize + 2,
    )
    for line in legend.get_lines():
        line.set_linewidth(3.0)
        line.set_alpha(1.0)

    plt.tight_layout()

    if do_save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, file_name), bbox_inches="tight", format="svg"
        )

    plt.show()
