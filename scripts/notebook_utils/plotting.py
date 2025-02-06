from typing import Optional, Tuple
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns


REGRESSION_COLORPALETTE = ["#4CAF50", "#2196F3", "#424242"]
TRUEPARAM_COLORPALETTE = ["#57a7a8", "#506eaf", "#b04fa4"]
PREDPARAM_COLORPALETTE = ["#00FFED", "#0055FF", "#FF00A5"]
ELBO_COLOR = "#2C3E50"


def plot_elbo(
    num_iterations,
    elbo_values,
    elbo_color,
    initial_percentage,
    final_percentage,
    save_dir: str = None,
    file_name=None,
):
    """
    Plots the ELBO over iterations along with two zoomed-in plots for the first initial_percentage
    and last final_percentage of iterations.

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
    """
    sns.set_theme(style="whitegrid")

    # Create a figure with GridSpec: the top row (row 0) will be the main plot;
    # the bottom row (row 1) is split into two columns for the zoomed-in plots.
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[2, 1])

    ax_main = fig.add_subplot(gs[0, :])
    sns.lineplot(x=range(num_iterations), y=elbo_values, ax=ax_main, color=elbo_color)
    ax_main.set_title("ELBO over Iterations", fontsize=15)
    ax_main.set_xlabel("Iteration", fontsize=13)
    ax_main.set_ylabel("ELBO", fontsize=13)

    # Calculate ranges for zoomed plots
    num_first = max(1, int(num_iterations * initial_percentage))
    num_last = max(1, int(num_iterations * final_percentage))
    start_last = num_iterations - num_last

    # First zoomed plot
    ax_first = fig.add_subplot(gs[1, 0])
    sns.lineplot(
        x=range(num_first), y=elbo_values[:num_first], ax=ax_first, color=elbo_color
    )
    ax_first.set_title(
        f"First {initial_percentage*100:.0f}% of Iterations", fontsize=13
    )
    ax_first.set_xlabel("Iteration", fontsize=12)
    ax_first.set_ylabel("ELBO", fontsize=12)

    # Second zoomed plot
    ax_last = fig.add_subplot(gs[1, 1])
    sns.lineplot(
        x=range(start_last, num_iterations),
        y=elbo_values[start_last:],
        ax=ax_last,
        color=elbo_color,
    )
    ax_last.set_title(f"Last {final_percentage*100:.0f}% of Iterations", fontsize=13)
    ax_last.set_xlabel("Iteration", fontsize=12)
    ax_last.set_ylabel("ELBO", fontsize=12)

    plt.tight_layout()

    if (file_name is None) != (save_dir is None):
        # This condition is True if exactly one of the two is provided.
        raise ValueError(
            "For saving, both a file name and a save directory must be provided."
        )
    elif file_name and save_dir:
        # Both file_name and save_dir are provided; proceed with saving.
        base, _ = os.path.splitext(file_name)
        file_name_svg = base + ".svg"
        full_file_path = os.path.join(save_dir, file_name_svg)
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")

    plt.show()


def plot_regression_results(
    scatter_x: jnp.ndarray,
    scatter_y: jnp.ndarray,
    x_pred: jnp.ndarray,
    pred_mean: jnp.ndarray,
    lower_hdi_bound: np.ndarray,
    upper_hdi_bound: np.ndarray,
    hdi_alpha: float,
    regression_colorpalette: list[str],
    scatter_label: str,
    regression_label: str,
    hdi_label: str,
    xlabel: str,
    ylabel: str,
    title: str,
    fig_size=(12, 7),
    save_dir: str = None,
    file_name=None,
):
    """
    Plots regression results including observed data, a regression line, and the HDI interval.

    Parameters:
        scatter_x (jnp.ndarray):
            Array of x-values for the observed data points.
        scatter_y (jnp.ndarray):
            Array of y-values for the observed data points.
        x_pred (jnp.ndarray):
            Array of x-values used for generating predictions.
        pred_mean (jnp.ndarray):
            Array of predicted mean values corresponding to x_pred.
        lower_hdi_bound (np.ndarray):
            Array representing the lower bounds of the Highest Density Interval (HDI).
        upper_hdi_bound (np.ndarray):
            Array representing the upper bounds of the Highest Density Interval (HDI).
        hdi_alpha (float):
            Transparency level (between 0 and 1) for the HDI shaded region.
        regression_colorpalette (list[str]):
            List of color strings in the order [scatter_color, regression_line_color, hdi_color].
        scatter_label (str):
            Label for the scatter plot legend.
        regression_label (str):
            Label for the regression line legend.
        hdi_label (str):
            Label for the HDI interval legend.
        xlabel (str):
            Label for the x-axis.
        ylabel (str):
            Label for the y-axis.
        title (str):
            Title of the plot.
        fig_size (tuple[int, int], optional):
            Dimensions of the figure in inches as (width, height). Defaults to (12, 7).
        save_dir (Optional[str], optional):
            Directory path to save the plot. If None, the plot is not saved.
        file_name (Optional[str], optional):
            Filename for saving the plot (saves as SVG if provided). Defaults to None.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=fig_size)

    sns.scatterplot(
        x=scatter_x,
        y=scatter_y,
        color=regression_colorpalette[0],
        label=scatter_label,
        alpha=0.7,
        ax=ax,
    )

    sns.lineplot(
        x=x_pred,
        y=pred_mean,
        color=regression_colorpalette[1],
        label=regression_label,
        linewidth=2.2,
        ax=ax,
    )

    ax.fill_between(
        x_pred,
        lower_hdi_bound,
        upper_hdi_bound,
        color=regression_colorpalette[2],
        alpha=hdi_alpha,
        label=hdi_label,
    )

    sns.lineplot(
        x=x_pred,
        y=lower_hdi_bound,
        color=regression_colorpalette[2],
        ax=ax,
        alpha=hdi_alpha,
    )
    sns.lineplot(
        x=x_pred,
        y=upper_hdi_bound,
        color=regression_colorpalette[2],
        ax=ax,
        alpha=hdi_alpha,
    )

    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if (file_name is None) != (save_dir is None):
        # This condition is True if exactly one of the two is provided.
        raise ValueError(
            "For saving, both a file name and a save directory must be provided."
        )
    elif file_name and save_dir:
        # Both file_name and save_dir are provided; proceed with saving.
        base, _ = os.path.splitext(file_name)
        file_name_svg = base + ".svg"
        full_file_path = os.path.join(save_dir, file_name_svg)
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")

    plt.show()


def plot_synthetic_data(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    scatterplot_color: str,
    line_palette: list[str],
    lines: list,
    scatter_xlabel: str,
    scatter_ylabel: str,
    scatter_title: str,
    line_xlabel: str,
    line_title: str,
    line_labels: list,
    file_name: str = None,
    save_dir: str = None,
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
        line_palette: list[str]
            List of colors for each line in the line plot.
        lines: list
            List of 1D arrays, each representing a line to plot (e.g., true function, predictions).
        scatter_xlabel: str
            X-axis label for the scatter plot.
        scatter_ylabel: str
            Y-axis label for the scatter plot.
        scatter_title: str
            Title for the scatter plot.
        line_xlabel: str
            X-axis label for the line plot.
        line_title: str
            Title for the line plot.
        line_labels: list
            Legend labels for each line in the line plot.
        file_name: str
            Filename to save the plot (saves as SVG).
        save_dir: Optional[str], optional
            Directory path to save the plot. Uses current directory if None.
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
        base, _ = os.path.splitext(file_name)
        file_name_svg = base + ".svg"
        full_file_path = os.path.join(save_dir, file_name_svg)
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")

    plt.show()


def plot_data(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    scatterplot_color: str,
    scatter_xlabel: str,
    scatter_ylabel: str,
    scatter_title: str,
    file_name: str = None,
    save_dir: str = None,
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
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6))

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
        base, _ = os.path.splitext(file_name)
        file_name_svg = base + ".svg"
        full_file_path = os.path.join(save_dir, file_name_svg)
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")
    plt.show()


def plot_true_predicted_comparison(
    X: jnp.ndarray,
    true_parameter_values: list[jnp.ndarray],
    predicted_parameter_values: list[jnp.ndarray],
    true_palette: list[str],
    pred_palette: list[str],
    line_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    fig_size: Tuple[int, int] = (10, 6),
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
) -> None:
    """
    Plots a comparison of the true generated parameter values per observation i against the predicted ones
    The function works for any number parameters.

    Parameters:
        X: jnp.ndarray
            1D array of x-values.
        true_parameter_values: list[jnp.ndarray]
            list of 1D arrays, each containing the original line's y-values.
        predicted_parameter_values: list[jnp.ndarray]
            list of 1D arrays, each containing the predicted line's y-values.
        true_palette: list[str]
            list of color strings for the true parameter.
        pred_palette: list[str]
            list of color strings for the predicted lines.
        parameter_labels: list[str]
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
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=fig_size)

    # Sort X and reorder the original and predicted arrays accordingly.
    sort_idx = jnp.argsort(X)
    xs_sorted = X[sort_idx]
    original_sorted = [line[sort_idx] for line in true_parameter_values]
    predicted_sorted = [line[sort_idx] for line in predicted_parameter_values]

    # Iterate over all lines and plot:
    for i, (orig, pred, label) in enumerate(
        zip(original_sorted, predicted_sorted, line_labels)
    ):
        ax.plot(
            xs_sorted,
            orig,
            linestyle="--",
            color=true_palette[i],
            label=f"True {label}",
        )
        ax.plot(
            xs_sorted,
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
        # Both file_name and save_dir are provided; proceed with saving.
        base, _ = os.path.splitext(file_name)
        file_name_svg = base + ".svg"
        full_file_path = os.path.join(save_dir, file_name_svg)
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")
    plt.show()
