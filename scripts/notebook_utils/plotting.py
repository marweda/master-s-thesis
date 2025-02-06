from typing import Optional, Tuple
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns


def plot_elbo(
    num_iterations,
    elbo_values,
    elbo_color,
    initial_percentage,
    final_percentage,
    save_dir=None,
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

    if file_name and save_dir is not None:
        full_file_path = os.path.join(save_dir, file_name) if save_dir else file_name
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")
    elif file_name != save_dir is not None:
        raise ValueError("For saving both a file name and a save dir has to be given.")

    plt.show()


def plot_regression_results(
    scatter_x,
    scatter_y,
    line_x,
    regression_y,
    lower_hdi_bound,
    upper_hdi_bound,
    hdi_alpha,
    palette,
    scatter_label,
    regression_label,
    interval_label,
    xlabel,
    ylabel,
    title,
    fig_size=(12, 7),
    save_dir=None,
    file_name=None,
):
    """
    Plots regression results including scatter data, regression line, and HDI interval.

    Parameters:
        scatter_x: array-like
            X-values of the observed data points.
        scatter_y: array-like
            Y-values of the observed data points.
        line_x: array-like
            X-values for the regression line and HDI intervals.
        regression_y: array-like
            Predicted Y-values of the regression line.
        lower_hdi_bound: array-like
            Lower bounds of the Highest Density Interval (HDI).
        upper_hdi_bound: array-like
            Upper bounds of the Highest Density Interval (HDI).
        hdi_alpha: float
            Transparency level (0-1) for the HDI shaded region.
        palette: list[str]
            List of colors in the order [scatter_color, regression_line_color, hdi_color].
        scatter_label: str
            Legend label for the scatter plot.
        regression_label: str
            Legend label for the regression line.
        interval_label: str
            Legend label for the HDI interval.
        xlabel: str
            Label for the x-axis.
        ylabel: str
            Label for the y-axis.
        title: str
            Title of the plot.
        fig_size: Tuple[int, int], optional
            Figure dimensions (width, height). Default is (12, 7).
        save_dir: Optional[str], optional
            Directory path to save the plot. If None, the plot is not saved.
        file_name: Optional[str], optional
            Filename to save the plot. Saves as SVG if provided.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=fig_size)

    sns.scatterplot(
        x=scatter_x,
        y=scatter_y,
        color=palette[0],
        label=scatter_label,
        alpha=0.7,
        ax=ax,
    )

    sns.lineplot(
        x=line_x,
        y=regression_y,
        color=palette[1],
        label=regression_label,
        linewidth=2.2,
        ax=ax,
    )

    ax.fill_between(
        line_x,
        lower_hdi_bound,
        upper_hdi_bound,
        color=palette[2],
        alpha=hdi_alpha,
        label=interval_label,
    )

    sns.lineplot(x=line_x, y=lower_hdi_bound, color=palette[2], ax=ax, alpha=hdi_alpha)
    sns.lineplot(x=line_x, y=upper_hdi_bound, color=palette[2], ax=ax, alpha=hdi_alpha)

    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if file_name and save_dir is not None:
        base, _ = os.path.splitext(file_name)
        file_name_svg = base + ".svg"
        full_file_path = (
            os.path.join(save_dir, file_name_svg) if save_dir else file_name_svg
        )
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")
    elif file_name != save_dir is not None:
        raise ValueError("For saving both a file name and a save dir has to be given.")

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
    file_name: str,
    save_dir=None,
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

    # Save the plot as an SVG if file_name is provided.
    if file_name and save_dir is not None:
        base, _ = os.path.splitext(file_name)
        file_name_svg = base + ".svg"
        full_file_path = (
            os.path.join(save_dir, file_name_svg) if save_dir else file_name_svg
        )
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")
    elif file_name != save_dir is not None:
        raise ValueError("For saving both a file name and a save dir has to be given.")
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

    if file_name and save_dir is not None:
        base, _ = os.path.splitext(file_name)
        file_name_svg = base + ".svg"
        full_file_path = (
            os.path.join(save_dir, file_name_svg) if save_dir else file_name_svg
        )
        plt.savefig(full_file_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {full_file_path}")
    elif file_name != save_dir is not None:
        raise ValueError("For saving both a file name and a save dir has to be given.")
    plt.show()
