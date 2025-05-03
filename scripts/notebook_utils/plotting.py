from typing import Optional, Tuple, List, Dict
import os
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
import matplotlib.ticker as ticker
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
PEAK_COLOR = "#820020"
QQ_SCATTER_QUANTILES_COLOR = "#2C3E50"  # 688797
COLORBLIND_PALETTE = [
    "#715cd6",
    "#1bbd51",
    "#44AA99",
    "#88CCEE",
    "#CC6677",
    "#AA4499",
    "#882255",
    "#BDAE65",
]
SUPP_PROPORTION_COLOR = "#FFDB58"
LOGPDF_COLOR = "#2CA02C"
# VIOLIN_REFERENCE_COLOR = "#BDAE65"  # "#CC6677"


def plot_mean_excesses(
    mean_excesses: Dict,
    color: str = GPD_MEAN_COLOR,
    hdi_color: str = GPD_MEAN_HDI_COLOR,
    do_save: bool = False,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    fig_size: Tuple[float, float] = (10, 6),
    title: str = "Mean Excesses with 95% HDI across Quantiles",
    x_label: str = "Quantile",
    y_label: str = "Mean Excess",
) -> None:
    """
    Plots mean excesses with HDI intervals across quantiles, using gradient based on number of excesses.
    """
    # Extract and sort quantiles
    quantiles = sorted(mean_excesses.keys(), key=float)
    excess_counts = [mean_excesses[q]["#excesses"] for q in quantiles]

    # Create colormap based on number of excesses
    norm = plt.Normalize(min(excess_counts), max(excess_counts))
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Create figure
    plt.figure(figsize=fig_size)
    sns.set_style("whitegrid")
    ax = plt.gca()

    # Plot each point individually with color scaling
    for q in quantiles:
        data = mean_excesses[q]
        color_strength = norm(data["#excesses"])

        # Convert array values to scalars if needed
        mean = np.array(data["excess_mean"]).item()
        lower = (
            np.array(data["lower_mean_hdi"]).mean()
            if isinstance(data["lower_mean_hdi"], np.ndarray)
            else data["lower_mean_hdi"]
        )
        upper = (
            np.array(data["upper_mean_hdi"]).mean()
            if isinstance(data["upper_mean_hdi"], np.ndarray)
            else data["upper_mean_hdi"]
        )

        # Plot individual point with error bar
        plt.errorbar(
            x=q,
            y=mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt="o",
            ecolor=cmap(color_strength),
            elinewidth=2,
            capsize=5,
            capthick=2,
            markersize=8,
            markerfacecolor=cmap(color_strength),
            markeredgecolor="white",
        )

    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Number of Excesses", rotation=270, labelpad=15)

    # Add connecting line
    sorted_quantiles = sorted(quantiles)
    sorted_means = [mean_excesses[q]["excess_mean"].item() for q in sorted_quantiles]
    ax.plot(
        sorted_quantiles,
        sorted_means,
        color=color,
        linestyle="--",
        alpha=0.5,
        label="Trend Line",
    )

    # Formatting
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Save handling
    if do_save:
        if not (save_dir and filename):
            raise ValueError("Both save_dir and filename must be provided for saving")

        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path, bbox_inches="tight", dpi=300, format="svg")
        print(f"Plot saved to {full_path}")
        plt.show()
        plt.close()

    plt.show()


def plot_kde(
    svi_posterior_dict: dict,
    mcmc_dict: dict,
    main_title: str,
    fig_size: tuple,
    save_dir: str,
    file_name: str,
    do_save: bool,
):
    """Plot KDEs with y-labels only in first column and preserved ticks."""
    # Layout configuration
    config = {
        "title_size": 9,
        "label_size": 8,
        "tick_size": 6,
        "row_spacing": 1.0,
        "col_spacing": 1.0,
        "legend_xoffset": 0.25,
        "legend_yoffset": 0.21,
        "main_rows": 6,
        "main_cols": 7,
        "special_cols": [2, 4],
        "tick_pad": 1,
        "x_num_ticks": 3,
        "y_num_ticks": 3,
    }

    # Create figure with slightly larger size
    fig = plt.figure(figsize=fig_size)

    # Create grid with better spacing
    gs = fig.add_gridspec(
        nrows=8,
        ncols=7,
        hspace=config["row_spacing"],
        wspace=config["col_spacing"],
        height_ratios=[1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
    )

    # Dictionary to store axis objects for later reference
    all_axes = {}

    # Access samples
    svi_samples = svi_posterior_dict["samples"][0]
    mcmc_samples = mcmc_dict["samples"][0]

    # Plot gamma parameters
    for section in [("spline_scale_coef", 0), ("spline_shape_coef", 3)]:
        param_key, row_offset = section
        for i in range(21):
            row = i // 7 + row_offset
            col = i % 7
            ax = fig.add_subplot(gs[row, col])
            all_axes[(row, col)] = ax  # Store axis for later reference

            svi_data = svi_samples[param_key][:, i].flatten()
            mcmc_data = mcmc_samples[param_key].reshape(-1, 21)[:, i].flatten()
            sns.kdeplot(svi_data, ax=ax, color="blue", label="SVI")
            sns.kdeplot(mcmc_data, ax=ax, color="orange", label="MCMC")

            # Set title with proper LaTeX formatting
            ax.set_title(
                f"{'scale' if 'scale' in param_key else 'shape'} $\gamma_{{{i+1}}}$",
                fontsize=config["title_size"],
            )

            # Set y-label only for first column
            ax.set_ylabel("Density" if col == 0 else "", fontsize=config["label_size"])

            # Adjust tick parameters - closer to axis
            ax.tick_params(
                axis="both", labelsize=config["tick_size"], pad=config["tick_pad"]
            )

            # Control number of ticks
            ax.xaxis.set_major_locator(plt.MaxNLocator(config["x_num_ticks"]))
            ax.yaxis.set_major_locator(plt.MaxNLocator(config["y_num_ticks"]))

    # Plot intercepts and lambdas (no y-labels)
    for row_idx, params in enumerate(
        [
            (6, ["intercept_scale", "intercept_shape"], r"$\beta_0$"),  # Use raw string
            (
                7,
                ["lambda_smooth_scale", "lambda_smooth_shape"],
                r"$\lambda^2$",
            ),  # Use raw string
        ]
    ):
        row, param_keys, symbol = params
        for i, param_key in enumerate(param_keys):
            col = config["special_cols"][i]
            ax = fig.add_subplot(gs[row, col])
            all_axes[(row, col)] = ax  # Store axis for later reference

            svi_data = svi_samples[param_key].flatten()
            mcmc_data = mcmc_samples[param_key].flatten()
            sns.kdeplot(svi_data, ax=ax, color="blue")
            sns.kdeplot(mcmc_data, ax=ax, color="orange")

            # Set specific x-axis limits for lambda plots
            if row == 7:  # Lambda row
                if col == 2:  # First lambda plot
                    ax.set_xlim(left=-0.03, right=0.03)
                elif col == 4:  # Second lambda plot
                    ax.set_xlim(right=0.02)

            # Set title with proper LaTeX formatting and parameter name
            param_name = param_key.split("_")[-1]
            ax.set_title(f"{param_name} {symbol}", fontsize=config["title_size"])

            ax.set_ylabel("")  # Explicitly remove y-label

            # Adjust tick parameters - closer to axis
            ax.tick_params(
                axis="both", labelsize=config["tick_size"], pad=config["tick_pad"]
            )

            # Control number of ticks
            ax.xaxis.set_major_locator(plt.MaxNLocator(config["x_num_ticks"]))
            ax.yaxis.set_major_locator(plt.MaxNLocator(config["y_num_ticks"]))

    # Add unified legend
    handles = [
        plt.Line2D([0], [0], color="blue", lw=2, label="SVI"),
        plt.Line2D([0], [0], color="orange", lw=2, label="MCMC"),
    ]
    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(config["legend_xoffset"], config["legend_yoffset"]),
        fontsize=config["label_size"],
    )

    # Add main title
    # fig.suptitle(main_title, y=0.94, fontsize=13)

    # Adjust layout to use available space better
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Add horizontal separator lines between rows with proper horizontal boundaries
    def add_horizontal_lines():
        # Define the rows to add lines between (0-indexed)
        row_pairs = [(2, 3), (5, 6), (6, 7), (7, 8)]

        # Find the leftmost and rightmost axes to determine horizontal boundaries
        min_x = 1.0
        max_x = 0.0

        for (row, col), ax in all_axes.items():
            bbox = ax.get_position()
            min_x = min(min_x, bbox.x0)
            max_x = max(max_x, bbox.x1)

        # Add some margin to avoid exact edges
        margin = 0.01
        left_boundary = min_x + margin
        right_boundary = max_x - margin

        # Add horizontal lines
        for row1, row2 in row_pairs:
            # Find representative axes for each row
            ax1 = None
            ax2 = None

            # Look for axes in these rows
            for col in range(7):
                if (row1, col) in all_axes:
                    ax1 = all_axes[(row1, col)]
                    break

            for col in range(7):
                if (row2, col) in all_axes:
                    ax2 = all_axes[(row2, col)]
                    break

            if ax1 and ax2:
                # Get bounding boxes
                bbox1 = ax1.get_position()
                bbox2 = ax2.get_position()

                # Calculate y-position (midway between rows)
                y_pos = (bbox1.y0 + bbox2.y1) / 2

                # Create line with constrained width
                line = plt.Line2D(
                    [left_boundary, right_boundary],
                    [y_pos, y_pos],
                    transform=fig.transFigure,
                    color="black",
                    linestyle="-",
                    linewidth=0.5,
                )
                fig.add_artist(line)

    # Add the horizontal lines
    add_horizontal_lines()

    # Save if requested
    if do_save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()


def plot_posterior_comparison(
    map_hdi_results: Dict[str, Dict[str, np.ndarray]],
    x_values: np.ndarray,
    fig_size: Tuple[float, float] = (12, 8),
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    do_save: bool = True,
    title: str = "Posterior Parameter Comparison",
    x_label: str = "Days since Jan/01/1980",
    y_labels: Tuple[str, str] = ("GPD Scale Parameter", "GPD Shape Parameter"),
    colors: Tuple[str, str] = ("blue", "orange"),
    hdi_alpha: float = 0.2,
):
    """Plot comparison of posterior estimates with MAPs and HDIs."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size)

    # Plot scale parameter results
    for method, color in zip(["svi", "mcmc"], colors):
        ax1.plot(
            x_values,
            map_hdi_results[method]["scale_map"],
            color=color,
            label=f"{method.upper()} Estimate",
        )
        ax1.fill_between(
            x_values,
            map_hdi_results[method]["scale_hdi"][:, 0],
            map_hdi_results[method]["scale_hdi"][:, 1],
            color=color,
            alpha=hdi_alpha,
            label=f"{method.upper()} 95% HDI",
        )

    # Plot shape parameter results
    for method, color in zip(["svi", "mcmc"], colors):
        ax2.plot(x_values, map_hdi_results[method]["shape_map"], color=color)
        ax2.fill_between(
            x_values,
            map_hdi_results[method]["shape_hdi"][:, 0],
            map_hdi_results[method]["shape_hdi"][:, 1],
            color=color,
            alpha=hdi_alpha,
        )

    # Formatting
    ax1.set_title(y_labels[0], fontsize=12)
    ax2.set_title(y_labels[1], fontsize=12)
    ax1.set_xlabel(x_label, fontsize=11)
    ax2.set_xlabel(x_label, fontsize=11)

    for ax in [ax1, ax2]:
        ax.set_ylabel("Parameter Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center")

    # plt.suptitle(title, y=0.98, fontsize=14)
    plt.tight_layout()

    # Save handling
    if do_save:
        if not (save_dir and file_name):
            raise ValueError("Both save_dir and file_name required for saving")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_wasserstein_violinplot(
    wd_dict: dict,
    svi_epoch_order: List[str],
    sample_size_order: List[str],
    svi_colors: List[str],
    mcmc_color: str = "gray",
    main_title: str = "Wasserstein Distance Comparison to MCMC Baseline: 100 runs",
    font_sizes: Dict[str, int] = None,
    figsize: Tuple[float, float] = (20, 15),
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    do_save: bool = True,
    exceptions: List[Tuple[str, str]] = [
        ("spline_shape_coef", "N=250"),
        ("spline_scale_coef", "N=250"),
        ("spline_shape_coef", "N=500"),
        ("spline_scale_coef", "N=500"),
    ],
):
    """Plot Wasserstein distances from nested dictionary structure."""
    # Set default exceptions if None
    if exceptions is None:
        exceptions = []

    # Set default font sizes
    default_fonts = {
        "main_title": 16,
        "row_title": 14,
        "violin_title": 13,
        "axis_label": 12,
        "tick": 11,
    }
    font_sizes = {**default_fonts, **(font_sizes or {})}

    # Parameter to LaTeX title mapping
    param_key_to_title = {
        "spline_scale_coef": r"$\overrightarrow{\gamma}_{\sigma}$",
        "spline_shape_coef": r"$\overrightarrow{\gamma}_{\xi}$",
        "lambda_smooth_scale": r"$\lambda^2_{\sigma}$",
        "lambda_smooth_shape": r"$\lambda^2_{\xi}$",
        "intercept_scale": r"$\beta_{0\sigma}$",
        "intercept_shape": r"$\beta_{0\xi}$",
    }

    # Extract and structure data
    param_groups = []
    data_store = {}

    # Get ordered parameter groups
    first_epoch = next(iter(wd_dict.values()))
    first_size = next(iter(first_epoch.values()))
    svi_data = first_size["svi"]["wassersteindistances"]
    param_groups = ["full"] + sorted([k for k in svi_data["marginal"].keys()])

    # Initialize data store
    for size_key in sample_size_order:
        data_store[size_key] = {
            param: {"svi": [], "mcmc": None} for param in param_groups
        }

    # Populate SVI data
    for epoch in svi_epoch_order:
        if epoch not in wd_dict:
            continue
        epoch_data = wd_dict[epoch]
        for size_key in sample_size_order:
            if size_key not in epoch_data:
                continue
            size_data = epoch_data[size_key]
            svi_wd = size_data["svi"]["wassersteindistances"]

            # Process full posterior
            if "full" in svi_wd:
                data_store[size_key]["full"]["svi"].append(
                    jax.device_get(svi_wd["full"])
                )

            # Process marginals
            for param in param_groups[1:]:
                if param in svi_wd["marginal"]:
                    data_store[size_key][param]["svi"].append(
                        jax.device_get(svi_wd["marginal"][param])
                    )

    # Populate MCMC data
    first_epoch_data = wd_dict[svi_epoch_order[0]]
    for size_key in sample_size_order:
        if size_key not in first_epoch_data:
            continue
        mcmc_wd = first_epoch_data[size_key]["mcmc"]["wassersteindistances"]

        if "full" in mcmc_wd:
            data_store[size_key]["full"]["mcmc"] = jax.device_get(mcmc_wd["full"])

        for param in param_groups[1:]:
            if param in mcmc_wd["marginal"]:
                data_store[size_key][param]["mcmc"] = jax.device_get(
                    mcmc_wd["marginal"][param]
                )

    # Create figure and grid
    fig = plt.figure(figsize=figsize)

    # Helper function to check if a param/size is an exception
    def is_exception(param, size_key):
        for ex_param, ex_size in exceptions:
            if param == ex_param and size_key == ex_size:
                return True
        return False

    # MODIFIED: Make all plots broken axis by default, except those in the exceptions list
    broken_axes_plots = {}
    for param in param_groups:
        for size_key in sample_size_order:
            # Skip if this param/size combo is in the exceptions list
            if is_exception(param, size_key):
                continue

            data = data_store[size_key][param]

            # Skip if no data
            if not data["svi"] or data["mcmc"] is None:
                continue

            # Calculate max values
            mcmc_max = np.max(np.array(data["mcmc"]))
            svi_max = max(
                [np.max(np.array(arr)) for arr in data["svi"] if len(arr) > 0]
            )

            broken_axes_plots[(param, size_key)] = (mcmc_max, svi_max)

    # Adjust GridSpec to account for broken axes
    n_rows = len(param_groups)
    n_cols = len(sample_size_order)

    # Calculate how many additional rows we need
    additional_rows = 0
    for param in param_groups:
        has_broken = False
        for size_key in sample_size_order:
            if (param, size_key) in broken_axes_plots:
                has_broken = True
                break
        if has_broken:
            additional_rows += 1

    # Create a mapping from logical positions to grid positions
    row_map = {}
    current_row = 0

    for row_idx, param in enumerate(param_groups):
        row_map[param] = {"start": current_row}

        # Check if this parameter needs a broken axis in any column
        needs_broken = False
        for size_key in sample_size_order:
            if (param, size_key) in broken_axes_plots:
                needs_broken = True
                break

        # Add an extra row if needed
        if needs_broken:
            row_map[param]["is_broken"] = True
            row_map[param]["upper"] = current_row
            current_row += 1
            row_map[param]["lower"] = current_row
            current_row += 1
        else:
            row_map[param]["is_broken"] = False
            row_map[param]["main"] = current_row
            current_row += 1

    # MODIFIED: Increased spacing between rows
    broken_axis_spacing = {
        "between_broken": 0.075,  # Space between broken axis parts (unchanged)
        "between_rows": 0.4,  # Increased from 0.38 to 0.5 for more space between rows
    }

    # Create GridSpec with appropriate height ratios
    height_ratios = []
    for param in param_groups:
        if row_map[param]["is_broken"]:
            # Top part is smaller
            height_ratios.append(0.7)
            # Bottom part
            height_ratios.append(1.0)
        else:
            # Regular row
            height_ratios.append(1.5 if param == "full" else 1.0)

    gs = gridspec.GridSpec(
        len(height_ratios),
        n_cols,
        height_ratios=height_ratios,
        hspace=broken_axis_spacing["between_rows"],
        wspace=0.12,
    )

    # Generate axes
    axes = {}

    for param in param_groups:
        for col_idx, size_key in enumerate(sample_size_order):
            if row_map[param]["is_broken"]:
                if (param, size_key) in broken_axes_plots:
                    # Create broken axes
                    upper_row = row_map[param]["upper"]
                    lower_row = row_map[param]["lower"]

                    axes[(param, size_key, "upper")] = fig.add_subplot(
                        gs[upper_row, col_idx]
                    )
                    axes[(param, size_key, "lower")] = fig.add_subplot(
                        gs[lower_row, col_idx], sharex=axes[(param, size_key, "upper")]
                    )
                else:
                    # For exceptions, use a subplot that spans both rows
                    upper_row = row_map[param]["upper"]
                    lower_row = row_map[param]["lower"]

                    # Use both rows for this plot
                    axes[(param, size_key)] = fig.add_subplot(
                        gs[upper_row : lower_row + 1, col_idx]
                    )
            else:
                # Regular plot
                axes[(param, size_key)] = fig.add_subplot(
                    gs[row_map[param]["main"], col_idx]
                )

    for param in param_groups:
        if row_map[param]["is_broken"]:
            for size_key in sample_size_order:
                # Only process if this parameter/size combo has broken axes
                if (
                    (param, size_key) in broken_axes_plots
                    and (param, size_key, "upper") in axes
                    and (param, size_key, "lower") in axes
                ):

                    upper_ax = axes[(param, size_key, "upper")]
                    lower_ax = axes[(param, size_key, "lower")]

                    # Get current positions
                    upper_pos = upper_ax.get_position()
                    lower_pos = lower_ax.get_position()

                    # Calculate new position with reduced gap
                    new_lower_y = (
                        upper_pos.y0 - broken_axis_spacing["between_broken"] * 0.69
                    )

                    # Set new position for lower axis (maintain other dimensions)
                    lower_ax.set_position(
                        [lower_pos.x0, new_lower_y, lower_pos.width, lower_pos.height]
                    )

    # Generate SVI labels
    svi_labels = []
    for epoch in svi_epoch_order:
        formatted_epoch = (
            f"E={epoch[1:]}" if epoch.startswith("E") and epoch[1:].isdigit() else epoch
        )
        svi_labels.append(f"SVI\n{formatted_epoch}")

    # Plotting logic
    first_row_y_top = None

    for row_idx, param in enumerate(param_groups):
        is_first_row = row_idx == 0

        for col_idx, size_key in enumerate(sample_size_order):
            # Get data for this cell
            data = data_store[size_key][param]

            # Skip empty data
            if not data["svi"] and data["mcmc"] is None:
                continue

            # Prepare data arrays
            svi_arrays = [np.array(arr) for arr in data["svi"]]
            mcmc_array = (
                np.array(data["mcmc"]) if data["mcmc"] is not None else np.array([])
            )
            all_data = svi_arrays + [mcmc_array]

            # Create DataFrame
            plot_data = []
            for i, arr in enumerate(all_data):
                if arr.size == 0:
                    continue
                for val in arr.flatten():
                    plot_data.append(
                        {
                            "value": val,
                            "method": svi_labels[i] if i < len(svi_labels) else "MCMC",
                            "color": (
                                svi_colors[i] if i < len(svi_labels) else mcmc_color
                            ),
                        }
                    )
            df = pd.DataFrame(plot_data)

            if df.empty:
                continue

            # Handle broken axis case
            if (param, size_key) in broken_axes_plots:
                # Get the max values
                mcmc_max = broken_axes_plots[(param, size_key)][0]
                svi_max = broken_axes_plots[(param, size_key)][1]

                # Custom break points based on parameter and sample size
                if param == "lambda_smooth_scale":
                    if size_key == "N=250":
                        break_point = 2.0 * mcmc_max
                    elif size_key == "N=500":
                        break_point = 1.1 * mcmc_max
                    elif size_key == "N=1000":
                        break_point = 0.002 * mcmc_max
                    else:
                        break_point = 2.0 * mcmc_max
                elif param == "lambda_smooth_shape":
                    if size_key == "N=250":
                        break_point = 1.1 * mcmc_max
                    elif size_key == "N=500":
                        break_point = 1.8 * mcmc_max
                    elif size_key == "N=1000":
                        break_point = 0.08 * mcmc_max
                    else:
                        break_point = 0.8 * mcmc_max
                elif param == "full":
                    if size_key == "N=250":
                        break_point = 1.0 * mcmc_max
                    elif size_key == "N=500":
                        break_point = 1.1 * mcmc_max
                    elif size_key == "N=1000":
                        break_point = 0.05 * mcmc_max
                    else:
                        break_point = 0.8 * mcmc_max
                elif param == "intercept_scale":
                    if size_key == "N=250":
                        break_point = 1.1 * mcmc_max
                    elif size_key == "N=500":
                        break_point = 1.8 * mcmc_max
                    elif size_key == "N=1000":
                        break_point = 0.4 * mcmc_max
                    else:
                        break_point = 0.8 * mcmc_max
                elif param == "intercept_shape":
                    if size_key == "N=250":
                        break_point = 1.1 * mcmc_max
                    elif size_key == "N=500":
                        break_point = 1.1 * mcmc_max
                    elif size_key == "N=1000":
                        break_point = 1.0 * mcmc_max
                    else:
                        break_point = 0.8 * mcmc_max
                elif param == "spline_scale_coef":
                    if size_key == "N=250":
                        break_point = 1.1 * mcmc_max
                    elif size_key == "N=500":
                        break_point = 1.8 * mcmc_max
                    elif size_key == "N=1000":
                        break_point = 0.05 * mcmc_max
                    else:
                        break_point = 0.8 * mcmc_max
                elif param == "spline_shape_coef":
                    if size_key == "N=250":
                        break_point = 1.1 * mcmc_max
                    elif size_key == "N=500":
                        break_point = 1.8 * mcmc_max
                    elif size_key == "N=1000":
                        break_point = 0.3 * mcmc_max
                    else:
                        break_point = 0.8 * mcmc_max
                else:
                    # Default for other parameters
                    break_point = 2.0 * mcmc_max

                # Custom break points based on parameter and sample size
                if param == "lambda_smooth_scale":
                    if size_key == "N=250":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=500":
                        beginning_point = svi_max * 0.85
                    elif size_key == "N=1000":
                        beginning_point = svi_max * 0.8
                    else:
                        beginning_point = svi_max * 0.8
                elif param == "lambda_smooth_shape":
                    if size_key == "N=250":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=500":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=1000":
                        beginning_point = svi_max * 0.8
                    else:
                        beginning_point = svi_max * 0.8
                elif param == "full":
                    if size_key == "N=250":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=500":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=1000":
                        beginning_point = svi_max * 0.8
                    else:
                        beginning_point = svi_max * 0.8
                elif param == "intercept_scale":
                    if size_key == "N=250":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=500":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=1000":
                        beginning_point = svi_max * 0.8
                    else:
                        beginning_point = svi_max * 0.8
                elif param == "intercept_shape":
                    if size_key == "N=250":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=500":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=1000":
                        beginning_point = svi_max * 0.8
                    else:
                        beginning_point = svi_max * 0.8
                elif param == "spline_scale_coef":
                    if size_key == "N=250":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=500":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=1000":
                        beginning_point = svi_max * 0.8
                    else:
                        beginning_point = svi_max * 0.8
                elif param == "spline_shape_coef":
                    if size_key == "N=250":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=500":
                        beginning_point = svi_max * 0.8
                    elif size_key == "N=1000":
                        beginning_point = svi_max * 0.8
                    else:
                        beginning_point = svi_max * 0.8
                else:
                    # Default for other parameters
                    beginning_point = svi_max * 0.8

                # Create upper plot
                ax_top = axes[(param, size_key, "upper")]
                sns.violinplot(
                    x="method",
                    y="value",
                    data=df,
                    palette=svi_colors + [mcmc_color],
                    inner="quartile",
                    cut=0,
                    ax=ax_top,
                    order=svi_labels + ["MCMC"],
                    saturation=0.75,
                    linewidth=1.5,  # Slightly wider quartile lines
                )
                ax_top.set_ylabel("")

                # Create lower plot
                ax_bottom = axes[(param, size_key, "lower")]
                sns.violinplot(
                    x="method",
                    y="value",
                    data=df,
                    palette=svi_colors + [mcmc_color],
                    inner="quartile",
                    cut=0,
                    ax=ax_bottom,
                    order=svi_labels + ["MCMC"],
                    saturation=0.75,
                    linewidth=1.5,  # Slightly wider quartile lines
                )
                ax_bottom.set_ylabel("")

                # Set limits for broken axis
                ax_top.set_ylim(bottom=beginning_point, top=svi_max * 1.1)
                ax_bottom.set_ylim(0, break_point * 1.0)

                # Add diagonal break lines
                d = 0.015  # Size of diagonal lines
                kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
                ax_top.plot((-d, +d), (-d, +d), **kwargs)
                ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

                kwargs.update(transform=ax_bottom.transAxes)
                ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

                # Style adjustments - keep top and right spines for the lambda plots
                # Only remove bottom spine for top plot
                for spine in ["left", "top", "right"]:
                    ax_top.spines[spine].set_visible(True)
                ax_top.spines["bottom"].set_visible(False)

                # Keep all spines for bottom plot
                for spine in ["left", "bottom", "right"]:
                    ax_bottom.spines[spine].set_visible(True)
                ax_bottom.spines["top"].set_visible(False)

                ax_top.set_xlabel("")
                ax_top.set_ylabel("")
                ax_bottom.set_xlabel("")

                # Hide x ticks on top plot
                ax_top.xaxis.set_visible(False)

                if is_first_row:
                    # For first row, show labels on TOP part of broken axis
                    ax_top.xaxis.set_visible(True)
                    ax_top.xaxis.set_ticks_position("top")
                    ax_top.xaxis.set_label_position("top")
                    
                    # Special handling for the middle column (N=500)
                    if size_key == "N=500":
                        ax_top.tick_params(
                            axis="x",
                            labelrotation=45,
                            labelsize=font_sizes["tick"],
                            top=True,
                            bottom=False,
                            pad=25,  # Extra padding for N=500
                        )
                    else:
                        ax_top.tick_params(
                            axis="x",
                            labelrotation=45,
                            labelsize=font_sizes["tick"],
                            top=True,
                            bottom=False,
                        )
                    
                    # Hide labels on bottom part
                    ax_bottom.xaxis.set_visible(False)
                else:
                    # For other rows, hide both top and bottom x-axis labels
                    ax_top.xaxis.set_visible(False)
                    ax_bottom.xaxis.set_visible(False)

                # Add y-label on the bottom plot only
                if col_idx == 0:
                    # Change: Set "SD" for specific parameters instead of "WD"
                    if param in ["full", "spline_scale_coef", "spline_shape_coef"]:
                        ax_bottom.set_ylabel("SD", fontsize=font_sizes["axis_label"])
                    else:
                        ax_bottom.set_ylabel("WD", fontsize=font_sizes["axis_label"])

                # Remove one of the legends
                if hasattr(ax_top, "legend_") and ax_top.legend_ is not None:
                    ax_top.legend_.remove()

            else:
                # Non-broken plot (either a regular row or an exception in a broken row)
                if row_map[param]["is_broken"]:
                    # This is an exception plot in a broken row
                    ax = axes[(param, size_key)]

                    # Regular violin plot for the full-height exception
                    sns.violinplot(
                        x="method",
                        y="value",
                        data=df,
                        palette=svi_colors + [mcmc_color],
                        inner="quartile",
                        cut=0,
                        ax=ax,
                        order=svi_labels + ["MCMC"],
                        saturation=0.75,
                        linewidth=1.5,
                    )

                    # Style adjustments for consistency
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                    # Axis label positioning
                    if is_first_row:
                        ax.xaxis.set_ticks_position("top")
                        ax.xaxis.set_label_position("top")
                        ax.tick_params(
                            axis="x",
                            labelrotation=45,
                            labelsize=font_sizes["tick"],
                            top=True,
                            bottom=False,
                        )
                    else:
                        ax.xaxis.set_visible(False)

                    if col_idx == 0:
                        if param in ["full", "spline_scale_coef", "spline_shape_coef"]:
                            ax.set_ylabel("SD", fontsize=font_sizes["axis_label"])
                        else:
                            ax.set_ylabel("WD", fontsize=font_sizes["axis_label"])
                else:
                    # Regular plot in a non-broken row
                    ax = axes[(param, size_key)]

                    sns.violinplot(
                        x="method",
                        y="value",
                        data=df,
                        palette=svi_colors + [mcmc_color],
                        inner="quartile",
                        cut=0,
                        ax=ax,
                        order=svi_labels + ["MCMC"],
                        saturation=0.75,
                        linewidth=1.5,
                    )

                    # Style adjustments
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                    # Axis label positioning
                    if is_first_row:
                        ax.xaxis.set_ticks_position("top")
                        ax.xaxis.set_label_position("top")
                        if size_key == "N=500":
                            ax.tick_params(
                                axis="x",
                                labelrotation=45,
                                labelsize=font_sizes["tick"],
                                top=True,
                                bottom=False,
                                pad=25,
                            )
                        else:
                            ax.tick_params(
                                axis="x",
                                labelrotation=45,
                                labelsize=font_sizes["tick"],
                                top=True,
                                bottom=False,
                            )
                    else:
                        ax.xaxis.set_visible(False)

                    if col_idx == 0:
                        # Change: Set "SD" for specific parameters instead of "WD"
                        if param in ["full", "spline_scale_coef", "spline_shape_coef"]:
                            ax.set_ylabel("SD", fontsize=font_sizes["axis_label"])
                        else:
                            ax.set_ylabel("WD", fontsize=font_sizes["axis_label"])

        # MODIFIED: Fixed row titles logic to work with exceptions
        # Add row titles
        regular_title_offset = 0.002
        joint_post_title_offset = 0.007
        last_row_title_offset = -0.004  # Increased offset for the last row
        is_last_row = row_idx == len(param_groups) - 1

        # Choose appropriate offset
        if param == "full":
            title_offset = joint_post_title_offset
        elif is_last_row:
            title_offset = last_row_title_offset  # Special larger offset for last row
        else:
            title_offset = regular_title_offset
        # Find leftmost and rightmost axes for title placement - consider ALL column axes
        first_ax = None
        last_ax = None

        # Get all axes for this parameter row, in column order
        all_row_axes = []
        for size_key in sample_size_order:
            ax_key = None
            # Check for different possible axis keys in this order
            if (param, size_key) in axes:  # Full-height exception axis
                ax_key = (param, size_key)
            elif (param, size_key, "upper") in axes:  # Upper part of broken axis
                ax_key = (param, size_key, "upper")
                
            if ax_key is not None:
                all_row_axes.append(axes[ax_key])

        # Use first and last axis to determine title position if axes were found
        if all_row_axes:
            first_ax = all_row_axes[0]
            last_ax = all_row_axes[-1]

        # If we found axes, add the title
        if first_ax is not None and last_ax is not None:
            center_pos = (first_ax.get_position().x0 + last_ax.get_position().x1) / 2
            row_y_top = first_ax.get_position().y1

            # Modified title generation
            if param == "full":
                title_text = "Joint Posterior"
            else:
                formatted_param = param_key_to_title.get(param, param)
                title_text = f"Marginal Posterior: {formatted_param}"

            fig.text(
                center_pos,
                row_y_top + title_offset,
                title_text,
                rotation=0,
                va="bottom",
                ha="center",
                fontsize=font_sizes["row_title"],
            )

            if is_first_row:
                first_row_y_top = row_y_top

    # Add column headers (N=xxx)
    if first_row_y_top is not None:
        for col_idx, size_key in enumerate(sample_size_order):
            if size_key == "N=500":
                column_header_y = first_row_y_top + 0.075
            else:
                column_header_y = first_row_y_top + 0.055

            for param in param_groups:
                ax_key = None
                if param == "full":
                    if (param, size_key) in axes:
                        ax_key = (param, size_key)
                    elif (param, size_key, "upper") in axes:
                        ax_key = (param, size_key, "upper")
                elif row_map[param]["is_broken"]:
                    if (param, size_key, "upper") in axes:
                        ax_key = (param, size_key, "upper")
                    elif (param, size_key) in axes:
                        ax_key = (param, size_key)
                else:
                    if (param, size_key) in axes:
                        ax_key = (param, size_key)

                if ax_key in axes:
                    ax = axes[ax_key]
                    x_center = (ax.get_position().x0 + ax.get_position().x1) / 2
                    fig.text(
                        x_center,
                        column_header_y,
                        size_key,
                        fontsize=font_sizes["violin_title"],
                        ha="center",
                        va="bottom",
                    )
                    break

    # Save and show
    if file_name and save_dir and do_save:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{file_name.split('.')[0]}.svg")
        plt.savefig(path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {path}")

    plt.show()
    plt.close()


def plot_elbo(
    num_iterations: int,
    title: str,
    elbo_values: List[float],
    elbo_color: str,
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    apply_ma: bool = False,
    window: Optional[int] = None,
    tick_label_plain: bool = False,
    do_save: bool = True,
):
    """
    ELBO plot visualization
    """
    sns.set_theme(style="whitegrid")
    # Convert and validate data
    elbo_values = np.asarray(elbo_values)
    if elbo_values.size == 0:
        raise ValueError("Input arrays must not be empty.")

    # Calculate threshold for moving average
    threshold = elbo_values[0]
    clipped_elbo = np.minimum(elbo_values, threshold)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the ELBO
    sns.lineplot(
        x=range(num_iterations),
        y=elbo_values,
        color=elbo_color,
        ax=ax,
        linewidth=1.5,
        label="ELBO",
    )

    # Add moving average if requested (original ELBO)
    if apply_ma:
        ma = pd.Series(clipped_elbo).rolling(window=window).mean()
        sns.lineplot(
            x=range(num_iterations),
            y=ma,
            color=elbo_color,
            linestyle="--",
            ax=ax,
            label="Moving Average",
        )

    # Configure axes
    # ax.set_title(title, fontsize=14)
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("ELBO", fontsize=13)
    ax.set_xlim(0, num_iterations - 1)
    if tick_label_plain:
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    # Remove legend when not needed
    if not apply_ma:
        ax.get_legend().remove() if ax.get_legend() else None

    # Save and show
    if file_name and save_dir and do_save:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{file_name.split('.')[0]}.svg")
        plt.savefig(path, bbox_inches="tight", format="svg")
        print(f"Plot saved to {path}")
    plt.show()


def plot_oos_counts(
    processed_data: Dict,
    colors: List[str],
    N: int,
    peak_color: str,
    vi_label: str = "OOS count over S VI samples",
    update_label: str = "OOS count with current optimal parameters",
    num_epochs: int = 200,
    figsize: Tuple[int, int] = (12, 6),
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    do_save: bool = False,
    x_label: str = "Epoch",
    y_label: str = "symlog Out-of-Support Count with 0.1 threshold",
    title: str = "Out-of-Support Responses Over Epochs",
    legend_loc: str = "best",
    dpi: int = 300,
):
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    ax.set_yscale("symlog", linthresh=0.1)
    epochs = np.arange(1, num_epochs + 1)
    oos_vi = processed_data.get("oos_vi_counts", None)
    oos_update = processed_data.get("oos_update_counts", None)
    if oos_vi is not None:
        oos_vi = oos_vi[:num_epochs]
        ax.bar(epochs, oos_vi, color=colors[0], label=vi_label, edgecolor=colors[0])
    if oos_update is not None:
        oos_update = oos_update[:num_epochs]
        ax.plot(
            epochs,
            oos_update,
            color=colors[1],
            label=update_label,
            linewidth=6.0,
        )
        peaks = np.asarray(processed_data["elbo_peaks"][:num_epochs])
        # if peaks.any():
        #     peak_x = np.where(peaks)[0] + 1
        #     peak_y = oos_vi[peak_x - 1]
        #     ax.scatter(
        #         peak_x,
        #         peak_y,
        #         color=peak_color,
        #         marker="X",
        #         s=80,
        #         alpha=0.25,
        #         label="ELBO peaks",
        #         zorder=10,
        #         edgecolors="black",
        #     )
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title + f": N={N}", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    if oos_vi is not None or oos_update is not None:
        ax.legend(loc=legend_loc, framealpha=0.9, fontsize=11)
    if file_name and save_dir:
        if do_save:
            file_name = (
                f"{file_name}.svg" if not file_name.endswith(".svg") else file_name
            )
            full_path = os.path.join(save_dir, file_name)
            plt.savefig(full_path, format="svg", bbox_inches="tight")
            print(f"Plot saved to {full_path}")
        else:
            warnings.warn("Plot not saved. Set do_save=True to save.", UserWarning)
    elif file_name or save_dir:
        raise ValueError("Both file_name and save_dir required for saving")
    plt.tight_layout()
    plt.show()


def plot_elbo_analysis(
    processed_data: Dict,
    colors: List[str],
    N: int,
    num_epochs: int = 200,
    bar_alpha: float = 1.0,
    figsize: Tuple[int, int] = (12, 6),
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    do_save: bool = False,
    x_label: str = "Epoch",
    y_label: str = "Component Contribution to ELBO",
    title: str = "Negative ELBO Component Break Down",
    legend_loc: str = "best",
    dpi: int = 300,
):
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    epochs = np.arange(0, num_epochs)
    log_comps = processed_data["log_components"]

    # Define all components and their labels (including OOS)
    components = ["logq", "prior", "response_pos", "response_neg", "response_oos"]
    labels = [
        "VI distribution neg. log pdf",
        "Prior neg. log pdf",
        "GPD shape neg. log pdf",
        "GPD shape neg. log pdf",
        "GPD OOS neg. log pdf",
    ]

    # Create dictionaries to store positive and negative parts
    pos_data_by_comp = {}  # Maps component name to its positive data
    neg_data_by_comp = {}  # Maps component name to its negative data (absolute values)

    # First pass: Extract positive and negative parts for each component
    valid_components = []  # Components that exist in log_comps
    for comp in components:
        if comp in log_comps:
            valid_components.append(comp)
            data = log_comps[comp][:num_epochs]

            # Extract positive part (negative values set to 0)
            pos_data = np.maximum(data, 0)
            if np.any(pos_data > 0):
                pos_data_by_comp[comp] = pos_data

            # Extract negative part (absolute value of negative parts)
            neg_data = np.abs(np.minimum(data, 0))
            if np.any(neg_data > 0):
                neg_data_by_comp[comp] = neg_data

    # Create arrays for stacking, maintaining component order
    pos_arrays = []
    neg_arrays = []
    pos_colors = []
    neg_colors = []
    pos_comp_indices = {}  # Maps component to its index in pos_arrays
    neg_comp_indices = {}  # Maps component to its index in neg_arrays

    # Build stacking arrays in component order
    for i, comp in enumerate(valid_components):
        comp_idx = components.index(comp)
        comp_color = colors[comp_idx % len(colors)]

        # Add positive part if it exists
        if comp in pos_data_by_comp:
            pos_comp_indices[comp] = len(pos_arrays)
            pos_arrays.append(pos_data_by_comp[comp])
            pos_colors.append(comp_color)

        # Add negative part if it exists
        if comp in neg_data_by_comp:
            neg_comp_indices[comp] = len(neg_arrays)
            neg_arrays.append(neg_data_by_comp[comp])
            neg_colors.append(comp_color)

    # Plot positive stack
    if pos_arrays:
        stack_pos = ax.stackplot(
            epochs,
            pos_arrays,
            colors=pos_colors,
            alpha=bar_alpha,
            labels=["nolegend"] * len(pos_arrays),
        )

    # Plot negative stack
    if neg_arrays:
        # Invert negative values for plotting below zero
        neg_arrays_inv = [-arr for arr in neg_arrays]
        stack_neg = ax.stackplot(
            epochs,
            neg_arrays_inv,
            colors=neg_colors,
            alpha=bar_alpha,
            labels=["nolegend"] * len(neg_arrays),
        )

    # Add outlines for all components in the original order
    for i, comp in enumerate(valid_components):
        comp_idx = components.index(comp)
        comp_color = colors[comp_idx % len(colors)]

        # Add outline for positive part if it exists
        if comp in pos_comp_indices:
            pos_index = pos_comp_indices[comp]
            comp_pos_data = pos_arrays[pos_index]

            # Calculate the position in the stack
            pos_base = np.zeros_like(pos_arrays[0])
            for j in range(pos_index):
                pos_base += pos_arrays[j]

            # Special handling for response_oos
            if comp == "response_oos":
                y_pos = pos_base + comp_pos_data
                ax.plot(
                    epochs,
                    y_pos,
                    color=comp_color,
                    linewidth=2,
                    label=labels[comp_idx],
                    zorder=10,
                )
            else:
                # Mask where component's positive data is zero
                y_pos = pos_base + comp_pos_data
                mask = comp_pos_data == 0
                y_pos_masked = np.where(mask, np.nan, y_pos)
                ax.plot(
                    epochs,
                    y_pos_masked,
                    color=comp_color,
                    linewidth=2,
                    label=labels[comp_idx],
                    zorder=10,
                )

        # Add outline for negative part
        if comp in neg_comp_indices:
            neg_index = neg_comp_indices[comp]
            comp_neg_data = neg_arrays[neg_index]

            # Calculate the position in the stack
            neg_base = np.zeros_like(neg_arrays[0])
            for j in range(neg_index):
                neg_base += neg_arrays[j]

            # Only add a label if we didn't already add one for positive values
            label = None if comp in pos_comp_indices else labels[comp_idx]

            # Special handling for response_oos
            if comp == "response_oos":
                y_neg = -(neg_base + comp_neg_data)
                ax.plot(
                    epochs, y_neg, color=comp_color, linewidth=2, label=label, zorder=10
                )
            else:
                # Mask where component's negative data is zero
                y_neg = -(neg_base + comp_neg_data)
                mask = comp_neg_data == 0
                y_neg_masked = np.where(mask, np.nan, y_neg)
                ax.plot(
                    epochs,
                    y_neg_masked,
                    color=comp_color,
                    linewidth=2,
                    label=label,
                    zorder=10,
                )

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title + f": N={N}", fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    # Legend handling
    handles, labels = ax.get_legend_handles_labels()

    # Remove any "nolegend" entries
    legend_entries = [(h, l) for h, l in zip(handles, labels) if l != "nolegend"]
    if legend_entries:
        ax.legend(
            [h for h, l in legend_entries],
            [l for h, l in legend_entries],
            loc=legend_loc,
            framealpha=0.9,
            fontsize=11,
            ncol=2,
        )

    # Saving logic
    if file_name and save_dir:
        if do_save:
            file_name = (
                f"{file_name}.svg" if not file_name.endswith(".svg") else file_name
            )
            full_path = os.path.join(save_dir, file_name)
            plt.savefig(full_path, format="svg", bbox_inches="tight")
            print(f"Plot saved to {full_path}")
        else:
            warnings.warn(
                "ELBO Component Break Down plot not saved. Set do_save=True to save.",
                UserWarning,
            )
    elif file_name or save_dir:
        raise ValueError("Both file_name and save_dir required for saving")

    plt.tight_layout()
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
    posterior_param_hdis_lower: Optional[List[jnp.ndarray]] = None,
    posterior_param_hdis_upper: Optional[List[jnp.ndarray]] = None,
    posterior_param_hdi_alphas: Optional[List[float]] = None,
    posterior_param_hdi_labels: Optional[List[str]] = None,
    posterior_param_hdi_colors: Optional[List[str]] = None,
    posterior_param_x_label: Optional[str] = None,
    posterior_param_y_label: Optional[str] = None,
    posterior_param_line_labels: Optional[List[str]] = None,
    posterior_param_titles: Optional[List[str]] = None,
    posterior_fig_size: tuple = (12, 4),
    save_dir: Optional[str] = None,
    y_origin: Optional[float] = None,
    do_save: bool = False,
    y_break_lower1: Optional[float] = None,
    y_break_upper1: Optional[float] = None,
    y_break_lower2: Optional[float] = None,
    y_break_upper2: Optional[float] = None,
    y_lim_upper: Optional[float] = None,
    break_height1: float = 0.05,
    break_height2: float = 0.1,
    break_height3: float = 1.0,
    hspace_subplot: float = 0.01,
    y_log_scale: bool = False,
    custom_y_ticks: Optional[np.ndarray] = None,
):
    """Plots regression results with optional 1 or 2 y-axis breaks."""
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

    # Determine number of breaks based on provided parameters
    num_breaks = 0
    if all([y_break_lower1 is not None, y_break_upper1 is not None]):
        num_breaks = 1
        if all([y_break_lower2 is not None, y_break_upper2 is not None]):
            num_breaks = 2

    # Calculate max data y if breaks are requested
    max_data_y = None
    if num_breaks > 0:
        max_data = []
        if scatter_y is not None:
            max_data.append(jnp.max(scatter_y).item())
        for line in regression_lines:
            if line is not None:
                max_data.append(jnp.max(line).item())
        for upper in hdi_upper_bounds:
            if upper is not None:
                max_data.append(jnp.max(upper).item())
        max_data_y = max(max_data) if max_data else 0

    # Initialize main_axes here
    main_axes = []

    # Create figure layout with potential breaks
    if pred_params_provided:
        num_pred = len(posterior_param_lines) if posterior_param_lines else 0
        main_height, sub_height = fig_size[1], posterior_fig_size[1]

        # Determine height ratios based on number of breaks
        if num_breaks == 0:
            main_height_ratios = [1.0]
        elif num_breaks == 1:
            main_height_ratios = [break_height1, break_height2]
        else:  # num_breaks == 2
            main_height_ratios = [break_height1, break_height2, break_height3]

        # Calculate total height considering the break_height parameters
        total_height = main_height * sum(main_height_ratios) + num_pred * sub_height
        total_width = max(fig_size[0], posterior_fig_size[0])

        fig = plt.figure(figsize=(total_width, total_height))

        # Use main_height_ratios for the main axes and [sub_height/main_height] for each pred axis
        height_ratios = main_height_ratios + [sub_height / main_height] * num_pred

        gs = gridspec.GridSpec(
            num_breaks + 1 + num_pred,
            1,
            height_ratios=height_ratios,
        )

        # Create main axes based on number of breaks
        main_axes = []
        for i in range(num_breaks + 1):
            main_axes.append(
                fig.add_subplot(gs[i], sharex=main_axes[0] if i > 0 else None)
            )
        pred_axes = [fig.add_subplot(gs[i + num_breaks + 1]) for i in range(num_pred)]
    else:
        if num_breaks > 0:
            # Determine height ratios based on number of breaks
            if num_breaks == 1:
                height_ratios = [break_height1, break_height2]
            else:  # num_breaks == 2
                height_ratios = [break_height1, break_height2, break_height3]

            # Use GridSpec for control over subplot heights
            fig = plt.figure(figsize=(fig_size[0], fig_size[1] * sum(height_ratios)))
            gs = gridspec.GridSpec(num_breaks + 1, 1, height_ratios=height_ratios)

            # Create axes from GridSpec
            main_axes = [fig.add_subplot(gs[i]) for i in range(num_breaks + 1)]

            # Make x-axis shared among all subplots
            for i in range(1, num_breaks + 1):
                main_axes[i].sharex(main_axes[0])

            plt.subplots_adjust(hspace=hspace_subplot)
        else:
            fig, ax_main = plt.subplots(figsize=fig_size)
            main_axes = [ax_main]

    # Collect all plotted elements for a single legend
    legend_elements = []
    legend_labels = []

    # Plot data on all relevant axes
    for i, ax in enumerate(main_axes):
        # Apply log scale to y-axis if requested
        if y_log_scale:
            ax.set_yscale("log")
            if custom_y_ticks is not None:
                ax.set_yticks(custom_y_ticks)
                ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda y, pos: f"log({y})")
                )
            # ax.tick_params(axis='y', which='major', labelsize=11)

        # Determine if this is the axis where we want to show the legend
        # (Only collect labels on the first axis for consolidated legend later)
        is_first_axis = i == 0

        # Scatter plots
        if scatter_x is not None and scatter_y is not None:
            facecolors = "None" if marker == "o" else scatter_color
            scatter = sns.scatterplot(
                x=scatter_x,
                y=scatter_y,
                marker=marker,
                facecolors=facecolors,
                edgecolor=scatter_color,
                label=(
                    scatter_label if is_first_axis else None
                ),  # Only add label on first axis
                alpha=0.65,
                ax=ax,
                s=scatter_size,
            )

            # For the first axis, collect legend elements
            if is_first_axis and scatter_label:
                handles, labels = ax.get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    if l == scatter_label and l not in legend_labels:
                        legend_elements.append(h)
                        legend_labels.append(l)

            # Remove the automatically created legend
            if ax.get_legend():
                ax.get_legend().remove()

            if excess_mask is not None:
                excess_color_ = excess_color or "#F44336"
                excess = sns.scatterplot(
                    x=scatter_x[excess_mask],
                    y=scatter_y[excess_mask],
                    marker=marker,
                    color=excess_color_,
                    facecolors=excess_color_,
                    edgecolor=excess_color_,
                    label=(
                        "Excess" if is_first_axis else None
                    ),  # Only add label on first axis
                    alpha=0.7,
                    ax=ax,
                    s=scatter_size,
                )

                # For the first axis, collect legend elements
                if is_first_axis:
                    handles, labels = ax.get_legend_handles_labels()
                    for h, l in zip(handles, labels):
                        if l == "Excess" and l not in legend_labels:
                            legend_elements.append(h)
                            legend_labels.append(l)

                # Remove the automatically created legend
                if ax.get_legend():
                    ax.get_legend().remove()

        # Regression lines
        for line_x, line, color, style, label, alpha in zip(
            line_xs,
            regression_lines,
            regression_line_colors,
            regression_line_styles,
            regression_line_labels,
            line_alphas,
        ):
            line_plot = sns.lineplot(
                x=line_x,
                y=line,
                color=color,
                linestyle=style,
                label=label if is_first_axis else None,  # Only add label on first axis
                linewidth=2.5,
                ax=ax,
                alpha=alpha,
            )

            # For the first axis, collect legend elements
            if is_first_axis and label:
                handles, labels = ax.get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    if l == label and l not in legend_labels:
                        legend_elements.append(h)
                        legend_labels.append(l)

            # Remove the automatically created legend
            if ax.get_legend():
                ax.get_legend().remove()

        # HDIs
        for line_x, lower, upper, alpha, color, label in zip(
            line_xs_hdi,
            hdi_lower_bounds,
            hdi_upper_bounds,
            hdi_alphas,
            hdi_colors,
            hdi_labels,
        ):
            # Only use the label on the first axis
            ax.fill_between(
                line_x,
                lower,
                upper,
                color=color,
                alpha=alpha,
                label=label if is_first_axis else None,
            )

            # For the first axis, collect legend elements
            if is_first_axis and label:
                handles, labels = ax.get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    if l == label and l not in legend_labels:
                        legend_elements.append(h)
                        legend_labels.append(l)

            # Lines for HDI bounds (no labels)
            sns.lineplot(x=line_x, y=lower, color=color, alpha=alpha, ax=ax, label=None)
            sns.lineplot(x=line_x, y=upper, color=color, alpha=alpha, ax=ax, label=None)

            # Remove the automatically created legend
            if ax.get_legend():
                ax.get_legend().remove()

    # Configure axis limits and breaks
    if num_breaks > 0:
        # Set y-limits for each section
        if num_breaks == 1:
            main_axes[0].set_ylim(y_break_upper1, max_data_y + y_lim_upper)
            main_axes[1].set_ylim(
                y_origin if y_origin is not None else 0, y_break_lower1
            )
        elif num_breaks == 2:
            main_axes[0].set_ylim(y_break_upper2, max_data_y + y_lim_upper)
            main_axes[1].set_ylim(y_break_upper1, y_break_lower2)
            main_axes[2].set_ylim(
                y_origin if y_origin is not None else 0, y_break_lower1
            )

        # Configure axis spines and break markers
        d = 0.01  # Break marker size
        for i in range(len(main_axes) - 1):
            # Hide spines between sections
            main_axes[i].spines["bottom"].set_visible(False)
            main_axes[i + 1].spines["top"].set_visible(False)
            main_axes[i].xaxis.set_visible(False)

            # Add diagonal break markers
            kwargs = dict(transform=main_axes[i].transAxes, color="k", clip_on=False)
            main_axes[i].plot((-d, +d), (-d, +d), **kwargs)  # Top-left diagonal
            main_axes[i].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

            kwargs = dict(
                transform=main_axes[i + 1].transAxes, color="k", clip_on=False
            )
            main_axes[i + 1].plot(
                (-d, +d), (1 - d, 1 + d), **kwargs
            )  # Bottom-left diagonal
            main_axes[i + 1].plot(
                (1 - d, 1 + d), (1 - d, 1 + d), **kwargs
            )  # Bottom-right diagonal

        # Set labels on bottommost axis
        main_axes[-1].set_xlabel(x_label, fontsize=13)
        main_axes[-1].set_ylabel(y_label, fontsize=13)
        # main_axes[0].set_title(title, fontsize=14)

        # Use the consolidated legend elements
        main_axes[-1].legend(legend_elements, legend_labels, fontsize=12, loc="best")
    else:
        # Original non-broken axis handling
        if y_origin is not None:
            main_axes[0].set_ylim(bottom=y_origin)
        main_axes[0].set_xlabel(x_label, fontsize=13)
        main_axes[0].set_ylabel(y_label, fontsize=13)
        # main_axes[0].set_title(title, fontsize=14)

        # Use the consolidated legend elements for non-broken axis too
        if legend_elements and legend_labels:
            main_axes[0].legend(legend_elements, legend_labels, fontsize=12, loc="best")
        else:
            # Fallback to getting legend from the axis if no consolidated elements
            main_axes[0].legend(fontsize=12, loc="best")

    # Plot predictive parameter subplots
    if pred_params_provided and posterior_param_lines:
        for (
            ax,
            line,
            color,
            label,
            sub_title,
            lower,
            upper,
            alpha,
            hdi_label,
            hdi_color,
        ) in zip(
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
                posterior_param_x,
                lower,
                upper,
                color=hdi_color,
                alpha=alpha,
                label=hdi_label,
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


def plot_elbo_behavior_checks(
    normalized_logpdfs: jnp.ndarray,
    support_checks_proportions: jnp.ndarray,
    elbo_values: jnp.ndarray,
    support_color: str = "#FFDB58",
    logpdf_color: str = "#2CA02C",
    peak_color: str = "#800020",
    x_label: str = "Epoch",
    y1_label: str = "Proportion",
    y2_label: str = "Normalized mean GPD log pdfs",
    title: str = None,
    do_save: bool = False,
    file_name: str = None,
    save_dir: str = None,
    N: int = None,
    label_size: int = 13,
    title_size: int = 15,
    figsize: tuple = (10, 6),
):
    """Plot ELBO behavior checks with dual y-axes and peak markers"""
    # Convert JAX arrays to numpy
    normalized_logpdfs = np.asarray(normalized_logpdfs)
    support_checks = np.asarray(support_checks_proportions)
    elbo_values = np.asarray(elbo_values)

    # Validate inputs
    if not (len(normalized_logpdfs) == len(support_checks) == len(elbo_values)):
        raise ValueError("All input arrays must have the same length")

    if elbo_values.size == 0:
        raise ValueError("elbo_values must not be empty")

    # Process ELBO values
    threshold = elbo_values[0]
    is_peak = elbo_values > threshold
    peak_x = np.where(is_peak)[0] + 1  # 1-based indexing

    # Create plot
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    x_values = np.arange(1, len(normalized_logpdfs) + 1)

    # Plot support checks proportions
    ax1.plot(
        x_values,
        support_checks,
        color=support_color,
        linewidth=2,
        label="Support Validity Proportion",
    )
    ax1.set_xlabel(x_label, size=label_size)
    ax1.set_ylabel(y1_label, color=support_color, size=label_size)
    ax1.tick_params(axis="y", labelcolor=support_color)

    # Plot normalized logpdf means
    ax2.plot(
        x_values,
        normalized_logpdfs,
        color=logpdf_color,
        linewidth=2,
        label="Mean log pdf",
    )
    ax2.set_ylabel(y2_label, color=logpdf_color, size=label_size)
    ax2.tick_params(axis="y", labelcolor=logpdf_color)

    # Plot ELBO peaks
    if peak_x.size > 0:
        ax2.scatter(
            peak_x,
            np.ones_like(peak_x),
            color=peak_color,
            marker="x",
            s=100,
            linewidths=2,
            label="Peaking ELBO values",
            zorder=5,
        )

    # Configure title
    title = title or f"ELBO Behavior Checks: N={N}" if N else "ELBO Behavior Checks"
    plt.title(title, fontsize=title_size, pad=20)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, loc="bottom right", fontsize=label_size
    )

    # Handle saving
    if file_name and save_dir:
        if do_save:
            file_name = (
                f"{file_name}.svg" if not file_name.endswith(".svg") else file_name
            )
            full_path = os.path.join(save_dir, file_name)
            plt.savefig(full_path, format="svg", bbox_inches="tight")
            print(f"Plot saved to {full_path}")
        else:
            warnings.warn(
                "ELBO behavior plot not saved. Set do_save=True to save.", UserWarning
            )
    elif file_name or save_dir:
        raise ValueError("Both file_name and save_dir required for saving")

    plt.tight_layout()
    plt.show()


def plot_gpd_qq_plot(
    sigma_hat,
    xi_hat,
    Y_excesses,
    qq_line_color="#607D8B",  # Default slate gray color
    quantiles_color="#1A237E",  # Default indigo color
    file_name=None,
    save_dir=None,
    do_save=True,
    custom_y_ticks=None,
    use_log_scale=False,  # Changed default to False as we're already transforming
):
    # Convert inputs to numpy arrays if they aren't already
    Y_excesses = np.array(Y_excesses)
    sigma_hat = np.array(sigma_hat)
    xi_hat = np.array(xi_hat)

    # Sort the excesses
    sorted_indices = np.argsort(Y_excesses)
    sorted_excesses = Y_excesses[sorted_indices]

    # If parameters are arrays with same length as excesses, sort them too
    if len(sigma_hat) > 1:
        sigma_hat = sigma_hat[sorted_indices]
    if len(xi_hat) > 1:
        xi_hat = xi_hat[sorted_indices]

    # Calculate plotting positions (probability points)
    k = len(sorted_excesses)
    u = np.arange(1, k + 1) / (k + 1)  # Probability points

    # Transformation to exponential scale (theoretical quantiles)
    # This is -log(1-u[i]) as per the first source
    theoretical_exponential_quantiles = -np.log(1 - u)

    # Transform empirical excesses to exponential scale
    # Using the transformation: Ỹ = (1/ξ) × log{1 + ξ((Y - u)/σ)}
    transformed_excesses = np.zeros_like(sorted_excesses)

    epsilon = 1e-8
    for i in range(k):
        # Get parameters for this point
        xi = xi_hat[i] if len(xi_hat) > 1 else xi_hat.item()
        sigma = sigma_hat[i] if len(sigma_hat) > 1 else sigma_hat.item()

        # Apply transformation based on xi value
        if abs(xi) < epsilon:
            # For xi ≈ 0, the transformation is Y/sigma (limiting case)
            transformed_excesses[i] = sorted_excesses[i] / sigma
        else:
            # Standard transformation formula from source 1
            transformed_excesses[i] = (1.0 / xi) * np.log(
                1.0 + xi * (sorted_excesses[i] / sigma)
            )

    # Set Seaborn theme
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Create the scatter plot (now plotting transformed empirical vs theoretical exponential)
    sns.scatterplot(
        x=theoretical_exponential_quantiles,
        y=transformed_excesses,
        color=quantiles_color,
        label="Transformed Quantile Pairs",
        edgecolor=quantiles_color,
        alpha=0.8,
    )

    # Find min and max for reference line
    min_val = min(theoretical_exponential_quantiles.min(), transformed_excesses.min())
    max_val = max(theoretical_exponential_quantiles.max(), transformed_excesses.max())

    # Get the current axes
    ax = plt.gca()

    # Linear scale reference line
    line_range = np.linspace(min_val, max_val, 1000)
    sns.lineplot(
        x=line_range,
        y=line_range,
        color=qq_line_color,
        linestyle="--",
        label="45° Line",
    )

    # Apply log scale if requested (though usually not needed with this approach)
    if use_log_scale:
        # Ensure min_val is positive for log scale
        min_val = max(min_val, epsilon)

        # Set log scale for both axes
        ax.set_xscale("log")
        ax.set_yscale("log")

        if custom_y_ticks:
            # Set custom ticks if within range
            valid_ticks = custom_y_ticks[
                (custom_y_ticks >= min_val) & (custom_y_ticks <= max_val)
            ]

            if len(valid_ticks) > 0:
                ax.set_xticks(valid_ticks)
                ax.set_yticks(valid_ticks)

        # Format the tick labels
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # Set labels for log scale
        plt.xlabel("log(Theoretical Exponential Quantiles)")
        plt.ylabel("log(Transformed Excesses)")
    else:
        # Set labels for linear scale
        plt.xlabel("Theoretical Exponential Quantiles")
        plt.ylabel("Transformed Excesses")

    # Set title
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
    # ax1.set_title(scatter_title, fontsize=14)
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
    # ax2.set_title(line_title, fontsize=14)
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
    # ax.set_title(scatter_title, fontsize=14)

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


def plot_support_line(
    x: jnp.array,
    y: jnp.array,
    x_label: str = "Epoch",
    y_label: str = "Proportion of responses within support",
    main_title: str = "Responses' Support Validity Progression",
    figsize: Tuple[float, float] = (12, 8),
    color: str = "blue",
    y_cutoff_lower: Optional[float] = 0.57,
    y_cutoff_upper: float = 1.005,
    do_save: bool = True,
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
) -> None:
    """
    Plots the progression of response validity over epochs for a single N.
    """
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Single line plot without legend
    sns.lineplot(
        x=x,
        y=y,
        color=color,
        ax=ax,
        linewidth=2.0,
        linestyle="-",
    )

    # Set axis limits and labels
    lower_bound = y_cutoff_lower if y_cutoff_lower is not None else 0.0
    ax.set_ylim(lower_bound, y_cutoff_upper)
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)

    # Add grid and title
    ax.grid(True)
    plt.title(main_title, fontsize=14)
    plt.tight_layout()
    plt.show()

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
                "The single support_check plot has not been saved. Flag do_save=True for saving.",
                category=UserWarning,
            )
    elif file_name or save_dir:
        raise ValueError(
            "Both file_name and save_dir must be provided to save the plot."
        )


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
    main_title: str = "Responses' Support Validity Progression: 100 Runs per N\nUsing the Epoch's current Optimal Parameters",
    x_label: str = "Epoch",
    y_label: str = "Proportion of responses within support",
    figsize: Tuple[float, float] = (12, 8),
    legend_fontsize: int = 12,
    x_cut: Optional[int] = None,
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

    # Prepare consistent color mapping based on original Ns list
    present_Ns = df["N"].unique().tolist()
    # Sort present_Ns according to their original position in Ns
    present_Ns_sorted = sorted(present_Ns, key=lambda x: Ns.index(int(x.split("=")[1])))
    # Get corresponding color indices from original colorpalette
    color_indices = [Ns.index(int(n.split("=")[1])) for n in present_Ns_sorted]
    colors = [colorpalette[i] for i in color_indices]

    # Create a color dictionary for consistent mapping
    color_dict = {n: colors[i] for i, n in enumerate(present_Ns_sorted)}

    if x_cut is None:
        fig, ax = plt.subplots(figsize=figsize)

        # Use the original plotting code with color_palette parameter
        sns.lineplot(
            data=df,
            x="Epoch",
            y="Proportion",
            hue="N",
            units="Run",
            alpha=0.5,
            palette=color_dict,  # Use the consistent color mapping
            legend=True,
            ax=ax,
            estimator=None,
            linestyle="-",
            linewidth=2.0,
        )

        # Set axis limits
        lower_bound = y_cutoff_lower if y_cutoff_lower is not None else 0
        ax.set_xlim(1, epochs_max)
        ax.set_ylim(lower_bound, y_cutoff_upper)

        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.grid(True)

        plt.title(main_title, fontsize=14)

        # Create custom legend with consistent colors
        handles = [
            plt.Line2D([0], [0], color=color_dict[n], label=n, lw=3)
            for n in present_Ns_sorted
        ]
        legend = ax.legend(
            handles=handles,
            loc="best",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 2,
        )
        for line in legend.get_lines():
            line.set_linewidth(3.0)
            line.set_alpha(1.0)

        plt.tight_layout()

        # Save if requested
        if do_save and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, file_name), bbox_inches="tight", format="svg"
            )

        plt.show()

    else:
        right_start = max(epochs_max - x_prolonging, x_cut + 1)
        left_df = df[df["Epoch"] <= x_cut]
        right_df = df[df["Epoch"] >= right_start]

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

        # Plot left side with consistent colors
        sns.lineplot(
            data=left_df,
            x="Epoch",
            y="Proportion",
            hue="N",
            units="Run",
            alpha=0.3,
            palette=color_dict,  # Use the consistent color mapping
            legend=False,
            ax=ax1,
            estimator=None,
            linestyle="-",
            linewidth=2.0,
        )
        # Plot right side with consistent colors
        sns.lineplot(
            data=right_df,
            x="Epoch",
            y="Proportion",
            hue="N",
            units="Run",
            alpha=0.3,
            palette=color_dict,  # Use the consistent color mapping
            legend=False,
            ax=ax2,
            estimator=None,
            linestyle="-",
            linewidth=2.0,
        )

        # Set axis limits
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

        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel(y_label, fontsize=14)
        ax2.set_xlabel(None, fontsize=14)
        ax2.set_ylabel(None, fontsize=14)
        ax1.xaxis.set_label_coords(0.8, -0.05)

        plt.suptitle(main_title, y=0.93, fontsize=15)

        # Create custom legend with consistent colors
        handles = [
            plt.Line2D([0], [0], color=color_dict[n], label=n, lw=3)
            for n in present_Ns_sorted
        ]
        legend = ax2.legend(
            handles=handles,
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


def plot_support_checks_histogram(
    support_data_dict: Dict,
    Ns: List[int],
    epochs_max: int,
    colorpalette: List[str],
    save_dir: str = None,
    file_name: str = "support_checks_histogram.svg",
    do_save: bool = True,
    x_label: str = "Proportions",
    y_label: str = "Relative frequency of proportions",
    figsize: Tuple[float, float] = (8, 6),
    legend_fontsize: int = 12,
) -> None:
    # Build DataFrame for last epoch
    title = f"Epoch {epochs_max}'s Support Validity Frequencies: 100 Runs per N\nUsing the final optimal parameters"
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
    # plt.title(title, fontsize=14)
    ax.minorticks_on()
    ax.grid(True, which="both")  # 'both' will show grid for both major and minor ticks
    # Or for more customization:
    ax.grid(True, which="major", linestyle="-")
    ax.grid(True, which="minor", linestyle=":")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter("{x:.1f}")
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

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


def plot_support_checks_histogram_mcmc(
    support_data_dict: Dict,
    Ns: List[int],
    colorpalette: List[str],
    save_dir: str = None,
    file_name: str = "support_checks_histogram_mcmc.svg",
    do_save: bool = True,
    x_label: str = "Proportions",
    y_label: str = "Rel. freq. of proportions",
    figsize: Tuple[float, float] = (8, 6),
    legend_fontsize: int = 12,
) -> None:
    # Build DataFrame for last epoch
    title = f"MCMC Support Validity Frequencies: 100 Runs per N\nUsing the final optimal parameters"
    data = []
    for N in Ns:
        N_str = str(N)
        runs = support_data_dict[N_str]
        for i in range(len(runs)):
            mean = runs[i]
            data.append(
                {
                    "N": f"N={N}",
                    "Proportion": mean,
                    "Run": f"{N}_{i}",
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
    # plt.title(title, fontsize=14)
    ax.minorticks_on()
    ax.grid(True, which="both")  # 'both' will show grid for both major and minor ticks
    # Or for more customization:
    ax.grid(True, which="major", linestyle="-")
    ax.grid(True, which="minor", linestyle=":")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter("{x:.1f}")
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

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


def plot_parameter_kdes(
    svi_samples,
    mcmc_samples,
    svi_full_posterior,
    mcmc_full_posterior,
    fig_title,
    n_cols=8,
    n_rows=6,
    figsize=(24, 18),
    color_svi="#1f77b4",
    color_mcmc="#ff7f0e",
    line_width=1.5,
    alpha=0.7,
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    save_dir=None,
    file_name=None,
    dpi=300,
    format="svg",
    show_plot=True,
):
    # Convert JAX arrays to numpy and extract parameters
    svi_samples = {k: np.array(v) for k, v in svi_samples.items()}
    mcmc_samples = {k: np.array(v) for k, v in mcmc_samples.items()}
    params = ["Full Posterior"] + sorted(
        [k for k in svi_samples if k != "Full Posterior"]
    )

    # Create figure and grid
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.ravel()

    # Plot each parameter
    for idx, param in enumerate(params):
        ax = axs[idx]
        if param == "Full Posterior":
            svi_data = np.array(svi_full_posterior)
            mcmc_data = np.array(mcmc_full_posterior)
        else:
            svi_data = svi_samples[param]
            mcmc_data = mcmc_samples[param]

        sns.kdeplot(
            svi_data,
            ax=ax,
            color=color_svi,
            linewidth=line_width,
            label="SVI",
            fill=True,
            alpha=alpha,
        )
        sns.kdeplot(
            mcmc_data,
            ax=ax,
            color=color_mcmc,
            linewidth=line_width,
            label="MCMC",
            fill=True,
            alpha=alpha,
        )

        ax.set_title(param, fontsize=title_fontsize, pad=8)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.set_xlabel("Value", fontsize=label_fontsize)
        ax.set_ylabel("Density", fontsize=label_fontsize)

        if idx == 0:
            ax.legend(fontsize=legend_fontsize, frameon=False)

    # Hide empty subplots
    for idx in range(len(params), n_rows * n_cols):
        axs[idx].axis("off")

    # plt.suptitle(fig_title, y=0.99, fontsize=title_fontsize + 2, fontweight="bold")
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)

    # Save handling
    if save_dir and file_name:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, f"{file_name}.{format}")
        plt.savefig(full_path, bbox_inches="tight", dpi=dpi)

    if show_plot:
        plt.show()
    else:
        plt.close()
