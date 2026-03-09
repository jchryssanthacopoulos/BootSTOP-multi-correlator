"""Utility functions for analyzing experiment results."""

from typing import Optional

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from multicorrelator.spin_partition import SpinPartition
from multicorrelator.utils.config import load_config_file


DELTA_EPSILON_TRUTH = 1.41267
LAMBDA_SSSS_TRUTH = 1.106396
LAMBDA_EEEE_TRUTH = 2.348358
LAMBDA_SSEE_TRUTH = 1.611898
DELTA_SIGMA_TRUTH = 0.518154
LAMBDA_SESE_TRUTH = 1.106396


def load_results(experiment: dict) -> pd.DataFrame:
    """Load the results from the given files.

    Args:
        experiment: The experiment dictionary containing keys 'experiment_dir' and 'files'

    Returns:
        The concatenated DataFrame

    """
    files = [f'{experiment['experiment_dir']}/{file}' for file in experiment['files']]

    dataframes = [pd.read_csv(file) for file in files]

    # Remove empty DataFrames
    dataframes = [df for df in dataframes if not df.empty]

    if not dataframes:
        return pd.DataFrame()

    results = pd.concat(dataframes, axis=0, ignore_index=True)

    # Trim columns after "crossing_error"
    if "crossing_error" in results.columns:
        results = results.loc[:, : "crossing_error"]

    return results


def get_spectra(
        config_file: str,
        data: pd.DataFrame,
        error_threshold: Optional[float] = None
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Load the positive and negative parity spectra from the data.

    Args:
        config_file: The configuration file to use
        data: The data to load the spectra from
        error_threshold: The threshold for crossing error above which a given experiment is ignored

    Returns:
        The positive and negative parity spectra

    """
    # load the spectrum structure
    spin_partition_dict = load_config_file(config_file)
    spin_partition = SpinPartition(**spin_partition_dict)

    positive_parity_dfs = []
    negative_parity_dfs = []

    for _, row in data.iterrows():
        crossing_error = row['crossing_error']

        # ignore experiment if crossing error is above threshold
        if error_threshold is not None and crossing_error > error_threshold:
            continue

        # drop evolution, island, and crossing_error columns
        x = row.values[2:-1]

        # seed the spin partition with the data
        spin_partition.from_array(x)

        df1 = spin_partition.print_positive_parity_dataframe()
        # add the crossing error to the dataframe
        df1['crossing_error'] = crossing_error

        df2 = spin_partition.print_negative_parity_dataframe()
        df2['crossing_error'] = crossing_error

        positive_parity_dfs.append(df1)
        negative_parity_dfs.append(df2)

    return positive_parity_dfs, negative_parity_dfs


def aggregate_dataframes(
        dataframes: list[pd.DataFrame],
        inverse_error_weighting: Optional[bool] = False
) -> pd.DataFrame:
    """Aggregates a list of DataFrames by calculating the weighted average and standard deviation for numeric columns.

    Args:
        dataframes: List of DataFrames to aggregate.
        inverse_error_weighting: If True, weights each DataFrame's contribution by the inverse of its 'crossing_error'.

    Returns:
        A new DataFrame with weighted average and standard deviation for numeric columns.

    """
    if not dataframes:
        return pd.DataFrame()

    # Extract the inverse of the first `crossing_error` value from each DataFrame
    weights = (
        np.array([1 / df["crossing_error"].iloc[0] for df in dataframes])
        if inverse_error_weighting else np.ones(len(dataframes))
    )

    # Normalize so that sum(weights) = 1
    weights /= np.sum(weights)

    # Concatenate all DataFrames while preserving source indices
    combined_df = pd.concat(dataframes, keys=range(len(dataframes)), ignore_index=False)

    # Extract numeric columns
    numeric_df = combined_df.select_dtypes(include="number")

    # Assign normalized weights to each DataFrame's rows
    weight_map = {i: w for i, w in enumerate(weights)}
    combined_df["weight"] = combined_df.index.get_level_values(0).map(weight_map)

    # Compute weighted mean
    weighted_avg = numeric_df.mul(combined_df["weight"], axis=0).groupby(level=1).sum()

    # Compute unbiased weighted standard deviation
    weighted_variance = (
        numeric_df.sub(weighted_avg, level=1).pow(2)
        .mul(combined_df["weight"], axis=0)
        .groupby(level=1).sum()
    )

    # Apply Bessel's correction (for unbiased std calculation)
    n_dfs = len(dataframes)
    if n_dfs > 1:
        weighted_variance *= n_dfs / (n_dfs - 1)

    weighted_std = np.sqrt(weighted_variance)

    # If the standard deviation is zero, replace it with NaN
    weighted_std = weighted_std.replace(0, np.nan)

    # Append suffixes
    avg_df = weighted_avg.add_suffix("_avg")
    std_df = weighted_std.add_suffix("_std")

    # Combine results
    aggregated_df = pd.concat([avg_df, std_df], axis=1).reset_index()

    # Remove unwanted columns
    aggregated_df = aggregated_df.drop(columns=["index"], errors="ignore")
    aggregated_df = aggregated_df.drop(columns=["crossing_error_avg", "crossing_error_std"], errors="ignore")

    # Clean up spin column
    if "spin_std" in aggregated_df.columns:
        aggregated_df = aggregated_df.drop(columns=["spin_std"])
    if "spin_avg" in aggregated_df.columns:
        aggregated_df = aggregated_df.rename(columns={"spin_avg": "spin"})

    return aggregated_df


def get_spectra_and_aggregate(
        config_file: str,
        data: pd.DataFrame,
        error_threshold: Optional[float] = None,
        inverse_error_weighting: Optional[bool] = False,
        data_label: Optional[str] = None
) -> pd.DataFrame:
    """Load the spectra from the data and aggregate them, returning results for epsilon and sigma.

    Args:
        config_file: The configuration file to use
        result: The data to load the spectra from
        error_threshold: The threshold for crossing error above which a given experiment is ignored
        inverse_error_weighting: If True, weights each DataFrame's contribution by the inverse of its 'crossing_error'
        data_label: An optional label to add to the resulting DataFrame

    Returns:
        The aggregated DataFrame

    """
    positive_parity_dfs, negative_parity_dfs = get_spectra(config_file, data, error_threshold=error_threshold)

    # If no valid experiments were found, return an empty DataFrame
    if not positive_parity_dfs:
        return pd.DataFrame()

    # Aggregate the DataFrames
    positive_parity_df_agg = aggregate_dataframes(
        positive_parity_dfs,
        inverse_error_weighting=inverse_error_weighting
    )
    negative_parity_df_agg = aggregate_dataframes(
        negative_parity_dfs,
        inverse_error_weighting=inverse_error_weighting
    )

    # Extract the second operator in the positive spectrum, which corresponds to epsilon
    epsilon_df = positive_parity_df_agg.iloc[[1]].copy()
    epsilon_df["operator"] = "epsilon"

    # Extract the first operator in the negative spectrum, which corresponds to sigma
    sigma_df = negative_parity_df_agg.iloc[[0]].copy()
    sigma_df["operator"] = "sigma"

    # Combine the two DataFrames
    combined_df = pd.concat([epsilon_df, sigma_df], ignore_index=True)

    # Drop 'spin' column
    if "spin" in combined_df.columns:
        combined_df = combined_df.drop(columns=["spin"])

    # Add a label column
    if data_label is not None:
        combined_df["label"] = data_label
        first_columns = ["label", "operator"]
    else:
        first_columns = ["operator"]

    column_order = first_columns + [col for col in combined_df.columns if col not in first_columns]

    return combined_df[column_order]


def plot_bell_curve(
        df: pd.DataFrame,
        row_index: int,
        avg_col: str,
        std_col: str,
        truth_val: float,
        xlabel: str,
        ax,
        x_min: float,
        x_max: float,
        sample_points: list[float] = None
):
    """Plots a bell curve for a specific row in the aggregated DataFrame.

    Args:
        df: The aggregated DataFrame containing the data
        row_index: The index of the row to plot
        avg_col: The column name for the average value
        std_col: The column name for the standard deviation
        truth_val: The true value to compare with
        xlabel: The x-axis label
        ax: The axes object where the plot will be created
        x_min: The minimum x value for the plot
        x_max: The maximum x value for the plot
        sample_points: Specific x values to plot on the bell curve

    """
    if row_index < 0 or row_index >= len(df):
        raise IndexError("Row index out of range.")

    if avg_col not in df.columns or std_col not in df.columns:
        raise ValueError("Specified columns not found in the DataFrame.")

    # Extract the average and standard deviation
    avg = df.loc[row_index, avg_col]
    std = df.loc[row_index, std_col]

    # Generate x values for plotting
    x = np.linspace(x_min, x_max, 500)

    # Compute the bell curve
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - avg) / std) ** 2)

    # Plot the bell curve
    ax.plot(x, y)
    ax.set_xlabel(xlabel)

    # Highlight the ±1σ region
    x_shaded = np.linspace(avg - 1 * std, avg + 1 * std, 500)
    y_shaded = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_shaded - avg) / std) ** 2)
    ax.fill_between(x_shaded, 0, y_shaded, alpha=0.2, color='blue', label="±1$\\sigma$")

    # Vertical lines for the mean and truth
    ax.axvline(avg, color='green', linestyle='--', label="Mean")
    ax.axvline(truth_val, color='red', linestyle='--', label="True Value")

    if sample_points:
        sample_y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.array(sample_points) - avg) / std) ** 2)
        ax.scatter(sample_points, sample_y, color='green', s=10, zorder=5)

    ax.legend(fontsize=7)
    ax.grid()


def plot_bell_curve_series(
        series: pd.Series,
        truth_val: float,
        xlabel: str,
        ax,
        x_min: float,
        x_max: float,
        crossing_error: pd.Series = None
):
    """Plots a bell curve for a given Series with samples overlaid.

    Optionally overlays an inverse crossing error weighted mean and std.

    Args:
        series: A pandas Series of numeric values (the samples)
        truth_val: The true value to compare with
        xlabel: The x-axis label
        ax: The axes object where the plot will be created
        x_min: The minimum x value for the plot
        x_max: The maximum x value for the plot
        crossing_error: Optional pandas Series of crossing errors for weighting

    """
    if series.empty:
        raise ValueError("Series is empty.")

    # Compute simple mean and std
    avg = series.mean()
    std = series.std()

    # Generate x values for plotting
    x = np.linspace(x_min, x_max, 500)

    # Plot the standard bell curve
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - avg) / std) ** 2)
    ax.plot(x, y, color='orange', label="Unweighted Fit")

    # Shade ±1σ region
    x_shaded = np.linspace(avg - std, avg + std, 500)
    y_shaded = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_shaded - avg) / std) ** 2)
    ax.fill_between(x_shaded, 0, y_shaded, alpha=0.2, color='orange', label="Unweighted ±1$\\sigma$")

    # Vertical line for unweighted mean
    ax.axvline(avg, color='orange', linestyle='--', label="Unweighted Mean")

    # If crossing_error is provided, compute weighted mean/std
    if crossing_error is not None:
        if len(series) != len(crossing_error):
            raise ValueError("series and crossing_error must have the same length.")
        weights = 1 / crossing_error
        weighted_avg = np.sum(series * weights) / np.sum(weights)
        weighted_std = np.sqrt(np.sum(weights * (series - weighted_avg) ** 2) / np.sum(weights))

        # Plot weighted bell curve
        y_w = (1 / (weighted_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - weighted_avg) / weighted_std) ** 2)
        ax.plot(x, y_w, color='blue', label="Weighted Fit")

        # Shade ±1σ region for weighted
        x_w_shaded = np.linspace(weighted_avg - weighted_std, weighted_avg + weighted_std, 500)
        y_w_shaded = (
            (1 / (weighted_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_w_shaded - weighted_avg) / weighted_std) ** 2)
        )
        ax.fill_between(x_w_shaded, 0, y_w_shaded, alpha=0.2, color='blue', label="Weighted ±1$\\sigma$")

        # Vertical line for weighted mean
        ax.axvline(weighted_avg, color='blue', linestyle='--', label="Weighted Mean")

    # Vertical line for truth
    ax.axvline(truth_val, color='red', linestyle='--', label="True Value")

    ax.set_xlabel(xlabel)
    ax.legend(fontsize=7)
    ax.grid()


def plot_crossing_error_cdf(
        error_series_list: list[pd.Series],
        labels: list[str],
        filename_base: str = None,
        max_error_to_save: float = None
):
    """Plot the CDF of crossing errors for multiple experiments.

    Args:
        error_series_list: List of crossing error Series
        labels: List of labels for the experiments
        filename_base: Optional filename to save the CDF data
        max_error_to_save: Maximum error value to save in the output file

    """
    plt.figure(figsize=(6, 5))

    for error_series, label in zip(error_series_list, labels):
        sorted_errors = np.sort(error_series)  # Sort the errors
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)  # Compute CDF values
        plt.plot(sorted_errors, cdf, '.-', label=label, linewidth=2)

        if max(error_series) < max_error_to_save:
            sorted_errors = np.append(sorted_errors, max_error_to_save)
            cdf = np.append(cdf, 1)

        # Save to a file
        if filename_base:
            data_to_save = pd.DataFrame({"errors": sorted_errors, "cdf": cdf})
            label_to_save = label.replace("=", "_")
            data_to_save.to_csv(f"{filename_base}_{label_to_save}.txt", sep="\t", index=False, float_format="%.6f")

    plt.xlabel("Crossing Error")
    plt.ylabel("CDF")
    plt.title("CDF of Crossing Errors")
    plt.legend()
    plt.grid()
    plt.xlim(-1, 100)


def plot_delta_samples(
        delta_epsilon_samples: pd.Series,
        delta_sigma_samples: pd.Series,
        crossing_error: pd.Series,
        results: dict = None,
        xlim: tuple = None,
        ylim: tuple = None,
        ax: plt.Axes = None,
        title: str = None
):
    """Plot the delta samples in the epsilon-sigma plane with color coding for crossing error.

    Args:
        delta_epsilon_samples: The delta epsilon samples
        delta_sigma_samples: The delta sigma samples
        crossing_error: The crossing error values
        results: Dict containing weighted/unweighted (avg, std) for Δσ and Δε
        xlim: The x-axis limits
        ylim: The y-axis limits
        ax: The axes object where the plot will be created
        title: The title of the plot

    """
    if xlim is None:
        xlim = (0, 7.5)
    if ylim is None:
        ylim = (0, 7.5)

    # Create new figure/axis if none provided
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    sc = ax.scatter(delta_epsilon_samples, delta_sigma_samples, c=crossing_error, cmap='viridis', s=10)
    ax.scatter(DELTA_EPSILON_TRUTH, DELTA_SIGMA_TRUTH, color='red', marker='x', s=20)

    # Plot ellipses if results provided
    if results is not None and "Δε" in results and "Δσ" in results:
        for key, style in zip(["unweighted", "weighted"], [("orange", ":"), ("blue", "--")]):
            eps_avg, eps_std = results["Δε"][key]
            sig_avg, sig_std = results["Δσ"][key]

            # Center of ellipse
            ax.scatter(eps_avg, sig_avg, color=style[0], marker="o", s=20, label=f"{key.capitalize()} Mean")

            # Add ellipse (1σ region)
            ellipse = Ellipse(
                (eps_avg, sig_avg),
                width=2 * eps_std,  # 2σ total width
                height=2 * sig_std,  # 2σ total height
                edgecolor=style[0],
                linestyle=style[1],
                facecolor="none",
                linewidth=2,
                label=f"{key.capitalize()} ±1σ"
            )
            ax.add_patch(ellipse)

    # Colorbar requires a figure
    if ax.figure:
        cbar = ax.figure.colorbar(sc, ax=ax)
        cbar.set_label('Crossing Error')

    ax.set_xlabel('$\\Delta_{\\epsilon}$')
    ax.set_ylabel('$\\Delta_{\\sigma}$')

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Scaling Dimension Estimates (True = Red)')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    ax.grid()


def get_p_value(samples: list, true_mean: float, alpha: float) -> float:
    """Computes the p-value for a one-sample t-test.

    Args:
        samples: The sample data
        true_mean: The true mean to compare with
        alpha: The significance level

    Returns:
        The p-value

    """
    samples = np.array(samples)

    sample_mean = np.mean(samples)
    sample_std = np.std(samples, ddof=1)
    n = len(samples)

    # t-test (use t-distribution because we estimate variance)
    t_stat = (sample_mean - true_mean) / (sample_std / np.sqrt(n))
    p_value = 2 * t.sf(np.abs(t_stat), df=n - 1)    # Two-tailed test

    # Print results
    if p_value < alpha:
        print("Reject the null hypothesis: The sample mean is significantly different from the true mean.")
    else:
        print("Fail to reject the null hypothesis: The sample mean is not significantly different from the true mean.")

    return p_value


def compute_weighted_estimates(df: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    """Compute inverse-crossing-error weighted estimates for deltas and OPE coefficients.

    Prints results in the form Estimate = mean ± std dev.

    """
    # Define variables to process: {df_column: pretty_label}
    variables = {
        "delta_negative_z2_sigma_op_1": "Δσ",
        "delta_positive_z2_epsilon_op_1": "Δε",
        "ope_coeff_positive_z2_epsilon_op_1_ssss": "λσσε",
        "ope_coeff_positive_z2_epsilon_op_1_eeee": "λεεε",
        "ope_coeff_negative_z2_sigma_op_1_sese": "λσεσ",
    }

    # Ground truth values
    true_values = {
        "Δσ": 0.5181489,
        "Δε": 1.412625,
        "λσσε": 1.0518537,
        "λεεε": 1.532435,
        "λσεσ": 1.0518537
    }

    # Compute weights
    crossing_error = df["crossing_error"].to_numpy()
    weights = 1 / crossing_error
    weights /= weights.sum()

    results = {}

    for col, label in variables.items():
        if col not in df.columns:
            print(f"Warning: column '{col}' not found in DataFrame, skipping.")
            continue
        values = df[col].to_numpy()

        # Weighted
        w_avg = np.average(values, weights=weights)
        w_std = np.sqrt(np.average((values - w_avg) ** 2, weights=weights))

        # Unweighted
        uw_avg = values.mean()
        uw_std = values.std(ddof=0)

        results[label] = {
            "weighted": (w_avg, w_std),
            "unweighted": (uw_avg, uw_std),
            "truth": true_values.get(label, None),
        }

    # Print nicely
    print("-" * 70)
    print(f"{'Variable':10s} | {'Weighted (±σ)':23s} | {'Unweighted (±σ)':19s} | {'Truth':10s}")
    print("-" * 70)
    for label, vals in results.items():
        w_avg, w_std = vals["weighted"]
        uw_avg, uw_std = vals["unweighted"]
        truth = vals["truth"]
        truth_str = f"{truth:.6f}" if truth is not None else "-"
        print(f"{label:10s} | {w_avg:.6f} ± {w_std:.6f}     | {uw_avg:.6f} ± {uw_std:.6f} | {truth_str}")
    print("-" * 70)

    return results
