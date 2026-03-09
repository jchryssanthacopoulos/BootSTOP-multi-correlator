"""Utility functions for calculating metrics."""

import numpy as np


def get_total_relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sort_and_remove_zeros: bool = True,
    average_axis: int = 1
) -> np.ndarray:
    """Calculate the total relative error between true and predicted values.

    Args:
        y_true: True values, with shape (n_samples, n_features)
        y_pred: Predicted values, with shape (n_samples, n_features)
        sort_and_remove_zeros: Whether to sort the errors and remove zeros
        average_axis: Axis along which to compute the norms

    Returns:
        A sorted array of total relative errors, excluding zeros

    """
    y_true_norm = np.linalg.norm(y_true, axis=average_axis)
    y_diff_norm = np.linalg.norm(y_true - y_pred, axis=average_axis)

    # Add a small value to avoid division by zero
    y_true_norm[y_true_norm == 0] = 1e-10

    total_relative_error = y_diff_norm / y_true_norm

    if not sort_and_remove_zeros:
        return total_relative_error

    # Sort and remove zeros
    sorted_total_relative_error = np.sort(total_relative_error)
    sorted_total_relative_error = sorted_total_relative_error[sorted_total_relative_error > 0]

    return sorted_total_relative_error


def get_average_component_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sort_and_remove_zeros: bool = True,
    average_axis: int = 1,
    threshold: float = 1e-15
) -> np.ndarray:
    """Calculate the average component error between true and predicted values.

    Args:
        y_true: True values, with shape (n_samples, n_features)
        y_pred: Predicted values, with shape (n_samples, n_features)
        sort_and_remove_zeros: Whether to sort the errors and remove zeros
        average_axis: Axis along which to compute the average
        threshold: Threshold for treating small true values as zero (for hybrid error)

    Returns:
        A sorted array of average component errors, excluding zeros

    """
    # # Add a small value to avoid division by zero
    # y_true[y_true == 0] = 1e-10

    # average_component_error = (np.abs(y_true - y_pred) / np.abs(y_true)).mean(axis=average_axis)

    # if not sort_and_remove_zeros:
    #     return average_component_error

    # sorted_average_component_error = np.sort(average_component_error)
    # sorted_average_component_error = sorted_average_component_error[sorted_average_component_error > 0]

    # return sorted_average_component_error

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    abs_true = np.abs(y_true)
    abs_diff = np.abs(y_true - y_pred)

    # Relative error (safe because masked)
    relative_error = abs_diff / abs_true

    # Absolute error
    absolute_error = abs_diff

    # Mask: where true value is small
    small_mask = abs_true < threshold

    # Hybrid error
    component_error = np.where(small_mask, absolute_error, relative_error)

    average_component_error = component_error.mean(axis=average_axis)

    if not sort_and_remove_zeros:
        return average_component_error

    sorted_error = np.sort(average_component_error)

    return sorted_error[sorted_error > 0]
