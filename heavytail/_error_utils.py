# Copyright (c) 2025 Mohammadjavad Vakili. All rights reserved.

import numpy as np


def _relative_error_array(x: np.ndarray, y: np.ndarray) -> float:  # pragma: no cover
    """Compute the relative error between two arrays.

    Parameters
    ----------
    x : np.ndarray
        First array.
    y : np.ndarray
        Second array.

    Returns
    -------
    float
        Relative error between the two arrays.

    """
    return np.linalg.norm(x - y, ord="fro") / np.linalg.norm(y, ord="fro")


def _relative_error_scalar(x: float, y: float) -> float:  # pragma: no cover
    """Compute the relative error between two scalars.

    Parameters
    ----------
    x : float
        First scalar.
    y : float
        Second scalar.

    Returns
    -------
    float
        Relative error between the two scalars.

    """
    return np.abs(x - y) / np.abs(y) if y != 0 else np.inf
