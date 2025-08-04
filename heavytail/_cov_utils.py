# Copyright (c) 2025 Mohammadjavad Vakili. All rights reserved.
import numpy as np


def _check_positive_definite(matrix: np.ndarray) -> bool:  # pragma: no cover
    """Check if a matrix is positive definite.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to check.

    Returns
    -------
    bool
        True if the matrix is positive definite, False otherwise.

    """
    return np.all(np.linalg.eigvals(matrix) > 0)


def _symmetrizer(matrix: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Symmetrize a matrix by averaging it with its transpose.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to symmetrize.

    Returns
    -------
    np.ndarray
        The symmetrized matrix.

    """
    return (matrix + matrix.T) / 2
