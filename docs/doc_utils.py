# Copyright (c) 2024 Mohammadjavad Vakili. All rights reserved.
"""Utility functions for generating 2D data with outliers and plotting covariance contours."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_t


def generate_2d_data_with_outliers(
    n_samples: int = 100,
    n_outliers: int = 10,
    mean: tuple = (0, 0),
    cov: list | None = None,
    outlier_range: float = 8,
) -> np.ndarray:
    """
    Generate synthetic 2D data with outliers.

    Parameters
    ----------
    n_samples : int
        Number of normal data samples to generate.
    n_outliers : int
        Number of outlier samples to generate.
    mean : tuple
        Mean of the normal data distribution.
    cov : list or np.ndarray
        Covariance matrix for the normal data.
    outlier_range : float
        Range for generating outlier values.

    Returns
    -------
    np.ndarray
        Combined array of normal data and outliers, shape (n_samples + n_outliers, 2).
    """
    if cov is None:
        cov = [[1, 0.5], [0.5, 1]]
    # Generate normal data
    rng = np.random.default_rng()
    data = rng.multivariate_normal(mean, cov, n_samples)
    # Generate outliers
    outliers = rng.uniform(low=-outlier_range, high=outlier_range, size=(n_outliers, 2))
    # Combine data and outliers
    return np.vstack([data, outliers])


def generate_2d_student_t_data(
    n_samples: int = 100,
    cov: list | None = None,
    nu: float = 4.0,
) -> np.ndarray:
    """
    Generate synthetic 2D data from a multivariate Student's t-distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    cov : list or np.ndarray
        Covariance matrix for the data.
    nu : float
        Degrees of freedom for the Student's t-distribution.

    Returns
    -------
    np.ndarray
        Generated data of shape (n_samples, 2).
    """
    if cov is None:
        cov = [[1, 0], [0, 1]]
    scatter = (nu / (nu - 2)) * np.array(cov)
    return multivariate_t.rvs(loc=[0, 0], shape=scatter, df=nu, size=n_samples)


def plot_covariance_contour(
        mean: tuple[float, float],
        cov: np.ndarray,
        ax: Axes | None = None,
        n_std: float = 2.0,
        **kwargs: dict[str, Any],
) -> Axes:
    """Plot covariance ellipse for 2D data.

    Parameters
    ----------
        mean: tuple[float, float]
            Mean vector (2D).
        cov: np.ndarray
            Covariance matrix (2x2).
        ax: Optional[plt.Axes]
            Matplotlib axis to plot on. If None, uses current axis.
        n_std: float
            Factor for scaling the standard deviations for ellipse size.
        **kwargs: dict[str, Any]
            Additional keyword arguments for Ellipse.

    Returns
    -------
        plt.axes.Axes
            The matplotlib axis with the covariance ellipse and data plotted.
    """
    if ax is None:
        ax = plt.gca()
    # Eigenvalue decomposition
    evals, evecs = np.linalg.eigh(cov)
    order = evals.argsort()[::-1]
    evals, evecs = evals[order], evecs[:, order]
    theta = np.degrees(np.arctan2(*evecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(evals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, edgecolor="red", fc="None", lw=2, **kwargs)
    ax.add_patch(ellip)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return ax


def plot_2d_data(data: np.ndarray, ax: Axes | None = None) -> Axes:
    """Plot 2D data points.

    Parameters
    ----------
    data: np.ndarray
        2D data points of shape (n_samples, 2).
    ax: Optional[plt.Axes]
        Matplotlib axis to plot on. If None, uses current axis.

    Returns
    -------
    plt.axes.Axes
        The matplotlib axis with the 2D data points plotted.
    """
    if ax is None:
        ax = plt.gca()
    ax.scatter(data[:, 0], data[:, 1], s=10)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return ax
