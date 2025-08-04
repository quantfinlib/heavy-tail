# Copyright (c) 2025 Mohammadjavad Vakili. All rights reserved.
"""Implementation of Tyler's estimator of covariance matrix for heavy-tail data.

Author: Mohammadjavad Vakili

This code implements Tyler's estimator, which is a robust estimator of the covariance matrix suitable for heavy-tailed distributions.
It is particularly useful in scenarios where traditional estimators may fail due to the presence of outliers or non-normality in the data.

Source:
Tyler, D. E. (1987). A distribution-free M-estimator of multivariate scatter. The Annals of Statistics, 15(1), 234_251.
"""

import numpy as np
from numpy.linalg import solve

from heavytail._cov_utils import _check_positive_definite, _symmetrizer
from heavytail._error_utils import _relative_error_array


def tyler_covariance(data: np.ndarray, epsilon: float = 1e-6, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    r"""Compute Tyler's estimator of the covariance matrix.

    Let $\mathbf{X}$ be the input data matrix of shape $\(n_{samples}, n_{features}\)$ with mean zero,
    where $n_{samples}$ is the number of observations and $n_{features}$ is the number of features.
    For brevity of notation, we denote $T = n_{samples}$ and $N = n_{features}$.
    In the context of multivariate time-series data, $T$ is the length of the time series
    and $N$ is the number of variables.

    The input data matrix $\mathbf{X}$ is defined as follows:

    $$
    \mathbf{X} = \big[ \mathbf{x}_1  \& \mathbf{x}_2  \&  \ldots  \&  \mathbf{x}_T \big]
    $$

    where $\mathbf{x}_t$ is the $t$-th observation vector of length $N$.
    For each observation $\mathbf{x}_t$, we can define a normalized version as follows:

    $$
    {\mathbf{s}}_t = \frac{\mathbf{x}_t}{\|\mathbf{x}_t\|} \quad \text{for } t = 1, \ldots, T
    $$

    where $\|\mathbf{x}_t\| = \sqrt{\mathbf{x}_t^T \mathbf{x}_t}$ is the Euclidean norm of the vector $\mathbf{x}_t$.

    Tyler demonstrated that the pdf of the normalized observations $p(\mathbf{s}_t)$
    follows an angular distribution regardless of the tail behavior of $\mathbf{x}_t$:

    $$
    p(\mathbf{s}_t) \propto \frac{1}{\| \Sigma  \|^{1/2}} \big( \mathbf{s}_t^T \Sigma^{-1} \mathbf{s}_t \big) ^ {-\frac{N}{2}},
    $$

    where $\| \Sigma \|$ is the determinant of the scatter matrix $\Sigma$.

    Taking the logarithm of the pdf, we have:
    $$
    \log p(\mathbf{s}_t) = -\frac{N}{2} \log \big( \mathbf{s}_t^T \Sigma^{-1} \mathbf{s}_t \big) - \frac{1}{2} \log \| \Sigma  \|
    $$

    Substituting the normalized observations into the log-likelihood function, we obtain:
    $$
    \log p(\mathbf{s}_t) = -\frac{N}{2} \log \big( \big( \mathbf{x}_t^T \Sigma^{-1} \mathbf{x}_t \big) - \log \big( \|\mathbf{x}_t\|^2 \big) \big) - \frac{1}{2} \log \| \Sigma  \|
    $$

    We can obtain the negative-log-likelihood function of the scatter matrix $\Sigma$ by summing over all observations:

    $$
    \mathcal{L}(\Sigma) = -\sum_{t=1}^{T} \log p(\mathbf{s}_t) = \frac{N}{2} \sum_{t=1}^{T} \log \big( \mathbf{x}_t^T \Sigma^{-1} \mathbf{x}_t \big) + \frac{1}{2} T \log \| \Sigma  \|
    $$

    Therefore, the maximum likelihood estrimator (MLE) of the scatter matrix $\Sigma$ can be obatinaed by minimizing the following expression:

    $$
    \hat{\Sigma} = \arg\min_{\Sigma} \mathcal{L}(\Sigma) = \arg\min_{\Sigma} \Big[ \frac{N}{T} \sum_{t=1}^{T} \log \big( \mathbf{x}_t^T \Sigma^{-1} \mathbf{x}_t \big) + \log \| \Sigma  \| \Big]
    $$

    Solution to this optimization problem can be obtained using an iterative approach, where we update the estimate of the covariance matrix $\Sigma$ until convergence:


    $$
    \hat{\Sigma}^{(k+1)} = \frac{N}{T} \sum_{t=1}^{T} \frac{\mathbf{x}_t \mathbf{x}_t^T}{\mathbf{x}_t^T \hat{\Sigma}^{(k)^{-1}} \mathbf{x}_t}
    $$

    Existence of the solution is guaranteed as long as $ T > N $.

    The parameter $\Sigma$ can only be estimated up to a scale factor. This scale factor can be found heuristically
    by the robustly estimating the variances of the $N$ variables, and requiring the diagonal elements of the covariance matrix
    to be equal to the estimated variances. This is done by scaling the covariance matrix $\hat{\Sigma}$ as follows:

    First, we compute the variances of the $N$ variables using the median absolute deviation (MAD) as follows:
    $$
    \hat{\sigma}_i^2 = \text{MAD}(\mathbf{x}_i) = 1.4826 \cdot \text{median}(|\mathbf{x}_i - \text{median}(\mathbf{x}_i)|)^2
    $$

    Next, we scale the covariance matrix $\hat{\Sigma}$ by requiring the diagonal elements to be equal to the estimated variances:
    $$
    \hat{\Sigma}_{ii} = \hat{\sigma}_i^2, \quad i = 1, \ldots, N
    $$
    This can be achieved by scaling the covariance matrix as follows:

    $$
    \hat{\Sigma} \rightarrow \frac{1}{N} \mathbf{1}^{T} \cdot \hat{\sigma}^{2} \cdot\hat{\Sigma},
    $$
    where $\mathbf{1}$ is a vector of ones of length $N$ and $\hat{\sigma}^{2}$ is the vector of estimated variances.

    Parameters
    ----------
    data : ndarray
        Input data matrix of shape (n_samples, n_features).
    epsilon : float, optional
        Small value to add to the diagonal elements of the initial guess of the covariance matrix.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    C : ndarray
        Estimated covariance matrix.

    Raises
    ------
    ValueError
        If the estimated covariance matrix is not positive definite.

    References
    ----------
    Tyler, D. E. (1987). A distribution-free M-estimator of multivariate scatter. The Annals of Statistics, 15(1), 234_251.

    https://bookdown.org/palomar/portfoliooptimizationbook/3.5-heavy-tail-ML.html#tylers-estimator

    """  # noqa: E501
    n_samples, n_features = data.shape
    mu = np.median(data, axis=0)  # Shape (N,)
    # Center the data by subtracting the median
    data_cen = data - mu  # Shape (T, N)
    # Initial guess for covariance matrix
    cov = np.cov(data_cen, rowvar=False)  # Shape (N, N)
    cov /= np.trace(cov)  # Normalize to have unit trace

    # Iterative Tyler's estimator of covariance matrix
    for _ in range(max_iter):
        cov_prev = cov.copy()
        cov_prev += epsilon * np.eye(n_features)
        z = solve(cov_prev, data_cen.T).T  # Shape (T, N)
        data_inv_cov_data = np.sum(data_cen * z, axis=1)  # Shape (T,)
        weights = 1 / np.maximum(data_inv_cov_data, epsilon)  # Shape(T, ). Avoid division by zero
        # Update covariance matrix
        cov = (n_samples / data_cen.shape[0]) * (data_cen.T @ np.diag(weights) @ data_cen)  # Shape (N, N)
        # Check for convergence
        if _relative_error_array(x=cov, y=cov_prev) < tol:
            break
    # Estimate the variances of the features using median absolute deviation
    sigma2 = _median_absolute_deviation(data)
    # Scale the covariance matrix to match the estimated variances
    cov = np.outer(np.ones(n_features), sigma2) * cov / np.diag(cov)
    cov = _symmetrizer(cov)  # Ensure symmetry
    if not _check_positive_definite(cov):
        msg = "Estimated covariance matrix is not positive definite."
        raise ValueError(msg)
    return cov


def _median_absolute_deviation(x: np.ndarray) -> np.ndarray:
    """Compute the median absolute deviation (MAD) for each feature in the data matrix X.

    Parameters
    ----------
    x : np.ndarray
        Input data matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Estimated variances for each feature.
    """
    return (1.4826**2) * np.median(np.abs(x - np.median(x, axis=0)), axis=0) ** 2
