"""Test suite for Tyler's covariance estimator on heavy-tailed data."""


import numpy as np
import pytest
from scipy.stats import multivariate_t

from heavytail._cov_utils import _check_positive_definite
from heavytail.tyler import _median_absolute_deviation, tyler_covariance

rng = np.random.default_rng(42)  # For reproducibility


def generate_heavytail_data(n_samples, n_features, nu) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic heavy-tailed data from a multivariate t-distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples (observations).
    n_features : int
        Number of features (variables).
    nu : float
        Degrees of freedom for the multivariate t-distribution.

    Returns
    -------
    np.ndarray
        Generated data of shape (T, N).
    np.ndarray
        The covariance matrix used to generate the data.
    """
    mu = np.zeros(n_features)
    u = rng.normal(0, np.sqrt(0.1), size=(n_features, int(0.3 * n_features)))
    sigma_cov = u @ u.T + np.eye(n_features)
    sigma_scatter = (nu / (nu - 2)) * sigma_cov
    return multivariate_t.rvs(df=nu, loc=mu, shape=sigma_scatter, size=n_samples, random_state=rng), sigma_cov


@pytest.fixture
def heavytail_data():
    """Fixture to generate heavy-tailed data for testing."""
    n_samples, n_features, nu = 10000, 2, 4
    return generate_heavytail_data(n_samples, n_features, nu)[0]


@pytest.mark.parametrize("n_samples", [1000, 10000])
@pytest.mark.parametrize("n_features", [2, 5, 10])
@pytest.mark.parametrize("nu", [2.5, 4, 10])
def test_tyler_vs_scm_error(n_samples, n_features, nu):
    """Test Tyler's covariance estimator against the sample covariance matrix (SCM).

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    nu : float
        Degrees of freedom for the multivariate t-distribution.
    """
    x, true_cov = generate_heavytail_data(n_samples, n_features, nu)

    sigma_scm = np.cov(x, rowvar=False)
    sigma_tyler = tyler_covariance(x)

    scm_error = np.linalg.norm(sigma_scm - true_cov, ord="fro")
    tyler_error = np.linalg.norm(sigma_tyler - true_cov, ord="fro")

    assert tyler_error < scm_error, "Tyler's estimator should have lower error than SCM."


@pytest.mark.parametrize("max_iter", [10, 50, 100])
@pytest.mark.parametrize("tol", [1e-3, 1e-4, 1e-5])
def test_tyler_covariance(heavytail_data, max_iter, tol):
    """Test Tyler's covariance estimator on heavy-tailed data.

    Parameters
    ----------
    heavytail_data : np.ndarray
        Data generated from a multivariate t-distribution.
    max_iter : int
        Maximum number of iterations for the Tyler's estimator.
    tol : float
        Tolerance for convergence.
    """ 
    sigma_estimated = tyler_covariance(heavytail_data, max_iter=max_iter, tol=tol)
    assert sigma_estimated.shape[0] == sigma_estimated.shape[1], "Covariance matrix is not square."
    assert sigma_estimated.shape[0] == heavytail_data.shape[1], "Covariance matrix does not match data dimensions."
    assert np.all(np.isfinite(sigma_estimated)), "Covariance matrix contains non-finite values."
    assert np.allclose(sigma_estimated, sigma_estimated.T), "Covariance matrix is not symmetric."    
    # Check if the estimated covariance is almost symmetric
    assert np.allclose(sigma_estimated, sigma_estimated.T, rtol=1e-2), (
        "Covariance matrix is not symmetric within tolerance."
    )
    # Check if the covariance matrix is positive definite
    assert _check_positive_definite(sigma_estimated), "Covariance matrix is not positive definite."

def test_median_absolute_deviation(heavytail_data):
    """Test the median absolute deviation (MAD) function."""
    mad = _median_absolute_deviation(heavytail_data)
    assert mad.shape[0] == heavytail_data.shape[1], "MAD vector does not match feature dimensions."
    assert np.all(np.isfinite(mad)), "MAD vector contains non-finite values."
