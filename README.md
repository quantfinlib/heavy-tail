# heavytail

[![codecov](https://codecov.io/gh/quantfinlib/heavy-tail/graph/badge.svg?token=Z60B2PYJ44)](https://codecov.io/gh/quantfinlib/heavy-tail)
[![tests](https://github.com/quantfinlib/heavy-tail/actions/workflows/test.yml/badge.svg)](https://github.com/quantfinlib/heavy-tail/actions/workflows/test.yml)
[![docs](https://github.com/quantfinlib/heavy-tail/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/quantfinlib/heavy-tail/actions/workflows/gh-pages.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/quantfinlib/heavy-tail/blob/main/LICENSE)




Covariance estimation in the presence of fat tails and outliers


## Documentation

For detailed description of the API and usage examples, please visit our most up-to-date [documentation](https://quantfinlib.github.io/heavy-tail/).


## Installation

### Install the package locally

```bash
$ uv pip install -e .[docs, tests, dev]
```

### Install the package from PyPI

```bash
```


## Usage

```python
from heavytail.tyler import tyler_covariance
tyler_cov = tyler_covariance(data=data, max_iter=100, tol=1e-6)
```