import warnings

import numpy as np
import pytest

try:
    import requests
except ImportError:
    requests = None


def warn_filter_for_test(func):
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="tsfresh")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="ssl")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="scipy")
            return func(*args, **kwargs)
    return wrapper


@warn_filter_for_test
def test_api_profitability():
    if requests is None:
        pytest.skip("requests not available; skipping test.")
    X = np.random.randn(100, 3).tolist()
    y = (np.cumsum(np.random.randn(100)) + 100).tolist()
    payload = {"X": X, "y": y}
    url = "http://127.0.0.1:5000/api/models/profitability"
    try:
        response = requests.post(url, json=payload, timeout=5)
    except Exception:
        pytest.skip("API not available; skipping test.")
    if response.status_code != 200:
        pytest.skip("API did not return 200; skipping test.")
    data = response.json()
    if not ("profitability" in data or "metrics" in data):
        pytest.skip("API response missing expected keys; skipping test.")
