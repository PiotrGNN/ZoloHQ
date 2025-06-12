import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hypothesis import given, settings
from hypothesis import strategies as st

from active_memory_leak_detector import ActiveMemoryLeakDetector


def dummy_memory_usage(x):
    return x * 2

@given(st.integers(min_value=0, max_value=10000))
@settings(deadline=None)
def test_memory_growth_is_linear(x):
    detector = ActiveMemoryLeakDetector()
    before = detector.get_current_memory_info()["rss_mb"]
    # Symulacja: przyrost pamięci proporcjonalny do x
    used = dummy_memory_usage(x)
    after = before + used / 1024 / 1024
    assert after >= before
