import pytest
from active_memory_leak_detector import ActiveMemoryLeakDetector

def test_set_baseline_and_snapshot():
    detector = ActiveMemoryLeakDetector()
    detector.set_baseline()
    snap = detector.take_snapshot("test")
    assert "rss_mb" in snap
    assert "timestamp" in snap

def test_analyze_memory_patterns_insufficient():
    detector = ActiveMemoryLeakDetector()
    result = detector.analyze_memory_patterns()
    assert "error" in result

def test_generate_report():
    detector = ActiveMemoryLeakDetector()
    detector.set_baseline()
    detector.take_snapshot("test")
    report = detector.generate_report()
    assert "analysis" in report
    assert "recommendations" in report
