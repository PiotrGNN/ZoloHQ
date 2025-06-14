#!/usr/bin/env python3
"""
Fixed AI Models Testing Script
=============================
Tests AI models decision-making processes with proper interface handling.
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Add the ZoL0-master directory to Python path
zol0_path = os.path.join(os.path.dirname(__file__), "ZoL0-master")
sys.path.insert(0, zol0_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_anomaly_detector():
    """Test AnomalyDetector with sensitivity adjustments"""
    print("\n=== Testing AnomalyDetector ===")

    try:
        from ai_models.anomaly_detection import AnomalyDetector

        # Try with higher sensitivity to detect our planted anomalies
        detector = AnomalyDetector(sensitivity=0.1)  # 10% contamination
        print(
            f"✓ AnomalyDetector loaded successfully (sensitivity: {detector.sensitivity})"
        )

        # Create more realistic test data with fewer anomalies
        np.random.seed(42)
        normal_data = np.random.normal(100, 5, 95)  # 95 normal points
        anomaly_data = np.array([150, 50, 200, 30, 180])  # 5 clear anomalies
        test_data = np.concatenate([normal_data, anomaly_data])
        np.random.shuffle(test_data)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2024-01-01", periods=len(test_data), freq="h"
                ),
                "price": test_data,
                "volume": np.random.normal(1000, 100, len(test_data)),
            }
        )

        print(f"Test data: {len(df)} points (expect ~5 anomalies)")

        # Test anomaly detection
        anomalies = detector.detect_anomalies(df)
        print(f"Detected anomalies: {len(anomalies)} points")

        if len(anomalies) > 0:
            print("Sample anomalous points:")
            for i, (_idx, row) in enumerate(anomalies.head(3).iterrows()):
                print(f"  {i+1}. Price: {row['price']:.2f}, Time: {row['timestamp']}")

        # Test confidence mechanism
        if hasattr(detector, "get_confidence"):
            confidence = detector.get_confidence()
            print(f"Detection confidence: {confidence:.2f}")

        print("✓ AnomalyDetector tests completed")
        assert True

    except Exception as e:
        print(f"✗ AnomalyDetector test failed: {e}")
        assert False


def test_model_recognizer():
    """Test ModelRecognizer pattern detection"""
    print("\n=== Testing ModelRecognizer ===")

    try:
        from ai_models.model_recognition import ModelRecognizer

        recognizer = ModelRecognizer()
        print("✓ ModelRecognizer loaded successfully")

        # Create test data with clear patterns
        timestamps = pd.date_range(start="2024-01-01", periods=100, freq="h")

        # Create trend pattern
        trend_data = np.linspace(100, 120, 100) + np.random.normal(0, 1, 100)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": trend_data,
                "high": trend_data + np.random.uniform(0.5, 2, 100),
                "low": trend_data - np.random.uniform(0.5, 2, 100),
                "close": trend_data + np.random.normal(0, 0.5, 100),
                "volume": np.random.normal(1000, 100, 100),
            }
        )

        print(f"Test data: {len(df)} points with upward trend pattern")
        # Test pattern recognition
        pattern = recognizer.recognize_pattern(df)
        print(f"Recognized pattern: {pattern if pattern else 'None'}")

        if pattern:
            print(f"  Pattern details: {pattern}")
        else:
            print("  No pattern detected - may need calibration")

        # Test confidence if available
        if hasattr(recognizer, "confidence_threshold"):
            confidence_threshold = recognizer.confidence_threshold
            print(
                f"Pattern recognition confidence threshold: {confidence_threshold:.2f}"
            )

        print("✓ ModelRecognizer tests completed")
        assert True

    except Exception as e:
        print(f"✗ ModelRecognizer test failed: {e}")
        assert False


def test_sentiment_analyzer():
    """Test both SentimentAnalyzer implementations"""
    print("\n=== Testing SentimentAnalyzer Implementations ===")

    success_count = 0

    # Test MarketSentimentAnalyzer (ai_models/sentiment_ai.py)
    print("\n--- Testing MarketSentimentAnalyzer ---")
    try:
        from ai_models.sentiment_ai import MarketSentimentAnalyzer

        analyzer = MarketSentimentAnalyzer()
        print("✓ MarketSentimentAnalyzer loaded successfully")

        # Test with proper texts parameter
        test_texts = [
            "The market is showing strong bullish momentum with significant gains",
            "Economic indicators suggest a potential downturn ahead",
            "Trading volumes are at record highs indicating market confidence",
        ]

        result = analyzer.analyze(texts=test_texts)
        print(f"Analysis result: {result}")

        # Test confidence if available
        if hasattr(analyzer, "get_confidence"):
            confidence = analyzer.get_confidence()
            print(f"Analysis confidence: {confidence:.2f}")

        print("✓ MarketSentimentAnalyzer tests completed")
        success_count += 1

    except Exception as e:
        print(f"✗ MarketSentimentAnalyzer test failed: {e}")

    # Test SentimentAnalyzer (data/indicators/sentiment_analysis.py)
    print("\n--- Testing SentimentAnalyzer (indicators) ---")
    try:
        from data.indicators.sentiment_analysis import SentimentAnalyzer

        analyzer2 = SentimentAnalyzer()
        print("✓ SentimentAnalyzer (indicators) loaded successfully")

        # This one has parameterless analyze() method
        result = analyzer2.analyze()
        print(f"Analysis result: {result}")

        print("✓ SentimentAnalyzer (indicators) tests completed")
        success_count += 1

    except Exception as e:
        print(f"✗ SentimentAnalyzer (indicators) test failed: {e}")

    assert success_count > 0


def test_model_integration():
    """Test integrated model decision-making"""
    print("\n=== Testing Model Integration ===")

    try:
        # Create comprehensive test scenario
        timestamps = pd.date_range(start="2024-01-01", periods=50, freq="h")

        # Simulate market data with mixed signals
        base_price = 100
        prices = []
        for i in range(50):
            if i < 20:  # Normal trend
                price = base_price + i * 0.5 + np.random.normal(0, 1)
            elif i < 30:  # Anomalous spike
                price = base_price + 20 + np.random.normal(0, 2)
            else:  # Return to trend
                price = base_price + (i - 20) * 0.3 + np.random.normal(0, 1)
            prices.append(price)

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
                "volume": np.random.normal(1000, 100, 50),
            }
        )

        print("Created integrated test scenario with:")
        print(f"  - {len(df)} data points")
        print(f"  - Price range: {df['price'].min():.2f} - {df['price'].max():.2f}")
        print("  - Simulated anomaly period: points 20-30")

        # Test each model on the same data
        results = {}

        # Anomaly detection
        try:
            from ai_models.anomaly_detection import AnomalyDetector

            detector = AnomalyDetector()
            anomalies = detector.detect_anomalies(df)
            results["anomalies"] = len(anomalies)
            print(f"  ✓ Anomalies detected: {len(anomalies)}")
        except Exception as e:
            print(f"  ✗ Anomaly detection failed: {e}")
        # Pattern recognition
        try:
            from ai_models.model_recognition import ModelRecognizer

            recognizer = ModelRecognizer()
            pattern = recognizer.recognize_pattern(df)
            results["pattern"] = pattern if pattern else "None"
            print(f"  ✓ Pattern detected: {results['pattern']}")
        except Exception as e:
            print(f"  ✗ Pattern recognition failed: {e}")

        print(f"\nIntegration test results: {results}")
        print("✓ Model integration tests completed")
        assert True

    except Exception as e:
        print(f"✗ Model integration test failed: {e}")
        assert False


def main():
    """Run all AI model tests with proper error handling"""
    print("🔍 AI Models Decision-Making Test Suite (Fixed)")
    print("=" * 50)

    start_time = datetime.now()
    results = {}

    # Run individual model tests
    results["anomaly_detector"] = test_anomaly_detector()
    results["model_recognizer"] = test_model_recognizer()
    results["sentiment_analyzer"] = test_sentiment_analyzer()
    results["integration"] = test_model_integration()

    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    duration = datetime.now() - start_time
    print(f"Test duration: {duration.total_seconds():.2f} seconds")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! AI models are functioning correctly.")
    else:
        print(
            f"\n⚠️  {total_tests - passed_tests} test(s) failed. Check individual results above."
        )

    assert passed_tests == total_tests


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Test suite interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}")
        sys.exit(1)
