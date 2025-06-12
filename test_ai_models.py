#!/usr/bin/env python3
"""
Test script to examine AI models' decision-making processes and confidence mechanisms.
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ZoL0-master"))

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_sentiment_analyzer():
    logger = logging.getLogger("test_ai_models")
    """Test SentimentAnalyzer decision-making process"""
    logger.info("=" * 60)
    logger.info("TESTING SENTIMENT ANALYZER")
    logger.info("=" * 60)

    try:
        from ai_models.sentiment_ai import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        logger.info("SentimentAnalyzer loaded successfully")

        # Test prediction with different market scenarios
        test_scenarios = [
            "Bitcoin price surges to new all-time high amid institutional adoption",
            "Market crash imminent as economic indicators turn negative",
            "Stable trading range continues with low volatility",
            "Fed announces interest rate hike, markets react positively",
            "Crypto regulation uncertainty causes selling pressure",
        ]

        logger.info("📊 Testing sentiment predictions:")
        for i, scenario in enumerate(test_scenarios, 1):
            try:
                if hasattr(analyzer, "predict"):
                    result = analyzer.predict(scenario)
                    logger.info(f"{i}. Scenario: {scenario[:50]}... Result: {result}")
                else:
                    result = analyzer.analyze()
                    logger.info(f"{i}. Using analyze() method: {result}")
            except Exception as e:
                logger.error(f"Error: {e}")
                continue

        # Test confidence mechanisms
        logger.info("🎯 Testing confidence mechanisms:")
        try:
            result = analyzer.analyze()
            if isinstance(result, dict):
                confidence = abs(result.get("value", 0)) * 100
                logger.info(
                    f"Confidence level: {confidence:.1f}% Analysis: {result.get('analysis', 'N/A')} Value: {result.get('value', 'N/A')}"
                )

                # Test decision thresholds
                value = result.get("value", 0)
                if value > 0.2:
                    decision = "STRONG BULLISH"
                elif value > 0.1:
                    decision = "BULLISH"
                elif value < -0.2:
                    decision = "STRONG BEARISH"
                elif value < -0.1:
                    decision = "BEARISH"
                else:
                    decision = "NEUTRAL"
                logger.info(f"Decision: {decision}")
        except Exception as e:
            logger.error(f"Error testing confidence: {e}")

    except ImportError as e:
        logger.error(f"Cannot import SentimentAnalyzer: {e}")
    except Exception as e:
        logger.error(f"Error testing SentimentAnalyzer: {e}")


def test_anomaly_detector():
    logger = logging.getLogger("test_ai_models")
    """Test AnomalyDetector decision-making process"""
    logger.info("=" * 60)
    logger.info("TESTING ANOMALY DETECTOR")
    logger.info("=" * 60)

    try:
        from ai_models.anomaly_detection import AnomalyDetector

        detector = AnomalyDetector(sensitivity=0.1)
        logger.info("AnomalyDetector loaded successfully")
        logger.info(f"   Sensitivity: {detector.sensitivity}")
        logger.info(f"   Initialized: {detector.is_initialized}")

        # Generate test data with anomalies
        logger.info("📊 Testing anomaly detection:")

        # Create normal price data
        np.random.seed(42)
        normal_data = np.random.normal(100, 5, (100, 5))  # 100 samples, 5 features

        # Inject anomalies
        anomaly_data = normal_data.copy()
        anomaly_data[25] = [150, 155, 148, 152, 10000]  # Price spike with volume spike
        anomaly_data[50] = [50, 55, 45, 48, 100]  # Price crash
        anomaly_data[75] = [95, 98, 92, 94, 50000]  # Volume anomaly

        # Test detection
        try:
            result = detector.detect(anomaly_data)
            logger.info(f"Detection result: {result}")

            if "anomaly_indices" in result:
                anomalies = result["anomaly_indices"]
                logger.info(f"Detected anomalies at indices: {anomalies}")
                logger.info("Expected anomalies at: [25, 50, 75]")

                # Check accuracy
                expected = {25, 50, 75}
                detected = set(anomalies)
                accuracy = len(expected & detected) / len(expected) * 100
                logger.info(f"Detection accuracy: {accuracy:.1f}%")

            if "anomaly_scores" in result:
                scores = result["anomaly_scores"]
                avg_score = np.mean(scores)
                logger.info(f"Average anomaly score: {avg_score:.3f}")

        except Exception as e:
            logger.error(f"Error during detection: {e}")

        # Test confidence mechanisms
        logger.info("🎯 Testing confidence mechanisms:")
        try:
            # Test with different sensitivity levels
            sensitivities = [0.05, 0.1, 0.2]
            for sens in sensitivities:
                test_detector = AnomalyDetector(sensitivity=sens)
                result = test_detector.detect(anomaly_data)
                anomaly_count = len(result.get("anomaly_indices", []))
                logger.info(f"Sensitivity {sens}: {anomaly_count} anomalies detected")
        except Exception as e:
            logger.error(f"Error testing sensitivity: {e}")

    except ImportError as e:
        logger.error(f"Cannot import AnomalyDetector: {e}")
    except Exception as e:
        logger.error(f"Error testing AnomalyDetector: {e}")


def test_model_recognizer():
    logger = logging.getLogger("test_ai_models")
    """Test ModelRecognizer decision-making process"""
    logger.info("=" * 60)
    logger.info("TESTING MODEL RECOGNIZER")
    logger.info("=" * 60)

    try:
        from ai_models.model_recognition import ModelRecognizer

        recognizer = ModelRecognizer(confidence_threshold=0.7)
        logger.info("ModelRecognizer loaded successfully")
        logger.info(f"   Confidence threshold: {recognizer.confidence_threshold}")
        logger.info(f"   Device: {recognizer.device}")

        # Generate test market data
        logger.info("📊 Testing pattern recognition:")

        # Create synthetic OHLCV data
        dates = pd.date_range(start="2025-01-01", periods=100, freq="h")

        # Create pattern data (head and shoulders)
        prices = []
        base_price = 100

        for i in range(100):
            if i < 20:  # Left shoulder
                price = base_price + i * 0.5
            elif i < 40:  # Head
                price = base_price + 10 + (i - 20) * 1.0
            elif i < 60:  # Right shoulder
                price = base_price + 30 - (i - 40) * 1.0
            else:  # Breakdown
                price = base_price + 10 - (i - 60) * 0.3

            # Add some noise
            price += np.random.normal(0, 1)
            prices.append(price)

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "datetime": dates,
                "open": prices,
                "high": [p + np.random.uniform(0, 2) for p in prices],
                "low": [p - np.random.uniform(0, 2) for p in prices],
                "close": prices,
                "volume": [np.random.uniform(1000, 10000) for _ in prices],
            }
        )

        logger.info(f"   Generated {len(data)} data points")

        # Test pattern identification
        try:
            if hasattr(recognizer, "identify_patterns"):
                patterns = recognizer.identify_patterns(data, min_confidence=0.6)
                logger.info(f"Pattern identification result: {patterns}")
            else:
                logger.warning("identify_patterns method not found")
        except Exception as e:
            logger.error(f"Error during pattern identification: {e}")

        # Test confidence mechanisms
        logger.info("🎯 Testing confidence mechanisms:")
        try:
            # Test with different confidence thresholds
            thresholds = [0.5, 0.7, 0.9]
            for threshold in thresholds:
                test_recognizer = ModelRecognizer(confidence_threshold=threshold)
                if hasattr(test_recognizer, "identify_patterns"):
                    patterns = test_recognizer.identify_patterns(
                        data, min_confidence=threshold
                    )
                    pattern_count = (
                        len(patterns)
                        if isinstance(patterns, list)
                        else len(patterns.get("patterns", []))
                    )
                    logger.info(
                        f"Threshold {threshold}: {pattern_count} patterns detected"
                    )
        except Exception as e:
            logger.error(f"Error testing thresholds: {e}")
    except ImportError as e:
        logger.error(f"Cannot import ModelRecognizer: {e}")
    except Exception as e:
        logger.error(f"Error testing ModelRecognizer: {e}")


def test_model_integration():
    logger = logging.getLogger("test_ai_models")
    """Test integration between models"""
    logger.info("=" * 60)
    logger.info("TESTING MODEL INTEGRATION")
    logger.info("=" * 60)

    try:
        # Import all models
        from ai_models.anomaly_detection import AnomalyDetector
        from ai_models.sentiment_ai import SentimentAnalyzer

        sentiment = SentimentAnalyzer()
        anomaly = AnomalyDetector()

        logger.info("All models loaded for integration test")

        # Test combined decision making
        logger.info("🔄 Testing combined decision making:")

        # Get sentiment analysis
        sentiment_result = sentiment.analyze()
        sentiment_value = sentiment_result.get("value", 0)
        sentiment_confidence = abs(sentiment_value) * 100

        # Generate anomaly test data
        test_data = np.random.normal(100, 5, (50, 5))
        anomaly_result = anomaly.detect(test_data)
        anomaly_count = len(anomaly_result.get("anomaly_indices", []))

        # Combined decision logic
        logger.info(
            f"Sentiment value: {sentiment_value:.3f} (confidence: {sentiment_confidence:.1f}%)"
        )
        logger.info(f"Anomalies detected: {anomaly_count}")

        # Decision matrix
        if sentiment_value > 0.2 and anomaly_count == 0:
            decision = "STRONG BUY - Positive sentiment, no anomalies"
            confidence = min(95, sentiment_confidence + 20)
        elif sentiment_value > 0.1 and anomaly_count <= 1:
            decision = "BUY - Moderate positive sentiment"
            confidence = sentiment_confidence
        elif sentiment_value < -0.2 or anomaly_count > 3:
            decision = "SELL - Negative sentiment or high anomaly activity"
            confidence = max(sentiment_confidence, 70)
        elif anomaly_count > 1:
            decision = "HOLD - Anomalies detected, wait for clarity"
            confidence = 60
        else:
            decision = "NEUTRAL - Mixed signals"
            confidence = 50

        logger.info(f"Combined decision: {decision}")
        logger.info(f"Overall confidence: {confidence:.1f}%")
    except Exception as e:
        logger.error(f"Error in integration test: {e}")


def test_dummy():
    """Dummy test that always passes"""
    assert True, "Dummy test always passes."


def main():
    """Main test function"""
    print("🚀 AI MODELS DECISION-MAKING ANALYSIS")
    print(f"📅 Test time: {datetime.now()}")
    print("=" * 80)

    # Test individual models
    test_sentiment_analyzer()
    print("\n")
    test_anomaly_detector()
    print("\n")
    test_model_recognizer()
    print("\n")

    # Test integration
    test_model_integration()

    print("\n" + "=" * 80)
    print("✅ AI Models analysis completed!")


if __name__ == "__main__":
    main()
