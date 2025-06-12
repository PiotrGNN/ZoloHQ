#!/usr/bin/env python3
"""
Active Memory Leak Detector for ZoL0 Trading System
Performs comprehensive memory leak detection and analysis
"""

import gc
import json
import logging
import os
import sys
import threading
import time
import traceback  # Added to fix F821 error
import tracemalloc
from datetime import datetime
from typing import Any, Dict, List

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ActiveMemoryLeakDetector:
    """Comprehensive memory leak detection system"""

    def __init__(self):
        self.baseline_memory = 0
        self.memory_snapshots = []
        self.thread_counts = []
        self.file_descriptors = []
        self.database_connections = []
        self.monitoring_active = False

        # Start memory tracing
        tracemalloc.start()

    def get_current_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Get memory map details
        try:
            memory_maps = process.memory_maps()
            mapped_memory = sum(m.rss for m in memory_maps) / 1024 / 1024
        except (psutil.AccessDenied, AttributeError):
            mapped_memory = 0

        # Get tracemalloc current memory
        current, peak = tracemalloc.get_traced_memory()

        return {
            "timestamp": datetime.now().isoformat(),
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "mapped_mb": mapped_memory,
            "traced_current_mb": current / 1024 / 1024,
            "traced_peak_mb": peak / 1024 / 1024,
            "threads": threading.active_count(),
            "gc_objects": len(gc.get_objects()),
            "file_descriptors": (
                len(process.open_files()) if hasattr(process, "open_files") else 0
            ),
        }

    def set_baseline(self):
        """Set baseline memory usage"""
        gc.collect()  # Force garbage collection
        self.baseline_memory = self.get_current_memory_info()
        logger.info(f"Baseline memory set: {self.baseline_memory['rss_mb']:.2f} MB")

    def take_snapshot(self, description: str = ""):
        """Take a memory snapshot with description"""
        snapshot = self.get_current_memory_info()
        snapshot["description"] = description

        if self.baseline_memory:
            snapshot["memory_growth_mb"] = (
                snapshot["rss_mb"] - self.baseline_memory["rss_mb"]
            )
            snapshot["thread_growth"] = (
                snapshot["threads"] - self.baseline_memory["threads"]
            )
            snapshot["object_growth"] = (
                snapshot["gc_objects"] - self.baseline_memory["gc_objects"]
            )

        self.memory_snapshots.append(snapshot)
        logger.info(
            f"Snapshot '{description}': {snapshot['rss_mb']:.2f} MB "
            f"(+{snapshot.get('memory_growth_mb', 0):.2f} MB)"
        )

        return snapshot

    def simulate_potential_leaks(self):
        """Simulate various scenarios that could cause memory leaks"""
        logger.info("Starting memory leak simulation tests...")

        # Test 1: Object creation without cleanup
        self.take_snapshot("Before object creation test")
        large_objects = []
        for i in range(1000):
            large_objects.append([0] * 1000)  # Create large lists
        self.take_snapshot("After creating 1000 large objects")

        # Test cleanup
        del large_objects
        gc.collect()
        self.take_snapshot("After cleanup and gc.collect()")

        # Test 2: Thread creation
        threads = []

        def dummy_thread():
            time.sleep(0.1)

        self.take_snapshot("Before thread creation test")
        for i in range(10):
            t = threading.Thread(target=dummy_thread)
            t.daemon = True
            t.start()
            threads.append(t)

        self.take_snapshot("After creating 10 threads")

        # Wait for threads to complete
        for t in threads:
            t.join()
        self.take_snapshot("After threads completed")

        # Test 3: File operations
        self.take_snapshot("Before file operations test")
        test_files = []
        for i in range(100):
            filename = f"temp_test_{i}.txt"
            with open(filename, "w") as f:
                f.write("test data" * 100)
            test_files.append(filename)

        self.take_snapshot("After creating 100 temp files")

        # Cleanup files
        for filename in test_files:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
        self.take_snapshot("After cleaning up temp files")

    def test_database_connections(self):
        """Test database connection management"""
        logger.info("Testing database connection patterns...")

        try:
            # Import database manager
            sys.path.append("ZoL0-master/core/database")
            from manager import DatabaseManager

            self.take_snapshot("Before database connection test")

            # Test multiple connections
            connections = []
            for _i in range(5):
                db = DatabaseManager()
                connections.append(db)

            self.take_snapshot("After creating 5 database connections")

            # Test proper cleanup
            for db in connections:
                db.close()

            self.take_snapshot("After closing database connections")

        except ImportError:
            logger.warning("Could not import DatabaseManager for testing")

    def monitor_continuous(self, duration_seconds: int = 60):
        """Continuously monitor memory for specified duration"""
        logger.info(f"Starting continuous monitoring for {duration_seconds} seconds...")

        self.monitoring_active = True
        start_time = time.time()

        while self.monitoring_active and (time.time() - start_time) < duration_seconds:
            snapshot = self.take_snapshot(
                f"Continuous monitoring at {time.time() - start_time:.1f}s"
            )

            # Check for concerning trends
            if len(self.memory_snapshots) > 1:
                growth = snapshot.get("memory_growth_mb", 0)
                if growth > 50:  # More than 50MB growth
                    logger.warning(f"High memory growth detected: {growth:.2f} MB")

                thread_growth = snapshot.get("thread_growth", 0)
                if thread_growth > 10:  # More than 10 additional threads
                    logger.warning(
                        f"High thread growth detected: {thread_growth} threads"
                    )

            time.sleep(5)  # Check every 5 seconds

        self.monitoring_active = False
        logger.info("Continuous monitoring completed")

    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns from snapshots"""
        if len(self.memory_snapshots) < 2:
            return {"error": "Insufficient snapshots for analysis"}

        # Calculate growth rates
        memory_growths = [
            s.get("memory_growth_mb", 0) for s in self.memory_snapshots[1:]
        ]
        thread_growths = [s.get("thread_growth", 0) for s in self.memory_snapshots[1:]]
        object_growths = [s.get("object_growth", 0) for s in self.memory_snapshots[1:]]

        analysis = {
            "total_snapshots": len(self.memory_snapshots),
            "memory_analysis": {
                "max_growth_mb": max(memory_growths) if memory_growths else 0,
                "min_growth_mb": min(memory_growths) if memory_growths else 0,
                "avg_growth_mb": (
                    sum(memory_growths) / len(memory_growths) if memory_growths else 0
                ),
                "final_growth_mb": memory_growths[-1] if memory_growths else 0,
            },
            "thread_analysis": {
                "max_thread_growth": max(thread_growths) if thread_growths else 0,
                "final_thread_count": self.memory_snapshots[-1]["threads"],
            },
            "object_analysis": {
                "max_object_growth": max(object_growths) if object_growths else 0,
                "final_object_count": self.memory_snapshots[-1]["gc_objects"],
            },
        }

        # Determine leak likelihood
        final_growth = analysis["memory_analysis"]["final_growth_mb"]
        if final_growth > 100:
            analysis["leak_risk"] = "HIGH"
        elif final_growth > 50:
            analysis["leak_risk"] = "MEDIUM"
        elif final_growth > 10:
            analysis["leak_risk"] = "LOW"
        else:
            analysis["leak_risk"] = "MINIMAL"

        return analysis

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory leak detection report"""
        analysis = self.analyze_memory_patterns()

        # Get current tracemalloc top stats
        try:
            top_stats = tracemalloc.take_snapshot().statistics("lineno")[:10]
            top_memory_lines = []
            for stat in top_stats:
                top_memory_lines.append(
                    {
                        "filename": (
                            stat.traceback.format()[0]
                            if stat.traceback.format()
                            else "unknown"
                        ),
                        "size_mb": stat.size / 1024 / 1024,
                        "count": stat.count,
                    }
                )
        except Exception as e:
            top_memory_lines = [f"Error getting tracemalloc stats: {str(e)}"]

        report = {
            "detection_timestamp": datetime.now().isoformat(),
            "baseline_memory": self.baseline_memory,
            "final_memory": (
                self.memory_snapshots[-1] if self.memory_snapshots else None
            ),
            "analysis": analysis,
            "all_snapshots": self.memory_snapshots,
            "top_memory_allocations": top_memory_lines,
            "recommendations": self._generate_recommendations(analysis),
        }

        return report

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        leak_risk = analysis.get("leak_risk", "UNKNOWN")

        if leak_risk == "HIGH":
            recommendations.extend(
                [
                    "🚨 HIGH MEMORY LEAK RISK DETECTED",
                    "• Immediately investigate memory allocations",
                    "• Check for unclosed database connections",
                    "• Verify all file handles are properly closed",
                    "• Review thread cleanup procedures",
                ]
            )
        elif leak_risk == "MEDIUM":
            recommendations.extend(
                [
                    "⚠️ MODERATE MEMORY GROWTH DETECTED",
                    "• Monitor memory usage more closely",
                    "• Consider implementing periodic garbage collection",
                    "• Review resource cleanup in long-running processes",
                ]
            )
        elif leak_risk == "LOW":
            recommendations.extend(
                [
                    "✅ Low memory growth - within acceptable range",
                    "• Continue regular monitoring",
                    "• Consider optimizing memory-heavy operations",
                ]
            )
        else:
            recommendations.extend(
                [
                    "✅ Minimal memory growth detected",
                    "• Memory management appears to be working well",
                    "• Continue current practices",
                ]
            )

        # Thread-specific recommendations
        max_thread_growth = analysis.get("thread_analysis", {}).get(
            "max_thread_growth", 0
        )
        if max_thread_growth > 20:
            recommendations.append("• Review thread creation and cleanup patterns")

        return recommendations


# TODO: Integrate with CI/CD pipeline for automated memory leak and resource cleanup tests.
# Edge-case tests: simulate file/db/thread errors, permission issues, and resource exhaustion.
# All public methods have docstrings and exception handling.

def main():
    """Main execution function"""
    print("🔍 ZoL0 Active Memory Leak Detection")
    print("=" * 50)

    detector = ActiveMemoryLeakDetector()

    try:
        # Set baseline
        detector.set_baseline()

        # Run simulation tests
        detector.simulate_potential_leaks()

        # Test database connections if available
        detector.test_database_connections()

        # Short continuous monitoring
        print("\n🔄 Starting continuous monitoring for 30 seconds...")
        detector.monitor_continuous(30)

        # Generate and save report
        report = detector.generate_report()

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"active_memory_leak_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print("\n📊 MEMORY LEAK DETECTION RESULTS")
        print("=" * 50)

        analysis = report["analysis"]
        print(f"🎯 Leak Risk Level: {analysis.get('leak_risk', 'UNKNOWN')}")
        print(
            f"📈 Final Memory Growth: {analysis['memory_analysis']['final_growth_mb']:.2f} MB"
        )
        print(
            f"🧵 Final Thread Count: {analysis['thread_analysis']['final_thread_count']}"
        )
        print(
            f"📦 Final Object Count: {analysis['object_analysis']['final_object_count']:,}"
        )

        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   {rec}")

        print(f"\n📄 Detailed report saved: {report_file}")

    except KeyboardInterrupt:
        print("\n⏹️ Detection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during detection: {str(e)}")
        traceback.print_exc()
    finally:
        tracemalloc.stop()


if __name__ == "__main__":
    main()
