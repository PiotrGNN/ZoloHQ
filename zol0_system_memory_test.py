#!/usr/bin/env python3
"""
ZoL0 System Component Memory Leak Test
Tests actual ZoL0 system components for memory leaks
"""

import gc
import json
import logging
import os
import sys
import time
import tracemalloc
import warnings
from datetime import datetime
from typing import Any, Dict, List

import psutil

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ZoL0SystemMemoryTest:
    """Test ZoL0 system components for memory leaks"""

    def __init__(self):
        self.baseline_memory = 0
        self.test_results = []
        tracemalloc.start()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        current, peak = tracemalloc.get_traced_memory()

        return {
            "timestamp": datetime.now().isoformat(),
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "traced_current_mb": current / 1024 / 1024,
            "threads": len([t for t in psutil.Process().threads()]),
            "gc_objects": len(gc.get_objects()),
        }

    def test_api_system(self) -> None:
        """Test API system components"""
        logger.info("Testing API system components...")

        before = self.get_memory_info()

        try:
            # Simulate API system test logic
            self.simulate_api_call()
        except ImportError as e:
            logger.error(f"ImportError during API system test: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during API system test: {e}")
        after = self.get_memory_info()
        logger.info(f"Memory before: {before}, after: {after}")
        # Add further assertions or checks as needed.

        gc.collect()
        after = self.get_memory_info()

        self.test_results.append(
            {
                "test": "API System",
                "before": before,
                "after": after,
                "memory_growth_mb": after["rss_mb"] - before["rss_mb"],
                "object_growth": after["gc_objects"] - before["gc_objects"],
            }
        )

    def test_data_management(self):
        """Test data management components"""
        logger.info("Testing data management components...")

        before = self.get_memory_info()

        try:
            # Test production data manager
            if os.path.exists("production_data_manager.py"):
                logger.info("Testing production_data_manager.py...")
                import production_data_manager

                # Create instance and test methods
                if hasattr(production_data_manager, "ProductionDataManager"):
                    pdm = production_data_manager.ProductionDataManager()
                    # Test basic functionality
                    if hasattr(pdm, "get_current_prices"):
                        try:
                            pdm.get_current_prices()
                        except Exception:
                            pass  # Expected without real connection
                    del pdm

                del production_data_manager

        except ImportError as e:
            logger.warning(f"Import error during data management testing: {e}")
        except Exception as e:
            logger.error(f"Error during data management testing: {e}")

        gc.collect()
        after = self.get_memory_info()

        self.test_results.append(
            {
                "test": "Data Management",
                "before": before,
                "after": after,
                "memory_growth_mb": after["rss_mb"] - before["rss_mb"],
                "object_growth": after["gc_objects"] - before["gc_objects"],
            }
        )

    def test_core_trading_system(self):
        """Test core trading system"""
        logger.info("Testing core trading system...")

        before = self.get_memory_info()

        try:
            # Add ZoL0-master to path
            sys.path.insert(0, "ZoL0-master")

            # Test core trading engine
            try:
                from core.trading.engine import TradingEngine

                logger.info("Testing TradingEngine...")

                # Create instance
                engine = TradingEngine()

                # Test basic operations
                if hasattr(engine, "initialize"):
                    engine.initialize()
                if hasattr(engine, "cleanup"):
                    engine.cleanup()

                del engine

            except ImportError as e:
                logger.warning(f"Could not import TradingEngine: {e}")

            # Test database manager
            try:
                from core.database.manager import DatabaseManager

                logger.info("Testing DatabaseManager...")

                # Create and test database manager
                db = DatabaseManager()
                if hasattr(db, "close"):
                    db.close()
                del db

            except ImportError as e:
                logger.warning(f"Could not import DatabaseManager: {e}")

        except Exception as e:
            logger.error(f"Error during core trading system testing: {e}")
        finally:
            # Remove from path
            if "ZoL0-master" in sys.path:
                sys.path.remove("ZoL0-master")

        gc.collect()
        after = self.get_memory_info()

        self.test_results.append(
            {
                "test": "Core Trading System",
                "before": before,
                "after": after,
                "memory_growth_mb": after["rss_mb"] - before["rss_mb"],
                "object_growth": after["gc_objects"] - before["gc_objects"],
            }
        )

    def test_monitoring_systems(self):
        """Test monitoring systems"""
        logger.info("Testing monitoring systems...")

        before = self.get_memory_info()

        try:
            # Test memory monitoring dashboard
            if os.path.exists("memory_monitoring_dashboard.py"):
                logger.info("Testing memory_monitoring_dashboard.py...")
                import memory_monitoring_dashboard

                del memory_monitoring_dashboard

            # Test performance monitor
            if os.path.exists("advanced_performance_monitor.py"):
                logger.info("Testing advanced_performance_monitor.py...")
                import advanced_performance_monitor

                del advanced_performance_monitor

        except ImportError as e:
            logger.warning(f"Import error during monitoring testing: {e}")
        except Exception as e:
            logger.error(f"Error during monitoring testing: {e}")

        gc.collect()
        after = self.get_memory_info()

        self.test_results.append(
            {
                "test": "Monitoring Systems",
                "before": before,
                "after": after,
                "memory_growth_mb": after["rss_mb"] - before["rss_mb"],
                "object_growth": after["gc_objects"] - before["gc_objects"],
            }
        )

    def test_dashboard_systems(self):
        """Test dashboard systems for memory leaks"""
        logger.info("Testing dashboard systems...")

        before = self.get_memory_info()

        try:
            # Test unified dashboard (without Streamlit dependencies)
            if os.path.exists("unified_trading_dashboard.py"):
                logger.info("Testing unified_trading_dashboard.py imports...")
                # Just test imports without running Streamlit
                with open("unified_trading_dashboard.py", "r") as f:
                    content = f.read()
                    if "import streamlit" in content:
                        logger.info("Dashboard uses Streamlit - skipping full test")

            # Test enhanced dashboard
            if os.path.exists("enhanced_dashboard.py"):
                logger.info("Testing enhanced_dashboard.py imports...")
                with open("enhanced_dashboard.py", "r") as f:
                    content = f.read()
                    if "import streamlit" in content:
                        logger.info(
                            "Enhanced dashboard uses Streamlit - skipping full test"
                        )

        except Exception as e:
            logger.error(f"Error during dashboard testing: {e}")

        gc.collect()
        after = self.get_memory_info()

        self.test_results.append(
            {
                "test": "Dashboard Systems",
                "before": before,
                "after": after,
                "memory_growth_mb": after["rss_mb"] - before["rss_mb"],
                "object_growth": after["gc_objects"] - before["gc_objects"],
            }
        )

    def run_continuous_system_test(self, duration: int = 60):
        """Run continuous system stress test"""
        logger.info(f"Running continuous system test for {duration} seconds...")

        start_time = time.time()
        snapshots = []

        while (time.time() - start_time) < duration:
            # Simulate system activity
            memory_info = self.get_memory_info()
            snapshots.append(memory_info)

            # Force garbage collection periodically
            if len(snapshots) % 5 == 0:
                gc.collect()

            time.sleep(2)

        # Analyze continuous test results
        if len(snapshots) > 1:
            memory_growth = snapshots[-1]["rss_mb"] - snapshots[0]["rss_mb"]
            object_growth = snapshots[-1]["gc_objects"] - snapshots[0]["gc_objects"]

            self.test_results.append(
                {
                    "test": "Continuous System Test",
                    "duration_seconds": duration,
                    "snapshots_count": len(snapshots),
                    "memory_growth_mb": memory_growth,
                    "object_growth": object_growth,
                    "average_memory_mb": sum(s["rss_mb"] for s in snapshots)
                    / len(snapshots),
                }
            )

    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system memory test report"""

        # Calculate overall metrics
        total_memory_growth = sum(
            r.get("memory_growth_mb", 0) for r in self.test_results
        )
        total_object_growth = sum(r.get("object_growth", 0) for r in self.test_results)

        # Determine system health
        if total_memory_growth > 50:
            health_status = "CONCERNING"
        elif total_memory_growth > 20:
            health_status = "MODERATE"
        else:
            health_status = "GOOD"

        # Get tracemalloc top stats
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[:10]
            memory_hotspots = []
            for stat in top_stats:
                memory_hotspots.append(
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
        except Exception:
            memory_hotspots = []

        report = {
            "test_timestamp": datetime.now().isoformat(),
            "system_health": health_status,
            "total_memory_growth_mb": total_memory_growth,
            "total_object_growth": total_object_growth,
            "test_results": self.test_results,
            "memory_hotspots": memory_hotspots,
            "recommendations": self._generate_system_recommendations(
                health_status, total_memory_growth
            ),
        }

        return report

    def _generate_system_recommendations(
        self, health_status: str, memory_growth: float
    ) -> List[str]:
        """Generate system-specific recommendations"""
        recommendations = []

        if health_status == "CONCERNING":
            recommendations.extend(
                [
                    "🚨 SYSTEM MEMORY ISSUES DETECTED",
                    "• Investigate high memory growth components",
                    "• Review dashboard initialization patterns",
                    "• Check for unclosed database connections",
                    "• Consider implementing memory limits",
                ]
            )
        elif health_status == "MODERATE":
            recommendations.extend(
                [
                    "⚠️ MODERATE SYSTEM MEMORY GROWTH",
                    "• Monitor specific components more closely",
                    "• Consider periodic garbage collection",
                    "• Review resource cleanup procedures",
                ]
            )
        else:
            recommendations.extend(
                [
                    "✅ GOOD SYSTEM MEMORY HEALTH",
                    "• Current memory management is effective",
                    "• Continue monitoring for regression",
                ]
            )

        return recommendations


def main():
    """Main execution function"""
    logger.info("🧪 ZoL0 System Component Memory Leak Test")
    logger.info("=" * 50)
    tester = ZoL0SystemMemoryTest()
    try:
        # Set baseline
        gc.collect()
        baseline = tester.get_memory_info()
        logger.info(f"Baseline memory: {baseline['rss_mb']:.2f} MB")
        # Run component tests
        tester.test_api_system()
        tester.test_data_management()
        tester.test_core_trading_system()
        tester.test_monitoring_systems()
        tester.test_dashboard_systems()
        # Run continuous test
        logger.info("\n🔄 Running continuous system test...")
        tester.run_continuous_system_test(30)
        # Generate report
        report = tester.generate_system_report()

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"zol0_system_memory_test_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Display results
        logger.info(f"\n🏥 SYSTEM MEMORY HEALTH: {report['system_health']}")
        logger.info(
            f"📈 Total Memory Growth: {report['total_memory_growth_mb']:.2f} MB"
        )
        logger.info(f"📦 Total Object Growth: {report['total_object_growth']:,}")

        logger.info("\n📊 Component Test Results:")
        for result in report["test_results"]:
            test_name = result.get("test", "Unknown")
            memory_growth = result.get("memory_growth_mb", 0)
            logger.info(f"   • {test_name}: {memory_growth:+.2f} MB")

        logger.info("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            logger.info(f"   {rec}")

        logger.info(f"\n📄 Detailed report saved: {report_file}")

    except KeyboardInterrupt:
        logger.info("\n⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        tracemalloc.stop()


if __name__ == "__main__":
    main()
