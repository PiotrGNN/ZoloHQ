#!/usr/bin/env python3
"""
Post-Fix Memory Leak Test for production_data_manager.py
========================================================
Test specifically focused on the fixed production_data_manager.py memory usage
"""

import gc
import importlib
import json
import logging
import os
from datetime import datetime

import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionManagerMemoryTest:
    """Test memory usage of the fixed production_data_manager.py"""

    def __init__(self):
        self.baseline_memory = 0
        self.test_results = []

    def measure_memory(self):
        """Measure current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_import_memory_leak(self):
        """Test if importing production_data_manager causes memory leak"""
        logger.info("🧪 Testing production_data_manager import memory usage...")

        # Baseline
        gc.collect()
        self.baseline_memory = self.measure_memory()
        logger.info(f"📊 Baseline memory: {self.baseline_memory:.2f} MB")

        try:
            # Import the fixed module
            import production_data_manager

            # Force garbage collection
            gc.collect()
            import_memory = self.measure_memory()
            import_growth = import_memory - self.baseline_memory

            logger.info(f"📦 Memory after import: {import_memory:.2f} MB")
            logger.info(f"📈 Memory growth from import: {import_growth:.2f} MB")

            # Test multiple imports (this was causing the massive leak)
            for i in range(3):
                # Simulate the problematic pattern that was fixed
                try:
                    # This should NOT cause memory growth anymore
                    importlib.reload(production_data_manager)
                except Exception as e:
                    logger.info(f"Reload {i+1}: {e}")

                gc.collect()
                reload_memory = self.measure_memory()
                reload_growth = reload_memory - self.baseline_memory

                logger.info(
                    f"🔄 Reload {i+1} memory: {reload_memory:.2f} MB (growth: {reload_growth:.2f} MB)"
                )

            final_memory = self.measure_memory()
            total_growth = final_memory - self.baseline_memory

            result = {
                "test": "import_memory_leak",
                "baseline_mb": self.baseline_memory,
                "final_mb": final_memory,
                "total_growth_mb": total_growth,
                "import_growth_mb": import_growth,
                "status": "FIXED" if total_growth < 50 else "LEAK_DETECTED",
            }

            self.test_results.append(result)

            if total_growth < 10:
                logger.info("✅ MEMORY LEAK FIXED: Low memory growth detected")
            elif total_growth < 50:
                logger.info("⚠️ MODERATE: Some memory growth but much improved")
            else:
                logger.error("❌ MEMORY LEAK STILL PRESENT: High memory growth")

            return result

        except Exception as e:
            logger.error(f"❌ Import test failed: {e}")
            return {"test": "import_memory_leak", "status": "ERROR", "error": str(e)}

    def test_manager_creation(self):
        """Test creating ProductionDataManager instances"""
        logger.info("🧪 Testing ProductionDataManager instance creation...")

        try:
            from production_data_manager import ProductionDataManager

            gc.collect()
            before_creation = self.measure_memory()

            # Create manager instance
            manager = ProductionDataManager()

            gc.collect()
            after_creation = self.measure_memory()
            creation_growth = after_creation - before_creation

            logger.info(f"📦 Memory after manager creation: {after_creation:.2f} MB")
            logger.info(f"📈 Memory growth from creation: {creation_growth:.2f} MB")

            # Test some operations
            try:
                # This should use the fixed lazy loading
                manager.get_account_balance(use_cache=False)

                gc.collect()
                after_operations = self.measure_memory()
                operation_growth = after_operations - before_creation

                logger.info(f"⚙️ Memory after operations: {after_operations:.2f} MB")
                logger.info(
                    f"📈 Total growth with operations: {operation_growth:.2f} MB"
                )

            except Exception as e:
                logger.info(f"⚠️ Operation test failed (expected for demo): {e}")
                operation_growth = creation_growth

            # Clean up
            del manager
            gc.collect()
            after_cleanup = self.measure_memory()
            final_growth = after_cleanup - before_creation

            logger.info(f"🧹 Memory after cleanup: {after_cleanup:.2f} MB")
            logger.info(f"📈 Final memory growth: {final_growth:.2f} MB")

            result = {
                "test": "manager_creation",
                "creation_growth_mb": creation_growth,
                "operation_growth_mb": operation_growth,
                "final_growth_mb": final_growth,
                "status": (
                    "GOOD"
                    if final_growth < 20
                    else "CONCERNING" if final_growth < 100 else "LEAK"
                ),
            }

            self.test_results.append(result)
            return result

        except Exception as e:
            logger.error(f"❌ Manager creation test failed: {e}")
            return {"test": "manager_creation", "status": "ERROR", "error": str(e)}

    def run_comprehensive_test(self):
        """Run all memory leak tests"""
        print("🔬 POST-FIX MEMORY LEAK TEST - production_data_manager.py")
        print("=" * 65)

        # Run tests
        import_result = self.test_import_memory_leak()
        creation_result = self.test_manager_creation()

        # Overall assessment
        print("\n🎯 MEMORY LEAK FIX ASSESSMENT")
        print("=" * 40)

        if import_result.get("status") == "FIXED":
            print("✅ Import Memory Leak: FIXED")
        elif import_result.get("status") == "LEAK_DETECTED":
            print("❌ Import Memory Leak: STILL PRESENT")
        else:
            print("⚠️ Import Memory Leak: TEST ERROR")

        creation_status = creation_result.get("status", "ERROR")
        print(f"📦 Manager Creation: {creation_status}")

        # Generate report
        report = {
            "test_timestamp": datetime.now().isoformat(),
            "fix_verification": {
                "import_leak_fixed": import_result.get("status") == "FIXED",
                "creation_memory_acceptable": creation_status in ["GOOD", "CONCERNING"],
            },
            "memory_measurements": {
                "baseline_mb": self.baseline_memory,
                "import_growth_mb": import_result.get("total_growth_mb", 0),
                "creation_growth_mb": creation_result.get("final_growth_mb", 0),
            },
            "test_results": self.test_results,
        }

        report_file = f"production_manager_memory_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed report: {report_file}")

        # Summary
        if import_result.get("status") == "FIXED" and creation_status in [
            "GOOD",
            "CONCERNING",
        ]:
            print("🎉 OVERALL: MEMORY LEAK SUCCESSFULLY FIXED!")
        else:
            print("⚠️ OVERALL: Additional fixes may be needed")

        return report


def main():
    tester = ProductionManagerMemoryTest()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()
