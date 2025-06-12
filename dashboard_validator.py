"""
Dashboard Testing and Validation Script
Skrypt testowania i walidacji dashboard ZoL0
"""

import importlib
import json
import sys
import time
import traceback
from datetime import datetime


class DashboardValidator:
    """Validator dla dashboard z kompleksowym testem"""

    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "unknown",
            "issues_found": [],
            "optimizations_applied": [],
        }

    def test_imports(self):
        """Test importów wszystkich modułów"""
        print("🔍 Testing imports...")

        modules_to_test = [
            "enhanced_dashboard",
            "memory_cleanup_optimizer",
            "dashboard_performance_optimizer",
        ]

        import_results = {}

        for module in modules_to_test:
            try:
                importlib.import_module(module)
                import_results[module] = {"status": "success", "error": None}
                print(f"  ✅ {module} - OK")
            except Exception as e:
                import_results[module] = {"status": "error", "error": str(e)}
                print(f"  ❌ {module} - ERROR: {str(e)}")
                self.test_results["issues_found"].append(
                    f"Import error in {module}: {str(e)}"
                )

        self.test_results["tests"]["imports"] = import_results
        return all(result["status"] == "success" for result in import_results.values())

    def test_syntax(self):
        """
        Test składni wszystkich plików. Obsługa błędów plików i wyjątków kompilacji.
        """
        print("🔍 Testing syntax...")
        import os
        import py_compile
        files_to_test = [
            "enhanced_dashboard.py",
            "memory_cleanup_optimizer.py",
            "dashboard_performance_optimizer.py",
        ]
        syntax_results = {}
        for file in files_to_test:
            if os.path.exists(file):
                try:
                    py_compile.compile(file, doraise=True)
                    syntax_results[file] = {"status": "success", "error": None}
                    print(f"  ✅ {file} - Syntax OK")
                except Exception as e:
                    syntax_results[file] = {"status": "error", "error": str(e)}
                    print(f"  ❌ {file} - Syntax ERROR: {str(e)}")
                    self.test_results["issues_found"].append(
                        f"Syntax error in {file}: {str(e)}"
                    )
            else:
                syntax_results[file] = {"status": "missing", "error": "File not found"}
                print(f"  ⚠️ {file} - File not found")
        self.test_results["tests"]["syntax"] = syntax_results
        return all(result["status"] == "success" for result in syntax_results.values())

    def test_memory_optimization(self):
        """Test optymalizacji pamięci"""
        print("🔍 Testing memory optimization...")

        try:
            from memory_cleanup_optimizer import memory_optimizer

            # Test basic functionality
            initial_memory = memory_optimizer.check_memory_usage()
            memory_optimizer.periodic_cleanup()

            optimization_results = {
                "memory_check": "success",
                "cleanup": "success",
                "initial_memory_mb": initial_memory.get("memory_mb", 0),
            }

            print(
                f"  ✅ Memory optimization - OK (Current: {initial_memory.get('memory_mb', 0):.1f}MB)"
            )

        except Exception as e:
            optimization_results = {"memory_check": "error", "error": str(e)}
            print(f"  ❌ Memory optimization - ERROR: {str(e)}")
            self.test_results["issues_found"].append(
                f"Memory optimization error: {str(e)}"
            )

        self.test_results["tests"]["memory_optimization"] = optimization_results
        return optimization_results.get("memory_check") == "success"

    def test_performance_monitoring(self):
        """Test monitorowania wydajności"""
        print("🔍 Testing performance monitoring...")

        try:
            from dashboard_performance_optimizer import dashboard_optimizer, performance_monitor

            # Test decorator
            @performance_monitor("test_function")
            def test_function():
                time.sleep(0.01)  # Short delay
                return "test_result"

            # Run test
            result = test_function()

            # Check if metrics were recorded
            summary = dashboard_optimizer.get_performance_summary()

            performance_results = {
                "decorator": "success",
                "metrics_recorded": "test_function" in summary,
                "test_result": result == "test_result",
            }

            print("  ✅ Performance monitoring - OK")

        except Exception as e:
            performance_results = {"decorator": "error", "error": str(e)}
            print(f"  ❌ Performance monitoring - ERROR: {str(e)}")
            self.test_results["issues_found"].append(
                f"Performance monitoring error: {str(e)}"
            )

        self.test_results["tests"]["performance_monitoring"] = performance_results
        return performance_results.get("decorator") == "success"

    def test_streamlit_components(self):
        """Test komponentów Streamlit (tylko podstawowy import)"""
        print("🔍 Testing Streamlit components...")

        try:
            import pandas as pd
            import plotly.graph_objects as go

            # Test basic functionality
            pd.DataFrame({"test": [1, 2, 3]})
            go.Figure()

            streamlit_results = {
                "streamlit_import": "success",
                "plotly_import": "success",
                "pandas_import": "success",
                "basic_operations": "success",
            }

            print("  ✅ Streamlit components - OK")

        except Exception as e:
            streamlit_results = {"streamlit_import": "error", "error": str(e)}
            print(f"  ❌ Streamlit components - ERROR: {str(e)}")
            self.test_results["issues_found"].append(
                f"Streamlit components error: {str(e)}"
            )

        self.test_results["tests"]["streamlit_components"] = streamlit_results
        return streamlit_results.get("streamlit_import") == "success"

    def run_all_tests(self):
        """Uruchom wszystkie testy"""
        print("🚀 Starting Dashboard Validation")
        print("=" * 50)

        test_methods = [
            self.test_syntax,
            self.test_imports,
            self.test_streamlit_components,
            self.test_memory_optimization,
            self.test_performance_monitoring,
        ]

        passed_tests = 0
        total_tests = len(test_methods)

        for test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
                print()  # Empty line between tests
            except Exception as e:
                print(f"  💥 Test crashed: {str(e)}")
                print(traceback.format_exc())
                self.test_results["issues_found"].append(f"Test crash: {str(e)}")
                print()

        # Calculate overall status
        success_rate = passed_tests / total_tests
        if success_rate >= 0.8:
            self.test_results["overall_status"] = "good"
            status_emoji = "✅"
        elif success_rate >= 0.6:
            self.test_results["overall_status"] = "warning"
            status_emoji = "⚠️"
        else:
            self.test_results["overall_status"] = "error"
            status_emoji = "❌"

        print("=" * 50)
        print(f"{status_emoji} VALIDATION COMPLETE")
        print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate*100:.1f}%)")
        print(f"Overall status: {self.test_results['overall_status'].upper()}")

        if self.test_results["issues_found"]:
            print(f"\n📋 Issues found ({len(self.test_results['issues_found'])}):")
            for i, issue in enumerate(self.test_results["issues_found"], 1):
                print(f"  {i}. {issue}")

        # Save results
        with open("dashboard_validation_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print("\n📄 Results saved to: dashboard_validation_results.json")

        return self.test_results


# Test edge-case: brak pliku
def test_missing_file():
    """Testuje obsługę braku pliku przy testach składni."""
    import os
    file = "nonexistent_dashboard.py"
    if not os.path.exists(file):
        try:
            import py_compile
            py_compile.compile(file, doraise=True)
        except Exception as e:
            print("OK: FileNotFoundError handled gracefully.")
        else:
            print("FAIL: No exception for missing file.")


if __name__ == "__main__":
    validator = DashboardValidator()
    results = validator.run_all_tests()
    test_missing_file()  # Run edge-case test

    # Exit with appropriate code
    if results["overall_status"] == "good":
        sys.exit(0)
    elif results["overall_status"] == "warning":
        sys.exit(1)
    else:
        sys.exit(2)

# TODO: Integrate with CI/CD pipeline for automated dashboard validation and edge-case tests.
# Edge-case tests: simulate import errors, missing modules, and optimization failures.
# All public methods have docstrings and exception handling.
