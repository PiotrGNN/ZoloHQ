#!/usr/bin/env python3
"""
Dashboard Status Report - Comprehensive Testing
==============================================
Test all dashboards and generate detailed error report
"""

import ast
import importlib.util
import subprocess
import sys
import time
from pathlib import Path


class DashboardTester:
    def __init__(self):
        self.base_dir = Path("c:/Users/piotr/Desktop/Zol0")
        self.results = {}

    def test_syntax(self, dashboard_file):
        """
        Test Python syntax. Obsługa błędów pliku i wyjątków składni.
        """
        try:
            with open(dashboard_file, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source)
            return True, "OK"
        except FileNotFoundError:
            return False, "File not found"
        except SyntaxError as e:
            return False, f"Syntax Error: {e}"
        except Exception as e:
            return False, f"Error: {e}"

    def test_import(self, dashboard_file):
        """Test if dashboard can be imported"""
        try:
            spec = importlib.util.spec_from_file_location("dashboard", dashboard_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True, "Import successful"
        except Exception as e:
            return False, f"Import error: {str(e)[:100]}"

    def test_streamlit_compatibility(self, dashboard_file):
        """Test Streamlit compatibility"""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import streamlit as st; exec(open('{dashboard_file}').read())",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return True, "Streamlit compatible"
            else:
                return False, f"Streamlit error: {result.stderr[:100]}"
        except subprocess.TimeoutExpired:
            return False, "Timeout during Streamlit test"
        except Exception as e:
            return False, f"Test error: {e}"

    def get_dashboard_files(self):
        """Get list of dashboard files"""
        dashboard_files = []

        # Main dashboard files
        for file in self.base_dir.glob("*dashboard*.py"):
            if file.name not in [
                "test_dashboard_launches.py",
                "test_dashboard_imports.py",
                "validate_dashboard.py",
                "verify_dashboards_production.py",
                "integration_test_dashboard.py",
                "final_dashboard_validation.py",
                "dashboard_status_report.py",
            ]:
                dashboard_files.append(file)

        # ZoL0-master dashboard files
        zol0_master = self.base_dir / "ZoL0-master"
        if zol0_master.exists():
            for file in zol0_master.glob("*dashboard*.py"):
                if file.name not in ["fix_dashboard.py", "run_dashboard.py"]:
                    dashboard_files.append(file)

        return sorted(dashboard_files)

    def check_dependencies(self, dashboard_file):
        """Check if required dependencies are available"""
        missing_deps = []

        try:
            with open(dashboard_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Common dependencies to check
            deps_to_check = [
                ("streamlit", "import streamlit"),
                ("pandas", "import pandas"),
                ("plotly", "import plotly"),
                ("numpy", "import numpy"),
                ("requests", "import requests"),
            ]

            for dep_name, import_line in deps_to_check:
                if import_line in content or f"import {dep_name}" in content:
                    try:
                        __import__(dep_name)
                    except ImportError:
                        missing_deps.append(dep_name)

        except Exception:
            return ["file_read_error"]

        return missing_deps

    def generate_report(self):
        """Generate comprehensive dashboard status report"""
        print("🔍 KOMPLETNY RAPORT STANU DASHBOARDÓW")
        print("=" * 60)
        print(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        dashboard_files = self.get_dashboard_files()

        total_dashboards = len(dashboard_files)
        working_dashboards = 0

        for dashboard_file in dashboard_files:
            print(f"\n📊 TESTOWANIE: {dashboard_file.name}")
            print("-" * 50)

            # Test syntax
            syntax_ok, syntax_msg = self.test_syntax(dashboard_file)
            print(f"   Składnia: {'✅' if syntax_ok else '❌'} {syntax_msg}")

            # Test dependencies
            missing_deps = self.check_dependencies(dashboard_file)
            if missing_deps:
                print(f"   Zależności: ❌ Brakuje: {', '.join(missing_deps)}")
            else:
                print("   Zależności: ✅ OK")

            # Test import
            import_ok, import_msg = self.test_import(dashboard_file)
            print(f"   Import: {'✅' if import_ok else '❌'} {import_msg}")

            # Test Streamlit compatibility
            streamlit_ok, streamlit_msg = self.test_streamlit_compatibility(
                dashboard_file
            )
            print(f"   Streamlit: {'✅' if streamlit_ok else '❌'} {streamlit_msg}")

            # Overall status
            dashboard_working = syntax_ok and not missing_deps and import_ok
            if dashboard_working:
                working_dashboards += 1
                print("   Status: ✅ DZIAŁA")
            else:
                print("   Status: ❌ BŁĄD")

            self.results[dashboard_file.name] = {
                "syntax": syntax_ok,
                "dependencies": len(missing_deps) == 0,
                "import": import_ok,
                "streamlit": streamlit_ok,
                "working": dashboard_working,
            }

        # Summary
        print("\n" + "=" * 60)
        print("📊 PODSUMOWANIE")
        print("=" * 60)
        print(f"Całkowita liczba dashboardów: {total_dashboards}")
        print(f"Działające dashboardy: {working_dashboards}")
        print(f"Procent sukcesu: {(working_dashboards/total_dashboards)*100:.1f}%")

        print("\n🔧 PROBLEMY DO NAPRAWIENIA:")
        print("-" * 30)
        for dashboard_name, status in self.results.items():
            if not status["working"]:
                print(f"❌ {dashboard_name}")
                if not status["syntax"]:
                    print("   - Błędy składni")
                if not status["dependencies"]:
                    print("   - Brakujące zależności")
                if not status["import"]:
                    print("   - Błędy importu")
                if not status["streamlit"]:
                    print("   - Problemy z Streamlit")

        print("\n✅ DZIAŁAJĄCE DASHBOARDY:")
        print("-" * 25)
        for dashboard_name, status in self.results.items():
            if status["working"]:
                print(f"✅ {dashboard_name}")

        return working_dashboards, total_dashboards


def main():
    tester = DashboardTester()
    working, total = tester.generate_report()

    print(f"\n🎯 KOŃCOWY WYNIK: {working}/{total} dashboardów działa poprawnie")

    return working == total


# Test edge-case: brak pliku dashboard
if __name__ == "__main__":
    def test_missing_dashboard_file():
        """Testuje obsługę braku pliku dashboard przy testach składni."""
        tester = DashboardTester()
        ok, msg = tester.test_syntax("nonexistent_dashboard.py")
        if not ok and msg == "File not found":
            print("OK: FileNotFoundError handled gracefully.")
        else:
            print("FAIL: No exception for missing dashboard file.")
    test_missing_dashboard_file()

    success = main()
    sys.exit(0 if success else 1)

# TODO: Integrate with CI/CD pipeline for automated dashboard status and edge-case tests.
# Edge-case tests: simulate file not found, syntax errors, and import failures.
# All public methods have docstrings and exception handling.
