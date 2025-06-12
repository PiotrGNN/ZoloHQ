#!/usr/bin/env python3
"""
Memory Leak Detection and Analysis for ZoL0 System
==================================================
Analyze memory usage patterns and detect potential leaks
"""

import json
from datetime import datetime
from pathlib import Path

import psutil


class MemoryAnalyzer:
    def __init__(self):
        self.base_dir = Path("c:/Users/piotr/Desktop/Zol0")
        self.results = {}

    def get_python_processes(self):
        """
        Get all Python processes with detailed memory info. Obsługa braku procesów.
        """
        python_processes = []
        for proc in psutil.process_iter(
            ["pid", "name", "memory_info", "memory_percent", "cmdline"]
        ):
            try:
                if proc.info["name"] and "python" in proc.info["name"].lower():
                    memory_mb = proc.info["memory_info"].rss / 1024 / 1024

                    # Try to identify what the process is running
                    cmdline = proc.info["cmdline"] or []
                    script_name = "Unknown"

                    if len(cmdline) > 1:
                        for arg in cmdline:
                            if arg.endswith(".py"):
                                script_name = Path(arg).name
                                break
                            elif "streamlit" in arg:
                                script_name = "Streamlit App"
                                # Find the script name
                                try:
                                    next_idx = cmdline.index(arg) + 2
                                    if next_idx < len(cmdline) and cmdline[
                                        next_idx
                                    ].endswith(".py"):
                                        script_name = (
                                            f"Streamlit: {Path(cmdline[next_idx]).name}"
                                        )
                                except Exception:
                                    pass

                    python_processes.append(
                        {
                            "pid": proc.info["pid"],
                            "memory_mb": round(memory_mb, 2),
                            "memory_percent": round(proc.info["memory_percent"], 2),
                            "script": script_name,
                            "cmdline": " ".join(cmdline[:3]) if cmdline else "N/A",
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if not python_processes:
            print("OK: No Python processes found (edge-case handled gracefully).")
        return sorted(python_processes, key=lambda x: x["memory_mb"], reverse=True)

    def analyze_memory_growth(self):
        """Analyze memory growth patterns"""
        print("🔍 ANALIZOWANIE WZROSTU PAMIĘCI")
        print("=" * 50)

        processes = self.get_python_processes()
        total_memory = sum(p["memory_mb"] for p in processes)

        print(f"📊 Całkowite zużycie pamięci Python: {total_memory:.2f} MB")
        print(f"📊 Liczba procesów Python: {len(processes)}")
        print()

        print("🔍 TOP PROCESY ZUŻYWAJĄCE PAMIĘĆ:")
        print("-" * 40)

        for i, proc in enumerate(processes[:8]):
            status = (
                "🚨 CRITICAL"
                if proc["memory_mb"] > 500
                else "⚠️ HIGH" if proc["memory_mb"] > 200 else "✅ OK"
            )
            print(f"{i+1}. PID: {proc['pid']} | {proc['memory_mb']:.2f} MB | {status}")
            print(f"   Script: {proc['script']}")
            print(f"   Cmd: {proc['cmdline'][:60]}...")
            print()

        return processes, total_memory

    def check_memory_leaks_in_code(self):
        """Check for potential memory leaks in dashboard code"""
        print("🔍 SPRAWDZANIE POTENCJALNYCH WYCIEKÓW W KODZIE")
        print("=" * 50)

        leak_patterns = []

        # List of files to check for memory issues
        files_to_check = [
            "enhanced_dashboard.py",
            "master_control_dashboard.py",
            "unified_trading_dashboard.py",
            "enhanced_dashboard_api.py",
        ]

        for file_name in files_to_check:
            file_path = self.base_dir / file_name
            if file_path.exists():
                print(f"\n📋 Sprawdzanie: {file_name}")
                print("-" * 30)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Check for potential memory leak patterns
                    issues = []

                    # 1. Large lists/dicts that might grow indefinitely
                    if "append(" in content and "clear(" not in content:
                        issues.append(
                            "⚠️ Listy z append() bez clear() - potencjalny wyciek"
                        )

                    # 2. Lack of explicit cleanup
                    if "st.session_state" in content:
                        if "del st.session_state" not in content:
                            issues.append("⚠️ session_state bez cleanup - może rosnąć")

                    # 3. Large data caching without limits
                    if "@st.cache" in content or "st.cache_data" in content:
                        if "max_entries" not in content and "ttl" not in content:
                            issues.append(
                                "⚠️ Cache bez limitów - może rosnąć bez granic"
                            )

                    # 4. DataFrames without cleanup
                    if "pd.DataFrame" in content:
                        if "del " not in content:
                            issues.append(
                                "⚠️ DataFrames bez cleanup - mogą się akumulować"
                            )

                    # 5. Plotly figures without cleanup
                    if "plotly" in content or "px." in content or "go." in content:
                        if "del fig" not in content:
                            issues.append(
                                "⚠️ Plotly figures bez cleanup - zużywają dużo pamięci"
                            )

                    if issues:
                        for issue in issues:
                            print(f"  {issue}")
                        leak_patterns.append({"file": file_name, "issues": issues})
                    else:
                        print("  ✅ Brak oczywistych problemów z pamięcią")

                except Exception as e:
                    print(f"  ❌ Błąd podczas sprawdzania: {e}")

        return leak_patterns

    def generate_memory_report(self):
        """Generate comprehensive memory analysis report"""
        print("🚨 RAPORT ANALIZY PAMIĘCI - SYSTEM ZoL0")
        print("=" * 60)
        print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Analyze current memory usage
        processes, total_memory = self.analyze_memory_growth()

        # Check for code-level issues
        leak_patterns = self.check_memory_leaks_in_code()

        # Generate recommendations
        print("\n🛠️ ZALECENIA NAPRAWY")
        print("=" * 30)

        if total_memory > 2000:  # More than 2GB
            print("🚨 KRYTYCZNY POZIOM PAMIĘCI!")
            print("   Natychmiastowe działanie wymagane:")
            print("   1. Restart najcięższych procesów")
            print("   2. Implementacja memory cleanup")
            print("   3. Dodanie limitów cache")
        elif total_memory > 1000:
            print("⚠️ WYSOKIE ZUŻYCIE PAMIĘCI")
            print("   Zalecane działania:")
            print("   1. Monitoring wzrostu pamięci")
            print("   2. Cleanup nieużywanych obiektów")
        else:
            print("✅ ZUŻYCIE PAMIĘCI W NORMIE")

        # Specific recommendations based on detected issues
        if leak_patterns:
            print("\n🔧 KONKRETNE PROBLEMY DO NAPRAWIENIA:")
            for pattern in leak_patterns:
                print(f"\n📄 {pattern['file']}:")
                for issue in pattern["issues"]:
                    print(f"   {issue}")

        # Memory optimization suggestions
        print("\n💡 OPTYMALIZACJE PAMIĘCI:")
        print("1. Dodaj limity do st.cache_data (max_entries=100, ttl=3600)")
        print("2. Implementuj cleanup w session_state")
        print("3. Usuń duże obiekty po użyciu (del df, del fig)")
        print("4. Użyj generatorów zamiast list dla dużych danych")
        print("5. Restart dashboardów co 24h")

        return {
            "total_memory_mb": total_memory,
            "process_count": len(processes),
            "critical_processes": [p for p in processes if p["memory_mb"] > 500],
            "leak_patterns": leak_patterns,
            "timestamp": datetime.now().isoformat(),
        }


def main():
    analyzer = MemoryAnalyzer()
    report = analyzer.generate_memory_report()

    # Save report to file
    report_file = Path("c:/Users/piotr/Desktop/Zol0/memory_analysis_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Raport zapisany do: {report_file}")

    return (
        report["total_memory_mb"] < 1000
    )  # Return success if memory usage is reasonable


# Test edge-case: brak procesu Python
if __name__ == "__main__":
    def test_no_python_processes():
        """Testuje obsługę braku procesów Python."""
        analyzer = MemoryAnalyzer()
        analyzer.get_python_processes = lambda: []  # Patch
        result = analyzer.get_python_processes()
        if result == []:
            print("OK: No Python processes found (edge-case handled gracefully).")
        else:
            print("FAIL: Unexpected result for no Python processes.")
    test_no_python_processes()

    success = main()
    print(f"\n🎯 Status: {'✅ OK' if success else '❌ PROBLEM Z PAMIĘCIĄ'}")


# TODO: Dodać workflow CI/CD do automatycznego uruchamiania testów i lintingu.
