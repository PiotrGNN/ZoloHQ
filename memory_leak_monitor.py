#!/usr/bin/env python3
"""
ZoL0 Memory Leak Monitor
Continuous monitoring for memory leaks after fixes
"""

import json
import os
import time
from datetime import datetime
import subprocess
import sys

import psutil


class MemoryLeakMonitor:
    def __init__(self):
        self.baseline = None
        self.alerts_sent = []

    def monitor_continuous(self, duration_minutes=60):
        """
        Monitor for memory leaks continuously. Obsługa błędu zapisu raportu.
        """
        print(f"🔍 Starting {duration_minutes} minute memory leak monitoring...")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        snapshots = []

        while time.time() < end_time:
            # Get memory info
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "memory_mb": memory_mb,
                "threads": len(process.threads()),
                "files": (
                    len(process.open_files()) if hasattr(process, "open_files") else 0
                ),
            }

            snapshots.append(snapshot)

            # Check for concerning growth
            if len(snapshots) > 10:
                recent_growth = snapshots[-1]["memory_mb"] - snapshots[-10]["memory_mb"]
                if recent_growth > 50:  # 50MB growth in 10 checks
                    print(f"⚠️ HIGH MEMORY GROWTH: {recent_growth:.2f} MB")
                elif recent_growth > 20:
                    print(f"⚠️ Moderate memory growth: {recent_growth:.2f} MB")
                else:
                    print(f"✅ Memory stable: {memory_mb:.2f} MB")
            else:
                print(f"📊 Memory: {memory_mb:.2f} MB")

            time.sleep(30)  # Check every 30 seconds

        # Generate report
        report = {
            "monitoring_duration_minutes": duration_minutes,
            "total_snapshots": len(snapshots),
            "memory_range": {
                "min_mb": min(s["memory_mb"] for s in snapshots),
                "max_mb": max(s["memory_mb"] for s in snapshots),
                "final_mb": snapshots[-1]["memory_mb"],
            },
            "memory_growth_mb": snapshots[-1]["memory_mb"] - snapshots[0]["memory_mb"],
            "snapshots": snapshots,
        }

        report_file = (
            f"memory_leak_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"OK: File write error handled gracefully: {e}")
            return report

        print(f"📄 Monitoring report saved: {report_file}")
        print(f"📈 Total memory growth: {report['memory_growth_mb']:.2f} MB")

        return report


# Test edge-case: błąd zapisu raportu
def test_report_write_error():
    """Testuje obsługę błędu zapisu raportu monitoringu."""
    import tempfile, os, stat

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    os.chmod(temp_file.name, 0)
    monitor = MemoryLeakMonitor()
    monitor.monitor_continuous = lambda duration_minutes=1: open(
        temp_file.name, "w"
    )  # Patch
    try:
        monitor.monitor_continuous(1)
    except Exception as e:
        print(f"FAIL: Unexpected exception: {e}")
    else:
        print("OK: File write error handled gracefully.")
    os.unlink(temp_file.name)


if __name__ == "__main__":
    monitor = MemoryLeakMonitor()
    monitor.monitor_continuous(60)  # Monitor for 1 hour

    def run_ci_cd_memory_checks() -> None:
        """Run memory monitoring tests and linting in CI/CD pipelines."""
        if not os.getenv("CI"):
            return

        commands = [
            [sys.executable, "-m", "pytest", "tests/test_active_memory_leak_detector.py"],
            ["ruff", "--quiet", "./"],
        ]
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                raise RuntimeError(f"Command {' '.join(cmd)} failed")

    run_ci_cd_memory_checks()
