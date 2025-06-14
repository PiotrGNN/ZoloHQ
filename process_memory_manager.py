#!/usr/bin/env python3
"""
ZoL0 Process Memory Manager
Manages and restarts high-memory processes to prevent memory leaks
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List

import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessMemoryManager:
    """Manages memory-heavy processes in ZoL0 system"""

    def __init__(self, base_path: str = r"c:\Users\piotr\Desktop\Zol0"):
        self.base_path = base_path
        self.memory_threshold_mb = 400  # 400MB threshold for restart
        self.critical_threshold_mb = 600  # 600MB critical threshold
        self.process_info = {}

    def get_python_processes(self) -> List[Dict[str, Any]]:
        """Get all Python processes with memory info"""
        processes = []

        for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
            try:
                if proc.info["name"] == "python.exe":
                    cmdline = proc.info["cmdline"]
                    if cmdline and len(cmdline) > 1:
                        script_name = os.path.basename(cmdline[1])
                        memory_mb = proc.info["memory_info"].rss / 1024 / 1024

                        processes.append(
                            {
                                "pid": proc.info["pid"],
                                "script": script_name,
                                "memory_mb": memory_mb,
                                "cmdline": " ".join(cmdline),
                                "process": proc,
                            }
                        )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return sorted(processes, key=lambda x: x["memory_mb"], reverse=True)

    def identify_zol0_processes(self) -> List[Dict[str, Any]]:
        """Identify ZoL0-related processes"""
        all_processes = self.get_python_processes()
        zol0_processes = []

        zol0_scripts = [
            "enhanced_dashboard.py",
            "master_control_dashboard.py",
            "unified_trading_dashboard.py",
            "enhanced_dashboard_api.py",
            "dashboard.py",
        ]

        for proc in all_processes:
            if any(script in proc["script"] for script in zol0_scripts):
                zol0_processes.append(proc)

        return zol0_processes

    def get_critical_processes(self) -> List[Dict[str, Any]]:
        """Get processes using critical amounts of memory"""
        zol0_processes = self.identify_zol0_processes()
        critical_processes = []

        for proc in zol0_processes:
            if proc["memory_mb"] > self.critical_threshold_mb:
                critical_processes.append(proc)

        return critical_processes

    def terminate_process_safely(self, proc_info: Dict[str, Any]) -> bool:
        """Safely terminate a process"""
        try:
            process = proc_info["process"]
            pid = proc_info["pid"]
            script = proc_info["script"]

            logger.info(
                f"Terminating process {pid} ({script}) - Memory: {proc_info['memory_mb']:.1f}MB"
            )

            # Try graceful termination first
            process.terminate()

            # Wait up to 10 seconds for graceful shutdown
            try:
                process.wait(timeout=10)
                logger.info(f"Process {pid} terminated gracefully")
                return True
            except psutil.TimeoutExpired:
                # Force kill if graceful termination fails
                logger.warning(f"Force killing process {pid}")
                process.kill()
                process.wait(timeout=5)
                return True

        except Exception as e:
            logger.error(f"Error terminating process {proc_info['pid']}: {e}")
            return False

    def restart_dashboard_services(self):
        """Restart dashboard services using the system restart script"""
        try:
            logger.info("Restarting ZoL0 system services...")

            # Stop existing services first
            critical_processes = self.get_critical_processes()
            for proc in critical_processes:
                self.terminate_process_safely(proc)

            # Wait a moment for cleanup
            time.sleep(5)

            # Restart the system
            restart_script = os.path.join(self.base_path, "restart_system.py")
            if os.path.exists(restart_script):
                logger.info("Starting system restart...")
                subprocess.Popen(["python", restart_script], cwd=self.base_path)

                logger.info("System restart initiated")
                return True
            else:
                logger.error(f"Restart script not found: {restart_script}")
                return False

        except Exception as e:
            logger.error(f"Error restarting services: {e}")
            return False

    def memory_analysis_report(self) -> Dict[str, Any]:
        """Generate detailed memory analysis report"""
        zol0_processes = self.identify_zol0_processes()
        critical_processes = self.get_critical_processes()

        total_memory = sum(proc["memory_mb"] for proc in zol0_processes)
        critical_memory = sum(proc["memory_mb"] for proc in critical_processes)

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_processes": len(zol0_processes),
            "critical_processes": len(critical_processes),
            "total_memory_mb": total_memory,
            "critical_memory_mb": critical_memory,
            "memory_threshold_mb": self.memory_threshold_mb,
            "critical_threshold_mb": self.critical_threshold_mb,
            "processes": zol0_processes,
            "recommendations": [],
        }

        # Generate recommendations
        if critical_processes:
            report["recommendations"].append(
                f"🚨 CRITICAL: {len(critical_processes)} processes using {critical_memory:.1f}MB total"
            )
            report["recommendations"].append("Immediate restart recommended")

        if total_memory > 2000:  # 2GB total
            report["recommendations"].append(
                f"⚠️ HIGH: Total memory usage {total_memory:.1f}MB exceeds 2GB"
            )

        if len(zol0_processes) > 10:
            report["recommendations"].append(
                f"🔍 INFO: {len(zol0_processes)} ZoL0 processes running"
            )

        return report

    def force_memory_cleanup(self):
        """Force immediate memory cleanup of critical processes"""
        logger.info("Starting force memory cleanup...")

        critical_processes = self.get_critical_processes()

        if not critical_processes:
            logger.info("No critical processes found")
            return True

        logger.info(f"Found {len(critical_processes)} critical processes")

        for proc in critical_processes:
            logger.info(
                f"Restarting: {proc['script']} (PID: {proc['pid']}, Memory: {proc['memory_mb']:.1f}MB)"
            )
            self.terminate_process_safely(proc)

        # Wait for cleanup
        time.sleep(3)

        # Restart services
        return self.restart_dashboard_services()

    def monitor_and_manage(self, restart_if_critical: bool = True):
        """Monitoruje i zarządza pamięcią procesu. Restartuje proces jeśli przekroczone limity krytyczne."""
        try:
            import psutil, os
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            if rss_mb > 1024 and restart_if_critical:
                self._restart_process()
                self._log_critical(f"Restart z powodu przekroczenia limitu pamięci: {rss_mb:.2f} MB")
                return False
            return True
        except Exception as e:
            self._log_critical(f"Błąd monitorowania pamięci: {e}")
            return False

    def save_report(self, report: dict, filename: str = None):
        """Zapisuje raport pamięci do pliku JSON."""
        import json, datetime
        if not filename:
            filename = f"memory_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self._log_info(f"Raport pamięci zapisany: {filename}")
            return filename
        except Exception as e:
            self._log_critical(f"Błąd zapisu raportu: {e}")
            return None

    def _restart_process(self):
        import os, sys
        self._log_critical("Restartuję proces...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def _log_critical(self, msg):
        import logging
        logging.critical(msg)

    def _log_info(self, msg):
        import logging
        logging.info(msg)


if __name__ == "__main__":
    print("ZoL0 Process Memory Manager")
    print("===========================")

    manager = ProcessMemoryManager()

    # Generate and display memory report
    success = manager.monitor_and_manage(restart_if_critical=True)

    if success:
        print("\n✅ Memory management completed successfully")
    else:
        print("\n❌ Memory management encountered issues")

    # Save detailed report
    report = manager.memory_analysis_report()
    report_file = manager.save_report(report)
    print(f"📄 Detailed report saved: {report_file}")

# CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml
