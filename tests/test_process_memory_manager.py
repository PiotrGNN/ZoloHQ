import pytest
import os
from process_memory_manager import ProcessMemoryManager
import psutil

def test_get_python_processes():
    manager = ProcessMemoryManager()
    procs = manager.get_python_processes()
    assert isinstance(procs, list)
    for proc in procs:
        assert "pid" in proc and "memory_mb" in proc

def test_memory_analysis_report():
    manager = ProcessMemoryManager()
    report = manager.memory_analysis_report()
    assert "total_processes" in report
    assert "total_memory_mb" in report

def test_terminate_process_safely_handles_invalid():
    manager = ProcessMemoryManager()
    fake_proc = {"process": psutil.Process(os.getpid()), "pid": 999999, "script": "fake.py", "memory_mb": 0}
    # Should not raise, even if process is already dead
    manager.terminate_process_safely(fake_proc)
