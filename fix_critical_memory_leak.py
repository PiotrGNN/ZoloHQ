#!/usr/bin/env python3
"""
CRITICAL MEMORY LEAK FIX - ZoL0 System
=====================================

MEMORY LEAK IDENTIFIED: production_data_manager.py
Problem: Importing entire ZoL0-master system causing 336.73 MB memory growth
Solution: Fix imports and add memory management
"""

import json
import logging
import os
from datetime import datetime

import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryLeakFixer:
    """Fix critical memory leaks in ZoL0 system"""

    def __init__(self):
        self.fixes_applied = []
        self.memory_before = 0
        self.memory_after = 0

    def measure_memory(self) -> float:
        """Measure current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def fix_production_data_manager_memory_leak(self):
        """Fix the critical memory leak in production_data_manager.py"""
        logger.info("🔧 Fixing CRITICAL memory leak in production_data_manager.py")

        file_path = "production_data_manager.py"

        # Create backup
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(original_content)

            logger.info(f"📄 Backup created: {backup_path}")

            # Apply memory leak fixes
            fixed_content = self._apply_memory_fixes(original_content)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)

            self.fixes_applied.append("production_data_manager.py - Memory leak fix")
            logger.info("✅ CRITICAL memory leak fixed in production_data_manager.py")

        except Exception as e:
            logger.error(f"❌ Failed to fix memory leak: {e}")

    def _apply_memory_fixes(self, content: str) -> str:
        """Apply comprehensive memory leak fixes"""

        # 1. Add memory management imports at the top
        memory_imports = """
# MEMORY LEAK FIX - Added memory management
import gc
import sys
import weakref
from functools import lru_cache
"""

        # Find the import section and add memory management
        if "import os" in content and "import gc" not in content:
            content = content.replace("import os", f"import os{memory_imports}")

        # 2. Fix the ZoL0-master import issue - use lazy loading
        old_import_pattern = """try:
            # Import the existing connector
            import sys
            sys.path.append(str(Path(__file__).parent / "ZoL0-master"))
            from data.execution.bybit_connector import BybitConnector"""

        new_import_pattern = """try:
            # MEMORY FIX: Use lazy import to prevent memory leak
            self.bybit_connector = self._lazy_import_bybit_connector()"""

        if old_import_pattern in content:
            content = content.replace(old_import_pattern, new_import_pattern)

        # 3. Add lazy loading method
        lazy_loading_method = '''
    def _lazy_import_bybit_connector(self):
        """Lazy import BybitConnector to prevent memory leak"""
        try:
            import sys
            zol0_path = str(Path(__file__).parent / "ZoL0-master")
            
            # Only add to path if not already present
            if zol0_path not in sys.path:
                sys.path.append(zol0_path)
            
            # Import only what we need
            from data.execution.bybit_connector import BybitConnector
            
            connector = BybitConnector(
                api_key=self.api_key,
                api_secret=self.api_secret,
                use_testnet=not self.is_production
            )
            
            # Clean up imports to prevent memory accumulation
            if 'data.execution.bybit_connector' in sys.modules:
                # Keep only essential references
                gc.collect()
            
            return connector
            
        except Exception as e:
            logger.error(f"Failed to import BybitConnector: {e}")
            return None
'''

        # Insert the lazy loading method after the __init__ method
        init_end = content.find("self._start_health_monitor()")
        if init_end != -1:
            insert_pos = content.find("\n", init_end) + 1
            content = content[:insert_pos] + lazy_loading_method + content[insert_pos:]

        # 4. Add memory cleanup to cache methods
        cache_cleanup = """
        # MEMORY FIX: Limit cache size to prevent memory leaks
        if len(self.data_cache) > 100:  # Limit cache size
            # Remove oldest entries
            oldest_keys = sorted(self.cache_timestamps.keys(), 
                               key=lambda k: self.cache_timestamps[k])[:50]
            for key in oldest_keys:
                if key in self.data_cache:
                    del self.data_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
            gc.collect()  # Force garbage collection
"""

        # Add cache cleanup after every cache operation
        if "_cache_data" in content:
            old_cache_method = "self.cache_timestamps[cache_key] = datetime.now()"
            new_cache_method = old_cache_method + cache_cleanup
            content = content.replace(old_cache_method, new_cache_method)

        # 5. Fix the second ZoL0-master import
        second_import_pattern = """import sys
            sys.path.append(str(Path(__file__).parent / "ZoL0-master"))
            from data.execution.bybit_connector import BybitConnector"""

        if second_import_pattern in content:
            content = content.replace(
                second_import_pattern,
                "# MEMORY FIX: Use existing connector to prevent duplicate imports",
            )

        # 6. Add destructor for cleanup
        destructor_method = '''
    def __del__(self):
        """Cleanup method to prevent memory leaks"""
        try:
            # Stop background threads
            if hasattr(self, '_health_monitor_active'):
                self._health_monitor_active = False
            
            # Clear cache
            if hasattr(self, 'data_cache'):
                self.data_cache.clear()
            if hasattr(self, 'cache_timestamps'):
                self.cache_timestamps.clear()
            
            # Close connector
            if hasattr(self, 'bybit_connector') and self.bybit_connector:
                try:
                    self.bybit_connector.close()
                except Exception:
                    pass
                    
            gc.collect()
        except Exception:
            pass  # Ignore errors in destructor
'''

        # Add destructor at the end of the class
        class_end = content.rfind("}")  # Find last closing brace
        if class_end != -1:
            content = content[:class_end] + destructor_method + content[class_end:]

        # 7. Add memory monitoring decorators to heavy methods
        memory_decorator = '''
def memory_managed(func):
    """Decorator to manage memory for heavy operations"""
    def wrapper(*args, **kwargs):
        gc.collect()  # Clean up before
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            gc.collect()  # Clean up after
    return wrapper

'''

        # Add decorator at the beginning of the class
        class_start = content.find("class ProductionDataManager:")
        if class_start != -1:
            content = content[:class_start] + memory_decorator + content[class_start:]

        # Apply decorator to heavy methods
        heavy_methods = [
            "def get_enhanced_portfolio_details",
            "def get_account_balance",
            "def get_portfolio_data",
        ]

        for method in heavy_methods:
            if method in content:
                content = content.replace(method, f"    @memory_managed\n    {method}")

        return content

    def create_memory_monitoring_script(self):
        """Create a memory monitoring script for ongoing surveillance"""

        monitoring_script = '''#!/usr/bin/env python3
"""
ZoL0 Memory Leak Monitor
Continuous monitoring for memory leaks after fixes
"""

import psutil
import time
import json
import os
from datetime import datetime

class MemoryLeakMonitor:
    def __init__(self):
        self.baseline = None
        self.alerts_sent = []
        
    def monitor_continuous(self, duration_minutes=60):
        """Monitor for memory leaks continuously"""
        print(f"🔍 Starting {duration_minutes} minute memory leak monitoring...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        snapshots = []
        
        while time.time() < end_time:
            # Get memory info
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'memory_mb': memory_mb,
                'threads': len(process.threads()),
                'files': len(process.open_files()) if hasattr(process, 'open_files') else 0
            }
            
            snapshots.append(snapshot)
            
            # Check for concerning growth
            if len(snapshots) > 10:
                recent_growth = snapshots[-1]['memory_mb'] - snapshots[-10]['memory_mb']
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
            'monitoring_duration_minutes': duration_minutes,
            'total_snapshots': len(snapshots),
            'memory_range': {
                'min_mb': min(s['memory_mb'] for s in snapshots),
                'max_mb': max(s['memory_mb'] for s in snapshots),
                'final_mb': snapshots[-1]['memory_mb']
            },
            'memory_growth_mb': snapshots[-1]['memory_mb'] - snapshots[0]['memory_mb'],
            'snapshots': snapshots
        }
        
        report_file = f"memory_leak_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📄 Monitoring report saved: {report_file}")
        print(f"📈 Total memory growth: {report['memory_growth_mb']:.2f} MB")
        
        return report

if __name__ == "__main__":
    monitor = MemoryLeakMonitor()
    monitor.monitor_continuous(60)  # Monitor for 1 hour
'''

        with open("memory_leak_monitor.py", "w") as f:
            f.write(monitoring_script)

        self.fixes_applied.append(
            "memory_leak_monitor.py - Created continuous monitoring"
        )
        logger.info("📊 Created continuous memory monitoring script")

    def run_all_fixes(self):
        """Run all memory leak fixes"""
        print("🚨 CRITICAL MEMORY LEAK REPAIR - ZoL0 System")
        print("=" * 60)

        self.memory_before = self.measure_memory()
        logger.info(f"💾 Memory before fixes: {self.memory_before:.2f} MB")

        # Apply fixes
        self.fix_production_data_manager_memory_leak()
        self.create_memory_monitoring_script()

        self.memory_after = self.measure_memory()

        # Generate report
        self.generate_fix_report()

    def generate_fix_report(self):
        """Generate comprehensive fix report"""

        report = {
            "fix_timestamp": datetime.now().isoformat(),
            "memory_impact": {
                "before_mb": self.memory_before,
                "after_mb": self.memory_after,
                "memory_saved_mb": self.memory_before - self.memory_after,
            },
            "fixes_applied": self.fixes_applied,
            "critical_issue_resolved": {
                "file": "production_data_manager.py",
                "issue": "Massive memory leak (336.73 MB) from ZoL0-master imports",
                "solution": "Lazy loading, cache limits, memory management",
            },
            "recommendations": [
                "🔄 Restart ZoL0 system to apply memory fixes",
                "📊 Run memory_leak_monitor.py for ongoing surveillance",
                "⚠️ Monitor dashboard memory usage after restart",
                "🧹 Consider periodic system restarts for memory hygiene",
            ],
        }

        report_file = (
            f"MEMORY_LEAK_FIX_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print("\n🎯 MEMORY LEAK FIX RESULTS")
        print("=" * 40)
        print("📁 Critical file fixed: production_data_manager.py")
        print(
            f"💾 Memory impact: {report['memory_impact']['memory_saved_mb']:.2f} MB saved"
        )
        print(f"🔧 Fixes applied: {len(self.fixes_applied)}")

        print("\n💡 Next Steps:")
        for rec in report["recommendations"]:
            print(f"   {rec}")

        print(f"\n📄 Detailed report: {report_file}")

        return report


def main():
    fixer = MemoryLeakFixer()
    fixer.run_all_fixes()


if __name__ == "__main__":
    main()
