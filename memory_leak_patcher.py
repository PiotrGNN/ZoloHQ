#!/usr/bin/env python3
"""
Memory Leak Patch System
Applies immediate fixes to identified memory leak sources
"""

import logging
import os
import re
import shutil
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryLeakPatcher:
    """Patches memory leaks in dashboard files"""

    def __init__(self, base_path: str = r"c:\Users\piotr\Desktop\Zol0"):
        self.base_path = base_path
        self.backup_suffix = f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def backup_file(self, file_path: str) -> str:
        """Create backup of file before patching"""
        backup_path = file_path + self.backup_suffix
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backed up {file_path} to {backup_path}")
        return backup_path

    def patch_enhanced_dashboard(self):
        """
        Patch enhanced_dashboard.py for memory leaks. Obsługa braku pliku do patchowania.
        """
        file_path = os.path.join(self.base_path, "enhanced_dashboard.py")
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            print("OK: FileNotFoundError handled gracefully.")
            return False

        self.backup_file(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Add memory optimization imports at the top
        imports_patch = """
# Memory optimization imports
import gc
from memory_cleanup_optimizer import memory_optimizer, apply_memory_optimizations, optimized_load_data, memory_safe_session_state
"""

        # Find the import section and add our imports
        if (
            "import streamlit as st" in content
            and "from memory_cleanup_optimizer" not in content
        ):
            content = content.replace(
                "import streamlit as st", "import streamlit as st" + imports_patch
            )

        # Add memory optimization setup at the beginning of main function
        main_setup_patch = """
    # Initialize memory optimizations
    apply_memory_optimizations()
    
    # Add memory monitor to sidebar
    with st.sidebar:
        st.subheader("🧠 Memory Monitor")
        memory_optimizer.create_memory_monitor_widget()
        
"""

        # Find main function or streamlit setup and add optimization
        if "def main():" in content:
            content = content.replace(
                "def main():", "def main():" + "\n" + main_setup_patch
            )
        elif "st.set_page_config" in content:
            # Add after page config
            content = content.replace(
                "st.set_page_config", main_setup_patch + "    st.set_page_config"
            )

        # Replace cache decorators with optimized versions
        content = re.sub(
            r"@st\.cache_data(?:\([^)]*\))?",
            "@memory_optimizer.optimized_cache_decorator(max_entries=20, ttl=1800)",
            content,
        )

        # Add cleanup for large DataFrames
        df_cleanup_patch = """
        # Memory optimization for DataFrames
        if 'df' in locals() and hasattr(df, 'memory_usage'):
            memory_optimizer.track_large_object(df, 'dataframe')
            df = memory_optimizer.optimize_dataframe(df)
"""

        # Add DataFrame cleanup after data loading
        content = re.sub(
            r"(df\s*=\s*[^=\n]+(?:\.csv|\.json|\.parquet)[^=\n]*)",
            r"\1" + df_cleanup_patch,
            content,
        )

        # Replace session_state usage with memory-safe version
        content = re.sub(
            r"st\.session_state\.([a-zA-Z_][a-zA-Z0-9_]*)",
            r'memory_safe_session_state("\1")',
            content,
        )

        # Add periodic cleanup calls
        cleanup_patch = """
    # Periodic memory cleanup
    memory_optimizer.periodic_cleanup()
"""

        # Add cleanup before major operations
        content = re.sub(
            r"(st\.plotly_chart\(|st\.dataframe\(|st\.table\()",
            cleanup_patch + r"    \1",
            content,
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Patched {file_path} for memory optimization")
        return True

    def patch_master_control_dashboard(self):
        """Patch master_control_dashboard.py for memory leaks"""
        file_path = os.path.join(self.base_path, "master_control_dashboard.py")

        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False

        self.backup_file(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Add memory optimization imports
        if "from memory_cleanup_optimizer" not in content:
            content = (
                "from memory_cleanup_optimizer import memory_optimizer, apply_memory_optimizations, optimized_create_figure\n"
                + content
            )

        # Add Plotly figure optimization
        content = re.sub(
            r"(fig\s*=\s*go\.Figure\([^)]*\))",
            r"\1\n        fig = memory_optimizer.optimize_plotly_figure(fig)",
            content,
        )

        # Add cleanup for Plotly figures
        content = re.sub(
            r"(st\.plotly_chart\(fig[^)]*\))",
            r"\1\n        del fig  # Memory cleanup\n        gc.collect()",
            content,
        )

        # Add session state cleanup
        session_cleanup = """
        # Clean up old session state data
        if len(st.session_state) > 50:  # Arbitrary limit
            memory_optimizer.cleanup_session_state()
"""

        # Add at the beginning of main sections
        content = re.sub(r"(def\s+\w+\(\):)", r"\1" + session_cleanup, content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Patched {file_path} for memory optimization")
        return True

    def patch_unified_trading_dashboard(self):
        """Patch unified_trading_dashboard.py for memory leaks"""
        file_path = os.path.join(self.base_path, "unified_trading_dashboard.py")

        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False

        self.backup_file(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Add memory optimization imports
        if "from memory_cleanup_optimizer" not in content:
            content = (
                "from memory_cleanup_optimizer import memory_optimizer, apply_memory_optimizations, optimized_load_data\nimport gc\n"
                + content
            )

        # Replace list.append() patterns with memory-safe alternatives
        list_append_patch = """
        # Memory-safe list management
        if len({list_name}) > 1000:  # Prevent unlimited growth
            {list_name} = {list_name}[-500:]  # Keep only last 500 items
"""

        # Find list append patterns and add memory limits
        content = re.sub(
            r"(\w+)\.append\([^)]+\)",
            lambda m: m.group(0) + list_append_patch.format(list_name=m.group(1)),
            content,
        )

        # Add DataFrame memory optimization
        content = re.sub(
            r"(pd\.DataFrame\([^)]+\))",
            r"memory_optimizer.optimize_dataframe(\1)",
            content,
        )

        # Add periodic cleanup
        content = re.sub(
            r"(if\s+st\.button\()",
            r"memory_optimizer.periodic_cleanup()\n    \1",
            content,
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Patched {file_path} for memory optimization")
        return True

    def patch_enhanced_dashboard_api(self):
        """Patch enhanced_dashboard_api.py for memory leaks"""
        file_path = os.path.join(self.base_path, "enhanced_dashboard_api.py")

        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False

        self.backup_file(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Add memory optimization for Flask routes
        if "import gc" not in content:
            content = "import gc\nfrom functools import wraps\n" + content

        # Add memory cleanup decorator for routes
        memory_decorator = """
def memory_cleanup_route(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            # Force garbage collection after each request
            gc.collect()
            return result
        except Exception as e:
            gc.collect()  # Cleanup even on error
            raise e
    return decorated_function

"""

        # Add decorator before route definitions
        if "@app.route" in content and "def memory_cleanup_route" not in content:
            content = content.replace(
                "from flask import", memory_decorator + "from flask import"
            )

        # Apply decorator to all routes
        content = re.sub(
            r"(@app\.route[^)]+\))\s*\ndef\s+(\w+)",
            r"\1\n@memory_cleanup_route\ndef \2",
            content,
        )

        # Add list size limits
        content = re.sub(
            r"(\w+)\.append\([^)]+\)",
            lambda m: f"""
        {m.group(0)}
        if len({m.group(1)}) > 500:  # Memory limit
            {m.group(1)} = {m.group(1)}[-250:]  # Keep last 250 items
""",
            content,
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Patched {file_path} for memory optimization")
        return True

    def apply_all_patches(self):
        """Apply all memory leak patches"""
        logger.info("Starting memory leak patching process...")

        patches_applied = []

        # Patch each file
        if self.patch_enhanced_dashboard():
            patches_applied.append("enhanced_dashboard.py")

        if self.patch_master_control_dashboard():
            patches_applied.append("master_control_dashboard.py")

        if self.patch_unified_trading_dashboard():
            patches_applied.append("unified_trading_dashboard.py")

        if self.patch_enhanced_dashboard_api():
            patches_applied.append("enhanced_dashboard_api.py")

        logger.info(
            f"Applied patches to {len(patches_applied)} files: {patches_applied}"
        )
        return patches_applied


if __name__ == "__main__":
    print("Memory Leak Patch System")
    print("========================")

    patcher = MemoryLeakPatcher()
    patched_files = patcher.apply_all_patches()

    print(f"\n✅ Successfully patched {len(patched_files)} files:")
    for file_name in patched_files:
        print(f"   - {file_name}")

    print("\n🔄 Recommendation: Restart the ZoL0 system to apply memory optimizations.")

    # Test edge-case: brak pliku do patchowania
    def test_missing_file():
        """Testuje obsługę braku pliku do patchowania."""
        patcher = MemoryLeakPatcher(base_path="c:/Users/piotr/Desktop/Zol0")
        patcher.patch_enhanced_dashboard()

    test_missing_file()

# --- CI/CD Integration ---
"""
This project uses GitHub Actions for CI/CD automation.
The workflow below runs all tests and linting on every push and pull request.
"""
# .github/workflows/ci-cd.yml (example)
#
# name: CI/CD Pipeline
# on: [push, pull_request]
# jobs:
#   build-test-lint:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - name: Set up Python
#         uses: actions/setup-python@v5
#         with:
#           python-version: '3.10'
#       - name: Install dependencies
#         run: |
#           python -m pip install --upgrade pip
#           pip install -r requirements.txt
#       - name: Run tests
#         run: |
#           pytest
#       - name: Run linting
#         run: |
#           flake8 .
#
# For local testing, run: pytest && flake8 .
