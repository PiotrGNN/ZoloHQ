#!/usr/bin/env python3
"""
Memory Cleanup Optimizer for ZoL0 System
Implements memory leak fixes and optimization strategies
"""

import gc
import logging
import weakref
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Advanced memory optimization for Streamlit dashboards"""

    def __init__(self):
        self.memory_limit_mb = 500  # 500MB limit per dashboard
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.now()
        self.object_tracker = weakref.WeakSet()

    def setup_streamlit_optimizations(self):
        """Configure Streamlit for optimal memory usage"""
        if "memory_optimizer" not in st.session_state:
            st.session_state.memory_optimizer = True
            st.session_state.cleanup_counter = 0
            st.session_state.large_objects = []

            # Set up cache with memory limits
            st.cache_data.clear()

    def optimized_cache_decorator(self, max_entries: int = 50, ttl: int = 1800):
        """Enhanced cache decorator with memory limits"""
        return st.cache_data(
            max_entries=max_entries, ttl=ttl, show_spinner=False, persist="disk"
        )

    def track_large_object(self, obj: Any, name: str = None):
        """Track large objects for cleanup"""
        if hasattr(st.session_state, "large_objects"):
            st.session_state.large_objects.append(
                {
                    "object": obj,
                    "name": name or str(type(obj)),
                    "created": datetime.now(),
                }
            )

    def cleanup_session_state(self):
        """Clean up old session state data"""
        try:
            keys_to_remove = []
            current_time = datetime.now()

            for key in st.session_state.keys():
                if key.startswith("temp_") or key.startswith("cache_"):
                    # Remove temporary keys older than 30 minutes
                    if hasattr(st.session_state[key], "created"):
                        if current_time - st.session_state[key].created > timedelta(
                            minutes=30
                        ):
                            keys_to_remove.append(key)

            for key in keys_to_remove:
                del st.session_state[key]

            logger.info(f"Cleaned up {len(keys_to_remove)} session state keys")

        except Exception as e:
            logger.error(f"Error cleaning session state: {e}")

    def cleanup_large_objects(self):
        """Clean up tracked large objects"""
        try:
            if hasattr(st.session_state, "large_objects"):
                cleaned = 0
                remaining_objects = []
                current_time = datetime.now()

                for obj_info in st.session_state.large_objects:
                    # Clean objects older than 15 minutes
                    if current_time - obj_info["created"] > timedelta(minutes=15):
                        try:
                            del obj_info["object"]
                            cleaned += 1
                        except Exception:
                            logging.exception(
                                "Exception occurred in memory_cleanup_optimizer at line 94"
                            )
                    else:
                        remaining_objects.append(obj_info)

                st.session_state.large_objects = remaining_objects
                logger.info(f"Cleaned up {cleaned} large objects")

        except Exception as e:
            logger.error(f"Error cleaning large objects: {e}")

    def force_garbage_collection(self):
        """Force aggressive garbage collection"""
        try:
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            return collected
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
            return 0

    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            return {
                "memory_mb": memory_mb,
                "memory_percent": process.memory_percent(),
                "is_critical": memory_mb > self.memory_limit_mb,
            }
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return {"memory_mb": 0, "memory_percent": 0, "is_critical": False}

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        try:
            # Convert object columns to category if beneficial
            for col in df.select_dtypes(include=["object"]):
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype("category")

            # Downcast numeric columns
            for col in df.select_dtypes(include=["int64"]):
                df[col] = pd.to_numeric(df[col], downcast="integer")

            for col in df.select_dtypes(include=["float64"]):
                df[col] = pd.to_numeric(df[col], downcast="float")

            return df
        except Exception as e:
            logger.error(f"Error optimizing DataFrame: {e}")
            return df

    def optimize_plotly_figure(self, fig: go.Figure) -> go.Figure:
        """Optimize Plotly figure for memory efficiency"""
        try:
            # Reduce data points if too many
            for trace in fig.data:
                if hasattr(trace, "x") and len(trace.x) > 10000:
                    # Sample data points for visualization
                    sample_size = 5000
                    step = len(trace.x) // sample_size
                    if step > 1:
                        trace.x = trace.x[::step]
                        if hasattr(trace, "y"):
                            trace.y = trace.y[::step]

            # Optimize layout
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,  # Fixed height to save memory
            )

            return fig
        except Exception as e:
            logger.error(f"Error optimizing Plotly figure: {e}")
            return fig

    def periodic_cleanup(self):
        """Perform periodic cleanup if needed"""
        current_time = datetime.now()
        if current_time - self.last_cleanup > timedelta(seconds=self.cleanup_interval):
            logger.info("Starting periodic cleanup...")

            # Check memory usage
            memory_info = self.check_memory_usage()

            if memory_info["is_critical"]:
                logger.warning(
                    f"Critical memory usage: {memory_info['memory_mb']:.1f}MB"
                )

                # Aggressive cleanup
                self.cleanup_session_state()
                self.cleanup_large_objects()
                st.cache_data.clear()

                # Force garbage collection
                self.force_garbage_collection()

            self.last_cleanup = current_time
            st.session_state.cleanup_counter = (
                st.session_state.get("cleanup_counter", 0) + 1
            )

            return memory_info

        return None

    def create_memory_monitor_widget(self):
        """Create a memory monitoring widget for dashboards"""
        memory_info = self.check_memory_usage()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Memory Usage",
                f"{memory_info['memory_mb']:.1f} MB",
                delta=f"{memory_info['memory_percent']:.1f}%",
            )

        with col2:
            status = "🚨 CRITICAL" if memory_info["is_critical"] else "✅ OK"
            st.metric("Status", status)

        with col3:
            cleanup_count = st.session_state.get("cleanup_counter", 0)
            st.metric("Cleanups", cleanup_count)

        # Show cleanup button if memory is critical
        if memory_info["is_critical"]:
            if st.button("🧹 Force Cleanup", type="primary"):
                with st.spinner("Cleaning up memory..."):
                    self.cleanup_session_state()
                    self.cleanup_large_objects()
                    self.force_garbage_collection()
                    st.cache_data.clear()
                    st.rerun()


# Global optimizer instance
memory_optimizer = MemoryOptimizer()


def apply_memory_optimizations():
    """Apply memory optimizations to current Streamlit session"""
    memory_optimizer.setup_streamlit_optimizations()
    memory_optimizer.periodic_cleanup()


def optimized_load_data(load_func, *args, **kwargs):
    """Wrapper for data loading with memory optimization"""

    @memory_optimizer.optimized_cache_decorator(max_entries=20, ttl=1800)
    def cached_load_func(*args, **kwargs):
        df = load_func(*args, **kwargs)
        return memory_optimizer.optimize_dataframe(df)

    return cached_load_func(*args, **kwargs)


def optimized_create_figure(create_func, *args, **kwargs):
    """Wrapper for figure creation with memory optimization"""
    fig = create_func(*args, **kwargs)
    optimized_fig = memory_optimizer.optimize_plotly_figure(fig)
    memory_optimizer.track_large_object(optimized_fig, "plotly_figure")
    return optimized_fig


def memory_safe_session_state(key: str, default_value: Any = None):
    """Memory-safe session state access"""
    if key not in st.session_state:
        st.session_state[key] = default_value

    # Return the value directly, don't create nested structures for simple values
    return st.session_state[key]


if __name__ == "__main__":
    print("Memory Cleanup Optimizer for ZoL0 System")
    print("=========================================")

    # Test memory optimization
    optimizer = MemoryOptimizer()
    memory_info = optimizer.check_memory_usage()

    print(f"Current Memory Usage: {memory_info['memory_mb']:.1f} MB")
    print(f"Memory Percentage: {memory_info['memory_percent']:.1f}%")
    print(f"Critical Status: {'YES' if memory_info['is_critical'] else 'NO'}")

    if memory_info["is_critical"]:
        print("\n🚨 Performing emergency cleanup...")
        optimizer.force_garbage_collection()

        # Re-check after cleanup
        new_memory_info = optimizer.check_memory_usage()
        print(f"After Cleanup: {new_memory_info['memory_mb']:.1f} MB")
