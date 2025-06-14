#!/usr/bin/env python3
"""
Memory Cleanup Optimizer for ZoL0 System
Implements memory leak fixes and optimization strategies
"""

import csv
import gc
import io
import logging
import sys
import weakref
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette_exporter import PrometheusMiddleware, handle_metrics
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEYS = {
    "admin-key": "admin",
    "memory-key": "memory",
    "partner-key": "partner",
    "premium-key": "premium",
}
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

# --- Monetization & SaaS Hooks ---
PREMIUM_API_KEYS = {"premium-key", "partner-key"}
PARTNER_WEBHOOKS = {"partner-key": "https://partner.example.com/webhook"}


def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return API_KEYS[api_key]


# --- FastAPI async API ---
mem_api = FastAPI(title="ZoL0 Memory Cleanup Optimizer API", version="2.0")
mem_api.add_middleware(PrometheusMiddleware)
mem_api.add_route("/metrics", handle_metrics)


# --- Monetization: Usage metering and billing endpoint ---
@mem_api.get("/api/usage", dependencies=[Depends(get_api_key)])
async def api_usage(role: str = Depends(get_api_key)):
    # Example: Return usage stats for billing/monetization
    # In production, integrate with billing/analytics backend
    return {
        "role": role,
        "memory_usage_mb": memory_optimizer.check_memory_usage().get("memory_mb", 0),
        "cleanups": getattr(st.session_state, "cleanup_counter", 0),
        "timestamp": datetime.now().isoformat(),
    }


# --- Monetization: Premium memory optimization endpoint ---
@mem_api.post("/api/premium/optimize", dependencies=[Depends(get_api_key)])
async def api_premium_optimize(role: str = Depends(get_api_key)):
    # Only allow premium/partner keys
    if role not in PREMIUM_API_KEYS:
        raise HTTPException(status_code=403, detail="Premium access required")
    memory_optimizer.cleanup_session_state()
    memory_optimizer.cleanup_large_objects()
    memory_optimizer.force_garbage_collection()
    st.cache_data.clear()
    return {"status": "premium optimization complete"}


# --- SaaS/Partner webhook endpoint ---
@mem_api.post("/api/partner/webhook", dependencies=[Depends(get_api_key)])
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    # In production, process partner webhook payload
    logger.info(f"Received partner webhook: {payload}")
    return {"status": "received", "payload": payload}


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

    def upload_report(self, report_path: str, api_key: str):
        # Monetization: SaaS memory profiling, upload to S3 or dashboard
        # In production, integrate with S3 or SaaS endpoint
        logger.info(f"Uploading memory report: {report_path} for {api_key}")
        if api_key in PARTNER_WEBHOOKS:
            # In production, send to partner webhook
            pass


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


# --- Pydantic Models ---
class MemoryQuery(BaseModel):
    pass


class BatchMemoryQuery(BaseModel):
    queries: list[MemoryQuery]


# --- Endpoints ---
@mem_api.get("/")
async def root():
    return {
        "status": "ok",
        "service": "ZoL0 Memory Cleanup Optimizer API",
        "version": "2.0",
    }


@mem_api.get("/api/health")
async def api_health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "ZoL0 Memory Cleanup Optimizer API",
        "version": "2.0",
    }


@mem_api.get(
    "/api/memory/status", dependencies=[Depends(get_api_key)]
)
async def api_memory_status(role: str = Depends(get_api_key)):
    return memory_optimizer.check_memory_usage()


@mem_api.post(
    "/api/memory/optimize", dependencies=[Depends(get_api_key)]
)
async def api_memory_optimize(role: str = Depends(get_api_key)):
    memory_optimizer.cleanup_session_state()
    memory_optimizer.cleanup_large_objects()
    memory_optimizer.force_garbage_collection()
    return memory_optimizer.check_memory_usage()


@mem_api.post(
    "/api/memory/batch", dependencies=[Depends(get_api_key)]
)
async def api_memory_batch(req: BatchMemoryQuery, role: str = Depends(get_api_key)):
    return {"results": [memory_optimizer.check_memory_usage() for _ in req.queries]}


@mem_api.get(
    "/api/export/csv", dependencies=[Depends(get_api_key)]
)
async def api_export_csv(role: str = Depends(get_api_key)):
    mem = memory_optimizer.check_memory_usage()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(mem.keys()))
    writer.writeheader()
    writer.writerow(mem)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv")


@mem_api.get(
    "/api/export/prometheus", dependencies=[Depends(get_api_key)]
)
async def api_export_prometheus(role: str = Depends(get_api_key)):
    mem = memory_optimizer.check_memory_usage()
    return PlainTextResponse(
        f"# HELP memory_usage_mb Memory usage in MB\nmemory_usage_mb {mem.get('memory_mb', 0)}",
        media_type="text/plain",
    )


@mem_api.get(
    "/api/report", dependencies=[Depends(get_api_key)]
)
async def api_report(role: str = Depends(get_api_key)):
    return {"status": "report generated (stub)"}


@mem_api.get(
    "/api/recommendations", dependencies=[Depends(get_api_key)]
)
async def api_recommendations(role: str = Depends(get_api_key)):
    mem = memory_optimizer.check_memory_usage()
    recs = []
    if mem.get("is_critical"):
        recs.append("Reduce data size, clear cache, or restart dashboard.")
    if mem.get("memory_mb", 0) > 400:
        recs.append("Consider scaling infrastructure or optimizing code.")
    return {"recommendations": recs}


@mem_api.get(
    "/api/premium/score", dependencies=[Depends(get_api_key)]
)
async def api_premium_score(role: str = Depends(get_api_key)):
    mem = memory_optimizer.check_memory_usage()
    score = max(0, 1000 - mem.get("memory_mb", 0) * 2)
    return {"score": score}


@mem_api.get(
    "/api/saas/tenant/{tenant_id}/report", dependencies=[Depends(get_api_key)]
)
async def api_saas_tenant_report(tenant_id: str, role: str = Depends(get_api_key)):
    mem = memory_optimizer.check_memory_usage()
    return {"tenant_id": tenant_id, "report": mem}


@mem_api.get(
    "/api/partner/webhook", dependencies=[Depends(get_api_key)]
)
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    return {"status": "received", "payload": payload}


@mem_api.get("/api/test/edge-case")
async def api_edge_case():
    try:
        raise RuntimeError("Simulated memory optimizer edge-case error")
    except Exception as e:
        return {"edge_case": str(e)}


@mem_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})


# --- CI/CD test suite ---
import unittest


class TestMemoryOptimizerAPI(unittest.TestCase):
    def test_status(self):
        result = memory_optimizer.check_memory_usage()
        assert "memory_mb" in result

    def test_optimization(self):
        memory_optimizer.force_garbage_collection()
        result = memory_optimizer.check_memory_usage()
        assert "memory_mb" in result


if __name__ == "__main__":
    if "streamlit" in sys.argv[0]:
        # Only run Streamlit UI if explicitly called
        pass
    elif "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        uvicorn.run("memory_cleanup_optimizer:mem_api", host="0.0.0.0", port=8505, reload=True)
