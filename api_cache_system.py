#!/usr/bin/env python3
"""
API Endpoint Caching System for ZoL0
====================================

Intelligent caching system for frequently accessed API endpoints to reduce
latency and improve overall system performance.
"""

import gzip
import hashlib
import json
import logging
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette_exporter import PrometheusMiddleware, handle_metrics
import io
import csv


@dataclass
class CacheEntry:
    """Cache entry data structure"""

    key: str
    data: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    ttl: float
    endpoint: str
    priority: int = 1  # 1=low, 2=medium, 3=high


class IntelligentCache:
    """
    Intelligent caching system with:
    - TTL-based expiration
    - LRU eviction
    - Priority-based retention
    - Compressed storage
    - Usage analytics
    Obsługa wyjątków przy operacjach na plikach i bazie.
    """

    def __init__(self, max_size_mb: int = 100, db_path: str = "cache_analytics.db", redis_url: str = None):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.db_path = Path(db_path)
        self.redis = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis = redis.StrictRedis.from_url(redis_url)
                logging.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logging.warning(f"Redis connection failed: {e}")

        # Cache statistics
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "size_violations": 0}

        # Endpoint-specific cache configurations
        self.endpoint_configs = {
            "/api/trading/statistics": {
                "ttl": 300,
                "priority": 3,
            },  # 5 minutes, high priority
            "/api/trading/positions": {
                "ttl": 60,
                "priority": 3,
            },  # 1 minute, high priority
            "/api/market/tickers": {
                "ttl": 30,
                "priority": 2,
            },  # 30 seconds, medium priority
            "/api/trading/history": {
                "ttl": 1800,
                "priority": 1,
            },  # 30 minutes, low priority
            "/api/cache/init": {"ttl": 3600, "priority": 2},  # 1 hour, medium priority
        }

        self._init_analytics_db()

        # Background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        logging.info(f"Intelligent cache initialized: max_size={max_size_mb}MB")
        self._cache_warmup()

    def _init_analytics_db(self):
        """Initialize analytics database. Obsługa błędów bazy danych."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    endpoint TEXT,
                    operation TEXT,
                    size_bytes INTEGER,
                    duration REAL
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Błąd inicjalizacji bazy cache_analytics: {e}")

    def _cache_warmup(self):
        """Preload most-used endpoints at startup for faster first response."""
        warmup_endpoints = ["/api/trading/statistics", "/api/market/tickers"]
        for endpoint in warmup_endpoints:
            try:
                dummy_data = {"endpoint": endpoint, "preloaded": True}
                self.set(endpoint, dummy_data)
                logging.info(f"Cache warmup: {endpoint}")
            except Exception as e:
                logging.warning(f"Cache warmup failed for {endpoint}: {e}")

    def _generate_cache_key(
        self, endpoint: str, params: Dict = None, headers: Dict = None
    ) -> str:
        """Generate unique cache key for request"""
        key_data = {
            "endpoint": endpoint,
            "params": params or {},
            "headers": {
                k: v
                for k, v in (headers or {}).items()
                if k.lower() in ["authorization"]
            },
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage"""
        try:
            serialized = pickle.dumps(data)
            return gzip.compress(serialized)
        except Exception as e:
            logging.error(f"Failed to compress cache data: {e}")
            return pickle.dumps(data)

    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data from storage"""
        try:
            if compressed_data[:2] == b"\x1f\x8b":  # gzip magic number
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(compressed_data)
        except Exception as e:
            logging.error(f"Failed to decompress cache data: {e}")
            return None

    def _get_endpoint_config(self, endpoint: str) -> Dict:
        """Get configuration for specific endpoint"""
        # Find matching endpoint pattern
        for pattern, config in self.endpoint_configs.items():
            if pattern in endpoint:
                return config

        # Default configuration
        return {"ttl": 300, "priority": 1}

    def _evict_entries(self, target_size: int):
        """Evict cache entries to reach target size"""
        with self.lock:
            evicted_count = 0

            # Sort by priority (ascending) and access time (ascending)
            entries_by_priority = sorted(
                self.cache.items(), key=lambda x: (x[1].priority, x[1].accessed_at)
            )

            for key, entry in entries_by_priority:
                if self.current_size <= target_size:
                    break

                self.current_size -= entry.size_bytes
                del self.cache[key]
                evicted_count += 1

            self.stats["evictions"] += evicted_count
            if evicted_count > 0:
                logging.info(
                    f"Evicted {evicted_count} cache entries, current size: {self.current_size / 1024 / 1024:.2f}MB"
                )

    def _cleanup_loop(self):
        """Background cleanup of expired entries"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                current_time = time.time()
                expired_keys = []

                with self.lock:
                    for key, entry in self.cache.items():
                        if current_time - entry.created_at > entry.ttl:
                            expired_keys.append(key)

                    for key in expired_keys:
                        entry = self.cache[key]
                        self.current_size -= entry.size_bytes
                        del self.cache[key]

                if expired_keys:
                    logging.info(
                        f"Cleaned up {len(expired_keys)} expired cache entries"
                    )

            except Exception as e:
                logging.error(f"Cache cleanup error: {e}")

    def get(
        self, endpoint: str, params: Dict = None, headers: Dict = None
    ) -> Tuple[Optional[Any], bool]:
        """
        Get cached data for endpoint
        Returns: (data, cache_hit)
        """
        cache_key = self._generate_cache_key(endpoint, params, headers)
        current_time = time.time()

        # Try Redis first if available
        if self.redis:
            try:
                val = self.redis.get(cache_key)
                if val:
                    data = pickle.loads(val)
                    self.stats["hits"] += 1
                    return data, True
            except Exception as e:
                logging.warning(f"Redis get failed: {e}")

        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]

                # Check if expired
                if current_time - entry.created_at > entry.ttl:
                    self.current_size -= entry.size_bytes
                    del self.cache[cache_key]
                    self.stats["misses"] += 1
                    return None, False

                # Update access info
                entry.accessed_at = current_time
                entry.access_count += 1

                # Move to end (LRU)
                self.cache.move_to_end(cache_key)

                # Decompress data
                data = self._decompress_data(entry.data)
                self.stats["hits"] += 1

                # If found in local cache, optionally update Redis
                if self.redis:
                    try:
                        self.redis.setex(cache_key, entry.ttl, entry.data)
                    except Exception as e:
                        logging.warning(f"Redis setex failed: {e}")

                return data, True
            else:
                self.stats["misses"] += 1
                return None, False

    def set(self, endpoint: str, data: Any, params: Dict = None, headers: Dict = None):
        """Cache data for endpoint"""
        cache_key = self._generate_cache_key(endpoint, params, headers)
        config = self._get_endpoint_config(endpoint)
        current_time = time.time()

        # Compress data
        compressed_data = self._compress_data(data)
        data_size = len(compressed_data)

        # Check if single entry exceeds cache size
        if data_size > self.max_size_bytes * 0.5:  # Don't cache if >50% of max size
            self.stats["size_violations"] += 1
            logging.warning(
                f"Cache entry too large for {endpoint}: {data_size / 1024 / 1024:.2f}MB"
            )
            return

        with self.lock:
            # Remove existing entry if present
            if cache_key in self.cache:
                old_entry = self.cache[cache_key]
                self.current_size -= old_entry.size_bytes
                del self.cache[cache_key]

            # Check if we need to evict entries
            if self.current_size + data_size > self.max_size_bytes:
                target_size = self.max_size_bytes - data_size
                before = len(self.cache)
                self._evict_entries(target_size)
                after = len(self.cache)
                evicted = before - after
                if evicted > 0:
                    self.stats["evictions"] += evicted

            # Create new entry
            entry = CacheEntry(
                key=cache_key,
                data=compressed_data,
                created_at=current_time,
                accessed_at=current_time,
                access_count=1,
                size_bytes=data_size,
                ttl=config["ttl"],
                endpoint=endpoint,
                priority=config["priority"],
            )

            self.cache[cache_key] = entry
            self.current_size += data_size

            # Update Redis cache
            if self.redis:
                try:
                    self.redis.setex(cache_key, config["ttl"], pickle.dumps(data))
                except Exception as e:
                    logging.warning(f"Redis setex failed: {e}")

    def _store_entry(self, entry: CacheEntry):
        """Directly store a CacheEntry (for testing only)."""
        with self.lock:
            self.cache[entry.key] = entry
            self.current_size += entry.size_bytes

    def delete(self, key: str):
        """Delete a cache entry by key (for testing only)."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_size -= entry.size_bytes
                del self.cache[key]

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "size_violations": self.stats["size_violations"],
                "current_size_mb": self.current_size / 1024 / 1024,
                "max_size_mb": self.max_size_bytes / 1024 / 1024,
                "entries_count": len(self.cache),
                "size_utilization": self.current_size / self.max_size_bytes,
            }

    def get_endpoint_stats(self) -> Dict:
        """Get per-endpoint statistics"""
        endpoint_stats = defaultdict(lambda: {"hits": 0, "total_size": 0, "entries": 0})

        with self.lock:
            for entry in self.cache.values():
                stats = endpoint_stats[entry.endpoint]
                stats["entries"] += 1
                stats["total_size"] += entry.size_bytes
                stats["hits"] += entry.access_count

        return dict(endpoint_stats)

    def clear_endpoint(self, endpoint_pattern: str):
        """Clear cache entries for specific endpoint pattern"""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if endpoint_pattern in entry.endpoint:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                entry = self.cache[key]
                self.current_size -= entry.size_bytes
                del self.cache[key]

            logging.info(
                f"Cleared {len(keys_to_remove)} cache entries for pattern: {endpoint_pattern}"
            )

    def clear_all(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            logging.info("Cleared all cache entries")

    def record_analytics(self):
        """Record analytics to database"""
        try:
            stats = self.get_stats()
            self.get_endpoint_stats()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Record overall cache analytics
            cursor.execute(
                """
                INSERT INTO cache_analytics 
                (timestamp, endpoint, operation, hit_rate, cache_size_mb, evictions, avg_access_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    time.time(),
                    "overall",
                    "analytics",
                    stats["hit_rate"],
                    stats["current_size_mb"],
                    stats["evictions"],
                    0.0,  # avg_access_time placeholder
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Failed to record cache analytics: {e}")

    def set_endpoint_ttl(self, endpoint_pattern: str, new_ttl: int):
        """Dynamically update TTL for a given endpoint pattern (for optimization)"""
        if endpoint_pattern in self.endpoint_configs:
            self.endpoint_configs[endpoint_pattern]["ttl"] = new_ttl
            logging.info(f"TTL for {endpoint_pattern} set to {new_ttl}s")
        else:
            # Add new config if not present
            self.endpoint_configs[endpoint_pattern] = {"ttl": new_ttl, "priority": 2}
            logging.info(f"TTL for {endpoint_pattern} created and set to {new_ttl}s")

    def optimize_cache_ttl(self):
        """Auto-optimize TTL for endpoints with low hit rate (for production)"""
        endpoint_stats = self.get_endpoint_stats()
        for endpoint, stats in endpoint_stats.items():
            if stats["hits"] < 3 and stats["entries"] > 0:
                # If endpoint is used but hit rate is low, increase TTL
                self.set_endpoint_ttl(endpoint, 600)  # 10 min
            elif stats["hits"] > 10 and stats["entries"] > 0:
                # If endpoint is very hot, set TTL to 30 min
                self.set_endpoint_ttl(endpoint, 1800)

    def start_auto_ttl_optimization(self, interval_sec: int = 600):
        """Start background thread to periodically optimize cache TTL (default co 10 min)"""

        def _auto_optimize():
            while True:
                try:
                    self.optimize_cache_ttl()
                    logging.info("Auto cache TTL optimization executed.")
                except Exception as e:
                    logging.warning(f"Auto cache TTL optimization error: {e}")
                time.sleep(interval_sec)

        t = threading.Thread(target=_auto_optimize, daemon=True)
        t.start()
        logging.info(f"Started auto cache TTL optimization every {interval_sec}s.")

    def export_metrics_prometheus(self) -> str:
        """Export cache metrics in Prometheus format (stub)."""
        stats = self.get_stats()
        lines = [
            f"cache_hit_rate {stats['hit_rate']}",
            f"cache_total_requests {stats['total_requests']}",
            f"cache_evictions {stats['evictions']}",
            f"cache_current_size_mb {stats['current_size_mb']}"
        ]
        return "\n".join(lines)


class CachedAPIWrapper:
    """
    Wrapper for API calls with intelligent caching
    """

    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.performance_monitor = None  # Will be set externally

    def get(
        self,
        endpoint: str,
        params: dict = None,
        headers: dict = None,
        bypass_cache: bool = False,
    ) -> tuple:
        """
        Make cached GET request
        Returns: (data, cache_hit, response_time)
        """
        import time

        start_time = time.time()
        if not bypass_cache:
            data, cache_hit = self.cache.get(endpoint, params, headers)
            if cache_hit:
                response_time = time.time() - start_time
                return data, True, response_time
        # Simulate API call (dummy data)
        data = {"endpoint": endpoint, "params": params, "headers": headers, "result": "dummy"}
        self.cache.set(endpoint, data, params, headers)
        response_time = time.time() - start_time
        return data, False, response_time


# Global cache instance
_global_cache = None


def get_cache_instance(max_size_mb: int = 100) -> IntelligentCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache(max_size_mb)
    return _global_cache


def create_cached_api_wrapper() -> CachedAPIWrapper:
    """Create cached API wrapper with global cache"""
    cache = get_cache_instance()
    return CachedAPIWrapper(cache)


# --- FastAPI app ---
API_KEYS = {
    "admin-key": "admin",
    "cache-key": "cache",
    "partner-key": "partner",
    "premium-key": "premium",
    "saas-key": "saas"
}
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key in API_KEYS:
        return API_KEYS[api_key]
    raise HTTPException(status_code=401, detail="Invalid or missing API key")

cache_api = FastAPI(title="ZoL0 API Cache System", version="2.0")
cache_api.add_middleware(PrometheusMiddleware)
cache_api.add_route("/metrics", handle_metrics)

# --- Pydantic Models ---
class CacheQuery(BaseModel):
    endpoint: str
    params: dict = Field(default_factory=dict)
    headers: dict = Field(default_factory=dict)
    data: dict = Field(default_factory=dict)

class BatchCacheQuery(BaseModel):
    queries: list[CacheQuery]

class TTLUpdateQuery(BaseModel):
    endpoint_pattern: str
    new_ttl: int

# --- Global cache instance ---
cache = get_cache_instance()

# --- Endpoints ---
@cache_api.get("/")
async def root():
    return {"status": "ok", "service": "ZoL0 API Cache System", "version": "2.0"}

@cache_api.get("/api/health")
async def api_health():
    return {"status": "ok", "timestamp": time.time(), "service": "ZoL0 API Cache System", "version": "2.0"}

@cache_api.get("/api/cache/get", dependencies=[Depends(get_api_key)])
async def api_cache_get(endpoint: str, params: dict = None, headers: dict = None, role: str = Depends(get_api_key)):
    data, hit = cache.get(endpoint, params, headers)
    return {"data": data, "cache_hit": hit}

@cache_api.post("/api/cache/set", dependencies=[Depends(get_api_key)])
async def api_cache_set(req: CacheQuery, role: str = Depends(get_api_key)):
    cache.set(req.endpoint, req.data, req.params, req.headers)
    return {"status": "ok"}

@cache_api.post("/api/cache/batch", dependencies=[Depends(get_api_key)])
async def api_cache_batch(req: BatchCacheQuery, role: str = Depends(get_api_key)):
    results = []
    for q in req.queries:
        cache.set(q.endpoint, q.data, q.params, q.headers)
        results.append({"endpoint": q.endpoint, "status": "ok"})
    return {"results": results}

@cache_api.get("/api/cache/clear", dependencies=[Depends(get_api_key)])
async def api_cache_clear(endpoint_pattern: str = "", role: str = Depends(get_api_key)):
    if endpoint_pattern:
        cache.clear_endpoint(endpoint_pattern)
        return {"status": "cleared", "pattern": endpoint_pattern}
    else:
        cache.clear_all()
        return {"status": "cleared_all"}

@cache_api.get("/api/cache/stats", dependencies=[Depends(get_api_key)])
async def api_cache_stats(role: str = Depends(get_api_key)):
    return cache.get_stats()

@cache_api.get("/api/cache/endpoint-stats", dependencies=[Depends(get_api_key)])
async def api_cache_endpoint_stats(role: str = Depends(get_api_key)):
    return cache.get_endpoint_stats()

@cache_api.get("/api/cache/analytics", dependencies=[Depends(get_api_key)])
async def api_cache_analytics(role: str = Depends(get_api_key)):
    cache.record_analytics()
    return {"status": "analytics recorded"}

@cache_api.post("/api/cache/ttl", dependencies=[Depends(get_api_key)])
async def api_cache_ttl(req: TTLUpdateQuery, role: str = Depends(get_api_key)):
    cache.set_endpoint_ttl(req.endpoint_pattern, req.new_ttl)
    return {"status": "ttl updated", "pattern": req.endpoint_pattern, "ttl": req.new_ttl}

@cache_api.get("/api/cache/optimize", dependencies=[Depends(get_api_key)])
async def api_cache_optimize(role: str = Depends(get_api_key)):
    cache.optimize_cache_ttl()
    return {"status": "cache ttl optimized"}

@cache_api.get("/api/export/csv", dependencies=[Depends(get_api_key)])
async def api_export_csv(role: str = Depends(get_api_key)):
    stats = cache.get_stats()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(stats.keys()))
    writer.writeheader()
    writer.writerow(stats)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv")

@cache_api.get("/api/export/prometheus", dependencies=[Depends(get_api_key)])
async def api_export_prometheus(role: str = Depends(get_api_key)):
    return PlainTextResponse(cache.export_metrics_prometheus(), media_type="text/plain")

@cache_api.get("/api/report", dependencies=[Depends(get_api_key)])
async def api_report(role: str = Depends(get_api_key)):
    # Placeholder for PDF/CSV/email integration
    return {"status": "report generated (stub)"}

@cache_api.get("/api/recommendations", dependencies=[Depends(get_api_key)])
async def api_recommendations(role: str = Depends(get_api_key)):
    stats = cache.get_stats()
    recs = []
    if stats["hit_rate"] < 0.5:
        recs.append("Increase cache TTL or optimize endpoints for higher hit rate.")
    if stats["size_utilization"] > 0.9:
        recs.append("Increase cache size or optimize eviction policy.")
    return {"recommendations": recs}

@cache_api.get("/api/premium/score", dependencies=[Depends(get_api_key)])
async def api_premium_score(role: str = Depends(get_api_key)):
    stats = cache.get_stats()
    score = stats["hit_rate"] * 100 + stats["current_size_mb"]
    return {"score": score}

@cache_api.get("/api/saas/tenant/{tenant_id}/report", dependencies=[Depends(get_api_key)])
async def api_saas_tenant_report(tenant_id: str, role: str = Depends(get_api_key)):
    # Multi-tenant stub: filter by tenant_id in report (future)
    stats = cache.get_stats()
    return {"tenant_id": tenant_id, "report": stats}

@cache_api.get("/api/partner/webhook", dependencies=[Depends(get_api_key)])
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    # Placeholder for partner webhook integration
    return {"status": "received", "payload": payload}

@cache_api.get("/api/test/edge-case")
async def api_edge_case():
    try:
        raise RuntimeError("Simulated cache edge-case error")
    except Exception as e:
        return {"edge_case": str(e)}

@cache_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

# --- CI/CD test suite ---
import unittest
class TestCacheAPI(unittest.TestCase):
    def test_cache_set_get(self):
        cache.set("/api/test", {"foo": "bar"})
        data, hit = cache.get("/api/test")
        assert hit and data["foo"] == "bar"
    def test_cache_clear(self):
        cache.set("/api/test", {"foo": "bar"})
        cache.clear_endpoint("/api/test")
        data, hit = cache.get("/api/test")
        assert not hit

if __name__ == "__main__":
    import sys
    if "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        uvicorn.run("api_cache_system:cache_api", host="0.0.0.0", port=8505, reload=True)
