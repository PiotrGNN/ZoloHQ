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

    def __init__(self, max_size_mb: int = 100, db_path: str = "cache_analytics.db"):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.db_path = Path(db_path)

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


if __name__ == "__main__":
    # Test the caching system
    import random

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("api_cache_system_test")

    logger.info("Testing API Caching System...")

    # Create cache and wrapper
    cache = IntelligentCache(max_size_mb=10)
    api_wrapper = CachedAPIWrapper(cache)

    # Test endpoints
    test_endpoints = [
        "/api/trading/statistics",
        "/api/trading/positions",
        "/api/market/tickers",
        "/api/trading/history",
    ]

    # Simulate API usage
    logger.info("Simulating API usage...")
    for i in range(20):
        endpoint = random.choice(test_endpoints)
        params = {"test": i, "random": random.randint(1, 100)}
        data, cache_hit, response_time = api_wrapper.get(endpoint, params)
        logger.info(
            f"Request {i+1}: {endpoint} - Cache: {'HIT' if cache_hit else 'MISS'} - Time: {response_time*1000:.1f}ms"
        )

    # Print statistics
    logger.info("Cache Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("Endpoint Statistics:")
    endpoint_stats = cache.get_endpoint_stats()
    for endpoint, stats in endpoint_stats.items():
        logger.info(f"  {endpoint}: {stats}")

    # Test edge-case: błąd zapisu do bazy
    def test_db_permission_error():
        """Testuje obsługę błędu zapisu do bazy cache_analytics."""
        import tempfile, os, stat
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        os.chmod(temp_file.name, 0)
        try:
            cache = IntelligentCache(db_path=temp_file.name)
            cache._init_analytics_db()
        except Exception as e:
            print(f"FAIL: Unexpected exception: {e}")
        else:
            print("OK: PermissionError handled gracefully.")
        os.unlink(temp_file.name)

    test_db_permission_error()

# CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml
# (TODO usunięty po wdrożeniu automatyzacji)
