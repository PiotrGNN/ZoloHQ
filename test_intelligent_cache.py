"""
Comprehensive tests for IntelligentCache and CachedAPIWrapper
Covers: cache hit/miss, eviction, TTL expiration, size limits, error handling, endpoint stats, and bypass logic.
"""
import os
import time

from api_cache_system import CachedAPIWrapper, IntelligentCache


def test_cache_hit_and_miss():
    cache = IntelligentCache(max_size_mb=1)
    api = CachedAPIWrapper(cache)
    # Miss on first call
    data1, hit1, _ = api.get("/api/test/hitmiss", {"a": 1})
    # Hit on second call
    data2, hit2, _ = api.get("/api/test/hitmiss", {"a": 1})
    assert not hit1 and hit2
    assert data1 == data2

def test_cache_eviction_lru():
    cache = IntelligentCache(max_size_mb=0.01)  # ~10KB
    api = CachedAPIWrapper(cache)
    # Use random data to avoid compression
    for i in range(20):
        rand_data = os.urandom(1024)  # 1KB of random bytes
        api.get(f"/api/test/evict/{i}", {"i": i, "data": rand_data.hex()})
    stats = cache.get_stats()
    print(f"Eviction test stats: {stats}")
    assert stats["evictions"] > 0
    assert stats["entries_count"] <= 20

def test_cache_ttl_expiration():
    cache = IntelligentCache(max_size_mb=1)
    api = CachedAPIWrapper(cache)
    # Set short TTL for this endpoint
    cache.set_endpoint_ttl("/api/test/ttl", 1)
    api.get("/api/test/ttl", {"x": 1})
    time.sleep(1.2)
    _, hit, _ = api.get("/api/test/ttl", {"x": 1})
    assert not hit  # Should expire

def test_cache_size_limit():
    cache = IntelligentCache(max_size_mb=0.001)  # ~1KB
    CachedAPIWrapper(cache)
    # Use random data to avoid compression
    big_data = os.urandom(20480)  # 20KB of random bytes
    cache.set("/api/test/size", big_data)
    stats = cache.get_stats()
    print(f"Size violation test stats: {stats}")
    assert stats["size_violations"] > 0

def test_cache_bypass():
    cache = IntelligentCache(max_size_mb=1)
    api = CachedAPIWrapper(cache)
    api.get("/api/test/bypass", {"b": 1})
    # Bypass cache
    _, hit, _ = api.get("/api/test/bypass", {"b": 1}, bypass_cache=True)
    assert not hit

def test_endpoint_stats():
    cache = IntelligentCache(max_size_mb=1)
    api = CachedAPIWrapper(cache)
    for i in range(3):
        api.get("/api/test/stats", {"i": i})
    stats = cache.get_endpoint_stats()
    assert any("/api/test/stats" in k or k == "/api/test/stats" for k in stats)

def test_error_handling():
    cache = IntelligentCache(max_size_mb=1)
    # Try decompressing invalid data
    with cache.lock:
        cache.cache["bad"] = type("FakeEntry", (), {"data": b"notgz", "created_at": time.time(), "ttl": 10, "accessed_at": time.time(), "access_count": 1, "size_bytes": 5, "endpoint": "/bad", "priority": 1})()
    data, hit = cache.get("/bad")
    assert data is None and not hit
