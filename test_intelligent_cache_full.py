"""
Full test suite for IntelligentCache and CachedAPIWrapper
Covers: hit/miss, eviction, TTL, size violation, bypass, endpoint stats, error handling, concurrent access, and stress.
"""
import os
import random
import threading
import time

from api_cache_system import CachedAPIWrapper, IntelligentCache


def test_cache_hit_and_miss():
    cache = IntelligentCache(max_size_mb=1)
    api = CachedAPIWrapper(cache)
    data1, hit1, _ = api.get("/api/test/hitmiss", {"a": 1})
    data2, hit2, _ = api.get("/api/test/hitmiss", {"a": 1})
    assert not hit1 and hit2
    assert data1 == data2

def test_cache_eviction_lru():
    cache = IntelligentCache(max_size_mb=0.01)
    api = CachedAPIWrapper(cache)
    for i in range(20):
        rand_data = os.urandom(1024)
        api.get(f"/api/test/evict/{i}", {"i": i, "data": rand_data.hex()})
    stats = cache.get_stats()
    assert stats["evictions"] > 0
    assert stats["entries_count"] <= 20

def test_cache_ttl_expiration():
    cache = IntelligentCache(max_size_mb=1)
    api = CachedAPIWrapper(cache)
    cache.set_endpoint_ttl("/api/test/ttl", 1)
    api.get("/api/test/ttl", {"x": 1})
    time.sleep(1.2)
    _, hit, _ = api.get("/api/test/ttl", {"x": 1})
    assert not hit

def test_cache_size_limit():
    cache = IntelligentCache(max_size_mb=0.001)
    CachedAPIWrapper(cache)
    big_data = os.urandom(20480)
    cache.set("/api/test/size", big_data)
    stats = cache.get_stats()
    assert stats["size_violations"] > 0

def test_cache_bypass():
    cache = IntelligentCache(max_size_mb=1)
    api = CachedAPIWrapper(cache)
    api.get("/api/test/bypass", {"b": 1})
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
    with cache.lock:
        cache.cache["bad"] = type("FakeEntry", (), {"data": b"notgz", "created_at": time.time(), "ttl": 10, "accessed_at": time.time(), "access_count": 1, "size_bytes": 5, "endpoint": "/bad", "priority": 1})()
    data, hit = cache.get("/bad")
    assert data is None and not hit

def worker(api, thread_id, results):
    for i in range(100):
        key = f"/api/test/concurrent/{random.randint(0, 30)}"
        data = os.urandom(256)
        if random.random() < 0.5:
            api.cache.set(key, data)
        else:
            api.get(key)
        if i % 10 == 0:
            time.sleep(0.01)
    results[thread_id] = True

def test_concurrent_cache_stress():
    cache = IntelligentCache(max_size_mb=2)
    api = CachedAPIWrapper(cache)
    threads = []
    results = [False] * 10
    for t in range(10):
        th = threading.Thread(target=worker, args=(api, t, results))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    assert all(results)
    stats = cache.get_stats()
    assert stats["hits"] > 0 or stats["misses"] > 0
    assert stats["entries_count"] > 0

def test_cache_stress_high_volume():
    cache = IntelligentCache(max_size_mb=2)
    api = CachedAPIWrapper(cache)
    for i in range(1000):
        key = f"/api/test/stress/{i%50}"
        data = os.urandom(128)
        if i % 2 == 0:
            api.cache.set(key, data)
        else:
            api.get(key)
    stats = cache.get_stats()
    assert stats["hits"] > 0
    assert stats["misses"] > 0
    assert stats["entries_count"] > 0
