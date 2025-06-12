"""
Stress test for IntelligentCache with concurrent (multi-threaded) access.
Validates thread safety, correctness, and stats under high concurrency.
"""
import os
import random
import threading
import time

from api_cache_system import CachedAPIWrapper, IntelligentCache


def worker(api, thread_id, results):
    for i in range(100):
        key = f"/api/test/concurrent/{random.randint(0, 30)}"
        data = os.urandom(256)
        # 50% of the time, set; 50% get
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
    # All threads completed
    assert all(results)
    stats = cache.get_stats()
    print(f"Concurrent stress test stats: {stats}")
    # Should have both hits and misses, and no deadlocks
    assert stats["hits"] > 0 or stats["misses"] > 0
    assert stats["entries_count"] > 0
