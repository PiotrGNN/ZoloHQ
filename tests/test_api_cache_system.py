import pytest
import os
from api_cache_system import IntelligentCache, CacheEntry
import tempfile
import shutil

def test_cache_entry_lifecycle():
    cache = IntelligentCache(max_size_mb=1)
    entry = CacheEntry(
        key="testkey",
        data={"foo": "bar"},
        created_at=0,
        accessed_at=0,
        access_count=0,
        size_bytes=10,
        ttl=1,
        endpoint="/api/test"
    )
    cache._store_entry(entry)
    assert cache.get("testkey") is not None
    cache.delete("testkey")
    assert cache.get("testkey") is None

def test_cache_expiry():
    cache = IntelligentCache(max_size_mb=1)
    entry = CacheEntry(
        key="expirekey",
        data="expire",
        created_at=0,
        accessed_at=0,
        access_count=0,
        size_bytes=10,
        ttl=0.01,
        endpoint="/api/test"
    )
    cache._store_entry(entry)
    import time
    time.sleep(0.02)
    assert cache.get("expirekey") is None

def test_cache_db_permission_error(tmp_path):
    db_path = tmp_path / "test_cache.db"
    db_path.write_text("")
    os.chmod(db_path, 0o000)
    try:
        cache = IntelligentCache(max_size_mb=1, db_path=str(db_path))
        with pytest.raises(Exception):
            cache._save_analytics()
    finally:
        os.chmod(db_path, 0o666)
        db_path.unlink()
