import sys
import time

sys.path.append(".")

print("Testing ProductionDataManager directly...")
start = time.time()

try:
    from production_data_manager import ProductionDataManager

    manager = ProductionDataManager()

    print(f"Manager initialization time: {time.time() - start:.2f}s")

    # Test with cache
    start_cache = time.time()
    data_cache = manager.get_portfolio_data(use_cache=True)
    cache_time = time.time() - start_cache
    print(f"get_portfolio_data(use_cache=True) time: {cache_time:.2f}s")
    print(f'Data source: {data_cache.get("data_source", "unknown")}')

    # Test without cache (fresh API call)
    start_fresh = time.time()
    data_fresh = manager.get_portfolio_data(use_cache=False)
    fresh_time = time.time() - start_fresh
    print(f"get_portfolio_data(use_cache=False) time: {fresh_time:.2f}s")
    print(f'Data source: {data_fresh.get("data_source", "unknown")}')

except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.2f}s: {e}")
    import traceback

    traceback.print_exc()
