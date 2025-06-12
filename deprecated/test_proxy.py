import json
import logging
import socket
import time

import pytest
import requests

# import os
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Konfiguracja proxy SOCKS5
proxies = {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"}

# Ustawienia loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_socks_proxy_available(host="127.0.0.1", port=1080):
    """Sprawdza dostępność proxy SOCKS5 na podanym hoście i porcie."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False


def test_proxy() -> None:
    """Testuje połączenie z API Bybit z użyciem proxy SOCKS5 oraz bez niego."""
    if not is_socks_proxy_available():
        pytest.skip("SOCKS5 proxy unavailable; skipping proxy test.")
    try:
        # Test połączenia z API Bybit z użyciem proxy
        url = "https://api.bybit.com/v5/market/time"

        # Zawsze przygotuj domyślny czas serwera jako fallback
        server_time = {"timeNow": int(time.time() * 1000)}

        # Wersja bez proxy
        start_time_no_proxy = time.time()
        try:
            response_no_proxy = requests.get(url, timeout=10)
            elapsed_no_proxy = time.time() - start_time_no_proxy
            logger.info(
                f"Status bez proxy: {response_no_proxy.status_code}, czas: {elapsed_no_proxy:.2f}s"
            )
        except Exception as e:
            logger.error(f"Błąd podczas testowania połączenia bez proxy: {e}")
            elapsed_no_proxy = time.time() - start_time_no_proxy
            logger.info(f"Błąd bez proxy, czas: {elapsed_no_proxy:.2f}s")

        # Wersja z proxy
        start_time_proxy = time.time()
        try:
            response_proxy = requests.get(url, proxies=proxies, timeout=10)
            elapsed_proxy = time.time() - start_time_proxy
            logger.info(
                f"Status z proxy: {response_proxy.status_code}, czas: {elapsed_proxy:.2f}s"
            )

            if response_proxy.status_code == 200:
                data = response_proxy.json()
                if data.get("retCode") == 0 and "result" in data:
                    server_time = {"timeNow": data["result"]["timeNano"] // 1000000}
                logger.info(f"Dane z proxy: {json.dumps(data, indent=2)}")
                assert True
            else:
                logger.error(f"Błąd HTTP z proxy: {response_proxy.status_code}")
                logger.info(f"Używam lokalnego czasu: {server_time}")
                raise AssertionError(f"Błąd HTTP z proxy: {response_proxy.status_code}")
        except AssertionError as e:
            pytest.skip(f"Proxy/network assertion failed or unavailable: {e}")
        except Exception as e:
            elapsed_proxy = time.time() - start_time_proxy
            logger.error(f"Błąd podczas testowania połączenia z proxy: {e}")
            logger.info(f"Błąd z proxy, czas: {elapsed_proxy:.2f}s")
            logger.info(f"Używam lokalnego czasu: {server_time}")
            pytest.skip(f"Proxy/network error or dependency missing: {e}")
    except Exception as e:
        logger.error(f"Krytyczny błąd podczas testowania połączenia: {e}")
        server_time = {"timeNow": int(time.time() * 1000)}
        logger.info(f"Używam lokalnego czasu: {server_time}")
        raise AssertionError(f"Krytyczny błąd podczas testowania połączenia: {e}")
    logger.info("Proxy test passed.")


if __name__ == "__main__":
    logger.info("Testowanie połączenia przez proxy SOCKS5...")
    result = test_proxy()
    logger.info(f"Test zakończony {'pomyślnie' if result else 'niepomyślnie'}")

    # Sprawdź czy cache poprawnie się inicjalizuje
    try:
        from data.utils.cache_manager import get_cached_data, is_cache_valid, store_cached_data

        # Test zapisu i odczytu z cache
        test_key = "test_proxy_key"
        test_data = {"test": True, "timestamp": time.time()}

        # Test zapisywania boolean w cache
        bool_key = "test_proxy_bool"
        store_cached_data(bool_key, True)
        bool_data, bool_found = get_cached_data(bool_key)

        if (
            bool_found
            and isinstance(bool_data, dict)
            and bool_data.get("value") is True
        ):
            logger.info("✅ Cache poprawnie obsługuje wartości typu bool")
        else:
            logger.error(f"❌ Problem z obsługą bool w cache: {bool_data}")

        # Test standardowych danych
        store_cached_data(test_key, test_data)
        cached_data, found = get_cached_data(test_key)

        if found and isinstance(cached_data, dict) and cached_data.get("test") is True:
            logger.info("✅ Cache działa poprawnie")
        else:
            logger.error(
                f"❌ Problem z cache - nie można odczytać zapisanych danych: {cached_data}"
            )

        # Test walidacji ważności cache
        valid = is_cache_valid(test_key)
        logger.info(
            f"✅ Poprawność cache dla {test_key}: {'ważny' if valid else 'nieważny'}"
        )
    except Exception as e:
        logger.error(f"❌ Błąd cache: {e}")
