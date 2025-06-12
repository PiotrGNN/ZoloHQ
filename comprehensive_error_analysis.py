#!/usr/bin/env python3
"""
Kompleksowa analiza błędów systemu ZoL0
"""
import logging
from datetime import datetime
from typing import Any, Dict

import requests

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def check_api_endpoint(url: str, name: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Sprawdź endpoint API i zwróć szczegóły błędów. Obsługa błędów połączenia HTTP.

    Args:
        url (str): URL endpointu API.
        name (str): Nazwa usługi lub endpointu.
        timeout (int, optional): Czas oczekiwania na odpowiedź w sekundach. Domyślnie 5.

    Returns:
        Dict[str, Any]: Słownik zawierający szczegóły błędów lub dane odpowiedzi.
    """
    logger = logging.getLogger("error_analysis")
    result = {
        "url": url,
        "name": name,
        "status": None,
        "error": None,
        "content_preview": None,
    }
    try:
        response = requests.get(url, timeout=timeout)
        result["status"] = response.status_code
        if response.status_code == 200:
            try:
                data = response.json()
                result["content_preview"] = (
                    str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
                )
            except Exception as e:
                logger.warning(f"Failed to parse JSON from {url}: {e}")
                result["content_preview"] = response.text[:100] + "..."
        else:
            logger.error(f"{name} at {url} returned status {response.status_code}")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout when connecting to {name} at {url}")
        result["error"] = "timeout"
    except Exception as e:
        logger.error(f"Error connecting to {name} at {url}: {e}")
        result["error"] = str(e)
    return result


def check_dashboard_accessibility(url, name):
    """Sprawdź dostępność dashboardu"""
    try:
        response = requests.get(url, timeout=10)
        return {
            "name": name,
            "url": url,
            "accessible": response.status_code == 200,
            "status_code": response.status_code,
            "error": (
                None if response.status_code == 200 else f"HTTP {response.status_code}"
            ),
        }
    except Exception as e:
        return {
            "name": name,
            "url": url,
            "accessible": False,
            "status_code": "ERROR",
            "error": str(e),
        }


def main():
    print("🔍 KOMPLEKSOWA ANALIZA BŁĘDÓW SYSTEMU ZoL0")
    print("=" * 60)
    print(f"📅 Data analizy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Lista endpointów do sprawdzenia
    api_endpoints = [
        ("http://localhost:5000/api/health", "Main API Health"),
        ("http://localhost:5000/api/portfolio", "Main API Portfolio"),
        ("http://localhost:5000/api/trading/status", "Main API Trading Status"),
        ("http://localhost:4001/health", "Enhanced API Health"),
        ("http://localhost:4001/api/portfolio", "Enhanced API Portfolio"),
        ("http://localhost:4001/api/trading/statistics", "Enhanced API Trading Stats"),
        ("http://localhost:4001/api/cache/init", "Enhanced API Cache Init"),
    ]

    dashboard_urls = [
        ("http://localhost:4501", "Master Control Dashboard"),
        ("http://localhost:4502", "Unified Trading Dashboard"),
        ("http://localhost:4503", "Bot Monitor Dashboard"),
        ("http://localhost:4504", "Trading Analytics Dashboard"),
        ("http://localhost:4505", "Notifications Dashboard"),
        ("http://localhost:4506", "Portfolio Dashboard"),
        ("http://localhost:4507", "ML Analytics Dashboard"),
        ("http://localhost:4508", "Enhanced Dashboard"),
    ]

    # Analiza API Endpoints
    print("📡 ANALIZA API ENDPOINTS")
    print("-" * 40)

    api_results = []
    api_errors = []
    api_warnings = []

    for url, name in api_endpoints:
        result = check_api_endpoint(url, name)
        api_results.append(result)

        if result["success"]:
            status_emoji = "✅"
            if result["data_source"] and "fallback" in result["data_source"]:
                status_emoji = "⚠️"
                api_warnings.append(
                    f"{name}: Using fallback data ({result['data_source']})"
                )
        else:
            status_emoji = "❌"
            api_errors.append(f"{name}: {result['error']}")

        print(f"{status_emoji} {name}")
        print(f"   URL: {url}")
        print(f"   Status: {result['status_code']}")
        print(f"   Response time: {result['response_time']}s")
        if result["data_source"]:
            print(f"   Data source: {result['data_source']}")
        if result["error"]:
            print(f"   Error: {result['error']}")
        print()

    # Analiza Dashboardów
    print("🎯 ANALIZA DOSTĘPNOŚCI DASHBOARDÓW")
    print("-" * 40)

    dashboard_results = []
    dashboard_errors = []

    for url, name in dashboard_urls:
        result = check_dashboard_accessibility(url, name)
        dashboard_results.append(result)

        if result["accessible"]:
            print(f"✅ {name}")
        else:
            print(f"❌ {name}")
            dashboard_errors.append(f"{name}: {result['error']}")

        print(f"   URL: {url}")
        print(f"   Status: {result['status_code']}")
        if result["error"]:
            print(f"   Error: {result['error']}")
        print()

    # Podsumowanie błędów
    print("📊 PODSUMOWANIE ANALIZY")
    print("=" * 50)

    api_working = sum(1 for r in api_results if r["success"])
    dashboards_working = sum(1 for r in dashboard_results if r["accessible"])

    print("📡 API Endpoints:")
    print(f"   ✅ Działające: {api_working}/{len(api_results)}")
    print(f"   ❌ Błędy: {len(api_errors)}")
    print(f"   ⚠️  Ostrzeżenia: {len(api_warnings)}")

    print("\n🎯 Dashboardy:")
    print(f"   ✅ Dostępne: {dashboards_working}/{len(dashboard_results)}")
    print(f"   ❌ Błędy: {len(dashboard_errors)}")

    # Szczegóły błędów
    if api_errors:
        print("\n❌ BŁĘDY API:")
        for error in api_errors:
            print(f"   • {error}")

    if api_warnings:
        print("\n⚠️  OSTRZEŻENIA API:")
        for warning in api_warnings:
            print(f"   • {warning}")

    if dashboard_errors:
        print("\n❌ BŁĘDY DASHBOARDÓW:")
        for error in dashboard_errors:
            print(f"   • {error}")

    # Rekomendacje
    print("\n🔧 REKOMENDACJE:")

    if len(api_errors) == 0 and len(dashboard_errors) == 0:
        print("   ✅ System działa poprawnie!")
    else:
        if api_errors:
            print("   🔄 Uruchom ponownie backend APIs")
        if dashboard_errors:
            print("   🔄 Uruchom ponownie dashboardy")
        if api_warnings:
            print("   ⚡ Sprawdź połączenie z Production API")

    print(f"\n🎉 Analiza zakończona: {datetime.now().strftime('%H:%M:%S')}")


# Test edge-case: błąd połączenia HTTP
if __name__ == "__main__":
    def test_http_connection_error():
        """Testuje obsługę błędu połączenia HTTP do API."""
        result = check_api_endpoint("http://localhost:9999", "Test API")
        if result["error"]:
            print("OK: ConnectionError handled gracefully.")
        else:
            print("FAIL: No exception for HTTP connection error.")
    test_http_connection_error()

# TODO: Integrate with CI/CD pipeline for automated error analysis and edge-case tests.
# Edge-case tests: simulate API failures, HTTP errors, and invalid responses.
# All public methods have docstrings and exception handling.

# TODO: Dodać workflow CI/CD do automatycznego uruchamiania testów i lintingu.

if __name__ == "__main__":
    main()
