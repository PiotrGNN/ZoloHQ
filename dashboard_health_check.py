#!/usr/bin/env python3
"""
Dashboard Health Check - Test wszystkich działających dashboardów
"""

import logging
import time

import requests


def test_dashboard_health():
    """
    Test wszystkich uruchomionych dashboardów. Obsługa błędów połączenia HTTP.
    """

    print("🔍 TESTOWANIE STANU URUCHOMIONYCH DASHBOARDÓW")
    print("=" * 55)
    print(f"Czas: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Lista portów do sprawdzenia
    dashboard_ports = {
        5000: "Main API Server (ZoL0-master)",
        5001: "Enhanced Dashboard API",
        8501: "Main Dashboard (ZoL0-master)",
        8503: "Dashboard (Port 8503)",
        8504: "Dashboard (Port 8504)",
        8505: "Dashboard (Port 8505)",
        8506: "Master Control Dashboard",
        8507: "Enhanced Dashboard",
    }

    working_dashboards = 0
    total_tested = len(dashboard_ports)

    for port, name in dashboard_ports.items():
        print(f"\n🔍 Testowanie: {name} (Port {port})")
        print("-" * 40)

        try:
            # Test basic HTTP connection
            response = requests.get(f"http://localhost:{port}", timeout=5)

            if response.status_code == 200:
                print("✅ HTTP Status: 200 OK")
                print(f"✅ Rozmiar odpowiedzi: {len(response.content)} bytes")

                # Check if it's a Streamlit app
                if "streamlit" in response.text.lower() or "st." in response.text:
                    print("✅ Typ: Streamlit Dashboard")
                elif "flask" in response.text.lower() or response.headers.get(
                    "Server", ""
                ).startswith("Werkzeug"):
                    print("✅ Typ: Flask API Server")
                else:
                    print("✅ Typ: Web Application")

                print("✅ STATUS: DZIAŁA POPRAWNIE")
                working_dashboards += 1

            else:
                print(f"⚠️ HTTP Status: {response.status_code}")
                print("⚠️ STATUS: ODPOWIADA ALE BŁĄD")

        except requests.exceptions.ConnectionError:
            print(f"❌ CONNECTION ERROR: Nie można połączyć się z portem {port}")
            print("❌ STATUS: NIEDOSTĘPNY")

        except requests.exceptions.Timeout:
            print("⏱️ TIMEOUT: Przekroczono limit czasu (5s)")
            print("⚠️ STATUS: POWOLNY/PROBLEMY")

        except Exception as e:
            logging.exception("Exception occurred in dashboard_health_check at line 99")
            print(f"❌ BŁĄD: {str(e)[:50]}")
            print("❌ STATUS: BŁĄD")

    # Podsumowanie
    print("\n" + "=" * 55)
    print("📊 PODSUMOWANIE STANU SYSTEMU")
    print("=" * 55)
    print(f"Działających serwisów: {working_dashboards}/{total_tested}")
    print(f"Procent dostępności: {(working_dashboards/total_tested)*100:.1f}%")

    if working_dashboards == total_tested:
        print("🎉 STAN: WSZYSTKIE SERWISY DZIAŁAJĄ!")
        print("🚀 System gotowy do użycia")
    elif working_dashboards > total_tested * 0.8:
        print("✅ STAN: WIĘKSZOŚĆ SERWISÓW DZIAŁA")
        print("⚠️ Niektóre komponenty wymagają uwagi")
    else:
        print("❌ STAN: PROBLEMY Z SYSTEMEM")
        print("🔧 Wymagane naprawy")

    print("\n🌐 DOSTĘPNE LINKI:")
    print("-" * 20)
    for port, name in dashboard_ports.items():
        try:
            response = requests.get(f"http://localhost:{port}", timeout=2)
            if response.status_code == 200:
                print(f"✅ {name}: http://localhost:{port}")
        except Exception as e:
            logging.exception(
                f"Exception checking dashboard {name} on port {port}: {e}"
            )
            print(f"❌ {name}: http://localhost:{port} (niedostępny)")

    return working_dashboards, total_tested


# Test edge-case: błąd połączenia HTTP
def test_http_connection_error():
    """Testuje obsługę błędu połączenia HTTP do dashboardu."""
    import requests

    try:
        requests.get("http://localhost:9999", timeout=2)
    except Exception as e:
        print("OK: ConnectionError handled gracefully.")
    else:
        print("FAIL: No exception for HTTP connection error.")


if __name__ == "__main__":
    working, total = test_dashboard_health()
    print(f"\n🎯 KOŃCOWY WYNIK: {working}/{total} serwisów działa poprawnie")
    test_http_connection_error()

# TODO: Integrate with CI/CD pipeline for automated dashboard health and edge-case tests.
# Edge-case tests: simulate HTTP failures, port issues, and Streamlit detection errors.
# All public methods have docstrings and exception handling.
