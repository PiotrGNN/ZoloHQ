#!/usr/bin/env python3
"""
ZoL0 Unified Dashboard Launcher
==============================
Skrypt do uruchamiania zunifikowanego dashboardu ZoL0 Trading System
Łączy wszystkie dashboardy w jeden interfejs z zakładkami
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def check_dependencies():
    """Sprawdź czy wszystkie wymagane pakiety są zainstalowane"""
    required_packages = ["streamlit", "pandas", "numpy", "plotly", "requests"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Brakujące pakiety: {', '.join(missing_packages)}")
        print("Instalacja:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✅ Wszystkie wymagane pakiety są zainstalowane")
    return True


def start_unified_dashboard():
    """Uruchom zunifikowany dashboard"""

    print("🚀 ZoL0 Unified Trading Dashboard")
    print("=" * 50)

    # Sprawdź zależności
    if not check_dependencies():
        return False

    # Sprawdź czy plik dashboardu istnieje
    dashboard_file = Path(__file__).parent / "unified_trading_dashboard.py"
    if not dashboard_file.exists():
        print(f"❌ Nie znaleziono pliku: {dashboard_file}")
        return False

    print(f"📁 Uruchamianie dashboardu: {dashboard_file}")
    print("🌐 URL: http://localhost:8500")
    print("⏹️  Zatrzymanie: Ctrl+C")
    print()

    try:
        # Uruchom Streamlit
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_file),
            "--server.port",
            "8500",
            "--server.address",
            "0.0.0.0",
            "--theme.base",
            "dark",
            "--theme.primaryColor",
            "#667eea",
            "--theme.backgroundColor",
            "#0e1117",
            "--theme.secondaryBackgroundColor",
            "#262730",
            "--theme.textColor",
            "#fafafa",
        ]

        print("🔧 Komenda uruchomienia:")
        print(" ".join(cmd))
        print()

        # Poczekaj chwilę, a potem otwórz przeglądarkę
        def open_browser():
            time.sleep(3)
            try:
                webbrowser.open("http://localhost:8500")
                print("🌐 Dashboard otwarty w przeglądarce")
            except Exception:
                print("⚠️  Otwórz ręcznie: http://localhost:8500")

        import threading

        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()

        # Uruchom Streamlit
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n👋 Dashboard zatrzymany przez użytkownika")
        return True
    except Exception as e:
        print(f"❌ Błąd uruchamiania: {e}")
        return False


def main():
    """Główna funkcja"""

    # Przejdź do katalogu skryptu
    os.chdir(Path(__file__).parent)

    print("🎯 ZoL0 Unified Trading Dashboard Launcher")
    print("==========================================")
    print()

    # Pokaż informacje o systemie
    print("📊 Status systemu:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Katalog: {os.getcwd()}")

    # Sprawdź tryb produkcyjny
    production_mode = os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true"
    if production_mode:
        print("   🟢 Tryb: PRODUKCYJNY (prawdziwe dane Bybit)")
    else:
        print("   🟡 Tryb: DEWELOPERSKI (symulowane dane)")

    print()

    # Uruchom dashboard
    success = start_unified_dashboard()

    if success:
        print("✅ Dashboard zakończył działanie pomyślnie")
    else:
        print("❌ Problem z uruchomieniem dashboardu")
        input("Naciśnij Enter aby zakończyć...")


if __name__ == "__main__":
    main()
