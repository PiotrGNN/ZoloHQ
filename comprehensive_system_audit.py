#!/usr/bin/env python3
"""
comprehensive_system_audit.py
=============================
Kompleksowy audyt systemu tradingowego - sprawdza czy wszystkie komponenty używają prawdziwych danych
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Globalne ignorowanie ostrzeżeń
warnings.filterwarnings("ignore")

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Załaduj zmienne środowiskowe
try:
    from dotenv import load_dotenv

    load_dotenv()
    logger.info("✅ Zmienne środowiskowe załadowane z .env")
except ImportError:
    logger.warning("⚠️ python-dotenv nie jest dostępny, używam zmiennych systemowych")
except Exception as e:
    logger.error(f"⚠️ Błąd ładowania .env: {e}")


class SystemAuditor:
    """Audytor systemu tradingowego"""

    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []

    def audit_environment(self) -> Dict[str, Any]:
        """Audyt konfiguracji środowiska"""
        logger = logging.getLogger("SystemAuditor")
        logger.info("🔍 AUDYT ŚRODOWISKA")
        logger.info("=" * 50)

        env_checks = {
            "TRADING_MODE": os.getenv("TRADING_MODE"),
            "BYBIT_PRODUCTION_ENABLED": os.getenv("BYBIT_PRODUCTION_ENABLED"),
            "BYBIT_TESTNET": os.getenv("BYBIT_TESTNET"),
            "BYBIT_API_KEY": os.getenv("BYBIT_API_KEY"),
            "BYBIT_API_SECRET": os.getenv("BYBIT_API_SECRET"),
        }

        status = True
        details = {}

        for key, value in env_checks.items():
            if key in ["BYBIT_API_KEY", "BYBIT_API_SECRET"]:
                has_value = bool(value and len(value) > 0)
                display_value = "***" if has_value else "BRAK"
            else:
                has_value = bool(value)
                display_value = value or "BRAK"

            result = "✅" if has_value else "❌"
            logger.info(f"{result} {key}: {display_value}")
            details[key] = {"value": display_value, "valid": has_value}

            if not has_value:
                status = False

        # Sprawdź czy trading mode jest na production
        if env_checks.get("TRADING_MODE") != "production":
            status = False
            self.errors.append("TRADING_MODE nie jest ustawiony na 'production'")
            # Sprawdź czy Bybit production jest włączony
        bybit_production = env_checks.get("BYBIT_PRODUCTION_ENABLED", "")
        if bybit_production and bybit_production.lower() != "true":
            status = False
            self.errors.append("BYBIT_PRODUCTION_ENABLED nie jest ustawiony na 'true'")
        elif not bybit_production:
            status = False
            self.errors.append("BYBIT_PRODUCTION_ENABLED nie jest zdefiniowany")

        return {"status": status, "details": details}

    def audit_bybit_connector(self) -> Dict[str, Any]:
        """Audyt połączenia z Bybit"""
        print("\n🔍 AUDYT BYBIT CONNECTOR")
        print("=" * 50)

        try:
            # Import connector
            sys.path.append(str(Path(__file__).parent / "ZoL0-master"))
            from data.execution.bybit_connector import BybitConnector

            # Inicjalizacja
            api_key = os.getenv("BYBIT_API_KEY")
            api_secret = os.getenv("BYBIT_API_SECRET")

            if not api_key or not api_secret:
                print("❌ Brak kluczy API")
                return {"status": False, "error": "Brak kluczy API"}

            # Test połączenia
            connector = BybitConnector(
                api_key=api_key,
                api_secret=api_secret,
                use_testnet=False,  # Force production
            )

            # Test server time
            print("🕒 Test czasu serwera...")
            time_result = connector.get_server_time()
            time_ok = time_result.get("success", False)
            print(
                f"{'✅' if time_ok else '❌'} Czas serwera: {time_result.get('server_time', 'BŁĄD')}"
            )

            # Test wallet balance
            print("💰 Test salda portfela...")
            balance_result = connector.get_wallet_balance()
            balance_ok = balance_result.get("success", False)
            source = balance_result.get("source", "unknown")

            if balance_ok:
                print(f"✅ Saldo pobrane pomyślnie (źródło: {source})")

                # Sprawdź czy to prawdziwe dane API
                if source == "api":
                    print("✅ UŻYWA PRAWDZIWYCH DANYCH API!")
                elif source in ["simulation", "demo", "testnet_simulation"]:
                    print("❌ UŻYWA SYMULOWANYCH DANYCH!")
                    return {
                        "status": False,
                        "error": f"Używa symulowanych danych: {source}",
                    }
                else:
                    print(f"⚠️ Nieznane źródło danych: {source}")

                # Pokaż szczegóły salda
                balances = balance_result.get("balances", {})
                if isinstance(balances, dict) and "list" in balances:
                    for account in balances["list"][:2]:
                        account_type = account.get("accountType", "Unknown")
                        total_equity = account.get("totalEquity", "0")
                        print(f"   💰 {account_type}: {total_equity} USD")

            else:
                error = balance_result.get("error", "Nieznany błąd")
                print(f"❌ Błąd pobierania salda: {error}")
                return {"status": False, "error": error}
            # Test market data
            print("📊 Test danych rynkowych...")
            try:
                if hasattr(connector, "get_ohlcv"):
                    market_result = connector.get_ohlcv("BTCUSDT", "1h", limit=5)
                    # get_ohlcv returns a list, not a dict with success field
                    if isinstance(market_result, list) and len(market_result) > 0:
                        print("✅ Dane rynkowe pobrane pomyślnie")
                    else:
                        print("❌ Brak danych rynkowych")
                elif hasattr(connector, "get_klines"):
                    market_result = connector.get_klines("BTCUSDT", "1h", limit=5)
                    # get_klines returns a list, not a dict with success field
                    if isinstance(market_result, list) and len(market_result) > 0:
                        print("✅ Dane rynkowe pobrane pomyślnie")
                    else:
                        print("❌ Brak danych rynkowych")
                else:
                    print("⚠️ Brak metody get_ohlcv/get_klines")

            except Exception as e:
                print(f"❌ Błąd testu danych rynkowych: {e}")

            return {
                "status": time_ok and balance_ok and source == "api",
                "details": {
                    "server_time": time_ok,
                    "wallet_balance": balance_ok,
                    "data_source": source,
                    "using_real_data": source == "api",
                },
            }

        except Exception as e:
            error_msg = f"Błąd audytu Bybit connector: {e}"
            print(f"❌ {error_msg}")
            logger.exception("Szczegóły błędu:")
            return {"status": False, "error": error_msg}

    def audit_production_manager(self) -> Dict[str, Any]:
        """Audyt Production Data Manager"""
        print("\n🔍 AUDYT PRODUCTION DATA MANAGER")
        print("=" * 50)

        try:
            from production_data_manager import ProductionDataManager

            manager = ProductionDataManager()

            # Sprawdź konfigurację
            is_production = manager.is_production
            has_credentials = bool(manager.api_key and manager.api_secret)

            print(
                f"{'✅' if is_production else '❌'} Tryb produkcyjny: {is_production}"
            )
            print(
                f"{'✅' if has_credentials else '❌'} Poświadczenia API: {'Tak' if has_credentials else 'Nie'}"
            )

            # Sprawdź status połączenia
            connection_status = manager.connection_status
            bybit_connected = connection_status.get("bybit", {}).get("connected", False)
            print(
                f"{'✅' if bybit_connected else '❌'} Połączenie Bybit: {'Połączony' if bybit_connected else 'Rozłączony'}"
            )

            # Test pobierania danych
            try:
                print("📊 Test pobierania danych rynkowych...")
                market_data = manager.get_market_data("BTCUSDT")
                if market_data:
                    source = market_data.get("source", "unknown")
                    price = market_data.get("price", "N/A")
                    print(f"✅ Dane pobrane (źródło: {source}, cena: {price})")

                    real_data = source not in ["simulation", "demo", "fallback"]
                    return {
                        "status": is_production
                        and has_credentials
                        and bybit_connected
                        and real_data,
                        "details": {
                            "production_mode": is_production,
                            "has_credentials": has_credentials,
                            "connected": bybit_connected,
                            "data_source": source,
                            "using_real_data": real_data,
                        },
                    }
                else:
                    print("❌ Nie udało się pobrać danych rynkowych")
                    return {"status": False, "error": "Brak danych rynkowych"}

            except Exception as e:
                print(f"❌ Błąd pobierania danych: {e}")
                return {"status": False, "error": str(e)}

        except Exception as e:
            error_msg = f"Błąd audytu Production Manager: {e}"
            print(f"❌ {error_msg}")
            return {"status": False, "error": error_msg}

    def audit_configuration_files(self) -> Dict[str, Any]:
        """Audyt plików konfiguracyjnych"""
        print("\n🔍 AUDYT PLIKÓW KONFIGURACYJNYCH")
        print("=" * 50)

        config_files = ["production_api_config.json", "production_config.json", ".env"]

        status = True
        details = {}

        for config_file in config_files:
            file_path = Path(config_file)
            if file_path.exists():
                print(f"✅ {config_file}: Istnieje")

                try:
                    if config_file.endswith(".json"):
                        with open(file_path, "r") as f:
                            config = json.load(f)

                        # Sprawdź konfigurację production/testnet
                        if config_file == "production_api_config.json":
                            # Sprawdź czy ma sekcję production z prawidłowym URL
                            production_section = (
                                config.get("api_configuration", {})
                                .get("bybit", {})
                                .get("production", {})
                            )
                            if (
                                production_section.get("base_url")
                                == "https://api.bybit.com"
                            ):
                                print("   ✅ Zawiera poprawną production URL")
                                details[config_file] = {
                                    "status": True,
                                    "type": "production",
                                }
                            else:
                                print("   ❌ Brak poprawnej production URL")
                                details[config_file] = {
                                    "status": False,
                                    "type": "invalid",
                                }
                                status = False
                        elif config_file == "production_config.json":
                            # Sprawdź czy jest ustawiony na production
                            if (
                                config.get("environment") == "production"
                                and config.get("bybit_production_enabled") is True
                            ):
                                print("   ✅ Konfiguracja production")
                                details[config_file] = {
                                    "status": True,
                                    "type": "production",
                                }
                            else:
                                print("   ❌ Nie jest skonfigurowany na production")
                                details[config_file] = {
                                    "status": False,
                                    "type": "testnet",
                                }
                                status = False
                        else:
                            # Ogólne sprawdzenie dla innych plików JSON
                            config_str = json.dumps(config).lower()
                            if (
                                "api.bybit.com" in config_str
                                and "testnet" not in config_str
                            ):
                                print("   ✅ Zawiera production URLs")
                                details[config_file] = {
                                    "status": True,
                                    "type": "production",
                                }
                            elif "testnet" in config_str:
                                print("   ❌ Zawiera testnet URLs")
                                details[config_file] = {
                                    "status": False,
                                    "type": "testnet",
                                }
                                status = False
                            else:
                                print("   ⚠️ Niejasna konfiguracja")
                                details[config_file] = {
                                    "status": True,
                                    "type": "unclear",
                                }

                    elif config_file == ".env":
                        with open(file_path, "r") as f:
                            env_content = f.read()

                        if "TRADING_MODE=production" in env_content:
                            print("   ✅ TRADING_MODE=production")
                        else:
                            print("   ❌ TRADING_MODE nie jest ustawiony na production")
                            status = False

                        details[config_file] = {"status": True, "content_checked": True}

                except Exception as e:
                    print(f"   ❌ Błąd odczytu: {e}")
                    details[config_file] = {"status": False, "error": str(e)}
                    status = False
            else:
                print(f"❌ {config_file}: Nie istnieje")
                details[config_file] = {"status": False, "error": "Plik nie istnieje"}

        return {"status": status, "details": details}

    def audit_dashboard_files(self) -> Dict[str, Any]:
        """Audyt plików dashboardów"""
        print("\n🔍 AUDYT PLIKÓW DASHBOARDÓW")
        print("=" * 50)

        dashboard_files = [
            "unified_trading_dashboard.py",
            "enhanced_dashboard.py",
            "master_control_dashboard.py",
            "advanced_trading_analytics.py",
        ]

        status = True
        details = {}

        for dashboard in dashboard_files:
            file_path = Path(dashboard)
            if file_path.exists():
                print(f"✅ {dashboard}: Istnieje")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Sprawdź integrację z prawdziwymi danymi
                    has_production_manager = (
                        "ProductionDataManager" in content
                        or "production_data_manager" in content
                        or "get_production_data" in content
                    )
                    has_bybit_connector = (
                        "BybitConnector" in content or "bybit_connector" in content
                    )
                    has_real_data_integration = any(
                        keyword in content.lower()
                        for keyword in [
                            "production_data_manager",
                            "real.*data",
                            "api.*data",
                            "bybit.*api",
                            "get_production_data",
                            "production_manager",
                        ]
                    )

                    print(
                        f"   {'✅' if has_production_manager else '❌'} ProductionDataManager: {'Tak' if has_production_manager else 'Nie'}"
                    )
                    print(
                        f"   {'✅' if has_bybit_connector else '❌'} BybitConnector: {'Tak' if has_bybit_connector else 'Nie'}"
                    )
                    print(
                        f"   {'✅' if has_real_data_integration else '❌'} Integracja prawdziwych danych: {'Tak' if has_real_data_integration else 'Nie'}"
                    )

                    file_status = (
                        has_production_manager
                        or has_bybit_connector
                        or has_real_data_integration
                    )
                    details[dashboard] = {
                        "status": file_status,
                        "has_production_manager": has_production_manager,
                        "has_bybit_connector": has_bybit_connector,
                        "has_real_data_integration": has_real_data_integration,
                    }

                    if not file_status:
                        status = False

                except Exception as e:
                    print(f"   ❌ Błąd analizy: {e}")
                    details[dashboard] = {"status": False, "error": str(e)}
                    status = False
            else:
                print(f"❌ {dashboard}: Nie istnieje")
                details[dashboard] = {"status": False, "error": "Plik nie istnieje"}

        return {"status": status, "details": details}

    def run_complete_audit(self) -> Dict[str, Any]:
        """Uruchom kompletny audyt systemu"""
        print("🚀 KOMPLEKSOWY AUDYT SYSTEMU TRADINGOWEGO")
        print("=" * 60)
        print(f"Czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        audit_results = {}

        # Wykonaj wszystkie audyty
        audit_results["environment"] = self.audit_environment()
        audit_results["bybit_connector"] = self.audit_bybit_connector()
        audit_results["production_manager"] = self.audit_production_manager()
        audit_results["configuration_files"] = self.audit_configuration_files()
        audit_results["dashboard_files"] = self.audit_dashboard_files()

        # Podsumowanie
        print("\n" + "=" * 60)
        print("📋 PODSUMOWANIE AUDYTU")
        print("=" * 60)

        total_tests = len(audit_results)
        passed_tests = sum(
            1 for result in audit_results.values() if result.get("status", False)
        )

        for test_name, result in audit_results.items():
            status_icon = "✅" if result.get("status", False) else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}")

            if not result.get("status", False) and "error" in result:
                print(f"   💥 Błąd: {result['error']}")

        percentage = (passed_tests / total_tests) * 100
        print(
            f"\n📊 WYNIK: {passed_tests}/{total_tests} testów przeszło ({percentage:.1f}%)"
        )

        if percentage >= 80:
            print("🎉 SYSTEM GOTOWY DO UŻYCIA Z PRAWDZIWYMI DANYMI!")
        elif percentage >= 60:
            print("⚠️ SYSTEM CZĘŚCIOWO GOTOWY - WYMAGANE POPRAWKI")
        else:
            print("❌ SYSTEM NIE GOTOWY - WYMAGANE ZNACZNE POPRAWKI")

        # Rekomendacje
        if self.errors:
            print("\n🔧 WYMAGANE NATYCHMIASTOWE POPRAWKI:")
            for error in self.errors:
                print(f"   • {error}")

        if self.warnings:
            print("\n⚠️ OSTRZEŻENIA:")
            for warning in self.warnings:
                print(f"   • {warning}")

        # Zapisz raport
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": audit_results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "percentage": percentage,
                "errors": self.errors,
                "warnings": self.warnings,
            },
        }

        with open("comprehensive_audit_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n📄 Raport zapisany do: comprehensive_audit_report.json")

        return report


# TODO: Integrate with CI/CD pipeline for automated audit and edge-case tests.
# Edge-case tests: simulate missing env vars, config errors, and audit failures.
# All public methods have docstrings and exception handling.

def main():
    """Główna funkcja"""
    auditor = SystemAuditor()
    return auditor.run_complete_audit()


if __name__ == "__main__":
    main()
