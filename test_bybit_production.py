from data.execution.bybit_connector import BybitConnector

# Klucze produkcyjne są już ustawione w projekcie, więc nie podajemy ich jawnie.
# Upewniamy się, że testujemy na produkcji (use_testnet=False)


def test_production_wallet_and_ticker():
    connector = BybitConnector(use_testnet=False)
    print("Test: Pobieranie salda portfela (PRODUKCJA)")
    wallet = connector.get_wallet_balance()
    print("Wynik wallet:", wallet)

    print("Test: Pobieranie tickera BTCUSDT (PRODUKCJA)")
    ticker = connector.get_ticker("BTCUSDT")
    print("Wynik ticker:", ticker)


if __name__ == "__main__":
    test_production_wallet_and_ticker()
