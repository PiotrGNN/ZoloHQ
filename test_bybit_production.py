import pytest
import os
from bybit_connector import BybitConnector

# Klucze produkcyjne są już ustawione w projekcie, więc nie podajemy ich jawnie.
# Upewniamy się, że testujemy na produkcji (testnet=False)

@pytest.mark.asyncio
async def test_production_wallet_and_ticker():
    if not os.getenv("BYBIT_API_KEY") or not os.getenv("BYBIT_API_SECRET"):
        pytest.skip("BYBIT_API_KEY or BYBIT_API_SECRET not set in environment.")
    connector = BybitConnector(testnet=False)
    print("Test: Pobieranie salda portfela (PRODUKCJA)")
    wallet = await connector.get_wallet_balance()
    print("Wynik wallet:", wallet)

    print("Test: Pobieranie tickera BTCUSDT (PRODUKCJA)")
    ticker = await connector.get_ticker("BTCUSDT")
    print("Wynik ticker:", ticker)
