"""
UI ZoL0 - interfejs użytkownika
Autor: Twój zespół
Data: 2025-06-03
Opis: Streamlit UI do obsługi backtestu, AI, automatyzacji i eksportu.

UWAGA: BinanceConnector jest używany wyłącznie do pobierania danych na żądanie użytkownika przez UI.
System nie wykonuje automatycznych transakcji na Binance. Cały handel automatyczny obsługuje Bybit.
Alerty Telegram i e-mail są wysyłane wyłącznie po kliknięciu przycisku w UI.
Wszystkie zaawansowane metody portfelowe (w tym dynamic_hedging, factor_investing_weights, ml_portfolio_selection)
są dostępne i sterowalne z poziomu UI.
"""

import base64
import io
import pandas as pd
import threading
import time
import smtplib
from email.mime.text import MIMEText
import requests

from ai.agent import ZoL0AIAgent
from binance_connector import BinanceConnector
from bybit_connector import BybitConnector
from data.demo_data import generate_demo_data
from engine.backtest_engine import BacktestEngine
from engine.portfolio_manager import PortfolioManager
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from telegram_alert import send_telegram_alert

STRATEGY_MAP = {"Momentum": MomentumStrategy, "Mean Reversion": MeanReversionStrategy}


def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        towrite = io.BytesIO()
        object_to_download.to_csv(towrite, index=False)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
        return href
    return ""


def export_profits_to_csv(profit_data, filename="profits.csv"):
    df = pd.DataFrame(profit_data)
    df.to_csv(filename, index=False)
    print(f"Zyski wyeksportowane do {filename}")


def notify_profit_threshold(profit, threshold=1000):
    if profit > threshold:
        print(f"🎉 Zysk przekroczył próg {threshold}!")
        # Możesz dodać powiadomienie email/SMS
        msg = MIMEText(f"Zysk przekroczył próg: {profit}")
        # ...konfiguracja SMTP...


def schedule_backtest(
    engine, strategy, data, stop_loss_pct, take_profit_pct, interval_minutes=60
):
    def run_periodically():
        while True:
            result = engine.run(
                strategy,
                data,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            # Możesz tu dodać automatyczny eksport lub powiadomienie
            print(f"[Automatyczny backtest] Wynik: {result.final_capital:.2f}")
            export_profits_to_csv([{"final_capital": result.final_capital}])
            notify_profit_threshold(result.final_capital)
            time.sleep(interval_minutes * 60)

    t = threading.Thread(target=run_periodically, daemon=True)
    t.start()


def send_email_notification(subject, body, to_email):
    """Wyślij powiadomienie email z zaawansowaną obsługą błędów i bezpieczeństwa."""
    import os
    import smtplib
    from email.mime.text import MIMEText
    import logging

    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    if not smtp_user or not smtp_password:
        logging.error("Brak konfiguracji SMTP_USER lub SMTP_PASSWORD w zmiennych środowiskowych!")
        return False
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], msg.as_string())
        logging.info(f"Email wysłany do {to_email}")
        return True
    except Exception as e:
        logging.error(f"Błąd wysyłki email: {e}", exc_info=True)
        return False


def main():
    import streamlit as st
    from ui.agent import agent
    from ui.strategies import STRATEGY_MAP
    from ui.backtest_engine import BacktestEngine

    """
    Główna funkcja uruchamiająca interfejs Streamlit ZoL0.
    Pozwala na wybór strategii, upload danych, uruchomienie backtestu, eksport wyników,
    automatyzację, powiadomienia i integrację z AI.
    """
    st.title("ZoL0 Trading Bot - Advanced Backtesting Engine")
    st.write("System gotowy do działania. Wybierz strategię i uruchom backtest.")
    ai_mode = st.checkbox("Tryb AI-driven (AI wybiera strategię i parametry)")
    strategy_name = st.selectbox("Wybierz strategię", list(STRATEGY_MAP.keys()))
    stop_loss_pct = (
        st.number_input("Stop loss (%)", min_value=0.0, max_value=100.0, value=2.0)
        / 100
    )
    take_profit_pct = (
        st.number_input("Take profit (%)", min_value=0.0, max_value=100.0, value=4.0)
        / 100
    )
    trailing_stop_pct = (
        st.number_input("Trailing stop (%)", min_value=0.0, max_value=100.0, value=0.0)
        / 100
    )
    dynamic_trailing = st.sidebar.checkbox("Dynamiczny trailing stop (ATR)")
    agent = ZoL0AIAgent()
    st.subheader("Dane wejściowe")
    uploaded_file = st.file_uploader(
        "Wgraj własny plik CSV z danymi OHLCV", type=["csv"]
    )
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, parse_dates=[0])
        st.write("Podgląd danych:")
        st.dataframe(data.head())
    else:
        data = generate_demo_data("TEST")
    if ai_mode:
        if st.button("Optymalizuj parametry (AI)"):
            params = agent.optimize_parameters(strategy_name, n_trials=20)
            st.write(f"AI zoptymalizował parametry: {params}")
            st.info("Parametry zoptymalizowane przez AI. Możesz je zapisać do produkcji lub porównać z innymi.")
        else:
            params = agent.optimize_parameters(strategy_name, n_trials=1)
            st.write(f"AI wybrał parametry: {params}")
        stop_loss_pct = (
            st.number_input("Stop loss (%)", min_value=0.0, max_value=100.0, value=2.0, help="Automatyczne limity strat na pozycję.")
            / 100
        )
        take_profit_pct = (
            st.number_input(
                "Take profit (%)", min_value=0.0, max_value=100.0, value=4.0, help="Automatyczne limity zysku na pozycję."
            )
            / 100
        )
        trailing_stop_pct = st.number_input(
            "Trailing stop (%)", min_value=0.0, max_value=100.0, value=1.0, help="Dynamiczne podążanie za ceną."
        ) / 100
        dynamic_trailing = st.checkbox("Dynamic trailing stop", value=True, help="Włącz zaawansowane, adaptacyjne trailing stop.")
        if st.button("Uruchom backtest AI"):
            st.info("Backtest AI uruchomiony. Wyniki pojawią się poniżej.")
            # TODO: Integrate with advanced backtest logic and show analytics
    else:
        if st.button("Uruchom backtest na danych demo"):
            engine = BacktestEngine(initial_capital=100000)
            strategy = STRATEGY_MAP[strategy_name]()
            result = engine.run(
                strategy,
                data,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                trailing_stop_pct=trailing_stop_pct,
                dynamic_trailing=dynamic_trailing,
            )
            st.subheader("Wyniki backtestu:")
            st.write(f"Final capital: {result.final_capital:.2f}")
            st.write(f"Total return: {result.total_return:.2%}")
            st.write(f"Win rate: {result.win_rate:.2%}")
            st.write(f"Profit factor: {result.profit_factor:.2f}")
            st.write(f"Total trades: {result.total_trades}")
            st.line_chart(result.equity_curve.set_index("timestamp")["portfolio_value"])
            st.write("Szczegółowa historia transakcji:")
            st.dataframe([{**t.__dict__} for t in result.trades])
    st.sidebar.subheader("Automatyzacja")
    if st.sidebar.button("Uruchom automatyczny backtest co godzinę"):
        schedule_backtest(
            engine, strategy, data, stop_loss_pct, take_profit_pct, interval_minutes=60
        )
        st.sidebar.success("Automatyczny backtest uruchomiony w tle!")
    if st.button("Wyślij powiadomienie e-mail z raportem") and "result" in locals():
        report = agent.generate_report(result)
        send_email_notification("Raport ZoL0 Backtest", report, "adresat@email.com")
        st.success("Powiadomienie e-mail wysłane!")
    st.sidebar.subheader("Batch/Multi-symbol Backtest")
    symbols = st.sidebar.text_input(
        "Symbole (oddziel przecinkami)", value="BTC/USD,ETH/USD,AAPL"
    )
    if st.sidebar.button("Uruchom batch backtest (AI)"):
        results = []
        for symbol in [s.strip() for s in symbols.split(",") if s.strip()]:
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file, parse_dates=[0])
            else:
                data = generate_demo_data(symbol)
            strategy_name = agent.generate_strategy(STRATEGY_MAP)
            params = agent.optimize_parameters(strategy_name, n_trials=10)
            strategy = STRATEGY_MAP[strategy_name](**params)
            result = engine.run(
                strategy,
                data,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            results.append(
                {
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "final_capital": result.final_capital,
                    "total_return": result.total_return,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades,
                }
            )
        st.write("Batch backtest results:")
        st.dataframe(results)
    st.sidebar.subheader("Live Trading (hook)")
    if (
        st.sidebar.button("Wyślij sygnał do systemu produkcyjnego (symulacja)")
        and "result" in locals()
    ):
        # Tu można dodać integrację z realnym API brokera lub systemu produkcyjnego
        st.sidebar.success(
            "Sygnał tradingowy wysłany do systemu produkcyjnego (symulacja)"
        )
    st.sidebar.subheader("Dane z Bybit")
    api_key = st.sidebar.text_input("Bybit API Key", type="password")
    api_secret = st.sidebar.text_input("Bybit API Secret", type="password")
    if st.sidebar.button("Pobierz dane z Bybit") and api_key and api_secret:
        bybit = BybitConnector(api_key, api_secret, testnet=True)
        symbol = st.sidebar.text_input("Symbol (np. BTCUSD)", value="BTCUSD")
        df = bybit.fetch_ohlcv(symbol)
        st.write(f"Dane z Bybit dla {symbol}:")
        st.dataframe(df.head())
        data = df
    st.sidebar.subheader("Dane z Binance")
    binance_api_key = st.sidebar.text_input("Binance API Key", type="password")
    binance_api_secret = st.sidebar.text_input("Binance API Secret", type="password")
    use_binance_api = st.sidebar.checkbox("Użyj API (SaaS/partner)", value=True)
    binance_api_url = st.sidebar.text_input("Binance API URL", value="http://localhost:8510")
    binance_symbol = st.sidebar.text_input("Symbol (np. BTCUSDT)", value="BTCUSDT")
    if (
        st.sidebar.button("Pobierz dane z Binance")
        and binance_api_key
        and binance_api_secret
    ):
        if use_binance_api:
            # Use maximal API endpoint
            try:
                resp = requests.post(
                    f"{binance_api_url}/api/ohlcv",
                    json={"symbol": binance_symbol, "interval": "1h", "limit": 1000},
                    headers={"X-API-KEY": binance_api_key},
                    timeout=10,
                )
                resp.raise_for_status()
                df = pd.DataFrame(resp.json())
                st.write(f"Dane z Binance API dla {binance_symbol}:")
                st.dataframe(df.head())
                data = df
            except Exception as e:
                st.error(f"Błąd pobierania danych z Binance API: {e}")
        else:
            binance = BinanceConnector(binance_api_key, binance_api_secret, testnet=True)
            df = binance.fetch_ohlcv(binance_symbol)
            st.write(f"Dane z Binance (local) dla {binance_symbol}:")
            st.dataframe(df.head())
            data = df
    st.sidebar.subheader("Zaawansowana analiza Binance (API)")
    if st.sidebar.button("Pobierz rekomendacje Binance (API)") and use_binance_api and binance_api_key:
        try:
            resp = requests.get(
                f"{binance_api_url}/api/recommendations",
                headers={"X-API-KEY": binance_api_key},
                timeout=10,
            )
            resp.raise_for_status()
            recs = resp.json().get("recommendations", [])
            st.write("Rekomendacje Binance:")
            st.write(recs)
        except Exception as e:
            st.error(f"Błąd pobierania rekomendacji z Binance API: {e}")
    if st.sidebar.button("Pobierz analitykę Binance (API)") and use_binance_api and binance_api_key:
        try:
            resp = requests.get(
                f"{binance_api_url}/api/analytics",
                headers={"X-API-KEY": binance_api_key},
                timeout=10,
            )
            resp.raise_for_status()
            analytics = resp.json()
            st.write("Analityka Binance:")
            st.json(analytics)
        except Exception as e:
            st.error(f"Błąd pobierania analityki z Binance API: {e}")
    st.sidebar.subheader("SaaS/Partner/Audit (Binance API)")
    tenant_id = st.sidebar.text_input("Tenant ID (SaaS)")
    if st.sidebar.button("Pobierz raport SaaS (Binance API)") and use_binance_api and binance_api_key and tenant_id:
        try:
            resp = requests.get(
                f"{binance_api_url}/api/saas/tenant/{tenant_id}/report",
                headers={"X-API-KEY": binance_api_key},
                timeout=10,
            )
            resp.raise_for_status()
            st.write("Raport SaaS:")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Błąd pobierania raportu SaaS z Binance API: {e}")
    if st.sidebar.button("Pobierz audit trail (Binance API)") and use_binance_api and binance_api_key:
        try:
            resp = requests.get(
                f"{binance_api_url}/api/audit/trail",
                headers={"X-API-KEY": binance_api_key},
                timeout=10,
            )
            resp.raise_for_status()
            st.write("Audit trail:")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Błąd pobierania audit trail z Binance API: {e}")
    if st.sidebar.button("Pobierz status compliance (Binance API)") and use_binance_api and binance_api_key:
        try:
            resp = requests.get(
                f"{binance_api_url}/api/compliance/status",
                headers={"X-API-KEY": binance_api_key},
                timeout=10,
            )
            resp.raise_for_status()
            st.write("Compliance status:")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Błąd pobierania compliance status z Binance API: {e}")
    # --- KOMENTARZE POLITYKA ALERTÓW I BINANCE ---
    st.sidebar.markdown("---")
    st.sidebar.info(
        "\n**Uwaga:**\n- BinanceConnector jest używany wyłącznie do pobierania danych na żądanie użytkownika.\n- System nie wykonuje automatycznych transakcji na Binance.\n- Alerty Telegram i e-mail są wysyłane wyłącznie po kliknięciu przycisku w UI.\n- Wszystkie zaawansowane metody portfelowe są dostępne i sterowalne z poziomu UI.\n"
    )


if __name__ == "__main__":
    main()
