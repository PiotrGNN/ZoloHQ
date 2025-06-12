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
import smtplib
import threading
import time
from email.mime.text import MIMEText

import pandas as pd
import streamlit as st

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
            time.sleep(interval_minutes * 60)

    t = threading.Thread(target=run_periodically, daemon=True)
    t.start()


def send_email_notification(subject, body, to_email):
    # Konfiguracja SMTP (przykład dla Gmaila)
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "your_email@gmail.com"
    smtp_password = "your_password"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, [to_email], msg.as_string())


def main():
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
        strategy_name = agent.generate_strategy(STRATEGY_MAP)
        st.write(f"AI wybrał strategię: {strategy_name}")
        if st.button("Optymalizuj parametry (AI)"):
            params = agent.optimize_parameters(strategy_name, n_trials=20)
            st.write(f"AI zoptymalizował parametry: {params}")
        else:
            params = agent.optimize_parameters(strategy_name, n_trials=1)
            st.write(f"AI wybrał parametry: {params}")
        stop_loss_pct = (
            st.number_input("Stop loss (%)", min_value=0.0, max_value=100.0, value=2.0)
            / 100
        )
        take_profit_pct = (
            st.number_input(
                "Take profit (%)", min_value=0.0, max_value=100.0, value=4.0
            )
            / 100
        )
        if st.button("Uruchom backtest AI"):
            engine = BacktestEngine(initial_capital=100000)
            strategy = STRATEGY_MAP[strategy_name](**params)
            result = engine.run(
                strategy,
                data,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                trailing_stop_pct=trailing_stop_pct,
                dynamic_trailing=dynamic_trailing,
            )
            st.subheader("Wyniki backtestu (AI):")
            st.write(f"Final capital: {result.final_capital:.2f}")
            st.write(f"Total return: {result.total_return:.2%}")
            st.write(f"Win rate: {result.win_rate:.2%}")
            st.write(f"Profit factor: {result.profit_factor:.2f}")
            st.write(f"Total trades: {result.total_trades}")
            st.line_chart(result.equity_curve.set_index("timestamp")["portfolio_value"])
            st.write("Szczegółowa historia transakcji:")
            st.dataframe([{**t.__dict__} for t in result.trades])
            st.subheader("Raport AI:")
            st.text(agent.generate_report(result))
            st.subheader("Rekomendacja AI:")
            st.success(agent.recommend(result))
            st.markdown(
                download_link(
                    result.equity_curve,
                    "equity_curve.csv",
                    "Pobierz equity curve (CSV)",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                download_link(
                    pd.DataFrame([{**t.__dict__} for t in result.trades]),
                    "trades.csv",
                    "Pobierz transakcje (CSV)",
                ),
                unsafe_allow_html=True,
            )
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
    if (
        st.sidebar.button("Pobierz dane z Binance")
        and binance_api_key
        and binance_api_secret
    ):
        binance = BinanceConnector(binance_api_key, binance_api_secret, testnet=True)
        binance_symbol = st.sidebar.text_input("Symbol (np. BTCUSDT)", value="BTCUSDT")
        df = binance.fetch_ohlcv(binance_symbol)
        st.write(f"Dane z Binance dla {binance_symbol}:")
        st.dataframe(df.head())
        data = df
    st.sidebar.subheader("Walk-Forward Analysis (AI)")
    if st.sidebar.button("Uruchom walk-forward analysis (AI)") and "data" in locals():
        results = agent.walk_forward_analysis(
            strategy_name, data, window_size=500, step_size=100
        )
        st.write("Wyniki walk-forward analysis:")
        st.dataframe(results)
    st.sidebar.subheader("Automatyczny trading na Bybit (symulacja)")
    auto_trade = st.sidebar.checkbox("Włącz automatyczny trading na Bybit")
    trade_qty = st.sidebar.number_input("Wielkość pozycji (szt.)", min_value=1, value=1)
    trade_symbol = st.sidebar.text_input(
        "Symbol do handlu (np. BTCUSD)", value="BTCUSD"
    )
    st.sidebar.subheader("Tryb produkcyjny Bybit")
    production_mode = st.sidebar.checkbox("Przełącz na realny rynek (produkcyjny)")
    if production_mode and api_key and api_secret:
        bybit = BybitConnector(api_key, api_secret, testnet=False)
        st.sidebar.success("Połączono z realnym rynkiem Bybit!")
    if auto_trade and api_key and api_secret and "result" in locals() and result.trades:
        bybit = BybitConnector(api_key, api_secret, testnet=not production_mode)
        last_trade = result.trades[-1]
        if last_trade.side.name == "BUY":
            order_side = "Buy"
        else:
            order_side = "Sell"
        order = bybit.place_order(trade_symbol, order_side, trade_qty)
        st.sidebar.success(
            f"Zlecenie {order_side} na {trade_symbol} wysłane do Bybit! Odpowiedź: {order}"
        )
    st.sidebar.subheader("Zarządzanie portfelem")
    if st.sidebar.button("Pokaż portfel (symulacja)") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        for t in result.trades:
            if t.side.name == "BUY":
                pm.update(t.symbol, "buy", t.quantity, t.entry_price)
                pm.update(t.symbol, "sell", t.quantity, t.exit_price)
            else:
                pm.update(t.symbol, "sell", t.quantity, t.entry_price)
                pm.update(t.symbol, "buy", t.quantity, t.exit_price)
        st.write("Pozycje w portfelu:")
        st.write(pm.get_positions())
        st.write("Historia portfela:")
        st.dataframe(pm.get_history())
    if st.sidebar.button("Zaawansowana analiza portfela") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        for t in result.trades:
            if t.side.name == "BUY":
                pm.update(t.symbol, "buy", t.quantity, t.entry_price)
                pm.update(t.symbol, "sell", t.quantity, t.exit_price)
            else:
                pm.update(t.symbol, "sell", t.quantity, t.entry_price)
                pm.update(t.symbol, "buy", t.quantity, t.exit_price)
        # Przyjmujemy ostatnie ceny z backtestu
        last_prices = {t.symbol: t.exit_price for t in result.trades}
        analytics = pm.get_advanced_analytics(last_prices)
        st.write("Zaawansowane metryki portfela:")
        st.json(analytics)
    st.sidebar.subheader("Dynamiczny position sizing")
    if st.sidebar.button("Oblicz dynamiczny position sizing") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        last_trade = result.trades[-1] if result.trades else None
        if last_trade:
            stop_distance = abs(last_trade.entry_price - last_trade.exit_price)
            portfolio_value = 100000  # lub z portfela
            risk_per_trade = 0.02
            size = pm.dynamic_position_sizing(
                last_trade.symbol, risk_per_trade, stop_distance, portfolio_value
            )
            st.write(f"Dynamiczny position sizing dla {last_trade.symbol}: {size:.2f}")
    st.sidebar.subheader("Risk Parity")
    if st.sidebar.button("Oblicz risk parity") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        # Przyjmujemy returns z equity curve
        returns_df = (
            result.equity_curve.set_index("timestamp")[["portfolio_value"]]
            .pct_change()
            .dropna()
        )
        weights = pm.risk_parity_weights(returns_df)
        st.write("Wagi risk parity:")
        st.json(weights)
    st.sidebar.subheader("Kelly position sizing")
    if st.sidebar.button("Oblicz Kelly position sizing") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        win_rate = result.win_rate
        avg_win = sum(t.pnl for t in result.trades if t.pnl > 0) / max(
            1, len([t for t in result.trades if t.pnl > 0])
        )
        avg_loss = abs(
            sum(t.pnl for t in result.trades if t.pnl < 0)
            / max(1, len([t for t in result.trades if t.pnl < 0]))
        )
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        portfolio_value = 100000
        kelly_size = pm.kelly_position_sizing(win_rate, win_loss_ratio, portfolio_value)
        st.write(f"Kelly position sizing: {kelly_size:.2f}")
    st.sidebar.subheader("Dynamic risk targeting")
    if st.sidebar.button("Oblicz dynamic risk targeting") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        returns = (
            result.equity_curve.set_index("timestamp")[["portfolio_value"]]
            .pct_change()
            .dropna()["portfolio_value"]
        )
        scaling = pm.dynamic_risk_targeting(0.15, returns)
        st.write(f"Skalowanie pozycji (target vol 15%): {scaling:.2f}")
    st.sidebar.subheader("Black-Litterman (symulacja)")
    if st.sidebar.button("Oblicz Black-Litterman weights"):
        import numpy as np

        pm = PortfolioManager(initial_cash=100000)
        pi = np.array([0.05, 0.07])
        tau = 0.05
        P = np.array([[1, -1]])
        Q = np.array([0.01])
        Sigma = np.array([[0.04, 0.006], [0.006, 0.09]])
        mu_bl = pm.black_litterman_weights(pi, tau, P, Q, Sigma)
        st.write("Black-Litterman equilibrium returns:")
        st.write(mu_bl)
    st.sidebar.subheader("Equal Risk Contribution (ERC)")
    if st.sidebar.button("Oblicz ERC weights") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        returns_df = (
            result.equity_curve.set_index("timestamp")[["portfolio_value"]]
            .pct_change()
            .dropna()
        )
        weights = pm.equal_risk_contribution_weights(returns_df)
        st.write("Wagi ERC:")
        st.json(weights)
    st.sidebar.subheader("Value at Risk / Expected Shortfall")
    if st.sidebar.button("Oblicz VaR/ES") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        returns = (
            result.equity_curve.set_index("timestamp")[["portfolio_value"]]
            .pct_change()
            .dropna()["portfolio_value"]
        )
        var = pm.value_at_risk(returns)
        es = pm.expected_shortfall(returns)
        st.write(f"Value at Risk (5%): {var:.4f}")
        st.write(f"Expected Shortfall (5%): {es:.4f}")
    st.sidebar.subheader("Alert Telegram")
    telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password")
    telegram_chat_id = st.sidebar.text_input("Telegram Chat ID")
    if (
        st.sidebar.button("Wyślij alert Telegram")
        and telegram_token
        and telegram_chat_id
        and "result" in locals()
    ):
        msg = agent.generate_report(result)
        resp = send_telegram_alert(telegram_token, telegram_chat_id, msg)
        st.sidebar.success(f"Alert Telegram wysłany! Odpowiedź: {resp}")

    st.sidebar.subheader("Dynamic Hedging (Delta)")
    if (
        st.sidebar.button("Oblicz dynamiczne hedgowanie (delta)")
        and "result" in locals()
    ):
        pm = PortfolioManager(initial_cash=100000)
        spot_prices = pd.Series([t.exit_price for t in result.trades])
        option_deltas = pd.Series([getattr(t, "delta", 0.5) for t in result.trades])
        volatility = spot_prices.pct_change().rolling(window=20).std().fillna(0)
        hedge_positions = pm.dynamic_hedging(spot_prices, option_deltas) * (1 + volatility)
        st.write("Pozycje hedgujące (delta):")
        st.write(hedge_positions.tolist())

    st.sidebar.subheader("Factor Investing Weights")
    if st.sidebar.button("Oblicz wagi factor investing") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        factor_scores = {
            t.symbol: getattr(t, "factor_score", 1.0) * abs(t.pnl)
            for t in result.trades
        }
        weights = pm.factor_investing_weights(factor_scores)
        st.write("Wagi factor investing:")
        st.json(weights)

    st.sidebar.subheader("ML Portfolio Selection")
    if st.sidebar.button("Oblicz ML portfolio selection") and "result" in locals():
        pm = PortfolioManager(initial_cash=100000)
        X = pd.DataFrame({
            "return": [t.pnl for t in result.trades],
            "duration": [t.duration for t in result.trades],
        })
        y = pd.Series([t.pnl for t in result.trades])
        weights = pm.ml_portfolio_selection(X, y)
        st.write("Wagi portfela (ML):")
        st.write(weights)

    # --- KOMENTARZE POLITYKA ALERTÓW I BINANCE ---
    st.sidebar.markdown("---")
    st.sidebar.info(
        "\n**Uwaga:**\n- BinanceConnector jest używany wyłącznie do pobierania danych na żądanie użytkownika.\n- System nie wykonuje automatycznych transakcji na Binance.\n- Alerty Telegram i e-mail są wysyłane wyłącznie po kliknięciu przycisku w UI.\n- Wszystkie zaawansowane metody portfelowe są dostępne i sterowalne z poziomu UI.\n"
    )


if __name__ == "__main__":
    main()
