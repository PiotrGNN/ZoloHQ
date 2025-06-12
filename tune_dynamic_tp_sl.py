import importlib
import importlib.util
import inspect
import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from advanced_alert_management import send_alert
from engine.backtest_engine import BacktestEngine

st.set_page_config(page_title="ZoL0 TP/SL Optymalizacja", page_icon="🔧", layout="wide")

# --- Automatyczna rejestracja strategii (w tym AI/ML) ---
STRATEGY_CLASSES = {}
STRATEGY_PATH = os.path.join(
    os.path.dirname(__file__), "ZoL0-master", "data", "strategies"
)
for fname in os.listdir(STRATEGY_PATH):
    if (
        fname.endswith(".py")
        and not fname.startswith("__")
        and "strategy" in fname.lower()
    ):
        mod_name = fname[:-3]
        try:
            mod = importlib.import_module(f"ZoL0-master.data.strategies.{mod_name}")
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if inspect.isclass(obj) and hasattr(obj, "generate_signals"):
                    STRATEGY_CLASSES[obj.__name__] = obj
        except Exception:
            pass  # Możesz dodać logowanie błędów importu

# --- Integracja AI/ML strategii ---
try:
    from ZoL0_master.data.strategies.AI_strategy_generator import AIStrategyGenerator

    class AI_ML_Strategy:
        def __init__(self, data, target):
            self.generator = AIStrategyGenerator(data, target)
            self.selected_features = self.generator.select_features(
                method="correlation", threshold=0.1
            )
            self.strategy = self.generator.generate_strategy(
                features=self.selected_features
            )

        def generate_signals(self, data):
            """Generate trading signals using an ensemble of ML models."""
            X = data[self.selected_features]
            X_norm = (X - X.mean()) / (X.std() + 1e-9)
            preds = np.vstack([m.predict(X_norm) for m in self.strategy["models"]])
            vote = np.mean(preds, axis=0)
            return np.where(vote > 0, 1, -1)

        def calculate_position_size(self, signal, current_price, portfolio_value):
            risk_pct = 0.01
            notional = portfolio_value * risk_pct
            return max(notional / current_price, 1.0)

    STRATEGY_CLASSES["AI_ML_Strategy"] = AI_ML_Strategy
except Exception:
    pass


# --- Funkcja rankingująca strategie ---
def score_strategy(metrics):
    # Wielokryterialny scoring: możesz rozbudować o kolejne metryki
    score = 0
    if metrics is None:
        return -1e9  # Dyskwalifikuj strategię bez metryk
    score += 2 * (getattr(metrics, "sharpe_ratio", 0) or 0)
    score += 1 * (getattr(metrics, "win_rate", 0) or 0)
    score += 1 * (getattr(metrics, "total_return", 0) or 0)
    score -= 2 * abs(getattr(metrics, "max_drawdown", 0) or 0)
    score -= 1 * (getattr(metrics, "volatility", 0) or 0)
    return score


# Import dynamic switcher (dynamic_strategy_switcher.py)
spec = importlib.util.spec_from_file_location(
    "dynamic_strategy_switcher",
    "ZoL0-master/data/strategies/dynamic_strategy_switcher.py",
)
dynamic_switcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamic_switcher)

status_queue = queue.Queue()


class RealTimeStrategyManager:
    def __init__(self):
        self.switcher = dynamic_switcher.DynamicStrategySwitcher(
            cooldown_period=60, hysteresis_threshold=0.02
        )
        self.log = []
        self.last_market_data = None
        self.last_sentiment = None
        self.last_strategy = None

    def simulate_market(self):
        # Symulacja danych rynkowych i sentymentu (w realu: pobierz z API)
        dates = pd.date_range(start="2025-01-01", periods=50, freq="h")
        prices = np.linspace(100, 110, 50) + np.random.normal(0, 0.5, 50)
        volumes = np.random.randint(1000, 1500, 50)
        market_data = pd.DataFrame({"close": prices, "volume": volumes}, index=dates)
        sentiment = {
            "POSITIVE": np.random.uniform(0.3, 0.7),
            "NEGATIVE": np.random.uniform(0.3, 0.7),
        }
        return market_data, sentiment

    def step(self):
        market_data, sentiment = self.simulate_market()
        strategy = self.switcher.switch_strategy(market_data, sentiment)
        log_entry = f"{time.strftime('%H:%M:%S')} | Strategia: {strategy} | Sentiment: {sentiment}"
        self.log.append(log_entry)
        self.last_market_data = market_data
        self.last_sentiment = sentiment
        self.last_strategy = strategy
        return strategy, log_entry

    def get_log(self, n=20):
        return self.log[-n:]


class RealTimeStrategyAgent:
    def __init__(self):
        self.current_strategy = None
        self.current_params = None
        self.current_metrics = None
        self.running = False
        self.status_log = []
        self.last_update = None
        # Automatyczna lista strategii
        self.available_strategies = list(STRATEGY_CLASSES.keys())
        self.engine = BacktestEngine(initial_capital=100000)

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        self.status_log.append(entry)
        status_queue.put(entry)

    def choose_best_strategy(self, n_trials=50):
        best = None
        best_score = -1e9
        best_name = None
        explanations = []
        for strat_name in self.available_strategies:
            strat_cls = STRATEGY_CLASSES[strat_name]
            # Domyślne parametry (możesz rozbudować o grid search lub AI tuning)
            default_params = {}
            sig = inspect.signature(strat_cls.__init__)
            for k, v in sig.parameters.items():
                if v.default is not inspect.Parameter.empty and k != "self":
                    default_params[k] = v.default
            # Optymalizacja parametrów
            result = self.engine.auto_optimize_strategy(
                strat_name, n_trials=n_trials, dynamic_tp_sl=True
            )
            metrics = result["metrics"]
            score = score_strategy(metrics)
            explanations.append(
                f"{strat_name}: score={score:.2f}, Sharpe={getattr(metrics, 'sharpe_ratio', None)}, win_rate={getattr(metrics, 'win_rate', None)}, drawdown={getattr(metrics, 'max_drawdown', None)}"
            )
            if score > best_score:
                best_score = score
                best = (strat_name, result["best_params"], metrics)
                result["best_params"]
                best_name = strat_name
        self.log("Ranking strategii:\n" + "\n".join(explanations))
        self.log(f"Wybrano: {best_name} (score={best_score:.2f})")
        return best

    def run(self, interval_sec=300):
        self.running = True
        while self.running:
            self.log("Rozpoczynam wybór i optymalizację strategii...")
            strat, params, metrics = self.choose_best_strategy(n_trials=50)
            self.current_strategy = strat
            self.current_params = params
            self.current_metrics = metrics
            self.last_update = datetime.now()
            self.log(f"Wybrano strategię: {strat}, metryki: {metrics}")
            # Zapisz do plików produkcyjnych
            with open("production_dynamic_tp_sl_params.json", "w") as f2:
                json.dump(params, f2, indent=2)
            # Alert o zmianie
            send_alert(
                f"[RT-Agent] Nowa strategia: {strat}, metryki: {metrics}", level="info"
            )
            # Czekaj do kolejnej iteracji lub szybciej jeśli metryki się pogorszą
            for _ in range(int(interval_sec / 5)):
                if not self.running:
                    break
                time.sleep(5)

    def stop(self):
        self.running = False


def run_optimization():
    strategy_name = st.selectbox("Wybierz strategię", list(STRATEGY_CLASSES.keys()))
    n_trials = st.slider("Liczba prób optymalizacji", 10, 200, 100, 10)
    engine = BacktestEngine(initial_capital=100000)
    with st.spinner("Optymalizuję... To może chwilę potrwać..."):
        result = engine.auto_optimize_strategy(
            strategy_name, n_trials=n_trials, dynamic_tp_sl=True
        )
    best_params = result["best_params"]
    metrics = result["metrics"]
    st.success("Optymalizacja zakończona!")
    st.subheader("Najlepsze parametry:")
    st.json(best_params)
    st.subheader("Metryki:")
    st.json(
        {
            "total_return": getattr(metrics, "total_return", None),
            "sharpe_ratio": getattr(metrics, "sharpe_ratio", None),
            "win_rate": getattr(metrics, "win_rate", None),
            "profit_factor": getattr(metrics, "profit_factor", None),
        }
    )
    # Zapisz do plików i wersjonuj
    out = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy_name,
        "best_params": best_params,
        "metrics": {
            "total_return": getattr(metrics, "total_return", None),
            "sharpe_ratio": getattr(metrics, "sharpe_ratio", None),
            "win_rate": getattr(metrics, "win_rate", None),
            "profit_factor": getattr(metrics, "profit_factor", None),
        },
    }
    with open("best_dynamic_tp_sl_params.json", "w") as f:
        json.dump(out, f, indent=2)
    with open("production_dynamic_tp_sl_params.json", "w") as f2:
        json.dump(best_params, f2, indent=2)
    deploy_dir = Path("deploy_history")
    deploy_dir.mkdir(exist_ok=True)
    deploy_file = (
        deploy_dir
        / f"production_dynamic_tp_sl_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    shutil.copy("production_dynamic_tp_sl_params.json", deploy_file)
    with open("deploy_history/deployments.log", "a", encoding="utf-8") as logf:
        logf.write(
            f"{datetime.now().isoformat()} | {strategy_name} | {best_params} | {metrics}\n"
        )
    send_alert(
        f"Wdrożono nowe parametry TP/SL do produkcji: {best_params} | Metryki: {metrics}",
        level="success",
    )
    st.info(f"Deployment versioned as {deploy_file} and logged.")
    st.info(
        "Aby wykonać rollback, skopiuj wybrany plik z deploy_history do production_dynamic_tp_sl_params.json."
    )
    # Walidacja
    min_sharpe = 0.5
    min_win_rate = 0.5
    min_total_return = 0
    if metrics and (
        getattr(metrics, "sharpe_ratio", 0) < min_sharpe
        or getattr(metrics, "win_rate", 0) < min_win_rate
        or getattr(metrics, "total_return", 0) < min_total_return
    ):
        st.error(
            "Nowe parametry NIE spełniają minimalnych wymagań jakości! Wdrożenie zablokowane."
        )
        send_alert(
            f"Blokada wdrożenia: Sharpe={getattr(metrics, 'sharpe_ratio', None)}, win_rate={getattr(metrics, 'win_rate', None)}, total_return={getattr(metrics, 'total_return', None)}",
            level="warning",
        )
        return


def tab_optimization():
    st.header("🔧 Optymalizacja dynamicznego TP/SL (ZoL0)")
    st.write(
        "Uruchom optymalizację najlepszych parametrów TP/SL dla strategii tradingowych. Wyniki zostaną zapisane i wdrożone do produkcji."
    )
    if st.button("Rozpocznij optymalizację!"):
        run_optimization()


def tab_switcher():
    st.header("🔄 Symulacja dynamicznego przełączania strategii")
    st.write(
        "System sam wybierze i zoptymalizuje strategię, a także będzie ją zmieniać w czasie rzeczywistym, jeśli przestanie być skuteczna."
    )
    if "switcher_manager" not in st.session_state:
        st.session_state.switcher_manager = RealTimeStrategyManager()
    if "switcher_log" not in st.session_state:
        st.session_state.switcher_log = []
    if st.button("Wykonaj krok symulacji (real-time)"):
        strategy, log_entry = st.session_state.switcher_manager.step()
        st.session_state.switcher_log.append(log_entry)
        st.success(f"Wybrana strategia: {strategy}")
    st.subheader("Log działań (ostatnie 20):")
    st.text_area(
        "Log", value="\n".join(st.session_state.switcher_log[-20:]), height=300
    )
    st.info(
        "Możesz uruchomić wiele kroków, by zobaczyć jak system adaptuje się do rynku."
    )


def tab_agent():
    st.header("🤖 Real-Time Strategy Agent & Optymalizacja TP/SL")
    if "agent" not in st.session_state:
        st.session_state.agent = RealTimeStrategyAgent()
        st.session_state.agent_thread = None
        st.session_state.agent_running = False
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sterowanie agentem")
        if not st.session_state.agent_running:
            if st.button("Uruchom agenta (auto-strategia)"):
                st.session_state.agent_running = True
                st.session_state.agent_thread = threading.Thread(
                    target=st.session_state.agent.run, args=(120,), daemon=True
                )
                st.session_state.agent_thread.start()
        else:
            if st.button("Zatrzymaj agenta"):
                st.session_state.agent.stop()
                st.session_state.agent_running = False
        st.write(f"Aktualna strategia: {st.session_state.agent.current_strategy}")
        st.write(f"Parametry: {st.session_state.agent.current_params}")
        st.write(f"Metryki: {st.session_state.agent.current_metrics}")
        st.write(f"Ostatnia aktualizacja: {st.session_state.agent.last_update}")
    with col2:
        st.subheader("Log działań agenta (real-time)")
        log_placeholder = st.empty()
        logs = list(st.session_state.agent.status_log)
        # Pobierz nowe logi z kolejki
        while not status_queue.empty():
            logs.append(status_queue.get())
        log_placeholder.text("\n".join(logs[-30:]))
        st.write("Logi odświeżają się automatycznie po akcji lub odświeżeniu strony.")


def tab_logs():
    st.header("📜 Logi systemowe (ostatnie 100 wpisów)")
    logs = []
    try:
        with open("deploy_history/deployments.log", "r", encoding="utf-8") as f:
            logs = f.readlines()
    except Exception:
        st.info("Brak logów wdrożeń.")
    st.text_area("Logi wdrożeń", value="".join(logs[-100:]), height=400)


# --- MAIN DASHBOARD ---
tabs = st.tabs(
    ["Optymalizacja TP/SL", "Symulacja Przełączania", "Agent Real-Time", "Logi"]
)

with tabs[0]:
    tab_optimization()
with tabs[1]:
    tab_switcher()
with tabs[2]:
    tab_agent()
with tabs[3]:
    tab_logs()
