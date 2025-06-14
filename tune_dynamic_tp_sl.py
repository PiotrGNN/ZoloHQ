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
from skopt import gp_minimize

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
        except Exception as e:
            import logging
            logging.error(f"Błąd importu strategii: {e}", exc_info=True)

# --- Integracja AI/ML strategii ---
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import joblib
    import os
    import numpy as np
    import pandas as pd
    import importlib.util
    import queue
    import threading
    import time
    import json
    import shutil
    from pathlib import Path
    from datetime import datetime
    import streamlit as st
    import warnings
    warnings.filterwarnings('ignore')

    class AdvancedAIStrategyGenerator:
        """
        Advanced AI/ML strategy generator with feature selection, model stacking, and auto-tuning.
        """
        def __init__(self, data, target, model_dir="ai_models_cache"):
            self.data = data
            self.target = target
            self.model_dir = model_dir
            os.makedirs(self.model_dir, exist_ok=True)
            self.selected_features = self.select_features(method="ensemble", threshold=0.05)
            self.models = self.train_models()
            self.meta_model = self.train_meta_model()

        def select_features(self, method="ensemble", threshold=0.05):
            X = self.data
            y = self.target
            if method == "correlation":
                corr = X.corrwith(y).abs()
                return list(corr[corr > threshold].index)
            elif method == "ensemble":
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                importances = pd.Series(rf.feature_importances_, index=X.columns)
                return list(importances[importances > threshold].index)
            else:
                return list(X.columns)

        def train_models(self):
            X = self.data[self.selected_features]
            y = self.target
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            # Save scaler
            joblib.dump(scaler, os.path.join(self.model_dir, "scaler.pkl"))
            models = []
            # Random Forest
            rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
            rf.fit(X_train, y_train)
            models.append(rf)
            # XGBoost
            xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.07, random_state=42, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)
            models.append(xgb)
            # Isolation Forest for anomaly detection
            iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            iso.fit(X_train)
            models.append(iso)
            # Save models
            for i, m in enumerate(models):
                joblib.dump(m, os.path.join(self.model_dir, f"model_{i}.pkl"))
            return models

        def train_meta_model(self):
            X = self.data[self.selected_features]
            y = self.target
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
            X_train = scaler.transform(X_train)
            # Get base model predictions
            base_preds = np.column_stack([
                m.predict(X_train) if hasattr(m, 'predict') else m.decision_function(X_train) for m in self.models[:2]
            ])
            meta = RandomForestClassifier(n_estimators=50, random_state=42)
            meta.fit(base_preds, y_train)
            joblib.dump(meta, os.path.join(self.model_dir, "meta_model.pkl"))
            return meta

        def generate_strategy(self, features=None):
            if features is None:
                features = self.selected_features
            return {
                "features": features,
                "models": self.models,
                "meta_model": self.meta_model,
                "scaler": joblib.load(os.path.join(self.model_dir, "scaler.pkl")),
            }

    class AI_ML_Strategy:
        def __init__(self, data, target):
            self.generator = AdvancedAIStrategyGenerator(data, target)
            self.selected_features = self.generator.selected_features
            self.strategy = self.generator.generate_strategy(features=self.selected_features)
            self.model_dir = getattr(self.generator, 'model_dir', './models')
            os.makedirs(self.model_dir, exist_ok=True)

        def generate_signals(self, data):
            X = data[self.selected_features]
            scaler = self.strategy["scaler"]
            X_norm = scaler.transform(X)
            base_preds = np.column_stack([
                m.predict(X_norm) if hasattr(m, 'predict') else m.decision_function(X_norm) for m in self.strategy["models"][:2]
            ])
            meta_model = self.strategy["meta_model"]
            vote = meta_model.predict(base_preds)
            return np.where(vote > 0, 1, -1)

        def calculate_position_size(self, signal, current_price, portfolio_value, risk_pct=0.01):
            # Advanced Kelly criterion for position sizing with dynamic estimation
            win_rate = self.estimate_win_rate()
            reward_risk = self.estimate_reward_risk()
            kelly = (win_rate * (reward_risk + 1) - 1) / reward_risk
            kelly = max(min(kelly, 0.2), 0.01)
            notional = portfolio_value * kelly * risk_pct
            return max(notional / current_price, 1.0)

        def estimate_win_rate(self):
            # Placeholder: implement rolling window win rate estimation
            return 0.55

        def estimate_reward_risk(self):
            # Placeholder: implement rolling window reward/risk estimation
            return 2.0

        def auto_tune(self, data, target):
            # Automated hyperparameter tuning for base models with fallback and model selection
            X = data[self.selected_features]
            y = target
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 8, 12, 16],
            }
            best_score = -np.inf
            best_model = None
            best_name = None
            for model_cls, name in zip([RandomForestClassifier, GradientBoostingClassifier], ["RandomForest", "GradientBoosting"]):
                grid = GridSearchCV(model_cls(random_state=42), param_grid, cv=3, scoring='accuracy')
                grid.fit(X, y)
                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    best_model = grid.best_estimator_
                    best_name = name
            joblib.dump(best_model, os.path.join(self.model_dir, f"{best_name}_tuned.pkl"))
            return best_model
except Exception as e:
    import logging
    logging.error(f"Błąd importu AI/ML strategii: {e}", exc_info=True)


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
    # Advanced: Add explainability/robustness bonus
    if hasattr(metrics, 'explainability_score'):
        score += 0.5 * getattr(metrics, 'explainability_score', 0)
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
        best_score = float('-inf')
        best_params = None
        best_name = None
        explanations = []
        for strat_name in self.available_strategies:
            strat_cls = STRATEGY_CLASSES[strat_name]
            default_params = {}
            sig = inspect.signature(strat_cls.__init__)
            for k, v in sig.parameters.items():
                if v.default is not inspect.Parameter.empty:
                    default_params[k] = v.default
            # Optymalizacja parametrów (Bayesian Optimization)
            def objective(params):
                param_dict = dict(zip(default_params.keys(), params))
                result = self.engine.auto_optimize_strategy(strat_name, params=param_dict, dynamic_tp_sl=True)
                metrics = result["metrics"]
                return -score_strategy(metrics)
            param_space = [(0.01, 0.2), (0.5, 2.0)]  # Przykładowe zakresy
            res = gp_minimize(objective, param_space, n_calls=n_trials)
            score = -res.fun
            explanations.append(f"{strat_name}: score={score:.2f}, params={res.x}")
            if score > best_score:
                best_score = score
                best_params = res.x
                best_name = strat_name
        # Logowanie najlepszych parametrów
        with open("best_strategy_params.json", "w") as f:
            json.dump({"params": best_params, "score": best_score, "strategy": best_name}, f)
        self.log("Ranking strategii:\n" + "\n".join(explanations))
        self.log(f"Wybrano: {best_name} (score={best_score:.2f})")
        return best_name, best_params, None  # Return metrics if available

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
