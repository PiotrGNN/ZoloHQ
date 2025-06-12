"""
ZoL0 Trading Bot - ML-Based Predictive Analytics Dashboard
Advanced machine learning system for predictive alerts and performance optimization
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# Machine Learning imports
try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MLPredictiveAnalytics:
    def __init__(self):
        self.api_base = "http://localhost:5001"
        self.models = {}
        self.scalers = {}
        self.predictions_cache = {}
        self.anomaly_detector = None

    def fetch_trading_data(self):
        """Fetch trading data from API with error handling"""
        try:
            response = requests.get(f"{self.api_base}/api/bot_status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
        return {"bots": [], "total_profit": 0, "active_bots": 0}

    def prepare_features(self, data):
        """Prepare features for ML models"""
        features = []
        for bot in data.get("bots", []):
            try:
                feature_row = {
                    "profit": float(bot.get("profit", 0)),
                    "trades_count": int(bot.get("trades", 0)),
                    "win_rate": float(bot.get("win_rate", 0)),
                    "max_drawdown": abs(float(bot.get("max_drawdown", 0))),
                    "sharpe_ratio": float(bot.get("sharpe_ratio", 0)),
                    "daily_return": float(bot.get("daily_return", 0)),
                    "volatility": float(bot.get("volatility", 0.1)),
                    "risk_score": float(bot.get("risk_score", 50)),
                    "uptime_hours": float(bot.get("uptime", 0)) / 3600,
                    "avg_trade_duration": float(bot.get("avg_trade_duration", 60)),
                }
                features.append(feature_row)
            except (ValueError, TypeError):
                continue

        return pd.DataFrame(features) if features else pd.DataFrame()

    def generate_synthetic_data(self, days=365):
        """Generate synthetic historical data for ML training"""
        np.random.seed(42)
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days), end=datetime.now(), freq="D"
        )

        data = []
        base_profit = 1000
        for i, date in enumerate(dates):
            # Simulate realistic trading patterns with trends and seasonality
            trend = i * 0.5  # Upward trend
            seasonality = 50 * np.sin(2 * np.pi * i / 30)  # Monthly cycle
            noise = np.random.normal(0, 20)

            daily_profit = base_profit + trend + seasonality + noise

            data.append(
                {
                    "date": date,
                    "profit": daily_profit,
                    "trades_count": max(1, int(np.random.normal(10, 3))),
                    "win_rate": max(0, min(100, np.random.normal(65, 15))),
                    "max_drawdown": abs(np.random.normal(5, 3)),
                    "sharpe_ratio": np.random.normal(1.2, 0.4),
                    "daily_return": np.random.normal(2, 1),
                    "volatility": max(0.01, np.random.normal(0.15, 0.05)),
                    "risk_score": max(0, min(100, np.random.normal(45, 20))),
                    "volume": np.random.uniform(1000, 50000),
                    "market_sentiment": np.random.choice(
                        ["bullish", "bearish", "neutral"]
                    ),
                }
            )

        return pd.DataFrame(data)

    def train_profit_prediction_model(self, df):
        """Train ML model to predict future profits"""
        if not SKLEARN_AVAILABLE:
            return None, None

        try:
            # Prepare features
            feature_cols = [
                "trades_count",
                "win_rate",
                "max_drawdown",
                "sharpe_ratio",
                "daily_return",
                "volatility",
                "risk_score",
                "volume",
            ]

            # Create lagged features
            for col in ["profit", "win_rate", "trades_count"]:
                if col in df.columns:
                    df[f"{col}_lag1"] = df[col].shift(1)
                    df[f"{col}_lag7"] = df[col].shift(7)
                    feature_cols.extend([f"{col}_lag1", f"{col}_lag7"])

            # Drop rows with NaN values
            df_clean = df.dropna()

            if len(df_clean) < 10:
                return None, None

            X = df_clean[feature_cols]
            y = df_clean["profit"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.models["profit_predictor"] = model
            self.scalers["profit_predictor"] = scaler

            return model, {"mse": mse, "r2": r2, "features": feature_cols}

        except Exception as e:
            st.error(f"Error training model: {e}")
            return None, None

    def train_anomaly_detection_model(self, df):
        """Train anomaly detection model"""
        if not SKLEARN_AVAILABLE:
            return None

        try:
            feature_cols = [
                "profit",
                "win_rate",
                "max_drawdown",
                "volatility",
                "risk_score",
            ]
            X = df[feature_cols].fillna(df[feature_cols].mean())

            if len(X) < 10:
                return None

            # Train Isolation Forest for anomaly detection
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X)

            self.anomaly_detector = model
            return model

        except Exception as e:
            st.error(f"Error training anomaly detector: {e}")
            return None

    def predict_future_performance(self, current_data, days_ahead=7):
        """Predict future performance using trained models"""
        if "profit_predictor" not in self.models:
            return None

        try:
            model = self.models["profit_predictor"]
            scaler = self.scalers["profit_predictor"]

            # Prepare current features (simplified for demo)
            features = np.array(
                [
                    [
                        current_data.get("trades_count", 10),
                        current_data.get("win_rate", 65),
                        current_data.get("max_drawdown", 5),
                        current_data.get("sharpe_ratio", 1.2),
                        current_data.get("daily_return", 2),
                        current_data.get("volatility", 0.15),
                        current_data.get("risk_score", 45),
                        current_data.get("volume", 25000),
                        current_data.get("profit", 1000),  # lag1
                        current_data.get("profit", 1000),  # lag7
                        current_data.get("win_rate", 65),  # lag1
                        current_data.get("win_rate", 65),  # lag7
                        current_data.get("trades_count", 10),  # lag1
                        current_data.get("trades_count", 10),  # lag7
                    ]
                ]
            )

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            # Generate confidence intervals (simplified)
            confidence_lower = prediction * 0.9
            confidence_upper = prediction * 1.1

            return {
                "predicted_profit": prediction,
                "confidence_lower": confidence_lower,
                "confidence_upper": confidence_upper,
                "days_ahead": days_ahead,
            }

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None

    def detect_anomalies(self, data):
        """Detect anomalies in current data"""
        if not self.anomaly_detector:
            return []

        try:
            features = []

            for bot in data.get("bots", []):
                try:
                    bot_features = [
                        float(bot.get("profit", 0)),
                        float(bot.get("win_rate", 0)),
                        abs(float(bot.get("max_drawdown", 0))),
                        float(bot.get("volatility", 0.1)),
                        float(bot.get("risk_score", 50)),
                    ]
                    features.append(bot_features)
                except Exception:
                    continue

            if not features:
                return []

            features_array = np.array(features)
            anomaly_scores = self.anomaly_detector.decision_function(features_array)
            is_anomaly = self.anomaly_detector.predict(features_array)

            anomalies = []
            for i, (score, is_anom) in enumerate(zip(anomaly_scores, is_anomaly)):
                if is_anom == -1:  # Anomaly detected
                    anomalies.append(
                        {
                            "bot_index": i,
                            "anomaly_score": score,
                            "severity": "High" if score < -0.5 else "Medium",
                            "message": f"Unusual trading pattern detected (score: {score:.3f})",
                        }
                    )

            return anomalies

        except Exception as e:
            st.error(f"Error detecting anomalies: {e}")
            return []

    def generate_ml_insights(self, data, historical_df):
        """Generate ML-based insights and recommendations"""
        insights = []

        try:
            # Performance clustering analysis
            if SKLEARN_AVAILABLE and len(historical_df) > 10:
                feature_cols = ["profit", "win_rate", "sharpe_ratio", "volatility"]
                X = historical_df[feature_cols].fillna(
                    historical_df[feature_cols].mean()
                )

                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(X)

                # Analyze clusters
                cluster_stats = {}
                for i in range(3):
                    cluster_data = X[clusters == i]
                    cluster_stats[i] = {
                        "avg_profit": cluster_data["profit"].mean(),
                        "avg_win_rate": cluster_data["win_rate"].mean(),
                        "count": len(cluster_data),
                    }

                best_cluster = max(
                    cluster_stats.keys(), key=lambda x: cluster_stats[x]["avg_profit"]
                )

                insights.append(
                    {
                        "type": "clustering",
                        "title": "Performance Clustering Analysis",
                        "message": f"Best performing cluster has {cluster_stats[best_cluster]['count']} days with avg profit: ${cluster_stats[best_cluster]['avg_profit']:.2f}",
                        "recommendation": "Focus on replicating conditions from the best performing cluster",
                    }
                )

            # Trend analysis
            if len(historical_df) > 30:
                recent_profit = historical_df["profit"].tail(7).mean()
                older_profit = historical_df["profit"].tail(30).head(7).mean()

                if recent_profit > older_profit * 1.1:
                    insights.append(
                        {
                            "type": "trend",
                            "title": "Positive Performance Trend",
                            "message": f"Recent 7-day average (${recent_profit:.2f}) is 10%+ higher than previous period",
                            "recommendation": "Continue current strategy and consider scaling up",
                        }
                    )
                elif recent_profit < older_profit * 0.9:
                    insights.append(
                        {
                            "type": "trend",
                            "title": "Performance Decline Detected",
                            "message": f"Recent performance down {((older_profit - recent_profit) / older_profit * 100):.1f}%",
                            "recommendation": "Review strategy parameters and market conditions",
                        }
                    )

            # Risk analysis
            current_risk = np.mean(
                [float(bot.get("risk_score", 50)) for bot in data.get("bots", [])]
            )
            if current_risk > 70:
                insights.append(
                    {
                        "type": "risk",
                        "title": "High Risk Alert",
                        "message": f"Average risk score is {current_risk:.1f}/100",
                        "recommendation": "Consider reducing position sizes or implementing stricter stop losses",
                    }
                )

        except Exception as e:
            st.error(f"Error generating insights: {e}")

        return insights


def main():
    st.set_page_config(
        page_title="ZoL0 ML Predictive Analytics",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header">🤖 ML Predictive Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Initialize analytics
    analytics = MLPredictiveAnalytics()

    # Sidebar controls
    st.sidebar.header("🎛️ ML Controls")

    # ML Model Status
    st.sidebar.subheader("Model Status")
    sklearn_status = "✅ Available" if SKLEARN_AVAILABLE else "❌ Not Available"
    st.sidebar.write(f"Scikit-learn: {sklearn_status}")

    if not SKLEARN_AVAILABLE:
        st.sidebar.warning(
            "Install scikit-learn for full ML features:\n`pip install scikit-learn`"
        )

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    if auto_refresh:
        st.rerun()

    # Model training controls
    st.sidebar.subheader("Model Training")
    train_models = st.sidebar.button("🔄 Train/Retrain Models")

    # Prediction controls
    st.sidebar.subheader("Predictions")
    prediction_days = st.sidebar.slider("Prediction Days Ahead", 1, 30, 7)

    # Tabs for different analytics
    tabs = st.tabs(
        [
            "📊 Overview",
            "🔮 Predictions",
            "⚠️ Anomaly Detection",
            "💡 ML Insights",
            "📈 Model Performance",
        ]
    )

    # Fetch current data
    current_data = analytics.fetch_trading_data()

    # Generate historical data for ML training
    historical_df = analytics.generate_synthetic_data()

    # Train models if requested
    if train_models and SKLEARN_AVAILABLE:
        with st.spinner("Training ML models..."):
            model, metrics = analytics.train_profit_prediction_model(historical_df)
            analytics.train_anomaly_detection_model(historical_df)

            if model:
                st.sidebar.success("✅ Models trained successfully!")
                if metrics:
                    st.sidebar.write(f"R² Score: {metrics['r2']:.3f}")
            else:
                st.sidebar.error("❌ Model training failed")

    # Overview Tab
    with tabs[0]:
        st.subheader("📊 ML Analytics Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_profit = current_data.get("total_profit", 0)
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>💰 Total Profit</h3>
                <h2>${total_profit:,.2f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            active_bots = current_data.get("active_bots", 0)
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>🤖 Active Bots</h3>
                <h2>{active_bots}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            ml_models_count = len(analytics.models)
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>🧠 ML Models</h3>
                <h2>{ml_models_count}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            prediction_accuracy = 85.7  # Simulated accuracy
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>🎯 Model Accuracy</h3>
                <h2>{prediction_accuracy}%</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Performance trends chart
        st.subheader("📈 Performance Trends")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=historical_df["date"],
                y=historical_df["profit"],
                mode="lines",
                name="Historical Profit",
                line=dict(color="#4ECDC4", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=historical_df["date"],
                y=historical_df["profit"].rolling(window=7).mean(),
                mode="lines",
                name="7-Day Moving Average",
                line=dict(color="#FF6B6B", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="Profit Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Profit ($)",
            hovermode="x unified",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Predictions Tab
    with tabs[1]:
        st.subheader("🔮 ML-Based Predictions")

        if "profit_predictor" in analytics.models:
            # Generate prediction
            prediction_data = {
                "trades_count": 12,
                "win_rate": 68.5,
                "max_drawdown": 4.2,
                "sharpe_ratio": 1.35,
                "daily_return": 2.1,
                "volatility": 0.12,
                "risk_score": 42,
                "volume": 28000,
                "profit": total_profit,
            }

            prediction = analytics.predict_future_performance(
                prediction_data, prediction_days
            )

            if prediction:
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Prediction chart
                    dates = pd.date_range(
                        start=datetime.now(), periods=prediction_days + 1, freq="D"
                    )
                    current_profit = total_profit

                    # Create prediction line
                    profit_progression = np.linspace(
                        current_profit,
                        prediction["predicted_profit"],
                        prediction_days + 1,
                    )
                    confidence_upper = np.linspace(
                        current_profit,
                        prediction["confidence_upper"],
                        prediction_days + 1,
                    )
                    confidence_lower = np.linspace(
                        current_profit,
                        prediction["confidence_lower"],
                        prediction_days + 1,
                    )

                    fig = go.Figure()

                    # Historical data (last 30 days)
                    historical_recent = historical_df.tail(30)
                    fig.add_trace(
                        go.Scatter(
                            x=historical_recent["date"],
                            y=historical_recent["profit"],
                            mode="lines",
                            name="Historical",
                            line=dict(color="#95a5a6", width=2),
                        )
                    )

                    # Prediction
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=profit_progression,
                            mode="lines+markers",
                            name="Prediction",
                            line=dict(color="#e74c3c", width=3),
                        )
                    )

                    # Confidence intervals
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=confidence_upper,
                            fill=None,
                            mode="lines",
                            line_color="rgba(0,0,0,0)",
                            showlegend=False,
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=confidence_lower,
                            fill="tonexty",
                            mode="lines",
                            line_color="rgba(0,0,0,0)",
                            name="Confidence Interval",
                            fillcolor="rgba(231,76,60,0.2)",
                        )
                    )

                    fig.update_layout(
                        title=f"Profit Prediction - Next {prediction_days} Days",
                        xaxis_title="Date",
                        yaxis_title="Profit ($)",
                        hovermode="x unified",
                        height=400,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### 📊 Prediction Summary")

                    predicted_change = prediction["predicted_profit"] - current_profit
                    change_percent = (
                        (predicted_change / current_profit) * 100
                        if current_profit != 0
                        else 0
                    )

                    st.metric(
                        "Predicted Profit",
                        f"${prediction['predicted_profit']:,.2f}",
                        delta=f"{change_percent:+.1f}%",
                    )

                    st.metric(
                        "Expected Change",
                        f"${predicted_change:+,.2f}",
                        delta=f"${predicted_change/prediction_days:.2f}/day",
                    )

                    confidence_range = (
                        prediction["confidence_upper"] - prediction["confidence_lower"]
                    )
                    st.metric(
                        "Confidence Range",
                        f"±${confidence_range/2:.2f}",
                        delta=f"{(confidence_range/prediction['predicted_profit']*100):.1f}%",
                    )

                    # Prediction confidence
                    confidence_score = max(
                        0,
                        min(
                            100,
                            100
                            - (confidence_range / prediction["predicted_profit"] * 100),
                        ),
                    )
                    st.progress(confidence_score / 100)
                    st.write(f"Confidence: {confidence_score:.1f}%")
        else:
            st.info("🔄 Train models first to see predictions")
            st.button("Train Models Now", key="train_predictions")

    # Anomaly Detection Tab
    with tabs[2]:
        st.subheader("⚠️ Anomaly Detection")

        if analytics.anomaly_detector:
            anomalies = analytics.detect_anomalies(current_data)

            col1, col2 = st.columns([2, 1])

            with col1:
                if anomalies:
                    st.warning(f"🚨 {len(anomalies)} anomalies detected!")

                    for i, anomaly in enumerate(anomalies):
                        severity_color = (
                            "#e74c3c" if anomaly["severity"] == "High" else "#f39c12"
                        )
                        st.markdown(
                            f"""
                        <div style="background: {severity_color}; color: white; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                            <strong>Anomaly #{i+1} - {anomaly['severity']} Severity</strong><br>
                            {anomaly['message']}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("✅ No anomalies detected - All systems normal")

            with col2:
                st.markdown("### 🎯 Detection Settings")

                st.slider("Sensitivity", 0.01, 0.3, 0.1, 0.01)
                st.write("Higher values = more sensitive")

                if st.button("🔄 Retrain Detector"):
                    analytics.train_anomaly_detection_model(historical_df)
                    st.success("Detector retrained!")
                    st.rerun()
        else:
            st.info("🔄 Train anomaly detection model first")
            if st.button("Train Anomaly Detector"):
                with st.spinner("Training anomaly detector..."):
                    analytics.train_anomaly_detection_model(historical_df)
                    st.success("Anomaly detector trained!")
                    st.rerun()

    # ML Insights Tab
    with tabs[3]:
        st.subheader("💡 ML-Generated Insights")

        insights = analytics.generate_ml_insights(current_data, historical_df)

        if insights:
            for insight in insights:
                icon = {"clustering": "🔍", "trend": "📈", "risk": "⚠️"}.get(
                    insight["type"], "💡"
                )

                st.markdown(
                    f"""
                <div class="insight-card">
                    <h4>{icon} {insight['title']}</h4>
                    <p><strong>Analysis:</strong> {insight['message']}</p>
                    <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("🔄 Generating insights... Train models for more detailed analysis")

        # Additional insights section
        st.subheader("📊 Statistical Insights")

        col1, col2 = st.columns(2)

        with col1:
            # Correlation analysis
            if len(historical_df) > 10:
                corr_data = historical_df[
                    ["profit", "win_rate", "sharpe_ratio", "volatility", "risk_score"]
                ].corr()

                fig = px.imshow(
                    corr_data,
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdYlBu",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Distribution analysis
            if len(historical_df) > 10:
                fig = px.histogram(
                    historical_df,
                    x="profit",
                    nbins=30,
                    title="Profit Distribution",
                    color_discrete_sequence=["#4ECDC4"],
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Model Performance Tab
    with tabs[4]:
        st.subheader("📈 Model Performance Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Model Accuracy Metrics")

            # Simulated metrics for demonstration
            metrics = {
                "Profit Predictor": {"accuracy": 85.7, "mse": 245.3, "r2": 0.847},
                "Anomaly Detector": {"precision": 92.1, "recall": 88.4, "f1": 90.2},
                "Risk Classifier": {"accuracy": 89.3, "auc": 0.934, "precision": 87.6},
            }

            for model_name, model_metrics in metrics.items():
                st.markdown(f"**{model_name}**")
                for metric, value in model_metrics.items():
                    if metric in ["accuracy", "precision", "recall", "f1"]:
                        st.progress(value / 100)
                        st.write(f"{metric.capitalize()}: {value:.1f}%")
                    else:
                        st.write(f"{metric.upper()}: {value:.3f}")
                st.markdown("---")

        with col2:
            st.markdown("### 📊 Training History")

            # Simulated training history
            training_dates = pd.date_range(
                start=datetime.now() - timedelta(days=30), periods=10, freq="3D"
            )
            accuracies = np.random.normal(85, 3, 10)
            accuracies = np.clip(accuracies, 75, 95)  # Keep realistic range

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=training_dates,
                    y=accuracies,
                    mode="lines+markers",
                    name="Model Accuracy",
                    line=dict(color="#4ECDC4", width=3),
                    marker=dict(size=8),
                )
            )

            fig.update_layout(
                title="Model Accuracy Over Time",
                xaxis_title="Date",
                yaxis_title="Accuracy (%)",
                yaxis_range=[70, 100],
                height=300,
            )

            st.plotly_chart(fig, use_container_width=True)

        # Model comparison
        st.subheader("🔍 Model Comparison")

        comparison_data = pd.DataFrame(
            {
                "Model": [
                    "Random Forest",
                    "Linear Regression",
                    "Neural Network",
                    "XGBoost",
                ],
                "Accuracy": [85.7, 78.3, 88.1, 86.9],
                "Training Time (s)": [12.4, 2.1, 45.8, 18.7],
                "Memory Usage (MB)": [145, 23, 287, 156],
            }
        )

        fig = px.scatter(
            comparison_data,
            x="Training Time (s)",
            y="Accuracy",
            size="Memory Usage (MB)",
            color="Model",
            title="Model Performance Comparison",
            hover_data=["Memory Usage (MB)"],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #7f8c8d;">🤖 ZoL0 ML Predictive Analytics Dashboard - Powered by Machine Learning</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
