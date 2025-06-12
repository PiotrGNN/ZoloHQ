#!/usr/bin/env python3
"""
Optimization patches for unified_trading_dashboard.py
"""


def render_realtime_market_data_fixed():
    """Naprawiona wersja funkcji render_realtime_market_data"""
    return '''
def render_realtime_market_data():
    """Renderuj dane rynkowe w czasie rzeczywistym"""
    # Periodic memory cleanup
    memory_optimizer.periodic_cleanup()
    
    st.header("📊 Dane Rynkowe w Czasie Rzeczywistym")
    dashboard = memory_safe_session_state('unified_dashboard')
    if dashboard is None:
        st.error("Błąd: UnifiedDashboard nie został zainicjalizowany w session_state.")
        return
    
    # Get real market data
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT']
    market_data = []
    data_source = "demo_fallback"
    
    if dashboard and dashboard.production_manager and dashboard.production_mode:
        # Use real production data
        try:
            for symbol in symbols:
                market_result = dashboard.production_manager.get_market_data(symbol)
                if market_result.get("success"):
                    data = market_result.get("result", {})
                    
                    price = float(data.get('lastPrice', 0))
                    change_24h = float(data.get('price24hPcnt', 0)) * 100
                    volume_24h = float(data.get('volume24h', 0))
                    
                    market_data.append({
                        'Symbol': symbol,
                        'Price': f"${price:,.2f}",
                        'Change 24h': f"{change_24h:+.2f}%",
                        'Volume': f"${volume_24h:,.0f}",
                        'Status': '🟢 Live Data'
                    })
                else:
                    # Fallback for failed API call
                    market_data.append({
                        'Symbol': symbol,
                        'Price': "N/A",
                        'Change 24h': "N/A",
                        'Volume': "N/A",
                        'Status': '🔴 No Data'
                    })
            
            # Memory-safe list management
            if len(market_data) > 1000:
                market_data = market_data[-500:]
            
            data_source = "production_api"
            st.info("📡 **Real-time data from Bybit production API**")
            
        except Exception as e:
            st.warning(f"Production data error: {e}")
            # Fall back to demo data
            for symbol in symbols:
                price = np.random.uniform(20000, 70000) if 'BTC' in symbol else np.random.uniform(1000, 4000)
                change = np.random.uniform(-5, 5)
                volume = np.random.uniform(1000000, 50000000)
                
                market_data.append({
                    'Symbol': symbol,
                    'Price': f"${price:,.2f}",
                    'Change 24h': f"{change:+.2f}%",
                    'Volume': f"${volume:,.0f}",
                    'Status': '🟡 Demo Data'
                })
            
            data_source = "demo_fallback"
    else:
        # Demo data fallback
        for symbol in symbols:
            price = np.random.uniform(20000, 70000) if 'BTC' in symbol else np.random.uniform(1000, 4000)
            change = np.random.uniform(-5, 5)
            volume = np.random.uniform(1000000, 50000000)
            
            market_data.append({
                'Symbol': symbol,
                'Price': f"${price:,.2f}",
                'Change 24h': f"{change:+.2f}%",
                'Volume': f"${volume:,.0f}",
                'Status': '🟡 Demo Data'
            })
        
        # Check if Enhanced Dashboard API is available
        try:
            response = requests.get(f"{dashboard.api_base}/health", timeout=3)
            if response.status_code == 200:
                st.info("🔗 **Demo market data** - Enhanced Dashboard API available for real data")
            else:
                st.warning("🟡 **Demo market data** - Backend services unavailable")
        except Exception as e:
            st.warning("🟡 **Demo market data** - Production manager and backend API unavailable")
    
    # Memory-safe list management
    if len(market_data) > 1000:
        market_data = market_data[-500:]
    
    # Status display based on data source
    if data_source == "production_api":
        st.success("📡 **Real market data from Bybit production API**")
    elif data_source == "enhanced_api":
        st.info("🔗 **Market data from Enhanced Dashboard API**")
    else:
        st.warning("🟡 **Using demo market data** - APIs unavailable")
    
    # Tabela danych rynkowych
    df = memory_optimizer.optimize_dataframe(pd.DataFrame(market_data))
    st.dataframe(df, use_container_width=True)
    
    # Real-time price charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Get real BTC historical data
        if dashboard and dashboard.production_manager and dashboard.production_mode:
            try:
                historical_data = dashboard.production_manager.get_historical_data("BTCUSDT", "1h", 24)
                if not historical_data.empty and 'close' in historical_data.columns:
                    times = historical_data.index
                    btc_prices = historical_data['close']
                    chart_title = "Bitcoin (BTC/USDT) - 24h Real Data"
                    data_source_text = "📡 Live"
                else:
                    # Fallback to demo data
                    times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                         end=datetime.now(), freq='1H')
                    btc_prices = 45000 + np.cumsum(np.random.normal(0, 100, len(times)))
                    chart_title = "Bitcoin (BTC/USDT) - 24h Demo Data"
                    data_source_text = "🟡 Demo"
            except Exception as e:
                # Fallback to demo data
                times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                     end=datetime.now(), freq='1H')
                btc_prices = 45000 + np.cumsum(np.random.normal(0, 100, len(times)))
                chart_title = f"Bitcoin (BTC/USDT) - Demo (Error: {str(e)[:30]})"
                data_source_text = "🔴 Error"
        else:
            # Demo data
            times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                 end=datetime.now(), freq='1H')
            btc_prices = 45000 + np.cumsum(np.random.normal(0, 100, len(times)))
            chart_title = "Bitcoin (BTC/USDT) - 24h Demo Data"
            data_source_text = "🟡 Demo"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=btc_prices,
            mode='lines',
            name=f'BTC/USDT {data_source_text}',
            line=dict(color='#f7931a', width=2)
        ))
        
        fig.update_layout(
            title=chart_title,
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Get real ETH historical data
        if dashboard and dashboard.production_manager and dashboard.production_mode:
            try:
                historical_data = dashboard.production_manager.get_historical_data("ETHUSDT", "1h", 24)
                if not historical_data.empty and 'close' in historical_data.columns:
                    times = historical_data.index
                    eth_prices = historical_data['close']
                    chart_title = "Ethereum (ETH/USDT) - 24h Real Data"
                    data_source_text = "📡 Live"
                else:
                    # Fallback to demo data
                    times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                         end=datetime.now(), freq='1H')
                    eth_prices = 3000 + np.cumsum(np.random.normal(0, 50, len(times)))
                    chart_title = "Ethereum (ETH/USDT) - 24h Demo Data"
                    data_source_text = "🟡 Demo"
            except Exception as e:
                # Fallback to demo data
                times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                     end=datetime.now(), freq='1H')
                eth_prices = 3000 + np.cumsum(np.random.normal(0, 50, len(times)))
                chart_title = f"Ethereum (ETH/USDT) - Demo (Error: {str(e)[:30]})"
                data_source_text = "🔴 Error"
        else:
            # Demo data
            times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                 end=datetime.now(), freq='1H')
            eth_prices = 3000 + np.cumsum(np.random.normal(0, 50, len(times)))
            chart_title = "Ethereum (ETH/USDT) - 24h Demo Data"
            data_source_text = "🟡 Demo"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=eth_prices,
            mode='lines',
            name=f'ETH/USDT {data_source_text}',
            line=dict(color='#627eea', width=2)
        ))
        
        fig.update_layout(
            title=chart_title,
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
'''


if __name__ == "__main__":
    print("Patches for unified_trading_dashboard.py optimization")
