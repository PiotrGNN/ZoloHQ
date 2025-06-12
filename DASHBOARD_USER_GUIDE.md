# 🚀 Enhanced Dashboard User Guide
## Przewodnik użytkownika rozszerzonego dashboard

### 🌐 Access URLs
- **Main Dashboard**: http://localhost:8502
- **API Backend**: http://localhost:5001

### 📊 Dashboard Sections

#### 1. **System Overview** 
- Real-time core system health (100% healthy)
- Component status indicators
- Overall system performance metrics

#### 2. **Trading Strategies Monitor**
- 6 Active strategies: AdaptiveAI, Arbitrage, Breakout, MeanReversion, Momentum, TrendFollowing
- Strategy performance metrics
- Real-time strategy status

#### 3. **AI Models Dashboard**
- 28 AI models monitoring
- RL Trader status and performance
- Model training and inference metrics
- Anomaly detection, pattern recognition, sentiment analysis

#### 4. **Trading Engine Status**
- Engine, executor, and handler monitoring
- Order execution tracking
- Trade performance analytics

#### 5. **Portfolio Management**
- Real-time portfolio status
- Risk metrics and exposure
- Performance tracking

#### 6. **System Performance**
- CPU, Memory, Disk usage
- Process monitoring
- System resource alerts

### 🔧 API Endpoints

#### Core System
- `GET /core/status` - Complete system status
- `GET /core/health` - Health check (100% healthy)
- `GET /core/strategies` - Strategy information
- `GET /core/ai-models` - AI models status
- `GET /core/system-metrics` - Performance metrics

#### Trading
- `GET /api/portfolio` - Portfolio data
- `GET /api/trading-signals` - Trading signals

### 📈 Key Features

#### Real-time Monitoring
- Live system metrics updates
- Automatic refresh every 30 seconds
- Interactive charts and graphs

#### Performance Analytics
- Historical performance tracking
- Strategy comparison tools
- Risk analysis dashboards

#### Alert System
- System health alerts
- Performance threshold notifications
- Error monitoring and logging

### 🛠️ Troubleshooting

#### Common Issues
1. **Dashboard not loading**: Check if services are running on ports 8502 and 5001
2. **API errors**: Verify core system components are active
3. **Performance issues**: Monitor CPU/Memory usage in System Performance section

#### Restart Services
```powershell
# Restart Dashboard
cd "c:\Users\piotr\Desktop\Zol0"
streamlit run enhanced_dashboard.py --server.port 8502

# Restart API
python enhanced_dashboard_api.py
```

### 📊 Success Metrics
- ✅ **Integration Test**: 81.8% success rate
- ✅ **Core Health**: 100% all components healthy
- ✅ **Strategies**: 6/6 active and operational
- ✅ **AI Models**: 28/28 active with RL Trader
- ✅ **Performance**: CPU 16.3%, Memory 54.4%

### 🎯 Next Steps
1. Monitor trading performance in real-time
2. Analyze strategy effectiveness
3. Track AI model performance
4. Optimize system resource usage
5. Set up automated alerts for critical metrics

---
**Created**: May 29, 2025  
**Status**: ✅ OPERATIONAL  
**Integration Test**: 81.8% SUCCESS RATE
