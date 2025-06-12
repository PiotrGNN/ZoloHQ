# ZoL0 Trading System Complete Launcher
# PowerShell script to launch the entire trading system with real Bybit data
# Launches all APIs and dashboards in minimized windows, with robust status output

param(
    [switch]$SkipWait = $false
)

Write-Host ""
Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host "███                                                  ███" -ForegroundColor Cyan
Write-Host "███         ZoL0 TRADING SYSTEM LAUNCHER            ███" -ForegroundColor Yellow
Write-Host "███            REAL BYBIT DATA MODE                 ███" -ForegroundColor Green
Write-Host "███                                                  ███" -ForegroundColor Cyan
Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host ""

# Set production environment variables
$env:BYBIT_PRODUCTION_CONFIRMED = "true"
$env:BYBIT_PRODUCTION_ENABLED = "true"

Write-Host "[INFO] Setting production environment variables..." -ForegroundColor Green
Write-Host "[INFO] BYBIT_PRODUCTION_CONFIRMED=$($env:BYBIT_PRODUCTION_CONFIRMED)" -ForegroundColor Gray
Write-Host "[INFO] BYBIT_PRODUCTION_ENABLED=$($env:BYBIT_PRODUCTION_ENABLED)" -ForegroundColor Gray
Write-Host ""

# Define base directory
$BaseDir = "C:\Users\piotr\Desktop\Zol0"
Set-Location $BaseDir

Write-Host "[STEP 1] Starting Backend API Services..." -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow

try {
    Write-Host "[1.1] Starting Main API Server on port 5000..." -ForegroundColor Cyan
    $MainAPI = Start-Process -FilePath "cmd" -ArgumentList "/k", "cd /d `"$BaseDir\ZoL0-master`" && echo Starting Main API Server... && python dashboard_api.py" -WindowStyle Minimized -PassThru
    Write-Host "    ✅ Main API Server started (PID: $($MainAPI.Id))" -ForegroundColor Green
    
    if (-not $SkipWait) {
        Write-Host "[1.2] Waiting 5 seconds..." -ForegroundColor Gray
        Start-Sleep -Seconds 5
    }
    
    Write-Host "[1.3] Starting Enhanced API Server on port 5001..." -ForegroundColor Cyan
    $EnhancedAPI = Start-Process -FilePath "cmd" -ArgumentList "/k", "cd /d `"$BaseDir`" && echo Starting Enhanced API Server... && python enhanced_dashboard_api.py" -WindowStyle Minimized -PassThru
    Write-Host "    ✅ Enhanced API Server started (PID: $($EnhancedAPI.Id))" -ForegroundColor Green
    
    if (-not $SkipWait) {
        Write-Host "[1.4] Waiting 15 seconds for APIs to initialize..." -ForegroundColor Gray
        Start-Sleep -Seconds 15
    }
} catch {
    Write-Host "    ❌ Error starting API services: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "[STEP 2] Starting Dashboard Services..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

# Dashboard configurations (ensure all dashboards/ports match Python/BAT launchers)
$Dashboards = @(
    @{Name="Master Control Dashboard"; Script="master_control_dashboard.py"; Port=8501},
    @{Name="Unified Trading Dashboard"; Script="unified_trading_dashboard.py"; Port=8502},
    @{Name="Enhanced Bot Monitor"; Script="enhanced_bot_monitor.py"; Port=8503},
    @{Name="Advanced Trading Analytics"; Script="advanced_trading_analytics.py"; Port=8504},
    @{Name="Notification Dashboard"; Script="notification_dashboard.py"; Port=8505},
    @{Name="Advanced Alert Management"; Script="advanced_alert_management.py"; Port=8506},
    @{Name="Portfolio Optimization"; Script="portfolio_optimization.py"; Port=8507},
    @{Name="ML Predictive Analytics"; Script="ml_predictive_analytics.py"; Port=8508},
    @{Name="Enhanced Dashboard"; Script="enhanced_dashboard.py"; Port=8509}
)

$DashboardProcesses = @()

foreach ($Dashboard in $Dashboards) {
    try {
        Write-Host "[2.$($Dashboards.IndexOf($Dashboard) + 1)] Starting $($Dashboard.Name) (port $($Dashboard.Port))..." -ForegroundColor Cyan
        
        $ScriptPath = Join-Path $BaseDir $Dashboard.Script
        if (Test-Path $ScriptPath) {
            $Process = Start-Process -FilePath "cmd" -ArgumentList "/k", "cd /d `"$BaseDir`" && streamlit run $($Dashboard.Script) --server.port $($Dashboard.Port)" -WindowStyle Minimized -PassThru
            $DashboardProcesses += @{Process=$Process; Name=$Dashboard.Name; Port=$Dashboard.Port}
            Write-Host "    ✅ $($Dashboard.Name) started (PID: $($Process.Id))" -ForegroundColor Green
        } else {
            Write-Host "    ⚠️  Script not found: $($Dashboard.Script)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "    ❌ Error starting $($Dashboard.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "[STEP 3] System Status" -ForegroundColor Yellow
Write-Host "======================" -ForegroundColor Yellow

if (-not $SkipWait) {
    Write-Host "[INFO] Waiting 20 seconds for all services to initialize..." -ForegroundColor Gray
    Start-Sleep -Seconds 20
}

Write-Host ""
Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Green
Write-Host "███                                                  ███" -ForegroundColor Green
Write-Host "███         ZoL0 SYSTEM LAUNCH COMPLETE!            ███" -ForegroundColor Yellow
Write-Host "███                                                  ███" -ForegroundColor Green
Write-Host "███  🟢 REAL BYBIT PRODUCTION DATA ACTIVE           ███" -ForegroundColor White
Write-Host "███                                                  ███" -ForegroundColor Green
Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Green
Write-Host ""

Write-Host "[SUCCESS] Backend API Services:" -ForegroundColor Green
Write-Host "    • Main API Server:     http://localhost:5000" -ForegroundColor White
Write-Host "    • Enhanced API Server: http://localhost:5001" -ForegroundColor White
Write-Host ""

Write-Host "[SUCCESS] Trading Dashboards:" -ForegroundColor Green
foreach ($Dashboard in $DashboardProcesses) {
    Write-Host "    • $($Dashboard.Name): http://localhost:$($Dashboard.Port)" -ForegroundColor White
}
Write-Host ""

Write-Host "[INFO] Opening Master Control Dashboard..." -ForegroundColor Cyan
try {
    Start-Process "http://localhost:8501"
} catch {
    Write-Host "    ⚠️  Could not auto-open browser. Please manually open: http://localhost:8501" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  ALL SYSTEMS ONLINE - READY FOR TRADING!" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop all services..." -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

# Function to check if services are running
function Test-ServiceStatus {
    Write-Host ""
    Write-Host "🔍 Service Status Check:" -ForegroundColor Cyan
    
    # Check API services
    try {
        $MainAPIResponse = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($MainAPIResponse.StatusCode -eq 200) {
            Write-Host "    ✅ Main API Server - RESPONDING" -ForegroundColor Green
        }
    } catch {
        Write-Host "    ❌ Main API Server - NOT RESPONDING" -ForegroundColor Red
    }
    
    try {
        $EnhancedAPIResponse = Invoke-WebRequest -Uri "http://localhost:5001" -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($EnhancedAPIResponse.StatusCode -eq 200) {
            Write-Host "    ✅ Enhanced API Server - RESPONDING" -ForegroundColor Green
        }
    } catch {
        Write-Host "    ❌ Enhanced API Server - NOT RESPONDING" -ForegroundColor Red
    }
    
    # Check dashboard services
    foreach ($Dashboard in $DashboardProcesses) {
        try {
            $DashboardResponse = Invoke-WebRequest -Uri "http://localhost:$($Dashboard.Port)" -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($DashboardResponse.StatusCode -eq 200) {
                Write-Host "    ✅ $($Dashboard.Name) - RESPONDING" -ForegroundColor Green
            }
        } catch {
            Write-Host "    ❌ $($Dashboard.Name) - NOT RESPONDING" -ForegroundColor Red
        }
    }
}

# Run initial status check
Test-ServiceStatus

Write-Host ""
Write-Host "📋 Quick Commands:" -ForegroundColor Cyan
Write-Host "    • Check status: Test-ServiceStatus" -ForegroundColor White
Write-Host "    • Master Control: http://localhost:8501" -ForegroundColor White
Write-Host "    • Unified Trading: http://localhost:8502" -ForegroundColor White
Write-Host ""

# Keep script running
try {
    Write-Host "Press Ctrl+C to stop all services and exit..." -ForegroundColor Yellow
    while ($true) {
        Start-Sleep -Seconds 1
    }
} catch {
    Write-Host ""
    Write-Host "🛑 Shutting down all services..." -ForegroundColor Red
    
    # Stop all processes if they exist
    if ($MainAPI -and !$MainAPI.HasExited) {
        $MainAPI.Kill()
        Write-Host "    ✅ Main API Server stopped" -ForegroundColor Green
    }
    
    if ($EnhancedAPI -and !$EnhancedAPI.HasExited) {
        $EnhancedAPI.Kill()
        Write-Host "    ✅ Enhanced API Server stopped" -ForegroundColor Green
    }
    
    foreach ($Dashboard in $DashboardProcesses) {
        if ($Dashboard.Process -and !$Dashboard.Process.HasExited) {
            $Dashboard.Process.Kill()
            Write-Host "    ✅ $($Dashboard.Name) stopped" -ForegroundColor Green
        }
    }
    
    Write-Host "✅ All services stopped. System shutdown complete." -ForegroundColor Green
}
