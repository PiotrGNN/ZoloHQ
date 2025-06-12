#!/usr/bin/env python3
"""
Production Usage Monitoring for ZoL0
====================================

Comprehensive production monitoring system that tracks real usage patterns,
performance metrics, and provides actionable insights for optimization.
"""

import json
import logging
import sqlite3
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import schedule

# Import our monitoring systems
from advanced_performance_monitor import PerformanceMonitor
from api_cache_system import CachedAPIWrapper, IntelligentCache


@dataclass
class UsagePattern:
    """Usage pattern analysis data"""

    endpoint: str
    hour_of_day: int
    day_of_week: int
    request_count: int
    avg_response_time: float
    peak_usage: bool
    cache_efficiency: float


@dataclass
class ProductionAlert:
    """Production alert data structure"""

    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'performance', 'rate_limit', 'cache', 'system'
    message: str
    endpoint: Optional[str]
    metric_value: float
    threshold: float
    timestamp: float
    resolved: bool = False


class ProductionMonitor:
    """
    Comprehensive production monitoring system
    """

    def __init__(self, db_path: str = "production_monitoring.db"):
        self.db_path = Path(db_path)
        self.performance_monitor = PerformanceMonitor()
        self.cache = IntelligentCache()
        self.cached_api = CachedAPIWrapper(self.cache)

        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.alert_queue = deque(maxlen=1000)

        # Usage pattern tracking
        self.usage_patterns = defaultdict(list)
        self.hourly_usage = defaultdict(int)
        self.daily_usage = defaultdict(int)

        # Performance thresholds
        self.thresholds = {
            "response_time_critical": 10.0,  # seconds
            "response_time_warning": 5.0,  # seconds
            "cache_hit_rate_warning": 0.5,  # 50%
            "rate_limit_violations": 5,  # per hour
            "cpu_usage_warning": 80.0,  # percent
            "memory_usage_warning": 85.0,  # percent
            "error_rate_warning": 0.05,  # 5%
        }

        # Real-time metrics
        self.real_time_metrics = {
            "requests_per_minute": deque(maxlen=60),
            "response_times": deque(maxlen=100),
            "cache_hit_rates": deque(maxlen=60),
            "error_rates": deque(maxlen=60),
            "system_load": deque(maxlen=60),
        }

        self._init_production_db()
        self._setup_scheduled_tasks()

        logging.info("Production monitoring system initialized")

    def _init_production_db(self):
        """Initialize production monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Usage patterns table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    endpoint TEXT,
                    hour_of_day INTEGER,
                    day_of_week INTEGER,
                    request_count INTEGER,
                    avg_response_time REAL,
                    peak_usage BOOLEAN,
                    cache_efficiency REAL
                )
            """
            )

            # Production alerts table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS production_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE,
                    severity TEXT,
                    category TEXT,
                    message TEXT,
                    endpoint TEXT,
                    metric_value REAL,
                    threshold REAL,
                    timestamp REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at REAL
                )
            """
            )

            # Real-time metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS realtime_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    metric_type TEXT,
                    value REAL,
                    metadata TEXT
                )
            """
            )

            # Optimization recommendations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    category TEXT,
                    recommendation TEXT,
                    priority INTEGER,
                    estimated_improvement REAL,
                    implementation_effort TEXT,
                    applied BOOLEAN DEFAULT FALSE
                )
            """
            )

            # Weekly reports table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS weekly_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    report_data TEXT
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_patterns(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_endpoint ON usage_patterns(endpoint)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON production_alerts(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON production_alerts(severity)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_realtime_timestamp ON realtime_metrics(timestamp)"
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Failed to initialize production database: {e}")

    def _setup_scheduled_tasks(self):
        """Setup scheduled monitoring tasks"""
        # Hourly usage pattern analysis
        schedule.every().hour.do(self._analyze_usage_patterns)

        # Daily optimization recommendations
        schedule.every().day.at("02:00").do(self._generate_optimization_recommendations)

        # Weekly performance report
        schedule.every().week.do(self._generate_weekly_report)

        # Real-time monitoring every minute
        schedule.every().minute.do(self._collect_realtime_metrics)

    def start_monitoring(self):
        """Start production monitoring"""
        if self.monitoring_active:
            logging.warning("Production monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()

        logging.info("Production monitoring started")

    def stop_monitoring(self):
        """Stop production monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logging.info("Production monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Run scheduled tasks
                schedule.run_pending()

                # Check for alerts
                self._check_alerts()

                # Update real-time metrics
                self._update_realtime_metrics()

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(30)  # Wait longer on error

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        duration: float,
        success: bool,
        cache_hit: bool = False,
        **kwargs,
    ):
        """Record API request for monitoring"""
        # Record in performance monitor
        self.performance_monitor.record_api_call(
            endpoint, method, duration, success, cache_hit=cache_hit, **kwargs
        )

        # Update usage patterns
        current_time = datetime.now()
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()

        pattern_key = f"{endpoint}:{hour_of_day}:{day_of_week}"
        self.usage_patterns[pattern_key].append(
            {
                "timestamp": time.time(),
                "duration": duration,
                "success": success,
                "cache_hit": cache_hit,
            }
        )

        # Update real-time metrics
        self.real_time_metrics["requests_per_minute"].append(time.time())
        self.real_time_metrics["response_times"].append(duration)
        self.real_time_metrics["cache_hit_rates"].append(1.0 if cache_hit else 0.0)
        self.real_time_metrics["error_rates"].append(0.0 if success else 1.0)

    def _collect_realtime_metrics(self):
        """Collect real-time system metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent

            # Cache metrics
            cache_stats = self.cache.get_stats()

            # Performance metrics
            recent_response_times = list(self.real_time_metrics["response_times"])
            avg_response_time = (
                statistics.mean(recent_response_times) if recent_response_times else 0
            )

            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            metrics = [
                ("cpu_usage", cpu_usage),
                ("memory_usage", memory_usage),
                ("cache_hit_rate", cache_stats["hit_rate"]),
                ("avg_response_time", avg_response_time),
                ("cache_size_mb", cache_stats["current_size_mb"]),
            ]

            for metric_type, value in metrics:
                cursor.execute(
                    """
                    INSERT INTO realtime_metrics (timestamp, metric_type, value, metadata)
                    VALUES (?, ?, ?, ?)
                """,
                    (time.time(), metric_type, value, "{}"),
                )

            conn.commit()
            conn.close()

            # Update real-time collections
            self.real_time_metrics["system_load"].append(cpu_usage)

        except Exception as e:
            logging.error(f"Failed to collect real-time metrics: {e}")

    def _check_alerts(self):
        """Check for alert conditions"""
        try:
            time.time()

            # Check response time alerts
            recent_times = list(self.real_time_metrics["response_times"])
            if recent_times:
                avg_response_time = statistics.mean(recent_times)
                max_response_time = max(recent_times)

                if max_response_time > self.thresholds["response_time_critical"]:
                    self._create_alert(
                        "critical",
                        "performance",
                        f"Critical response time detected: {max_response_time:.2f}s",
                        metric_value=max_response_time,
                        threshold=self.thresholds["response_time_critical"],
                    )
                elif avg_response_time > self.thresholds["response_time_warning"]:
                    self._create_alert(
                        "warning",
                        "performance",
                        f"High average response time: {avg_response_time:.2f}s",
                        metric_value=avg_response_time,
                        threshold=self.thresholds["response_time_warning"],
                    )

            # Check cache efficiency
            cache_stats = self.cache.get_stats()
            if cache_stats["hit_rate"] < self.thresholds["cache_hit_rate_warning"]:
                self._create_alert(
                    "medium",
                    "cache",
                    f"Low cache hit rate: {cache_stats['hit_rate']:.1%}",
                    metric_value=cache_stats["hit_rate"],
                    threshold=self.thresholds["cache_hit_rate_warning"],
                )

            # Check system resources
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent

            if cpu_usage > self.thresholds["cpu_usage_warning"]:
                self._create_alert(
                    "high",
                    "system",
                    f"High CPU usage: {cpu_usage:.1f}%",
                    metric_value=cpu_usage,
                    threshold=self.thresholds["cpu_usage_warning"],
                )

            if memory_usage > self.thresholds["memory_usage_warning"]:
                self._create_alert(
                    "high",
                    "system",
                    f"High memory usage: {memory_usage:.1f}%",
                    metric_value=memory_usage,
                    threshold=self.thresholds["memory_usage_warning"],
                )

        except Exception as e:
            logging.error(f"Alert checking error: {e}")

    def _create_alert(
        self,
        severity: str,
        category: str,
        message: str,
        endpoint: str = None,
        metric_value: float = 0,
        threshold: float = 0,
    ):
        """Create new alert"""
        alert_id = f"{category}_{severity}_{int(time.time())}"

        alert = ProductionAlert(
            alert_id=alert_id,
            severity=severity,
            category=category,
            message=message,
            endpoint=endpoint,
            metric_value=metric_value,
            threshold=threshold,
            timestamp=time.time(),
        )

        self.alert_queue.append(alert)

        # Store in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO production_alerts 
                (alert_id, severity, category, message, endpoint, metric_value, threshold, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.alert_id,
                    alert.severity,
                    alert.category,
                    alert.message,
                    alert.endpoint,
                    alert.metric_value,
                    alert.threshold,
                    alert.timestamp,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Failed to store alert: {e}")

        logging.warning(f"ALERT [{severity.upper()}] {category}: {message}")

    def _analyze_usage_patterns(self):
        """Analyze usage patterns for optimization opportunities"""
        try:
            current_time = time.time()
            hour_ago = current_time - 3600

            # Analyze patterns from the last hour
            patterns_analysis = []

            for pattern_key, requests in self.usage_patterns.items():
                recent_requests = [r for r in requests if r["timestamp"] > hour_ago]

                if not recent_requests:
                    continue

                endpoint, hour_of_day, day_of_week = pattern_key.split(":")

                avg_response_time = statistics.mean(
                    [r["duration"] for r in recent_requests]
                )
                cache_efficiency = statistics.mean(
                    [r["cache_hit"] for r in recent_requests]
                )
                request_count = len(recent_requests)

                # Determine if this is peak usage
                total_requests = len(requests)
                peak_usage = (
                    request_count > total_requests * 0.1
                )  # More than 10% of total requests in 1 hour

                pattern = UsagePattern(
                    endpoint=endpoint,
                    hour_of_day=int(hour_of_day),
                    day_of_week=int(day_of_week),
                    request_count=request_count,
                    avg_response_time=avg_response_time,
                    peak_usage=peak_usage,
                    cache_efficiency=cache_efficiency,
                )

                patterns_analysis.append(pattern)

            # Store patterns
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for pattern in patterns_analysis:
                cursor.execute(
                    """
                    INSERT INTO usage_patterns 
                    (timestamp, endpoint, hour_of_day, day_of_week, request_count, 
                     avg_response_time, peak_usage, cache_efficiency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        current_time,
                        pattern.endpoint,
                        pattern.hour_of_day,
                        pattern.day_of_week,
                        pattern.request_count,
                        pattern.avg_response_time,
                        pattern.peak_usage,
                        pattern.cache_efficiency,
                    ),
                )

            conn.commit()
            conn.close()

            logging.info(f"Analyzed {len(patterns_analysis)} usage patterns")

        except Exception as e:
            logging.error(f"Usage pattern analysis error: {e}")

    def _generate_optimization_recommendations(self):
        """Generate optimization recommendations based on usage data"""
        try:
            recommendations = []

            # Analyze cache efficiency
            cache_stats = self.cache.get_stats()
            if cache_stats["hit_rate"] < 0.7:
                recommendations.append(
                    {
                        "category": "cache",
                        "recommendation": f"Increase cache TTL for frequently accessed endpoints. Current hit rate: {cache_stats['hit_rate']:.1%}",
                        "priority": 2,
                        "estimated_improvement": 0.3,
                        "implementation_effort": "low",
                    }
                )

            # Analyze endpoint performance
            slow_endpoints = self._identify_slow_endpoints()
            for endpoint, avg_time in slow_endpoints:
                recommendations.append(
                    {
                        "category": "performance",
                        "recommendation": f"Optimize {endpoint} - average response time: {avg_time:.2f}s",
                        "priority": 3,
                        "estimated_improvement": 0.5,
                        "implementation_effort": "medium",
                    }
                )

            # Analyze rate limiting efficiency
            rate_limit_recommendations = self._analyze_rate_limiting()
            recommendations.extend(rate_limit_recommendations)

            # Store recommendations
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for rec in recommendations:
                cursor.execute(
                    """
                    INSERT INTO optimization_recommendations 
                    (timestamp, category, recommendation, priority, estimated_improvement, implementation_effort)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        time.time(),
                        rec["category"],
                        rec["recommendation"],
                        rec["priority"],
                        rec["estimated_improvement"],
                        rec["implementation_effort"],
                    ),
                )

            conn.commit()
            conn.close()

            logging.info(
                f"Generated {len(recommendations)} optimization recommendations"
            )

        except Exception as e:
            logging.error(f"Recommendation generation error: {e}")

    def _identify_slow_endpoints(self) -> List[Tuple[str, float]]:
        """Identify endpoints with poor performance"""
        try:
            # Get recent performance data
            perf_data = self.performance_monitor.get_performance_summary(hours=24)

            slow_endpoints = []
            for endpoint, stats in perf_data.get("endpoints", {}).items():
                avg_time = stats.get("avg_duration", 0)
                if avg_time > 3.0:  # Endpoints taking more than 3 seconds
                    slow_endpoints.append((endpoint, avg_time))

            return sorted(slow_endpoints, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logging.error(f"Slow endpoint identification error: {e}")
            return []

    def _analyze_rate_limiting(self) -> List[Dict]:
        """Analyze rate limiting efficiency"""
        recommendations = []

        try:
            # Get rate limiting data
            rate_limit_data = self.performance_monitor.get_rate_limit_analysis()

            if rate_limit_data.get("violations", 0) > 10:
                recommendations.append(
                    {
                        "category": "rate_limit",
                        "recommendation": f"Consider increasing rate limits. Current violations: {rate_limit_data['violations']}",
                        "priority": 2,
                        "estimated_improvement": 0.2,
                        "implementation_effort": "low",
                    }
                )

            if rate_limit_data.get("avg_wait_time", 0) > 1.0:
                recommendations.append(
                    {
                        "category": "rate_limit",
                        "recommendation": f"Rate limiting causing delays. Average wait: {rate_limit_data['avg_wait_time']:.2f}s",
                        "priority": 3,
                        "estimated_improvement": 0.4,
                        "implementation_effort": "medium",
                    }
                )

        except Exception as e:
            logging.error(f"Rate limiting analysis error: {e}")

        return recommendations

    def get_production_dashboard_data(self) -> Dict:
        """Get data for production monitoring dashboard"""
        try:
            # Real-time metrics
            recent_response_times = list(self.real_time_metrics["response_times"])
            recent_cache_hits = list(self.real_time_metrics["cache_hit_rates"])
            recent_errors = list(self.real_time_metrics["error_rates"])

            # System metrics
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
            }

            # Cache metrics
            cache_stats = self.cache.get_stats()

            # Recent alerts
            recent_alerts = list(self.alert_queue)[-10:]  # Last 10 alerts

            # Performance summary
            perf_summary = self.performance_monitor.get_performance_summary(hours=1)

            return {
                "realtime_metrics": {
                    "avg_response_time": (
                        statistics.mean(recent_response_times)
                        if recent_response_times
                        else 0
                    ),
                    "cache_hit_rate": (
                        statistics.mean(recent_cache_hits) if recent_cache_hits else 0
                    ),
                    "error_rate": (
                        statistics.mean(recent_errors) if recent_errors else 0
                    ),
                    "requests_per_minute": len(
                        [
                            t
                            for t in self.real_time_metrics["requests_per_minute"]
                            if time.time() - t < 60
                        ]
                    ),
                },
                "system_metrics": system_metrics,
                "cache_metrics": cache_stats,
                "recent_alerts": [asdict(alert) for alert in recent_alerts],
                "performance_summary": perf_summary,
                "monitoring_status": {
                    "active": self.monitoring_active,
                    "uptime": time.time()
                    - (recent_alerts[0].timestamp if recent_alerts else time.time()),
                },
            }

        except Exception as e:
            logging.error(f"Dashboard data generation error: {e}")
            return {}

    def get_optimization_report(self) -> Dict:
        """Get comprehensive optimization report"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get recent recommendations
            cursor.execute(
                """
                SELECT * FROM optimization_recommendations 
                WHERE timestamp > ? AND applied = FALSE
                ORDER BY priority DESC, timestamp DESC
            """,
                (time.time() - 86400,),
            )  # Last 24 hours

            recommendations = cursor.fetchall()

            # Get usage patterns
            cursor.execute(
                """
                SELECT endpoint, AVG(avg_response_time) as avg_time, 
                       AVG(cache_efficiency) as cache_eff, SUM(request_count) as total_requests
                FROM usage_patterns 
                WHERE timestamp > ?
                GROUP BY endpoint
                ORDER BY total_requests DESC
            """,
                (time.time() - 86400,),
            )

            usage_patterns = cursor.fetchall()

            conn.close()

            return {
                "recommendations": recommendations,
                "usage_patterns": usage_patterns,
                "cache_stats": self.cache.get_stats(),
                "performance_summary": self.performance_monitor.get_performance_summary(
                    hours=24
                ),
                "generated_at": time.time(),
            }

        except Exception as e:
            logging.error(f"Optimization report generation error: {e}")
            return {}

    def _generate_weekly_report(self):
        """Generate weekly performance report"""
        try:
            logging.info("Generating weekly performance report...")

            # Get weekly data
            weekly_data = self._get_weekly_performance_data()

            # Generate report
            report = {
                "period": "weekly",
                "generated_at": time.time(),
                "summary": weekly_data,
                "recommendations": self._get_weekly_recommendations(weekly_data),
            }

            # Store report
            self._store_weekly_report(report)

            logging.info("Weekly performance report generated successfully")

        except Exception as e:
            logging.error(f"Weekly report generation error: {e}")

    def _get_weekly_performance_data(self) -> Dict:
        """Get performance data for the last week"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            week_ago = time.time() - (7 * 24 * 3600)

            # Get usage patterns
            cursor.execute(
                """
                SELECT endpoint, AVG(avg_response_time) as avg_time,
                       AVG(cache_efficiency) as cache_eff,
                       SUM(request_count) as total_requests
                FROM usage_patterns 
                WHERE timestamp > ?
                GROUP BY endpoint
            """,
                (week_ago,),
            )

            usage_data = cursor.fetchall()

            # Get alert summary
            cursor.execute(
                """
                SELECT severity, COUNT(*) as count
                FROM production_alerts 
                WHERE timestamp > ?
                GROUP BY severity
            """,
                (week_ago,),
            )

            alert_data = cursor.fetchall()

            conn.close()

            return {
                "usage_patterns": usage_data,
                "alert_summary": alert_data,
                "period_start": week_ago,
                "period_end": time.time(),
            }

        except Exception as e:
            logging.error(f"Weekly data retrieval error: {e}")
            return {}

    def _get_weekly_recommendations(self, weekly_data: Dict) -> List[str]:
        """Get recommendations based on weekly data"""
        recommendations = []

        try:
            usage_patterns = weekly_data.get("usage_patterns", [])
            alert_summary = weekly_data.get("alert_summary", [])

            # Analyze usage patterns
            if usage_patterns:
                slow_endpoints = [
                    p for p in usage_patterns if p[1] > 3.0
                ]  # avg_time > 3s
                if slow_endpoints:
                    recommendations.append(
                        "Optimize slow endpoints for better performance"
                    )

                low_cache_endpoints = [
                    p for p in usage_patterns if p[2] < 0.5
                ]  # cache_eff < 50%
                if low_cache_endpoints:
                    recommendations.append(
                        "Improve cache efficiency for low-performing endpoints"
                    )

            # Analyze alerts
            if alert_summary:
                critical_alerts = [a for a in alert_summary if a[0] == "critical"]
                if critical_alerts:
                    recommendations.append("Address critical performance issues")

            return recommendations

        except Exception as e:
            logging.error(f"Weekly recommendations error: {e}")
            return ["Review system performance manually"]

    def _store_weekly_report(self, report: Dict):
        """Store weekly report in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO weekly_reports (timestamp, report_data)
                VALUES (?, ?)
            """,
                (time.time(), json.dumps(report)),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Weekly report storage error: {e}")

    def _update_realtime_metrics(self):
        """Update real-time metrics for monitoring dashboard"""
        try:
            time.time()

            # Update system metrics
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent

            # Update real-time metric collections
            self.real_time_metrics["cpu_usage"].append(cpu_usage)
            self.real_time_metrics["memory_usage"].append(memory_usage)
            self.real_time_metrics["system_load"].append(cpu_usage)

            # Get cache metrics
            cache_stats = self.cache.get_stats()
            self.real_time_metrics["cache_hit_rate"].append(
                cache_stats.get("hit_rate", 0)
            )

            # Get recent performance data
            perf_summary = self.performance_monitor.get_performance_summary(hours=1)
            if perf_summary and "avg_response_time" in perf_summary:
                self.real_time_metrics["response_times"].append(
                    perf_summary["avg_response_time"]
                )

            # Update active connections count
            self.real_time_metrics["active_connections"].append(1)  # Simplified

            logging.debug(
                f"Real-time metrics updated: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%"
            )

        except Exception as e:
            logging.error(f"Failed to update real-time metrics: {e}")


# Global production monitor instance
_production_monitor = None


def get_production_monitor() -> ProductionMonitor:
    """Get global production monitor instance"""
    global _production_monitor
    if _production_monitor is None:
        _production_monitor = ProductionMonitor()
    return _production_monitor


if __name__ == "__main__":
    # Test production monitoring
    logging.basicConfig(level=logging.INFO)

    print("Testing Production Monitoring System...")

    monitor = ProductionMonitor()
    monitor.start_monitoring()

    # Simulate API usage
    import random

    endpoints = [
        "/api/trading/statistics",
        "/api/trading/positions",
        "/api/market/tickers",
    ]

    print("Simulating production API usage...")
    for _i in range(50):
        endpoint = random.choice(endpoints)
        duration = random.uniform(0.1, 3.0)
        success = random.random() > 0.05  # 95% success rate
        cache_hit = random.random() > 0.3  # 70% cache hit rate

        monitor.record_api_request(endpoint, "GET", duration, success, cache_hit)
        time.sleep(0.1)

    # Wait for monitoring
    time.sleep(5)

    # Get dashboard data
    dashboard_data = monitor.get_production_dashboard_data()
    print("\nProduction Dashboard Data:")
    print(json.dumps(dashboard_data, indent=2, default=str))

    # Get optimization report
    optimization_report = monitor.get_optimization_report()
    print("\nOptimization Report:")
    print(json.dumps(optimization_report, indent=2, default=str))

    monitor.stop_monitoring()
    print("\nProduction monitoring test completed!")
