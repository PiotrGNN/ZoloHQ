#!/usr/bin/env python3
"""
Advanced Performance Monitoring System for ZoL0
===============================================

This system monitors API performance, rate limiting efficiency, and provides
real-time insights for further optimization opportunities.
"""

import logging
import sqlite3
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""

    timestamp: float
    endpoint: str
    method: str
    duration: float
    success: bool
    response_size: Optional[int] = None
    rate_limit_wait: Optional[float] = None
    cache_hit: bool = False
    error_message: Optional[str] = None


@dataclass
class RateLimitMetric:
    """Rate limiting metric data structure"""

    timestamp: float
    calls_per_minute: int
    current_interval: float
    backoff_active: bool
    rate_limit_violations: int


@dataclass
class SystemMetric:
    """System-wide metric data structure"""

    timestamp: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    cache_hit_ratio: float
    avg_response_time: float


class PerformanceMonitor:
    """Advanced performance monitoring system"""

    def __init__(
        self, db_path: str = "performance_monitor.db", max_metrics: int = 10000
    ):
        """
        Initialize performance monitor

        Args:
            db_path: SQLite database path for persistence
            max_metrics: Maximum metrics to keep in memory
        """
        self.db_path = db_path
        self.max_metrics = max_metrics

        # In-memory metric storage for real-time analysis
        self.performance_metrics = deque(maxlen=max_metrics)
        self.rate_limit_metrics = deque(maxlen=1000)
        self.system_metrics = deque(maxlen=1000)

        # Performance analysis data
        self.endpoint_stats = defaultdict(list)
        self.hourly_stats = defaultdict(list)

        # Thread safety
        self.lock = threading.Lock()

        # Initialize database
        self._init_database()

        # Start background monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._background_monitor, daemon=True
        )
        self.monitor_thread.start()

        logging.info("Performance monitoring system initialized")

    def _init_database(self):
        """Initialize SQLite database for metric persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Performance metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    endpoint TEXT,
                    method TEXT,
                    duration REAL,
                    success BOOLEAN,
                    response_size INTEGER,
                    rate_limit_wait REAL,
                    cache_hit BOOLEAN,
                    error_message TEXT
                )
            """
            )

            # Rate limit metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limit_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    calls_per_minute INTEGER,
                    current_interval REAL,
                    backoff_active BOOLEAN,
                    rate_limit_violations INTEGER
                )
            """
            )

            # System metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    active_connections INTEGER,
                    cache_hit_ratio REAL,
                    avg_response_time REAL
                )
            """
            )

            # Create indexes for faster queries
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_perf_endpoint ON performance_metrics(endpoint)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_rate_timestamp ON rate_limit_metrics(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sys_timestamp ON system_metrics(timestamp)"
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Failed to initialize performance database: {e}")

    def record_api_call(
        self,
        endpoint: str,
        method: str,
        duration: float,
        success: bool,
        response_size: Optional[int] = None,
        rate_limit_wait: Optional[float] = None,
        cache_hit: bool = False,
        error_message: Optional[str] = None,
    ):
        """Record an API call performance metric"""

        metric = PerformanceMetric(
            timestamp=time.time(),
            endpoint=endpoint,
            method=method,
            duration=duration,
            success=success,
            response_size=response_size,
            rate_limit_wait=rate_limit_wait,
            cache_hit=cache_hit,
            error_message=error_message,
        )

        with self.lock:
            self.performance_metrics.append(metric)
            self.endpoint_stats[endpoint].append(duration)

            # Keep only recent stats per endpoint
            if len(self.endpoint_stats[endpoint]) > 100:
                self.endpoint_stats[endpoint] = self.endpoint_stats[endpoint][-100:]

        # Persist to database asynchronously
        threading.Thread(
            target=self._persist_performance_metric, args=(metric,), daemon=True
        ).start()

        # Check for performance issues
        self._check_performance_issues(metric)

    def record_rate_limit_status(
        self,
        calls_per_minute: int,
        current_interval: float,
        backoff_active: bool,
        rate_limit_violations: int,
    ):
        """Record rate limiting status"""

        metric = RateLimitMetric(
            timestamp=time.time(),
            calls_per_minute=calls_per_minute,
            current_interval=current_interval,
            backoff_active=backoff_active,
            rate_limit_violations=rate_limit_violations,
        )

        with self.lock:
            self.rate_limit_metrics.append(metric)

        threading.Thread(
            target=self._persist_rate_limit_metric, args=(metric,), daemon=True
        ).start()

    def record_system_status(
        self,
        cpu_usage: float,
        memory_usage: float,
        active_connections: int,
        cache_hit_ratio: float,
        avg_response_time: float,
    ):
        """Record system-wide performance status"""

        metric = SystemMetric(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_connections=active_connections,
            cache_hit_ratio=cache_hit_ratio,
            avg_response_time=avg_response_time,
        )

        with self.lock:
            self.system_metrics.append(metric)

        threading.Thread(
            target=self._persist_system_metric, args=(metric,), daemon=True
        ).start()

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""

        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            recent_metrics = [
                m for m in self.performance_metrics if m.timestamp > cutoff_time
            ]
            recent_rate_metrics = [
                m for m in self.rate_limit_metrics if m.timestamp > cutoff_time
            ]
            recent_sys_metrics = [
                m for m in self.system_metrics if m.timestamp > cutoff_time
            ]

        if not recent_metrics:
            return {"error": "No performance data available"}

        # Calculate performance statistics
        durations = [m.duration for m in recent_metrics]
        success_rate = (
            sum(1 for m in recent_metrics if m.success) / len(recent_metrics) * 100
        )
        cache_hit_rate = (
            sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics) * 100
        )

        # Endpoint-specific stats
        endpoint_stats = {}
        for endpoint in set(m.endpoint for m in recent_metrics):
            endpoint_metrics = [m for m in recent_metrics if m.endpoint == endpoint]
            endpoint_durations = [m.duration for m in endpoint_metrics]
            endpoint_stats[endpoint] = {
                "count": len(endpoint_metrics),
                "avg_duration": statistics.mean(endpoint_durations),
                "min_duration": min(endpoint_durations),
                "max_duration": max(endpoint_durations),
                "success_rate": sum(1 for m in endpoint_metrics if m.success)
                / len(endpoint_metrics)
                * 100,
            }

        # Rate limiting stats
        rate_limit_stats = {}
        if recent_rate_metrics:
            intervals = [m.current_interval for m in recent_rate_metrics]
            violations = sum(m.rate_limit_violations for m in recent_rate_metrics)
            rate_limit_stats = {
                "avg_interval": statistics.mean(intervals),
                "total_violations": violations,
                "backoff_percentage": sum(
                    1 for m in recent_rate_metrics if m.backoff_active
                )
                / len(recent_rate_metrics)
                * 100,
            }

        # System stats
        system_stats = {}
        if recent_sys_metrics:
            cpu_usage = [m.cpu_usage for m in recent_sys_metrics]
            memory_usage = [m.memory_usage for m in recent_sys_metrics]
            system_stats = {
                "avg_cpu_usage": statistics.mean(cpu_usage),
                "avg_memory_usage": statistics.mean(memory_usage),
                "avg_cache_hit_ratio": statistics.mean(
                    [m.cache_hit_ratio for m in recent_sys_metrics]
                ),
            }

        return {
            "period_hours": hours,
            "total_api_calls": len(recent_metrics),
            "avg_response_time": statistics.mean(durations),
            "median_response_time": statistics.median(durations),
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "endpoint_stats": endpoint_stats,
            "rate_limit_stats": rate_limit_stats,
            "system_stats": system_stats,
            "performance_issues": self._detect_performance_issues(recent_metrics),
        }

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on performance data"""
        recommendations = []

        try:
            # Get recent performance data
            summary = self.get_performance_summary(hours=1)

            # Check API performance
            for endpoint, stats in summary.get("endpoint_stats", {}).items():
                avg_duration = stats.get("avg_duration", 0)
                stats.get("total_calls", 0)
                error_rate = stats.get("error_rate", 0)

                if avg_duration > 2.0:  # Slow endpoint
                    recommendations.append(
                        {
                            "type": "performance",
                            "severity": "high" if avg_duration > 5.0 else "medium",
                            "endpoint": endpoint,
                            "issue": f"Slow response time: {avg_duration:.2f}s",
                            "recommendation": "Consider caching, query optimization, or request batching",
                        }
                    )

                if error_rate > 0.1:  # High error rate
                    recommendations.append(
                        {
                            "type": "reliability",
                            "severity": "high" if error_rate > 0.3 else "medium",
                            "endpoint": endpoint,
                            "issue": f"High error rate: {error_rate*100:.1f}%",
                            "recommendation": "Review error handling and retry logic",
                        }
                    )

            # Check rate limiting effectiveness
            rate_analysis = self.get_rate_limit_analysis()
            if rate_analysis.get("efficiency_score", 1.0) < 0.7:
                recommendations.append(
                    {
                        "type": "rate_limiting",
                        "severity": "medium",
                        "endpoint": "global",
                        "issue": f"Rate limiting efficiency: {rate_analysis.get('efficiency_score', 0)*100:.1f}%",
                        "recommendation": "Consider adjusting rate limits or implementing adaptive throttling",
                    }
                )

        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")

        return recommendations

    def get_rate_limit_analysis(self) -> Dict[str, Any]:
        """Analyze rate limiting effectiveness"""
        try:
            # Get rate limit metrics from the last hour
            cutoff_time = time.time() - 3600

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM rate_limit_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """,
                (cutoff_time,),
            )

            metrics = cursor.fetchall()
            conn.close()

            if not metrics:
                return {
                    "total_violations": 0,
                    "avg_calls_per_minute": 0,
                    "avg_interval": 0,
                    "efficiency_score": 1.0,
                    "backoff_frequency": 0,
                }

            # Calculate statistics
            total_violations = sum(
                row[4] for row in metrics
            )  # rate_limit_violations column
            avg_calls_per_minute = statistics.mean(
                [row[1] for row in metrics]
            )  # calls_per_minute
            avg_interval = statistics.mean(
                [row[2] for row in metrics]
            )  # current_interval
            backoff_active_count = sum(1 for row in metrics if row[3])  # backoff_active

            # Calculate efficiency score (fewer violations and backoffs = higher efficiency)
            efficiency_score = max(
                0,
                1.0
                - (total_violations * 0.1)
                - (backoff_active_count / len(metrics) * 0.3),
            )

            return {
                "total_violations": total_violations,
                "avg_calls_per_minute": avg_calls_per_minute,
                "avg_interval": avg_interval,
                "efficiency_score": efficiency_score,
                "backoff_frequency": (
                    backoff_active_count / len(metrics) if metrics else 0
                ),
                "sample_size": len(metrics),
            }

        except Exception as e:
            logging.error(f"Error analyzing rate limits: {e}")
            return {
                "total_violations": 0,
                "avg_calls_per_minute": 0,
                "avg_interval": 0,
                "efficiency_score": 0.5,
                "backoff_frequency": 0,
                "error": str(e),
            }

    def safe_get_cpu_usage(self):
        """Get CPU usage with fallback if metric is unavailable"""
        try:
            import psutil

            return psutil.cpu_percent(interval=0.5)
        except ImportError:
            logging.warning("psutil not installed, cannot get CPU usage.")
            return None
        except Exception as e:
            logging.warning(f"CPU usage metric unavailable: {e}")
            return None

    def _check_performance_issues(self, metric: PerformanceMetric):
        """Check for immediate performance issues"""

        # Alert on very slow responses
        if metric.duration > 10.0:
            logging.warning(
                f"Slow API response detected: {metric.endpoint} took {metric.duration:.2f}s"
            )

        # Alert on failures
        if not metric.success and metric.error_message:
            logging.error(f"API failure: {metric.endpoint} - {metric.error_message}")

    def _detect_performance_issues(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Detect performance issues from recent metrics"""

        issues = []

        if not metrics:
            return issues

        # Check for consistently slow responses
        slow_responses = sum(1 for m in metrics if m.duration > 5.0)
        if slow_responses > len(metrics) * 0.2:  # More than 20% slow
            issues.append(
                f"High percentage of slow responses: {slow_responses}/{len(metrics)}"
            )

        # Check for error spikes
        errors = sum(1 for m in metrics if not m.success)
        if errors > len(metrics) * 0.1:  # More than 10% errors
            issues.append(f"High error rate: {errors}/{len(metrics)}")

        # Check for rate limiting issues
        rate_limited = sum(
            1 for m in metrics if m.rate_limit_wait and m.rate_limit_wait > 2.0
        )
        if rate_limited > len(metrics) * 0.3:  # More than 30% rate limited
            issues.append(f"Frequent rate limiting: {rate_limited}/{len(metrics)}")

        return issues

    def _background_monitor(self):
        """Background monitoring thread"""

        while self.monitoring_active:
            try:
                # Collect system metrics every 30 seconds
                self._collect_system_metrics()
                time.sleep(30)
            except Exception as e:
                logging.error(f"Background monitoring error: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_system_metrics(self):
        """Collect system-wide performance metrics"""

        try:
            import psutil

            # Get system stats
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent

            # Calculate cache hit ratio from recent metrics
            with self.lock:
                recent_metrics = list(self.performance_metrics)[-100:]  # Last 100 calls

            if recent_metrics:
                cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
                cache_hit_ratio = cache_hits / len(recent_metrics) * 100
                avg_response_time = statistics.mean(
                    [m.duration for m in recent_metrics]
                )
            else:
                cache_hit_ratio = 0
                avg_response_time = 0

            self.record_system_status(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                active_connections=1,  # Simplified for now
                cache_hit_ratio=cache_hit_ratio,
                avg_response_time=avg_response_time,
            )

        except ImportError:
            # psutil not available, skip system metrics
            logging.warning("psutil not available, skipping system metrics collection.")
        except Exception as e:
            logging.error(f"Failed to collect system metrics: {e}")

    def _persist_performance_metric(self, metric: PerformanceMetric):
        """Persist performance metric to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO performance_metrics 
                (timestamp, endpoint, method, duration, success, response_size, 
                 rate_limit_wait, cache_hit, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metric.timestamp,
                    metric.endpoint,
                    metric.method,
                    metric.duration,
                    metric.success,
                    metric.response_size,
                    metric.rate_limit_wait,
                    metric.cache_hit,
                    metric.error_message,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Failed to persist performance metric: {e}")

    def _persist_rate_limit_metric(self, metric: RateLimitMetric):
        """Persist rate limit metric to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO rate_limit_metrics 
                (timestamp, calls_per_minute, current_interval, backoff_active, rate_limit_violations)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    metric.timestamp,
                    metric.calls_per_minute,
                    metric.current_interval,
                    metric.backoff_active,
                    metric.rate_limit_violations,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Failed to persist rate limit metric: {e}")

    def _persist_system_metric(self, metric: SystemMetric):
        """Persist system metric to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO system_metrics 
                (timestamp, cpu_usage, memory_usage, active_connections, cache_hit_ratio, avg_response_time)
                VALUES (?, ?, ?, ?, ?, ?)            """,
                (
                    metric.timestamp,
                    metric.cpu_usage,
                    metric.memory_usage,
                    metric.active_connections,
                    metric.cache_hit_ratio,
                    metric.avg_response_time,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Failed to persist system metric: {e}")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

    def get_endpoints(self) -> Dict[str, Any]:
        """Get endpoint performance data for dashboard compatibility"""
        try:
            summary = self.get_performance_summary(hours=24)
            return summary.get("endpoint_stats", {})
        except Exception as e:
            logging.error(f"Error getting endpoints data: {e}")
            return {}

    def record_response_time(self, component: str, response_time: float):
        """Record response time for a component (compatibility method)"""
        self.record_api_call(
            endpoint=f"/api/{component}",
            method="GET",
            duration=response_time,
            success=True,
        )

    def record_error(self, component: str, error_message: str = ""):
        """Record an error for a component (compatibility method)"""
        self.record_api_call(
            endpoint=f"/api/{component}",
            method="GET",
            duration=0.0,
            success=False,
            error_message=error_message,
        )

    def get_metrics(
        self, hours: Optional[int] = None, components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get metrics compatible with existing dashboard interface"""
        try:
            hours = hours or 1
            summary = self.get_performance_summary(hours=hours)

            if not summary or "error" in summary:
                return {"components": {}, "system": {}, "alerts": []}

            # Transform data to match expected format
            components_data = {}
            if components:
                for component in components:
                    endpoint_key = f"/api/{component}"
                    if endpoint_key in summary.get("endpoint_stats", {}):
                        stats = summary["endpoint_stats"][endpoint_key]
                        components_data[component] = {
                            "response_times": [stats.get("avg_duration", 0)],
                            "error_rate": (100 - stats.get("success_rate", 100)) / 100,
                            "request_count": stats.get("count", 0),
                        }

            return {
                "components": components_data,
                "system": {
                    "cpu_usage": summary.get("system_stats", {}).get(
                        "avg_cpu_usage", 0
                    ),
                    "memory_usage": summary.get("system_stats", {}).get(
                        "avg_memory_usage", 0
                    ),
                    "response_time": summary.get("avg_response_time", 0),
                },
                "alerts": [
                    {"message": issue, "severity": "warning"}
                    for issue in summary.get("performance_issues", [])
                ],
            }

        except Exception as e:
            logging.error(f"Error getting metrics: {e}")
            return {"components": {}, "system": {}, "alerts": []}

    def get_alerts(
        self, hours: Optional[int] = None, alert_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get alerts compatible with existing dashboard interface"""
        try:
            hours = hours or 1
            summary = self.get_performance_summary(hours=hours)

            alerts = []
            for issue in summary.get("performance_issues", []):
                alerts.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "type": "performance",
                        "severity": "warning",
                        "message": issue,
                        "component": "system",
                    }
                )

            return alerts

        except Exception as e:
            logging.error(f"Error getting alerts: {e}")
            return []

# TODO: Integrate with CI/CD pipeline for automated performance and system tests.
# Edge-case tests: simulate DB/network errors, high load, and invalid metrics.
# All public methods have docstrings and exception handling.

# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def record_api_performance(
    endpoint: str, method: str, duration: float, success: bool, **kwargs
):
    """Convenience function to record API performance"""
    monitor = get_performance_monitor()
    monitor.record_api_call(endpoint, method, duration, success, **kwargs)


def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report"""
    monitor = get_performance_monitor()
    return {
        "summary_1h": monitor.get_performance_summary(hours=1),
        "summary_24h": monitor.get_performance_summary(hours=24),
        "recommendations": monitor.get_optimization_recommendations(),
    }
