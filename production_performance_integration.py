#!/usr/bin/env python3
"""
Production Performance Monitoring Integration for ZoL0
=====================================================

Integrates all performance monitoring components into the existing ZoL0 system
for comprehensive production monitoring and optimization.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Import all monitoring components
from advanced_performance_monitor import PerformanceMonitor
from advanced_rate_limit_optimizer import get_adaptive_rate_limiter
from api_cache_system import CachedAPIWrapper, get_cache_instance
from production_usage_monitor import get_production_monitor


class ProductionPerformanceSystem:
    """
    Comprehensive production performance monitoring and optimization system
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Initialize all monitoring components
        self.performance_monitor = PerformanceMonitor()
        self.cache = get_cache_instance(
            max_size_mb=self.config.get("cache_size_mb", 100)
        )
        self.cached_api = CachedAPIWrapper(self.cache)
        self.production_monitor = get_production_monitor()
        self.rate_limiter = get_adaptive_rate_limiter()

        # System state
        self.system_active = False
        self.monitoring_threads = []

        # Performance tracking
        self.performance_baseline = {}
        self.optimization_history = []

        logging.info("Production performance system initialized")

    def start_production_monitoring(self):
        """Start all production monitoring components"""
        if self.system_active:
            logging.warning("Production monitoring already active")
            return

        try:
            # Start individual components
            self.production_monitor.start_monitoring()
            self.rate_limiter.start_adaptive_optimization()

            # Set up integrations
            self._setup_monitoring_integrations()

            self.system_active = True

            logging.info("🚀 Production performance monitoring system started")

            # Log startup summary
            self._log_startup_summary()

        except Exception as e:
            logging.error(f"Failed to start production monitoring: {e}")
            self.stop_production_monitoring()

    def stop_production_monitoring(self):
        """Stop all production monitoring components"""
        try:
            # Stop components
            if hasattr(self.production_monitor, "stop_monitoring"):
                self.production_monitor.stop_monitoring()

            if hasattr(self.rate_limiter, "stop_adaptive_optimization"):
                self.rate_limiter.stop_adaptive_optimization()

            # Stop monitoring threads
            for thread in self.monitoring_threads:
                if thread.is_alive():
                    thread.join(timeout=5)

            self.system_active = False

            logging.info("Production performance monitoring system stopped")

        except Exception as e:
            logging.error(f"Error stopping production monitoring: {e}")

    def _setup_monitoring_integrations(self):
        """Set up integrations between monitoring components"""

        # Link cached API wrapper to performance monitor
        self.cached_api.performance_monitor = self.performance_monitor

        # Start cache analytics recording
        cache_analytics_thread = threading.Thread(
            target=self._cache_analytics_loop, daemon=True
        )
        cache_analytics_thread.start()
        self.monitoring_threads.append(cache_analytics_thread)

        # Start performance correlation analysis
        correlation_thread = threading.Thread(
            target=self._performance_correlation_loop, daemon=True
        )
        correlation_thread.start()
        self.monitoring_threads.append(correlation_thread)

    def _cache_analytics_loop(self):
        """Background cache analytics recording"""
        while self.system_active:
            try:
                self.cache.record_analytics()
                time.sleep(300)  # Record every 5 minutes
            except Exception as e:
                logging.error(f"Cache analytics error: {e}")
                time.sleep(60)

    def _performance_correlation_loop(self):
        """Background performance correlation analysis"""
        while self.system_active:
            try:
                self._analyze_performance_correlations()
                time.sleep(600)  # Analyze every 10 minutes
            except Exception as e:
                logging.error(f"Performance correlation error: {e}")
                time.sleep(120)

    def _analyze_performance_correlations(self):
        """Analyze correlations between different performance metrics"""
        try:
            # Get recent performance data
            self.performance_monitor.get_performance_summary(hours=1)
            self.cache.get_stats()

            # Simple correlation analysis
            correlations = {
                "cache_hit_rate_vs_response_time": self._calculate_cache_response_correlation(),
                "rate_limit_efficiency": self._calculate_rate_limit_efficiency(),
                "system_load_impact": self._calculate_system_load_impact(),
            }

            # Log significant correlations
            for correlation, value in correlations.items():
                if abs(value) > 0.5:  # Significant correlation
                    logging.info(
                        f"Performance correlation detected: {correlation} = {value:.2f}"
                    )

        except Exception as e:
            logging.error(f"Performance correlation analysis error: {e}")

    def _calculate_cache_response_correlation(self) -> float:
        """Calculate correlation between cache hit rate and response time"""
        # Simplified correlation calculation
        # In a real implementation, this would use statistical correlation
        cache_stats = self.cache.get_stats()

        if cache_stats["hit_rate"] > 0.8:
            return -0.7  # High cache hit rate correlates with lower response times
        elif cache_stats["hit_rate"] < 0.3:
            return 0.5  # Low cache hit rate correlates with higher response times
        else:
            return 0.0  # Neutral correlation

    def _calculate_rate_limit_efficiency(self) -> float:
        """Calculate rate limiting efficiency"""
        # This would analyze rate limiting effectiveness
        return 0.8  # Placeholder

    def _calculate_system_load_impact(self) -> float:
        """Calculate system load impact on performance"""
        # This would analyze system resource impact
        return 0.3  # Placeholder

    def _log_startup_summary(self):
        """Log startup summary with system status"""
        try:
            cache_stats = self.cache.get_stats()

            summary = {
                "timestamp": datetime.now().isoformat(),
                "cache_max_size_mb": cache_stats["max_size_mb"],
                "monitoring_components": [
                    "PerformanceMonitor",
                    "IntelligentCache",
                    "ProductionMonitor",
                    "AdaptiveRateLimiter",
                ],
                "optimization_features": [
                    "Adaptive rate limiting",
                    "Intelligent caching",
                    "Real-time performance monitoring",
                    "Usage pattern analysis",
                    "Automated optimization recommendations",
                ],
            }

            logging.info(
                f"Production monitoring startup summary: {json.dumps(summary, indent=2)}"
            )

        except Exception as e:
            logging.error(f"Failed to log startup summary: {e}")

    def record_api_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Dict = None,
        headers: Dict = None,
        bypass_cache: bool = False,
    ) -> Tuple[Any, Dict]:
        """
        Enhanced API request recording with full monitoring integration
        Returns: (response_data, metrics)
        """
        start_time = time.time()

        try:
            # Make cached API request
            response_data, cache_hit, response_time = self.cached_api.get(
                endpoint, params, headers, bypass_cache
            )

            success = response_data is not None

            # Record in all monitoring systems

            # 1. Performance Monitor
            self.performance_monitor.record_api_call(
                endpoint, method, response_time, success, cache_hit=cache_hit
            )

            # 2. Production Monitor
            self.production_monitor.record_api_request(
                endpoint, method, response_time, success, cache_hit
            )

            # 3. Rate Limiter
            self.rate_limiter.record_api_call(
                endpoint,
                response_time,
                success,
                rate_limited=False,  # Would be determined by actual rate limiting
            )

            # Compile metrics
            metrics = {
                "endpoint": endpoint,
                "method": method,
                "response_time": response_time,
                "success": success,
                "cache_hit": cache_hit,
                "timestamp": start_time,
            }

            return response_data, metrics

        except Exception as e:
            error_time = time.time() - start_time

            # Record error in monitoring systems
            self.performance_monitor.record_api_call(
                endpoint, method, error_time, False, error_message=str(e)
            )

            self.production_monitor.record_api_request(
                endpoint, method, error_time, False, False
            )

            logging.error(f"API request error for {endpoint}: {e}")

            return None, {
                "endpoint": endpoint,
                "method": method,
                "response_time": error_time,
                "success": False,
                "cache_hit": False,
                "error": str(e),
                "timestamp": start_time,
            }

    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            # Get status from all components
            dashboard_data = self.production_monitor.get_production_dashboard_data()
            cache_stats = self.cache.get_stats()
            optimization_report = self.rate_limiter.get_optimization_report()
            perf_summary = self.performance_monitor.get_performance_summary(hours=1)

            # Calculate overall health score
            health_score = self._calculate_health_score(
                dashboard_data, cache_stats, perf_summary
            )

            return {
                "system_active": self.system_active,
                "health_score": health_score,
                "performance_summary": perf_summary,
                "cache_statistics": cache_stats,
                "real_time_metrics": dashboard_data.get("realtime_metrics", {}),
                "system_metrics": dashboard_data.get("system_metrics", {}),
                "recent_alerts": dashboard_data.get("recent_alerts", []),
                "optimization_recommendations": optimization_report.get(
                    "recommendations", []
                ),
                "monitoring_uptime": time.time()
                - (
                    dashboard_data.get("monitoring_status", {}).get(
                        "uptime", time.time()
                    )
                ),
                "last_updated": time.time(),
            }

        except Exception as e:
            logging.error(f"Failed to get comprehensive status: {e}")
            return {
                "system_active": self.system_active,
                "health_score": 0.0,
                "error": str(e),
                "last_updated": time.time(),
            }

    def _calculate_health_score(
        self, dashboard_data: Dict, cache_stats: Dict, perf_summary: Dict
    ) -> float:
        """Calculate overall system health score (0-1)"""
        try:
            scores = []

            # Response time score (lower is better)
            avg_response_time = dashboard_data.get("realtime_metrics", {}).get(
                "avg_response_time", 5.0
            )
            response_score = max(0, 1 - (avg_response_time / 5.0))  # 5s = 0 score
            scores.append(response_score * 0.3)

            # Cache efficiency score
            cache_hit_rate = cache_stats.get("hit_rate", 0)
            cache_score = cache_hit_rate
            scores.append(cache_score * 0.2)

            # Error rate score (lower is better)
            error_rate = dashboard_data.get("realtime_metrics", {}).get(
                "error_rate", 0.1
            )
            error_score = max(0, 1 - (error_rate * 10))  # 10% error = 0 score
            scores.append(error_score * 0.3)

            # System resource score
            cpu_usage = dashboard_data.get("system_metrics", {}).get("cpu_usage", 100)
            memory_usage = dashboard_data.get("system_metrics", {}).get(
                "memory_usage", 100
            )
            resource_score = max(0, 1 - max(cpu_usage, memory_usage) / 100)
            scores.append(resource_score * 0.2)

            return sum(scores)

        except Exception as e:
            logging.error(f"Health score calculation error: {e}")
            return 0.5  # Neutral score on error

    def get_optimization_opportunities(self) -> List[Dict]:
        """Get current optimization opportunities"""
        try:
            opportunities = []

            # Cache optimization opportunities
            cache_stats = self.cache.get_stats()
            if cache_stats["hit_rate"] < 0.6:
                opportunities.append(
                    {
                        "type": "cache_optimization",
                        "priority": "high",
                        "description": f"Cache hit rate is low ({cache_stats['hit_rate']:.1%})",
                        "recommendation": "Increase cache TTL for frequently accessed endpoints",
                        "estimated_improvement": "30-50% response time reduction",
                    }
                )

            # Rate limit optimization opportunities
            rate_limit_recommendations = (
                self.rate_limiter.get_optimization_recommendations()
            )
            for rec in rate_limit_recommendations[:3]:  # Top 3
                opportunities.append(
                    {
                        "type": "rate_limit_optimization",
                        "priority": "medium" if rec.confidence_score > 0.7 else "low",
                        "description": rec.recommendation_reason,
                        "recommendation": f"Adjust rate limits for {rec.endpoint}",
                        "estimated_improvement": f"{rec.expected_improvement:.0%} efficiency gain",
                    }
                )

            # Performance optimization opportunities
            perf_summary = self.performance_monitor.get_performance_summary(hours=24)
            if perf_summary and "endpoints" in perf_summary:
                slow_endpoints = [
                    (ep, stats["avg_duration"])
                    for ep, stats in perf_summary["endpoints"].items()
                    if stats["avg_duration"] > 3.0
                ]

                for endpoint, avg_time in slow_endpoints[:2]:  # Top 2 slow endpoints
                    opportunities.append(
                        {
                            "type": "endpoint_optimization",
                            "priority": "high",
                            "description": f"Endpoint {endpoint} has high response time ({avg_time:.2f}s)",
                            "recommendation": "Investigate and optimize endpoint performance",
                            "estimated_improvement": "40-60% response time reduction",
                        }
                    )

            return opportunities

        except Exception as e:
            logging.error(f"Failed to get optimization opportunities: {e}")
            return []

    def apply_optimization_recommendation(self, optimization_id: str) -> bool:
        """Apply an optimization recommendation"""
        try:
            # This would implement automatic optimization application
            # For now, just log the request
            logging.info(f"Optimization application requested: {optimization_id}")

            # In a real implementation, this would:
            # 1. Validate the optimization
            # 2. Apply configuration changes
            # 3. Monitor results
            # 4. Rollback if needed

            return True

        except Exception as e:
            logging.error(f"Failed to apply optimization {optimization_id}: {e}")
            return False

    def generate_performance_report(self, hours: int = 24) -> Dict:
        """Generate comprehensive performance report"""
        try:
            # Get data from all components
            perf_summary = self.performance_monitor.get_performance_summary(hours)
            cache_stats = self.cache.get_stats()
            optimization_report = self.rate_limiter.get_optimization_report()
            comprehensive_status = self.get_comprehensive_status()
            opportunities = self.get_optimization_opportunities()

            # Calculate improvement metrics
            baseline_metrics = self.performance_baseline
            current_metrics = comprehensive_status.get("real_time_metrics", {})

            improvements = {}
            if baseline_metrics:
                for metric, current_value in current_metrics.items():
                    baseline_value = baseline_metrics.get(metric)
                    if baseline_value and baseline_value > 0:
                        improvement = (baseline_value - current_value) / baseline_value
                        improvements[metric] = improvement

            report = {
                "report_period_hours": hours,
                "generated_at": datetime.now().isoformat(),
                "system_health": {
                    "overall_score": comprehensive_status["health_score"],
                    "status": (
                        "healthy"
                        if comprehensive_status["health_score"] > 0.8
                        else "needs_attention"
                    ),
                },
                "performance_metrics": {
                    "summary": perf_summary,
                    "improvements": improvements,
                },
                "cache_performance": {
                    "statistics": cache_stats,
                    "efficiency_rating": (
                        "excellent"
                        if cache_stats["hit_rate"] > 0.8
                        else (
                            "good"
                            if cache_stats["hit_rate"] > 0.6
                            else "needs_improvement"
                        )
                    ),
                },
                "optimization_status": {
                    "active_optimizations": len(
                        optimization_report.get("recommendations", [])
                    ),
                    "opportunities": opportunities,
                },
                "alerts_summary": {
                    "recent_alerts": len(comprehensive_status.get("recent_alerts", [])),
                    "critical_issues": len(
                        [
                            alert
                            for alert in comprehensive_status.get("recent_alerts", [])
                            if alert.get("severity") == "critical"
                        ]
                    ),
                },
                "recommendations": {
                    "immediate_actions": [
                        opp for opp in opportunities if opp["priority"] == "high"
                    ],
                    "planned_optimizations": [
                        opp
                        for opp in opportunities
                        if opp["priority"] in ["medium", "low"]
                    ],
                },
            }

            return report

        except Exception as e:
            logging.error(f"Failed to generate performance report: {e}")
            return {"error": str(e), "generated_at": datetime.now().isoformat()}


# Global production system instance
_production_system = None


def get_production_system(config: Dict = None) -> ProductionPerformanceSystem:
    """Get global production system instance"""
    global _production_system
    if _production_system is None:
        _production_system = ProductionPerformanceSystem(config)
    return _production_system


if __name__ == "__main__":
    # Test production system integration
    logging.basicConfig(level=logging.INFO)

    print("Testing Production Performance System Integration...")

    # Initialize system
    config = {"cache_size_mb": 50, "monitoring_enabled": True}

    system = ProductionPerformanceSystem(config)
    system.start_production_monitoring()

    # Simulate API usage
    import random

    endpoints = [
        "/api/trading/statistics",
        "/api/trading/positions",
        "/api/market/tickers",
        "/api/cache/init",
    ]

    print("Simulating production API usage...")
    for i in range(30):
        endpoint = random.choice(endpoints)
        params = {"test": i}

        response_data, metrics = system.record_api_request(endpoint, params=params)

        if i % 10 == 0:
            print(
                f"Processed {i} requests - Last response time: {metrics['response_time']:.3f}s"
            )

    # Wait for monitoring to collect data
    time.sleep(3)

    # Get comprehensive status
    print("\nSystem Status:")
    status = system.get_comprehensive_status()
    print(json.dumps(status, indent=2, default=str))

    # Get optimization opportunities
    print("\nOptimization Opportunities:")
    opportunities = system.get_optimization_opportunities()
    for opp in opportunities:
        print(f"- {opp['type']} ({opp['priority']}): {opp['description']}")

    # Generate performance report
    print("\nPerformance Report:")
    report = system.generate_performance_report(hours=1)
    print(json.dumps(report, indent=2, default=str))

    system.stop_production_monitoring()
    print("\nProduction system integration test completed!")
