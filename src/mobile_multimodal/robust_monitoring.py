"""Comprehensive monitoring and observability for mobile multi-modal LLM."""

import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric measurement."""
    timestamp: float
    value: float
    tags: Dict[str, str]
    metric_name: str

@dataclass
class PerformanceTrace:
    """Performance trace for operations."""
    trace_id: str
    operation: str
    start_time: float
    end_time: Optional[float]
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]

class MetricsCollector:
    """High-performance metrics collection system."""
    
    def __init__(self, max_points: int = 10000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.traces = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric point."""
        with self._lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {},
                metric_name=name
            )
            self.metrics[name].append(point)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            self.record_metric(f"{name}.count", self.counters[name], tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            self.record_metric(f"{name}.gauge", value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record histogram data."""
        with self._lock:
            self.histograms[name].append(value)
            
            # Keep only last 1000 points for memory efficiency
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            
            # Record statistical metrics
            values = self.histograms[name]
            if values:
                self.record_metric(f"{name}.min", min(values), tags)
                self.record_metric(f"{name}.max", max(values), tags)
                self.record_metric(f"{name}.avg", sum(values) / len(values), tags)
                
                # Percentiles
                sorted_values = sorted(values)
                p50_idx = int(len(sorted_values) * 0.5)
                p95_idx = int(len(sorted_values) * 0.95)
                p99_idx = int(len(sorted_values) * 0.99)
                
                self.record_metric(f"{name}.p50", sorted_values[p50_idx], tags)
                self.record_metric(f"{name}.p95", sorted_values[p95_idx], tags)
                self.record_metric(f"{name}.p99", sorted_values[p99_idx], tags)
    
    def start_trace(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a performance trace."""
        trace_id = str(uuid.uuid4())
        trace = PerformanceTrace(
            trace_id=trace_id,
            operation=operation,
            start_time=time.time(),
            end_time=None,
            success=False,
            error_message=None,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.traces.append(trace)
        
        return trace_id
    
    def end_trace(self, trace_id: str, success: bool = True, error_message: Optional[str] = None):
        """End a performance trace."""
        end_time = time.time()
        
        with self._lock:
            # Find and update trace
            for trace in reversed(self.traces):
                if trace.trace_id == trace_id:
                    trace.end_time = end_time
                    trace.success = success
                    trace.error_message = error_message
                    
                    # Record performance metrics
                    duration = end_time - trace.start_time
                    self.record_histogram(
                        f"operation.duration.{trace.operation}",
                        duration * 1000,  # Convert to milliseconds
                        {"success": str(success)}
                    )
                    
                    self.increment_counter(
                        f"operation.count.{trace.operation}",
                        1,
                        {"success": str(success)}
                    )
                    break
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            summary = {
                "timestamp": time.time(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "recent_traces": []
            }
            
            # Add recent successful traces
            recent_traces = list(self.traces)[-10:]
            for trace in recent_traces:
                if trace.end_time:
                    summary["recent_traces"].append({
                        "operation": trace.operation,
                        "duration_ms": (trace.end_time - trace.start_time) * 1000,
                        "success": trace.success,
                        "error": trace.error_message
                    })
            
            # Add histogram statistics
            histogram_stats = {}
            for name, values in self.histograms.items():
                if values:
                    sorted_values = sorted(values)
                    histogram_stats[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p50": sorted_values[int(len(sorted_values) * 0.5)],
                        "p95": sorted_values[int(len(sorted_values) * 0.95)],
                        "p99": sorted_values[int(len(sorted_values) * 0.99)]
                    }
            
            summary["histograms"] = histogram_stats
            
            return summary

class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 5.0):
        """Start system monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics.set_gauge("system.cpu.percent", cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.set_gauge("system.memory.percent", memory.percent)
        self.metrics.set_gauge("system.memory.available_gb", memory.available / 1024**3)
        self.metrics.set_gauge("system.memory.used_gb", memory.used / 1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('.')
        self.metrics.set_gauge("system.disk.free_gb", disk.free / 1024**3)
        self.metrics.set_gauge("system.disk.used_percent", (disk.used / disk.total) * 100)

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.health_checks = {}
        self.last_check_time = {}
    
    def register_check(self, name: str, check_func, interval: float = 60.0):
        """Register a health check function."""
        self.health_checks[name] = {
            "func": check_func,
            "interval": interval,
            "last_result": None,
            "last_error": None
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            "timestamp": time.time(),
            "overall_status": "HEALTHY",
            "checks": {}
        }
        
        for name, check_config in self.health_checks.items():
            try:
                start_time = time.time()
                check_result = check_config["func"]()
                duration = (time.time() - start_time) * 1000
                
                results["checks"][name] = {
                    "status": "HEALTHY" if check_result else "UNHEALTHY",
                    "duration_ms": duration,
                    "result": check_result,
                    "error": None
                }
                
                check_config["last_result"] = check_result
                check_config["last_error"] = None
                
                if not check_result:
                    results["overall_status"] = "UNHEALTHY"
                
            except Exception as e:
                results["checks"][name] = {
                    "status": "ERROR",
                    "duration_ms": 0,
                    "result": None,
                    "error": str(e)
                }
                
                check_config["last_result"] = None
                check_config["last_error"] = str(e)
                results["overall_status"] = "UNHEALTHY"
        
        return results

class AlertManager:
    """Alert management system."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alert_rules = []
        self.active_alerts = {}
    
    def add_rule(self, name: str, condition_func, severity: str = "WARNING", cooldown: float = 300.0):
        """Add an alert rule."""
        self.alert_rules.append({
            "name": name,
            "condition": condition_func,
            "severity": severity,
            "cooldown": cooldown
        })
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alert rules and return active alerts."""
        current_time = time.time()
        new_alerts = []
        
        for rule in self.alert_rules:
            rule_name = rule["name"]
            
            try:
                # Check if alert condition is met
                triggered = rule["condition"](self.metrics)
                
                if triggered:
                    # Check if alert is in cooldown
                    if (rule_name not in self.active_alerts or 
                        current_time - self.active_alerts[rule_name]["timestamp"] > rule["cooldown"]):
                        
                        alert = {
                            "name": rule_name,
                            "severity": rule["severity"],
                            "timestamp": current_time,
                            "message": f"Alert {rule_name} triggered"
                        }
                        
                        self.active_alerts[rule_name] = alert
                        new_alerts.append(alert)
                        
                        logger.warning(f"ALERT: {alert['message']}")
                
                else:
                    # Clear alert if condition no longer met
                    if rule_name in self.active_alerts:
                        del self.active_alerts[rule_name]
                        logger.info(f"Alert {rule_name} cleared")
            
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
        
        return new_alerts

@contextmanager
def performance_trace(metrics: MetricsCollector, operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for performance tracing."""
    trace_id = metrics.start_trace(operation, metadata)
    try:
        yield trace_id
        metrics.end_trace(trace_id, success=True)
    except Exception as e:
        metrics.end_trace(trace_id, success=False, error_message=str(e))
        raise

class ObservabilityManager:
    """Centralized observability management."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics)
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager(self.metrics)
        self._setup_default_health_checks()
        self._setup_default_alerts()
    
    def start(self):
        """Start all monitoring components."""
        self.system_monitor.start_monitoring()
        logger.info("Observability system started")
    
    def stop(self):
        """Stop all monitoring components."""
        self.system_monitor.stop_monitoring()
        logger.info("Observability system stopped")
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        def memory_check():
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if memory usage > 90%
        
        def disk_check():
            disk = psutil.disk_usage('.')
            return (disk.free / disk.total) > 0.1  # Alert if < 10% free
        
        self.health_checker.register_check("memory", memory_check)
        self.health_checker.register_check("disk", disk_check)
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        def high_memory_alert(metrics):
            return metrics.gauges.get("system.memory.percent", 0) > 85
        
        def low_disk_alert(metrics):
            return metrics.gauges.get("system.disk.free_gb", 100) < 1.0
        
        self.alert_manager.add_rule("high_memory", high_memory_alert, "WARNING")
        self.alert_manager.add_rule("low_disk", low_disk_alert, "CRITICAL")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "metrics": self.metrics.get_metrics_summary(),
            "health": self.health_checker.run_all_checks(),
            "alerts": list(self.alert_manager.active_alerts.values())
        }