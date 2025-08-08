"""Advanced monitoring and observability for mobile multi-modal models."""

import json
import logging
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np

# Configure structured logging
class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record):
        log_data = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


@dataclass
class MetricPoint:
    """Single metric measurement."""
    timestamp: float
    name: str
    value: float
    tags: Dict[str, str]
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceStats:
    """Performance statistics for operations."""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    error_count: int = 0
    
    def add_measurement(self, execution_time: float, is_error: bool = False):
        """Add a new measurement."""
        self.count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        if is_error:
            self.error_count += 1
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / self.count if self.count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "avg_time_ms": self.avg_time * 1000,
            "min_time_ms": self.min_time * 1000 if self.min_time != float('inf') else 0,
            "max_time_ms": self.max_time * 1000,
            "error_count": self.error_count,
            "error_rate": self.error_rate
        }


class MetricCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics = deque(maxlen=max_points)
        self.aggregates = defaultdict(PerformanceStats)
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = ""):
        """Record a metric point."""
        with self._lock:
            point = MetricPoint(
                timestamp=time.time(),
                name=name,
                value=value,
                tags=tags or {},
                unit=unit
            )
            self.metrics.append(point)
    
    def record_execution_time(self, operation: str, execution_time: float, 
                            is_error: bool = False, tags: Dict[str, str] = None):
        """Record execution time for an operation."""
        with self._lock:
            self.aggregates[operation].add_measurement(execution_time, is_error)
            self.record_metric(
                f"{operation}_duration", 
                execution_time, 
                tags or {}, 
                "seconds"
            )
            if is_error:
                self.record_metric(f"{operation}_error", 1, tags or {}, "count")
    
    def get_metrics(self, name_filter: str = None, 
                   time_range: Tuple[float, float] = None) -> List[MetricPoint]:
        """Get metrics with optional filtering."""
        with self._lock:
            filtered = list(self.metrics)
            
            if name_filter:
                filtered = [m for m in filtered if name_filter in m.name]
            
            if time_range:
                start_time, end_time = time_range
                filtered = [m for m in filtered 
                           if start_time <= m.timestamp <= end_time]
            
            return filtered
    
    def get_aggregated_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated performance statistics."""
        with self._lock:
            return {op: stats.to_dict() for op, stats in self.aggregates.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            current_time = time.time()
            recent_metrics = [m for m in self.metrics 
                            if current_time - m.timestamp < 300]  # Last 5 minutes
            
            return {
                "total_metrics": len(self.metrics),
                "recent_metrics": len(recent_metrics),
                "operations": list(self.aggregates.keys()),
                "aggregated_stats": self.get_aggregated_stats()
            }


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self):
        self.alerts = deque(maxlen=1000)
        self.alert_handlers = []
        self._lock = threading.Lock()
        
        # Alert thresholds
        self.thresholds = {
            "high_error_rate": 0.1,  # 10%
            "high_latency": 5.0,     # 5 seconds
            "high_memory": 1024,     # 1GB
            "circuit_breaker_open": True
        }
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and trigger alerts."""
        current_time = time.time()
        
        # Check error rate
        for op_name, stats in metrics.get("aggregated_stats", {}).items():
            if stats["error_rate"] > self.thresholds["high_error_rate"]:
                self._trigger_alert("high_error_rate", {
                    "operation": op_name,
                    "error_rate": stats["error_rate"],
                    "threshold": self.thresholds["high_error_rate"]
                })
            
            if stats["avg_time_ms"] / 1000 > self.thresholds["high_latency"]:
                self._trigger_alert("high_latency", {
                    "operation": op_name,
                    "avg_latency_ms": stats["avg_time_ms"],
                    "threshold_ms": self.thresholds["high_latency"] * 1000
                })
    
    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger an alert."""
        alert = {
            "timestamp": time.time(),
            "type": alert_type,
            "details": details,
            "severity": self._get_alert_severity(alert_type)
        }
        
        with self._lock:
            self.alerts.append(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level."""
        severity_map = {
            "high_error_rate": "high",
            "high_latency": "medium",
            "high_memory": "medium",
            "circuit_breaker_open": "high"
        }
        return severity_map.get(alert_type, "low")
    
    def get_active_alerts(self, severity: str = None) -> List[Dict[str, Any]]:
        """Get active alerts."""
        with self._lock:
            alerts = list(self.alerts)
            
            if severity:
                alerts = [a for a in alerts if a["severity"] == severity]
            
            # Consider alerts from last hour as active
            current_time = time.time()
            return [a for a in alerts if current_time - a["timestamp"] < 3600]


class DashboardExporter:
    """Export metrics for dashboard visualization."""
    
    def __init__(self, export_dir: str = "metrics_export"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
    
    def export_prometheus_format(self, metrics: List[MetricPoint]) -> str:
        """Export metrics in Prometheus format."""
        prometheus_lines = []
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.name].append(metric)
        
        for name, points in metric_groups.items():
            # Add help and type
            prometheus_lines.append(f"# HELP {name} {name} metric")
            prometheus_lines.append(f"# TYPE {name} gauge")
            
            for point in points:
                # Format tags
                tags_str = ""
                if point.tags:
                    tags_list = [f'{k}="{v}"' for k, v in point.tags.items()]
                    tags_str = "{" + ",".join(tags_list) + "}"
                
                prometheus_lines.append(
                    f"{name}{tags_str} {point.value} {int(point.timestamp * 1000)}"
                )
        
        return "\n".join(prometheus_lines)
    
    def export_json_format(self, data: Dict[str, Any]) -> str:
        """Export data in JSON format."""
        return json.dumps(data, indent=2, default=str)
    
    def export_csv_format(self, metrics: List[MetricPoint]) -> str:
        """Export metrics in CSV format."""
        lines = ["timestamp,name,value,tags,unit"]
        
        for metric in metrics:
            tags_str = json.dumps(metric.tags) if metric.tags else ""
            lines.append(
                f"{metric.timestamp},{metric.name},{metric.value},"
                f'"{tags_str}",{metric.unit}'
            )
        
        return "\n".join(lines)
    
    def save_export(self, data: str, filename: str):
        """Save exported data to file."""
        file_path = self.export_dir / filename
        file_path.write_text(data, encoding='utf-8')


class ModelMonitor:
    """Comprehensive model monitoring system."""
    
    def __init__(self, model_name: str = "mobile_multimodal"):
        self.model_name = model_name
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.dashboard_exporter = DashboardExporter()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # System metrics
        self.system_metrics = SystemMetrics()
        
        # Setup default alert handlers
        self.alert_manager.add_alert_handler(self._log_alert)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger."""
        logger = logging.getLogger(f"monitor.{self.model_name}")
        logger.setLevel(logging.INFO)
        
        # Add structured formatter
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(StructuredFormatter())
            logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self, interval: float = 30.0):
        """Start background monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                sys_metrics = self.system_metrics.collect()
                for name, value in sys_metrics.items():
                    self.metric_collector.record_metric(
                        f"system_{name}", 
                        value, 
                        {"model": self.model_name}
                    )
                
                # Check thresholds and trigger alerts
                summary = self.metric_collector.get_summary()
                self.alert_manager.check_thresholds(summary)
                
                # Log monitoring status
                self.logger.debug("Monitoring check completed", extra={
                    "total_metrics": summary["total_metrics"],
                    "active_alerts": len(self.alert_manager.get_active_alerts())
                })
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
            
            time.sleep(interval)
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Default alert handler - log to structured logger."""
        self.logger.warning(f"ALERT: {alert['type']}", extra=alert)
    
    def record_inference(self, operation: str, duration: float, 
                        input_size: int = None, output_size: int = None,
                        is_error: bool = False, error_type: str = None):
        """Record inference operation metrics."""
        tags = {"model": self.model_name, "operation": operation}
        
        if input_size:
            tags["input_size"] = str(input_size)
        if output_size:
            tags["output_size"] = str(output_size)
        if is_error and error_type:
            tags["error_type"] = error_type
        
        self.metric_collector.record_execution_time(
            operation, duration, is_error, tags
        )
        
        # Record additional metrics
        self.metric_collector.record_metric(
            "inference_count", 1, tags, "count"
        )
        
        if input_size:
            self.metric_collector.record_metric(
                "input_size", input_size, tags, "bytes"
            )
    
    def record_custom_metric(self, name: str, value: float, 
                           tags: Dict[str, str] = None, unit: str = ""):
        """Record custom metric."""
        base_tags = {"model": self.model_name}
        if tags:
            base_tags.update(tags)
        
        self.metric_collector.record_metric(name, value, base_tags, unit)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for dashboard."""
        summary = self.metric_collector.get_summary()
        alerts = self.alert_manager.get_active_alerts()
        
        return {
            "model_name": self.model_name,
            "timestamp": time.time(),
            "metrics_summary": summary,
            "active_alerts": alerts,
            "system_health": self.system_metrics.get_health_status()
        }
    
    def export_metrics(self, format_type: str = "json", 
                      time_range: Tuple[float, float] = None) -> str:
        """Export metrics in specified format."""
        if format_type == "prometheus":
            metrics = self.metric_collector.get_metrics(time_range=time_range)
            return self.dashboard_exporter.export_prometheus_format(metrics)
        elif format_type == "csv":
            metrics = self.metric_collector.get_metrics(time_range=time_range)
            return self.dashboard_exporter.export_csv_format(metrics)
        else:  # json
            return self.dashboard_exporter.export_json_format(
                self.get_dashboard_data()
            )


class SystemMetrics:
    """Collect system-level metrics."""
    
    def collect(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        try:
            import psutil
            
            # CPU usage
            metrics["cpu_percent"] = psutil.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics["memory_percent"] = memory.percent
            metrics["memory_used_mb"] = memory.used / (1024 * 1024)
            metrics["memory_available_mb"] = memory.available / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics["disk_percent"] = (disk.used / disk.total) * 100
            metrics["disk_used_gb"] = disk.used / (1024 ** 3)
            metrics["disk_free_gb"] = disk.free / (1024 ** 3)
            
        except ImportError:
            # Fallback metrics without psutil
            try:
                import os
                load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
                metrics["load_average"] = load_avg
            except:
                pass
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        metrics = self.collect()
        
        health = {
            "status": "healthy",
            "checks": {}
        }
        
        # CPU check
        cpu_percent = metrics.get("cpu_percent", 0)
        health["checks"]["cpu"] = cpu_percent < 80
        if cpu_percent >= 80:
            health["status"] = "degraded"
        
        # Memory check
        memory_percent = metrics.get("memory_percent", 0)
        health["checks"]["memory"] = memory_percent < 85
        if memory_percent >= 85:
            health["status"] = "degraded"
        
        # Disk check
        disk_percent = metrics.get("disk_percent", 0)
        health["checks"]["disk"] = disk_percent < 90
        if disk_percent >= 90:
            health["status"] = "degraded"
        
        if health["status"] == "degraded":
            health["warnings"] = [
                check for check, passed in health["checks"].items() 
                if not passed
            ]
        
        return health


# Performance monitoring decorator
def monitor_performance(monitor: ModelMonitor, operation: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start_time = time.time()
            is_error = False
            error_type = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                is_error = True
                error_type = type(e).__name__
                raise
            finally:
                duration = time.time() - start_time
                monitor.record_inference(
                    op_name, duration, is_error=is_error, error_type=error_type
                )
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    print("Testing monitoring system...")
    
    # Create monitor
    monitor = ModelMonitor("test_model")
    
    # Record some metrics
    monitor.record_inference("generate_caption", 0.150, 1024, 256)
    monitor.record_inference("extract_text", 0.090, 2048, 128)
    monitor.record_inference("generate_caption", 0.200, 1024, 256, True, "TimeoutError")
    
    # Record custom metrics
    monitor.record_custom_metric("gpu_temperature", 75.5, unit="celsius")
    monitor.record_custom_metric("model_confidence", 0.89)
    
    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print(f"Dashboard data keys: {list(dashboard.keys())}")
    
    # Export metrics
    json_export = monitor.export_metrics("json")
    print(f"JSON export size: {len(json_export)} characters")
    
    # Test decorator
    @monitor_performance(monitor, "test_operation")
    def test_function():
        time.sleep(0.1)
        return "success"
    
    result = test_function()
    print(f"Decorated function result: {result}")
    
    # Get final summary
    summary = monitor.metric_collector.get_summary()
    print(f"Total operations tracked: {len(summary['operations'])}")
    
    print("Monitoring system test completed!")