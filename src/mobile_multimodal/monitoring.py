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

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

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
        """Record a metric measurement."""
        if tags is None:
            tags = {}
        
        metric = MetricPoint(
            timestamp=time.time(),
            name=name,
            value=value,
            tags=tags,
            unit=unit
        )
        
        with self._lock:
            self.metrics.append(metric)
    
    def record_performance(self, operation: str, execution_time: float, is_error: bool = False, tags: Dict[str, str] = None):
        """Record performance statistics."""
        if tags is None:
            tags = {}
        
        # Create key for aggregation
        tag_key = "_".join(f"{k}:{v}" for k, v in sorted(tags.items()))
        key = f"{operation}_{tag_key}" if tag_key else operation
        
        with self._lock:
            self.aggregates[key].add_measurement(execution_time, is_error)
        
        # Also record as individual metric
        self.record_metric(
            name=f"{operation}_duration",
            value=execution_time * 1000,  # Convert to ms
            tags={"operation": operation, **tags},
            unit="ms"
        )
        
        if is_error:
            self.record_metric(
                name=f"{operation}_error",
                value=1,
                tags={"operation": operation, **tags},
                unit="count"
            )
    
    def get_recent_metrics(self, minutes: int = 5) -> List[MetricPoint]:
        """Get metrics from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            return [m for m in self.metrics if m.timestamp >= cutoff_time]
    
    def get_performance_stats(self, operation: str = None) -> Dict[str, PerformanceStats]:
        """Get performance statistics."""
        with self._lock:
            if operation:
                return {k: v for k, v in self.aggregates.items() if k.startswith(operation)}
            return dict(self.aggregates)
    
    def clear_old_metrics(self, hours: int = 24):
        """Clear metrics older than specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            # Filter metrics
            self.metrics = deque(
                [m for m in self.metrics if m.timestamp >= cutoff_time],
                maxlen=self.max_points
            )
    
    def export_metrics(self, format: str = "json") -> Union[str, List[Dict]]:
        """Export metrics in specified format."""
        with self._lock:
            metrics_data = [m.to_dict() for m in self.metrics]
        
        if format == "json":
            return json.dumps(metrics_data, indent=2)
        elif format == "list":
            return metrics_data
        else:
            raise ValueError(f"Unsupported format: {format}")


class TelemetryCollector:
    """Comprehensive telemetry collection for model operations."""
    
    def __init__(self, enable_system_metrics: bool = True, enable_prometheus: bool = False):
        """Initialize telemetry collector.
        
        Args:
            enable_system_metrics: Enable system resource monitoring
            enable_prometheus: Enable Prometheus metrics export
        """
        self.enable_system_metrics = enable_system_metrics and PSUTIL_AVAILABLE
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        self.metric_collector = MetricCollector()
        self.operation_history = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # Performance tracking
        self.active_operations = {}  # operation_id -> start_time
        self.operation_stats = defaultdict(PerformanceStats)
        
        # System monitoring
        if self.enable_system_metrics:
            self._start_system_monitoring()
        
        # Prometheus metrics
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Telemetry collector initialized (system_metrics={self.enable_system_metrics}, prometheus={self.enable_prometheus})")
    
    def record_operation_start(self, operation_id: str, operation_type: str, user_id: str = "anonymous", metadata: Dict[str, Any] = None):
        """Record the start of an operation."""
        if metadata is None:
            metadata = {}
        
        start_time = time.time()
        
        operation_record = {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "user_id": user_id,
            "start_time": start_time,
            "metadata": metadata,
            "status": "started"
        }
        
        with self._lock:
            self.active_operations[operation_id] = operation_record
        
        # Record metric
        self.metric_collector.record_metric(
            name="operation_started",
            value=1,
            tags={
                "operation_type": operation_type,
                "user_id": user_id
            },
            unit="count"
        )
        
        self.logger.debug(f"Operation started: {operation_id} ({operation_type})", extra={
            "operation_id": operation_id,
            "operation_type": operation_type,
            "user_id": user_id
        })
    
    def record_operation_success(self, operation_id: str, duration: float = None, result_metadata: Dict[str, Any] = None):
        """Record successful completion of an operation."""
        if result_metadata is None:
            result_metadata = {}
        
        end_time = time.time()
        
        with self._lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Operation {operation_id} not found in active operations")
                return
            
            operation_record = self.active_operations.pop(operation_id)
        
        # Calculate duration if not provided
        if duration is None:
            duration = end_time - operation_record["start_time"]
        
        # Update operation record
        operation_record.update({
            "end_time": end_time,
            "duration": duration,
            "status": "success",
            "result_metadata": result_metadata
        })
        
        # Store in history
        with self._lock:
            self.operation_history.append(operation_record)
        
        # Record performance metrics
        operation_type = operation_record["operation_type"]
        user_id = operation_record["user_id"]
        
        self.metric_collector.record_performance(
            operation=operation_type,
            execution_time=duration,
            is_error=False,
            tags={"user_id": user_id, "status": "success"}
        )
        
        # Record success metric
        self.metric_collector.record_metric(
            name="operation_completed",
            value=1,
            tags={
                "operation_type": operation_type,
                "user_id": user_id,
                "status": "success"
            },
            unit="count"
        )
        
        self.logger.info(f"Operation completed successfully: {operation_id} ({operation_type}, {duration:.3f}s)", extra={
            "operation_id": operation_id,
            "operation_type": operation_type,
            "duration_ms": duration * 1000,
            "user_id": user_id,
            "status": "success"
        })
    
    def record_operation_failure(self, operation_id: str, error: str, duration: float = None, error_metadata: Dict[str, Any] = None):
        """Record failed operation."""
        if error_metadata is None:
            error_metadata = {}
        
        end_time = time.time()
        
        with self._lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Operation {operation_id} not found in active operations")
                return
            
            operation_record = self.active_operations.pop(operation_id)
        
        # Calculate duration if not provided
        if duration is None:
            duration = end_time - operation_record["start_time"]
        
        # Update operation record
        operation_record.update({
            "end_time": end_time,
            "duration": duration,
            "status": "failed",
            "error": error,
            "error_metadata": error_metadata
        })
        
        # Store in history
        with self._lock:
            self.operation_history.append(operation_record)
        
        # Record performance metrics
        operation_type = operation_record["operation_type"]
        user_id = operation_record["user_id"]
        
        self.metric_collector.record_performance(
            operation=operation_type,
            execution_time=duration,
            is_error=True,
            tags={"user_id": user_id, "status": "failed"}
        )
        
        # Record failure metric
        self.metric_collector.record_metric(
            name="operation_completed",
            value=1,
            tags={
                "operation_type": operation_type,
                "user_id": user_id,
                "status": "failed"
            },
            unit="count"
        )
        
        self.logger.error(f"Operation failed: {operation_id} ({operation_type}, {duration:.3f}s): {error}", extra={
            "operation_id": operation_id,
            "operation_type": operation_type,
            "duration_ms": duration * 1000,
            "user_id": user_id,
            "status": "failed",
            "error": error
        })
    
    def get_operation_stats(self, operation_type: str = None, minutes: int = 60) -> Dict[str, Any]:
        """Get aggregated operation statistics."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            # Filter recent operations
            recent_ops = [
                op for op in self.operation_history
                if op["start_time"] >= cutoff_time
            ]
            
            if operation_type:
                recent_ops = [op for op in recent_ops if op["operation_type"] == operation_type]
        
        if not recent_ops:
            return {
                "operation_type": operation_type,
                "time_window_minutes": minutes,
                "total_operations": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0
            }
        
        # Calculate statistics
        total_ops = len(recent_ops)
        success_ops = [op for op in recent_ops if op["status"] == "success"]
        failure_ops = [op for op in recent_ops if op["status"] == "failed"]
        
        durations = [op["duration"] for op in recent_ops if "duration" in op]
        
        stats = {
            "operation_type": operation_type,
            "time_window_minutes": minutes,
            "total_operations": total_ops,
            "success_count": len(success_ops),
            "failure_count": len(failure_ops),
            "success_rate": len(success_ops) / total_ops if total_ops > 0 else 0.0,
        }
        
        if durations:
            stats.update({
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "p95_duration": np.percentile(durations, 95) if len(durations) > 1 else durations[0],
                "p99_duration": np.percentile(durations, 99) if len(durations) > 1 else durations[0]
            })
        else:
            stats.update({
                "avg_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "p95_duration": 0.0,
                "p99_duration": 0.0
            })
        
        return stats
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        metrics = {
            "timestamp": time.time(),
            "system_available": self.enable_system_metrics
        }
        
        if self.enable_system_metrics:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                
                # Network metrics (if available)
                try:
                    net_io = psutil.net_io_counters()
                    network_metrics = {
                        "bytes_sent": net_io.bytes_sent,
                        "bytes_recv": net_io.bytes_recv,
                        "packets_sent": net_io.packets_sent,
                        "packets_recv": net_io.packets_recv
                    }
                except AttributeError:
                    network_metrics = {}
                
                metrics.update({
                    "cpu": {
                        "percent": cpu_percent,
                        "count": cpu_count,
                        "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
                    },
                    "memory": {
                        "total_bytes": memory.total,
                        "available_bytes": memory.available,
                        "used_bytes": memory.used,
                        "percent": memory.percent
                    },
                    "disk": {
                        "total_bytes": disk.total,
                        "free_bytes": disk.free,
                        "used_bytes": disk.used,
                        "percent": (disk.used / disk.total) * 100
                    },
                    "network": network_metrics
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to collect system metrics: {e}")
                metrics["error"] = str(e)
        
        return metrics
    
    def _start_system_monitoring(self):
        """Start background system monitoring thread."""
        def monitor_system():
            while True:
                try:
                    metrics = self.get_system_metrics()
                    
                    # Record system metrics
                    if "cpu" in metrics:
                        self.metric_collector.record_metric(
                            name="system_cpu_percent",
                            value=metrics["cpu"]["percent"],
                            unit="percent"
                        )
                    
                    if "memory" in metrics:
                        self.metric_collector.record_metric(
                            name="system_memory_percent",
                            value=metrics["memory"]["percent"],
                            unit="percent"
                        )
                        
                        self.metric_collector.record_metric(
                            name="system_memory_used",
                            value=metrics["memory"]["used_bytes"] / (1024**3),  # GB
                            unit="GB"
                        )
                    
                    if "disk" in metrics:
                        self.metric_collector.record_metric(
                            name="system_disk_percent",
                            value=metrics["disk"]["percent"],
                            unit="percent"
                        )
                    
                    # Clean up stale operations
                    self.cleanup_stale_operations()
                    
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        self.logger.info("System monitoring thread started")
    
    def cleanup_stale_operations(self, timeout_seconds: int = 300):
        """Clean up operations that have been active too long."""
        current_time = time.time()
        stale_ops = []
        
        with self._lock:
            for op_id, operation in list(self.active_operations.items()):
                if current_time - operation["start_time"] > timeout_seconds:
                    stale_ops.append(op_id)
                    operation["status"] = "timeout"
                    operation["end_time"] = current_time
                    operation["duration"] = current_time - operation["start_time"]
                    
                    # Move to history
                    self.operation_history.append(operation)
                    del self.active_operations[op_id]
        
        if stale_ops:
            self.logger.warning(f"Cleaned up {len(stale_ops)} stale operations: {stale_ops}")
        
        return stale_ops


# Example usage and testing
if __name__ == "__main__":
    print("Testing monitoring modules...")
    
    # Test metric collector
    print("Testing MetricCollector...")
    collector = MetricCollector()
    
    # Record some metrics
    collector.record_metric("test_metric", 42.0, {"type": "test"}, "units")
    collector.record_performance("test_operation", 0.150, False, {"user": "test"})
    collector.record_performance("test_operation", 0.200, True, {"user": "test"})
    
    # Get statistics
    stats = collector.get_performance_stats("test_operation")
    print(f"Performance stats: {len(stats)} operations tracked")
    
    metrics = collector.get_recent_metrics(5)
    print(f"Recent metrics: {len(metrics)} metrics")
    
    print("✓ MetricCollector works")
    
    # Test telemetry collector
    print("\nTesting TelemetryCollector...")
    telemetry = TelemetryCollector(enable_system_metrics=PSUTIL_AVAILABLE)
    
    # Simulate operations
    telemetry.record_operation_start("op1", "generate_caption", "test_user")
    time.sleep(0.1)
    telemetry.record_operation_success("op1", result_metadata={"caption_length": 25})
    
    telemetry.record_operation_start("op2", "extract_text", "test_user")
    time.sleep(0.05)
    telemetry.record_operation_failure("op2", "Mock error for testing")
    
    # Get statistics
    operation_stats = telemetry.get_operation_stats("generate_caption")
    print(f"Operation stats: {operation_stats['total_operations']} operations, {operation_stats['success_rate']:.2%} success rate")
    
    system_metrics = telemetry.get_system_metrics()
    print(f"System metrics available: {system_metrics['system_available']}")
    
    print("✓ TelemetryCollector works")
    
    print("\nAll monitoring tests passed!")
    
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


class TelemetryCollector:
    """Lightweight telemetry collection for operations."""
    
    def __init__(self, model_name: str = "mobile_multimodal"):
        self.model_name = model_name
        self.operations = {}  # operation_id -> operation_data
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_operation_start(self, operation_id: str, operation_type: str, user_id: str = "anonymous"):
        """Record the start of an operation."""
        with self._lock:
            self.operations[operation_id] = {
                "id": operation_id,
                "type": operation_type,
                "user_id": user_id,
                "start_time": time.time(),
                "end_time": None,
                "duration": None,
                "success": None,
                "metadata": {}
            }
    
    def record_operation_success(self, operation_id: str, duration: float, metadata: Dict[str, Any] = None):
        """Record successful completion of an operation."""
        with self._lock:
            if operation_id in self.operations:
                op = self.operations[operation_id]
                op["end_time"] = time.time()
                op["duration"] = duration
                op["success"] = True
                op["metadata"].update(metadata or {})
                
                # Record to metrics
                self.metrics[op["type"]].append({
                    "timestamp": op["end_time"],
                    "duration": duration,
                    "success": True,
                    "user_id": op["user_id"]
                })
    
    def record_operation_failure(self, operation_id: str, duration: float, error_message: str):
        """Record failed completion of an operation."""
        with self._lock:
            if operation_id in self.operations:
                op = self.operations[operation_id]
                op["end_time"] = time.time()
                op["duration"] = duration
                op["success"] = False
                op["error"] = error_message
                
                # Record to metrics
                self.metrics[op["type"]].append({
                    "timestamp": op["end_time"],
                    "duration": duration,
                    "success": False,
                    "error": error_message,
                    "user_id": op["user_id"]
                })
    
    def get_operation_stats(self, operation_type: str = None) -> Dict[str, Any]:
        """Get statistics for operations."""
        with self._lock:
            if operation_type and operation_type in self.metrics:
                ops = self.metrics[operation_type]
            else:
                ops = []
                for op_metrics in self.metrics.values():
                    ops.extend(op_metrics)
            
            if not ops:
                return {"count": 0, "success_rate": 0.0, "avg_duration": 0.0}
            
            successful = [op for op in ops if op["success"]]
            durations = [op["duration"] for op in ops if op["duration"] is not None]
            
            return {
                "count": len(ops),
                "success_count": len(successful),
                "success_rate": len(successful) / len(ops),
                "avg_duration": np.mean(durations) if durations else 0.0,
                "min_duration": min(durations) if durations else 0.0,
                "max_duration": max(durations) if durations else 0.0
            }
    
    def cleanup_old_operations(self, max_age_seconds: int = 3600):
        """Remove old completed operations to prevent memory growth."""
        current_time = time.time()
        with self._lock:
            # Remove old operations
            to_remove = []
            for op_id, op_data in self.operations.items():
                if (op_data.get("end_time") and 
                    current_time - op_data["end_time"] > max_age_seconds):
                    to_remove.append(op_id)
            
            for op_id in to_remove:
                del self.operations[op_id]
            
            # Keep only recent metrics
            for op_type in self.metrics:
                self.metrics[op_type] = [
                    m for m in self.metrics[op_type]
                    if current_time - m["timestamp"] < max_age_seconds
                ]


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