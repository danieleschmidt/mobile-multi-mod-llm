"""Enhanced monitoring and telemetry system for Generation 2 robustness."""

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

logger = logging.getLogger(__name__)


@dataclass
class OperationMetric:
    """Individual operation metric."""
    operation_id: str
    operation_type: str
    user_id: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def complete_success(self, metadata: Optional[Dict[str, Any]] = None):
        """Mark operation as successfully completed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = True
        if metadata:
            self.metadata = self.metadata or {}
            self.metadata.update(metadata)
    
    def complete_failure(self, error_message: str):
        """Mark operation as failed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = False
        self.error_message = error_message


class TelemetryCollector:
    """Advanced telemetry collection with comprehensive monitoring."""
    
    def __init__(self, collection_interval: float = 10.0, 
                 max_metrics_memory: int = 10000,
                 enable_file_export: bool = True,
                 export_directory: str = "telemetry_data"):
        """Initialize telemetry collector."""
        self.collection_interval = collection_interval
        self.max_metrics_memory = max_metrics_memory
        self.enable_file_export = enable_file_export
        self.export_directory = Path(export_directory)
        
        # Metric storage
        self.operation_metrics: deque = deque(maxlen=max_metrics_memory)
        self.system_metrics: deque = deque(maxlen=max_metrics_memory)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Real-time aggregations
        self.operation_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        self.latency_histograms = defaultdict(list)
        
        # Collection state
        self.collecting = False
        self.collection_thread = None
        self.last_network_stats = None
        
        # Export state
        self.last_export_time = 0
        self.export_interval = 300  # 5 minutes
        
        # Create export directory
        if self.enable_file_export:
            self.export_directory.mkdir(exist_ok=True)
        
        logger.info(f"Telemetry collector initialized (interval={collection_interval}s)")
    
    def record_operation_start(self, operation_id: str, operation_type: str, user_id: str) -> OperationMetric:
        """Record the start of an operation."""
        metric = OperationMetric(
            operation_id=operation_id,
            operation_type=operation_type,
            user_id=user_id,
            start_time=time.time()
        )
        
        # Update real-time counters
        self.operation_counters[operation_type] += 1
        
        return metric
    
    def record_operation_success(self, operation_id: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Record successful completion of an operation."""
        # Find and update the metric
        for metric in reversed(self.operation_metrics):
            if metric.operation_id == operation_id:
                metric.complete_success(metadata)
                break
        else:
            # Create new metric if not found
            metric = OperationMetric(
                operation_id=operation_id,
                operation_type="unknown",
                user_id="unknown",
                start_time=time.time() - duration,
                end_time=time.time(),
                duration=duration,
                success=True,
                metadata=metadata
            )
        
        self.operation_metrics.append(metric)
        
        # Update histograms
        self.latency_histograms[metric.operation_type].append(duration)
        if len(self.latency_histograms[metric.operation_type]) > 1000:
            self.latency_histograms[metric.operation_type].pop(0)
    
    def record_operation_failure(self, operation_id: str, duration: float, error_message: str):
        """Record failed completion of an operation."""
        # Find and update the metric
        for metric in reversed(self.operation_metrics):
            if metric.operation_id == operation_id:
                metric.complete_failure(error_message)
                break
        else:
            # Create new metric if not found
            metric = OperationMetric(
                operation_id=operation_id,
                operation_type="unknown",
                user_id="unknown",
                start_time=time.time() - duration,
                end_time=time.time(),
                duration=duration,
                success=False,
                error_message=error_message
            )
        
        self.operation_metrics.append(metric)
        
        # Update error counters
        self.error_counters[metric.operation_type] += 1
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get aggregated operation statistics."""
        if not self.operation_metrics:
            return {"error": "No operation metrics available"}
        
        # Calculate success rates by operation type
        stats_by_type = defaultdict(lambda: {"total": 0, "success": 0, "failure": 0, "durations": []})
        
        for metric in self.operation_metrics:
            if metric.success is not None:  # Only completed operations
                op_type = metric.operation_type
                stats_by_type[op_type]["total"] += 1
                
                if metric.success:
                    stats_by_type[op_type]["success"] += 1
                else:
                    stats_by_type[op_type]["failure"] += 1
                
                if metric.duration is not None:
                    stats_by_type[op_type]["durations"].append(metric.duration)
        
        # Calculate aggregated statistics
        aggregated_stats = {}
        for op_type, stats in stats_by_type.items():
            durations = stats["durations"]
            
            aggregated_stats[op_type] = {
                "total_operations": stats["total"],
                "success_count": stats["success"],
                "failure_count": stats["failure"],
                "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
                "avg_duration": np.mean(durations) if durations else 0,
                "median_duration": np.median(durations) if durations else 0,
                "p95_duration": np.percentile(durations, 95) if durations else 0,
                "p99_duration": np.percentile(durations, 99) if durations else 0
            }
        
        # Overall statistics
        total_operations = sum(stats["total"] for stats in stats_by_type.values())
        total_success = sum(stats["success"] for stats in stats_by_type.values())
        all_durations = [d for stats in stats_by_type.values() for d in stats["durations"]]
        
        return {
            "by_operation_type": aggregated_stats,
            "overall": {
                "total_operations": total_operations,
                "success_rate": total_success / total_operations if total_operations > 0 else 0,
                "avg_duration": np.mean(all_durations) if all_durations else 0,
                "median_duration": np.median(all_durations) if all_durations else 0
            }
        }

    def start_collection(self):
        """Start background system metric collection."""
        if self.collecting:
            logger.warning("Telemetry collection already running")
            return
        
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Telemetry collection started")
    
    def stop_collection(self):
        """Stop background metric collection."""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        logger.info("Telemetry collection stopped")
    
    def _collection_loop(self):
        """Main collection loop for system metrics."""
        while self.collecting:
            try:
                self._collect_system_metrics()
                
                # Export metrics periodically
                current_time = time.time()
                if (self.enable_file_export and 
                    current_time - self.last_export_time > self.export_interval):
                    self._export_metrics()
                    self.last_export_time = current_time
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in telemetry collection loop: {e}", exc_info=True)
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                memory_percent = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                
                # Network usage
                network = psutil.net_io_counters()
                network_sent_mb = network.bytes_sent / (1024 * 1024)
                network_recv_mb = network.bytes_recv / (1024 * 1024)
                
            else:
                # Fallback metrics without psutil
                cpu_percent = 0
                memory_mb = 0
                memory_percent = 0
                disk_percent = 0
                network_sent_mb = 0
                network_recv_mb = 0
            
            # Create system metric
            system_metric = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "memory_percent": memory_percent,
                "disk_usage_percent": disk_percent,
                "network_sent_mb": network_sent_mb,
                "network_recv_mb": network_recv_mb,
            }
            
            self.system_metrics.append(system_metric)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _export_metrics(self):
        """Export metrics to files."""
        try:
            timestamp = int(time.time())
            
            # Export operation metrics
            if self.operation_metrics:
                operation_file = self.export_directory / f"operations_{timestamp}.json"
                operations_data = [asdict(metric) for metric in self.operation_metrics]
                
                with open(operation_file, 'w') as f:
                    json.dump(operations_data, f, indent=2, default=str)
                
                logger.debug(f"Exported {len(operations_data)} operation metrics")
            
            # Export system metrics
            if self.system_metrics:
                system_file = self.export_directory / f"system_{timestamp}.json"
                system_data = list(self.system_metrics)
                
                with open(system_file, 'w') as f:
                    json.dump(system_data, f, indent=2)
                
                logger.debug(f"Exported {len(system_data)} system metrics")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


class PerformanceTracker:
    """Track and analyze performance patterns."""
    
    def __init__(self, telemetry_collector: TelemetryCollector):
        self.telemetry = telemetry_collector
        self.performance_baselines = {}
        self.performance_alerts = []
        
    def establish_baseline(self, operation_type: str, duration_threshold: float = None):
        """Establish performance baseline for an operation type."""
        stats = self.telemetry.get_operation_stats()
        if operation_type in stats.get("by_operation_type", {}):
            op_stats = stats["by_operation_type"][operation_type]
            
            # Use 95th percentile as baseline if not provided
            if duration_threshold is None:
                duration_threshold = op_stats.get("p95_duration", 1.0)
            
            self.performance_baselines[operation_type] = {
                "duration_threshold": duration_threshold,
                "success_rate_threshold": 0.95,
                "established_at": time.time(),
                "sample_count": op_stats.get("total_operations", 0)
            }
            
            logger.info(f"Performance baseline established for {operation_type}: {duration_threshold:.3f}s")
    
    def check_performance_degradation(self) -> List[Dict[str, Any]]:
        """Check for performance degradation against baselines."""
        alerts = []
        current_stats = self.telemetry.get_operation_stats()
        
        for op_type, baseline in self.performance_baselines.items():
            if op_type in current_stats.get("by_operation_type", {}):
                current = current_stats["by_operation_type"][op_type]
                
                # Check duration degradation
                if current["avg_duration"] > baseline["duration_threshold"] * 1.5:
                    alerts.append({
                        "type": "duration_degradation",
                        "operation_type": op_type,
                        "current_avg": current["avg_duration"],
                        "baseline_threshold": baseline["duration_threshold"],
                        "severity": "high" if current["avg_duration"] > baseline["duration_threshold"] * 2 else "medium"
                    })
                
                # Check success rate degradation
                if current["success_rate"] < baseline["success_rate_threshold"]:
                    alerts.append({
                        "type": "success_rate_degradation",
                        "operation_type": op_type,
                        "current_rate": current["success_rate"],
                        "baseline_threshold": baseline["success_rate_threshold"],
                        "severity": "critical" if current["success_rate"] < 0.8 else "high"
                    })
        
        # Store new alerts
        for alert in alerts:
            alert["timestamp"] = time.time()
            self.performance_alerts.append(alert)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = time.time() - 86400
        self.performance_alerts = [a for a in self.performance_alerts if a["timestamp"] > cutoff_time]
        
        return alerts


class AnomalyDetector:
    """Detect anomalies in system and operation metrics."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly detection
        self.metric_history = defaultdict(list)
    
    def add_metric_value(self, metric_name: str, value: float):
        """Add a new metric value for anomaly detection."""
        self.metric_history[metric_name].append({
            "timestamp": time.time(),
            "value": value
        })
        
        # Keep only recent values (last 1000)
        if len(self.metric_history[metric_name]) > 1000:
            self.metric_history[metric_name].pop(0)
    
    def detect_anomalies(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Detect anomalies in metric values using statistical methods."""
        if metric_name not in self.metric_history:
            return None
        
        values = [entry["value"] for entry in self.metric_history[metric_name]]
        
        if len(values) < 10:  # Need sufficient data
            return None
        
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        current_val = values[-1]
        
        # Z-score based anomaly detection
        z_score = abs(current_val - mean_val) / std_val if std_val > 0 else 0
        
        if z_score > self.sensitivity:
            return {
                "metric_name": metric_name,
                "current_value": current_val,
                "mean_value": mean_val,
                "std_deviation": std_val,
                "z_score": z_score,
                "severity": "high" if z_score > 3.0 else "medium",
                "timestamp": time.time()
            }
        
        return None


class HealthChecker:
    """Comprehensive health monitoring system."""
    
    def __init__(self, telemetry_collector: TelemetryCollector):
        self.telemetry = telemetry_collector
        self.health_checks = {}
        self.health_history = deque(maxlen=1000)
        self.last_health_check = 0
        self.check_interval = 60  # 1 minute
        
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def perform_health_checks(self) -> Dict[str, Any]:
        """Perform all registered health checks."""
        current_time = time.time()
        health_result = {
            "timestamp": current_time,
            "overall_status": "healthy",
            "checks": {},
            "metrics": {}
        }
        
        # Run all registered checks
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = check_func()
                health_result["checks"][check_name] = check_result
                
                # Update overall status if any check fails
                if not check_result.get("healthy", True):
                    health_result["overall_status"] = "unhealthy"
                    
            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                health_result["checks"][check_name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_result["overall_status"] = "unhealthy"
        
        # Add telemetry metrics
        try:
            stats = self.telemetry.get_operation_stats()
            if "overall" in stats:
                health_result["metrics"]["success_rate"] = stats["overall"]["success_rate"]
                health_result["metrics"]["avg_duration"] = stats["overall"]["avg_duration"]
        except Exception as e:
            logger.warning(f"Failed to get telemetry stats for health check: {e}")
        
        # Store health check result
        self.health_history.append(health_result)
        self.last_health_check = current_time
        
        return health_result
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary over time."""
        if not self.health_history:
            return {"error": "No health check history available"}
        
        # Calculate health trends
        recent_checks = list(self.health_history)[-10:]  # Last 10 checks
        healthy_count = sum(1 for check in recent_checks if check["overall_status"] == "healthy")
        
        return {
            "current_status": self.health_history[-1]["overall_status"],
            "last_check_time": self.last_health_check,
            "recent_health_rate": healthy_count / len(recent_checks),
            "total_checks": len(self.health_history),
            "check_interval": self.check_interval
        }