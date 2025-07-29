#!/usr/bin/env python3
"""Mobile telemetry collector for deployment observability.

This script collects and processes telemetry data from mobile deployments
of the Multi-Modal LLM for comprehensive observability.
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

try:
    import prometheus_client
    from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available. Install with: pip install prometheus-client")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MobileInferenceMetric:
    """Mobile inference performance metric."""
    timestamp: str
    device_id: str
    device_type: str
    os_version: str
    model_version: str
    task_type: str
    latency_ms: float
    memory_usage_mb: float
    battery_level: float
    cpu_temperature: Optional[float] = None
    gpu_temperature: Optional[float] = None
    accuracy_score: Optional[float] = None
    network_type: Optional[str] = None
    thermal_throttled: bool = False


@dataclass
class MobileDeploymentMetric:
    """Mobile deployment health metric."""
    timestamp: str
    device_id: str
    device_type: str
    app_version: str
    model_version: str
    load_time_ms: float
    crash_occurred: bool = False
    crash_reason: Optional[str] = None
    network_usage_mb: float = 0.0
    compatibility_score: float = 1.0


class MobileTelemetryCollector:
    """Collects and processes mobile telemetry data."""
    
    def __init__(self, config_file: str = "mobile-telemetry-config.json"):
        self.config = self._load_config(config_file)
        self.metrics_buffer = []
        self.deployment_buffer = []
        self.lock = threading.Lock()
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
            self.prometheus_server = None
        
        # Device classification cache
        self.device_cache = {}
        
        # Start background processing
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_metrics)
        self.processor_thread.daemon = True
        self.processor_thread.start()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load telemetry configuration."""
        default_config = {
            "collection_interval_seconds": 30,
            "batch_size": 100,
            "prometheus_port": 8090,
            "export_endpoints": [],
            "device_classification": {
                "flagship_threshold_ram_gb": 8,
                "midrange_threshold_ram_gb": 4,
                "budget_threshold_ram_gb": 3
            },
            "performance_thresholds": {
                "latency_warning_ms": 50,
                "latency_critical_ms": 100,
                "accuracy_warning": 0.85,
                "accuracy_critical": 0.80,
                "memory_warning_mb": 200,
                "memory_critical_mb": 300
            },
            "retention_days": 30
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            logger.info(f"Config file {config_file} not found, using defaults")
            return default_config
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.inference_latency = Histogram(
            'mobile_inference_latency_seconds',
            'Mobile inference latency in seconds',
            ['device_type', 'os_version', 'model_version', 'task_type'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'mobile_model_accuracy',
            'Model accuracy on mobile devices',
            ['device_type', 'model_version', 'quantization_level'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'mobile_memory_usage_mb',
            'Memory usage during mobile inference',
            ['device_type', 'model_size_mb'],
            registry=self.registry
        )
        
        self.battery_impact = Gauge(
            'mobile_battery_level',
            'Battery level during inference',
            ['device_type', 'charging_state'],
            registry=self.registry
        )
        
        self.thermal_throttling = Counter(
            'mobile_thermal_throttling_total',
            'Thermal throttling events',
            ['device_type'],
            registry=self.registry
        )
        
        self.app_crashes = Counter(
            'mobile_app_crashes_total',
            'Mobile app crashes',
            ['device_type', 'os_version', 'crash_reason'],
            registry=self.registry
        )
        
        self.model_load_time = Histogram(
            'mobile_model_load_time_seconds',
            'Model load time on mobile',
            ['device_type', 'storage_type', 'model_size_mb'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
    
    def start_prometheus_server(self, port: Optional[int] = None):
        """Start Prometheus metrics server."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available")
            return
        
        port = port or self.config["prometheus_port"]
        
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {port}")
            self.prometheus_server = port
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def classify_device(self, device_info: Dict[str, Any]) -> str:
        """Classify device into performance tiers."""
        device_key = f"{device_info.get('model', '')}_{device_info.get('ram_gb', 0)}"
        
        if device_key in self.device_cache:
            return self.device_cache[device_key]
        
        ram_gb = device_info.get('ram_gb', 0)
        cpu_cores = device_info.get('cpu_cores', 0)
        
        if ram_gb >= self.config["device_classification"]["flagship_threshold_ram_gb"]:
            classification = "flagship"
        elif ram_gb >= self.config["device_classification"]["midrange_threshold_ram_gb"]:
            classification = "midrange"
        elif ram_gb >= self.config["device_classification"]["budget_threshold_ram_gb"]:
            classification = "budget"
        else:
            classification = "entry_level"
        
        self.device_cache[device_key] = classification
        return classification
    
    def collect_inference_metric(self, metric: MobileInferenceMetric):
        """Collect mobile inference performance metric."""
        with self.lock:
            self.metrics_buffer.append(metric)
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and self.registry:
            try:
                labels = [
                    metric.device_type,
                    metric.os_version,
                    metric.model_version,
                    metric.task_type
                ]
                self.inference_latency.labels(*labels).observe(metric.latency_ms / 1000.0)
                
                if metric.accuracy_score is not None:
                    self.model_accuracy.labels(
                        metric.device_type,
                        metric.model_version,
                        "int2"  # Assuming INT2 quantization
                    ).set(metric.accuracy_score)
                
                self.memory_usage.labels(
                    metric.device_type,
                    "35"  # Model size from spec
                ).set(metric.memory_usage_mb)
                
                self.battery_impact.labels(
                    metric.device_type,
                    "unknown"  # Would need charging state info
                ).set(metric.battery_level)
                
                if metric.thermal_throttled:
                    self.thermal_throttling.labels(metric.device_type).inc()
                    
            except Exception as e:
                logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def collect_deployment_metric(self, metric: MobileDeploymentMetric):
        """Collect mobile deployment health metric."""
        with self.lock:
            self.deployment_buffer.append(metric)
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and self.registry:
            try:
                if metric.crash_occurred:
                    self.app_crashes.labels(
                        metric.device_type,
                        "unknown",  # Would need OS version
                        metric.crash_reason or "unknown"
                    ).inc()
                
                self.model_load_time.labels(
                    metric.device_type,
                    "internal",  # Storage type
                    "35"  # Model size
                ).observe(metric.load_time_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Failed to update deployment metrics: {e}")
    
    def _process_metrics(self):
        """Background thread to process collected metrics."""
        while self.running:
            try:
                time.sleep(self.config["collection_interval_seconds"])
                
                with self.lock:
                    if len(self.metrics_buffer) >= self.config["batch_size"]:
                        batch = self.metrics_buffer[:self.config["batch_size"]]
                        self.metrics_buffer = self.metrics_buffer[self.config["batch_size"]:]
                        
                        self._export_metrics_batch(batch)
                    
                    if len(self.deployment_buffer) >= self.config["batch_size"]:
                        batch = self.deployment_buffer[:self.config["batch_size"]]
                        self.deployment_buffer = self.deployment_buffer[self.config["batch_size"]:]
                        
                        self._export_deployment_batch(batch)
                
            except Exception as e:
                logger.error(f"Error processing metrics: {e}")
    
    def _export_metrics_batch(self, batch: List[MobileInferenceMetric]):
        """Export batch of inference metrics."""
        logger.info(f"Exporting batch of {len(batch)} inference metrics")
        
        # Convert to JSON for export
        json_batch = [asdict(metric) for metric in batch]
        
        # Save to local file
        timestamp = int(time.time())
        filename = f"mobile_inference_metrics_{timestamp}.json"
        filepath = Path("telemetry") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(json_batch, f, indent=2)
        
        # Export to configured endpoints
        for endpoint in self.config.get("export_endpoints", []):
            self._send_to_endpoint(endpoint, json_batch, "inference_metrics")
    
    def _export_deployment_batch(self, batch: List[MobileDeploymentMetric]):
        """Export batch of deployment metrics."""
        logger.info(f"Exporting batch of {len(batch)} deployment metrics")
        
        json_batch = [asdict(metric) for metric in batch]
        
        timestamp = int(time.time())
        filename = f"mobile_deployment_metrics_{timestamp}.json"
        filepath = Path("telemetry") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(json_batch, f, indent=2)
        
        for endpoint in self.config.get("export_endpoints", []):
            self._send_to_endpoint(endpoint, json_batch, "deployment_metrics")
    
    def _send_to_endpoint(self, endpoint: Dict[str, str], data: List[Dict], metric_type: str):
        """Send metrics to configured endpoint."""
        try:
            import requests
            
            payload = {
                "timestamp": datetime.now().isoformat(),
                "metric_type": metric_type,
                "data": data
            }
            
            response = requests.post(
                endpoint["url"],
                json=payload,
                headers=endpoint.get("headers", {}),
                timeout=30
            )
            response.raise_for_status()
            
            logger.info(f"Successfully sent {len(data)} metrics to {endpoint['name']}")
            
        except Exception as e:
            logger.error(f"Failed to send metrics to {endpoint['name']}: {e}")
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance report from collected metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            # Filter recent metrics
            recent_metrics = [
                m for m in self.metrics_buffer
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            recent_deployments = [
                m for m in self.deployment_buffer
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
        
        # Calculate statistics
        if recent_metrics:
            latencies = [m.latency_ms for m in recent_metrics]
            memory_usage = [m.memory_usage_mb for m in recent_metrics]
            accuracies = [m.accuracy_score for m in recent_metrics if m.accuracy_score is not None]
            
            report = {
                "time_range_hours": hours,
                "total_inferences": len(recent_metrics),
                "performance_stats": {
                    "latency_ms": {
                        "avg": sum(latencies) / len(latencies),
                        "p50": sorted(latencies)[len(latencies)//2],
                        "p95": sorted(latencies)[int(len(latencies)*0.95)],
                        "p99": sorted(latencies)[int(len(latencies)*0.99)]
                    },
                    "memory_usage_mb": {
                        "avg": sum(memory_usage) / len(memory_usage),
                        "max": max(memory_usage),
                        "min": min(memory_usage)
                    }
                },
                "device_breakdown": self._analyze_device_performance(recent_metrics),
                "quality_metrics": {
                    "accuracy": {
                        "avg": sum(accuracies) / len(accuracies) if accuracies else None,
                        "min": min(accuracies) if accuracies else None
                    }
                },
                "reliability_metrics": {
                    "total_deployments": len(recent_deployments),
                    "crashes": len([d for d in recent_deployments if d.crash_occurred]),
                    "crash_rate": len([d for d in recent_deployments if d.crash_occurred]) / len(recent_deployments) if recent_deployments else 0
                }
            }
        else:
            report = {
                "time_range_hours": hours,
                "total_inferences": 0,
                "message": "No metrics available for the specified time range"
            }
        
        return report
    
    def _analyze_device_performance(self, metrics: List[MobileInferenceMetric]) -> Dict[str, Any]:
        """Analyze performance by device type."""
        device_stats = {}
        
        for metric in metrics:
            device_type = metric.device_type
            if device_type not in device_stats:
                device_stats[device_type] = {
                    "count": 0,
                    "latencies": [],
                    "memory_usage": [],
                    "thermal_events": 0
                }
            
            device_stats[device_type]["count"] += 1
            device_stats[device_type]["latencies"].append(metric.latency_ms)
            device_stats[device_type]["memory_usage"].append(metric.memory_usage_mb)
            
            if metric.thermal_throttled:
                device_stats[device_type]["thermal_events"] += 1
        
        # Calculate summary statistics
        for device_type, stats in device_stats.items():
            latencies = stats["latencies"]
            memory = stats["memory_usage"]
            
            device_stats[device_type] = {
                "inference_count": stats["count"],
                "avg_latency_ms": sum(latencies) / len(latencies),
                "avg_memory_mb": sum(memory) / len(memory),
                "thermal_throttling_rate": stats["thermal_events"] / stats["count"],
                "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0
            }
        
        return device_stats
    
    def stop(self):
        """Stop the telemetry collector."""
        logger.info("Stopping mobile telemetry collector")
        self.running = False
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join(timeout=5)


def main():
    """Main entry point for mobile telemetry collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mobile telemetry collector")
    parser.add_argument("--config", default="mobile-telemetry-config.json",
                       help="Configuration file path")
    parser.add_argument("--prometheus-port", type=int, default=8090,
                       help="Prometheus metrics server port")
    parser.add_argument("--report", action="store_true",
                       help="Generate performance report and exit")
    parser.add_argument("--hours", type=int, default=24,
                       help="Hours of data for report generation")
    
    args = parser.parse_args()
    
    collector = MobileTelemetryCollector(args.config)
    
    if args.report:
        report = collector.generate_performance_report(args.hours)
        print(json.dumps(report, indent=2))
        return
    
    # Start Prometheus server
    collector.start_prometheus_server(args.prometheus_port)
    
    try:
        logger.info("Mobile telemetry collector started. Press Ctrl+C to stop.")
        
        # Example: Simulate some metrics for demonstration
        while True:
            time.sleep(60)  # Keep running
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        collector.stop()


if __name__ == "__main__":
    main()