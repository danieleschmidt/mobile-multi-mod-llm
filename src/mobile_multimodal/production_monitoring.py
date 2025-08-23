"""Production-Grade Monitoring and Observability for Mobile Multi-Modal AI.

Advanced monitoring, alerting, and observability system for production deployments
with real-time metrics, distributed tracing, and automated incident response.
"""

import json
import logging
import os
import time
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid
import hashlib
from datetime import datetime, timedelta
import asyncio

# Observability and monitoring libraries
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry, REGISTRY
    from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
except ImportError:
    # Mock Prometheus client for environments without it
    class Counter:
        def __init__(self, *args, **kwargs):
            self._value = 0
        def inc(self, value=1):
            self._value += value
        def get(self):
            return self._value
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            self._observations = []
        def observe(self, value):
            self._observations.append(value)
        def time(self):
            return HistogramTimer(self)
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            self._value = 0
        def set(self, value):
            self._value = value
        def inc(self, value=1):
            self._value += value
        def dec(self, value=1):
            self._value -= value
    
    def start_http_server(port):
        pass
    
    class CollectorRegistry:
        pass
    
    REGISTRY = CollectorRegistry()

class HistogramTimer:
    """Timer context for histograms."""
    def __init__(self, histogram):
        self.histogram = histogram
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if self.start_time:
            duration = time.time() - self.start_time
            self.histogram.observe(duration)

logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    """Configuration for metrics collection."""
    name: str
    type: str  # counter, gauge, histogram
    description: str
    labels: List[str] = None
    buckets: List[float] = None  # For histograms
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = []
        if self.buckets is None and self.type == "histogram":
            self.buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]

@dataclass  
class AlertRule:
    """Configuration for alerting rules."""
    name: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "not_equals"
    threshold: float
    duration: int  # seconds
    severity: str  # "critical", "warning", "info"
    description: str
    runbook_url: Optional[str] = None

@dataclass
class TraceSpan:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []

class ProductionMonitoring:
    """Production-grade monitoring and observability system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize production monitoring system.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or {}
        self.metrics_port = self.config.get("metrics_port", 8000)
        self.alert_webhook_url = self.config.get("alert_webhook_url")
        self.trace_sampling_rate = self.config.get("trace_sampling_rate", 0.1)
        
        # Initialize metrics registry
        self.metrics_registry = CollectorRegistry()
        self.custom_metrics = {}
        
        # Initialize core metrics
        self._init_core_metrics()
        
        # Alert management
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Distributed tracing
        self.traces = {}
        self.span_buffer = deque(maxlen=10000)
        
        # Health check system
        self.health_checks = {}
        self.health_status = "healthy"
        self.last_health_check = time.time()
        
        # Performance tracking
        self.performance_baselines = {}
        self.anomaly_detector = AnomalyDetector()
        
        # Thread management
        self._monitoring_thread = None
        self._alert_thread = None
        self._health_thread = None
        self._shutdown_event = threading.Event()
        
        # Initialize async monitoring
        self._async_loop = None
        self._async_tasks = []
        
        logger.info("Production monitoring system initialized")
    
    def _init_core_metrics(self):
        """Initialize core application metrics."""
        # Request metrics
        self.request_count = Counter(
            'mobile_multimodal_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.metrics_registry
        )
        
        self.request_duration = Histogram(
            'mobile_multimodal_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.metrics_registry,
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Model inference metrics
        self.inference_count = Counter(
            'mobile_multimodal_inferences_total',
            'Total number of model inferences',
            ['model_name', 'task', 'status'],
            registry=self.metrics_registry
        )
        
        self.inference_duration = Histogram(
            'mobile_multimodal_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_name', 'task'],
            registry=self.metrics_registry
        )
        
        self.inference_accuracy = Histogram(
            'mobile_multimodal_inference_accuracy',
            'Model inference accuracy scores',
            ['model_name', 'task'],
            registry=self.metrics_registry,
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'mobile_multimodal_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.metrics_registry
        )
        
        self.cpu_usage = Gauge(
            'mobile_multimodal_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.metrics_registry
        )
        
        self.gpu_usage = Gauge(
            'mobile_multimodal_gpu_usage_percent',
            'Current GPU usage percentage',
            ['gpu_id'],
            registry=self.metrics_registry
        )
        
        self.model_load_time = Histogram(
            'mobile_multimodal_model_load_duration_seconds',
            'Time taken to load models',
            ['model_name', 'format'],
            registry=self.metrics_registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'mobile_multimodal_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.metrics_registry
        )
        
        # Business metrics
        self.active_models = Gauge(
            'mobile_multimodal_active_models',
            'Number of currently active models',
            registry=self.metrics_registry
        )
        
        self.concurrent_requests = Gauge(
            'mobile_multimodal_concurrent_requests',
            'Number of concurrent requests being processed',
            registry=self.metrics_registry
        )
    
    def start_monitoring(self):
        """Start monitoring services."""
        try:
            # Start Prometheus metrics server
            start_http_server(self.metrics_port, registry=self.metrics_registry)
            logger.info(f"Metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
        
        # Start monitoring threads
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._alert_thread = threading.Thread(target=self._alert_loop, daemon=True) 
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        
        self._monitoring_thread.start()
        self._alert_thread.start()
        self._health_thread.start()
        
        # Start async monitoring loop
        self._start_async_monitoring()
        
        logger.info("All monitoring services started")
    
    def stop_monitoring(self):
        """Stop monitoring services."""
        self._shutdown_event.set()
        
        # Stop async tasks
        for task in self._async_tasks:
            task.cancel()
        
        if self._async_loop:
            self._async_loop.stop()
        
        logger.info("Monitoring services stopped")
    
    def _start_async_monitoring(self):
        """Start async monitoring loop in separate thread."""
        def run_async_loop():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            
            # Start async monitoring tasks
            self._async_tasks.append(
                self._async_loop.create_task(self._async_system_monitoring())
            )
            self._async_tasks.append(
                self._async_loop.create_task(self._async_trace_processing())
            )
            
            try:
                self._async_loop.run_forever()
            except Exception as e:
                logger.error(f"Async monitoring loop failed: {e}")
        
        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()
    
    def register_metric(self, metric_config: MetricConfig):
        """Register a custom metric.
        
        Args:
            metric_config: Metric configuration
        """
        if metric_config.type == "counter":
            metric = Counter(
                metric_config.name,
                metric_config.description,
                metric_config.labels,
                registry=self.metrics_registry
            )
        elif metric_config.type == "gauge":
            metric = Gauge(
                metric_config.name,
                metric_config.description,
                metric_config.labels,
                registry=self.metrics_registry
            )
        elif metric_config.type == "histogram":
            metric = Histogram(
                metric_config.name,
                metric_config.description,
                metric_config.labels,
                buckets=metric_config.buckets,
                registry=self.metrics_registry
            )
        else:
            raise ValueError(f"Unsupported metric type: {metric_config.type}")
        
        self.custom_metrics[metric_config.name] = metric
        logger.info(f"Registered custom metric: {metric_config.name}")
    
    def register_alert_rule(self, alert_rule: AlertRule):
        """Register an alert rule.
        
        Args:
            alert_rule: Alert rule configuration
        """
        self.alert_rules[alert_rule.name] = alert_rule
        logger.info(f"Registered alert rule: {alert_rule.name}")
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record request metrics.
        
        Args:
            method: HTTP method
            endpoint: Request endpoint
            status: HTTP status code
            duration: Request duration in seconds
        """
        self.request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_inference(self, model_name: str, task: str, duration: float, 
                        accuracy: Optional[float] = None, success: bool = True):
        """Record model inference metrics.
        
        Args:
            model_name: Name of the model
            task: Type of task (captioning, ocr, vqa, etc.)
            duration: Inference duration in seconds
            accuracy: Optional accuracy score
            success: Whether inference was successful
        """
        status = "success" if success else "error"
        self.inference_count.labels(model_name=model_name, task=task, status=status).inc()
        self.inference_duration.labels(model_name=model_name, task=task).observe(duration)
        
        if accuracy is not None:
            self.inference_accuracy.labels(model_name=model_name, task=task).observe(accuracy)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence.
        
        Args:
            error_type: Type of error
            component: Component where error occurred
        """
        self.error_count.labels(error_type=error_type, component=component).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.set(cpu_percent)
            
        except ImportError:
            # Fallback to basic system info
            import os
            try:
                # Basic memory info on Linux
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemAvailable:'):
                        mem_available = int(line.split()[1]) * 1024  # Convert KB to bytes
                        # Rough estimate of memory usage
                        total_mem = 8 * 1024 * 1024 * 1024  # Assume 8GB total
                        used_mem = total_mem - mem_available
                        self.memory_usage.set(used_mem)
                        break
            except:
                pass
    
    def start_trace(self, operation_name: str, parent_span_id: Optional[str] = None) -> TraceSpan:
        """Start a new distributed trace span.
        
        Args:
            operation_name: Name of the operation being traced
            parent_span_id: Optional parent span ID
            
        Returns:
            New trace span
        """
        # Generate trace and span IDs
        if parent_span_id:
            # Find parent trace
            parent_trace_id = None
            for trace_id, spans in self.traces.items():
                if parent_span_id in [s.span_id for s in spans]:
                    parent_trace_id = trace_id
                    break
            trace_id = parent_trace_id or str(uuid.uuid4())
        else:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        # Store span
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        self.traces[trace_id].append(span)
        
        return span
    
    def finish_trace(self, span: TraceSpan, tags: Optional[Dict[str, Any]] = None,
                    logs: Optional[List[Dict[str, Any]]] = None):
        """Finish a trace span.
        
        Args:
            span: Span to finish
            tags: Optional tags to add
            logs: Optional log entries to add
        """
        span.end_time = time.time()
        
        if tags:
            span.tags.update(tags)
        
        if logs:
            span.logs.extend(logs)
        
        # Add to span buffer for processing
        self.span_buffer.append(span)
    
    def trace_context(self, operation_name: str, parent_span_id: Optional[str] = None):
        """Context manager for distributed tracing.
        
        Args:
            operation_name: Name of the operation
            parent_span_id: Optional parent span ID
            
        Returns:
            Context manager that handles span lifecycle
        """
        return TraceContext(self, operation_name, parent_span_id)
    
    def register_health_check(self, name: str, check_fn: Callable[[], bool], 
                            interval: int = 30):
        """Register a health check.
        
        Args:
            name: Health check name
            check_fn: Function that returns True if healthy
            interval: Check interval in seconds
        """
        self.health_checks[name] = {
            "function": check_fn,
            "interval": interval,
            "last_check": 0,
            "status": "unknown"
        }
        logger.info(f"Registered health check: {name}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status.
        
        Returns:
            Health status information
        """
        return {
            "status": self.health_status,
            "timestamp": self.last_health_check,
            "checks": {name: check["status"] for name, check in self.health_checks.items()}
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Update system metrics
                self.update_system_metrics()
                
                # Clean up old traces
                self._cleanup_traces()
                
                # Process performance baselines
                self._update_performance_baselines()
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _alert_loop(self):
        """Alert processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Check alert rules
                for rule_name, rule in self.alert_rules.items():
                    self._evaluate_alert_rule(rule)
                
                # Clean up old alerts
                self._cleanup_alerts()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Alert loop error: {e}")
                time.sleep(5)
    
    def _health_loop(self):
        """Health check loop."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                overall_healthy = True
                
                # Run health checks
                for name, check_info in self.health_checks.items():
                    if current_time - check_info["last_check"] >= check_info["interval"]:
                        try:
                            is_healthy = check_info["function"]()
                            check_info["status"] = "healthy" if is_healthy else "unhealthy"
                            check_info["last_check"] = current_time
                            
                            if not is_healthy:
                                overall_healthy = False
                                
                        except Exception as e:
                            logger.error(f"Health check {name} failed: {e}")
                            check_info["status"] = "error"
                            overall_healthy = False
                
                # Update overall health status
                if overall_healthy:
                    self.health_status = "healthy"
                else:
                    self.health_status = "unhealthy"
                
                self.last_health_check = current_time
                
                time.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(10)
    
    async def _async_system_monitoring(self):
        """Async system monitoring tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Advanced system monitoring
                await self._monitor_gpu_usage()
                await self._monitor_network_io()
                await self._monitor_disk_usage()
                
                await asyncio.sleep(20)
                
            except Exception as e:
                logger.error(f"Async system monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _async_trace_processing(self):
        """Async trace processing."""
        while not self._shutdown_event.is_set():
            try:
                # Process span buffer
                spans_to_process = []
                while len(spans_to_process) < 100 and self.span_buffer:
                    spans_to_process.append(self.span_buffer.popleft())
                
                if spans_to_process:
                    await self._process_trace_spans(spans_to_process)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Async trace processing error: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_gpu_usage(self):
        """Monitor GPU usage asynchronously."""
        try:
            # Try to get GPU usage with nvidia-smi
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                gpu_usages = result.stdout.strip().split('\n')
                for i, usage in enumerate(gpu_usages):
                    try:
                        usage_percent = float(usage)
                        self.gpu_usage.labels(gpu_id=str(i)).set(usage_percent)
                    except ValueError:
                        pass
                        
        except Exception as e:
            # GPU monitoring not available
            pass
    
    async def _monitor_network_io(self):
        """Monitor network I/O asynchronously."""
        try:
            import psutil
            net_io = psutil.net_io_counters()
            
            # Store network stats for rate calculation
            if hasattr(self, '_last_net_io'):
                bytes_sent_rate = net_io.bytes_sent - self._last_net_io.bytes_sent
                bytes_recv_rate = net_io.bytes_recv - self._last_net_io.bytes_recv
                
                # Could add network rate metrics here
            
            self._last_net_io = net_io
            
        except Exception:
            pass
    
    async def _monitor_disk_usage(self):
        """Monitor disk usage asynchronously."""
        try:
            import psutil
            disk_usage = psutil.disk_usage('/')
            
            # Could add disk usage metrics here
            
        except Exception:
            pass
    
    async def _process_trace_spans(self, spans: List[TraceSpan]):
        """Process trace spans for analysis."""
        for span in spans:
            # Analyze span for performance anomalies
            if span.end_time and span.start_time:
                duration = span.end_time - span.start_time
                
                # Check against performance baselines
                if span.operation_name in self.performance_baselines:
                    baseline = self.performance_baselines[span.operation_name]
                    if duration > baseline["p95"] * 2:  # Significantly slower than P95
                        logger.warning(f"Slow trace detected: {span.operation_name} took {duration:.3f}s")
    
    def _evaluate_alert_rule(self, rule: AlertRule):
        """Evaluate an alert rule."""
        # Get current metric value
        metric_value = self._get_metric_value(rule.metric_name)
        
        if metric_value is None:
            return
        
        # Evaluate condition
        triggered = False
        if rule.condition == "greater_than" and metric_value > rule.threshold:
            triggered = True
        elif rule.condition == "less_than" and metric_value < rule.threshold:
            triggered = True
        elif rule.condition == "equals" and metric_value == rule.threshold:
            triggered = True
        elif rule.condition == "not_equals" and metric_value != rule.threshold:
            triggered = True
        
        current_time = time.time()
        
        if triggered:
            # Check if alert should fire (duration check)
            if rule.name not in self.active_alerts:
                self.active_alerts[rule.name] = {
                    "first_triggered": current_time,
                    "last_triggered": current_time,
                    "fired": False
                }
            else:
                self.active_alerts[rule.name]["last_triggered"] = current_time
                
                # Check if duration threshold is met
                if (current_time - self.active_alerts[rule.name]["first_triggered"] >= rule.duration and
                    not self.active_alerts[rule.name]["fired"]):
                    self._fire_alert(rule, metric_value)
                    self.active_alerts[rule.name]["fired"] = True
        else:
            # Clear active alert if condition no longer met
            if rule.name in self.active_alerts:
                self._resolve_alert(rule)
                del self.active_alerts[rule.name]
    
    def _fire_alert(self, rule: AlertRule, current_value: float):
        """Fire an alert."""
        alert = {
            "rule_name": rule.name,
            "severity": rule.severity,
            "description": rule.description,
            "metric_name": rule.metric_name,
            "current_value": current_value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "timestamp": time.time(),
            "runbook_url": rule.runbook_url
        }
        
        # Add to alert history
        self.alert_history.append(alert)
        
        # Send alert notification
        self._send_alert_notification(alert)
        
        logger.warning(f"Alert fired: {rule.name} - {rule.description}")
    
    def _resolve_alert(self, rule: AlertRule):
        """Resolve an alert."""
        resolve_alert = {
            "rule_name": rule.name,
            "severity": "resolved",
            "description": f"Alert resolved: {rule.description}",
            "timestamp": time.time()
        }
        
        self.alert_history.append(resolve_alert)
        logger.info(f"Alert resolved: {rule.name}")
    
    def _send_alert_notification(self, alert: Dict[str, Any]):
        """Send alert notification."""
        if self.alert_webhook_url:
            try:
                import requests
                response = requests.post(self.alert_webhook_url, json=alert, timeout=10)
                logger.info(f"Alert notification sent: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric."""
        try:
            # Map metric names to actual metrics
            metric_map = {
                "request_rate": lambda: self.request_count._value._value if hasattr(self.request_count, '_value') else 0,
                "error_rate": lambda: self.error_count._value._value if hasattr(self.error_count, '_value') else 0,
                "memory_usage": lambda: self.memory_usage._value._value if hasattr(self.memory_usage, '_value') else 0,
                "cpu_usage": lambda: self.cpu_usage._value._value if hasattr(self.cpu_usage, '_value') else 0
            }
            
            if metric_name in metric_map:
                return metric_map[metric_name]()
            
        except Exception as e:
            logger.error(f"Failed to get metric value for {metric_name}: {e}")
        
        return None
    
    def _cleanup_traces(self):
        """Clean up old traces."""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep traces for 1 hour
        
        traces_to_remove = []
        for trace_id, spans in self.traces.items():
            # Remove traces older than cutoff
            if all(span.start_time < cutoff_time for span in spans):
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.traces[trace_id]
    
    def _cleanup_alerts(self):
        """Clean up old active alerts."""
        current_time = time.time()
        alerts_to_remove = []
        
        for rule_name, alert_info in self.active_alerts.items():
            # Remove alerts that haven't been triggered recently
            if current_time - alert_info["last_triggered"] > 300:  # 5 minutes
                alerts_to_remove.append(rule_name)
        
        for rule_name in alerts_to_remove:
            del self.active_alerts[rule_name]
    
    def _update_performance_baselines(self):
        """Update performance baselines from historical data."""
        # This would analyze historical metrics to establish performance baselines
        # For now, implement basic placeholder
        pass
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        try:
            from prometheus_client import generate_latest
            return generate_latest(self.metrics_registry).decode('utf-8')
        except ImportError:
            return "# Prometheus client not available\n"
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace summary by trace ID.
        
        Args:
            trace_id: Trace identifier
            
        Returns:
            Trace summary or None if not found
        """
        if trace_id not in self.traces:
            return None
        
        spans = self.traces[trace_id]
        
        # Calculate trace summary
        start_time = min(span.start_time for span in spans)
        end_times = [span.end_time for span in spans if span.end_time]
        end_time = max(end_times) if end_times else None
        
        summary = {
            "trace_id": trace_id,
            "span_count": len(spans),
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time if end_time else None,
            "operations": [span.operation_name for span in spans],
            "status": "completed" if all(span.end_time for span in spans) else "in_progress"
        }
        
        return summary
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.
        
        Returns:
            Metrics summary
        """
        summary = {
            "timestamp": time.time(),
            "total_requests": getattr(self.request_count, '_value', {}).get('_value', 0) if hasattr(self.request_count, '_value') else 0,
            "total_inferences": getattr(self.inference_count, '_value', {}).get('_value', 0) if hasattr(self.inference_count, '_value') else 0,
            "total_errors": getattr(self.error_count, '_value', {}).get('_value', 0) if hasattr(self.error_count, '_value') else 0,
            "current_memory_usage": getattr(self.memory_usage, '_value', {}).get('_value', 0) if hasattr(self.memory_usage, '_value') else 0,
            "current_cpu_usage": getattr(self.cpu_usage, '_value', {}).get('_value', 0) if hasattr(self.cpu_usage, '_value') else 0,
            "health_status": self.health_status,
            "active_alerts": len(self.active_alerts),
            "active_traces": len(self.traces)
        }
        
        return summary


class TraceContext:
    """Context manager for distributed tracing."""
    
    def __init__(self, monitoring: ProductionMonitoring, operation_name: str, 
                 parent_span_id: Optional[str] = None):
        """Initialize trace context.
        
        Args:
            monitoring: Monitoring system
            operation_name: Operation being traced
            parent_span_id: Optional parent span ID
        """
        self.monitoring = monitoring
        self.operation_name = operation_name
        self.parent_span_id = parent_span_id
        self.span = None
    
    def __enter__(self) -> TraceSpan:
        """Start trace span."""
        self.span = self.monitoring.start_trace(self.operation_name, self.parent_span_id)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish trace span."""
        if self.span:
            tags = {}
            logs = []
            
            if exc_type:
                tags["error"] = True
                tags["error_type"] = exc_type.__name__
                logs.append({
                    "timestamp": time.time(),
                    "level": "error",
                    "message": str(exc_val)
                })
            
            self.monitoring.finish_trace(self.span, tags, logs)


class AnomalyDetector:
    """Simple anomaly detector for performance metrics."""
    
    def __init__(self, window_size: int = 100):
        """Initialize anomaly detector.
        
        Args:
            window_size: Size of the sliding window for analysis
        """
        self.window_size = window_size
        self.metric_windows = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_metric(self, metric_name: str, value: float):
        """Add a metric value for analysis.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.metric_windows[metric_name].append(value)
    
    def is_anomalous(self, metric_name: str, value: float, threshold: float = 2.0) -> bool:
        """Check if a value is anomalous.
        
        Args:
            metric_name: Name of the metric
            value: Value to check
            threshold: Number of standard deviations for anomaly threshold
            
        Returns:
            True if the value is anomalous
        """
        window = self.metric_windows[metric_name]
        
        if len(window) < 10:  # Need minimum data points
            return False
        
        import statistics
        mean = statistics.mean(window)
        stdev = statistics.stdev(window)
        
        if stdev == 0:
            return False
        
        z_score = abs(value - mean) / stdev
        return z_score > threshold


class MetricsCollector:
    """Custom metrics collector for Prometheus."""
    
    def __init__(self, monitoring: ProductionMonitoring):
        """Initialize metrics collector.
        
        Args:
            monitoring: Production monitoring system
        """
        self.monitoring = monitoring
    
    def collect(self):
        """Collect custom metrics."""
        # This would implement custom metric collection
        # For now, return empty list
        return []


def create_default_monitoring_config() -> Dict[str, Any]:
    """Create default monitoring configuration."""
    return {
        "metrics_port": 8000,
        "trace_sampling_rate": 0.1,
        "alert_rules": [
            {
                "name": "high_error_rate",
                "metric_name": "error_rate",
                "condition": "greater_than",
                "threshold": 0.05,  # 5% error rate
                "duration": 60,  # 1 minute
                "severity": "warning",
                "description": "High error rate detected"
            },
            {
                "name": "high_memory_usage",
                "metric_name": "memory_usage", 
                "condition": "greater_than",
                "threshold": 1024 * 1024 * 1024,  # 1GB
                "duration": 300,  # 5 minutes
                "severity": "warning",
                "description": "High memory usage detected"
            },
            {
                "name": "high_cpu_usage",
                "metric_name": "cpu_usage",
                "condition": "greater_than",
                "threshold": 80.0,  # 80% CPU
                "duration": 180,  # 3 minutes
                "severity": "warning",
                "description": "High CPU usage detected"
            }
        ],
        "health_checks": [
            {
                "name": "model_health",
                "interval": 30
            },
            {
                "name": "memory_health",
                "interval": 60
            }
        ]
    }