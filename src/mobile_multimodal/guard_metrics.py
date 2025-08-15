"""Advanced metrics collection and analysis for Pipeline Guard."""

import asyncio
import json
import logging
import sqlite3
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import psutil
import threading


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    component: str
    metric_name: str
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    evaluation_window_seconds: int = 300  # 5 minutes
    consecutive_breaches: int = 3


class MetricsCollector:
    """Collects and stores pipeline metrics."""
    
    def __init__(self, db_path: str = "pipeline_metrics.db"):
        """Initialize metrics collector.
        
        Args:
            db_path: Path to SQLite database for metrics storage
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=10000)  # In-memory buffer
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Start background processing
        self.is_running = False
        self._background_task = None
        
        self.logger.info("Metrics collector initialized")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_component_metric 
                ON metrics(component, metric_name)
            ''')
            
            conn.commit()
        finally:
            conn.close()
    
    def record_metric(self, component: str, metric_name: str, value: float, 
                     labels: Dict[str, str] = None):
        """Record a metric value.
        
        Args:
            component: Component name (e.g., 'training', 'quantization')
            metric_name: Metric name (e.g., 'accuracy', 'latency_ms')
            value: Metric value
            labels: Optional labels for the metric
        """
        metric = MetricPoint(
            timestamp=datetime.now(),
            component=component,
            metric_name=metric_name,
            value=value,
            labels=labels or {}
        )
        
        with self._lock:
            self.metrics_buffer.append(metric)
    
    async def start_collection(self):
        """Start background metrics collection."""
        if self.is_running:
            return
        
        self.is_running = True
        self._background_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop background metrics collection."""
        self.is_running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining metrics
        await self._flush_metrics()
        self.logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Background loop for collecting system metrics."""
        while self.is_running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Flush buffered metrics to database
                await self._flush_metrics()
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system", "cpu_usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system", "memory_usage_percent", memory.percent)
            self.record_metric("system", "memory_available_gb", memory.available / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            self.record_metric("system", "disk_usage_percent", disk_usage_percent)
            self.record_metric("system", "disk_free_gb", disk.free / (1024**3))
            
            # Load average (Unix systems)
            try:
                load_avg = psutil.getloadavg()
                self.record_metric("system", "load_average_1min", load_avg[0])
                self.record_metric("system", "load_average_5min", load_avg[1])
                self.record_metric("system", "load_average_15min", load_avg[2])
            except AttributeError:
                # getloadavg not available on Windows
                pass
            
            # Process count
            process_count = len(psutil.pids())
            self.record_metric("system", "process_count", process_count)
            
            # Network I/O (if available)
            try:
                net_io = psutil.net_io_counters()
                self.record_metric("system", "network_bytes_sent", net_io.bytes_sent)
                self.record_metric("system", "network_bytes_recv", net_io.bytes_recv)
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
    
    async def _flush_metrics(self):
        """Flush buffered metrics to database."""
        if not self.metrics_buffer:
            return
        
        metrics_to_flush = []
        with self._lock:
            metrics_to_flush = list(self.metrics_buffer)
            self.metrics_buffer.clear()
        
        if not metrics_to_flush:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                for metric in metrics_to_flush:
                    cursor.execute('''
                        INSERT INTO metrics (timestamp, component, metric_name, value, labels)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        metric.timestamp.isoformat(),
                        metric.component,
                        metric.metric_name,
                        metric.value,
                        json.dumps(metric.labels)
                    ))
                
                conn.commit()
                self.logger.debug(f"Flushed {len(metrics_to_flush)} metrics to database")
                
            finally:
                conn.close()
                
        except Exception as e:
            self.logger.error(f"Failed to flush metrics to database: {e}")
            # Put metrics back in buffer for retry
            with self._lock:
                self.metrics_buffer.extendleft(reversed(metrics_to_flush))
    
    def get_metrics(self, component: str = None, metric_name: str = None,
                   start_time: datetime = None, end_time: datetime = None,
                   limit: int = 1000) -> List[MetricPoint]:
        """Retrieve metrics from database.
        
        Args:
            component: Filter by component name
            metric_name: Filter by metric name
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of results
            
        Returns:
            List of metric points
        """
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT timestamp, component, metric_name, value, labels FROM metrics WHERE 1=1"
            params = []
            
            if component:
                query += " AND component = ?"
                params.append(component)
            
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            metrics = []
            for row in cursor.fetchall():
                timestamp_str, comp, name, value, labels_str = row
                labels = json.loads(labels_str) if labels_str else {}
                
                metrics.append(MetricPoint(
                    timestamp=datetime.fromisoformat(timestamp_str),
                    component=comp,
                    metric_name=name,
                    value=value,
                    labels=labels
                ))
            
            return metrics
            
        finally:
            conn.close()
    
    def get_aggregated_metrics(self, component: str, metric_name: str,
                              start_time: datetime, end_time: datetime,
                              aggregation: str = "avg") -> Optional[float]:
        """Get aggregated metric value over time period.
        
        Args:
            component: Component name
            metric_name: Metric name
            start_time: Start time
            end_time: End time
            aggregation: Aggregation type (avg, min, max, sum, count)
            
        Returns:
            Aggregated value or None if no data
        """
        metrics = self.get_metrics(
            component=component,
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        
        if aggregation == "avg":
            return statistics.mean(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "count":
            return len(values)
        elif aggregation == "median":
            return statistics.median(values)
        elif aggregation == "stdev":
            return statistics.stdev(values) if len(values) > 1 else 0
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation}")
    
    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Remove old metrics from database.
        
        Args:
            days_to_keep: Number of days of metrics to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            deleted_count = cursor.rowcount
            conn.commit()
            
            self.logger.info(f"Cleaned up {deleted_count} old metric records")
            
        finally:
            conn.close()


class AnomalyDetector:
    """Detects anomalies in pipeline metrics using statistical methods."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize anomaly detector.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = metrics_collector
        self.baselines = {}  # Component -> metric -> baseline stats
        self._lock = threading.RLock()
    
    def calculate_baselines(self, lookback_days: int = 7):
        """Calculate baseline statistics for metrics.
        
        Args:
            lookback_days: Number of days to use for baseline calculation
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Get all unique component/metric combinations
        conn = sqlite3.connect(self.metrics_collector.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT component, metric_name 
                FROM metrics 
                WHERE timestamp >= ? AND timestamp <= ?
            ''', (start_time.isoformat(), end_time.isoformat()))
            
            combinations = cursor.fetchall()
            
        finally:
            conn.close()
        
        # Calculate baselines for each combination
        with self._lock:
            for component, metric_name in combinations:
                try:
                    metrics = self.metrics_collector.get_metrics(
                        component=component,
                        metric_name=metric_name,
                        start_time=start_time,
                        end_time=end_time,
                        limit=10000
                    )
                    
                    if len(metrics) < 10:  # Need minimum data points
                        continue
                    
                    values = [m.value for m in metrics]
                    
                    baseline = {
                        "mean": statistics.mean(values),
                        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values),
                        "median": statistics.median(values),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99),
                        "sample_count": len(values),
                        "calculated_at": datetime.now().isoformat(),
                    }
                    
                    if component not in self.baselines:
                        self.baselines[component] = {}
                    
                    self.baselines[component][metric_name] = baseline
                    
                    self.logger.debug(
                        f"Calculated baseline for {component}.{metric_name}: "
                        f"mean={baseline['mean']:.2f}, stdev={baseline['stdev']:.2f}"
                    )
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to calculate baseline for {component}.{metric_name}: {e}"
                    )
        
        self.logger.info(f"Calculated baselines for {len(self.baselines)} components")
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def detect_anomalies(self, component: str, metric_name: str, 
                        current_value: float) -> Dict[str, Any]:
        """Detect if a metric value is anomalous.
        
        Args:
            component: Component name
            metric_name: Metric name
            current_value: Current metric value
            
        Returns:
            Anomaly detection result
        """
        baseline = self.baselines.get(component, {}).get(metric_name)
        if not baseline:
            return {
                "is_anomaly": False,
                "reason": "No baseline available",
                "severity": "unknown"
            }
        
        mean = baseline["mean"]
        stdev = baseline["stdev"]
        
        # Z-score based anomaly detection
        if stdev > 0:
            z_score = abs(current_value - mean) / stdev
        else:
            z_score = 0
        
        # Classify anomaly severity
        if z_score > 3.0:  # 3 standard deviations
            severity = "critical"
            is_anomaly = True
        elif z_score > 2.0:  # 2 standard deviations
            severity = "warning"
            is_anomaly = True
        else:
            severity = "normal"
            is_anomaly = False
        
        # Additional checks for extreme values
        if current_value > baseline["p99"]:
            if not is_anomaly:
                severity = "warning"
                is_anomaly = True
        
        return {
            "is_anomaly": is_anomaly,
            "severity": severity,
            "z_score": z_score,
            "baseline_mean": mean,
            "baseline_stdev": stdev,
            "current_value": current_value,
            "percentile_99": baseline["p99"],
            "reason": f"Z-score: {z_score:.2f}, threshold: 2.0/3.0"
        }
    
    def save_baselines(self, file_path: str = "anomaly_baselines.json"):
        """Save baselines to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.baselines, f, indent=2)
            self.logger.info(f"Baselines saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save baselines: {e}")
    
    def load_baselines(self, file_path: str = "anomaly_baselines.json"):
        """Load baselines from file."""
        try:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    self.baselines = json.load(f)
                self.logger.info(f"Baselines loaded from {file_path}")
            else:
                self.logger.warning(f"Baseline file {file_path} not found")
        except Exception as e:
            self.logger.error(f"Failed to load baselines: {e}")


class AlertManager:
    """Manages alerts based on metric thresholds and anomalies."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = metrics_collector
        self.thresholds: Dict[str, AlertThreshold] = {}
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []
        self._lock = threading.RLock()
        
        # Initialize default thresholds
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        default_thresholds = [
            AlertThreshold("cpu_usage_percent", 80.0, 95.0),
            AlertThreshold("memory_usage_percent", 85.0, 95.0),
            AlertThreshold("disk_usage_percent", 85.0, 95.0),
            AlertThreshold("load_average_1min", 5.0, 10.0),
            AlertThreshold("training_accuracy_drop", 5.0, 10.0),
            AlertThreshold("quantization_accuracy_drop", 3.0, 8.0),
            AlertThreshold("inference_latency_ms", 100.0, 200.0),
            AlertThreshold("export_failure_rate", 0.05, 0.1),  # 5% warning, 10% critical
        ]
        
        for threshold in default_thresholds:
            self.add_threshold(threshold)
    
    def add_threshold(self, threshold: AlertThreshold):
        """Add alert threshold.
        
        Args:
            threshold: Alert threshold configuration
        """
        with self._lock:
            self.thresholds[threshold.metric_name] = threshold
            self.logger.debug(f"Added threshold for {threshold.metric_name}")
    
    def remove_threshold(self, metric_name: str):
        """Remove alert threshold.
        
        Args:
            metric_name: Metric name to remove threshold for
        """
        with self._lock:
            if metric_name in self.thresholds:
                del self.thresholds[metric_name]
                self.logger.debug(f"Removed threshold for {metric_name}")
    
    def evaluate_thresholds(self) -> List[Dict]:
        """Evaluate all thresholds against recent metrics.
        
        Returns:
            List of new alerts generated
        """
        new_alerts = []
        end_time = datetime.now()
        
        with self._lock:
            for metric_name, threshold in self.thresholds.items():
                start_time = end_time - timedelta(seconds=threshold.evaluation_window_seconds)
                
                # Get recent metrics for this metric
                metrics = self.metrics_collector.get_metrics(
                    metric_name=metric_name,
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000
                )
                
                if not metrics:
                    continue
                
                # Group by component
                component_metrics = defaultdict(list)
                for metric in metrics:
                    component_metrics[metric.component].append(metric.value)
                
                # Evaluate each component
                for component, values in component_metrics.items():
                    if len(values) < threshold.consecutive_breaches:
                        continue
                    
                    # Check recent values for threshold breaches
                    recent_values = values[-threshold.consecutive_breaches:]
                    
                    critical_breaches = sum(1 for v in recent_values if v >= threshold.critical_threshold)
                    warning_breaches = sum(1 for v in recent_values if v >= threshold.warning_threshold)
                    
                    alert_key = f"{component}.{metric_name}"
                    
                    if critical_breaches >= threshold.consecutive_breaches:
                        alert = self._create_alert(
                            component, metric_name, "critical",
                            f"Critical threshold breached: {recent_values[-1]:.2f} >= {threshold.critical_threshold}",
                            {"threshold": threshold.critical_threshold, "current_value": recent_values[-1]}
                        )
                        new_alerts.append(alert)
                        self.active_alerts[alert_key] = alert
                        
                    elif warning_breaches >= threshold.consecutive_breaches:
                        alert = self._create_alert(
                            component, metric_name, "warning",
                            f"Warning threshold breached: {recent_values[-1]:.2f} >= {threshold.warning_threshold}",
                            {"threshold": threshold.warning_threshold, "current_value": recent_values[-1]}
                        )
                        new_alerts.append(alert)
                        self.active_alerts[alert_key] = alert
                        
                    else:
                        # Clear active alert if exists
                        if alert_key in self.active_alerts:
                            resolved_alert = self.active_alerts[alert_key].copy()
                            resolved_alert["resolved_at"] = end_time.isoformat()
                            resolved_alert["status"] = "resolved"
                            self.alert_history.append(resolved_alert)
                            del self.active_alerts[alert_key]
        
        return new_alerts
    
    def _create_alert(self, component: str, metric_name: str, severity: str,
                     message: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create alert dictionary.
        
        Args:
            component: Component name
            metric_name: Metric name
            severity: Alert severity
            message: Alert message
            metadata: Additional metadata
            
        Returns:
            Alert dictionary
        """
        alert = {
            "id": f"{component}.{metric_name}.{int(time.time())}",
            "component": component,
            "metric_name": metric_name,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "metadata": metadata,
        }
        
        self.alert_history.append(alert)
        self.logger.warning(f"Alert generated: {alert['id']} - {message}")
        
        return alert
    
    def get_active_alerts(self) -> List[Dict]:
        """Get list of active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alerts
        """
        with self._lock:
            return self.alert_history[-limit:]
    
    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
        """
        with self._lock:
            # Find and resolve in active alerts
            for key, alert in list(self.active_alerts.items()):
                if alert["id"] == alert_id:
                    alert["resolved_at"] = datetime.now().isoformat()
                    alert["status"] = "resolved"
                    alert["resolution_method"] = "manual"
                    del self.active_alerts[key]
                    self.logger.info(f"Manually resolved alert: {alert_id}")
                    break


def main():
    """CLI for metrics collection and analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Guard Metrics")
    parser.add_argument("--collect", action="store_true", help="Start metrics collection")
    parser.add_argument("--analyze", help="Analyze metrics for component")
    parser.add_argument("--baselines", action="store_true", help="Calculate anomaly baselines")
    parser.add_argument("--alerts", action="store_true", help="Show active alerts")
    parser.add_argument("--db", default="pipeline_metrics.db", help="Database path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    collector = MetricsCollector(args.db)
    
    if args.collect:
        async def run_collection():
            await collector.start_collection()
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await collector.stop_collection()
        
        asyncio.run(run_collection())
    
    elif args.analyze:
        # Show recent metrics for component
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        metrics = collector.get_metrics(
            component=args.analyze,
            start_time=start_time,
            end_time=end_time
        )
        
        print(f"Recent metrics for {args.analyze}:")
        for metric in metrics[-10:]:  # Last 10 metrics
            print(f"  {metric.timestamp}: {metric.metric_name} = {metric.value}")
    
    elif args.baselines:
        detector = AnomalyDetector(collector)
        detector.calculate_baselines()
        detector.save_baselines()
        print("Baselines calculated and saved")
    
    elif args.alerts:
        alert_manager = AlertManager(collector)
        alerts = alert_manager.evaluate_thresholds()
        
        print(f"Found {len(alerts)} new alerts:")
        for alert in alerts:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
        
        active_alerts = alert_manager.get_active_alerts()
        print(f"\nActive alerts: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  {alert['id']}: {alert['message']}")
    
    else:
        print("No action specified. Use --help for options.")


if __name__ == "__main__":
    main()