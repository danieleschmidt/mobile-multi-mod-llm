"""Enhanced monitoring capabilities for mobile multi-modal models."""

import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class AdvancedMetric:
    """Advanced metric with additional metadata."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]
    metadata: Dict[str, Any]

class AdvancedMetrics:
    """Advanced metrics collection and analysis."""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.metrics = deque(maxlen=buffer_size)
        self.aggregates = defaultdict(list)
        self._lock = threading.Lock()
        logger.info(f"AdvancedMetrics initialized with buffer_size={buffer_size}")
    
    def record_advanced_metric(self, name: str, value: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record an advanced metric with metadata."""
        if tags is None:
            tags = {}
        if metadata is None:
            metadata = {}
        
        metric = AdvancedMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags,
            metadata=metadata
        )
        
        with self._lock:
            self.metrics.append(metric)
            self.aggregates[name].append(value)
    
    def get_metric_summary(self, name: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive summary of a metric."""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        with self._lock:
            recent_metrics = [m for m in self.metrics if m.name == name and m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"name": name, "count": 0, "values": []}
        
        values = [m.value for m in recent_metrics]
        return {
            "name": name,
            "count": len(values),
            "values": values,
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "recent_timestamp": recent_metrics[-1].timestamp
        }
    
    def get_all_metrics(self, time_window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get all metrics within time window."""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        with self._lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        return [{
            "name": m.name,
            "value": m.value,
            "timestamp": m.timestamp,
            "tags": m.tags,
            "metadata": m.metadata
        } for m in recent_metrics]

class AlertManager:
    """Advanced alerting and notification system."""
    
    def __init__(self):
        self.alert_rules = []
        self.alert_history = []
        self.alert_callbacks = []
        logger.info("AlertManager initialized")
    
    def add_alert_rule(self, name: str, condition: callable, severity: str = "warning", threshold: float = 1.0):
        """Add an alert rule."""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "threshold": threshold,
            "triggered_count": 0,
            "last_triggered": None
        }
        self.alert_rules.append(rule)
        logger.info(f"Alert rule added: {name}")
    
    def check_alerts(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check all alert rules against current metrics."""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    alert = {
                        "rule_name": rule["name"],
                        "severity": rule["severity"],
                        "timestamp": time.time(),
                        "message": f"Alert triggered: {rule['name']}"
                    }
                    triggered_alerts.append(alert)
                    self.alert_history.append(alert)
                    rule["triggered_count"] += 1
                    rule["last_triggered"] = time.time()
                    
                    # Execute callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")
                            
            except Exception as e:
                logger.error(f"Alert rule evaluation failed for {rule['name']}: {e}")
        
        return triggered_alerts
    
    def add_alert_callback(self, callback: callable):
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)
        logger.info("Alert callback added")
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert["timestamp"] >= cutoff_time]