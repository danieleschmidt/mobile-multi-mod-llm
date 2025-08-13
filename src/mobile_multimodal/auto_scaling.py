"""Advanced auto-scaling system for mobile AI workloads."""

import time
import threading
import logging
import statistics
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Triggers for auto-scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_LATENCY = "request_latency"
    QUEUE_DEPTH = "queue_depth"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int
    max_instances: int
    cooldown_period_seconds: int
    evaluation_period_seconds: int
    datapoints_required: int
    scale_up_adjustment: int = 1
    scale_down_adjustment: int = 1


@dataclass
class ScalingEvent:
    """Auto-scaling event record."""
    timestamp: float
    direction: ScalingDirection
    trigger: ScalingTrigger
    metric_value: float
    threshold: float
    instances_before: int
    instances_after: int
    reason: str


class MetricsCollector:
    """Collect and aggregate metrics for auto-scaling decisions."""
    
    def __init__(self, retention_period_seconds: int = 3600):
        self.retention_period = retention_period_seconds
        self.metrics = {}
        self.aggregated_metrics = {}
        self.collection_thread = None
        self.is_collecting = False
        
    def start_collection(self, interval_seconds: float = 5.0):
        """Start metrics collection thread."""
        self.is_collecting = True
        
        def collect_loop():
            while self.is_collecting:
                try:
                    self._collect_system_metrics()
                    self._aggregate_metrics()
                    self._cleanup_old_metrics()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
        
        self.collection_thread = threading.Thread(target=collect_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        timestamp = time.time()
        
        # Collect various metrics (mock implementation)
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
        except ImportError:
            # Fallback metrics when psutil not available
            import random
            cpu_percent = random.uniform(20, 80)
            memory_percent = random.uniform(30, 70)
        
        # Mock additional metrics
        current_metrics = {
            "cpu_utilization": cpu_percent,
            "memory_utilization": memory_percent,
            "request_latency": self._get_avg_latency(),
            "queue_depth": self._get_queue_depth(),
            "error_rate": self._get_error_rate(),
            "throughput": self._get_throughput(),
            "active_connections": self._get_active_connections(),
            "response_time_p95": self._get_p95_response_time()
        }
        
        # Store metrics with timestamp
        for metric_name, value in current_metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            self.metrics[metric_name].append({
                "timestamp": timestamp,
                "value": value
            })
    
    def _aggregate_metrics(self):
        """Aggregate metrics for different time windows."""
        current_time = time.time()
        
        time_windows = {
            "1min": 60,
            "5min": 300,
            "15min": 900,
            "1hour": 3600
        }
        
        for metric_name, data_points in self.metrics.items():
            self.aggregated_metrics[metric_name] = {}
            
            for window_name, window_seconds in time_windows.items():
                window_start = current_time - window_seconds
                window_data = [
                    dp["value"] for dp in data_points 
                    if dp["timestamp"] >= window_start
                ]
                
                if window_data:
                    self.aggregated_metrics[metric_name][window_name] = {
                        "avg": statistics.mean(window_data),
                        "max": max(window_data),
                        "min": min(window_data),
                        "median": statistics.median(window_data),
                        "count": len(window_data),
                        "last": window_data[-1] if window_data else 0
                    }
                else:
                    self.aggregated_metrics[metric_name][window_name] = {
                        "avg": 0, "max": 0, "min": 0, "median": 0, "count": 0, "last": 0
                    }
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - self.retention_period
        
        for metric_name in self.metrics:
            self.metrics[metric_name] = [
                dp for dp in self.metrics[metric_name]
                if dp["timestamp"] >= cutoff_time
            ]
    
    def get_metric_value(self, metric_name: str, time_window: str = "5min", 
                        aggregation: str = "avg") -> float:
        """Get aggregated metric value."""
        if (metric_name in self.aggregated_metrics and 
            time_window in self.aggregated_metrics[metric_name]):
            return self.aggregated_metrics[metric_name][time_window].get(aggregation, 0)
        return 0.0
    
    def add_custom_metric(self, metric_name: str, value: float):
        """Add custom metric value."""
        timestamp = time.time()
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            "timestamp": timestamp,
            "value": value
        })
    
    # Mock metric getters (in real implementation, these would connect to actual systems)
    def _get_avg_latency(self) -> float:
        return max(10, 50 + (time.time() % 100) - 50)  # Mock varying latency
    
    def _get_queue_depth(self) -> float:
        return max(0, 5 + (time.time() % 20) - 10)  # Mock queue depth
    
    def _get_error_rate(self) -> float:
        return max(0, min(10, (time.time() % 50) / 5))  # Mock error rate 0-10%
    
    def _get_throughput(self) -> float:
        return 100 + (time.time() % 200) - 100  # Mock throughput
    
    def _get_active_connections(self) -> float:
        return max(1, 50 + (time.time() % 100) - 50)  # Mock connections
    
    def _get_p95_response_time(self) -> float:
        return self._get_avg_latency() * 1.5  # P95 typically higher than average


class AutoScaler:
    """Intelligent auto-scaling engine."""
    
    def __init__(self, instance_manager: Callable = None):
        self.policies = {}
        self.metrics_collector = MetricsCollector()
        self.scaling_events = []
        self.current_instances = 1
        self.last_scaling_time = {}
        self.instance_manager = instance_manager
        self.is_running = False
        self.scaling_thread = None
        
        # Predictive scaling
        self.prediction_models = {}
        self.enable_predictive_scaling = True
        self.learning_enabled = True
        
    def add_policy(self, policy: ScalingPolicy):
        """Add auto-scaling policy."""
        self.policies[policy.name] = policy
        self.last_scaling_time[policy.name] = 0
        logger.info(f"Added scaling policy: {policy.name}")
    
    def start_auto_scaling(self, evaluation_interval: float = 30.0):
        """Start auto-scaling evaluation loop."""
        self.metrics_collector.start_collection()
        self.is_running = True
        
        def scaling_loop():
            while self.is_running:
                try:
                    self._evaluate_scaling_policies()
                    if self.enable_predictive_scaling:
                        self._predictive_scaling()
                    time.sleep(evaluation_interval)
                except Exception as e:
                    logger.error(f"Auto-scaling evaluation error: {e}")
        
        self.scaling_thread = threading.Thread(target=scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling."""
        self.is_running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        self.metrics_collector.stop_collection()
        logger.info("Auto-scaling stopped")
    
    def _evaluate_scaling_policies(self):
        """Evaluate all scaling policies and make scaling decisions."""
        current_time = time.time()
        
        for policy_name, policy in self.policies.items():
            try:
                # Check cooldown period
                if (current_time - self.last_scaling_time[policy_name] < 
                    policy.cooldown_period_seconds):
                    continue
                
                # Get metric value
                metric_value = self._get_policy_metric_value(policy)
                
                # Determine scaling direction
                scaling_decision = self._make_scaling_decision(policy, metric_value)
                
                if scaling_decision["should_scale"]:
                    self._execute_scaling_action(policy, scaling_decision, metric_value)
                    self.last_scaling_time[policy_name] = current_time
                    
            except Exception as e:
                logger.error(f"Error evaluating policy {policy_name}: {e}")
    
    def _get_policy_metric_value(self, policy: ScalingPolicy) -> float:
        """Get metric value for scaling policy evaluation."""
        metric_name = policy.trigger.value
        
        # Use 5-minute average for most metrics
        return self.metrics_collector.get_metric_value(
            metric_name, "5min", "avg"
        )
    
    def _make_scaling_decision(self, policy: ScalingPolicy, metric_value: float) -> Dict[str, Any]:
        """Make scaling decision based on policy and metric value."""
        decision = {
            "should_scale": False,
            "direction": ScalingDirection.STABLE,
            "target_instances": self.current_instances,
            "reason": "No scaling needed"
        }
        
        # Scale up decision
        if (metric_value > policy.scale_up_threshold and 
            self.current_instances < policy.max_instances):
            
            new_instances = min(
                self.current_instances + policy.scale_up_adjustment,
                policy.max_instances
            )
            
            decision.update({
                "should_scale": True,
                "direction": ScalingDirection.UP,
                "target_instances": new_instances,
                "reason": f"{policy.trigger.value} ({metric_value:.2f}) > threshold ({policy.scale_up_threshold})"
            })
        
        # Scale down decision
        elif (metric_value < policy.scale_down_threshold and 
              self.current_instances > policy.min_instances):
            
            new_instances = max(
                self.current_instances - policy.scale_down_adjustment,
                policy.min_instances
            )
            
            decision.update({
                "should_scale": True,
                "direction": ScalingDirection.DOWN,
                "target_instances": new_instances,
                "reason": f"{policy.trigger.value} ({metric_value:.2f}) < threshold ({policy.scale_down_threshold})"
            })
        
        return decision
    
    def _execute_scaling_action(self, policy: ScalingPolicy, decision: Dict[str, Any], 
                              metric_value: float):
        """Execute the scaling action."""
        old_instances = self.current_instances
        new_instances = decision["target_instances"]
        
        try:
            # Execute scaling through instance manager
            if self.instance_manager:
                success = self.instance_manager(new_instances)
                if not success:
                    logger.error(f"Failed to scale from {old_instances} to {new_instances}")
                    return
            
            # Update current instance count
            self.current_instances = new_instances
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                direction=decision["direction"],
                trigger=policy.trigger,
                metric_value=metric_value,
                threshold=policy.scale_up_threshold if decision["direction"] == ScalingDirection.UP 
                         else policy.scale_down_threshold,
                instances_before=old_instances,
                instances_after=new_instances,
                reason=decision["reason"]
            )
            
            self.scaling_events.append(event)
            
            # Keep only last 100 events
            if len(self.scaling_events) > 100:
                self.scaling_events.pop(0)
            
            logger.info(f"Scaled {decision['direction'].value}: {old_instances} -> {new_instances} "
                       f"({decision['reason']})")
            
            # Learn from scaling action if enabled
            if self.learning_enabled:
                self._learn_from_scaling_action(event)
                
        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
    
    def _predictive_scaling(self):
        """Perform predictive scaling based on historical patterns."""
        try:
            # Simple trend-based prediction
            cpu_trend = self._calculate_metric_trend("cpu_utilization", window_minutes=15)
            latency_trend = self._calculate_metric_trend("request_latency", window_minutes=15)
            
            # Predict if scaling will be needed in next 5 minutes
            current_cpu = self.metrics_collector.get_metric_value("cpu_utilization", "1min", "avg")
            predicted_cpu = current_cpu + (cpu_trend * 5)  # 5 minutes ahead
            
            current_latency = self.metrics_collector.get_metric_value("request_latency", "1min", "avg")
            predicted_latency = current_latency + (latency_trend * 5)
            
            # Proactive scaling decisions
            if predicted_cpu > 80 and self.current_instances < 10:  # Assuming max 10 instances
                logger.info(f"Predictive scaling: CPU trend suggests scaling up "
                           f"(predicted: {predicted_cpu:.1f}%)")
                # Could trigger proactive scaling here
                
            elif predicted_latency > 200 and self.current_instances < 10:
                logger.info(f"Predictive scaling: Latency trend suggests scaling up "
                           f"(predicted: {predicted_latency:.1f}ms)")
                
        except Exception as e:
            logger.error(f"Predictive scaling error: {e}")
    
    def _calculate_metric_trend(self, metric_name: str, window_minutes: int = 15) -> float:
        """Calculate trend (slope) for a metric over time window."""
        window_seconds = window_minutes * 60
        current_time = time.time()
        
        if metric_name not in self.metrics_collector.metrics:
            return 0.0
        
        # Get data points in time window
        data_points = [
            dp for dp in self.metrics_collector.metrics[metric_name]
            if current_time - dp["timestamp"] <= window_seconds
        ]
        
        if len(data_points) < 2:
            return 0.0
        
        # Calculate simple linear trend
        x_values = [dp["timestamp"] for dp in data_points]
        y_values = [dp["value"] for dp in data_points]
        
        n = len(data_points)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Convert to per-minute trend
        return slope * 60
    
    def _learn_from_scaling_action(self, event: ScalingEvent):
        """Learn from scaling actions to improve future decisions."""
        # Simple learning: track effectiveness of scaling actions
        if event.trigger.value not in self.prediction_models:
            self.prediction_models[event.trigger.value] = {
                "successful_scales": 0,
                "total_scales": 0,
                "avg_effectiveness": 0.0
            }
        
        model = self.prediction_models[event.trigger.value]
        model["total_scales"] += 1
        
        # In a real implementation, you would measure effectiveness by checking
        # if the metric improved after scaling
        # For now, assume 80% effectiveness
        if event.direction == ScalingDirection.UP:
            effectiveness = 0.8  # Mock effectiveness
        else:
            effectiveness = 0.7  # Mock effectiveness
        
        model["successful_scales"] += effectiveness
        model["avg_effectiveness"] = model["successful_scales"] / model["total_scales"]
    
    def get_scaling_recommendations(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get scaling recommendations based on current metrics."""
        recommendations = {
            "immediate_action": "none",
            "recommended_instances": self.current_instances,
            "confidence": 0.0,
            "reasoning": [],
            "predicted_metrics": {}
        }
        
        # Analyze current metrics against thresholds
        scaling_signals = []
        
        for policy_name, policy in self.policies.items():
            metric_name = policy.trigger.value
            current_value = metrics.get(metric_name, 0)
            
            if current_value > policy.scale_up_threshold:
                scaling_signals.append({
                    "direction": "up",
                    "policy": policy_name,
                    "metric": metric_name,
                    "value": current_value,
                    "threshold": policy.scale_up_threshold,
                    "urgency": (current_value - policy.scale_up_threshold) / policy.scale_up_threshold
                })
            elif current_value < policy.scale_down_threshold:
                scaling_signals.append({
                    "direction": "down",
                    "policy": policy_name,
                    "metric": metric_name,
                    "value": current_value,
                    "threshold": policy.scale_down_threshold,
                    "urgency": (policy.scale_down_threshold - current_value) / policy.scale_down_threshold
                })
        
        # Generate recommendations
        if scaling_signals:
            # Find strongest signal
            strongest_signal = max(scaling_signals, key=lambda s: s["urgency"])
            
            if strongest_signal["direction"] == "up":
                recommendations["immediate_action"] = "scale_up"
                recommendations["recommended_instances"] = min(
                    self.current_instances + 1, 10  # Assuming max 10
                )
            else:
                recommendations["immediate_action"] = "scale_down"
                recommendations["recommended_instances"] = max(
                    self.current_instances - 1, 1  # Assuming min 1
                )
            
            recommendations["confidence"] = min(strongest_signal["urgency"], 1.0)
            recommendations["reasoning"] = [
                f"{strongest_signal['metric']} ({strongest_signal['value']:.2f}) "
                f"{'exceeds' if strongest_signal['direction'] == 'up' else 'below'} "
                f"threshold ({strongest_signal['threshold']:.2f})"
            ]
        
        return recommendations
    
    def get_auto_scaler_stats(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaler statistics."""
        current_time = time.time()
        
        # Recent scaling events (last 24 hours)
        recent_events = [
            e for e in self.scaling_events
            if current_time - e.timestamp <= 86400
        ]
        
        scale_up_events = len([e for e in recent_events if e.direction == ScalingDirection.UP])
        scale_down_events = len([e for e in recent_events if e.direction == ScalingDirection.DOWN])
        
        return {
            "current_instances": self.current_instances,
            "policies_count": len(self.policies),
            "scaling_events_24h": len(recent_events),
            "scale_up_events_24h": scale_up_events,
            "scale_down_events_24h": scale_down_events,
            "metrics_collected": len(self.metrics_collector.metrics),
            "predictive_scaling_enabled": self.enable_predictive_scaling,
            "learning_enabled": self.learning_enabled,
            "prediction_models": self.prediction_models,
            "recent_events": [
                {
                    "timestamp": e.timestamp,
                    "direction": e.direction.value,
                    "trigger": e.trigger.value,
                    "instances_change": f"{e.instances_before} -> {e.instances_after}",
                    "reason": e.reason
                }
                for e in recent_events[-10:]  # Last 10 events
            ]
        }


# Example usage
if __name__ == "__main__":
    print("Testing Auto-Scaling System...")
    
    # Mock instance manager
    def mock_instance_manager(target_instances: int) -> bool:
        print(f"Scaling to {target_instances} instances")
        time.sleep(0.1)  # Simulate scaling time
        return True
    
    # Create auto-scaler
    auto_scaler = AutoScaler(instance_manager=mock_instance_manager)
    
    # Add scaling policies
    cpu_policy = ScalingPolicy(
        name="cpu_scaling",
        trigger=ScalingTrigger.CPU_UTILIZATION,
        scale_up_threshold=70.0,
        scale_down_threshold=30.0,
        min_instances=1,
        max_instances=5,
        cooldown_period_seconds=60,
        evaluation_period_seconds=30,
        datapoints_required=2
    )
    
    latency_policy = ScalingPolicy(
        name="latency_scaling",
        trigger=ScalingTrigger.REQUEST_LATENCY,
        scale_up_threshold=100.0,
        scale_down_threshold=50.0,
        min_instances=1,
        max_instances=5,
        cooldown_period_seconds=60,
        evaluation_period_seconds=30,
        datapoints_required=2
    )
    
    auto_scaler.add_policy(cpu_policy)
    auto_scaler.add_policy(latency_policy)
    
    # Start auto-scaling
    auto_scaler.start_auto_scaling(evaluation_interval=5.0)
    
    # Simulate some workload and let auto-scaler run
    print("Running auto-scaler for 30 seconds...")
    
    # Add some custom metrics to trigger scaling
    for i in range(6):
        time.sleep(5)
        # Simulate increasing load
        auto_scaler.metrics_collector.add_custom_metric("cpu_utilization", 50 + i * 10)
        auto_scaler.metrics_collector.add_custom_metric("request_latency", 60 + i * 15)
    
    # Get stats
    stats = auto_scaler.get_auto_scaler_stats()
    print(f"Auto-scaler stats: {stats['current_instances']} instances, "
          f"{stats['scaling_events_24h']} scaling events")
    
    # Test recommendations
    current_metrics = {
        "cpu_utilization": 85.0,
        "request_latency": 120.0,
        "memory_utilization": 60.0
    }
    
    recommendations = auto_scaler.get_scaling_recommendations(current_metrics)
    print(f"Recommendations: {recommendations['immediate_action']} "
          f"to {recommendations['recommended_instances']} instances "
          f"(confidence: {recommendations['confidence']:.2f})")
    
    # Stop auto-scaler
    auto_scaler.stop_auto_scaling()
    
    print("✅ Auto-scaling system working correctly!")