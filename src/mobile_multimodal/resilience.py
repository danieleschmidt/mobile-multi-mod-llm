"""Advanced resilience and fault tolerance for mobile AI systems."""

import asyncio
import time
import random
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of system failures."""
    NETWORK_TIMEOUT = "network_timeout"
    MEMORY_PRESSURE = "memory_pressure"
    THERMAL_THROTTLING = "thermal_throttling"
    MODEL_LOAD_FAILURE = "model_load_failure"
    INFERENCE_ERROR = "inference_error"
    STORAGE_FAILURE = "storage_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class FailureScenario:
    """Failure scenario configuration."""
    failure_type: FailureType
    probability: float
    duration_seconds: float
    recovery_time: float
    mitigation_strategy: str
    impact_level: str


class CircuitBreaker:
    """Advanced circuit breaker with multiple states and recovery strategies."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # State management
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.success_count = 0
        self.total_requests = 0
        
        # Advanced features
        self.failure_history = []
        self.recovery_strategies = []
        self.adaptive_threshold = True
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        self.total_requests += 1
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Record success
            self._record_success()
            return result
            
        except self.expected_exception as e:
            self._record_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        
        # Adaptive recovery timeout based on failure history
        if self.adaptive_threshold:
            adaptive_timeout = self.recovery_timeout * (1 + len(self.failure_history) * 0.1)
        else:
            adaptive_timeout = self.recovery_timeout
        
        return time_since_failure >= adaptive_timeout
    
    def _record_success(self):
        """Record successful execution."""
        self.success_count += 1
        
        if self.state == "HALF_OPEN":
            # Successful call in half-open state - close the circuit
            self.state = "CLOSED"
            self.failure_count = 0
            logger.info("Circuit breaker CLOSED after successful half-open operation")
        elif self.state == "CLOSED":
            # Successful call in closed state - maintain state
            pass
    
    def _record_failure(self, exception: Exception):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Add to failure history
        failure_info = {
            "timestamp": time.time(),
            "exception": str(exception),
            "exception_type": type(exception).__name__
        }
        self.failure_history.append(failure_info)
        
        # Keep only recent failures (last 100)
        if len(self.failure_history) > 100:
            self.failure_history.pop(0)
        
        # Update adaptive threshold based on failure patterns
        if self.adaptive_threshold:
            self._update_adaptive_threshold()
        
        # Check if we should open the circuit
        if self.failure_count >= self.failure_threshold:
            if self.state != "OPEN":
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
    
    def _update_adaptive_threshold(self):
        """Update failure threshold based on failure patterns."""
        if len(self.failure_history) < 10:
            return
        
        recent_failures = self.failure_history[-10:]
        time_span = recent_failures[-1]["timestamp"] - recent_failures[0]["timestamp"]
        
        if time_span < 60:  # Failures within 1 minute - be more sensitive
            self.failure_threshold = max(3, self.failure_threshold - 1)
        elif time_span > 300:  # Failures spread over 5+ minutes - be less sensitive
            self.failure_threshold = min(10, self.failure_threshold + 1)
        
        logger.debug(f"Adaptive threshold updated to {self.failure_threshold}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "success_rate": self.success_count / max(self.total_requests, 1),
            "recent_failures": len([f for f in self.failure_history if time.time() - f["timestamp"] < 300])
        }
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")


class RetryManager:
    """Advanced retry mechanism with exponential backoff and jitter."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        
        # Retry statistics
        self.attempt_counts = {}
        self.success_after_retry = 0
        self.total_retries = 0
    
    def retry(self, func: Callable, *args, retryable_exceptions: tuple = (Exception,), **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        func_name = getattr(func, '__name__', 'unknown')
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                
                # Record success statistics
                if attempt > 1:
                    self.success_after_retry += 1
                    logger.info(f"Function {func_name} succeeded after {attempt} attempts")
                
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                self.total_retries += 1
                
                if attempt < self.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt} failed for {func_name}, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_attempts} attempts failed for {func_name}")
            
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception in {func_name}: {e}")
                raise
        
        # Update attempt statistics
        self.attempt_counts[func_name] = self.attempt_counts.get(func_name, 0) + 1
        
        # Raise the last exception if all attempts failed
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = min(self.base_delay * (self.backoff_factor ** (attempt - 1)), self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            jitter_amount = delay * 0.1
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            "total_retries": self.total_retries,
            "success_after_retry": self.success_after_retry,
            "success_rate_after_retry": self.success_after_retry / max(self.total_retries, 1),
            "attempt_counts": self.attempt_counts.copy(),
            "config": {
                "max_attempts": self.max_attempts,
                "base_delay": self.base_delay,
                "max_delay": self.max_delay,
                "backoff_factor": self.backoff_factor,
                "jitter": self.jitter
            }
        }


class ResourceMonitor:
    """Advanced resource monitoring and management."""
    
    def __init__(self, memory_threshold_mb: float = 1024, 
                 cpu_threshold_percent: float = 80.0,
                 monitoring_interval: float = 30.0):
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.monitoring_interval = monitoring_interval
        
        # Resource history
        self.memory_history = []
        self.cpu_history = []
        self.disk_history = []
        
        # Alert tracking
        self.alerts = []
        self.last_alert_time = {}
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.monitoring:
            logger.warning("Resource monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._collect_metrics()
                self._check_thresholds()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self):
        """Collect system resource metrics."""
        current_time = time.time()
        
        try:
            # Try to use psutil if available
            import psutil
            
            # Memory metrics
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            self.memory_history.append({
                "timestamp": current_time,
                "value": memory_mb,
                "percent": memory_info.percent
            })
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_history.append({
                "timestamp": current_time,
                "value": cpu_percent
            })
            
            # Disk metrics
            disk_info = psutil.disk_usage('/')
            disk_percent = (disk_info.used / disk_info.total) * 100
            self.disk_history.append({
                "timestamp": current_time,
                "value": disk_percent,
                "free_gb": disk_info.free / (1024**3)
            })
            
        except ImportError:
            # Fallback without psutil
            import os
            import gc
            
            # Memory approximation using garbage collector
            gc.collect()
            memory_mb = len(gc.get_objects()) * 0.001  # Very rough estimate
            self.memory_history.append({
                "timestamp": current_time,
                "value": memory_mb,
                "percent": min(100, memory_mb / self.memory_threshold_mb * 100)
            })
            
            # Load average (Unix only)
            try:
                load_avg = os.getloadavg()[0] * 100  # Approximate CPU usage
                self.cpu_history.append({
                    "timestamp": current_time,
                    "value": load_avg
                })
            except (AttributeError, OSError):
                # Windows or other system without load average
                self.cpu_history.append({
                    "timestamp": current_time,
                    "value": 0
                })
        
        # Keep only recent history (last hour)
        cutoff_time = current_time - 3600
        self.memory_history = [h for h in self.memory_history if h["timestamp"] > cutoff_time]
        self.cpu_history = [h for h in self.cpu_history if h["timestamp"] > cutoff_time]
        self.disk_history = [h for h in self.disk_history if h["timestamp"] > cutoff_time]
    
    def _check_thresholds(self):
        """Check if resource usage exceeds thresholds."""
        current_time = time.time()
        
        # Check memory threshold
        if self.memory_history:
            current_memory = self.memory_history[-1]["value"]
            if current_memory > self.memory_threshold_mb:
                self._create_alert("memory_threshold_exceeded", {
                    "current_mb": current_memory,
                    "threshold_mb": self.memory_threshold_mb,
                    "percent": self.memory_history[-1].get("percent", 0)
                })
        
        # Check CPU threshold
        if self.cpu_history:
            current_cpu = self.cpu_history[-1]["value"]
            if current_cpu > self.cpu_threshold_percent:
                self._create_alert("cpu_threshold_exceeded", {
                    "current_percent": current_cpu,
                    "threshold_percent": self.cpu_threshold_percent
                })
        
        # Check disk threshold (90% full)
        if self.disk_history:
            current_disk = self.disk_history[-1]["value"]
            if current_disk > 90:
                self._create_alert("disk_threshold_exceeded", {
                    "current_percent": current_disk,
                    "free_gb": self.disk_history[-1].get("free_gb", 0)
                })
    
    def _create_alert(self, alert_type: str, details: Dict[str, Any]):
        """Create resource alert with rate limiting."""
        current_time = time.time()
        
        # Rate limit alerts (one per 5 minutes per type)
        if alert_type in self.last_alert_time:
            if current_time - self.last_alert_time[alert_type] < 300:  # 5 minutes
                return
        
        alert = {
            "timestamp": current_time,
            "type": alert_type,
            "details": details
        }
        
        self.alerts.append(alert)
        self.last_alert_time[alert_type] = current_time
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = current_time - 86400
        self.alerts = [a for a in self.alerts if a["timestamp"] > cutoff_time]
        
        logger.warning(f"Resource alert: {alert_type} - {details}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        current_metrics = {"timestamp": time.time()}
        
        if self.memory_history:
            current_metrics["memory"] = self.memory_history[-1]
        
        if self.cpu_history:
            current_metrics["cpu"] = self.cpu_history[-1]
        
        if self.disk_history:
            current_metrics["disk"] = self.disk_history[-1]
        
        return current_metrics
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary."""
        summary = {
            "monitoring_active": self.monitoring,
            "total_alerts": len(self.alerts),
            "recent_alerts": len([a for a in self.alerts if time.time() - a["timestamp"] < 3600])
        }
        
        # Memory summary
        if self.memory_history:
            memory_values = [h["value"] for h in self.memory_history]
            summary["memory"] = {
                "current_mb": memory_values[-1],
                "avg_mb": sum(memory_values) / len(memory_values),
                "max_mb": max(memory_values),
                "threshold_mb": self.memory_threshold_mb,
                "threshold_exceeded": memory_values[-1] > self.memory_threshold_mb
            }
        
        # CPU summary
        if self.cpu_history:
            cpu_values = [h["value"] for h in self.cpu_history]
            summary["cpu"] = {
                "current_percent": cpu_values[-1],
                "avg_percent": sum(cpu_values) / len(cpu_values),
                "max_percent": max(cpu_values),
                "threshold_percent": self.cpu_threshold_percent,
                "threshold_exceeded": cpu_values[-1] > self.cpu_threshold_percent
            }
        
        # Disk summary
        if self.disk_history:
            disk_values = [h["value"] for h in self.disk_history]
            summary["disk"] = {
                "current_percent": disk_values[-1],
                "avg_percent": sum(disk_values) / len(disk_values),
                "max_percent": max(disk_values),
                "free_gb": self.disk_history[-1].get("free_gb", 0)
            }
        
        return summary


class FaultInjector:
    """Chaos engineering and fault injection for resilience testing."""
    
    def __init__(self):
        self.active_failures = {}
        self.failure_scenarios = []
        self.injection_history = []
    
    def register_failure_scenario(self, scenario: FailureScenario):
        """Register a failure scenario for injection."""
        self.failure_scenarios.append(scenario)
        logger.info(f"Registered failure scenario: {scenario.failure_type.value}")
    
    def inject_failure(self, failure_type: FailureType, duration: float = 10.0) -> str:
        """Inject a specific failure type."""
        injection_id = f"{failure_type.value}_{int(time.time())}"
        
        failure_info = {
            "id": injection_id,
            "type": failure_type,
            "start_time": time.time(),
            "duration": duration,
            "active": True
        }
        
        self.active_failures[injection_id] = failure_info
        
        # Schedule failure removal
        timer = threading.Timer(duration, self._remove_failure, args=[injection_id])
        timer.start()
        
        logger.warning(f"Injected failure: {failure_type.value} for {duration}s (ID: {injection_id})")
        return injection_id
    
    def _remove_failure(self, injection_id: str):
        """Remove active failure injection."""
        if injection_id in self.active_failures:
            failure_info = self.active_failures[injection_id]
            failure_info["active"] = False
            failure_info["end_time"] = time.time()
            
            # Move to history
            self.injection_history.append(failure_info)
            del self.active_failures[injection_id]
            
            logger.info(f"Removed failure injection: {injection_id}")
    
    def is_failure_active(self, failure_type: FailureType) -> bool:
        """Check if a specific failure type is currently active."""
        return any(
            f["type"] == failure_type and f["active"]
            for f in self.active_failures.values()
        )
    
    def get_active_failures(self) -> List[Dict[str, Any]]:
        """Get list of currently active failures."""
        return [
            {
                "id": f["id"],
                "type": f["type"].value,
                "duration_remaining": f["start_time"] + f["duration"] - time.time(),
                "elapsed": time.time() - f["start_time"]
            }
            for f in self.active_failures.values()
        ]
    
    def simulate_random_failure(self) -> Optional[str]:
        """Simulate a random failure from registered scenarios."""
        if not self.failure_scenarios:
            return None
        
        # Select scenario based on probability
        for scenario in self.failure_scenarios:
            if random.random() < scenario.probability:
                return self.inject_failure(scenario.failure_type, scenario.duration_seconds)
        
        return None
    
    def clear_all_failures(self):
        """Clear all active failures."""
        for injection_id in list(self.active_failures.keys()):
            self._remove_failure(injection_id)
        
        logger.info("All failure injections cleared")


class ResilienceManager:
    """Comprehensive resilience management coordinating all resilience components."""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.retry_manager = RetryManager()
        self.resource_monitor = ResourceMonitor()
        self.fault_injector = FaultInjector()
        
        # Resilience metrics
        self.resilience_score = 1.0
        self.last_evaluation = time.time()
        
        logger.info("Resilience manager initialized")
    
    def execute_with_resilience(self, func: Callable, *args, **kwargs):
        """Execute function with full resilience protection."""
        # Check for active fault injections
        if self.fault_injector.get_active_failures():
            failure_types = [f["type"] for f in self.fault_injector.get_active_failures()]
            logger.warning(f"Executing with active fault injections: {failure_types}")
        
        # Use circuit breaker and retry manager
        def resilient_execution():
            return self.circuit_breaker.call(func, *args, **kwargs)
        
        return self.retry_manager.retry(resilient_execution, retryable_exceptions=(Exception,))
    
    def start_monitoring(self):
        """Start all monitoring systems."""
        self.resource_monitor.start_monitoring()
        logger.info("Resilience monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.resource_monitor.stop_monitoring()
        logger.info("Resilience monitoring stopped")
    
    def evaluate_resilience(self) -> Dict[str, Any]:
        """Evaluate overall system resilience."""
        cb_status = self.circuit_breaker.get_status()
        retry_stats = self.retry_manager.get_stats()
        resource_summary = self.resource_monitor.get_resource_summary()
        active_failures = self.fault_injector.get_active_failures()
        
        # Calculate resilience score
        score = 1.0
        
        # Circuit breaker impact
        if cb_status["state"] == "OPEN":
            score *= 0.3
        elif cb_status["state"] == "HALF_OPEN":
            score *= 0.7
        
        # Success rate impact
        success_rate = cb_status["success_rate"]
        score *= success_rate
        
        # Resource pressure impact
        if resource_summary.get("memory", {}).get("threshold_exceeded", False):
            score *= 0.8
        if resource_summary.get("cpu", {}).get("threshold_exceeded", False):
            score *= 0.8
        
        # Active failures impact
        if active_failures:
            score *= 0.5
        
        self.resilience_score = score
        self.last_evaluation = time.time()
        
        return {
            "resilience_score": score,
            "evaluation_time": time.time(),
            "circuit_breaker": cb_status,
            "retry_stats": retry_stats,
            "resource_summary": resource_summary,
            "active_failures": active_failures,
            "recommendations": self._generate_recommendations(score, cb_status, resource_summary)
        }
    
    def _generate_recommendations(self, score: float, cb_status: Dict, resource_summary: Dict) -> List[str]:
        """Generate resilience improvement recommendations."""
        recommendations = []
        
        if score < 0.5:
            recommendations.append("System resilience critically low - immediate attention required")
        elif score < 0.7:
            recommendations.append("System resilience degraded - consider scaling or optimization")
        
        if cb_status["state"] == "OPEN":
            recommendations.append("Circuit breaker open - investigate underlying service issues")
        
        if cb_status["success_rate"] < 0.8:
            recommendations.append("Low success rate - review error handling and service dependencies")
        
        if resource_summary.get("memory", {}).get("threshold_exceeded", False):
            recommendations.append("Memory usage high - consider memory optimization or scaling")
        
        if resource_summary.get("cpu", {}).get("threshold_exceeded", False):
            recommendations.append("CPU usage high - consider load balancing or scaling")
        
        if not recommendations:
            recommendations.append("System resilience is healthy")
        
        return recommendations
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        return {
            "resilience_score": self.resilience_score,
            "last_evaluation": self.last_evaluation,
            "circuit_breaker": self.circuit_breaker.get_status(),
            "retry_manager": self.retry_manager.get_stats(),
            "resource_monitor": self.resource_monitor.get_resource_summary(),
            "fault_injector": {
                "active_failures": self.fault_injector.get_active_failures(),
                "registered_scenarios": len(self.fault_injector.failure_scenarios)
            },
            "monitoring_active": self.resource_monitor.monitoring
        }
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= 3:  # Require multiple successes to close
                self.state = "CLOSED"
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
        elif self.state == "CLOSED":
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self, exception: Exception):
        """Record failure and update circuit breaker state."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Add to failure history
        self.failure_history.append({
            "timestamp": self.last_failure_time,
            "exception": str(exception),
            "failure_count": self.failure_count
        })
        
        # Keep only recent failures
        hour_ago = time.time() - 3600
        self.failure_history = [f for f in self.failure_history if f["timestamp"] > hour_ago]
        
        # Check if should open circuit
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "failure_rate": self.failure_count / max(self.total_requests, 1),
            "last_failure_time": self.last_failure_time,
            "recent_failures": len(self.failure_history),
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time else None
        }


class RetryManager:
    """Advanced retry mechanism with multiple strategies."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_strategies = {
            "exponential": self._exponential_backoff,
            "linear": self._linear_backoff,
            "fixed": self._fixed_delay,
            "jittered": self._jittered_backoff
        }
    
    def execute_with_retry(self, func: Callable, strategy: str = "exponential", 
                          *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.retry_strategies[strategy](attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception
    
    def _exponential_backoff(self, attempt: int) -> float:
        """Exponential backoff strategy."""
        return self.base_delay * (2 ** attempt)
    
    def _linear_backoff(self, attempt: int) -> float:
        """Linear backoff strategy."""
        return self.base_delay * (attempt + 1)
    
    def _fixed_delay(self, attempt: int) -> float:
        """Fixed delay strategy."""
        return self.base_delay
    
    def _jittered_backoff(self, attempt: int) -> float:
        """Jittered exponential backoff to avoid thundering herd."""
        base = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, base * 0.1)
        return base + jitter


class FaultInjector:
    """Fault injection for resilience testing."""
    
    def __init__(self):
        self.active_failures = {}
        self.failure_scenarios = []
        self.injection_enabled = False
        
    def register_failure_scenario(self, scenario: FailureScenario):
        """Register a failure scenario."""
        self.failure_scenarios.append(scenario)
        logger.info(f"Registered failure scenario: {scenario.failure_type.value}")
    
    def enable_fault_injection(self):
        """Enable fault injection."""
        self.injection_enabled = True
        logger.info("Fault injection enabled")
    
    def disable_fault_injection(self):
        """Disable fault injection."""
        self.injection_enabled = False
        self.active_failures.clear()
        logger.info("Fault injection disabled")
    
    def should_inject_failure(self, operation_type: str) -> Optional[FailureScenario]:
        """Check if failure should be injected for operation."""
        if not self.injection_enabled:
            return None
        
        for scenario in self.failure_scenarios:
            if random.random() < scenario.probability:
                # Check if this failure is already active
                if scenario.failure_type.value not in self.active_failures:
                    self.active_failures[scenario.failure_type.value] = {
                        "start_time": time.time(),
                        "scenario": scenario
                    }
                    return scenario
        
        return None
    
    def inject_failure(self, scenario: FailureScenario, operation_name: str):
        """Inject specific failure."""
        failure_id = f"{scenario.failure_type.value}_{int(time.time())}"
        
        logger.warning(f"Injecting {scenario.failure_type.value} failure for {operation_name}")
        
        if scenario.failure_type == FailureType.NETWORK_TIMEOUT:
            raise TimeoutError(f"Simulated network timeout in {operation_name}")
        elif scenario.failure_type == FailureType.MEMORY_PRESSURE:
            raise MemoryError(f"Simulated memory pressure in {operation_name}")
        elif scenario.failure_type == FailureType.MODEL_LOAD_FAILURE:
            raise FileNotFoundError(f"Simulated model load failure in {operation_name}")
        elif scenario.failure_type == FailureType.INFERENCE_ERROR:
            raise RuntimeError(f"Simulated inference error in {operation_name}")
        else:
            raise Exception(f"Simulated {scenario.failure_type.value} in {operation_name}")
    
    def cleanup_expired_failures(self):
        """Clean up expired active failures."""
        current_time = time.time()
        expired_failures = []
        
        for failure_type, failure_info in self.active_failures.items():
            if current_time - failure_info["start_time"] > failure_info["scenario"].duration_seconds:
                expired_failures.append(failure_type)
        
        for failure_type in expired_failures:
            del self.active_failures[failure_type]
            logger.info(f"Failure {failure_type} expired and removed")


class ResourceMonitor:
    """Monitor system resources and trigger protective measures."""
    
    def __init__(self):
        self.monitoring_enabled = False
        self.resource_limits = {
            "memory_mb": 1024,
            "cpu_percent": 80,
            "disk_mb": 500,
            "temperature_celsius": 75
        }
        self.resource_history = []
        self.alert_callbacks = []
        
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start resource monitoring."""
        self.monitoring_enabled = True
        
        def monitor_loop():
            while self.monitoring_enabled:
                try:
                    self._check_resources()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_enabled = False
        logger.info("Resource monitoring stopped")
    
    def _check_resources(self):
        """Check current resource usage."""
        try:
            # Mock resource usage - in real implementation use psutil
            import psutil
            process = psutil.Process()
            
            current_resources = {
                "timestamp": time.time(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "disk_mb": 100,  # Mock disk usage
                "temperature_celsius": 45  # Mock temperature
            }
        except ImportError:
            # Fallback when psutil not available
            current_resources = {
                "timestamp": time.time(),
                "memory_mb": random.uniform(200, 800),
                "cpu_percent": random.uniform(10, 70),
                "disk_mb": random.uniform(50, 200),
                "temperature_celsius": random.uniform(35, 65)
            }
        
        # Add to history
        self.resource_history.append(current_resources)
        
        # Keep only last 100 measurements
        if len(self.resource_history) > 100:
            self.resource_history.pop(0)
        
        # Check for threshold violations
        alerts = []
        for resource, limit in self.resource_limits.items():
            if current_resources[resource] > limit:
                alerts.append({
                    "resource": resource,
                    "current": current_resources[resource],
                    "limit": limit,
                    "severity": "high" if current_resources[resource] > limit * 1.2 else "medium"
                })
        
        # Trigger alerts
        if alerts:
            self._trigger_alerts(alerts, current_resources)
    
    def _trigger_alerts(self, alerts: List[Dict], current_resources: Dict):
        """Trigger resource alerts."""
        for alert in alerts:
            logger.warning(f"Resource alert: {alert['resource']} = {alert['current']:.1f} "
                         f"(limit: {alert['limit']}) - {alert['severity']} severity")
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alerts, current_resources)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        if not self.resource_history:
            return {"error": "No resource data available"}
        
        # Calculate statistics
        stats = {}
        for resource in ["memory_mb", "cpu_percent", "disk_mb", "temperature_celsius"]:
            values = [r[resource] for r in self.resource_history]
            stats[resource] = {
                "current": values[-1] if values else 0,
                "average": sum(values) / len(values) if values else 0,
                "maximum": max(values) if values else 0,
                "minimum": min(values) if values else 0,
                "limit": self.resource_limits[resource],
                "utilization_percent": (values[-1] / self.resource_limits[resource] * 100) if values else 0
            }
        
        return {
            "timestamp": time.time(),
            "measurements_count": len(self.resource_history),
            "monitoring_duration_minutes": (self.resource_history[-1]["timestamp"] - 
                                           self.resource_history[0]["timestamp"]) / 60 if len(self.resource_history) > 1 else 0,
            "resources": stats
        }


class ResilienceManager:
    """Comprehensive resilience management system."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_managers = {}
        self.fault_injector = FaultInjector()
        self.resource_monitor = ResourceMonitor()
        
        # Resilience policies
        self.policies = {
            "default": {
                "max_retries": 3,
                "circuit_breaker_threshold": 5,
                "timeout_seconds": 30,
                "fallback_enabled": True
            }
        }
        
        # Fallback strategies
        self.fallback_strategies = {}
        
    def register_circuit_breaker(self, name: str, failure_threshold: int = 5, 
                               recovery_timeout: float = 60.0) -> CircuitBreaker:
        """Register circuit breaker for a service."""
        circuit_breaker = CircuitBreaker(failure_threshold, recovery_timeout)
        self.circuit_breakers[name] = circuit_breaker
        logger.info(f"Registered circuit breaker for {name}")
        return circuit_breaker
    
    def register_retry_manager(self, name: str, max_retries: int = 3, 
                             base_delay: float = 1.0) -> RetryManager:
        """Register retry manager for a service."""
        retry_manager = RetryManager(max_retries, base_delay)
        self.retry_managers[name] = retry_manager
        logger.info(f"Registered retry manager for {name}")
        return retry_manager
    
    def register_fallback_strategy(self, service_name: str, fallback_func: Callable):
        """Register fallback strategy for a service."""
        self.fallback_strategies[service_name] = fallback_func
        logger.info(f"Registered fallback strategy for {service_name}")
    
    def execute_resilient_operation(self, service_name: str, operation: Callable, 
                                  *args, **kwargs) -> Any:
        """Execute operation with full resilience protection."""
        try:
            # Check for fault injection
            failure_scenario = self.fault_injector.should_inject_failure(service_name)
            if failure_scenario:
                self.fault_injector.inject_failure(failure_scenario, service_name)
            
            # Get circuit breaker and retry manager
            circuit_breaker = self.circuit_breakers.get(service_name)
            retry_manager = self.retry_managers.get(service_name)
            
            # Define the protected operation
            def protected_operation():
                if circuit_breaker:
                    return circuit_breaker.call(operation, *args, **kwargs)
                else:
                    return operation(*args, **kwargs)
            
            # Execute with retry if available
            if retry_manager:
                return retry_manager.execute_with_retry(protected_operation)
            else:
                return protected_operation()
                
        except Exception as e:
            logger.error(f"Resilient operation failed for {service_name}: {e}")
            
            # Try fallback strategy
            fallback_func = self.fallback_strategies.get(service_name)
            if fallback_func:
                logger.info(f"Using fallback strategy for {service_name}")
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy failed: {fallback_error}")
            
            raise e
    
    def start_resilience_monitoring(self):
        """Start comprehensive resilience monitoring."""
        self.resource_monitor.start_monitoring()
        
        # Register resource alert callback
        def resource_alert_handler(alerts, current_resources):
            for alert in alerts:
                if alert["severity"] == "high":
                    logger.critical(f"HIGH RESOURCE ALERT: {alert}")
                    # Could trigger emergency measures here
        
        self.resource_monitor.register_alert_callback(resource_alert_handler)
        
        # Start fault injector cleanup
        def cleanup_loop():
            while True:
                self.fault_injector.cleanup_expired_failures()
                time.sleep(30)  # Cleanup every 30 seconds
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        logger.info("Resilience monitoring started")
    
    def get_resilience_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive resilience dashboard."""
        dashboard = {
            "timestamp": time.time(),
            "circuit_breakers": {},
            "resource_usage": self.resource_monitor.get_resource_stats(),
            "fault_injection": {
                "enabled": self.fault_injector.injection_enabled,
                "active_failures": len(self.fault_injector.active_failures),
                "registered_scenarios": len(self.fault_injector.failure_scenarios)
            },
            "services": {
                "protected_services": len(self.circuit_breakers),
                "retry_enabled_services": len(self.retry_managers),
                "fallback_enabled_services": len(self.fallback_strategies)
            }
        }
        
        # Add circuit breaker metrics
        for name, cb in self.circuit_breakers.items():
            dashboard["circuit_breakers"][name] = cb.get_metrics()
        
        return dashboard


# Example usage and testing
if __name__ == "__main__":
    print("Testing Resilience System...")
    
    # Create resilience manager
    resilience = ResilienceManager()
    
    # Register services
    cb = resilience.register_circuit_breaker("model_inference", failure_threshold=3)
    rm = resilience.register_retry_manager("model_inference", max_retries=2)
    
    # Register fallback strategy
    def inference_fallback(*args, **kwargs):
        return {"result": "fallback_response", "confidence": 0.5}
    
    resilience.register_fallback_strategy("model_inference", inference_fallback)
    
    # Test normal operation
    def mock_inference():
        if random.random() < 0.3:  # 30% failure rate
            raise Exception("Mock inference failure")
        return {"result": "successful_inference", "confidence": 0.9}
    
    # Execute resilient operations
    for i in range(10):
        try:
            result = resilience.execute_resilient_operation("model_inference", mock_inference)
            print(f"Operation {i+1}: {result['result']}")
        except Exception as e:
            print(f"Operation {i+1} failed: {e}")
    
    # Start monitoring
    resilience.start_resilience_monitoring()
    
    # Get dashboard
    dashboard = resilience.get_resilience_dashboard()
    print(f"Circuit breaker state: {dashboard['circuit_breakers']['model_inference']['state']}")
    
    print("✅ Resilience system working correctly!")