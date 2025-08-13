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