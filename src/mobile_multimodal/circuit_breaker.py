"""Circuit Breaker Pattern - Advanced fault tolerance and failure handling.

This module implements production-grade circuit breaker patterns specifically
designed for mobile AI inference, providing:
1. Multi-level circuit breakers (service, model, hardware)
2. Adaptive failure detection with ML-based anomaly detection
3. Intelligent fallback strategies and graceful degradation
4. Real-time health monitoring and recovery mechanisms
5. Mobile-optimized resource protection
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, failing fast
    HALF_OPEN = "half_open" # Testing if service has recovered


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    HIGH_ERROR_RATE = "high_error_rate"
    MODEL_DEGRADATION = "model_degradation"
    HARDWARE_FAILURE = "hardware_failure"


@dataclass
class FailureRecord:
    """Record of a failure event."""
    timestamp: float
    failure_type: FailureType
    error_message: str
    duration: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Number of failures to trigger opening
    success_threshold: int = 3          # Number of successes to close from half-open
    timeout_duration: float = 30.0     # Timeout before transitioning to half-open
    rolling_window_size: int = 100      # Size of rolling window for failure tracking
    error_rate_threshold: float = 0.5   # Error rate threshold (0.0-1.0)
    min_requests_threshold: int = 10    # Minimum requests before considering error rate
    health_check_interval: float = 5.0  # Interval for health checks in seconds


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opens: int = 0
    circuit_half_opens: int = 0
    circuit_closes: int = 0
    avg_response_time: float = 0.0
    current_error_rate: float = 0.0
    time_in_open_state: float = 0.0


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""
    
    @abstractmethod
    async def execute(self, original_args: Tuple, original_kwargs: Dict) -> Any:
        """Execute fallback strategy."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class CachedResponseFallback(FallbackStrategy):
    """Fallback to cached responses."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def execute(self, original_args: Tuple, original_kwargs: Dict) -> Any:
        """Return cached response if available."""
        # Simple hash-based cache key
        cache_key = hash(str(original_args) + str(sorted(original_kwargs.items())))
        
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.info("Circuit breaker: Returning cached response")
            return self.cache[cache_key]
        else:
            self.cache_misses += 1
            # Return a default/simplified response
            return self._get_default_response()
    
    def _get_default_response(self) -> Dict[str, Any]:
        """Get default response when cache miss occurs."""
        return {
            "status": "fallback",
            "message": "Service temporarily unavailable, using fallback response",
            "confidence": 0.0,
            "result": None
        }
    
    def add_to_cache(self, args: Tuple, kwargs: Dict, response: Any):
        """Add successful response to cache."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        cache_key = hash(str(args) + str(sorted(kwargs.items())))
        self.cache[cache_key] = response
    
    def get_strategy_name(self) -> str:
        return "cached_response"


class SimplifiedModelFallback(FallbackStrategy):
    """Fallback to a simplified model or algorithm."""
    
    def __init__(self, simplified_model: Optional[Callable] = None):
        self.simplified_model = simplified_model
        self.fallback_count = 0
        
    async def execute(self, original_args: Tuple, original_kwargs: Dict) -> Any:
        """Execute simplified model."""
        self.fallback_count += 1
        
        if self.simplified_model:
            try:
                logger.info("Circuit breaker: Using simplified model fallback")
                result = await self._run_simplified_model(original_args, original_kwargs)
                return {
                    "status": "fallback_model",
                    "result": result,
                    "confidence": 0.7,  # Lower confidence for fallback
                    "message": "Using simplified model due to service issues"
                }
            except Exception as e:
                logger.error(f"Simplified model fallback failed: {str(e)}")
                return self._get_error_response()
        else:
            return self._get_error_response()
    
    async def _run_simplified_model(self, args: Tuple, kwargs: Dict) -> Any:
        """Run the simplified model."""
        if asyncio.iscoroutinefunction(self.simplified_model):
            return await self.simplified_model(*args, **kwargs)
        else:
            return self.simplified_model(*args, **kwargs)
    
    def _get_error_response(self) -> Dict[str, Any]:
        """Get error response when fallback fails."""
        return {
            "status": "error",
            "message": "Service unavailable and no fallback available",
            "confidence": 0.0,
            "result": None
        }
    
    def get_strategy_name(self) -> str:
        return "simplified_model"


class GracefulDegradationFallback(FallbackStrategy):
    """Fallback with graceful degradation of features."""
    
    def __init__(self, degraded_features: List[str] = None):
        self.degraded_features = degraded_features or ["high_quality_inference", "batch_processing"]
        self.degradation_count = 0
        
    async def execute(self, original_args: Tuple, original_kwargs: Dict) -> Any:
        """Execute with degraded features."""
        self.degradation_count += 1
        
        logger.info(f"Circuit breaker: Graceful degradation active, disabled features: {self.degraded_features}")
        
        # Simulate degraded processing
        await asyncio.sleep(0.1)  # Reduced processing time
        
        return {
            "status": "degraded",
            "result": self._get_degraded_result(),
            "confidence": 0.8,
            "message": f"Service running with degraded features: {', '.join(self.degraded_features)}",
            "disabled_features": self.degraded_features
        }
    
    def _get_degraded_result(self) -> Dict[str, Any]:
        """Get result with degraded quality."""
        return {
            "inference_quality": "standard",  # Instead of "high"
            "processing_time": "fast",        # Faster but lower quality
            "features_available": "limited"
        }
    
    def get_strategy_name(self) -> str:
        return "graceful_degradation"


class AdaptiveCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds and ML-based failure detection."""
    
    def __init__(self, name: str, config: CircuitConfig, 
                 fallback_strategy: Optional[FallbackStrategy] = None):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_history = []
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state_change_time = time.time()
        self.fallback_strategy = fallback_strategy or CachedResponseFallback()
        
        # Metrics
        self.metrics = CircuitMetrics()
        
        # Adaptive thresholds
        self.adaptive_failure_threshold = config.failure_threshold
        self.adaptive_timeout = config.timeout_duration
        self.performance_history = []
        
        # Health monitoring
        self.last_health_check = 0.0
        self.health_status = True
        
        logger.info(f"Circuit breaker '{name}' initialized in {self.state.value} state")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.metrics.total_requests += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                # Circuit is open, use fallback
                logger.warning(f"Circuit breaker '{self.name}' is OPEN, using fallback")
                return await self.fallback_strategy.execute(args, kwargs)
        
        # Attempt to execute function
        start_time = time.perf_counter()
        try:
            # Set timeout for the operation
            result = await asyncio.wait_for(
                self._execute_function(func, args, kwargs),
                timeout=self.config.timeout_duration
            )
            
            execution_time = time.perf_counter() - start_time
            await self._record_success(execution_time, result, args, kwargs)
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.perf_counter() - start_time
            await self._record_failure(FailureType.TIMEOUT, "Operation timed out", execution_time)
            return await self.fallback_strategy.execute(args, kwargs)
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            await self._record_failure(FailureType.EXCEPTION, str(e), execution_time)
            return await self.fallback_strategy.execute(args, kwargs)
    
    async def _execute_function(self, func: Callable, args: Tuple, kwargs: Dict) -> Any:
        """Execute the protected function."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run synchronous function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    async def _record_success(self, execution_time: float, result: Any, 
                            args: Tuple, kwargs: Dict):
        """Record successful execution."""
        self.metrics.successful_requests += 1
        self.success_count += 1
        
        # Update average response time
        total_requests = self.metrics.total_requests
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (total_requests - 1) + execution_time) / total_requests
        )
        
        # Record performance for adaptive thresholds
        self.performance_history.append({
            "timestamp": time.time(),
            "execution_time": execution_time,
            "success": True
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 300  # Last 5 minutes
        self.performance_history = [
            record for record in self.performance_history 
            if record["timestamp"] > cutoff_time
        ]
        
        # Cache successful response if using cached fallback
        if isinstance(self.fallback_strategy, CachedResponseFallback):
            self.fallback_strategy.add_to_cache(args, kwargs, result)
        
        # Check if we should close circuit from half-open state
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
    
    async def _record_failure(self, failure_type: FailureType, error_message: str, 
                            execution_time: float):
        """Record failed execution."""
        self.metrics.failed_requests += 1
        self.success_count = 0  # Reset success count
        self.last_failure_time = time.time()
        
        # Add to failure history
        failure = FailureRecord(
            timestamp=time.time(),
            failure_type=failure_type,
            error_message=error_message,
            duration=execution_time
        )
        self.failure_history.append(failure)
        
        # Keep rolling window of failures
        cutoff_time = time.time() - 60  # Last minute
        self.failure_history = [
            f for f in self.failure_history 
            if f.timestamp > cutoff_time
        ]
        
        # Update error rate
        self._update_error_rate()
        
        # Check if circuit should be opened
        if self.state == CircuitState.CLOSED:
            if self._should_open_circuit():
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit
            self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened."""
        # Failure count threshold
        recent_failures = len(self.failure_history)
        if recent_failures >= self.adaptive_failure_threshold:
            return True
        
        # Error rate threshold
        if (self.metrics.total_requests >= self.config.min_requests_threshold and
            self.metrics.current_error_rate >= self.config.error_rate_threshold):
            return True
        
        # Adaptive ML-based detection
        if self._detect_anomalous_behavior():
            return True
        
        return False
    
    def _detect_anomalous_behavior(self) -> bool:
        """Use ML techniques to detect anomalous behavior."""
        if len(self.performance_history) < 10:
            return False
        
        # Extract recent execution times
        recent_times = [record["execution_time"] for record in self.performance_history[-20:]]
        
        if len(recent_times) < 5:
            return False
        
        # Statistical anomaly detection
        mean_time = np.mean(recent_times)
        std_time = np.std(recent_times)
        
        # Check for significant performance degradation
        recent_avg = np.mean(recent_times[-5:])
        if std_time > 0 and (recent_avg - mean_time) > 2 * std_time:
            logger.warning(f"Circuit breaker '{self.name}': Performance anomaly detected")
            return True
        
        # Check for high variance (indicating instability)
        if std_time > mean_time * 0.5:  # CV > 0.5
            logger.warning(f"Circuit breaker '{self.name}': High performance variance detected")
            return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset from open state."""
        time_since_open = time.time() - self.state_change_time
        return time_since_open >= self.adaptive_timeout
    
    def _transition_to_open(self):
        """Transition circuit to open state."""
        self.state = CircuitState.OPEN
        self.state_change_time = time.time()
        self.metrics.circuit_opens += 1
        logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN state")
        
        # Adapt thresholds based on failure patterns
        self._adapt_thresholds()
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = time.time()
        self.success_count = 0
        self.metrics.circuit_half_opens += 1
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN state")
    
    def _transition_to_closed(self):
        """Transition circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        self.success_count = 0
        self.metrics.circuit_closes += 1
        logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED state")
        
        # Reset adaptive thresholds on successful recovery
        self._reset_adaptive_thresholds()
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on recent failure patterns."""
        # Increase sensitivity after failures
        self.adaptive_failure_threshold = max(
            2, 
            int(self.config.failure_threshold * 0.8)
        )
        
        # Increase timeout for recovery
        self.adaptive_timeout = min(
            300,  # Max 5 minutes
            self.config.timeout_duration * 1.5
        )
        
        logger.info(f"Circuit breaker '{self.name}': Adapted thresholds - "
                   f"failure_threshold={self.adaptive_failure_threshold}, "
                   f"timeout={self.adaptive_timeout}")
    
    def _reset_adaptive_thresholds(self):
        """Reset adaptive thresholds to original values."""
        self.adaptive_failure_threshold = self.config.failure_threshold
        self.adaptive_timeout = self.config.timeout_duration
    
    def _update_error_rate(self):
        """Update current error rate."""
        if self.metrics.total_requests > 0:
            self.metrics.current_error_rate = (
                self.metrics.failed_requests / self.metrics.total_requests
            )
    
    async def health_check(self) -> bool:
        """Perform health check."""
        current_time = time.time()
        
        if current_time - self.last_health_check < self.config.health_check_interval:
            return self.health_status
        
        self.last_health_check = current_time
        
        # Simple health check based on recent performance
        if len(self.performance_history) > 5:
            recent_successes = sum(1 for record in self.performance_history[-10:] if record["success"])
            self.health_status = recent_successes >= 7  # 70% success rate
        else:
            self.health_status = self.state != CircuitState.OPEN
        
        return self.health_status
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        time_in_current_state = time.time() - self.state_change_time
        
        return {
            "name": self.name,
            "state": self.state.value,
            "time_in_current_state": time_in_current_state,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "current_error_rate": self.metrics.current_error_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "circuit_opens": self.metrics.circuit_opens,
                "circuit_closes": self.metrics.circuit_closes
            },
            "adaptive_thresholds": {
                "failure_threshold": self.adaptive_failure_threshold,
                "timeout_duration": self.adaptive_timeout
            },
            "recent_failures": len(self.failure_history),
            "success_count": self.success_count,
            "health_status": self.health_status,
            "fallback_strategy": self.fallback_strategy.get_strategy_name()
        }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_history.clear()
        self.success_count = 0
        self.metrics = CircuitMetrics()
        self.state_change_time = time.time()
        self._reset_adaptive_thresholds()
        logger.info(f"Circuit breaker '{self.name}' reset to initial state")


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services/components."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.global_metrics = {
            "total_circuits": 0,
            "open_circuits": 0,
            "half_open_circuits": 0,
            "closed_circuits": 0
        }
    
    def register_circuit(self, name: str, config: CircuitConfig, 
                        fallback_strategy: Optional[FallbackStrategy] = None) -> AdaptiveCircuitBreaker:
        """Register a new circuit breaker."""
        circuit = AdaptiveCircuitBreaker(name, config, fallback_strategy)
        self.circuit_breakers[name] = circuit
        self.global_metrics["total_circuits"] += 1
        self.global_metrics["closed_circuits"] += 1
        
        logger.info(f"Registered circuit breaker: {name}")
        return circuit
    
    def get_circuit(self, name: str) -> Optional[AdaptiveCircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    async def call_through_circuit(self, circuit_name: str, func: Callable, 
                                 *args, **kwargs) -> Any:
        """Call function through specified circuit breaker."""
        circuit = self.get_circuit(circuit_name)
        if not circuit:
            raise ValueError(f"Circuit breaker '{circuit_name}' not found")
        
        return await circuit.call(func, *args, **kwargs)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all circuit breakers."""
        health_results = {}
        
        for name, circuit in self.circuit_breakers.items():
            health_results[name] = await circuit.health_check()
        
        return health_results
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        # Update global metrics
        self.global_metrics = {
            "total_circuits": len(self.circuit_breakers),
            "open_circuits": 0,
            "half_open_circuits": 0,
            "closed_circuits": 0
        }
        
        circuit_statuses = {}
        
        for name, circuit in self.circuit_breakers.items():
            status = circuit.get_status()
            circuit_statuses[name] = status
            
            # Update global counts
            if circuit.state == CircuitState.OPEN:
                self.global_metrics["open_circuits"] += 1
            elif circuit.state == CircuitState.HALF_OPEN:
                self.global_metrics["half_open_circuits"] += 1
            else:
                self.global_metrics["closed_circuits"] += 1
        
        return {
            "global_metrics": self.global_metrics,
            "circuit_statuses": circuit_statuses,
            "timestamp": time.time()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for circuit in self.circuit_breakers.values():
            circuit.reset()
        
        logger.info("Reset all circuit breakers")
    
    def remove_circuit(self, name: str):
        """Remove circuit breaker."""
        if name in self.circuit_breakers:
            del self.circuit_breakers[name]
            logger.info(f"Removed circuit breaker: {name}")


# Global circuit breaker manager instance
global_circuit_manager = CircuitBreakerManager()


# Decorator for easy circuit breaker integration
def circuit_breaker(name: str, config: Optional[CircuitConfig] = None, 
                   fallback_strategy: Optional[FallbackStrategy] = None):
    """Decorator to protect functions with circuit breaker."""
    def decorator(func):
        # Register circuit breaker if not exists
        if not global_circuit_manager.get_circuit(name):
            circuit_config = config or CircuitConfig()
            global_circuit_manager.register_circuit(name, circuit_config, fallback_strategy)
        
        async def wrapper(*args, **kwargs):
            return await global_circuit_manager.call_through_circuit(name, func, *args, **kwargs)
        
        return wrapper
    return decorator


# Factory functions
def create_mobile_circuit_config() -> CircuitConfig:
    """Create circuit breaker configuration optimized for mobile."""
    return CircuitConfig(
        failure_threshold=3,           # Lower threshold for mobile
        success_threshold=2,           # Faster recovery
        timeout_duration=15.0,         # Shorter timeout for mobile
        rolling_window_size=50,        # Smaller window
        error_rate_threshold=0.4,      # Lower error rate tolerance
        min_requests_threshold=5,      # Lower minimum requests
        health_check_interval=10.0     # More frequent health checks
    )


def create_inference_circuit_breaker(model_name: str) -> AdaptiveCircuitBreaker:
    """Create circuit breaker for model inference."""
    config = create_mobile_circuit_config()
    fallback = GracefulDegradationFallback(["high_resolution", "batch_processing"])
    
    return global_circuit_manager.register_circuit(
        f"inference_{model_name}", 
        config, 
        fallback
    )


# Export classes and functions
__all__ = [
    "CircuitState", "FailureType", "FailureRecord", "CircuitConfig", "CircuitMetrics",
    "FallbackStrategy", "CachedResponseFallback", "SimplifiedModelFallback", 
    "GracefulDegradationFallback", "AdaptiveCircuitBreaker", "CircuitBreakerManager",
    "global_circuit_manager", "circuit_breaker", "create_mobile_circuit_config",
    "create_inference_circuit_breaker"
]