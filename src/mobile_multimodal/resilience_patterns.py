"""
Generation 2: Resilience Patterns Implementation
MAKE IT ROBUST - Circuit breakers, retries, bulkheads, and fault tolerance
"""

import asyncio
import enum
import functools
import logging
import random
import threading
import time
from collections import deque, defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError

# Enhanced logging setup
logger = logging.getLogger(__name__)
resilience_logger = logging.getLogger(f"{__name__}.resilience")
circuit_logger = logging.getLogger(f"{__name__}.circuit_breaker")

T = TypeVar('T')

class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class ResilienceStrategy(enum.Enum):
    """Resilience strategy types."""
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    RATE_LIMITER = "rate_limiter"
    FALLBACK = "fallback"

@dataclass
class RetryPolicy:
    """Retry configuration policy."""
    max_attempts: int = 3
    initial_delay: float = 0.1
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    jitter: bool = True
    exceptions: tuple = (Exception,)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: tuple = (Exception,)
    success_threshold: int = 3  # For half-open state
    timeout: float = 30.0
    
@dataclass
class BulkheadConfig:
    """Bulkhead isolation configuration."""
    max_concurrent_calls: int = 10
    max_queue_size: int = 100
    timeout: float = 30.0

@dataclass
class ResilienceMetrics:
    """Metrics for resilience patterns."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    circuit_breaker_activations: int = 0
    timeouts: int = 0
    fallback_executions: int = 0
    average_response_time: float = 0.0
    last_failure_time: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate()


class CircuitBreaker:
    """Circuit breaker pattern implementation with metrics."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker with configuration."""
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        self.lock = threading.RLock()
        self.metrics = ResilienceMetrics()
        
        circuit_logger.info(f"Circuit breaker initialized: threshold={config.failure_threshold}")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for circuit breaker."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.execute(func, *args, **kwargs)
        
        return wrapper
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        
        with self.lock:
            self.metrics.total_calls += 1
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    circuit_logger.info("Circuit breaker moved to HALF_OPEN")
                else:
                    self.metrics.failed_calls += 1
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
            
            elif self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    circuit_logger.info("Circuit breaker moved to CLOSED")
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            with self.lock:
                self.metrics.successful_calls += 1
                self._update_response_time(execution_time)
                
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                elif self.state == CircuitState.CLOSED:
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            if isinstance(e, self.config.expected_exception):
                with self.lock:
                    self.metrics.failed_calls += 1
                    self.metrics.last_failure_time = time.time()
                    self._update_response_time(execution_time)
                    
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.config.failure_threshold:
                        self.state = CircuitState.OPEN
                        self.metrics.circuit_breaker_activations += 1
                        circuit_logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise
    
    def _update_response_time(self, execution_time: float):
        """Update average response time."""
        if self.metrics.total_calls == 1:
            self.metrics.average_response_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * execution_time + (1 - alpha) * self.metrics.average_response_time
            )
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def get_metrics(self) -> ResilienceMetrics:
        """Get circuit breaker metrics."""
        return self.metrics
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            circuit_logger.info("Circuit breaker manually reset")


class RetryExecutor:
    """Retry pattern implementation with exponential backoff."""
    
    def __init__(self, policy: RetryPolicy):
        """Initialize retry executor with policy."""
        self.policy = policy
        self.metrics = ResilienceMetrics()
        
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for retry pattern."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.execute(func, *args, **kwargs)
        
        return wrapper
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        
        self.metrics.total_calls += 1
        last_exception = None
        
        for attempt in range(1, self.policy.max_attempts + 1):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.metrics.successful_calls += 1
                if attempt > 1:
                    self.metrics.retried_calls += 1
                
                self._update_response_time(execution_time)
                
                if attempt > 1:
                    resilience_logger.info(f"Function succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                last_exception = e
                
                if not isinstance(e, self.policy.exceptions):
                    # Exception not retryable
                    self.metrics.failed_calls += 1
                    self._update_response_time(execution_time)
                    raise
                
                if attempt == self.policy.max_attempts:
                    # Last attempt failed
                    self.metrics.failed_calls += 1
                    self._update_response_time(execution_time)
                    resilience_logger.error(f"All {self.policy.max_attempts} retry attempts failed")
                    raise
                
                # Calculate delay and wait
                delay = self.policy.calculate_delay(attempt)
                resilience_logger.warning(f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        # Should never reach here
        raise last_exception
    
    def _update_response_time(self, execution_time: float):
        """Update average response time."""
        if self.metrics.total_calls == 1:
            self.metrics.average_response_time = execution_time
        else:
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * execution_time + (1 - alpha) * self.metrics.average_response_time
            )
    
    async def execute_async(self, coro_func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with retry logic."""
        
        self.metrics.total_calls += 1
        last_exception = None
        
        for attempt in range(1, self.policy.max_attempts + 1):
            start_time = time.time()
            
            try:
                result = await coro_func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.metrics.successful_calls += 1
                if attempt > 1:
                    self.metrics.retried_calls += 1
                
                self._update_response_time(execution_time)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                last_exception = e
                
                if not isinstance(e, self.policy.exceptions):
                    self.metrics.failed_calls += 1
                    self._update_response_time(execution_time)
                    raise
                
                if attempt == self.policy.max_attempts:
                    self.metrics.failed_calls += 1
                    self._update_response_time(execution_time)
                    raise
                
                delay = self.policy.calculate_delay(attempt)
                resilience_logger.warning(f"Async attempt {attempt} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        raise last_exception


class Bulkhead:
    """Bulkhead isolation pattern implementation."""
    
    def __init__(self, config: BulkheadConfig):
        """Initialize bulkhead with configuration."""
        self.config = config
        self.semaphore = threading.Semaphore(config.max_concurrent_calls)
        self.queue = deque()
        self.active_calls = 0
        self.lock = threading.RLock()
        self.metrics = ResilienceMetrics()
        
        resilience_logger.info(f"Bulkhead initialized: max_concurrent={config.max_concurrent_calls}")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for bulkhead pattern."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.execute(func, *args, **kwargs)
        
        return wrapper
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with bulkhead isolation."""
        
        self.metrics.total_calls += 1
        
        # Check if queue is full
        with self.lock:
            if self.active_calls >= self.config.max_concurrent_calls:
                if len(self.queue) >= self.config.max_queue_size:
                    self.metrics.failed_calls += 1
                    raise BulkheadFullError("Bulkhead queue is full")
        
        # Acquire semaphore
        acquired = self.semaphore.acquire(timeout=self.config.timeout)
        if not acquired:
            self.metrics.timeouts += 1
            self.metrics.failed_calls += 1
            raise BulkheadTimeoutError("Bulkhead timeout")
        
        try:
            with self.lock:
                self.active_calls += 1
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            self.metrics.successful_calls += 1
            self._update_response_time(execution_time)
            
            return result
            
        except Exception as e:
            self.metrics.failed_calls += 1
            raise
            
        finally:
            with self.lock:
                self.active_calls -= 1
            self.semaphore.release()
    
    def _update_response_time(self, execution_time: float):
        """Update average response time."""
        if self.metrics.total_calls == 1:
            self.metrics.average_response_time = execution_time
        else:
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * execution_time + (1 - alpha) * self.metrics.average_response_time
            )
    
    def get_active_calls(self) -> int:
        """Get number of active calls."""
        return self.active_calls
    
    def get_metrics(self) -> ResilienceMetrics:
        """Get bulkhead metrics."""
        return self.metrics


class TimeoutGuard:
    """Timeout pattern implementation."""
    
    def __init__(self, timeout: float):
        """Initialize timeout guard."""
        self.timeout = timeout
        self.metrics = ResilienceMetrics()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for timeout pattern."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.execute(func, *args, **kwargs)
        
        return wrapper
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with timeout protection."""
        
        self.metrics.total_calls += 1
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            
            start_time = time.time()
            
            try:
                result = future.result(timeout=self.timeout)
                execution_time = time.time() - start_time
                
                self.metrics.successful_calls += 1
                self._update_response_time(execution_time)
                
                return result
                
            except FutureTimeoutError:
                execution_time = time.time() - start_time
                
                self.metrics.timeouts += 1
                self.metrics.failed_calls += 1
                self._update_response_time(execution_time)
                
                raise TimeoutError(f"Function execution timed out after {self.timeout}s")
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                self.metrics.failed_calls += 1
                self._update_response_time(execution_time)
                
                raise
    
    def _update_response_time(self, execution_time: float):
        """Update average response time."""
        if self.metrics.total_calls == 1:
            self.metrics.average_response_time = execution_time
        else:
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * execution_time + (1 - alpha) * self.metrics.average_response_time
            )


class FallbackExecutor:
    """Fallback pattern implementation."""
    
    def __init__(self, fallback_func: Callable, exceptions: tuple = (Exception,)):
        """Initialize fallback executor."""
        self.fallback_func = fallback_func
        self.exceptions = exceptions
        self.metrics = ResilienceMetrics()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for fallback pattern."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.execute(func, *args, **kwargs)
        
        return wrapper
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with fallback protection."""
        
        self.metrics.total_calls += 1
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            self.metrics.successful_calls += 1
            self._update_response_time(execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            if isinstance(e, self.exceptions):
                self.metrics.fallback_executions += 1
                resilience_logger.info(f"Executing fallback due to: {e}")
                
                # Execute fallback
                try:
                    fallback_result = self.fallback_func(*args, **kwargs)
                    self.metrics.successful_calls += 1
                    return fallback_result
                except Exception as fallback_error:
                    self.metrics.failed_calls += 1
                    resilience_logger.error(f"Fallback execution failed: {fallback_error}")
                    raise
            else:
                self.metrics.failed_calls += 1
                raise
            
            self._update_response_time(execution_time)
    
    def _update_response_time(self, execution_time: float):
        """Update average response time."""
        if self.metrics.total_calls == 1:
            self.metrics.average_response_time = execution_time
        else:
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * execution_time + (1 - alpha) * self.metrics.average_response_time
            )


class ResilienceManager:
    """Comprehensive resilience pattern manager."""
    
    def __init__(self):
        """Initialize resilience manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_executors: Dict[str, RetryExecutor] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.timeout_guards: Dict[str, TimeoutGuard] = {}
        self.fallback_executors: Dict[str, FallbackExecutor] = {}
        
        self.global_metrics = defaultdict(ResilienceMetrics)
        
        resilience_logger.info("ResilienceManager initialized")
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create and register circuit breaker."""
        circuit_breaker = CircuitBreaker(config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def create_retry_executor(self, name: str, policy: RetryPolicy) -> RetryExecutor:
        """Create and register retry executor."""
        retry_executor = RetryExecutor(policy)
        self.retry_executors[name] = retry_executor
        return retry_executor
    
    def create_bulkhead(self, name: str, config: BulkheadConfig) -> Bulkhead:
        """Create and register bulkhead."""
        bulkhead = Bulkhead(config)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    def create_timeout_guard(self, name: str, timeout: float) -> TimeoutGuard:
        """Create and register timeout guard."""
        timeout_guard = TimeoutGuard(timeout)
        self.timeout_guards[name] = timeout_guard
        return timeout_guard
    
    def create_fallback_executor(self, name: str, fallback_func: Callable, 
                               exceptions: tuple = (Exception,)) -> FallbackExecutor:
        """Create and register fallback executor."""
        fallback_executor = FallbackExecutor(fallback_func, exceptions)
        self.fallback_executors[name] = fallback_executor
        return fallback_executor
    
    def compose_resilience(self, *patterns) -> Callable:
        """Compose multiple resilience patterns."""
        
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                # Apply patterns in reverse order (innermost first)
                current_func = func
                
                for pattern in reversed(patterns):
                    current_func = pattern(current_func)
                
                return current_func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        
        health = {
            "circuit_breakers": {},
            "retry_executors": {},
            "bulkheads": {},
            "timeout_guards": {},
            "fallback_executors": {},
            "overall_health_score": 100
        }
        
        # Circuit breaker health
        for name, cb in self.circuit_breakers.items():
            metrics = cb.get_metrics()
            health["circuit_breakers"][name] = {
                "state": cb.get_state().value,
                "success_rate": metrics.success_rate(),
                "total_calls": metrics.total_calls,
                "activations": metrics.circuit_breaker_activations
            }
            
            # Reduce health score for open circuit breakers
            if cb.get_state() == CircuitState.OPEN:
                health["overall_health_score"] -= 20
        
        # Retry executor health
        for name, retry in self.retry_executors.items():
            metrics = retry.metrics
            health["retry_executors"][name] = {
                "success_rate": metrics.success_rate(),
                "retry_rate": metrics.retried_calls / max(metrics.total_calls, 1),
                "average_response_time": metrics.average_response_time
            }
        
        # Bulkhead health
        for name, bulkhead in self.bulkheads.items():
            metrics = bulkhead.get_metrics()
            health["bulkheads"][name] = {
                "active_calls": bulkhead.get_active_calls(),
                "max_concurrent": bulkhead.config.max_concurrent_calls,
                "utilization": bulkhead.get_active_calls() / bulkhead.config.max_concurrent_calls,
                "timeout_rate": metrics.timeouts / max(metrics.total_calls, 1)
            }
            
            # Reduce health score for high utilization
            utilization = bulkhead.get_active_calls() / bulkhead.config.max_concurrent_calls
            if utilization > 0.8:
                health["overall_health_score"] -= 10
        
        # Timeout guard health
        for name, timeout_guard in self.timeout_guards.items():
            metrics = timeout_guard.metrics
            health["timeout_guards"][name] = {
                "timeout_rate": metrics.timeouts / max(metrics.total_calls, 1),
                "average_response_time": metrics.average_response_time
            }
        
        # Fallback executor health
        for name, fallback in self.fallback_executors.items():
            metrics = fallback.metrics
            health["fallback_executors"][name] = {
                "fallback_rate": metrics.fallback_executions / max(metrics.total_calls, 1),
                "success_rate": metrics.success_rate()
            }
        
        return health


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

class BulkheadFullError(Exception):
    """Raised when bulkhead is full."""
    pass

class BulkheadTimeoutError(Exception):
    """Raised when bulkhead times out."""
    pass


# Convenience functions for common patterns
def resilient_function(retry_attempts: int = 3, timeout: float = 30.0, 
                      circuit_breaker_threshold: int = 5, 
                      fallback_func: Optional[Callable] = None):
    """Decorator for comprehensive resilience patterns."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        
        # Create patterns
        retry_policy = RetryPolicy(max_attempts=retry_attempts)
        retry_executor = RetryExecutor(retry_policy)
        
        circuit_config = CircuitBreakerConfig(failure_threshold=circuit_breaker_threshold)
        circuit_breaker = CircuitBreaker(circuit_config)
        
        timeout_guard = TimeoutGuard(timeout)
        
        patterns = [timeout_guard, circuit_breaker, retry_executor]
        
        if fallback_func:
            fallback_executor = FallbackExecutor(fallback_func)
            patterns.insert(0, fallback_executor)  # Fallback first
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_func = func
            
            # Apply patterns in reverse order
            for pattern in reversed(patterns):
                current_func = pattern(current_func)
            
            return current_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Example usage and testing
if __name__ == "__main__":
    print("Testing Resilience Patterns...")
    
    # Create resilience manager
    resilience_manager = ResilienceManager()
    
    # Test functions
    def flaky_function(should_fail: bool = False, delay: float = 0.1):
        """Test function that can fail or be slow."""
        time.sleep(delay)
        if should_fail:
            raise ValueError("Simulated failure")
        return "Success!"
    
    def fallback_function(*args, **kwargs):
        """Fallback function."""
        return "Fallback result"
    
    # Test retry pattern
    print("\n🔄 Testing Retry Pattern...")
    retry_policy = RetryPolicy(max_attempts=3, initial_delay=0.1)
    retry_executor = resilience_manager.create_retry_executor("test_retry", retry_policy)
    
    try:
        result = retry_executor.execute(flaky_function, should_fail=False)
        print(f"✅ Retry success: {result}")
    except Exception as e:
        print(f"❌ Retry failed: {e}")
    
    # Test circuit breaker pattern
    print("\n⚡ Testing Circuit Breaker Pattern...")
    circuit_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5.0)
    circuit_breaker = resilience_manager.create_circuit_breaker("test_circuit", circuit_config)
    
    # Trigger failures to open circuit
    for i in range(5):
        try:
            result = circuit_breaker.execute(flaky_function, should_fail=True)
        except Exception as e:
            print(f"   Attempt {i+1}: {type(e).__name__}")
    
    print(f"Circuit state: {circuit_breaker.get_state().value}")
    
    # Test bulkhead pattern
    print("\n🚧 Testing Bulkhead Pattern...")
    bulkhead_config = BulkheadConfig(max_concurrent_calls=2, timeout=5.0)
    bulkhead = resilience_manager.create_bulkhead("test_bulkhead", bulkhead_config)
    
    # Test concurrent execution
    import threading
    
    def concurrent_test():
        try:
            result = bulkhead.execute(flaky_function, delay=1.0)
            print(f"   Bulkhead call succeeded: {result}")
        except Exception as e:
            print(f"   Bulkhead call failed: {type(e).__name__}")
    
    threads = []
    for i in range(5):
        thread = threading.Thread(target=concurrent_test)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Test timeout pattern
    print("\n⏱️ Testing Timeout Pattern...")
    timeout_guard = resilience_manager.create_timeout_guard("test_timeout", timeout=0.5)
    
    try:
        result = timeout_guard.execute(flaky_function, delay=1.0)  # Will timeout
        print(f"✅ Timeout success: {result}")
    except Exception as e:
        print(f"❌ Timeout as expected: {type(e).__name__}")
    
    # Test fallback pattern
    print("\n🛡️ Testing Fallback Pattern...")
    fallback_executor = resilience_manager.create_fallback_executor(
        "test_fallback", fallback_function, (ValueError,)
    )
    
    try:
        result = fallback_executor.execute(flaky_function, should_fail=True)
        print(f"✅ Fallback success: {result}")
    except Exception as e:
        print(f"❌ Fallback failed: {e}")
    
    # Test composite resilience
    print("\n🔗 Testing Composite Resilience...")
    
    @resilient_function(retry_attempts=2, timeout=2.0, circuit_breaker_threshold=3, 
                       fallback_func=fallback_function)
    def protected_function():
        return flaky_function(should_fail=False)
    
    try:
        result = protected_function()
        print(f"✅ Protected function success: {result}")
    except Exception as e:
        print(f"❌ Protected function failed: {e}")
    
    # Display system health
    health = resilience_manager.get_system_health()
    print(f"\n📊 System Health Score: {health['overall_health_score']}")
    print(f"Circuit Breakers: {len(health['circuit_breakers'])}")
    print(f"Bulkheads: {len(health['bulkheads'])}")
    
    print("\n✅ Resilience Patterns test completed!")