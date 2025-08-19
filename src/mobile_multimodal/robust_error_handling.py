"""
Generation 2: MAKE IT ROBUST - Comprehensive Error Handling System
Advanced error handling, recovery patterns, and resilience mechanisms
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Type
from functools import wraps

# Enhanced logging setup
logger = logging.getLogger(__name__)
error_logger = logging.getLogger(f"{__name__}.errors")
recovery_logger = logging.getLogger(f"{__name__}.recovery")

class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for systematic handling."""
    VALIDATION = "validation"
    RESOURCE = "resource"
    NETWORK = "network"
    COMPUTATION = "computation"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    FAIL_FAST = "fail_fast"
    IGNORE = "ignore"

@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    stack_trace: Optional[str] = None
    system_state: Optional[Dict[str, Any]] = None
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_success: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RecoveryAction:
    """Recovery action definition."""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    timeout_seconds: float = 30.0
    fallback_function: Optional[Callable] = None
    condition_check: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RobustErrorHandler:
    """Comprehensive error handling system with recovery mechanisms."""
    
    def __init__(self, max_error_history: int = 1000):
        """Initialize robust error handler.
        
        Args:
            max_error_history: Maximum number of errors to keep in history
        """
        self.max_error_history = max_error_history
        self.error_history = []
        self.error_patterns = {}  # error_type -> pattern_info
        self.recovery_strategies = {}  # error_category -> RecoveryAction
        self.circuit_breakers = {}  # operation -> circuit_breaker_state
        
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_attempts": 0,
            "recovery_successes": 0,
            "circuit_breaker_activations": 0
        }
        
        # Default recovery strategies
        self._setup_default_recovery_strategies()
        
        logger.info("RobustErrorHandler initialized")
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies for different error categories."""
        self.recovery_strategies = {
            ErrorCategory.NETWORK: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=3,
                backoff_multiplier=2.0,
                timeout_seconds=10.0
            ),
            ErrorCategory.RESOURCE: RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADE,
                max_attempts=2,
                timeout_seconds=5.0
            ),
            ErrorCategory.COMPUTATION: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                max_attempts=1,
                timeout_seconds=15.0
            ),
            ErrorCategory.VALIDATION: RecoveryAction(
                strategy=RecoveryStrategy.FAIL_FAST,
                max_attempts=1
            ),
            ErrorCategory.SECURITY: RecoveryAction(
                strategy=RecoveryStrategy.FAIL_FAST,
                max_attempts=1
            ),
            ErrorCategory.CONFIGURATION: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                max_attempts=2
            ),
            ErrorCategory.EXTERNAL: RecoveryAction(
                strategy=RecoveryStrategy.CIRCUIT_BREAK,
                max_attempts=5,
                backoff_multiplier=1.5,
                timeout_seconds=20.0
            )
        }
    
    def handle_error(self, error: Exception, operation: str, 
                    user_id: str = None, request_id: str = None,
                    severity: ErrorSeverity = None,
                    category: ErrorCategory = None) -> ErrorContext:
        """Handle error with comprehensive context and recovery."""
        
        # Generate error ID
        error_id = f"err_{int(time.time() * 1000)}_{id(error)}"
        
        # Classify error if not provided
        if severity is None:
            severity = self._classify_error_severity(error)
        
        if category is None:
            category = self._classify_error_category(error)
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            operation=operation,
            user_id=user_id,
            request_id=request_id,
            stack_trace=traceback.format_exc(),
            system_state=self._capture_system_state()
        )
        
        # Log error based on severity
        self._log_error(error_context)
        
        # Update statistics
        self._update_error_stats(error_context)
        
        # Store in history
        self._store_error_history(error_context)
        
        # Detect error patterns
        self._analyze_error_patterns(error_context)
        
        # Check circuit breaker
        self._check_circuit_breaker(error_context)
        
        return error_context
    
    def attempt_recovery(self, error_context: ErrorContext, 
                        target_function: Callable, *args, **kwargs) -> Any:
        """Attempt error recovery using appropriate strategy."""
        
        recovery_action = self.recovery_strategies.get(
            error_context.category,
            RecoveryAction(strategy=RecoveryStrategy.FAIL_FAST)
        )
        
        error_context.recovery_attempted = True
        error_context.recovery_strategy = recovery_action.strategy
        
        recovery_logger.info(f"Attempting recovery for {error_context.error_id} using {recovery_action.strategy.value}")
        
        self.error_stats["recovery_attempts"] += 1
        
        try:
            if recovery_action.strategy == RecoveryStrategy.RETRY:
                result = self._retry_with_backoff(
                    target_function, recovery_action, *args, **kwargs
                )
            
            elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
                result = self._execute_fallback(
                    target_function, recovery_action, *args, **kwargs
                )
            
            elif recovery_action.strategy == RecoveryStrategy.GRACEFUL_DEGRADE:
                result = self._graceful_degradation(
                    target_function, recovery_action, *args, **kwargs
                )
            
            elif recovery_action.strategy == RecoveryStrategy.CIRCUIT_BREAK:
                result = self._circuit_breaker_recovery(
                    target_function, recovery_action, error_context, *args, **kwargs
                )
            
            elif recovery_action.strategy == RecoveryStrategy.FAIL_FAST:
                raise Exception(f"Fail-fast strategy: {error_context.error_message}")
            
            elif recovery_action.strategy == RecoveryStrategy.IGNORE:
                result = None
                recovery_logger.warning(f"Ignoring error {error_context.error_id}")
            
            else:
                raise Exception(f"Unknown recovery strategy: {recovery_action.strategy}")
            
            # Mark recovery as successful
            error_context.recovery_success = True
            self.error_stats["recovery_successes"] += 1
            
            recovery_logger.info(f"Recovery successful for {error_context.error_id}")
            return result
            
        except Exception as recovery_error:
            error_context.recovery_success = False
            recovery_logger.error(f"Recovery failed for {error_context.error_id}: {recovery_error}")
            raise recovery_error
    
    def _retry_with_backoff(self, target_function: Callable, 
                          recovery_action: RecoveryAction, *args, **kwargs) -> Any:
        """Retry function with exponential backoff."""
        
        for attempt in range(recovery_action.max_attempts):
            try:
                recovery_logger.debug(f"Retry attempt {attempt + 1}/{recovery_action.max_attempts}")
                
                result = target_function(*args, **kwargs)
                return result
                
            except Exception as e:
                if attempt == recovery_action.max_attempts - 1:
                    # Last attempt failed
                    raise e
                
                # Calculate backoff delay
                delay = (recovery_action.backoff_multiplier ** attempt) * 0.5
                recovery_logger.debug(f"Retry failed, waiting {delay:.2f}s before next attempt")
                time.sleep(delay)
        
        raise Exception("All retry attempts failed")
    
    def _execute_fallback(self, target_function: Callable,
                         recovery_action: RecoveryAction, *args, **kwargs) -> Any:
        """Execute fallback function if available."""
        
        if recovery_action.fallback_function:
            recovery_logger.debug("Executing fallback function")
            return recovery_action.fallback_function(*args, **kwargs)
        else:
            # Default fallback behavior
            recovery_logger.debug("No fallback function available, returning default response")
            return self._get_default_response()
    
    def _graceful_degradation(self, target_function: Callable,
                            recovery_action: RecoveryAction, *args, **kwargs) -> Any:
        """Implement graceful degradation strategy."""
        
        try:
            # Try with reduced functionality
            recovery_logger.debug("Attempting graceful degradation")
            
            # Modify arguments for simpler operation
            simplified_kwargs = kwargs.copy()
            
            # Remove complex parameters that might cause issues
            simplified_kwargs.pop('complex_mode', None)
            simplified_kwargs.pop('high_quality', None)
            
            return target_function(*args, **simplified_kwargs)
            
        except Exception as e:
            # Further degrade to minimal functionality
            recovery_logger.debug("Further degradation to minimal functionality")
            return self._get_minimal_response()
    
    def _circuit_breaker_recovery(self, target_function: Callable,
                                recovery_action: RecoveryAction,
                                error_context: ErrorContext, *args, **kwargs) -> Any:
        """Implement circuit breaker pattern."""
        
        operation = error_context.operation
        current_time = time.time()
        
        # Initialize circuit breaker state if not exists
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {
                "state": "closed",  # closed, open, half-open
                "failure_count": 0,
                "last_failure_time": 0,
                "timeout": 60  # seconds
            }
        
        cb_state = self.circuit_breakers[operation]
        
        # Check circuit breaker state
        if cb_state["state"] == "open":
            # Check if timeout has passed
            if current_time - cb_state["last_failure_time"] > cb_state["timeout"]:
                cb_state["state"] = "half-open"
                recovery_logger.info(f"Circuit breaker for {operation} moved to half-open")
            else:
                raise Exception(f"Circuit breaker open for {operation}")
        
        try:
            result = target_function(*args, **kwargs)
            
            # Success - reset circuit breaker
            cb_state["state"] = "closed"
            cb_state["failure_count"] = 0
            
            return result
            
        except Exception as e:
            # Failure - update circuit breaker
            cb_state["failure_count"] += 1
            cb_state["last_failure_time"] = current_time
            
            if cb_state["failure_count"] >= recovery_action.max_attempts:
                cb_state["state"] = "open"
                self.error_stats["circuit_breaker_activations"] += 1
                recovery_logger.warning(f"Circuit breaker opened for {operation}")
            
            raise e
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on error type and characteristics."""
        
        # Critical errors
        if isinstance(error, (MemoryError, SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(error, (FileNotFoundError, PermissionError, ConnectionError)):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if isinstance(error, (ValueError, TypeError, IndexError, KeyError)):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _classify_error_category(self, error: Exception) -> ErrorCategory:
        """Classify error category based on error type."""
        
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        
        if isinstance(error, (MemoryError, OSError)):
            return ErrorCategory.RESOURCE
        
        if isinstance(error, (ValueError, TypeError, AssertionError)):
            return ErrorCategory.VALIDATION
        
        if isinstance(error, (ArithmeticError, OverflowError)):
            return ErrorCategory.COMPUTATION
        
        if isinstance(error, (PermissionError, FileNotFoundError)):
            return ErrorCategory.SECURITY
        
        if isinstance(error, (ImportError, ModuleNotFoundError)):
            return ErrorCategory.CONFIGURATION
        
        return ErrorCategory.UNKNOWN
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context."""
        
        try:
            import psutil
            import platform
            
            return {
                "timestamp": time.time(),
                "platform": {
                    "system": platform.system(),
                    "python_version": platform.python_version(),
                    "architecture": platform.architecture()[0]
                },
                "memory": {
                    "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                    "percent_used": psutil.virtual_memory().percent
                },
                "cpu": {
                    "percent_used": psutil.cpu_percent(interval=0.1),
                    "core_count": psutil.cpu_count()
                },
                "disk": {
                    "free_space_gb": psutil.disk_usage('/').free / (1024 * 1024 * 1024),
                    "percent_used": psutil.disk_usage('/').percent
                }
            }
            
        except ImportError:
            # Fallback without psutil
            return {
                "timestamp": time.time(),
                "platform": {
                    "system": sys.platform,
                    "python_version": sys.version
                }
            }
        except Exception as e:
            return {
                "timestamp": time.time(),
                "error": f"Failed to capture system state: {e}"
            }
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level based on severity."""
        
        log_message = (
            f"Error {error_context.error_id}: {error_context.error_type} "
            f"in {error_context.operation} - {error_context.error_message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            error_logger.critical(log_message, extra={"error_context": asdict(error_context)})
        elif error_context.severity == ErrorSeverity.HIGH:
            error_logger.error(log_message, extra={"error_context": asdict(error_context)})
        elif error_context.severity == ErrorSeverity.MEDIUM:
            error_logger.warning(log_message, extra={"error_context": asdict(error_context)})
        else:
            error_logger.info(log_message, extra={"error_context": asdict(error_context)})
    
    def _update_error_stats(self, error_context: ErrorContext):
        """Update error statistics."""
        
        self.error_stats["total_errors"] += 1
        
        category_key = error_context.category.value
        self.error_stats["errors_by_category"][category_key] = (
            self.error_stats["errors_by_category"].get(category_key, 0) + 1
        )
        
        severity_key = error_context.severity.value
        self.error_stats["errors_by_severity"][severity_key] = (
            self.error_stats["errors_by_severity"].get(severity_key, 0) + 1
        )
    
    def _store_error_history(self, error_context: ErrorContext):
        """Store error in history with size management."""
        
        self.error_history.append(error_context)
        
        # Maintain maximum history size
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
    
    def _analyze_error_patterns(self, error_context: ErrorContext):
        """Analyze error patterns for proactive detection."""
        
        pattern_key = f"{error_context.error_type}_{error_context.category.value}"
        current_time = time.time()
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                "count": 0,
                "first_occurrence": current_time,
                "last_occurrence": current_time,
                "frequency": 0.0
            }
        
        pattern = self.error_patterns[pattern_key]
        pattern["count"] += 1
        pattern["last_occurrence"] = current_time
        
        # Calculate frequency (errors per hour)
        time_span = current_time - pattern["first_occurrence"]
        if time_span > 0:
            pattern["frequency"] = pattern["count"] / (time_span / 3600)
        
        # Check for concerning patterns
        if pattern["frequency"] > 10:  # More than 10 errors per hour
            logger.warning(f"High frequency error pattern detected: {pattern_key} "
                         f"({pattern['frequency']:.1f} errors/hour)")
    
    def _check_circuit_breaker(self, error_context: ErrorContext):
        """Check if circuit breaker should be activated."""
        
        operation = error_context.operation
        
        if operation in self.circuit_breakers:
            cb_state = self.circuit_breakers[operation]
            
            if cb_state["state"] == "closed" and cb_state["failure_count"] >= 5:
                cb_state["state"] = "open"
                cb_state["last_failure_time"] = time.time()
                self.error_stats["circuit_breaker_activations"] += 1
                logger.warning(f"Circuit breaker activated for {operation}")
    
    def _get_default_response(self) -> Dict[str, Any]:
        """Get default response for fallback scenarios."""
        return {
            "status": "fallback",
            "message": "Default response due to error recovery",
            "timestamp": time.time()
        }
    
    def _get_minimal_response(self) -> Dict[str, Any]:
        """Get minimal response for graceful degradation."""
        return {
            "status": "degraded",
            "message": "Minimal response due to service degradation",
            "timestamp": time.time()
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        
        # Calculate error rates
        recent_errors = [
            err for err in self.error_history
            if time.time() - err.timestamp < 3600  # Last hour
        ]
        
        recovery_rate = 0.0
        if self.error_stats["recovery_attempts"] > 0:
            recovery_rate = (
                self.error_stats["recovery_successes"] / 
                self.error_stats["recovery_attempts"]
            )
        
        return {
            **self.error_stats,
            "recovery_rate": recovery_rate,
            "recent_errors_count": len(recent_errors),
            "error_patterns": dict(self.error_patterns),
            "circuit_breaker_states": dict(self.circuit_breakers),
            "history_size": len(self.error_history)
        }
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Get detected error patterns."""
        return dict(self.error_patterns)
    
    def reset_circuit_breaker(self, operation: str) -> bool:
        """Manually reset circuit breaker for operation."""
        
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation] = {
                "state": "closed",
                "failure_count": 0,
                "last_failure_time": 0,
                "timeout": 60
            }
            logger.info(f"Circuit breaker reset for {operation}")
            return True
        
        return False
    
    def configure_recovery_strategy(self, category: ErrorCategory, 
                                  recovery_action: RecoveryAction):
        """Configure custom recovery strategy for error category."""
        
        self.recovery_strategies[category] = recovery_action
        logger.info(f"Updated recovery strategy for {category.value}")


# Decorator for robust error handling
def robust_operation(operation_name: str = None, 
                    error_handler: RobustErrorHandler = None,
                    severity: ErrorSeverity = None,
                    category: ErrorCategory = None,
                    auto_recover: bool = True):
    """Decorator for adding robust error handling to functions."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            handler = error_handler or getattr(wrapper, '_error_handler', None)
            
            if not handler:
                # Create default handler if none provided
                handler = RobustErrorHandler()
                wrapper._error_handler = handler
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                # Handle error
                error_context = handler.handle_error(
                    error=e,
                    operation=op_name,
                    severity=severity,
                    category=category
                )
                
                if auto_recover:
                    try:
                        # Attempt recovery
                        return handler.attempt_recovery(error_context, func, *args, **kwargs)
                    except Exception as recovery_error:
                        # Recovery failed, re-raise original error
                        raise e
                else:
                    # No recovery, re-raise error
                    raise e
        
        return wrapper
    return decorator


# Context manager for error handling
@contextmanager
def robust_context(operation_name: str, error_handler: RobustErrorHandler = None):
    """Context manager for robust error handling."""
    
    handler = error_handler or RobustErrorHandler()
    
    try:
        yield handler
    except Exception as e:
        error_context = handler.handle_error(
            error=e,
            operation=operation_name
        )
        
        # Re-raise with enhanced context
        raise Exception(f"Operation {operation_name} failed: {e}") from e


# Example usage and testing
if __name__ == "__main__":
    print("Testing Robust Error Handling System...")
    
    # Create error handler
    error_handler = RobustErrorHandler()
    
    # Test error classification
    try:
        raise ValueError("Test validation error")
    except Exception as e:
        error_context = error_handler.handle_error(e, "test_operation")
        print(f"✅ Error classified: {error_context.category.value} / {error_context.severity.value}")
    
    # Test decorator
    @robust_operation("test_function", error_handler=error_handler, auto_recover=True)
    def failing_function(should_fail: bool = True):
        if should_fail:
            raise ConnectionError("Network connection failed")
        return "Success!"
    
    # Test function with recovery
    try:
        result = failing_function(should_fail=True)
        print(f"✅ Function with recovery: {result}")
    except Exception as e:
        print(f"❌ Function failed: {e}")
    
    # Test context manager
    try:
        with robust_context("test_context", error_handler) as handler:
            raise FileNotFoundError("Config file missing")
    except Exception as e:
        print(f"✅ Context manager handled: {type(e).__name__}")
    
    # Display statistics
    stats = error_handler.get_error_statistics()
    print(f"\n📊 Error Statistics:")
    print(f"   Total errors: {stats['total_errors']}")
    print(f"   Recovery rate: {stats['recovery_rate']:.1%}")
    print(f"   Error patterns: {len(stats['error_patterns'])}")
    
    print("\n✅ Robust Error Handling System test completed!")