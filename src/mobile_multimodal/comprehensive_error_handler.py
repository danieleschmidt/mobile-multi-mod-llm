"""Comprehensive Error Handling System for Mobile Multi-Modal LLM.

This module implements enterprise-grade error handling including:
1. Hierarchical error classification and recovery strategies  
2. Automatic retry mechanisms with exponential backoff
3. Circuit breaker patterns for fault tolerance
4. Dead letter queues for failed operations
5. Comprehensive error analytics and reporting
6. Real-time error monitoring and alerting
"""

import asyncio
import functools
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
from collections import defaultdict, deque
import inspect
import sys
import warnings

import numpy as np

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"

class ErrorCategory(Enum):
    """Error categories for classification."""
    SYSTEM = "system"
    NETWORK = "network"
    MEMORY = "memory"
    COMPUTATION = "computation"
    VALIDATION = "validation"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    CORRUPTION = "corruption"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    ESCALATION = "escalation"
    IGNORE = "ignore"
    TERMINATE = "terminate"

@dataclass
class ErrorContext:
    """Context information for error occurrence."""
    function_name: str
    module_name: str
    line_number: int
    arguments: Dict[str, Any]
    local_variables: Dict[str, Any]
    stack_trace: str
    system_state: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict:
        return {
            "function_name": self.function_name,
            "module_name": self.module_name,
            "line_number": self.line_number,
            "arguments": self._serialize_args(self.arguments),
            "local_variables": self._serialize_args(self.local_variables),
            "stack_trace": self.stack_trace,
            "system_state": self.system_state,
            "timestamp": self.timestamp
        }
    
    def _serialize_args(self, args: Dict[str, Any]) -> Dict[str, str]:
        """Serialize arguments for JSON storage."""
        serialized = {}
        for key, value in args.items():
            try:
                if isinstance(value, (str, int, float, bool, type(None))):
                    serialized[key] = value
                elif isinstance(value, (list, dict, tuple)):
                    serialized[key] = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                else:
                    serialized[key] = f"<{type(value).__name__}>"
            except:
                serialized[key] = "<unserializable>"
        return serialized

@dataclass
class ErrorRecord:
    """Complete error record with context and metadata."""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    recovery_strategy: RecoveryStrategy
    retry_count: int = 0
    max_retries: int = 3
    resolved: bool = False
    resolution_time: Optional[float] = None
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict(),
            "recovery_strategy": self.recovery_strategy.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time,
            "impact_assessment": self.impact_assessment
        }

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    
    # State tracking
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        else:  # HALF_OPEN
            return self.half_open_calls < self.half_open_max_calls
    
    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN


class ErrorClassifier:
    """Classifies errors and determines recovery strategies."""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
        self.recovery_strategies = self._build_recovery_strategies()
    
    def _build_classification_rules(self) -> Dict[str, Tuple[ErrorCategory, ErrorSeverity]]:
        """Build error classification rules."""
        return {
            # System errors
            "MemoryError": (ErrorCategory.MEMORY, ErrorSeverity.CRITICAL),
            "SystemExit": (ErrorCategory.SYSTEM, ErrorSeverity.FATAL),
            "KeyboardInterrupt": (ErrorCategory.SYSTEM, ErrorSeverity.WARNING),
            
            # Network errors
            "ConnectionError": (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
            "TimeoutError": (ErrorCategory.TIMEOUT, ErrorSeverity.ERROR),
            "ConnectionRefusedError": (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
            "ConnectionResetError": (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
            
            # Validation errors
            "ValueError": (ErrorCategory.VALIDATION, ErrorSeverity.ERROR),
            "TypeError": (ErrorCategory.VALIDATION, ErrorSeverity.ERROR),
            "AssertionError": (ErrorCategory.VALIDATION, ErrorSeverity.ERROR),
            
            # Resource errors
            "FileNotFoundError": (ErrorCategory.RESOURCE, ErrorSeverity.ERROR),
            "PermissionError": (ErrorCategory.RESOURCE, ErrorSeverity.ERROR),
            "OSError": (ErrorCategory.SYSTEM, ErrorSeverity.ERROR),
            
            # Computation errors
            "ZeroDivisionError": (ErrorCategory.COMPUTATION, ErrorSeverity.ERROR),
            "OverflowError": (ErrorCategory.COMPUTATION, ErrorSeverity.ERROR),
            "FloatingPointError": (ErrorCategory.COMPUTATION, ErrorSeverity.ERROR),
            
            # Configuration errors
            "ConfigurationError": (ErrorCategory.CONFIGURATION, ErrorSeverity.ERROR),
            "ImportError": (ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL),
            "ModuleNotFoundError": (ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL),
            
            # Security errors  
            "SecurityError": (ErrorCategory.SECURITY, ErrorSeverity.CRITICAL),
            "PermissionError": (ErrorCategory.SECURITY, ErrorSeverity.CRITICAL)
        }
    
    def _build_recovery_strategies(self) -> Dict[Tuple[ErrorCategory, ErrorSeverity], RecoveryStrategy]:
        """Build recovery strategy mapping."""
        return {
            # Memory errors - try to recover gracefully
            (ErrorCategory.MEMORY, ErrorSeverity.CRITICAL): RecoveryStrategy.GRACEFUL_DEGRADATION,
            (ErrorCategory.MEMORY, ErrorSeverity.ERROR): RecoveryStrategy.FALLBACK,
            
            # Network errors - retry with backoff
            (ErrorCategory.NETWORK, ErrorSeverity.ERROR): RecoveryStrategy.RETRY,
            (ErrorCategory.TIMEOUT, ErrorSeverity.ERROR): RecoveryStrategy.RETRY,
            
            # Validation errors - usually permanent
            (ErrorCategory.VALIDATION, ErrorSeverity.ERROR): RecoveryStrategy.ESCALATION,
            
            # System errors - depends on severity
            (ErrorCategory.SYSTEM, ErrorSeverity.FATAL): RecoveryStrategy.TERMINATE,
            (ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL): RecoveryStrategy.CIRCUIT_BREAKER,
            (ErrorCategory.SYSTEM, ErrorSeverity.ERROR): RecoveryStrategy.FALLBACK,
            (ErrorCategory.SYSTEM, ErrorSeverity.WARNING): RecoveryStrategy.IGNORE,
            
            # Resource errors - try alternative sources
            (ErrorCategory.RESOURCE, ErrorSeverity.ERROR): RecoveryStrategy.FALLBACK,
            
            # Computation errors - fallback to safer alternatives
            (ErrorCategory.COMPUTATION, ErrorSeverity.ERROR): RecoveryStrategy.FALLBACK,
            
            # Configuration errors - usually require intervention
            (ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL): RecoveryStrategy.ESCALATION,
            (ErrorCategory.CONFIGURATION, ErrorSeverity.ERROR): RecoveryStrategy.FALLBACK,
            
            # Security errors - terminate or escalate
            (ErrorCategory.SECURITY, ErrorSeverity.CRITICAL): RecoveryStrategy.TERMINATE
        }
    
    def classify_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify an error and determine its category and severity."""
        error_type = type(error).__name__
        
        # Check direct mapping
        if error_type in self.classification_rules:
            return self.classification_rules[error_type]
        
        # Pattern-based classification
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ["memory", "allocation", "out of memory"]):
            return ErrorCategory.MEMORY, ErrorSeverity.CRITICAL
        elif any(keyword in error_message for keyword in ["network", "connection", "socket"]):
            return ErrorCategory.NETWORK, ErrorSeverity.ERROR
        elif any(keyword in error_message for keyword in ["timeout", "deadline"]):
            return ErrorCategory.TIMEOUT, ErrorSeverity.ERROR
        elif any(keyword in error_message for keyword in ["permission", "access", "denied"]):
            return ErrorCategory.SECURITY, ErrorSeverity.CRITICAL
        elif any(keyword in error_message for keyword in ["config", "setting", "parameter"]):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.ERROR
        else:
            # Default classification
            return ErrorCategory.SYSTEM, ErrorSeverity.ERROR
    
    def get_recovery_strategy(self, category: ErrorCategory, severity: ErrorSeverity) -> RecoveryStrategy:
        """Get appropriate recovery strategy for error classification."""
        strategy_key = (category, severity)
        return self.recovery_strategies.get(strategy_key, RecoveryStrategy.ESCALATION)


class DeadLetterQueue:
    """Dead letter queue for failed operations."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)
        self.processed_count = 0
        self.failed_count = 0
        
    def add_failed_operation(self, operation_data: Dict, error_record: ErrorRecord):
        """Add failed operation to dead letter queue."""
        dead_letter_entry = {
            "timestamp": time.time(),
            "operation_data": operation_data,
            "error_record": error_record.to_dict(),
            "retry_count": error_record.retry_count,
            "status": "queued"
        }
        
        self.queue.append(dead_letter_entry)
        logger.warning(f"Operation added to dead letter queue: {error_record.error_id}")
    
    def process_dead_letters(self, processor_func: Callable) -> Dict[str, int]:
        """Process items in dead letter queue."""
        results = {"processed": 0, "failed": 0, "skipped": 0}
        
        # Process a batch of items
        batch_size = min(10, len(self.queue))
        
        for _ in range(batch_size):
            if not self.queue:
                break
                
            entry = self.queue.popleft()
            
            try:
                # Try to process the failed operation
                success = processor_func(entry["operation_data"])
                
                if success:
                    entry["status"] = "processed"
                    results["processed"] += 1
                    self.processed_count += 1
                else:
                    entry["status"] = "failed_again"
                    results["failed"] += 1
                    self.failed_count += 1
                    # Re-add to queue if not exceeded max retries
                    if entry["retry_count"] < 3:
                        entry["retry_count"] += 1
                        self.queue.append(entry)
                        
            except Exception as e:
                logger.error(f"Error processing dead letter: {e}")
                entry["status"] = "error"
                results["failed"] += 1
                self.failed_count += 1
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dead letter queue statistics."""
        return {
            "queue_size": len(self.queue),
            "max_size": self.max_size,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "utilization": len(self.queue) / self.max_size,
            "oldest_entry": min([entry["timestamp"] for entry in self.queue], default=0),
            "status_distribution": self._get_status_distribution()
        }
    
    def _get_status_distribution(self) -> Dict[str, int]:
        """Get distribution of entry statuses."""
        distribution = defaultdict(int)
        for entry in self.queue:
            distribution[entry["status"]] += 1
        return dict(distribution)


class ErrorAnalytics:
    """Analytics engine for error patterns and trends."""
    
    def __init__(self, window_size_hours: int = 24):
        self.window_size_hours = window_size_hours
        self.error_history = deque(maxlen=10000)
        self.pattern_cache = {}
        self.last_analysis_time = 0
        
    def record_error(self, error_record: ErrorRecord):
        """Record error for analytics."""
        self.error_history.append(error_record)
        
        # Clear cache if needed
        if time.time() - self.last_analysis_time > 300:  # 5 minutes
            self.pattern_cache.clear()
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns and trends."""
        current_time = time.time()
        window_start = current_time - (self.window_size_hours * 3600)
        
        # Filter errors within time window
        recent_errors = [
            error for error in self.error_history
            if error.context.timestamp >= window_start
        ]
        
        if not recent_errors:
            return {"message": "No recent errors to analyze"}
        
        analysis = {
            "time_window_hours": self.window_size_hours,
            "total_errors": len(recent_errors),
            "error_rate_per_hour": len(recent_errors) / self.window_size_hours,
            "severity_distribution": self._analyze_severity_distribution(recent_errors),
            "category_distribution": self._analyze_category_distribution(recent_errors),
            "top_error_types": self._analyze_error_types(recent_errors),
            "recovery_success_rate": self._analyze_recovery_success(recent_errors),
            "temporal_patterns": self._analyze_temporal_patterns(recent_errors),
            "correlation_analysis": self._analyze_error_correlations(recent_errors),
            "prediction_insights": self._predict_error_trends(recent_errors)
        }
        
        self.last_analysis_time = current_time
        return analysis
    
    def _analyze_severity_distribution(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Analyze distribution of error severities."""
        distribution = defaultdict(int)
        for error in errors:
            distribution[error.severity.value] += 1
        
        total = len(errors)
        return {
            "counts": dict(distribution),
            "percentages": {k: (v / total) * 100 for k, v in distribution.items()},
            "critical_ratio": (distribution[ErrorSeverity.CRITICAL.value] + 
                             distribution[ErrorSeverity.FATAL.value]) / total
        }
    
    def _analyze_category_distribution(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Analyze distribution of error categories."""
        distribution = defaultdict(int)
        for error in errors:
            distribution[error.category.value] += 1
        
        total = len(errors)
        return {
            "counts": dict(distribution),
            "percentages": {k: (v / total) * 100 for k, v in distribution.items()},
            "top_categories": sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _analyze_error_types(self, errors: List[ErrorRecord]) -> List[Dict[str, Any]]:
        """Analyze most common error types."""
        type_counts = defaultdict(int)
        type_details = defaultdict(list)
        
        for error in errors:
            type_counts[error.error_type] += 1
            type_details[error.error_type].append({
                "severity": error.severity.value,
                "category": error.category.value,
                "resolved": error.resolved
            })
        
        top_types = []
        for error_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            details = type_details[error_type]
            top_types.append({
                "error_type": error_type,
                "count": count,
                "resolution_rate": sum(1 for d in details if d["resolved"]) / len(details),
                "severity_breakdown": self._get_severity_breakdown(details),
                "category_breakdown": self._get_category_breakdown(details)
            })
        
        return top_types
    
    def _analyze_recovery_success(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Analyze recovery success rates by strategy."""
        strategy_stats = defaultdict(lambda: {"total": 0, "resolved": 0})
        
        for error in errors:
            strategy = error.recovery_strategy.value
            strategy_stats[strategy]["total"] += 1
            if error.resolved:
                strategy_stats[strategy]["resolved"] += 1
        
        success_rates = {}
        for strategy, stats in strategy_stats.items():
            success_rates[strategy] = {
                "total_attempts": stats["total"],
                "successful_recoveries": stats["resolved"],
                "success_rate": stats["resolved"] / stats["total"] if stats["total"] > 0 else 0
            }
        
        return success_rates
    
    def _analyze_temporal_patterns(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Analyze temporal patterns in errors."""
        hourly_distribution = defaultdict(int)
        
        for error in errors:
            hour = datetime.fromtimestamp(error.context.timestamp).hour
            hourly_distribution[hour] += 1
        
        # Find peak error hours
        sorted_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "hourly_distribution": dict(hourly_distribution),
            "peak_error_hours": sorted_hours[:3],
            "error_clustering": self._detect_error_clustering(errors)
        }
    
    def _analyze_error_correlations(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Analyze correlations between different error types."""
        # Simple correlation analysis
        correlations = {}
        
        # Check for errors that occur close in time (within 5 minutes)
        time_window = 300  # 5 minutes
        
        for i, error1 in enumerate(errors):
            for j, error2 in enumerate(errors[i+1:], i+1):
                time_diff = abs(error1.context.timestamp - error2.context.timestamp)
                
                if time_diff <= time_window and error1.error_type != error2.error_type:
                    pair = tuple(sorted([error1.error_type, error2.error_type]))
                    correlations[pair] = correlations.get(pair, 0) + 1
        
        # Sort by correlation strength
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "time_window_seconds": time_window,
            "correlated_error_pairs": sorted_correlations[:10],
            "total_correlations_found": len(correlations)
        }
    
    def _predict_error_trends(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Simple error trend prediction."""
        if len(errors) < 10:
            return {"message": "Insufficient data for trend analysis"}
        
        # Split errors into time buckets for trend analysis
        bucket_size_minutes = 60  # 1 hour buckets
        buckets = defaultdict(int)
        
        min_time = min(error.context.timestamp for error in errors)
        
        for error in errors:
            bucket = int((error.context.timestamp - min_time) / (bucket_size_minutes * 60))
            buckets[bucket] += 1
        
        # Simple linear trend calculation
        if len(buckets) >= 3:
            bucket_times = sorted(buckets.keys())
            error_counts = [buckets[t] for t in bucket_times]
            
            # Calculate trend
            n = len(error_counts)
            sum_x = sum(range(n))
            sum_y = sum(error_counts)
            sum_xy = sum(i * count for i, count in enumerate(error_counts))
            sum_x2 = sum(i * i for i in range(n))
            
            trend_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
            
            return {
                "trend_direction": "increasing" if trend_slope > 0.1 else "decreasing" if trend_slope < -0.1 else "stable",
                "trend_slope": trend_slope,
                "recent_error_rate": error_counts[-1] if error_counts else 0,
                "prediction": "Error rate may increase" if trend_slope > 0.1 else "Error rate may decrease" if trend_slope < -0.1 else "Error rate appears stable"
            }
        
        return {"message": "Insufficient data for trend prediction"}
    
    def _get_severity_breakdown(self, details: List[Dict]) -> Dict[str, int]:
        """Get severity breakdown for error details."""
        breakdown = defaultdict(int)
        for detail in details:
            breakdown[detail["severity"]] += 1
        return dict(breakdown)
    
    def _get_category_breakdown(self, details: List[Dict]) -> Dict[str, int]:
        """Get category breakdown for error details."""
        breakdown = defaultdict(int)
        for detail in details:
            breakdown[detail["category"]] += 1
        return dict(breakdown)
    
    def _detect_error_clustering(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Detect if errors are clustering in time."""
        if len(errors) < 5:
            return {"clustering_detected": False}
        
        timestamps = [error.context.timestamp for error in errors]
        timestamps.sort()
        
        # Calculate inter-arrival times
        inter_arrivals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not inter_arrivals:
            return {"clustering_detected": False}
        
        avg_interval = np.mean(inter_arrivals)
        std_interval = np.std(inter_arrivals)
        
        # Simple clustering detection: many short intervals
        short_intervals = [interval for interval in inter_arrivals if interval < avg_interval - std_interval]
        clustering_ratio = len(short_intervals) / len(inter_arrivals)
        
        return {
            "clustering_detected": clustering_ratio > 0.3,
            "clustering_ratio": clustering_ratio,
            "avg_interval_seconds": avg_interval,
            "short_interval_count": len(short_intervals)
        }


class ComprehensiveErrorHandler:
    """Main comprehensive error handling system."""
    
    def __init__(self):
        self.error_classifier = ErrorClassifier()
        self.circuit_breakers = {}
        self.dead_letter_queue = DeadLetterQueue()
        self.error_analytics = ErrorAnalytics()
        self.error_records = {}
        
        # Configuration
        self.max_retries = 3
        self.base_retry_delay = 1.0
        self.max_retry_delay = 60.0
        self.backoff_multiplier = 2.0
        
        # Statistics
        self.total_errors_handled = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        
        logger.info("Comprehensive Error Handler initialized")
    
    def handle_error(self, error: Exception, context_data: Dict[str, Any] = None,
                    function_name: str = None, retry_count: int = 0) -> Any:
        """Main error handling entry point."""
        # Generate error context
        context = self._capture_error_context(error, context_data, function_name)
        
        # Classify error
        category, severity = self.error_classifier.classify_error(error)
        recovery_strategy = self.error_classifier.get_recovery_strategy(category, severity)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=self._generate_error_id(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            context=context,
            recovery_strategy=recovery_strategy,
            retry_count=retry_count,
            max_retries=self.max_retries
        )
        
        # Store error record
        self.error_records[error_record.error_id] = error_record
        self.error_analytics.record_error(error_record)
        self.total_errors_handled += 1
        
        # Log error
        self._log_error(error_record)
        
        # Execute recovery strategy
        try:
            result = self._execute_recovery_strategy(error_record, context_data)
            if result is not None:
                error_record.resolved = True
                error_record.resolution_time = time.time()
                self.successful_recoveries += 1
                return result
        except Exception as recovery_error:
            logger.error(f"Recovery strategy failed: {recovery_error}")
            self.failed_recoveries += 1
        
        # If recovery failed, add to dead letter queue
        if context_data:
            self.dead_letter_queue.add_failed_operation(context_data, error_record)
        
        # Re-raise if no recovery possible
        raise error
    
    def _capture_error_context(self, error: Exception, context_data: Dict[str, Any] = None,
                              function_name: str = None) -> ErrorContext:
        """Capture comprehensive error context."""
        # Get current frame information
        frame = inspect.currentframe()
        
        # Navigate up the stack to find the original error location
        while frame and frame.f_code.co_filename == __file__:
            frame = frame.f_back
        
        if frame:
            filename = frame.f_code.co_filename
            line_number = frame.f_lineno
            actual_function_name = frame.f_code.co_name
            local_vars = dict(frame.f_locals)
        else:
            filename = "unknown"
            line_number = 0
            actual_function_name = "unknown"
            local_vars = {}
        
        # Use provided function name or detected one
        func_name = function_name or actual_function_name
        
        # Get stack trace
        stack_trace = traceback.format_exc()
        
        # Capture system state
        system_state = self._capture_system_state()
        
        return ErrorContext(
            function_name=func_name,
            module_name=filename,
            line_number=line_number,
            arguments=context_data or {},
            local_variables=local_vars,
            stack_trace=stack_trace,
            system_state=system_state,
            timestamp=time.time()
        )
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "thread_count": process.num_threads(),
                "open_files": len(process.open_files()),
                "system_load": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except ImportError:
            return {
                "memory_usage_mb": "unavailable",
                "cpu_percent": "unavailable",
                "thread_count": threading.active_count(),
                "timestamp": time.time()
            }
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        import hashlib
        import secrets
        timestamp = str(time.time())
        random_data = secrets.token_hex(8)
        error_id = hashlib.md5(f"{timestamp}_{random_data}".encode()).hexdigest()[:12]
        return f"ERR_{error_id}"
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level."""
        log_message = (f"Error {error_record.error_id}: {error_record.error_type} - "
                      f"{error_record.error_message} "
                      f"(Category: {error_record.category.value}, "
                      f"Severity: {error_record.severity.value}, "
                      f"Strategy: {error_record.recovery_strategy.value})")
        
        if error_record.severity == ErrorSeverity.FATAL:
            logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error_record.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _execute_recovery_strategy(self, error_record: ErrorRecord, context_data: Dict[str, Any] = None) -> Any:
        """Execute the determined recovery strategy."""
        strategy = error_record.recovery_strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return self._execute_retry_strategy(error_record, context_data)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback_strategy(error_record, context_data)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._execute_graceful_degradation(error_record, context_data)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._execute_circuit_breaker_strategy(error_record, context_data)
        elif strategy == RecoveryStrategy.IGNORE:
            return self._execute_ignore_strategy(error_record, context_data)
        elif strategy == RecoveryStrategy.ESCALATION:
            return self._execute_escalation_strategy(error_record, context_data)
        elif strategy == RecoveryStrategy.TERMINATE:
            return self._execute_terminate_strategy(error_record, context_data)
        
        return None
    
    def _execute_retry_strategy(self, error_record: ErrorRecord, context_data: Dict[str, Any] = None) -> Any:
        """Execute retry strategy with exponential backoff."""
        if error_record.retry_count >= error_record.max_retries:
            logger.warning(f"Max retries ({error_record.max_retries}) exceeded for error {error_record.error_id}")
            return None
        
        # Calculate retry delay with exponential backoff
        delay = min(
            self.base_retry_delay * (self.backoff_multiplier ** error_record.retry_count),
            self.max_retry_delay
        )
        
        logger.info(f"Retrying operation in {delay:.2f}s (attempt {error_record.retry_count + 1})")
        time.sleep(delay)
        
        # Increment retry count
        error_record.retry_count += 1
        
        # Try to re-execute the original operation
        if context_data and "retry_function" in context_data:
            try:
                return context_data["retry_function"]()
            except Exception as retry_error:
                return self.handle_error(retry_error, context_data, retry_count=error_record.retry_count)
        
        return None
    
    def _execute_fallback_strategy(self, error_record: ErrorRecord, context_data: Dict[str, Any] = None) -> Any:
        """Execute fallback strategy."""
        logger.info(f"Executing fallback strategy for error {error_record.error_id}")
        
        if context_data and "fallback_function" in context_data:
            try:
                return context_data["fallback_function"]()
            except Exception as fallback_error:
                logger.error(f"Fallback strategy failed: {fallback_error}")
                return None
        
        # Generic fallback responses based on error category
        if error_record.category == ErrorCategory.COMPUTATION:
            return {"result": "fallback_computation_result", "degraded": True}
        elif error_record.category == ErrorCategory.NETWORK:
            return {"result": "cached_or_default_response", "degraded": True}
        elif error_record.category == ErrorCategory.MEMORY:
            return {"result": "simplified_response", "degraded": True}
        
        return {"result": "generic_fallback", "degraded": True}
    
    def _execute_graceful_degradation(self, error_record: ErrorRecord, context_data: Dict[str, Any] = None) -> Any:
        """Execute graceful degradation strategy."""
        logger.info(f"Executing graceful degradation for error {error_record.error_id}")
        
        # Return simplified/degraded response
        return {
            "result": "degraded_service_response",
            "degradation_reason": error_record.error_message,
            "service_level": "reduced",
            "error_id": error_record.error_id
        }
    
    def _execute_circuit_breaker_strategy(self, error_record: ErrorRecord, context_data: Dict[str, Any] = None) -> Any:
        """Execute circuit breaker strategy."""
        operation_name = context_data.get("operation_name", "default") if context_data else "default"
        
        # Get or create circuit breaker
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(name=operation_name)
        
        circuit_breaker = self.circuit_breakers[operation_name]
        circuit_breaker.record_failure()
        
        logger.warning(f"Circuit breaker {operation_name} state: {circuit_breaker.state.value}")
        
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            return {
                "result": "circuit_breaker_open",
                "retry_after_seconds": circuit_breaker.recovery_timeout,
                "error_id": error_record.error_id
            }
        
        return None
    
    def _execute_ignore_strategy(self, error_record: ErrorRecord, context_data: Dict[str, Any] = None) -> Any:
        """Execute ignore strategy (log and continue)."""
        logger.info(f"Ignoring error {error_record.error_id} as per strategy")
        return {"result": "error_ignored", "continued": True}
    
    def _execute_escalation_strategy(self, error_record: ErrorRecord, context_data: Dict[str, Any] = None) -> Any:
        """Execute escalation strategy (alert administrators)."""
        logger.critical(f"ESCALATION: Error {error_record.error_id} requires administrator attention")
        
        # In real implementation, this would send alerts via email, Slack, etc.
        escalation_data = {
            "error_id": error_record.error_id,
            "error_type": error_record.error_type,
            "severity": error_record.severity.value,
            "category": error_record.category.value,
            "context": error_record.context.to_dict(),
            "escalation_time": time.time()
        }
        
        # Mock escalation notification
        logger.critical(f"ALERT SENT TO ADMINISTRATORS: {json.dumps(escalation_data, indent=2)}")
        
        return None
    
    def _execute_terminate_strategy(self, error_record: ErrorRecord, context_data: Dict[str, Any] = None) -> Any:
        """Execute terminate strategy (controlled shutdown)."""
        logger.critical(f"TERMINATING due to critical error {error_record.error_id}")
        
        # In real implementation, this would perform clean shutdown
        # For demo purposes, we'll just log and raise
        termination_message = (f"System termination triggered by error {error_record.error_id}: "
                             f"{error_record.error_message}")
        
        logger.critical(termination_message)
        raise SystemExit(termination_message)
    
    def with_error_handling(self, retry_function: Callable = None, 
                           fallback_function: Callable = None,
                           operation_name: str = None):
        """Decorator for adding comprehensive error handling to functions."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                context_data = {
                    "function_name": func.__name__,
                    "args": args,
                    "kwargs": kwargs,
                    "retry_function": retry_function,
                    "fallback_function": fallback_function,
                    "operation_name": operation_name or func.__name__
                }
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return self.handle_error(e, context_data, func.__name__)
            
            return wrapper
        return decorator
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error handling statistics."""
        return {
            "total_errors_handled": self.total_errors_handled,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "recovery_success_rate": self.successful_recoveries / max(self.total_errors_handled, 1),
            "active_circuit_breakers": len(self.circuit_breakers),
            "circuit_breaker_states": {
                name: cb.state.value for name, cb in self.circuit_breakers.items()
            },
            "dead_letter_queue": self.dead_letter_queue.get_statistics(),
            "error_analytics": self.error_analytics.analyze_error_patterns(),
            "recent_errors": len([
                err for err in self.error_records.values()
                if time.time() - err.context.timestamp < 3600  # Last hour
            ])
        }
    
    def export_error_report(self, filepath: str, include_analytics: bool = True):
        """Export comprehensive error report."""
        report = {
            "report_timestamp": time.time(),
            "error_statistics": self.get_error_statistics(),
            "error_records": {
                error_id: record.to_dict()
                for error_id, record in self.error_records.items()
            },
            "configuration": {
                "max_retries": self.max_retries,
                "base_retry_delay": self.base_retry_delay,
                "max_retry_delay": self.max_retry_delay,
                "backoff_multiplier": self.backoff_multiplier
            }
        }
        
        if include_analytics:
            report["analytics"] = self.error_analytics.analyze_error_patterns()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Error report exported to {filepath}")


# Factory function
def create_error_handler() -> ComprehensiveErrorHandler:
    """Create comprehensive error handler."""
    return ComprehensiveErrorHandler()


# Example usage and testing
if __name__ == "__main__":
    print("🛡️ Comprehensive Error Handling System - Mobile Multi-Modal LLM")
    
    # Create error handler
    error_handler = create_error_handler()
    
    # Test error handling with different error types
    test_errors = [
        ValueError("Invalid input value"),
        MemoryError("Out of memory"),
        ConnectionError("Network connection failed"),
        TimeoutError("Operation timed out"),
    ]
    
    # Test each error type
    for i, error in enumerate(test_errors):
        print(f"\n🔍 Testing error {i+1}: {type(error).__name__}")
        
        try:
            error_handler.handle_error(
                error, 
                context_data={
                    "operation_name": f"test_operation_{i}",
                    "test_data": f"test_value_{i}",
                    "fallback_function": lambda: f"fallback_result_{i}"
                },
                function_name=f"test_function_{i}"
            )
        except Exception as e:
            print(f"  Final result: {type(e).__name__}: {e}")
    
    # Test decorator
    @error_handler.with_error_handling(
        fallback_function=lambda: "decorator_fallback_result"
    )
    def test_decorated_function(should_fail: bool = False):
        if should_fail:
            raise RuntimeError("Decorator test error")
        return "success"
    
    print(f"\n🧪 Testing decorator with success: {test_decorated_function(False)}")
    print(f"🧪 Testing decorator with failure: {test_decorated_function(True)}")
    
    # Get error statistics
    stats = error_handler.get_error_statistics()
    print(f"\n📊 Error Handler Statistics:")
    print(f"- Total errors handled: {stats['total_errors_handled']}")
    print(f"- Successful recoveries: {stats['successful_recoveries']}")
    print(f"- Recovery success rate: {stats['recovery_success_rate']:.2%}")
    print(f"- Dead letter queue size: {stats['dead_letter_queue']['queue_size']}")
    
    # Export error report
    error_handler.export_error_report("error_report.json")
    print("📋 Error report exported")
    
    print("\n✅ Comprehensive Error Handling demonstration completed!")