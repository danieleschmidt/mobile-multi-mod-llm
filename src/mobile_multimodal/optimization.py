"""Performance optimization and scaling capabilities for mobile multi-modal models."""

import asyncio
import concurrent.futures
import functools
import gc
import multiprocessing
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class ResourceLimits:
    """Resource usage limits and thresholds."""
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    max_concurrent_requests: int = 10
    max_queue_size: int = 100
    request_timeout_seconds: float = 30.0


@dataclass 
class PerformanceProfile:
    """Performance profile configuration."""
    batch_size: int = 8
    num_workers: int = 4
    enable_mixed_precision: bool = True
    enable_dynamic_batching: bool = True
    enable_model_parallel: bool = False
    cache_size_mb: int = 512
    prefetch_count: int = 2


class ResourceManager:
    """Manage system resources and enforce limits."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self._active_requests = 0
        self._request_queue = queue.Queue(maxsize=limits.max_queue_size)
        self._lock = threading.Lock()
        
        # Resource monitoring
        self._cpu_usage_history = []
        self._memory_usage_history = []
        self._last_gc_time = time.time()
    
    def can_accept_request(self) -> bool:
        """Check if system can accept a new request."""
        with self._lock:
            # Check active request limit
            if self._active_requests >= self.limits.max_concurrent_requests:
                return False
            
            # Check queue capacity
            if self._request_queue.full():
                return False
            
            # Check system resources
            if not self._check_system_resources():
                return False
            
            return True
    
    def _check_system_resources(self) -> bool:
        """Check current system resource usage."""
        try:
            if psutil is None:
                return True  # Skip checks if psutil unavailable
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._cpu_usage_history.append(cpu_percent)
            if len(self._cpu_usage_history) > 10:
                self._cpu_usage_history.pop(0)
            
            avg_cpu = sum(self._cpu_usage_history) / len(self._cpu_usage_history)
            if avg_cpu > self.limits.max_cpu_percent:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            self._memory_usage_history.append(memory_mb)
            if len(self._memory_usage_history) > 10:
                self._memory_usage_history.pop(0)
            
            if memory_mb > self.limits.max_memory_mb:
                return False
            
            return True
            
        except Exception:
            return True  # Allow on error
    
    @contextmanager
    def acquire_request_slot(self):
        """Context manager to acquire and release request slot."""
        if not self.can_accept_request():
            raise ResourceExhaustedError("System resource limits exceeded")
        
        with self._lock:
            self._active_requests += 1
        
        try:
            yield
        finally:
            with self._lock:
                self._active_requests -= 1
            
            # Trigger GC periodically
            current_time = time.time()
            if current_time - self._last_gc_time > 60:  # Every minute
                gc.collect()
                self._last_gc_time = current_time
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        stats = {
            "active_requests": self._active_requests,
            "queue_size": self._request_queue.qsize(),
            "resource_limits": {
                "max_memory_mb": self.limits.max_memory_mb,
                "max_cpu_percent": self.limits.max_cpu_percent,
                "max_concurrent_requests": self.limits.max_concurrent_requests
            }
        }
        
        if self._cpu_usage_history:
            stats["avg_cpu_percent"] = sum(self._cpu_usage_history) / len(self._cpu_usage_history)
        if self._memory_usage_history:
            stats["avg_memory_mb"] = sum(self._memory_usage_history) / len(self._memory_usage_history)
        
        return stats


class BatchProcessor:
    """Dynamic batching for efficient processing."""
    
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self._batch_queue = []
        self._batch_futures = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._processor_thread = None
        self._processing = False
    
    def start_processing(self, process_batch_func: Callable):
        """Start batch processing thread."""
        if self._processing:
            return
        
        self._processing = True
        self._processor_thread = threading.Thread(
            target=self._batch_processor_loop,
            args=(process_batch_func,),
            daemon=True
        )
        self._processor_thread.start()
    
    def stop_processing(self):
        """Stop batch processing."""
        self._processing = False
        with self._condition:
            self._condition.notify_all()
        
        if self._processor_thread:
            self._processor_thread.join(timeout=5)
    
    def submit_for_batching(self, item: Any) -> concurrent.futures.Future:
        """Submit item for batch processing."""
        future = concurrent.futures.Future()
        
        with self._condition:
            self._batch_queue.append((item, future))
            self._batch_futures.append(future)
            self._condition.notify()
        
        return future
    
    def _batch_processor_loop(self, process_batch_func: Callable):
        """Main batch processing loop."""
        while self._processing:
            with self._condition:
                # Wait for items or timeout
                while not self._batch_queue and self._processing:
                    self._condition.wait(timeout=self.max_wait_time)
                
                if not self._batch_queue:
                    continue
                
                # Collect batch
                batch_items = []
                batch_futures = []
                
                while (len(batch_items) < self.max_batch_size and 
                       self._batch_queue):
                    item, future = self._batch_queue.pop(0)
                    batch_items.append(item)
                    batch_futures.append(future)
            
            # Process batch outside lock
            if batch_items:
                try:
                    results = process_batch_func(batch_items)
                    
                    # Set results
                    for i, future in enumerate(batch_futures):
                        if i < len(results):
                            future.set_result(results[i])
                        else:
                            future.set_exception(
                                RuntimeError("Batch processing returned fewer results than expected")
                            )
                
                except Exception as e:
                    # Set exception for all futures
                    for future in batch_futures:
                        future.set_exception(e)


class ModelPool:
    """Pool of model instances for concurrent processing."""
    
    def __init__(self, model_factory: Callable, pool_size: int = None):
        self.model_factory = model_factory
        self.pool_size = pool_size or multiprocessing.cpu_count()
        self._pool = queue.Queue()
        self._lock = threading.Lock()
        self._initialized = False
    
    def initialize(self):
        """Initialize model pool."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            for _ in range(self.pool_size):
                model = self.model_factory()
                self._pool.put(model)
            
            self._initialized = True
    
    @contextmanager
    def get_model(self):
        """Get model from pool."""
        if not self._initialized:
            self.initialize()
        
        try:
            model = self._pool.get(timeout=30)
            yield model
        finally:
            self._pool.put(model)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool_size,
            "available_models": self._pool.qsize(),
            "initialized": self._initialized
        }


class CacheManager:
    """Advanced caching with LRU, size limits, and automatic cleanup."""
    
    def __init__(self, max_size_mb: int = 512, max_entries: int = 10000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        
        self._cache = {}
        self._access_order = []
        self._sizes = {}
        self._current_size = 0
        self._lock = threading.RLock()
        
        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Update access order
                self._access_order.remove(key)
                self._access_order.append(key)
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None
    
    def put(self, key: str, value: Any, size_hint: int = None) -> bool:
        """Put item in cache."""
        with self._lock:
            # Calculate size
            if size_hint is None:
                try:
                    size_hint = self._estimate_size(value)
                except:
                    size_hint = 1024  # Default size
            
            # Check if item would exceed cache limits
            if size_hint > self.max_size_bytes:
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._current_size -= self._sizes[key]
                self._access_order.remove(key)
            
            # Evict items if necessary
            while (self._current_size + size_hint > self.max_size_bytes or 
                   len(self._cache) >= self.max_entries):
                if not self._access_order:
                    break
                
                lru_key = self._access_order.pop(0)
                self._current_size -= self._sizes[lru_key]
                del self._cache[lru_key]
                del self._sizes[lru_key]
                self._evictions += 1
            
            # Add new item
            self._cache[key] = value
            self._sizes[key] = size_hint
            self._current_size += size_hint
            self._access_order.append(key)
            
            return True
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (str, bytes)):
            return len(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) 
                      for k, v in obj.items())
        else:
            # Rough estimate
            return 1024
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._sizes.clear()
            self._current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "entries": len(self._cache),
                "size_mb": self._current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions
            }


class PerformanceOptimizer:
    """Comprehensive performance optimization system."""
    
    def __init__(self, profile: PerformanceProfile):
        self.profile = profile
        self.resource_manager = ResourceManager(ResourceLimits())
        self.cache_manager = CacheManager(profile.cache_size_mb)
        self.batch_processor = BatchProcessor(profile.batch_size)
        
        # Thread pools
        self.io_executor = ThreadPoolExecutor(
            max_workers=profile.num_workers,
            thread_name_prefix="io-worker"
        )
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=profile.num_workers,
            thread_name_prefix="cpu-worker"
        )
        
        # Performance metrics
        self._optimization_stats = {
            "cache_hits": 0,
            "batch_processing_count": 0,
            "parallel_processing_count": 0
        }
    
    def optimize_inference(self, inference_func: Callable) -> Callable:
        """Optimize inference function with caching and batching."""
        @functools.wraps(inference_func)
        def optimized_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(inference_func.__name__, args, kwargs)
            
            # Try cache first
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self._optimization_stats["cache_hits"] += 1
                return cached_result
            
            # Use resource manager
            with self.resource_manager.acquire_request_slot():
                result = inference_func(*args, **kwargs)
            
            # Cache result
            self.cache_manager.put(cache_key, result)
            
            return result
        
        return optimized_wrapper
    
    def optimize_batch_processing(self, batch_func: Callable, items: List[Any]) -> List[Any]:
        """Optimize batch processing with dynamic batching."""
        if not items:
            return []
        
        if len(items) == 1:
            return [batch_func([items[0]])[0]]
        
        # Use batch processor
        futures = []
        for item in items:
            future = self.batch_processor.submit_for_batching(item)
            futures.append(future)
        
        # Start processing if not already started
        if not self.batch_processor._processing:
            self.batch_processor.start_processing(batch_func)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")
        
        self._optimization_stats["batch_processing_count"] += len(items)
        return results
    
    def optimize_parallel_processing(self, func: Callable, items: List[Any], 
                                   use_processes: bool = False) -> List[Any]:
        """Optimize with parallel processing."""
        if not items:
            return []
        
        executor = ProcessPoolExecutor if use_processes else self.cpu_executor
        
        if use_processes and isinstance(executor, type):
            # Create process pool for this operation
            with ProcessPoolExecutor(max_workers=self.profile.num_workers) as proc_executor:
                futures = [proc_executor.submit(func, item) for item in items]
                results = [future.result() for future in futures]
        else:
            # Use existing thread pool
            futures = [self.cpu_executor.submit(func, item) for item in items]
            results = [future.result() for future in futures]
        
        self._optimization_stats["parallel_processing_count"] += len(items)
        return results
    
    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for function call."""
        import hashlib
        import json
        
        # Create deterministic key
        key_data = {
            "function": func_name,
            "args": self._serialize_args(args),
            "kwargs": self._serialize_args(kwargs)
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _serialize_args(self, args) -> Any:
        """Serialize arguments for cache key generation."""
        if isinstance(args, (list, tuple)):
            return [self._serialize_args(arg) for arg in args]
        elif isinstance(args, dict):
            return {k: self._serialize_args(v) for k, v in args.items()}
        elif isinstance(args, np.ndarray):
            return f"ndarray_shape_{args.shape}_dtype_{args.dtype}_hash_{hash(args.data.tobytes())}"
        elif hasattr(args, '__dict__'):
            return str(type(args).__name__)
        else:
            return str(args)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "optimization_stats": self._optimization_stats.copy(),
            "cache_stats": self.cache_manager.get_stats(),
            "resource_stats": self.resource_manager.get_resource_stats()
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.batch_processor.stop_processing()
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)
        self.cache_manager.clear()


class AutoScaler:
    """Automatic scaling based on load and performance metrics."""
    
    def __init__(self):
        self.scaling_metrics = {
            "cpu_threshold_scale_up": 70.0,
            "cpu_threshold_scale_down": 30.0,
            "memory_threshold_scale_up": 70.0,
            "latency_threshold_scale_up": 2.0,  # seconds
            "error_rate_threshold": 0.05  # 5%
        }
        
        self.current_capacity = 1.0  # Current scaling factor
        self.min_capacity = 0.5
        self.max_capacity = 4.0
        self.scaling_history = []
    
    def should_scale(self, metrics: Dict[str, float]) -> Tuple[bool, float]:
        """Determine if scaling is needed and by how much."""
        scale_up_signals = 0
        scale_down_signals = 0
        
        # CPU utilization
        cpu_percent = metrics.get("avg_cpu_percent", 0)
        if cpu_percent > self.scaling_metrics["cpu_threshold_scale_up"]:
            scale_up_signals += 1
        elif cpu_percent < self.scaling_metrics["cpu_threshold_scale_down"]:
            scale_down_signals += 1
        
        # Memory utilization
        memory_percent = metrics.get("memory_percent", 0)
        if memory_percent > self.scaling_metrics["memory_threshold_scale_up"]:
            scale_up_signals += 1
        
        # Latency
        avg_latency = metrics.get("avg_latency_ms", 0) / 1000
        if avg_latency > self.scaling_metrics["latency_threshold_scale_up"]:
            scale_up_signals += 1
        
        # Error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.scaling_metrics["error_rate_threshold"]:
            scale_up_signals += 1
        
        # Make scaling decision
        if scale_up_signals >= 2:  # Need at least 2 signals
            new_capacity = min(self.current_capacity * 1.5, self.max_capacity)
            return True, new_capacity
        elif scale_down_signals >= 2 and scale_up_signals == 0:
            new_capacity = max(self.current_capacity * 0.8, self.min_capacity)
            return True, new_capacity
        
        return False, self.current_capacity
    
    def apply_scaling(self, new_capacity: float) -> Dict[str, Any]:
        """Apply scaling configuration."""
        scaling_event = {
            "timestamp": time.time(),
            "old_capacity": self.current_capacity,
            "new_capacity": new_capacity,
            "scaling_ratio": new_capacity / self.current_capacity
        }
        
        self.current_capacity = new_capacity
        self.scaling_history.append(scaling_event)
        
        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history.pop(0)
        
        return scaling_event
    
    def get_scaling_recommendations(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get scaling recommendations without applying them."""
        should_scale, new_capacity = self.should_scale(metrics)
        
        return {
            "should_scale": should_scale,
            "current_capacity": self.current_capacity,
            "recommended_capacity": new_capacity,
            "scaling_factor": new_capacity / self.current_capacity if self.current_capacity > 0 else 1.0,
            "scaling_history": self.scaling_history[-5:]  # Last 5 events
        }


class ResourceExhaustedError(Exception):
    """Raised when system resources are exhausted."""
    pass


# Decorators for performance optimization
def cached(cache_manager: CacheManager, ttl: int = 3600):
    """Decorator to cache function results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{func.__name__}:{hash((args, tuple(kwargs.items())))}"
            
            # Try cache first
            result = cache_manager.get(key_data)
            if result is not None:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.put(key_data, result)
            
            return result
        
        return wrapper
    return decorator


def async_executor(executor: concurrent.futures.Executor):
    """Decorator to run function in executor."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return executor.submit(func, *args, **kwargs)
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    print("Testing performance optimization system...")
    
    # Test resource manager
    resource_manager = ResourceManager(ResourceLimits(max_concurrent_requests=2))
    
    with resource_manager.acquire_request_slot():
        print("✅ Resource slot acquired")
        stats = resource_manager.get_resource_stats()
        print(f"   Active requests: {stats['active_requests']}")
    
    # Test cache manager
    cache = CacheManager(max_size_mb=10)
    
    cache.put("test_key", "test_value")
    result = cache.get("test_key")
    print(f"✅ Cache test: {result}")
    
    cache_stats = cache.get_stats()
    print(f"   Cache hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Test batch processor
    def mock_batch_func(items):
        return [f"processed_{item}" for item in items]
    
    batch_processor = BatchProcessor(max_batch_size=3)
    batch_processor.start_processing(mock_batch_func)
    
    futures = []
    for i in range(5):
        future = batch_processor.submit_for_batching(f"item_{i}")
        futures.append(future)
    
    results = [f.result(timeout=5) for f in futures]
    print(f"✅ Batch processing: {results}")
    
    batch_processor.stop_processing()
    
    # Test performance optimizer
    profile = PerformanceProfile(batch_size=4, num_workers=2)
    optimizer = PerformanceOptimizer(profile)
    
    @optimizer.optimize_inference
    def mock_inference(data):
        return f"result_for_{data}"
    
    # Test optimized function
    result1 = mock_inference("test_data")
    result2 = mock_inference("test_data")  # Should use cache
    print(f"✅ Optimized inference: {result1}, cached: {result2}")
    
    # Test auto-scaler
    scaler = AutoScaler()
    test_metrics = {
        "avg_cpu_percent": 75.0,
        "memory_percent": 80.0,
        "avg_latency_ms": 2500,
        "error_rate": 0.02
    }
    
    recommendations = scaler.get_scaling_recommendations(test_metrics)
    print(f"✅ Auto-scaling recommendations: {recommendations['should_scale']}")
    
    # Cleanup
    optimizer.cleanup()
    
    print("🎉 Performance optimization tests completed!")