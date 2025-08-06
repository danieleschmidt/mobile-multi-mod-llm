"""Performance optimization and scaling utilities for mobile AI deployment.

This module provides advanced optimization techniques including caching, concurrent processing,
resource pooling, and adaptive performance tuning for mobile multi-modal LLM inference.
"""

import asyncio
import logging
import multiprocessing
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import lru_cache, wraps
import weakref

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    nn = None
    DataLoader = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance monitoring data structure."""
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_fps: float = 0.0
    queue_length: int = 0
    error_count: int = 0
    timestamp: float = field(default_factory=time.time)


class AdaptiveCache:
    """Intelligent caching system with size limits and TTL."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        """Initialize adaptive cache with size and time limits."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with TTL check."""
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            value, creation_time = self.cache[key]
            
            # Check TTL
            if time.time() - creation_time > self.ttl_seconds:
                self._remove_key(key)
                self.miss_count += 1
                return None
            
            # Update access time for LRU
            self.access_times[key] = time.time()
            self.hit_count += 1
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Store item in cache with eviction if needed."""
        with self._lock:
            current_time = time.time()
            
            # Remove expired items
            self._cleanup_expired()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            # Store new item
            self.cache[key] = (value, current_time)
            self.access_times[key] = current_time
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache and access times."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def _cleanup_expired(self) -> None:
        """Remove expired items from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, creation_time) in self.cache.items()
            if current_time - creation_time > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": total_requests
        }
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0


class ModelPool:
    """Pool of model instances for concurrent processing."""
    
    def __init__(self, model_factory: Callable, pool_size: int = 4):
        """Initialize model pool with factory function."""
        self.model_factory = model_factory
        self.pool_size = pool_size
        self.models = queue.Queue(maxsize=pool_size)
        self.total_models = 0
        self._lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize model instances in the pool."""
        for _ in range(self.pool_size):
            try:
                model = self.model_factory()
                self.models.put(model)
                self.total_models += 1
                logger.info(f"Created model instance {self.total_models}")
            except Exception as e:
                logger.error(f"Failed to create model instance: {e}")
                break
    
    def get_model(self, timeout: float = 10.0):
        """Get model from pool with timeout."""
        try:
            return self.models.get(timeout=timeout)
        except queue.Empty:
            logger.warning("Model pool exhausted, creating temporary instance")
            return self.model_factory()
    
    def return_model(self, model):
        """Return model to pool."""
        try:
            self.models.put_nowait(model)
        except queue.Full:
            # Pool is full, model will be garbage collected
            pass
    
    def get_pool_stats(self) -> Dict[str, int]:
        """Get pool utilization statistics."""
        return {
            "available_models": self.models.qsize(),
            "total_models": self.total_models,
            "pool_size": self.pool_size,
            "utilization": (self.pool_size - self.models.qsize()) / self.pool_size
        }


class ResourceMonitor:
    """System resource monitoring for adaptive performance tuning."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize resource monitor."""
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        self.monitoring = False
        self.monitor_thread = None
        self._callbacks: List[Callable[[PerformanceMetrics], None]] = []
    
    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def add_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """Add callback for performance metrics updates."""
        self._callbacks.append(callback)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self._update_history(metrics)
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        metrics = PerformanceMetrics()
        
        try:
            import psutil
            
            # CPU usage
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            process = psutil.Process()
            metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            
        except ImportError:
            logger.debug("psutil not available for resource monitoring")
        
        return metrics
    
    def _update_history(self, metrics: PerformanceMetrics):
        """Update metrics history with size limit."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
    
    def get_average_metrics(self, window_size: int = 10) -> Optional[PerformanceMetrics]:
        """Get averaged metrics over recent window."""
        if not self.metrics_history:
            return None
        
        recent_metrics = self.metrics_history[-window_size:]
        
        return PerformanceMetrics(
            inference_time_ms=np.mean([m.inference_time_ms for m in recent_metrics]),
            memory_usage_mb=np.mean([m.memory_usage_mb for m in recent_metrics]),
            cpu_usage_percent=np.mean([m.cpu_usage_percent for m in recent_metrics]),
            cache_hit_rate=np.mean([m.cache_hit_rate for m in recent_metrics]),
            throughput_fps=np.mean([m.throughput_fps for m in recent_metrics]),
            queue_length=int(np.mean([m.queue_length for m in recent_metrics])),
            error_count=sum(m.error_count for m in recent_metrics)
        )


class OptimizedInferenceEngine:
    """High-performance inference engine with adaptive optimization."""
    
    def __init__(self, model_factory: Callable, 
                 pool_size: int = 4,
                 cache_size: int = 1000,
                 enable_batching: bool = True):
        """Initialize optimized inference engine."""
        self.model_pool = ModelPool(model_factory, pool_size)
        self.cache = AdaptiveCache(max_size=cache_size)
        self.enable_batching = enable_batching
        self.executor = ThreadPoolExecutor(max_workers=pool_size * 2)
        self.batch_queue = queue.Queue()
        self.batch_size = 8
        self.batch_timeout = 0.05  # 50ms
        self.resource_monitor = ResourceMonitor()
        
        # Performance tracking
        self.total_requests = 0
        self.total_errors = 0
        self.processing_times = []
        
        # Start background services
        self.resource_monitor.start_monitoring()
        if enable_batching:
            self._start_batch_processor()
    
    def _start_batch_processor(self):
        """Start background batch processing."""
        def batch_processor():
            batch = []
            last_batch_time = time.time()
            
            while True:
                try:
                    # Try to get item with timeout
                    try:
                        item = self.batch_queue.get(timeout=self.batch_timeout)
                        batch.append(item)
                    except queue.Empty:
                        pass
                    
                    # Process batch if full or timeout reached
                    current_time = time.time()
                    should_process = (
                        len(batch) >= self.batch_size or
                        (batch and current_time - last_batch_time > self.batch_timeout)
                    )
                    
                    if should_process and batch:
                        self._process_batch(batch)
                        batch = []
                        last_batch_time = current_time
                    
                except Exception as e:
                    logger.error(f"Batch processor error: {e}")
                    batch = []
        
        batch_thread = threading.Thread(target=batch_processor, daemon=True)
        batch_thread.start()
    
    def _process_batch(self, batch: List[Tuple[Any, asyncio.Future]]):
        """Process a batch of inference requests."""
        if not batch:
            return
        
        try:
            # Get model from pool
            model = self.model_pool.get_model()
            
            # Extract inputs from batch
            inputs = [item[0] for item in batch]
            futures = [item[1] for item in batch]
            
            try:
                # Batch inference (if supported)
                results = self._batch_inference(model, inputs)
                
                # Set results for futures
                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)
                        
            except Exception as e:
                # Set exception for all futures
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                        
            finally:
                # Return model to pool
                self.model_pool.return_model(model)
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all futures
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)
    
    def _batch_inference(self, model, inputs: List[Any]) -> List[Any]:
        """Perform batched inference."""
        # Default implementation - override for model-specific batching
        results = []
        for input_data in inputs:
            result = model(input_data)
            results.append(result)
        return results
    
    async def inference_async(self, input_data: Any, cache_key: Optional[str] = None) -> Any:
        """Asynchronous inference with caching."""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Check cache first
            if cache_key:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Submit for batch processing if enabled
            if self.enable_batching:
                future = asyncio.get_event_loop().create_future()
                self.batch_queue.put((input_data, future))
                result = await future
            else:
                # Direct inference
                result = await self._direct_inference_async(input_data)
            
            # Cache result
            if cache_key:
                self.cache.put(cache_key, result)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 1000:
                self.processing_times.pop(0)
            
            return result
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Inference failed: {e}")
            raise
    
    async def _direct_inference_async(self, input_data: Any) -> Any:
        """Direct asynchronous inference without batching."""
        loop = asyncio.get_event_loop()
        
        def inference_task():
            model = self.model_pool.get_model()
            try:
                return model(input_data)
            finally:
                self.model_pool.return_model(model)
        
        return await loop.run_in_executor(self.executor, inference_task)
    
    def inference_sync(self, input_data: Any, cache_key: Optional[str] = None) -> Any:
        """Synchronous inference wrapper."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.inference_async(input_data, cache_key))
        finally:
            loop.close()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_stats = self.cache.get_stats()
        pool_stats = self.model_pool.get_pool_stats()
        
        stats = {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_requests, 1),
            "cache": cache_stats,
            "model_pool": pool_stats,
            "batch_queue_size": self.batch_queue.qsize()
        }
        
        if self.processing_times:
            times = np.array(self.processing_times)
            stats["latency"] = {
                "mean_ms": float(np.mean(times)),
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times))
            }
        
        # Resource metrics
        avg_metrics = self.resource_monitor.get_average_metrics()
        if avg_metrics:
            stats["resources"] = {
                "cpu_usage_percent": avg_metrics.cpu_usage_percent,
                "memory_usage_mb": avg_metrics.memory_usage_mb
            }
        
        return stats
    
    def optimize_parameters(self):
        """Adaptively optimize performance parameters."""
        stats = self.get_performance_stats()
        
        # Adaptive batch size tuning
        if "latency" in stats:
            p95_latency = stats["latency"]["p95_ms"]
            
            if p95_latency > 100:  # High latency
                self.batch_size = max(1, self.batch_size - 1)
                self.batch_timeout = min(0.1, self.batch_timeout * 1.1)
            elif p95_latency < 50:  # Low latency
                self.batch_size = min(32, self.batch_size + 1)
                self.batch_timeout = max(0.01, self.batch_timeout * 0.9)
        
        # Adaptive cache size
        cache_hit_rate = stats["cache"]["hit_rate"]
        if cache_hit_rate < 0.5:  # Low hit rate
            self.cache.max_size = min(5000, int(self.cache.max_size * 1.2))
        elif cache_hit_rate > 0.9:  # High hit rate
            self.cache.max_size = max(100, int(self.cache.max_size * 0.9))
        
        logger.info(f"Optimized parameters: batch_size={self.batch_size}, "
                   f"batch_timeout={self.batch_timeout:.3f}, "
                   f"cache_size={self.cache.max_size}")
    
    def shutdown(self):
        """Gracefully shutdown inference engine."""
        self.resource_monitor.stop_monitoring()
        self.executor.shutdown(wait=True)
        logger.info("Inference engine shutdown complete")


class AutoScaler:
    """Automatic scaling based on load and performance metrics."""
    
    def __init__(self, inference_engine: OptimizedInferenceEngine):
        """Initialize auto-scaler."""
        self.inference_engine = inference_engine
        self.scaling_enabled = True
        self.scale_up_threshold = 80  # CPU percentage
        self.scale_down_threshold = 30
        self.min_pool_size = 2
        self.max_pool_size = 16
        self.scaling_cooldown = 60  # seconds
        self.last_scaling_time = 0
    
    def evaluate_scaling(self) -> Dict[str, Any]:
        """Evaluate if scaling is needed."""
        stats = self.inference_engine.get_performance_stats()
        current_time = time.time()
        
        scaling_decision = {
            "action": "none",
            "reason": "",
            "current_pool_size": stats["model_pool"]["pool_size"],
            "cpu_usage": stats.get("resources", {}).get("cpu_usage_percent", 0)
        }
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            scaling_decision["reason"] = "Scaling in cooldown period"
            return scaling_decision
        
        cpu_usage = stats.get("resources", {}).get("cpu_usage_percent", 0)
        current_pool_size = stats["model_pool"]["pool_size"]
        queue_size = stats["batch_queue_size"]
        
        # Scale up conditions
        if (cpu_usage > self.scale_up_threshold or queue_size > 10) and \
           current_pool_size < self.max_pool_size:
            scaling_decision["action"] = "scale_up"
            scaling_decision["reason"] = f"High load: CPU {cpu_usage}%, Queue {queue_size}"
        
        # Scale down conditions
        elif cpu_usage < self.scale_down_threshold and queue_size == 0 and \
             current_pool_size > self.min_pool_size:
            scaling_decision["action"] = "scale_down"
            scaling_decision["reason"] = f"Low load: CPU {cpu_usage}%"
        
        return scaling_decision
    
    def apply_scaling(self, action: str) -> bool:
        """Apply scaling decision."""
        if not self.scaling_enabled:
            return False
        
        if action == "scale_up":
            # Add model to pool (simplified implementation)
            self.last_scaling_time = time.time()
            logger.info("Scaled up model pool")
            return True
            
        elif action == "scale_down":
            # Remove model from pool (simplified implementation)
            self.last_scaling_time = time.time()
            logger.info("Scaled down model pool")
            return True
        
        return False


# Decorator for automatic caching and performance monitoring
def cached_inference(cache_key_func: Callable = None, ttl: float = 300.0):
    """Decorator for caching inference results."""
    def decorator(func):
        func_cache = AdaptiveCache(max_size=1000, ttl_seconds=ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Check cache
            result = func_cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            func_cache.put(cache_key, result)
            
            return result
        
        wrapper.cache = func_cache
        return wrapper
    return decorator


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator for monitoring function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = 0
        
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss
        except ImportError:
            pass
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise
        finally:
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            
            try:
                end_memory = process.memory_info().rss
                memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
            except:
                memory_delta = 0
            
            logger.info(f"Function {func.__name__}: "
                       f"time={execution_time:.2f}ms, "
                       f"memory_delta={memory_delta:.2f}MB, "
                       f"success={success}")
        
        return result
    return wrapper


if __name__ == "__main__":
    print("Performance optimization module loaded successfully!")
    
    # Example usage demonstration
    def dummy_model_factory():
        """Example model factory for testing."""
        return lambda x: x * 2  # Simple dummy model
    
    # Test adaptive cache
    cache = AdaptiveCache(max_size=100, ttl_seconds=60)
    cache.put("test", "value")
    cached_value = cache.get("test")
    print(f"Cache test: {cached_value}")
    print(f"Cache stats: {cache.get_stats()}")
    
    # Test model pool
    pool = ModelPool(dummy_model_factory, pool_size=2)
    model = pool.get_model()
    result = model(5)
    pool.return_model(model)
    print(f"Model pool test: {result}")
    print(f"Pool stats: {pool.get_pool_stats()}")
    
    logger.info("Optimization module ready for high-performance inference")