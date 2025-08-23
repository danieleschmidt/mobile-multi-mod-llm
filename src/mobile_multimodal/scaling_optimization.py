"""Advanced scaling and performance optimization for mobile multi-modal LLM."""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from queue import Queue, PriorityQueue
import weakref

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None

@dataclass
class ProcessingTask:
    """Processing task for batch processing."""
    task_id: str
    priority: int
    input_data: Any
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    
    def __lt__(self, other):
        return self.priority < other.priority

class IntelligentCache:
    """High-performance adaptive cache with intelligent eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = []  # LRU tracking
        self.size_tracking = {}  # Size tracking per key
        self.current_memory = 0
        self._lock = threading.RLock()
        
        # Adaptive parameters
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_running = True
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent access tracking."""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    self._evict_key(key)
                    self.miss_count += 1
                    return None
                
                # Update access metadata
                entry.access_count += 1
                
                # Update LRU order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.hit_count += 1
                return entry.value
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache with intelligent eviction."""
        with self._lock:
            # Calculate size
            value_size = self._calculate_size(value)
            
            # Check if single item exceeds memory limit
            if value_size > self.max_memory_bytes:
                logger.warning(f"Item too large for cache: {value_size} bytes")
                return False
            
            # Evict if necessary
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + value_size > self.max_memory_bytes):
                
                if not self._evict_lru():
                    logger.error("Failed to evict items from cache")
                    return False
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                size_bytes=value_size,
                ttl=ttl
            )
            
            # Remove old entry if updating
            if key in self.cache:
                self.current_memory -= self.cache[key].size_bytes
            
            self.cache[key] = entry
            self.current_memory += value_size
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return True
    
    def _evict_key(self, key: str):
        """Evict specific key from cache."""
        if key in self.cache:
            self.current_memory -= self.cache[key].size_bytes
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            self.eviction_count += 1
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self.access_order:
            return False
        
        lru_key = self.access_order[0]
        self._evict_key(lru_key)
        return True
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (str, bytes)):
            return len(obj.encode('utf-8')) if isinstance(obj, str) else len(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(self._calculate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._calculate_size(k) + self._calculate_size(v) 
                      for k, v in obj.items())
        else:
            # Rough estimate
            return 64  # Base object overhead
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while self.cleanup_running:
            try:
                with self._lock:
                    current_time = time.time()
                    expired_keys = []
                    
                    for key, entry in self.cache.items():
                        if entry.ttl and current_time - entry.timestamp > entry.ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._evict_key(key)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                time.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_mb": self.current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "eviction_count": self.eviction_count
            }
    
    def shutdown(self):
        """Shutdown cache cleanup."""
        self.cleanup_running = False
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)

class BatchProcessor:
    """High-performance batch processing with dynamic optimization."""
    
    def __init__(self, 
                 max_batch_size: int = 32,
                 max_wait_time: float = 0.1,
                 num_workers: int = 4):
        
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.num_workers = num_workers
        
        self.task_queue = PriorityQueue()
        self.result_futures = {}
        self.batch_stats = {
            "processed_batches": 0,
            "total_items": 0,
            "avg_batch_size": 0,
            "avg_processing_time": 0
        }
        
        self.workers = []
        self.running = False
    
    def start(self):
        """Start batch processing workers."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop, 
                args=(f"worker-{i}",),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Batch processor started with {self.num_workers} workers")
    
    def stop(self):
        """Stop batch processing workers."""
        self.running = False
        
        # Wake up workers
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        self.workers.clear()
        logger.info("Batch processor stopped")
    
    def submit(self, task: ProcessingTask) -> 'asyncio.Future':
        """Submit task for batch processing."""
        future = asyncio.Future()
        task.task_id = task.task_id or str(id(task))
        self.result_futures[task.task_id] = future
        self.task_queue.put(task)
        return future
    
    def _worker_loop(self, worker_name: str):
        """Main worker processing loop."""
        logger.info(f"Batch worker {worker_name} started")
        
        while self.running:
            try:
                batch = self._collect_batch()
                
                if batch is None:  # Shutdown signal
                    break
                
                if batch:
                    self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error in batch worker {worker_name}: {e}")
                time.sleep(0.1)
        
        logger.info(f"Batch worker {worker_name} stopped")
    
    def _collect_batch(self) -> Optional[List[ProcessingTask]]:
        """Collect tasks into a batch."""
        batch = []
        batch_start_time = time.time()
        
        # Get first task (blocking)
        try:
            first_task = self.task_queue.get(timeout=1.0)
            if first_task is None:  # Shutdown signal
                return None
            batch.append(first_task)
        except:
            return []
        
        # Collect additional tasks (non-blocking)
        while (len(batch) < self.max_batch_size and 
               time.time() - batch_start_time < self.max_wait_time):
            
            try:
                task = self.task_queue.get_nowait()
                if task is None:  # Shutdown signal
                    # Put it back for other workers
                    self.task_queue.put(None)
                    break
                batch.append(task)
            except:
                break  # Queue empty
        
        return batch
    
    def _process_batch(self, batch: List[ProcessingTask]):
        """Process a batch of tasks."""
        start_time = time.time()
        
        try:
            # Group by processing type for efficiency
            grouped_tasks = self._group_tasks_by_type(batch)
            
            # Process each group
            for task_type, tasks in grouped_tasks.items():
                results = self._process_task_group(task_type, tasks)
                
                # Set results in futures
                for task, result in zip(tasks, results):
                    future = self.result_futures.get(task.task_id)
                    if future and not future.done():
                        future.set_result(result)
                    
                    # Cleanup
                    if task.task_id in self.result_futures:
                        del self.result_futures[task.task_id]
            
            # Update statistics
            processing_time = time.time() - start_time
            self.batch_stats["processed_batches"] += 1
            self.batch_stats["total_items"] += len(batch)
            
            # Update running averages
            total_batches = self.batch_stats["processed_batches"]
            self.batch_stats["avg_batch_size"] = (
                (self.batch_stats["avg_batch_size"] * (total_batches - 1) + len(batch)) /
                total_batches
            )
            self.batch_stats["avg_processing_time"] = (
                (self.batch_stats["avg_processing_time"] * (total_batches - 1) + processing_time) /
                total_batches
            )
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
            # Set error in futures
            for task in batch:
                future = self.result_futures.get(task.task_id)
                if future and not future.done():
                    future.set_exception(e)
                
                if task.task_id in self.result_futures:
                    del self.result_futures[task.task_id]
    
    def _group_tasks_by_type(self, batch: List[ProcessingTask]) -> Dict[str, List[ProcessingTask]]:
        """Group tasks by processing type for batch efficiency."""
        grouped = {}
        
        for task in batch:
            task_type = task.metadata.get("type", "default") if task.metadata else "default"
            
            if task_type not in grouped:
                grouped[task_type] = []
            grouped[task_type].append(task)
        
        return grouped
    
    def _process_task_group(self, task_type: str, tasks: List[ProcessingTask]) -> List[Any]:
        """Process a group of tasks of the same type."""
        # This would be implemented with actual model inference
        # For now, simulate processing
        
        results = []
        for task in tasks:
            # Simulate processing based on task type
            if task_type == "image_captioning":
                result = f"Caption for image {task.task_id}"
            elif task_type == "text_processing":
                result = f"Processed text {task.task_id}"
            elif task_type == "ocr":
                result = f"OCR result for {task.task_id}"
            else:
                result = f"Processed {task.task_id}"
            
            results.append(result)
            
            # Simulate processing time
            time.sleep(0.001)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return dict(self.batch_stats)

class LoadBalancer:
    """Intelligent load balancer for distributed processing."""
    
    def __init__(self):
        self.endpoints = []
        self.endpoint_stats = {}
        self.current_index = 0
        self._lock = threading.Lock()
    
    def add_endpoint(self, endpoint_id: str, capacity: int = 100, current_load: int = 0):
        """Add processing endpoint."""
        with self._lock:
            endpoint = {
                "id": endpoint_id,
                "capacity": capacity,
                "current_load": current_load,
                "success_rate": 1.0,
                "avg_response_time": 0.1,
                "last_health_check": time.time()
            }
            
            self.endpoints.append(endpoint)
            self.endpoint_stats[endpoint_id] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "total_response_time": 0
            }
    
    def select_endpoint(self) -> Optional[str]:
        """Select best endpoint using weighted round-robin."""
        with self._lock:
            if not self.endpoints:
                return None
            
            # Filter healthy endpoints
            healthy_endpoints = [
                ep for ep in self.endpoints 
                if ep["current_load"] < ep["capacity"] * 0.9 and
                ep["success_rate"] > 0.5
            ]
            
            if not healthy_endpoints:
                # Fall back to least loaded
                healthy_endpoints = sorted(
                    self.endpoints,
                    key=lambda ep: ep["current_load"] / ep["capacity"]
                )[:1]
            
            if not healthy_endpoints:
                return None
            
            # Weighted selection based on capacity and performance
            best_endpoint = min(
                healthy_endpoints,
                key=lambda ep: (
                    ep["current_load"] / ep["capacity"] * 0.6 +
                    (1.0 - ep["success_rate"]) * 0.3 +
                    ep["avg_response_time"] * 0.1
                )
            )
            
            return best_endpoint["id"]
    
    def report_result(self, endpoint_id: str, success: bool, response_time: float):
        """Report processing result for endpoint."""
        with self._lock:
            if endpoint_id not in self.endpoint_stats:
                return
            
            stats = self.endpoint_stats[endpoint_id]
            stats["requests"] += 1
            
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            
            stats["total_response_time"] += response_time
            
            # Update endpoint metrics
            for endpoint in self.endpoints:
                if endpoint["id"] == endpoint_id:
                    endpoint["success_rate"] = (
                        stats["successes"] / stats["requests"] if stats["requests"] > 0 else 1.0
                    )
                    endpoint["avg_response_time"] = (
                        stats["total_response_time"] / stats["requests"] if stats["requests"] > 0 else 0.1
                    )
                    break

class AutoScaler:
    """Automatic scaling based on system metrics and load patterns."""
    
    def __init__(self):
        self.scaling_rules = []
        self.current_scale = 1
        self.min_scale = 1
        self.max_scale = 10
        self.scale_cooldown = 60.0  # seconds
        self.last_scale_time = 0
        
        # Metrics tracking
        self.metrics_history = []
        self.max_history = 100
    
    def add_scaling_rule(self, 
                        condition: Callable[[Dict[str, Any]], bool],
                        action: str,  # "scale_up" or "scale_down"
                        factor: float = 1.5):
        """Add auto-scaling rule."""
        self.scaling_rules.append({
            "condition": condition,
            "action": action,
            "factor": factor
        })
    
    def evaluate_scaling(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling is needed."""
        # Add metrics to history
        self.metrics_history.append({
            "timestamp": time.time(),
            "metrics": metrics
        })
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        # Check cooldown
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return None
        
        # Evaluate rules
        for rule in self.scaling_rules:
            if rule["condition"](metrics):
                if rule["action"] == "scale_up" and self.current_scale < self.max_scale:
                    new_scale = min(
                        self.max_scale,
                        int(self.current_scale * rule["factor"])
                    )
                    
                    if new_scale > self.current_scale:
                        return self._create_scaling_decision("scale_up", new_scale)
                
                elif rule["action"] == "scale_down" and self.current_scale > self.min_scale:
                    new_scale = max(
                        self.min_scale,
                        int(self.current_scale / rule["factor"])
                    )
                    
                    if new_scale < self.current_scale:
                        return self._create_scaling_decision("scale_down", new_scale)
        
        return None
    
    def _create_scaling_decision(self, action: str, new_scale: int) -> Dict[str, Any]:
        """Create scaling decision."""
        decision = {
            "action": action,
            "current_scale": self.current_scale,
            "target_scale": new_scale,
            "timestamp": time.time(),
            "reason": f"Auto-scaling triggered: {action}"
        }
        
        self.current_scale = new_scale
        self.last_scale_time = time.time()
        
        return decision

def smart_cache(max_size: int = 1000, ttl: Optional[float] = None):
    """Decorator for intelligent function result caching."""
    cache = IntelligentCache(max_size=max_size)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": sorted(kwargs.items())
            }
            key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.put(key, result, ttl=ttl)
            
            return result
        
        wrapper.cache_stats = cache.get_stats
        wrapper.cache_clear = lambda: setattr(cache, 'cache', {})
        
        return wrapper
    
    return decorator

# Example optimized functions
@smart_cache(max_size=500, ttl=300.0)
def optimized_image_preprocessing(image_hash: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Cached image preprocessing."""
    # Simulate expensive preprocessing
    time.sleep(0.01)
    return np.random.rand(*target_size, 3).astype(np.float32)

@smart_cache(max_size=1000, ttl=600.0)  
def optimized_model_inference(input_hash: str, model_config: str) -> Dict[str, Any]:
    """Cached model inference results."""
    # Simulate inference
    time.sleep(0.05)
    return {
        "prediction": "sample_output",
        "confidence": 0.95,
        "latency_ms": 50
    }