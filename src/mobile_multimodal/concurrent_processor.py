"""Concurrent Processing Engine - Advanced parallel processing for mobile AI inference.

This module implements production-grade concurrent processing with:
1. Adaptive thread pool management based on device capabilities
2. GPU/NPU/CPU workload distribution and scheduling
3. Batch processing optimization with dynamic batching
4. Pipeline parallelism for multi-stage inference
5. Mobile-aware resource management and power optimization
"""

import asyncio
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    mp = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessingUnit(Enum):
    """Types of processing units."""
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    EDGE_TPU = "edge_tpu"
    AUTO = "auto"


class BatchingStrategy(Enum):
    """Batching strategies for processing."""
    STATIC = "static"        # Fixed batch size
    DYNAMIC = "dynamic"      # Adaptive batch size
    PRIORITY = "priority"    # Priority-based batching
    LATENCY = "latency"      # Latency-optimized batching


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessingTask:
    """Task for concurrent processing."""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    processing_unit: ProcessingUnit = ProcessingUnit.AUTO
    max_batch_size: int = 1
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    
    @property
    def age(self) -> float:
        """Get task age in seconds."""
        return time.time() - self.created_time


@dataclass
class ProcessingResult:
    """Result of processing task."""
    task_id: str
    result: Any
    success: bool
    processing_time: float
    processing_unit: ProcessingUnit
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingMetrics:
    """Metrics for processing performance."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_processing_time: float = 0.0
    avg_queue_wait_time: float = 0.0
    throughput_per_second: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    batch_efficiency: float = 0.0


class DeviceCapabilityDetector:
    """Detects and manages device processing capabilities."""
    
    def __init__(self):
        self.capabilities = self._detect_capabilities()
        self.performance_scores = self._benchmark_devices()
        
    def _detect_capabilities(self) -> Dict[ProcessingUnit, bool]:
        """Detect available processing units."""
        capabilities = {
            ProcessingUnit.CPU: True,  # Always available
            ProcessingUnit.GPU: False,
            ProcessingUnit.NPU: False,
            ProcessingUnit.EDGE_TPU: False
        }
        
        # Detect GPU
        if TORCH_AVAILABLE:
            capabilities[ProcessingUnit.GPU] = torch.cuda.is_available()
        
        # Detect NPU (Qualcomm Hexagon, Apple Neural Engine, etc.)
        # This would require platform-specific detection code
        try:
            # Placeholder for NPU detection
            capabilities[ProcessingUnit.NPU] = self._detect_npu()
        except Exception:
            pass
        
        # Detect Edge TPU
        try:
            # Placeholder for Edge TPU detection
            capabilities[ProcessingUnit.EDGE_TPU] = self._detect_edge_tpu()
        except Exception:
            pass
        
        logger.info(f"Detected capabilities: {capabilities}")
        return capabilities
    
    def _detect_npu(self) -> bool:
        """Detect NPU availability."""
        # Platform-specific NPU detection would go here
        # For Android: Check for Qualcomm Hexagon SDK
        # For iOS: Check for Core ML with Neural Engine
        return False
    
    def _detect_edge_tpu(self) -> bool:
        """Detect Edge TPU availability."""
        try:
            # Try importing Edge TPU runtime
            import tflite_runtime.interpreter as tflite
            # Check for Edge TPU delegate
            return True
        except ImportError:
            return False
    
    def _benchmark_devices(self) -> Dict[ProcessingUnit, float]:
        """Benchmark processing units to get relative performance scores."""
        scores = {}
        
        for unit, available in self.capabilities.items():
            if available:
                scores[unit] = self._benchmark_single_device(unit)
            else:
                scores[unit] = 0.0
        
        logger.info(f"Performance scores: {scores}")
        return scores
    
    def _benchmark_single_device(self, unit: ProcessingUnit) -> float:
        """Benchmark a single processing unit."""
        try:
            if unit == ProcessingUnit.CPU:
                return self._benchmark_cpu()
            elif unit == ProcessingUnit.GPU and TORCH_AVAILABLE:
                return self._benchmark_gpu()
            else:
                return 1.0  # Default score
        except Exception as e:
            logger.warning(f"Benchmarking failed for {unit.value}: {str(e)}")
            return 1.0
    
    def _benchmark_cpu(self) -> float:
        """Benchmark CPU performance."""
        # Simple CPU benchmark using matrix multiplication
        start_time = time.perf_counter()
        
        # Create test matrices
        a = np.random.rand(500, 500).astype(np.float32)
        b = np.random.rand(500, 500).astype(np.float32)
        
        # Perform computation
        for _ in range(10):
            c = np.dot(a, b)
        
        elapsed_time = time.perf_counter() - start_time
        
        # Score inversely proportional to time (faster = higher score)
        return 1.0 / elapsed_time if elapsed_time > 0 else 1.0
    
    def _benchmark_gpu(self) -> float:
        """Benchmark GPU performance."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        
        try:
            device = torch.device('cuda')
            start_time = time.perf_counter()
            
            # Create test tensors
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            
            # Perform computation
            for _ in range(10):
                c = torch.mm(a, b)
                torch.cuda.synchronize()  # Wait for completion
            
            elapsed_time = time.perf_counter() - start_time
            return 1.0 / elapsed_time if elapsed_time > 0 else 1.0
            
        except Exception as e:
            logger.warning(f"GPU benchmark failed: {str(e)}")
            return 0.0
    
    def get_best_device(self, task_requirements: Dict[str, Any] = None) -> ProcessingUnit:
        """Get the best processing unit for a task."""
        # Filter available devices
        available_devices = [unit for unit, available in self.capabilities.items() if available]
        
        if not available_devices:
            return ProcessingUnit.CPU
        
        # Apply task requirements
        if task_requirements:
            memory_requirement = task_requirements.get('memory_mb', 0)
            compute_requirement = task_requirements.get('compute_intensity', 'medium')
            
            # GPU is better for high compute intensity
            if (compute_requirement == 'high' and 
                ProcessingUnit.GPU in available_devices and
                memory_requirement < 2048):  # GPU memory limit
                return ProcessingUnit.GPU
            
            # NPU is better for inference tasks
            if (task_requirements.get('task_type') == 'inference' and
                ProcessingUnit.NPU in available_devices):
                return ProcessingUnit.NPU
        
        # Return device with highest performance score
        best_device = max(available_devices, key=lambda x: self.performance_scores[x])
        return best_device


class ConcurrentProcessor:
    """Simple concurrent processing system for Generation 2."""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.active_workers = 0
        self._lock = threading.Lock()
        logger.info(f"ConcurrentProcessor initialized with {max_workers} workers")
    
    def process_concurrent(self, tasks: List[Any], processor_func: callable) -> List[Any]:
        """Process tasks concurrently."""
        import concurrent.futures
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            try:
                futures = [executor.submit(processor_func, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Task processing failed: {e}")
                        results.append(None)
            except Exception as e:
                logger.error(f"Concurrent processing failed: {e}")
        
        return results

class ThreadSafeCache:
    """Thread-safe cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self._lock = threading.Lock()
        logger.info(f"ThreadSafeCache initialized with max_size={max_size}")
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        with self._lock:
            return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self.cache)

class AdaptiveBatchProcessor:
    """Adaptive batch processing with dynamic batch size optimization."""
    
    def __init__(self, initial_batch_size: int = 8, max_batch_size: int = 32):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        self.batch_queue = asyncio.Queue()
        self.processing_batches = {}
        
    async def add_task(self, task: ProcessingTask) -> str:
        """Add task to batch processing queue."""
        batch_id = f"batch_{int(time.time() * 1000)}_{task.task_id}"
        await self.batch_queue.put((batch_id, task))
        return batch_id
    
    async def process_batches(self, processor_func: Callable) -> None:
        """Process tasks in adaptive batches."""
        current_batch = []
        batch_start_time = time.time()
        
        while True:
            try:
                # Collect tasks for batch
                timeout = 0.1  # 100ms batch timeout
                
                while len(current_batch) < self.current_batch_size:
                    try:
                        batch_id, task = await asyncio.wait_for(
                            self.batch_queue.get(), timeout=timeout
                        )
                        current_batch.append((batch_id, task))
                        
                        # Reduce timeout for subsequent tasks in batch
                        timeout = 0.01
                        
                    except asyncio.TimeoutError:
                        break
                
                # Process batch if we have tasks
                if current_batch:
                    await self._process_batch(current_batch, processor_func)
                    current_batch.clear()
                    batch_start_time = time.time()
                else:
                    # No tasks, wait a bit
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                # Clear current batch on error
                current_batch.clear()
    
    async def _process_batch(self, batch: List[Tuple[str, ProcessingTask]], 
                           processor_func: Callable) -> None:
        """Process a single batch of tasks."""
        start_time = time.perf_counter()
        
        try:
            # Extract tasks and prepare batch input
            batch_ids = [batch_id for batch_id, _ in batch]
            tasks = [task for _, task in batch]
            
            # Group tasks by function and processing unit
            grouped_tasks = self._group_tasks_for_batching(tasks)
            
            # Process each group
            for group_key, group_tasks in grouped_tasks.items():
                if asyncio.iscoroutinefunction(processor_func):
                    results = await processor_func(group_tasks)
                else:
                    results = processor_func(group_tasks)
                
                # Handle results
                for task, result in zip(group_tasks, results):
                    if task.callback:
                        if asyncio.iscoroutinefunction(task.callback):
                            await task.callback(result)
                        else:
                            task.callback(result)
            
            # Record performance
            processing_time = time.perf_counter() - start_time
            batch_efficiency = len(batch) / processing_time if processing_time > 0 else 0
            
            self.performance_history.append({
                "batch_size": len(batch),
                "processing_time": processing_time,
                "efficiency": batch_efficiency,
                "timestamp": time.time()
            })
            
            # Adapt batch size based on performance
            self._adapt_batch_size()
            
            logger.debug(f"Processed batch of {len(batch)} tasks in {processing_time:.4f}s")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
    
    def _group_tasks_for_batching(self, tasks: List[ProcessingTask]) -> Dict[Tuple, List[ProcessingTask]]:
        """Group tasks that can be batched together."""
        groups = {}
        
        for task in tasks:
            # Group by function name and processing unit
            group_key = (task.function.__name__, task.processing_unit)
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(task)
        
        return groups
    
    def _adapt_batch_size(self) -> None:
        """Adapt batch size based on recent performance."""
        if len(self.performance_history) < 5:
            return
        
        # Analyze recent performance
        recent_performance = self.performance_history[-5:]
        
        # Calculate efficiency trend
        efficiencies = [p["efficiency"] for p in recent_performance]
        avg_efficiency = np.mean(efficiencies)
        
        # Adjust batch size based on efficiency
        if avg_efficiency > 10:  # High efficiency, try larger batches
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
        elif avg_efficiency < 5:  # Low efficiency, try smaller batches
            self.current_batch_size = max(
                1,
                int(self.current_batch_size * 0.8)
            )
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""
    
    def __init__(self, name: str, processing_unit: ProcessingUnit = ProcessingUnit.AUTO):
        self.name = name
        self.processing_unit = processing_unit
        self.metrics = ProcessingMetrics()
        
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process data in this pipeline stage."""
        pass
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get processing metrics for this stage."""
        return self.metrics


class ConcurrentPipeline:
    """Concurrent processing pipeline with parallel stages."""
    
    def __init__(self, stages: List[PipelineStage], max_concurrent: int = 4):
        self.stages = stages
        self.max_concurrent = max_concurrent
        self.pipeline_semaphore = asyncio.Semaphore(max_concurrent)
        self.stage_queues = {stage.name: asyncio.Queue() for stage in stages}
        self.results_queue = asyncio.Queue()
        
    async def process_pipeline(self, input_data: Any, pipeline_id: str = None) -> Any:
        """Process data through the entire pipeline."""
        if pipeline_id is None:
            pipeline_id = f"pipeline_{int(time.time() * 1000)}"
        
        async with self.pipeline_semaphore:
            current_data = input_data
            
            for stage in self.stages:
                start_time = time.perf_counter()
                
                try:
                    current_data = await stage.process(current_data)
                    
                    processing_time = time.perf_counter() - start_time
                    stage.metrics.total_tasks += 1
                    stage.metrics.completed_tasks += 1
                    
                    # Update average processing time
                    stage.metrics.avg_processing_time = (
                        (stage.metrics.avg_processing_time * (stage.metrics.completed_tasks - 1) +
                         processing_time) / stage.metrics.completed_tasks
                    )
                    
                except Exception as e:
                    stage.metrics.failed_tasks += 1
                    logger.error(f"Pipeline stage {stage.name} failed: {str(e)}")
                    raise
            
            return current_data
    
    async def process_batch_pipeline(self, input_batch: List[Any]) -> List[Any]:
        """Process a batch of data through the pipeline."""
        tasks = []
        
        for i, data in enumerate(input_batch):
            task = asyncio.create_task(
                self.process_pipeline(data, f"batch_item_{i}")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def get_pipeline_metrics(self) -> Dict[str, ProcessingMetrics]:
        """Get metrics for all pipeline stages."""
        return {stage.name: stage.get_metrics() for stage in self.stages}


class ConcurrentProcessingEngine:
    """Main concurrent processing engine with adaptive resource management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Device detection
        self.device_detector = DeviceCapabilityDetector()
        
        # Thread pools for different types of work
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=self.config["cpu_workers"],
            thread_name_prefix="mobile_cpu"
        )
        
        self.io_executor = ThreadPoolExecutor(
            max_workers=self.config["io_workers"],
            thread_name_prefix="mobile_io"
        )
        
        # Task queues by priority
        self.task_queues = {
            priority: asyncio.PriorityQueue() 
            for priority in TaskPriority
        }
        
        # Batch processor
        self.batch_processor = AdaptiveBatchProcessor(
            initial_batch_size=self.config["initial_batch_size"],
            max_batch_size=self.config["max_batch_size"]
        )
        
        # Metrics
        self.global_metrics = ProcessingMetrics()
        
        # Worker tasks
        self.workers = []
        self.running = False
        
        logger.info("Concurrent processing engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for concurrent processing."""
        return {
            "cpu_workers": min(4, (mp.cpu_count() if mp else 2)),
            "io_workers": 2,
            "initial_batch_size": 4,
            "max_batch_size": 16,
            "queue_timeout": 1.0,
            "enable_batching": True,
            "enable_pipeline": True,
            "power_optimization": True
        }
    
    async def start(self):
        """Start the processing engine."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker tasks for each priority level
        for priority in TaskPriority:
            worker_task = asyncio.create_task(
                self._worker(priority)
            )
            self.workers.append(worker_task)
        
        # Start batch processor if enabled
        if self.config["enable_batching"]:
            batch_worker = asyncio.create_task(
                self.batch_processor.process_batches(self._process_batch)
            )
            self.workers.append(batch_worker)
        
        logger.info("Concurrent processing engine started")
    
    async def stop(self):
        """Stop the processing engine."""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Shutdown executors
        self.cpu_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
        
        logger.info("Concurrent processing engine stopped")
    
    async def submit_task(self, task: ProcessingTask) -> str:
        """Submit a task for processing."""
        self.global_metrics.total_tasks += 1
        
        # Determine best processing unit if AUTO
        if task.processing_unit == ProcessingUnit.AUTO:
            task.processing_unit = self.device_detector.get_best_device(
                task.metadata
            )
        
        # Submit to appropriate queue
        if task.max_batch_size > 1 and self.config["enable_batching"]:
            # Submit to batch processor
            batch_id = await self.batch_processor.add_task(task)
            return batch_id
        else:
            # Submit to priority queue
            priority_value = 4 - task.priority.value  # Higher priority = lower value
            await self.task_queues[task.priority].put((priority_value, task))
            return task.task_id
    
    async def _worker(self, priority: TaskPriority):
        """Worker for processing tasks of specific priority."""
        queue = self.task_queues[priority]
        
        while self.running:
            try:
                # Get task from queue
                priority_value, task = await asyncio.wait_for(
                    queue.get(),
                    timeout=self.config["queue_timeout"]
                )
                
                # Process task
                result = await self._process_single_task(task)
                
                # Call callback if provided
                if task.callback:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(result)
                    else:
                        task.callback(result)
                
            except asyncio.TimeoutError:
                continue  # No tasks available, continue waiting
            except Exception as e:
                logger.error(f"Worker error for priority {priority.name}: {str(e)}")
    
    async def _process_single_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single task."""
        start_time = time.perf_counter()
        
        try:
            # Select appropriate executor
            if task.processing_unit == ProcessingUnit.CPU:
                executor = self.cpu_executor
            else:
                executor = self.cpu_executor  # Default to CPU for now
            
            # Execute task
            loop = asyncio.get_event_loop()
            
            if asyncio.iscoroutinefunction(task.function):
                result = await task.function(*task.args, **task.kwargs)
            else:
                result = await loop.run_in_executor(
                    executor,
                    lambda: task.function(*task.args, **task.kwargs)
                )
            
            processing_time = time.perf_counter() - start_time
            
            # Update metrics
            self.global_metrics.completed_tasks += 1
            self.global_metrics.avg_processing_time = (
                (self.global_metrics.avg_processing_time * (self.global_metrics.completed_tasks - 1) +
                 processing_time) / self.global_metrics.completed_tasks
            )
            
            return ProcessingResult(
                task_id=task.task_id,
                result=result,
                success=True,
                processing_time=processing_time,
                processing_unit=task.processing_unit
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self.global_metrics.failed_tasks += 1
            
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            return ProcessingResult(
                task_id=task.task_id,
                result=None,
                success=False,
                processing_time=processing_time,
                processing_unit=task.processing_unit,
                error_message=str(e)
            )
    
    async def _process_batch(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """Process a batch of tasks."""
        results = []
        
        for task in tasks:
            result = await self._process_single_task(task)
            results.append(result)
        
        return results
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        queue_sizes = {
            priority.name: queue.qsize() 
            for priority, queue in self.task_queues.items()
        }
        
        return {
            "global_metrics": {
                "total_tasks": self.global_metrics.total_tasks,
                "completed_tasks": self.global_metrics.completed_tasks,
                "failed_tasks": self.global_metrics.failed_tasks,
                "success_rate": (
                    self.global_metrics.completed_tasks / 
                    max(self.global_metrics.total_tasks, 1)
                ),
                "avg_processing_time": self.global_metrics.avg_processing_time
            },
            "queue_sizes": queue_sizes,
            "device_capabilities": {
                unit.value: available 
                for unit, available in self.device_detector.capabilities.items()
            },
            "performance_scores": {
                unit.value: score
                for unit, score in self.device_detector.performance_scores.items()
            },
            "current_batch_size": self.batch_processor.current_batch_size,
            "running": self.running,
            "active_workers": len(self.workers)
        }
    
    def optimize_for_battery(self, battery_level: float):
        """Optimize processing for battery level."""
        if battery_level < 0.2:  # Low battery
            # Reduce worker count
            self.config["cpu_workers"] = max(1, self.config["cpu_workers"] // 2)
            # Increase batch size to reduce overhead
            self.batch_processor.max_batch_size = min(32, self.batch_processor.max_batch_size * 2)
            logger.info("Optimized processing for low battery")
        elif battery_level > 0.8:  # High battery
            # Restore full performance
            self.config["cpu_workers"] = min(4, mp.cpu_count() if mp else 2)
            logger.info("Restored full processing performance")


# Factory functions
def create_mobile_processing_engine() -> ConcurrentProcessingEngine:
    """Create processing engine optimized for mobile deployment."""
    config = {
        "cpu_workers": 2,           # Conservative for mobile
        "io_workers": 1,            # Limited I/O workers
        "initial_batch_size": 2,    # Smaller initial batch
        "max_batch_size": 8,        # Smaller max batch
        "queue_timeout": 0.5,       # Faster timeout
        "enable_batching": True,
        "enable_pipeline": True,
        "power_optimization": True
    }
    
    return ConcurrentProcessingEngine(config)


# Export classes and functions
__all__ = [
    "ProcessingUnit", "BatchingStrategy", "TaskPriority",
    "ProcessingTask", "ProcessingResult", "ProcessingMetrics",
    "DeviceCapabilityDetector", "AdaptiveBatchProcessor",
    "PipelineStage", "ConcurrentPipeline", "ConcurrentProcessingEngine",
    "create_mobile_processing_engine"
]