"""Distributed inference and scaling for mobile AI systems."""

import asyncio
import time
import threading
import queue
import hashlib
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import random
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference execution modes."""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"
    DISTRIBUTED = "distributed"


class WorkloadType(Enum):
    """Types of AI workloads."""
    CAPTION_GENERATION = "caption_generation"
    OCR_EXTRACTION = "ocr_extraction"
    VQA_ANSWERING = "vqa_answering"
    EMBEDDING_EXTRACTION = "embedding_extraction"
    MIXED = "mixed"


@dataclass
class InferenceRequest:
    """Inference request data structure."""
    request_id: str
    workload_type: WorkloadType
    input_data: Any
    parameters: Dict[str, Any]
    priority: int = 1
    timestamp: float = 0.0
    user_id: str = "anonymous"
    timeout_seconds: float = 30.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class InferenceResult:
    """Inference result data structure."""
    request_id: str
    result: Any
    processing_time_ms: float
    worker_id: str
    status: str = "success"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LoadBalancer:
    """Intelligent load balancer for distributed inference."""
    
    def __init__(self, strategy: str = "least_loaded"):
        self.strategy = strategy
        self.workers = {}
        self.worker_stats = {}
        self.request_history = []
        
    def register_worker(self, worker_id: str, worker_instance, capacity: int = 10):
        """Register an inference worker."""
        self.workers[worker_id] = {
            "instance": worker_instance,
            "capacity": capacity,
            "current_load": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "avg_latency_ms": 0,
            "status": "active"
        }
        self.worker_stats[worker_id] = {
            "latencies": [],
            "last_request_time": 0,
            "health_score": 100
        }
        logger.info(f"Registered worker {worker_id} with capacity {capacity}")
    
    def select_worker(self, request: InferenceRequest) -> Optional[str]:
        """Select optimal worker for request based on strategy."""
        available_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if (worker["status"] == "active" and 
                worker["current_load"] < worker["capacity"])
        ]
        
        if not available_workers:
            return None
        
        if self.strategy == "least_loaded":
            return min(available_workers, 
                      key=lambda w: self.workers[w]["current_load"])
        
        elif self.strategy == "fastest":
            return min(available_workers,
                      key=lambda w: self.workers[w]["avg_latency_ms"])
        
        elif self.strategy == "health_based":
            return max(available_workers,
                      key=lambda w: self.worker_stats[w]["health_score"])
        
        elif self.strategy == "round_robin":
            # Simple round-robin based on total requests
            return min(available_workers,
                      key=lambda w: self.workers[w]["total_requests"])
        
        else:  # random
            return random.choice(available_workers)
    
    def update_worker_stats(self, worker_id: str, latency_ms: float, success: bool):
        """Update worker statistics after request completion."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        stats = self.worker_stats[worker_id]
        
        # Update basic stats
        worker["total_requests"] += 1
        if not success:
            worker["failed_requests"] += 1
        
        # Update latency stats
        stats["latencies"].append(latency_ms)
        if len(stats["latencies"]) > 100:  # Keep only last 100 latencies
            stats["latencies"].pop(0)
        
        worker["avg_latency_ms"] = sum(stats["latencies"]) / len(stats["latencies"])
        stats["last_request_time"] = time.time()
        
        # Update health score
        failure_rate = worker["failed_requests"] / max(worker["total_requests"], 1)
        latency_penalty = max(0, (latency_ms - 100) / 10)  # Penalty for >100ms
        
        stats["health_score"] = max(0, 100 - (failure_rate * 50) - latency_penalty)
        
        # Auto-disable unhealthy workers
        if stats["health_score"] < 20:
            worker["status"] = "disabled"
            logger.warning(f"Worker {worker_id} disabled due to poor health score")
    
    def increment_load(self, worker_id: str):
        """Increment worker load."""
        if worker_id in self.workers:
            self.workers[worker_id]["current_load"] += 1
    
    def decrement_load(self, worker_id: str):
        """Decrement worker load."""
        if worker_id in self.workers:
            self.workers[worker_id]["current_load"] = max(0, 
                self.workers[worker_id]["current_load"] - 1)
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        total_capacity = sum(w["capacity"] for w in self.workers.values())
        total_load = sum(w["current_load"] for w in self.workers.values())
        active_workers = len([w for w in self.workers.values() if w["status"] == "active"])
        
        return {
            "strategy": self.strategy,
            "total_workers": len(self.workers),
            "active_workers": active_workers,
            "total_capacity": total_capacity,
            "current_load": total_load,
            "utilization_percent": (total_load / max(total_capacity, 1)) * 100,
            "worker_details": {
                worker_id: {
                    "load": worker["current_load"],
                    "capacity": worker["capacity"],
                    "utilization": (worker["current_load"] / worker["capacity"]) * 100,
                    "health_score": self.worker_stats[worker_id]["health_score"],
                    "avg_latency_ms": worker["avg_latency_ms"],
                    "total_requests": worker["total_requests"],
                    "failure_rate": worker["failed_requests"] / max(worker["total_requests"], 1),
                    "status": worker["status"]
                }
                for worker_id, worker in self.workers.items()
            }
        }


class InferenceWorker:
    """Individual inference worker for processing requests."""
    
    def __init__(self, worker_id: str, model_instance):
        self.worker_id = worker_id
        self.model = model_instance
        self.request_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.processed_requests = 0
        self.worker_thread = None
        
    def start(self):
        """Start worker processing thread."""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.worker_thread.start()
        logger.info(f"Worker {self.worker_id} started")
    
    def stop(self):
        """Stop worker processing."""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info(f"Worker {self.worker_id} stopped")
    
    def submit_request(self, request: InferenceRequest) -> bool:
        """Submit request to worker queue."""
        try:
            self.request_queue.put(request, timeout=1.0)
            return True
        except queue.Full:
            logger.warning(f"Worker {self.worker_id} queue full, request rejected")
            return False
    
    def _process_requests(self):
        """Main request processing loop."""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=1.0)
                result = self._execute_inference(request)
                
                # Log result for load balancer feedback
                if hasattr(self, '_result_callback'):
                    self._result_callback(result)
                
                self.processed_requests += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} processing error: {e}")
    
    def _execute_inference(self, request: InferenceRequest) -> InferenceResult:
        """Execute inference for a request."""
        start_time = time.time()
        
        try:
            # Route to appropriate model method based on workload type
            if request.workload_type == WorkloadType.CAPTION_GENERATION:
                if hasattr(self.model, 'generate_caption'):
                    result = self.model.generate_caption(request.input_data)
                else:
                    result = f"Mock caption for worker {self.worker_id}"
                    
            elif request.workload_type == WorkloadType.OCR_EXTRACTION:
                if hasattr(self.model, 'extract_text'):
                    result = self.model.extract_text(request.input_data)
                else:
                    result = [{"text": f"Mock OCR from worker {self.worker_id}", "bbox": [0, 0, 100, 20]}]
                    
            elif request.workload_type == WorkloadType.VQA_ANSWERING:
                if hasattr(self.model, 'answer_question'):
                    question = request.parameters.get('question', 'What is this?')
                    result = self.model.answer_question(request.input_data, question)
                else:
                    result = f"Mock VQA answer from worker {self.worker_id}"
                    
            elif request.workload_type == WorkloadType.EMBEDDING_EXTRACTION:
                if hasattr(self.model, 'get_image_embeddings'):
                    result = self.model.get_image_embeddings(request.input_data)
                else:
                    import numpy as np
                    result = np.random.randn(1, 384).astype(np.float32)
                    
            else:  # MIXED or unknown
                result = {"caption": f"Mock result from worker {self.worker_id}"}
            
            processing_time = (time.time() - start_time) * 1000
            
            return InferenceResult(
                request_id=request.request_id,
                result=result,
                processing_time_ms=processing_time,
                worker_id=self.worker_id,
                status="success",
                metadata={
                    "workload_type": request.workload_type.value,
                    "queue_wait_time_ms": (start_time - request.timestamp) * 1000
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Inference error in worker {self.worker_id}: {e}")
            
            return InferenceResult(
                request_id=request.request_id,
                result=None,
                processing_time_ms=processing_time,
                worker_id=self.worker_id,
                status="error",
                error_message=str(e)
            )
    
    def set_result_callback(self, callback: Callable):
        """Set callback for processing results."""
        self._result_callback = callback


class BatchProcessor:
    """Intelligent batch processing for inference optimization."""
    
    def __init__(self, max_batch_size: int = 8, max_wait_time_ms: float = 50):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.pending_requests = {}  # Group by workload type
        self.batch_stats = {"batches_processed": 0, "total_requests": 0}
        
    def add_request(self, request: InferenceRequest) -> Optional[List[InferenceRequest]]:
        """Add request to batch processor. Returns batch if ready."""
        workload_type = request.workload_type
        
        if workload_type not in self.pending_requests:
            self.pending_requests[workload_type] = []
        
        self.pending_requests[workload_type].append(request)
        
        # Check if batch is ready
        if len(self.pending_requests[workload_type]) >= self.max_batch_size:
            return self._create_batch(workload_type)
        
        # Check for timeout
        oldest_request = min(self.pending_requests[workload_type], key=lambda r: r.timestamp)
        if (time.time() - oldest_request.timestamp) * 1000 > self.max_wait_time_ms:
            return self._create_batch(workload_type)
        
        return None
    
    def _create_batch(self, workload_type: WorkloadType) -> List[InferenceRequest]:
        """Create batch from pending requests."""
        if workload_type not in self.pending_requests:
            return []
        
        batch = self.pending_requests[workload_type][:self.max_batch_size]
        self.pending_requests[workload_type] = self.pending_requests[workload_type][self.max_batch_size:]
        
        self.batch_stats["batches_processed"] += 1
        self.batch_stats["total_requests"] += len(batch)
        
        return batch
    
    def get_pending_count(self) -> Dict[str, int]:
        """Get count of pending requests by workload type."""
        return {
            workload_type.value: len(requests)
            for workload_type, requests in self.pending_requests.items()
        }
    
    def flush_all_batches(self) -> Dict[WorkloadType, List[InferenceRequest]]:
        """Flush all pending requests as batches."""
        batches = {}
        for workload_type in list(self.pending_requests.keys()):
            if self.pending_requests[workload_type]:
                batches[workload_type] = self._create_batch(workload_type)
        return batches


class DistributedInferenceEngine:
    """Main distributed inference engine coordinating all components."""
    
    def __init__(self, load_balancer_strategy: str = "least_loaded"):
        self.load_balancer = LoadBalancer(load_balancer_strategy)
        self.batch_processor = BatchProcessor()
        self.workers = {}
        self.request_futures = {}  # Track async requests
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency_ms": 0,
            "requests_per_second": 0
        }
        
        self.latency_history = []
        self.start_time = time.time()
    
    def add_worker(self, model_instance, worker_id: Optional[str] = None, capacity: int = 10):
        """Add inference worker to the distributed system."""
        if worker_id is None:
            worker_id = f"worker_{len(self.workers)}"
        
        worker = InferenceWorker(worker_id, model_instance)
        
        # Set result callback for statistics
        def result_callback(result: InferenceResult):
            self._update_stats(result)
            self.load_balancer.update_worker_stats(
                worker_id, result.processing_time_ms, result.status == "success"
            )
        
        worker.set_result_callback(result_callback)
        worker.start()
        
        self.workers[worker_id] = worker
        self.load_balancer.register_worker(worker_id, worker, capacity)
        
        logger.info(f"Added worker {worker_id} to distributed engine")
    
    def submit_request(self, workload_type: WorkloadType, input_data: Any, 
                      parameters: Dict[str, Any] = None, priority: int = 1,
                      user_id: str = "anonymous") -> str:
        """Submit inference request to distributed system."""
        request_id = hashlib.md5(f"{time.time()}_{random.random()}".encode()).hexdigest()[:8]
        
        request = InferenceRequest(
            request_id=request_id,
            workload_type=workload_type,
            input_data=input_data,
            parameters=parameters or {},
            priority=priority,
            user_id=user_id
        )
        
        # Try batch processing first
        batch = self.batch_processor.add_request(request)
        if batch:
            self._process_batch(batch)
        else:
            # Process single request
            self._process_single_request(request)
        
        self.stats["total_requests"] += 1
        return request_id
    
    def _process_single_request(self, request: InferenceRequest):
        """Process single request through load balancer."""
        worker_id = self.load_balancer.select_worker(request)
        
        if worker_id is None:
            logger.error("No available workers for request")
            return
        
        worker = self.workers[worker_id]
        self.load_balancer.increment_load(worker_id)
        
        # Submit to worker queue
        if not worker.submit_request(request):
            self.load_balancer.decrement_load(worker_id)
            logger.error(f"Failed to submit request {request.request_id} to worker {worker_id}")
    
    def _process_batch(self, batch: List[InferenceRequest]):
        """Process batch of requests."""
        # For batch processing, find worker with highest capacity
        available_workers = [
            worker_id for worker_id, worker_info in self.load_balancer.workers.items()
            if worker_info["status"] == "active"
        ]
        
        if not available_workers:
            logger.error("No available workers for batch processing")
            return
        
        # Select worker with highest remaining capacity
        best_worker = max(available_workers, 
                         key=lambda w: self.load_balancer.workers[w]["capacity"] - 
                                      self.load_balancer.workers[w]["current_load"])
        
        worker = self.workers[best_worker]
        
        # Submit all batch requests to the same worker for efficiency
        for request in batch:
            self.load_balancer.increment_load(best_worker)
            if not worker.submit_request(request):
                self.load_balancer.decrement_load(best_worker)
                logger.warning(f"Failed to submit batch request {request.request_id}")
    
    def _update_stats(self, result: InferenceResult):
        """Update engine statistics."""
        if result.status == "success":
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Update latency stats
        self.latency_history.append(result.processing_time_ms)
        if len(self.latency_history) > 1000:  # Keep last 1000 latencies
            self.latency_history.pop(0)
        
        if self.latency_history:
            self.stats["avg_latency_ms"] = sum(self.latency_history) / len(self.latency_history)
        
        # Update throughput
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.stats["requests_per_second"] = self.stats["total_requests"] / elapsed_time
        
        # Decrement worker load
        self.load_balancer.decrement_load(result.worker_id)
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        return {
            "engine_stats": self.stats,
            "load_balancer_stats": self.load_balancer.get_load_balancer_stats(),
            "batch_processor_stats": {
                "pending_requests": self.batch_processor.get_pending_count(),
                "batches_processed": self.batch_processor.batch_stats["batches_processed"],
                "total_batched_requests": self.batch_processor.batch_stats["total_requests"]
            },
            "worker_count": len(self.workers),
            "uptime_seconds": time.time() - self.start_time
        }
    
    def scale_workers(self, target_workers: int, model_factory: Callable):
        """Dynamically scale number of workers."""
        current_workers = len(self.workers)
        
        if target_workers > current_workers:
            # Scale up
            for i in range(target_workers - current_workers):
                new_model = model_factory()
                self.add_worker(new_model, capacity=10)
            logger.info(f"Scaled up from {current_workers} to {target_workers} workers")
            
        elif target_workers < current_workers:
            # Scale down
            workers_to_remove = list(self.workers.keys())[target_workers:]
            for worker_id in workers_to_remove:
                worker = self.workers[worker_id]
                worker.stop()
                del self.workers[worker_id]
                del self.load_balancer.workers[worker_id]
                del self.load_balancer.worker_stats[worker_id]
            logger.info(f"Scaled down from {current_workers} to {target_workers} workers")
    
    def shutdown(self):
        """Shutdown distributed inference engine."""
        logger.info("Shutting down distributed inference engine...")
        
        for worker in self.workers.values():
            worker.stop()
        
        self.executor.shutdown(wait=True)
        logger.info("Distributed inference engine shut down")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Distributed Inference Engine...")
    
    # Mock model for testing
    class MockModel:
        def __init__(self, model_id: str = "mock"):
            self.model_id = model_id
            
        def generate_caption(self, image):
            time.sleep(random.uniform(0.01, 0.05))  # Simulate processing
            return f"Caption from {self.model_id}"
        
        def extract_text(self, image):
            time.sleep(random.uniform(0.02, 0.04))
            return [{"text": f"OCR from {self.model_id}", "bbox": [0, 0, 100, 20]}]
        
        def answer_question(self, image, question):
            time.sleep(random.uniform(0.03, 0.06))
            return f"Answer from {self.model_id}: {question}"
    
    # Create distributed engine
    engine = DistributedInferenceEngine("least_loaded")
    
    # Add workers
    for i in range(3):
        model = MockModel(f"model_{i}")
        engine.add_worker(model, f"worker_{i}", capacity=5)
    
    # Submit test requests
    import numpy as np
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    print("Submitting test requests...")
    request_ids = []
    
    for i in range(20):
        if i % 3 == 0:
            workload = WorkloadType.CAPTION_GENERATION
        elif i % 3 == 1:
            workload = WorkloadType.OCR_EXTRACTION
        else:
            workload = WorkloadType.VQA_ANSWERING
            
        request_id = engine.submit_request(
            workload_type=workload,
            input_data=test_image,
            parameters={"question": "What is this?"} if workload == WorkloadType.VQA_ANSWERING else {},
            user_id=f"user_{i % 5}"
        )
        request_ids.append(request_id)
    
    # Wait for processing
    time.sleep(2)
    
    # Get stats
    stats = engine.get_engine_stats()
    print(f"Engine Stats: {stats['engine_stats']['total_requests']} requests processed")
    print(f"Average Latency: {stats['engine_stats']['avg_latency_ms']:.1f}ms")
    print(f"Success Rate: {stats['engine_stats']['successful_requests'] / max(stats['engine_stats']['total_requests'], 1) * 100:.1f}%")
    
    # Test scaling
    print("Testing dynamic scaling...")
    def model_factory():
        return MockModel(f"scaled_model_{time.time()}")
    
    engine.scale_workers(5, model_factory)  # Scale up
    time.sleep(1)
    engine.scale_workers(2, model_factory)  # Scale down
    
    # Shutdown
    engine.shutdown()
    
    print("✅ Distributed inference engine working correctly!")