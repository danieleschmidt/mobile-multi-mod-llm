"""
Generation 1: Real-time Multi-Modal Inference Pipeline
MAKE IT WORK - Simple but functional real-time processing capabilities
"""

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

try:
    import numpy as np
except ImportError:
    np = None

# Enhanced logging setup
logger = logging.getLogger(__name__)
pipeline_logger = logging.getLogger(f"{__name__}.pipeline")

@dataclass
class InferenceRequest:
    """Real-time inference request structure."""
    request_id: str
    user_id: str
    operation: str  # "caption", "ocr", "vqa", "embedding"
    image: Optional[Any] = None
    text: Optional[str] = None
    timestamp: float = 0.0
    priority: int = 0  # 0=low, 1=normal, 2=high, 3=critical
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def __lt__(self, other):
        """Enable comparison for priority queue."""
        return self.timestamp < other.timestamp

@dataclass 
class InferenceResult:
    """Real-time inference result structure."""
    request_id: str
    user_id: str
    operation: str
    result: Dict[str, Any]
    processing_time_ms: float
    timestamp: float
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RealTimeInferenceEngine:
    """Real-time multi-modal inference engine with queue-based processing."""
    
    def __init__(self, model_instance, max_workers: int = 4, queue_size: int = 1000):
        """Initialize real-time inference engine.
        
        Args:
            model_instance: MobileMultiModalLLM instance
            max_workers: Maximum number of worker threads
            queue_size: Maximum queue size for pending requests
        """
        self.model = model_instance
        self.max_workers = max_workers
        self.queue_size = queue_size
        
        # Request processing queue with priority support
        self.request_queue = queue.PriorityQueue(maxsize=queue_size)
        self.result_callbacks = {}  # request_id -> callback function
        
        # Worker thread management
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.workers_running = False
        self.worker_threads = []
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "queue_full_rejections": 0,
            "avg_processing_time_ms": 0.0,
            "requests_per_second": 0.0,
            "last_update_time": time.time()
        }
        
        # Request tracking
        self.active_requests = {}  # request_id -> InferenceRequest
        self.completed_requests = []  # Recent completed requests (last 100)
        
        logger.info(f"RealTimeInferenceEngine initialized with {max_workers} workers, queue size {queue_size}")
    
    def start(self):
        """Start the real-time inference engine."""
        if self.workers_running:
            logger.warning("Inference engine already running")
            return
        
        self.workers_running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"InferenceWorker-{i}",
                daemon=True
            )
            worker_thread.start()
            self.worker_threads.append(worker_thread)
        
        # Start metrics collection thread
        metrics_thread = threading.Thread(
            target=self._metrics_loop,
            name="MetricsCollector",
            daemon=True
        )
        metrics_thread.start()
        
        logger.info("Real-time inference engine started")
    
    def stop(self):
        """Stop the real-time inference engine."""
        if not self.workers_running:
            logger.warning("Inference engine not running")
            return
        
        self.workers_running = False
        
        # Signal workers to stop
        for _ in range(self.max_workers):
            try:
                self.request_queue.put((0, None), timeout=1.0)  # Sentinel value
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=2.0)
        
        self.worker_threads.clear()
        logger.info("Real-time inference engine stopped")
    
    def submit_request(self, request: InferenceRequest, 
                      callback: Optional[Callable[[InferenceResult], None]] = None) -> bool:
        """Submit inference request for processing.
        
        Args:
            request: Inference request to process
            callback: Optional callback function for result notification
            
        Returns:
            bool: True if request was queued, False if queue is full
        """
        try:
            # Priority queue uses negative priority for correct ordering (higher priority first)
            priority = -request.priority
            
            # Store callback if provided
            if callback:
                self.result_callbacks[request.request_id] = callback
            
            # Add to queue
            self.request_queue.put((priority, request), block=False)
            self.active_requests[request.request_id] = request
            self.metrics["total_requests"] += 1
            
            pipeline_logger.debug(f"Request {request.request_id} queued for {request.operation}")
            return True
            
        except queue.Full:
            self.metrics["queue_full_rejections"] += 1
            logger.warning(f"Request queue full, rejecting request {request.request_id}")
            return False
    
    async def submit_request_async(self, request: InferenceRequest) -> InferenceResult:
        """Submit inference request asynchronously and wait for result.
        
        Args:
            request: Inference request to process
            
        Returns:
            InferenceResult: Processing result
        """
        result_future = asyncio.Future()
        
        def callback(result: InferenceResult):
            if not result_future.done():
                result_future.set_result(result)
        
        success = self.submit_request(request, callback)
        if not success:
            # Queue full, return immediate error
            return InferenceResult(
                request_id=request.request_id,
                user_id=request.user_id,
                operation=request.operation,
                result={},
                processing_time_ms=0.0,
                timestamp=time.time(),
                success=False,
                error_message="Queue full - request rejected"
            )
        
        # Wait for result
        try:
            result = await asyncio.wait_for(result_future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            return InferenceResult(
                request_id=request.request_id,
                user_id=request.user_id,
                operation=request.operation,
                result={},
                processing_time_ms=30000.0,
                timestamp=time.time(),
                success=False,
                error_message="Request timeout"
            )
    
    def _worker_loop(self):
        """Main worker loop for processing inference requests."""
        worker_name = threading.current_thread().name
        logger.debug(f"Worker {worker_name} started")
        
        while self.workers_running:
            try:
                # Get next request from queue
                priority, request = self.request_queue.get(timeout=1.0)
                
                # Check for sentinel value (stop signal)
                if request is None:
                    break
                
                # Process the request
                result = self._process_request(request)
                
                # Store result and call callback if provided
                self._handle_result(result)
                
                # Mark task as done
                self.request_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.debug(f"Worker {worker_name} stopped")
    
    def _process_request(self, request: InferenceRequest) -> InferenceResult:
        """Process a single inference request."""
        start_time = time.time()
        
        try:
            pipeline_logger.debug(f"Processing request {request.request_id}: {request.operation}")
            
            # Route to appropriate model operation
            if request.operation == "caption" and request.image is not None:
                result_data = {"caption": self.model.generate_caption(request.image, user_id=request.user_id)}
                
            elif request.operation == "ocr" and request.image is not None:
                result_data = {"text_regions": self.model.extract_text(request.image)}
                
            elif request.operation == "vqa" and request.image is not None and request.text:
                result_data = {"answer": self.model.answer_question(request.image, request.text)}
                
            elif request.operation == "embedding" and request.image is not None:
                embeddings = self.model.get_image_embeddings(request.image)
                result_data = {
                    "embeddings": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                    "embedding_dim": embeddings.shape[-1] if hasattr(embeddings, 'shape') else 0
                }
                
            elif request.operation == "multi_task" and request.image is not None:
                # Multi-task inference - run multiple operations
                caption = self.model.generate_caption(request.image, user_id=request.user_id)
                ocr_text = self.model.extract_text(request.image)
                embeddings = self.model.get_image_embeddings(request.image)
                
                result_data = {
                    "caption": caption,
                    "text_regions": ocr_text,
                    "embeddings": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                    "multi_task": True
                }
                
            else:
                raise ValueError(f"Unsupported operation '{request.operation}' or missing required inputs")
            
            processing_time = (time.time() - start_time) * 1000
            
            result = InferenceResult(
                request_id=request.request_id,
                user_id=request.user_id,
                operation=request.operation,
                result=result_data,
                processing_time_ms=processing_time,
                timestamp=time.time(),
                success=True
            )
            
            pipeline_logger.debug(f"Request {request.request_id} completed in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            result = InferenceResult(
                request_id=request.request_id,
                user_id=request.user_id,
                operation=request.operation,
                result={},
                processing_time_ms=processing_time,
                timestamp=time.time(),
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Request {request.request_id} failed: {e}")
            return result
    
    def _handle_result(self, result: InferenceResult):
        """Handle processing result."""
        # Update metrics
        if result.success:
            self.metrics["completed_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update average processing time
        total_completed = self.metrics["completed_requests"] + self.metrics["failed_requests"]
        if total_completed > 0:
            current_avg = self.metrics["avg_processing_time_ms"]
            self.metrics["avg_processing_time_ms"] = (
                (current_avg * (total_completed - 1) + result.processing_time_ms) / total_completed
            )
        
        # Store in completed requests (keep last 100)
        self.completed_requests.append(result)
        if len(self.completed_requests) > 100:
            self.completed_requests.pop(0)
        
        # Remove from active requests
        if result.request_id in self.active_requests:
            del self.active_requests[result.request_id]
        
        # Call callback if provided
        if result.request_id in self.result_callbacks:
            callback = self.result_callbacks.pop(result.request_id)
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error for request {result.request_id}: {e}")
    
    def _metrics_loop(self):
        """Background metrics collection loop."""
        last_completed = 0
        
        while self.workers_running:
            time.sleep(5.0)  # Update every 5 seconds
            
            try:
                current_time = time.time()
                current_completed = self.metrics["completed_requests"] + self.metrics["failed_requests"]
                time_delta = current_time - self.metrics["last_update_time"]
                
                # Calculate requests per second
                if time_delta > 0:
                    requests_delta = current_completed - last_completed
                    self.metrics["requests_per_second"] = requests_delta / time_delta
                
                self.metrics["last_update_time"] = current_time
                last_completed = current_completed
                
                # Log metrics periodically
                if current_completed % 100 == 0 and current_completed > 0:
                    logger.info(f"Pipeline metrics: {self.get_metrics()}")
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics."""
        queue_size = self.request_queue.qsize()
        active_count = len(self.active_requests)
        
        return {
            **self.metrics,
            "queue_size": queue_size,
            "active_requests": active_count,
            "worker_threads": len(self.worker_threads),
            "workers_running": self.workers_running
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_size": self.request_queue.qsize(),
            "max_queue_size": self.queue_size,
            "queue_utilization": self.request_queue.qsize() / self.queue_size,
            "active_requests": len(self.active_requests),
            "workers_available": self.max_workers - len(self.active_requests)
        }
    
    def get_recent_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent processing results."""
        recent = self.completed_requests[-limit:] if limit > 0 else self.completed_requests
        return [asdict(result) for result in recent]


class StreamingInferenceManager:
    """Manager for streaming real-time inference from multiple sources."""
    
    def __init__(self, inference_engine: RealTimeInferenceEngine):
        self.inference_engine = inference_engine
        self.active_streams = {}  # stream_id -> stream_info
        self.stream_callbacks = {}  # stream_id -> callback
        
    def start_image_stream(self, stream_id: str, frame_rate_fps: float = 1.0,
                          operations: List[str] = None,
                          callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Start a streaming inference session.
        
        Args:
            stream_id: Unique identifier for the stream
            frame_rate_fps: Target frame rate for processing
            operations: List of operations to perform on each frame
            callback: Callback for streaming results
            
        Returns:
            Stream configuration
        """
        if operations is None:
            operations = ["caption", "ocr"]
        
        stream_info = {
            "stream_id": stream_id,
            "frame_rate_fps": frame_rate_fps,
            "operations": operations,
            "frames_processed": 0,
            "start_time": time.time(),
            "last_frame_time": 0,
            "active": True
        }
        
        self.active_streams[stream_id] = stream_info
        if callback:
            self.stream_callbacks[stream_id] = callback
        
        logger.info(f"Started stream {stream_id} at {frame_rate_fps} FPS with operations: {operations}")
        return stream_info
    
    def process_frame(self, stream_id: str, frame: Any, user_id: str = "stream_user") -> bool:
        """Process a single frame from a stream.
        
        Args:
            stream_id: Stream identifier
            frame: Image frame to process
            user_id: User identifier for the request
            
        Returns:
            bool: True if frame was submitted for processing
        """
        if stream_id not in self.active_streams:
            logger.warning(f"Stream {stream_id} not found")
            return False
        
        stream_info = self.active_streams[stream_id]
        if not stream_info["active"]:
            return False
        
        current_time = time.time()
        
        # Check frame rate limiting
        time_since_last = current_time - stream_info["last_frame_time"]
        min_interval = 1.0 / stream_info["frame_rate_fps"]
        
        if time_since_last < min_interval:
            return False  # Too soon, skip frame
        
        # Process each operation
        success_count = 0
        for operation in stream_info["operations"]:
            request_id = f"{stream_id}_{operation}_{stream_info['frames_processed']}"
            
            request = InferenceRequest(
                request_id=request_id,
                user_id=user_id,
                operation=operation,
                image=frame,
                priority=1,  # Normal priority for streaming
                metadata={
                    "stream_id": stream_id,
                    "frame_number": stream_info["frames_processed"],
                    "stream_timestamp": current_time
                }
            )
            
            # Define callback for stream results
            def stream_result_callback(result: InferenceResult, stream_id=stream_id):
                if stream_id in self.stream_callbacks:
                    try:
                        self.stream_callbacks[stream_id](result)
                    except Exception as e:
                        logger.error(f"Stream callback error: {e}")
            
            success = self.inference_engine.submit_request(request, stream_result_callback)
            if success:
                success_count += 1
        
        # Update stream info
        stream_info["frames_processed"] += 1
        stream_info["last_frame_time"] = current_time
        
        return success_count > 0
    
    def stop_stream(self, stream_id: str):
        """Stop a streaming session."""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["active"] = False
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            logger.info(f"Stopped stream {stream_id}")
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Get status of all active streams."""
        return {
            "active_streams": len([s for s in self.active_streams.values() if s["active"]]),
            "total_streams": len(self.active_streams),
            "streams": dict(self.active_streams)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Real-time Multi-Modal Inference Pipeline...")
    
    # Mock model for testing
    class MockModel:
        def generate_caption(self, image, user_id="test"):
            time.sleep(0.1)  # Simulate processing time
            return "Mock caption: A scene with various objects"
        
        def extract_text(self, image):
            time.sleep(0.05)
            return [{"text": "MOCK TEXT", "bbox": [10, 10, 100, 30], "confidence": 0.9}]
        
        def answer_question(self, image, question):
            time.sleep(0.08)
            return f"Mock answer for: {question}"
        
        def get_image_embeddings(self, image):
            if np:
                return np.random.randn(1, 384).astype(np.float32)
            return [[0.1] * 384]
    
    # Test real-time pipeline
    mock_model = MockModel()
    engine = RealTimeInferenceEngine(mock_model, max_workers=2)
    
    try:
        # Start the engine
        engine.start()
        time.sleep(0.5)  # Let workers start
        
        # Submit test requests
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) if np else "mock_image"
        
        requests = [
            InferenceRequest("req1", "user1", "caption", image=test_image, priority=2),
            InferenceRequest("req2", "user1", "ocr", image=test_image, priority=1),
            InferenceRequest("req3", "user1", "multi_task", image=test_image, priority=3),
        ]
        
        results = []
        def collect_result(result):
            results.append(result)
            print(f"✅ Result received: {result.operation} - Success: {result.success}")
        
        # Submit requests
        for req in requests:
            success = engine.submit_request(req, collect_result)
            print(f"Request {req.request_id} submitted: {success}")
        
        # Wait for processing
        time.sleep(2.0)
        
        # Check metrics
        metrics = engine.get_metrics()
        print(f"\n📊 Pipeline Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Test streaming
        print(f"\n🎥 Testing Streaming...")
        streaming_manager = StreamingInferenceManager(engine)
        
        def stream_callback(result):
            print(f"📡 Stream result: {result.operation} - {result.success}")
        
        # Start stream
        stream_config = streaming_manager.start_image_stream(
            "test_stream", 
            frame_rate_fps=2.0,
            operations=["caption", "ocr"],
            callback=stream_callback
        )
        print(f"Stream started: {stream_config}")
        
        # Process some frames
        for i in range(3):
            frame_success = streaming_manager.process_frame("test_stream", test_image)
            print(f"Frame {i} processed: {frame_success}")
            time.sleep(0.6)  # Respect frame rate
        
        # Wait for stream processing
        time.sleep(1.0)
        
        # Get final status
        stream_status = streaming_manager.get_stream_status()
        print(f"\n📈 Stream Status: {stream_status}")
        
        final_metrics = engine.get_metrics()
        print(f"\n📊 Final Metrics: {final_metrics}")
        
        print("\n✅ Real-time pipeline test completed successfully!")
        
    finally:
        # Clean up
        engine.stop()
        print("🛑 Engine stopped")