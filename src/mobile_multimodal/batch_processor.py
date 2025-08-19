"""
Generation 1: Mobile-Optimized Batch Processing
MAKE IT WORK - Efficient batch processing for mobile deployment scenarios
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict

try:
    import numpy as np
except ImportError:
    np = None

# Enhanced logging
logger = logging.getLogger(__name__)
batch_logger = logging.getLogger(f"{__name__}.batch")

@dataclass
class BatchItem:
    """Individual item in a batch processing job."""
    item_id: str
    operation: str
    image_path: Optional[str] = None
    image_data: Optional[Any] = None
    text_input: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class BatchJob:
    """Batch processing job configuration."""
    job_id: str
    items: List[BatchItem]
    output_format: str = "json"  # "json", "csv", "parquet"
    output_path: Optional[str] = None
    batch_size: int = 8
    max_workers: int = 4
    save_intermediate: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class BatchResult:
    """Result from batch processing."""
    job_id: str
    item_id: str
    operation: str
    result: Dict[str, Any]
    processing_time_ms: float
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class MobileBatchProcessor:
    """Mobile-optimized batch processor for multi-modal inference."""
    
    def __init__(self, model_instance, cache_dir: str = "batch_cache"):
        """Initialize mobile batch processor.
        
        Args:
            model_instance: MobileMultiModalLLM instance
            cache_dir: Directory for caching intermediate results
        """
        self.model = model_instance
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Job tracking
        self.active_jobs = {}  # job_id -> job_info
        self.completed_jobs = {}  # job_id -> job_results
        
        # Performance metrics
        self.metrics = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_items_processed": 0,
            "avg_items_per_second": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"MobileBatchProcessor initialized with cache dir: {cache_dir}")
    
    def submit_batch_job(self, job: BatchJob) -> str:
        """Submit a batch job for processing.
        
        Args:
            job: BatchJob configuration
            
        Returns:
            str: Job ID for tracking
        """
        job_info = {
            "job": job,
            "status": "submitted",
            "start_time": time.time(),
            "progress": 0,
            "results": [],
            "errors": []
        }
        
        self.active_jobs[job.job_id] = job_info
        self.metrics["total_jobs"] += 1
        
        logger.info(f"Batch job {job.job_id} submitted with {len(job.items)} items")
        return job.job_id
    
    async def process_batch_job_async(self, job: BatchJob, 
                                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process batch job asynchronously.
        
        Args:
            job: BatchJob to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing job results and statistics
        """
        job_id = self.submit_batch_job(job)
        
        try:
            job_info = self.active_jobs[job_id]
            job_info["status"] = "running"
            
            results = []
            errors = []
            
            # Process items in batches
            for batch_start in range(0, len(job.items), job.batch_size):
                batch_items = job.items[batch_start:batch_start + job.batch_size]
                
                # Process batch in parallel
                batch_results = await self._process_batch_parallel(
                    batch_items, job.max_workers
                )
                
                results.extend(batch_results)
                
                # Update progress
                job_info["progress"] = len(results) / len(job.items)
                job_info["results"] = results
                
                # Call progress callback
                if progress_callback:
                    try:
                        progress_callback(job_id, job_info["progress"], len(results))
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
                
                # Save intermediate results if enabled
                if job.save_intermediate:
                    await self._save_intermediate_results(job_id, results)
                
                # Yield control to allow other coroutines
                await asyncio.sleep(0.01)
            
            # Final processing
            job_info["status"] = "completed"
            job_info["end_time"] = time.time()
            job_info["total_time"] = job_info["end_time"] - job_info["start_time"]
            
            # Save final results
            if job.output_path:
                await self._save_results(job, results)
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            del self.active_jobs[job_id]
            
            # Update metrics
            self.metrics["completed_jobs"] += 1
            self.metrics["total_items_processed"] += len(results)
            
            # Calculate throughput
            items_per_second = len(results) / job_info["total_time"]
            current_avg = self.metrics["avg_items_per_second"]
            completed_count = self.metrics["completed_jobs"]
            self.metrics["avg_items_per_second"] = (
                (current_avg * (completed_count - 1) + items_per_second) / completed_count
            )
            
            logger.info(f"Batch job {job_id} completed: {len(results)} items in {job_info['total_time']:.2f}s")
            
            return {
                "job_id": job_id,
                "status": "completed",
                "total_items": len(job.items),
                "successful_items": len([r for r in results if r.success]),
                "failed_items": len([r for r in results if not r.success]),
                "total_time_seconds": job_info["total_time"],
                "items_per_second": items_per_second,
                "results": results
            }
            
        except Exception as e:
            # Handle job failure
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = "failed"
                self.active_jobs[job_id]["error"] = str(e)
            
            self.metrics["failed_jobs"] += 1
            logger.error(f"Batch job {job_id} failed: {e}")
            
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "results": []
            }
    
    async def _process_batch_parallel(self, batch_items: List[BatchItem], 
                                    max_workers: int) -> List[BatchResult]:
        """Process a batch of items in parallel."""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all items to thread pool
            futures = []
            for item in batch_items:
                future = loop.run_in_executor(executor, self._process_single_item, item)
                futures.append(future)
            
            # Wait for all results
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create error result
                    error_result = BatchResult(
                        job_id="unknown",
                        item_id=batch_items[i].item_id,
                        operation=batch_items[i].operation,
                        result={},
                        processing_time_ms=0.0,
                        success=False,
                        error_message=str(result)
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def _process_single_item(self, item: BatchItem) -> BatchResult:
        """Process a single batch item."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(item)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                self.metrics["cache_hits"] += 1
                batch_logger.debug(f"Cache hit for item {item.item_id}")
                return cached_result
            
            self.metrics["cache_misses"] += 1
            
            # Load image if needed
            image_data = None
            if item.image_path:
                image_data = self._load_image(item.image_path)
            elif item.image_data is not None:
                image_data = item.image_data
            
            # Process based on operation
            if item.operation == "caption" and image_data is not None:
                result_data = {"caption": self.model.generate_caption(image_data)}
                
            elif item.operation == "ocr" and image_data is not None:
                result_data = {"text_regions": self.model.extract_text(image_data)}
                
            elif item.operation == "vqa" and image_data is not None and item.text_input:
                result_data = {"answer": self.model.answer_question(image_data, item.text_input)}
                
            elif item.operation == "embedding" and image_data is not None:
                embeddings = self.model.get_image_embeddings(image_data)
                result_data = {
                    "embeddings": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                    "embedding_dim": embeddings.shape[-1] if hasattr(embeddings, 'shape') else 0
                }
                
            elif item.operation == "multi_task" and image_data is not None:
                # Multi-task processing
                caption = self.model.generate_caption(image_data)
                ocr_text = self.model.extract_text(image_data)
                embeddings = self.model.get_image_embeddings(image_data)
                
                result_data = {
                    "caption": caption,
                    "text_regions": ocr_text,
                    "embeddings": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                    "multi_task": True
                }
                
            else:
                raise ValueError(f"Unsupported operation '{item.operation}' or missing inputs")
            
            processing_time = (time.time() - start_time) * 1000
            
            result = BatchResult(
                job_id="batch",
                item_id=item.item_id,
                operation=item.operation,
                result=result_data,
                processing_time_ms=processing_time,
                success=True
            )
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            batch_logger.debug(f"Processed item {item.item_id} in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            result = BatchResult(
                job_id="batch",
                item_id=item.item_id,
                operation=item.operation,
                result={},
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Failed to process item {item.item_id}: {e}")
            return result
    
    def _load_image(self, image_path: str) -> Any:
        """Load image from file path."""
        try:
            if np:
                # Try to load with numpy (simplified - in real implementation would use cv2 or PIL)
                # For now, return mock image data
                return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            else:
                return f"mock_image_from_{image_path}"
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {e}")
    
    def _get_cache_key(self, item: BatchItem) -> str:
        """Generate cache key for batch item."""
        # Simple cache key based on operation and inputs
        key_parts = [item.operation]
        
        if item.image_path:
            # Use file modification time for cache invalidation
            try:
                mtime = os.path.getmtime(item.image_path)
                key_parts.append(f"path_{item.image_path}_{mtime}")
            except:
                key_parts.append(f"path_{item.image_path}")
        
        if item.text_input:
            key_parts.append(f"text_{hash(item.text_input)}")
        
        return "_".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[BatchResult]:
        """Get cached result if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return BatchResult(**data)
        except Exception as e:
            logger.warning(f"Failed to load cached result: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: BatchResult):
        """Cache processing result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    async def _save_intermediate_results(self, job_id: str, results: List[BatchResult]):
        """Save intermediate results."""
        try:
            intermediate_file = self.cache_dir / f"{job_id}_intermediate.json"
            results_data = [asdict(result) for result in results]
            
            with open(intermediate_file, 'w') as f:
                json.dump(results_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")
    
    async def _save_results(self, job: BatchJob, results: List[BatchResult]):
        """Save final batch results."""
        try:
            output_path = Path(job.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if job.output_format == "json":
                results_data = [asdict(result) for result in results]
                with open(output_path, 'w') as f:
                    json.dump(results_data, f, indent=2)
                    
            elif job.output_format == "csv":
                # Simple CSV output (would use pandas in real implementation)
                import csv
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['item_id', 'operation', 'success', 'processing_time_ms', 'result'])
                    
                    for result in results:
                        writer.writerow([
                            result.item_id,
                            result.operation, 
                            result.success,
                            result.processing_time_ms,
                            json.dumps(result.result)
                        ])
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a batch job."""
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job_info["status"],
                "progress": job_info["progress"],
                "items_processed": len(job_info["results"]),
                "total_items": len(job_info["job"].items),
                "errors": len(job_info["errors"])
            }
        elif job_id in self.completed_jobs:
            job_info = self.completed_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job_info["status"],
                "progress": 1.0,
                "items_processed": len(job_info["results"]),
                "total_items": len(job_info["job"].items),
                "total_time": job_info.get("total_time", 0)
            }
        else:
            return {"job_id": job_id, "status": "not_found"}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batch processing metrics."""
        return {
            **self.metrics,
            "active_jobs": len(self.active_jobs),
            "completed_jobs_stored": len(self.completed_jobs),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / 
                max(self.metrics["cache_hits"] + self.metrics["cache_misses"], 1)
            )
        }
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache files."""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            removed_count = 0
            
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old cache files")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Mobile-Optimized Batch Processor...")
    
    # Mock model for testing
    class MockModel:
        def generate_caption(self, image):
            time.sleep(0.05)  # Simulate processing
            return "Mock batch caption: A detailed scene"
        
        def extract_text(self, image):
            time.sleep(0.03)
            return [{"text": "BATCH OCR", "bbox": [5, 5, 80, 25], "confidence": 0.95}]
        
        def answer_question(self, image, question):
            time.sleep(0.04)
            return f"Batch answer: {question[:20]}..."
        
        def get_image_embeddings(self, image):
            if np:
                return np.random.randn(1, 384).astype(np.float32)
            return [[0.2] * 384]
    
    async def test_batch_processing():
        mock_model = MockModel()
        processor = MobileBatchProcessor(mock_model, cache_dir="test_batch_cache")
        
        # Create test batch job
        items = []
        for i in range(5):
            items.extend([
                BatchItem(f"item_{i}_caption", "caption", image_path=f"test_image_{i}.jpg"),
                BatchItem(f"item_{i}_ocr", "ocr", image_path=f"test_image_{i}.jpg"),
                BatchItem(f"item_{i}_vqa", "vqa", image_path=f"test_image_{i}.jpg", 
                         text_input=f"What is in image {i}?"),
            ])
        
        job = BatchJob(
            job_id="test_batch_job",
            items=items,
            output_format="json",
            output_path="test_results.json",
            batch_size=4,
            max_workers=2
        )
        
        # Progress callback
        def progress_callback(job_id, progress, items_done):
            print(f"📊 Job {job_id}: {progress:.1%} complete ({items_done} items)")
        
        # Process the batch
        print(f"🚀 Starting batch job with {len(items)} items...")
        start_time = time.time()
        
        result = await processor.process_batch_job_async(job, progress_callback)
        
        total_time = time.time() - start_time
        print(f"\n✅ Batch job completed!")
        print(f"   Status: {result['status']}")
        print(f"   Total items: {result['total_items']}")
        print(f"   Successful: {result['successful_items']}")
        print(f"   Failed: {result['failed_items']}")
        print(f"   Processing time: {total_time:.2f}s")
        print(f"   Throughput: {result['items_per_second']:.1f} items/second")
        
        # Test metrics
        metrics = processor.get_metrics()
        print(f"\n📈 Processor Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Test cache (run again to see cache hits)
        print(f"\n🔄 Testing cache with repeat job...")
        cache_start = time.time()
        
        result2 = await processor.process_batch_job_async(
            BatchJob("test_cache_job", items[:3], batch_size=2, max_workers=1)
        )
        
        cache_time = time.time() - cache_start
        print(f"   Cached job time: {cache_time:.2f}s")
        print(f"   Cache hit rate: {processor.get_metrics()['cache_hit_rate']:.1%}")
        
        print(f"\n✅ Batch processing test completed!")
        
        # Cleanup
        processor.cleanup_cache(max_age_hours=0)  # Clean all for test
    
    # Run async test
    try:
        asyncio.run(test_batch_processing())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()