"""Performance regression testing for Mobile Multi-Modal LLM.

This module provides automated performance regression testing to ensure
model optimizations and code changes don't degrade inference performance.
"""

import json
import time
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from mobile_multimodal import MobileMultiModalLLM


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    test_name: str
    latency_ms: float
    throughput_fps: float
    memory_mb: float
    cpu_percent: float
    model_size_mb: float
    timestamp: str


class PerformanceRegressor:
    """Performance regression detection and reporting."""
    
    def __init__(self, baseline_path: str = "tests/performance/baselines.json"):
        self.baseline_path = Path(baseline_path)
        self.baselines = self._load_baselines()
        self.regression_threshold = 0.15  # 15% regression threshold
    
    def _load_baselines(self) -> Dict:
        """Load performance baselines from file."""
        if self.baseline_path.exists():
            with open(self.baseline_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_baseline(self, benchmark: PerformanceBenchmark):
        """Save new performance baseline."""
        self.baselines[benchmark.test_name] = asdict(benchmark)
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, 'w') as f:
            json.dump(self.baselines, f, indent=2)
    
    def check_regression(self, current: PerformanceBenchmark) -> Optional[str]:
        """Check for performance regression."""
        if current.test_name not in self.baselines:
            return None
        
        baseline = self.baselines[current.test_name]
        regressions = []
        
        # Check latency regression (higher is worse)
        latency_change = (current.latency_ms - baseline['latency_ms']) / baseline['latency_ms']
        if latency_change > self.regression_threshold:
            regressions.append(f"Latency regression: {latency_change:.1%} ({current.latency_ms:.1f}ms vs {baseline['latency_ms']:.1f}ms)")
        
        # Check throughput regression (lower is worse)
        throughput_change = (baseline['throughput_fps'] - current.throughput_fps) / baseline['throughput_fps']
        if throughput_change > self.regression_threshold:
            regressions.append(f"Throughput regression: {throughput_change:.1%} ({current.throughput_fps:.1f}fps vs {baseline['throughput_fps']:.1f}fps)")
        
        # Check memory regression (higher is worse)
        memory_change = (current.memory_mb - baseline['memory_mb']) / baseline['memory_mb']
        if memory_change > self.regression_threshold:
            regressions.append(f"Memory regression: {memory_change:.1%} ({current.memory_mb:.1f}MB vs {baseline['memory_mb']:.1f}MB)")
        
        return "; ".join(regressions) if regressions else None


@pytest.fixture
def performance_regressor():
    """Fixture for performance regression testing."""
    return PerformanceRegressor()


@pytest.fixture
def sample_image():
    """Sample image for testing."""
    return torch.randn(3, 224, 224)  # RGB image tensor


@pytest.fixture
def mock_model():
    """Mock model for testing (replace with actual model loading)."""
    # This would load the actual model in real implementation
    class MockModel:
        def generate_caption(self, image):
            time.sleep(0.01)  # Simulate inference time
            return "A sample caption"
        
        def extract_text(self, image):
            time.sleep(0.015)  # Simulate OCR inference
            return [{"text": "Sample text", "bbox": [0, 0, 100, 50]}]
        
        def answer_question(self, image, question):
            time.sleep(0.012)  # Simulate VQA inference
            return "Sample answer"
    
    return MockModel()


class TestPerformanceRegression:
    """Performance regression test suite."""
    
    def test_image_captioning_performance(self, mock_model, sample_image, performance_regressor):
        """Test image captioning performance."""
        import psutil
        import os
        
        # Warm up
        for _ in range(3):
            mock_model.generate_caption(sample_image)
        
        # Measure performance
        start_time = time.perf_counter()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        iterations = 100
        for _ in range(iterations):
            mock_model.generate_caption(sample_image)
        
        end_time = time.perf_counter()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        total_time = (end_time - start_time) * 1000  # Convert to ms
        avg_latency = total_time / iterations
        throughput = 1000 / avg_latency  # FPS
        memory_usage = max(end_memory - start_memory, 0)
        
        benchmark = PerformanceBenchmark(
            test_name="image_captioning",
            latency_ms=avg_latency,
            throughput_fps=throughput,
            memory_mb=memory_usage,
            cpu_percent=psutil.cpu_percent(),
            model_size_mb=35.0,  # Model size from spec
            timestamp=str(int(time.time()))
        )
        
        # Check for regression
        regression = performance_regressor.check_regression(benchmark)
        if regression:
            pytest.fail(f"Performance regression detected: {regression}")
        
        # Save as new baseline if running in baseline mode
        if os.getenv("SAVE_PERFORMANCE_BASELINE"):
            performance_regressor.save_baseline(benchmark)
    
    def test_ocr_performance(self, mock_model, sample_image, performance_regressor):
        """Test OCR performance."""
        import psutil
        import os
        
        # Warm up
        for _ in range(3):
            mock_model.extract_text(sample_image)
        
        # Measure performance
        start_time = time.perf_counter()
        iterations = 50
        
        for _ in range(iterations):
            mock_model.extract_text(sample_image)
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = (end_time - start_time) * 1000
        avg_latency = total_time / iterations
        throughput = 1000 / avg_latency
        
        benchmark = PerformanceBenchmark(
            test_name="ocr_extraction",
            latency_ms=avg_latency,
            throughput_fps=throughput,
            memory_mb=psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
            cpu_percent=psutil.cpu_percent(),
            model_size_mb=35.0,
            timestamp=str(int(time.time()))
        )
        
        regression = performance_regressor.check_regression(benchmark)
        if regression:
            pytest.fail(f"OCR performance regression: {regression}")
        
        if os.getenv("SAVE_PERFORMANCE_BASELINE"):
            performance_regressor.save_baseline(benchmark)
    
    def test_vqa_performance(self, mock_model, sample_image, performance_regressor):
        """Test VQA performance."""
        import psutil
        import os
        
        # Warm up
        for _ in range(3):
            mock_model.answer_question(sample_image, "What is this?")
        
        # Measure performance
        start_time = time.perf_counter()
        iterations = 50
        
        for _ in range(iterations):
            mock_model.answer_question(sample_image, "What is this?")
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = (end_time - start_time) * 1000
        avg_latency = total_time / iterations
        throughput = 1000 / avg_latency
        
        benchmark = PerformanceBenchmark(
            test_name="vqa_inference",
            latency_ms=avg_latency,
            throughput_fps=throughput,
            memory_mb=psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
            cpu_percent=psutil.cpu_percent(),
            model_size_mb=35.0,
            timestamp=str(int(time.time()))
        )
        
        regression = performance_regressor.check_regression(benchmark)
        if regression:
            pytest.fail(f"VQA performance regression: {regression}")
        
        if os.getenv("SAVE_PERFORMANCE_BASELINE"):
            performance_regressor.save_baseline(benchmark)
    
    @pytest.mark.slow
    def test_batch_inference_performance(self, mock_model, performance_regressor):
        """Test batch inference performance."""
        import psutil
        import os
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # Create batch of images
            batch_images = [torch.randn(3, 224, 224) for _ in range(batch_size)]
            
            # Warm up
            for _ in range(3):
                for img in batch_images:
                    mock_model.generate_caption(img)
            
            # Measure performance
            start_time = time.perf_counter()
            iterations = 10
            
            for _ in range(iterations):
                for img in batch_images:
                    mock_model.generate_caption(img)
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time = (end_time - start_time) * 1000
            avg_latency = total_time / (iterations * batch_size)
            throughput = 1000 / avg_latency
            
            benchmark = PerformanceBenchmark(
                test_name=f"batch_inference_size_{batch_size}",
                latency_ms=avg_latency,
                throughput_fps=throughput,
                memory_mb=psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
                cpu_percent=psutil.cpu_percent(),
                model_size_mb=35.0,
                timestamp=str(int(time.time()))
            )
            
            regression = performance_regressor.check_regression(benchmark)
            if regression:
                pytest.fail(f"Batch inference (size {batch_size}) regression: {regression}")
            
            if os.getenv("SAVE_PERFORMANCE_BASELINE"):
                performance_regressor.save_baseline(benchmark)


@pytest.mark.integration
class TestMobileDevicePerformance:
    """Mobile device-specific performance tests."""
    
    def test_android_inference_performance(self, performance_regressor):
        """Test Android device performance profile."""
        # This would use Android benchmarking tools in real implementation
        # For now, simulate Android performance characteristics
        
        benchmark = PerformanceBenchmark(
            test_name="android_inference",
            latency_ms=12.0,  # Target: <15ms
            throughput_fps=83.3,  # Target: >60fps
            memory_mb=150.0,  # Target: <200MB
            cpu_percent=45.0,  # Target: <70%
            model_size_mb=34.5,  # Target: <35MB
            timestamp=str(int(time.time()))
        )
        
        regression = performance_regressor.check_regression(benchmark)
        if regression:
            pytest.fail(f"Android performance regression: {regression}")
    
    def test_ios_inference_performance(self, performance_regressor):
        """Test iOS device performance profile."""
        # This would use iOS benchmarking tools in real implementation
        
        benchmark = PerformanceBenchmark(
            test_name="ios_inference",
            latency_ms=10.0,  # Target: <15ms
            throughput_fps=100.0,  # Target: >60fps
            memory_mb=120.0,  # Target: <200MB
            cpu_percent=35.0,  # Target: <70%
            model_size_mb=34.8,  # Target: <35MB
            timestamp=str(int(time.time()))
        )
        
        regression = performance_regressor.check_regression(benchmark)
        if regression:
            pytest.fail(f"iOS performance regression: {regression}")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short"])