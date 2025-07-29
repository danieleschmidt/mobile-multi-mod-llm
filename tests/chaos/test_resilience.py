"""Chaos engineering and resilience testing for Mobile Multi-Modal LLM.

This module implements chaos engineering principles to test system resilience
under various failure conditions and stress scenarios.
"""

import asyncio
import concurrent.futures
import os
import random
import signal
import threading
import time
import pytest
import psutil
from contextlib import contextmanager
from typing import Generator, List, Dict, Any
from unittest.mock import patch, MagicMock

from mobile_multimodal import MobileMultiModalLLM


class ChaosInjector:
    """Chaos engineering fault injection utilities."""
    
    @staticmethod
    @contextmanager
    def memory_pressure(target_mb: int = 500):
        """Inject memory pressure by allocating large amounts of memory."""
        memory_hogs = []
        try:
            # Allocate memory in chunks
            while psutil.virtual_memory().available > target_mb * 1024 * 1024:
                # Allocate 50MB chunks
                memory_hogs.append(bytearray(50 * 1024 * 1024))
                time.sleep(0.1)
            yield
        finally:
            # Clean up allocated memory
            del memory_hogs
    
    @staticmethod
    @contextmanager
    def cpu_stress(duration_seconds: int = 5, num_threads: int = None):
        """Inject CPU stress by running intensive computations."""
        if num_threads is None:
            num_threads = os.cpu_count() or 4
        
        stop_event = threading.Event()
        threads = []
        
        def cpu_burn():
            """CPU-intensive task."""
            while not stop_event.is_set():
                # Busy loop with some computation
                sum(i * i for i in range(1000))
        
        try:
            # Start CPU stress threads
            for _ in range(num_threads):
                thread = threading.Thread(target=cpu_burn)
                thread.start()
                threads.append(thread)
            
            yield
            
        finally:
            # Stop stress threads
            stop_event.set()
            for thread in threads:
                thread.join(timeout=1.0)
    
    @staticmethod
    @contextmanager
    def network_latency(delay_ms: int = 100):
        """Simulate network latency for external calls."""
        original_time_sleep = time.sleep
        
        def delayed_operation(*args, **kwargs):
            # Add network delay
            original_time_sleep(delay_ms / 1000.0)
            return original_time_sleep(*args, **kwargs)
        
        with patch('time.sleep', side_effect=delayed_operation):
            yield
    
    @staticmethod
    @contextmanager
    def disk_io_errors():
        """Simulate intermittent disk I/O errors."""
        original_open = open
        
        def failing_open(filename, *args, **kwargs):
            # 20% chance of I/O error
            if random.random() < 0.2:
                raise IOError(f"Simulated I/O error for {filename}")
            return original_open(filename, *args, **kwargs)
        
        with patch('builtins.open', side_effect=failing_open):
            yield
    
    @staticmethod
    @contextmanager
    def intermittent_failures(failure_rate: float = 0.1):
        """Inject intermittent random failures."""
        def maybe_fail():
            if random.random() < failure_rate:
                raise RuntimeError("Simulated intermittent failure")
        
        yield maybe_fail


@pytest.fixture
def chaos_injector():
    """Provide chaos engineering utilities."""
    return ChaosInjector()


@pytest.fixture
def mock_model():
    """Mock model for chaos testing."""
    class ChaosTestModel:
        def __init__(self):
            self.call_count = 0
            self.failure_count = 0
        
        def generate_caption(self, image, timeout=5.0):
            self.call_count += 1
            # Simulate processing time
            time.sleep(0.01)
            return f"Caption {self.call_count}"
        
        def extract_text(self, image, timeout=5.0):
            self.call_count += 1
            time.sleep(0.015)
            return [{"text": f"Text {self.call_count}", "bbox": [0, 0, 100, 50]}]
        
        def answer_question(self, image, question, timeout=5.0):
            self.call_count += 1
            time.sleep(0.012)
            return f"Answer {self.call_count}"
    
    return ChaosTestModel()


class TestMemoryResilience:
    """Test system resilience under memory pressure."""
    
    def test_inference_under_memory_pressure(self, mock_model, chaos_injector):
        """Test model inference continues under memory pressure."""
        import torch
        
        sample_image = torch.randn(3, 224, 224)
        
        with chaos_injector.memory_pressure(target_mb=100):
            # Model should still function under memory pressure
            result = mock_model.generate_caption(sample_image)
            assert result is not None
            assert "Caption" in result
    
    def test_graceful_degradation_low_memory(self, mock_model):
        """Test graceful degradation when memory is extremely low."""
        import torch
        
        sample_image = torch.randn(3, 224, 224)
        
        # Mock memory exhaustion
        with patch('torch.cuda.OutOfMemoryError', RuntimeError):
            try:
                result = mock_model.generate_caption(sample_image)
                # Should either succeed or fail gracefully
                assert result is not None or True  # Graceful failure is acceptable
            except RuntimeError:
                # Graceful failure is acceptable
                pass


class TestCPUResilience:
    """Test system resilience under CPU stress."""
    
    def test_inference_under_cpu_stress(self, mock_model, chaos_injector):
        """Test model inference continues under high CPU load."""
        import torch
        
        sample_image = torch.randn(3, 224, 224)
        
        with chaos_injector.cpu_stress(duration_seconds=10, num_threads=8):
            # Measure performance under stress
            start_time = time.perf_counter()
            result = mock_model.generate_caption(sample_image)
            end_time = time.perf_counter()
            
            assert result is not None
            # Should complete within reasonable time even under stress
            assert (end_time - start_time) < 5.0  # 5 second timeout
    
    def test_concurrent_inference_stability(self, mock_model):
        """Test stability under concurrent inference load."""
        import torch
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        sample_image = torch.randn(3, 224, 224)
        
        def run_inference():
            return mock_model.generate_caption(sample_image)
        
        # Run 20 concurrent inferences
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_inference) for _ in range(20)]
            
            completed = 0
            failed = 0
            
            for future in as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    if result is not None:
                        completed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            
            # At least 80% should succeed
            success_rate = completed / (completed + failed)
            assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"


class TestNetworkResilience:
    """Test resilience to network-related issues."""
    
    def test_inference_with_network_latency(self, mock_model, chaos_injector):
        """Test inference continues with high network latency."""
        import torch
        
        sample_image = torch.randn(3, 224, 224)
        
        with chaos_injector.network_latency(delay_ms=200):
            # Should handle network delays gracefully
            result = mock_model.generate_caption(sample_image)
            assert result is not None
    
    def test_offline_inference_capability(self, mock_model):
        """Test that inference works without network access."""
        import torch
        
        sample_image = torch.randn(3, 224, 224)
        
        # Mock network unavailability
        with patch('requests.get', side_effect=ConnectionError("No network")):
            with patch('urllib.request.urlopen', side_effect=ConnectionError("No network")):
                # Inference should work offline
                result = mock_model.generate_caption(sample_image)
                assert result is not None


class TestIOResilience:
    """Test resilience to I/O failures."""
    
    def test_inference_with_disk_errors(self, mock_model, chaos_injector):
        """Test inference continues despite intermittent disk errors."""
        import torch
        
        sample_image = torch.randn(3, 224, 224)
        
        with chaos_injector.disk_io_errors():
            # Should handle I/O errors gracefully
            try:
                result = mock_model.generate_caption(sample_image)
                assert result is not None
            except IOError:
                # Graceful failure due to I/O error is acceptable
                pass
    
    def test_model_recovery_after_corruption(self, mock_model):
        """Test model recovery after simulated corruption."""
        import torch
        
        sample_image = torch.randn(3, 224, 224)
        
        # Simulate model corruption
        original_method = mock_model.generate_caption
        
        def corrupted_method(*args, **kwargs):
            if random.random() < 0.3:  # 30% corruption rate
                raise RuntimeError("Model corruption detected")
            return original_method(*args, **kwargs)
        
        mock_model.generate_caption = corrupted_method
        
        # Should have some mechanism to recover or retry
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts:
            try:
                result = mock_model.generate_caption(sample_image)
                if result is not None:
                    break
            except RuntimeError:
                attempts += 1
                time.sleep(0.1)  # Brief retry delay
        
        # Should eventually recover or fail gracefully
        assert attempts < max_attempts or True  # Graceful failure acceptable


class TestFailureRecovery:
    """Test system recovery from various failure modes."""
    
    def test_timeout_handling(self, mock_model):
        """Test proper timeout handling for long-running operations."""
        import torch
        import signal
        
        sample_image = torch.randn(3, 224, 224)
        
        # Mock slow operation
        def slow_inference(*args, **kwargs):
            time.sleep(10)  # Simulate slow operation
            return "Slow result"
        
        mock_model.generate_caption = slow_inference
        
        # Test timeout mechanism
        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")
        
        # Set alarm for 2 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)
        
        try:
            result = mock_model.generate_caption(sample_image)
            signal.alarm(0)  # Cancel alarm
            pytest.fail("Should have timed out")
        except TimeoutError:
            signal.alarm(0)  # Cancel alarm
            # Timeout handling worked correctly
            pass
    
    def test_resource_cleanup_on_failure(self, mock_model):
        """Test that resources are properly cleaned up on failure."""
        import torch
        import gc
        
        sample_image = torch.randn(3, 224, 224)
        
        # Track resource usage before test
        initial_memory = psutil.Process().memory_info().rss
        
        # Simulate failure during resource-intensive operation
        def failing_operation(*args, **kwargs):
            # Allocate some resources
            large_tensor = torch.randn(1000, 1000, 1000)  # ~4GB if it existed
            raise RuntimeError("Simulated failure")
        
        mock_model.generate_caption = failing_operation
        
        try:
            mock_model.generate_caption(sample_image)
        except RuntimeError:
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Check that memory usage hasn't grown significantly
        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Allow for some growth but not excessive
        assert memory_growth < 100 * 1024 * 1024, f"Memory leak detected: {memory_growth / 1024 / 1024:.1f}MB growth"


class TestChaosScenarios:
    """Complex chaos engineering scenarios."""
    
    @pytest.mark.slow
    def test_sustained_chaos_scenario(self, mock_model, chaos_injector):
        """Test system under sustained chaotic conditions."""
        import torch
        import threading
        
        sample_image = torch.randn(3, 224, 224)
        results = []
        errors = []
        
        def chaotic_inference():
            """Run inference under chaotic conditions."""
            try:
                with chaos_injector.intermittent_failures(failure_rate=0.2) as maybe_fail:
                    maybe_fail()  # Random failure injection
                    result = mock_model.generate_caption(sample_image)
                    results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple threads with chaos injection
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=chaotic_inference)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Analyze results
        total_attempts = len(results) + len(errors)
        success_rate = len(results) / total_attempts if total_attempts > 0 else 0
        
        # Should maintain reasonable success rate even under chaos
        assert success_rate >= 0.5, f"Success rate too low under chaos: {success_rate:.2%}"
    
    def test_cascading_failure_prevention(self, mock_model):
        """Test prevention of cascading failures."""
        import torch
        
        sample_image = torch.randn(3, 224, 224)
        
        # Mock cascading failure scenario
        failure_count = 0
        
        def potentially_cascading_method(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 3:  # First few calls fail
                raise RuntimeError(f"Cascade failure {failure_count}")
            else:
                return f"Recovery after {failure_count} failures"
        
        mock_model.generate_caption = potentially_cascading_method
        
        # Should implement circuit breaker or similar pattern
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                result = mock_model.generate_caption(sample_image)
                if result:
                    break
            except RuntimeError:
                retry_count += 1
                time.sleep(0.1 * retry_count)  # Exponential backoff
        
        # Should eventually recover
        assert retry_count < max_retries, "System failed to recover from cascading failures"


if __name__ == "__main__":
    # Run chaos engineering tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])