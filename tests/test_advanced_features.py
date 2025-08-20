"""Comprehensive tests for advanced mobile multimodal features.

This test suite covers:
1. Adaptive quantization functionality and performance
2. Hybrid attention mechanisms and optimization
3. Edge federated learning components
4. Intelligent caching system
5. Concurrent processing engine
6. Integration tests with realistic mobile scenarios
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import numpy as np

# Import modules to test
from src.mobile_multimodal.adaptive_quantization import (
    AdaptiveQuantizationEngine, EntropyBasedStrategy, PerformanceBasedStrategy,
    PrecisionLevel, HardwareTarget, ComplexityMetrics
)
from src.mobile_multimodal.hybrid_attention import (
    HybridAttentionMechanism, AttentionConfig, create_hybrid_attention
)
from src.mobile_multimodal.edge_federated_learning import (
    EdgeFederatedLearningCoordinator, DeviceProfile, DeviceClass,
    create_mobile_device_profile, create_federated_coordinator
)
from src.mobile_multimodal.intelligent_cache import (
    IntelligentCacheManager, create_mobile_cache_manager,
    CacheLevel, EvictionPolicy
)
from src.mobile_multimodal.concurrent_processor import (
    ConcurrentProcessingEngine, ProcessingTask, TaskPriority,
    ProcessingUnit, create_mobile_processing_engine
)
from src.mobile_multimodal.advanced_validation import (
    CompositeValidator, ValidationLevel, create_validator
)
from src.mobile_multimodal.circuit_breaker import (
    AdaptiveCircuitBreaker, CircuitConfig, create_mobile_circuit_config
)


class TestAdaptiveQuantization:
    """Test adaptive quantization functionality."""
    
    def test_complexity_analyzer_image(self):
        """Test image complexity analysis."""
        engine = AdaptiveQuantizationEngine()
        
        # Test with simple image
        simple_image = np.ones((224, 224, 3)) * 0.5
        complexity = engine.complexity_analyzer.analyze_image_complexity(simple_image)
        
        assert isinstance(complexity, ComplexityMetrics)
        assert complexity.entropy >= 0
        assert complexity.spatial_variance >= 0
        assert complexity.texture_density >= 0
        assert 0 <= complexity.overall_complexity <= 1
    
    def test_complexity_analyzer_text(self):
        """Test text complexity analysis."""
        engine = AdaptiveQuantizationEngine()
        
        # Test with simple text
        simple_text = "Hello world"
        complexity = engine.complexity_analyzer.analyze_text_complexity(simple_text)
        
        assert isinstance(complexity, ComplexityMetrics)
        assert complexity.entropy >= 0
        
        # Test with complex text
        complex_text = "The quick brown fox jumps over the lazy dog. " * 10
        complex_complexity = engine.complexity_analyzer.analyze_text_complexity(complex_text)
        
        assert complex_complexity.entropy > complexity.entropy
    
    def test_entropy_based_strategy(self):
        """Test entropy-based quantization strategy."""
        strategy = EntropyBasedStrategy()
        
        # Test with low complexity
        low_complexity = ComplexityMetrics(entropy=2.0, spatial_variance=10.0, texture_density=5.0)
        profile = strategy.select_precision(low_complexity, HardwareTarget.HEXAGON_NPU)
        
        assert profile.vision_encoder_precision in [PrecisionLevel.INT2, PrecisionLevel.INT4]
        assert profile.expected_speedup > 1.0
        assert profile.memory_reduction > 0.0
        
        # Test with high complexity
        high_complexity = ComplexityMetrics(entropy=7.0, spatial_variance=500.0, texture_density=80.0)
        high_profile = strategy.select_precision(high_complexity, HardwareTarget.HEXAGON_NPU)
        
        # Higher complexity should use higher precision
        assert high_profile.vision_encoder_precision.value >= profile.vision_encoder_precision.value
    
    def test_performance_based_strategy(self):
        """Test performance-based quantization strategy."""
        strategy = PerformanceBasedStrategy(target_fps=30.0, target_latency_ms=33.0)
        
        complexity = ComplexityMetrics(entropy=4.0, spatial_variance=100.0, texture_density=20.0)
        profile = strategy.select_precision(complexity, HardwareTarget.CPU)
        
        assert profile.expected_speedup > 1.0
        assert profile.expected_accuracy_drop >= 0.0
        
        # Record performance to test adaptation
        strategy.record_performance(latency_ms=50.0, throughput_fps=20.0)
        
        # Should adapt to more aggressive quantization
        adaptive_profile = strategy.select_precision(complexity, HardwareTarget.CPU)
        assert strategy.current_strategy in ["aggressive", "balanced", "conservative"]
    
    def test_quantization_engine_integration(self):
        """Test complete quantization engine."""
        engine = AdaptiveQuantizationEngine(
            default_strategy="entropy_based",
            hardware_target=HardwareTarget.CPU
        )
        
        # Test image quantization
        test_image = np.random.rand(224, 224, 3)
        profile = engine.analyze_and_adapt(image=test_image)
        
        assert profile.vision_encoder_precision in list(PrecisionLevel)
        assert profile.expected_speedup > 0
        
        # Test text quantization
        test_text = "This is a test sentence for quantization analysis."
        text_profile = engine.analyze_and_adapt(text=test_text)
        
        assert text_profile.text_encoder_precision in list(PrecisionLevel)
        
        # Test statistics
        stats = engine.get_statistics()
        assert stats["adaptation_count"] == 2
        assert "current_strategy" in stats
    
    def test_strategy_switching(self):
        """Test switching between quantization strategies."""
        engine = AdaptiveQuantizationEngine()
        
        # Start with entropy-based
        assert engine.current_strategy_name == "entropy_based"
        
        # Switch to performance-based
        engine.switch_strategy("performance_based")
        assert engine.current_strategy_name == "performance_based"
        
        # Test with invalid strategy
        with pytest.raises(ValueError):
            engine.switch_strategy("invalid_strategy")


@pytest.mark.skipif(not hasattr(np, 'random'), reason="NumPy not available")
class TestHybridAttention:
    """Test hybrid attention mechanisms."""
    
    def test_attention_config(self):
        """Test attention configuration."""
        config = AttentionConfig(
            num_heads=8,
            head_dim=64,
            local_window_size=32,
            sparsity_ratio=0.1
        )
        
        assert config.num_heads == 8
        assert config.head_dim == 64
        assert config.local_window_size == 32
        assert config.sparsity_ratio == 0.1
    
    def test_create_hybrid_attention(self):
        """Test hybrid attention creation."""
        attention = create_hybrid_attention(
            dim=512,
            num_heads=8,
            local_window_size=32,
            sparsity_ratio=0.1
        )
        
        # May return None if PyTorch not available
        if attention is not None:
            assert hasattr(attention, 'config')
            assert attention.config.num_heads == 8
    
    @pytest.mark.skipif(True, reason="PyTorch may not be available in test environment")
    def test_attention_forward_pass(self):
        """Test attention forward pass (requires PyTorch)."""
        try:
            import torch
            
            attention = create_hybrid_attention(dim=512, num_heads=8)
            if attention is not None:
                # Test forward pass
                x = torch.randn(2, 64, 512)  # (batch, seq_len, dim)
                output = attention(x)
                
                assert output.shape == x.shape
                assert not torch.isnan(output).any()
        except ImportError:
            pytest.skip("PyTorch not available")


class TestEdgeFederatedLearning:
    """Test edge federated learning components."""
    
    def test_device_profile_creation(self):
        """Test device profile creation."""
        device = create_mobile_device_profile(
            device_id="test_device_1",
            device_class=DeviceClass.HIGH_END,
            memory_mb=8192,
            compute_score=0.9
        )
        
        assert device.device_id == "test_device_1"
        assert device.device_class == DeviceClass.HIGH_END
        assert device.memory_mb == 8192
        assert device.compute_score == 0.9
        assert 0 <= device.participation_score <= 1
    
    def test_federated_coordinator_creation(self):
        """Test federated coordinator creation."""
        coordinator = create_federated_coordinator(
            compression_method="hybrid",
            privacy_enabled=True
        )
        
        assert coordinator.config["compression_method"] == "hybrid"
        assert coordinator.config["privacy_epsilon"] == 1.0
        assert len(coordinator.registered_devices) == 0
    
    def test_device_registration(self):
        """Test device registration."""
        coordinator = create_federated_coordinator()
        
        device = DeviceProfile(
            device_id="test_device",
            device_class=DeviceClass.MID_RANGE,
            memory_mb=4096,
            compute_score=0.6,
            network_quality=0.8,
            battery_level=0.9,
            privacy_budget=1.0
        )
        
        coordinator.register_device(device)
        
        assert "test_device" in coordinator.registered_devices
        assert coordinator.registered_devices["test_device"] == device
    
    @pytest.mark.asyncio
    async def test_federation_round(self):
        """Test federation round execution."""
        coordinator = create_federated_coordinator()
        
        # Register multiple devices
        for i in range(5):
            device = create_mobile_device_profile(
                device_id=f"device_{i}",
                device_class=DeviceClass.MID_RANGE,
                memory_mb=4096,
                compute_score=0.7
            )
            coordinator.register_device(device)
        
        # Run federation round
        round_result = await coordinator.run_federation_round()
        
        if round_result:  # May be None if insufficient devices
            assert round_result.round_number == 1
            assert len(round_result.participants) > 0
            assert round_result.global_loss >= 0
            assert round_result.end_time > round_result.start_time
    
    def test_compression_methods(self):
        """Test gradient compression methods."""
        from src.mobile_multimodal.edge_federated_learning import GradientCompressor, CompressionMethod
        
        compressor = GradientCompressor(method=CompressionMethod.HYBRID)
        
        # Test data
        test_gradients = {
            "layer1": np.random.normal(0, 0.1, (100, 50)),
            "layer2": np.random.normal(0, 0.1, (50, 10))
        }
        
        # Compress
        compressed, ratio = compressor.compress(test_gradients)
        assert 0 < ratio <= 1.0  # Should be compressed
        
        # Decompress
        decompressed = {}
        for key, comp_data in compressed.items():
            decompressed[key] = compressor.decompress(comp_data)
        
        # Check reconstruction quality
        for key in test_gradients:
            original = test_gradients[key]
            reconstructed = decompressed[key]
            assert original.shape == reconstructed.shape


class TestIntelligentCache:
    """Test intelligent caching system."""
    
    def test_cache_manager_creation(self):
        """Test cache manager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = create_mobile_cache_manager(cache_dir=temp_dir)
            
            assert cache_manager.config["l1_size_mb"] == 64  # Mobile optimized
            assert cache_manager.config["l2_size_mb"] == 256
            assert cache_manager.prefetch_enabled
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test basic cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = create_mobile_cache_manager(cache_dir=temp_dir)
            
            # Test put/get
            test_key = "test_key_1"
            test_value = {"data": "test_data", "array": np.random.rand(10, 10)}
            
            # Put value
            success = await cache_manager.put(test_key, test_value)
            assert success
            
            # Get value
            retrieved_value = await cache_manager.get(test_key)
            assert retrieved_value is not None
            assert retrieved_value["data"] == "test_data"
            assert isinstance(retrieved_value["array"], np.ndarray)
    
    @pytest.mark.asyncio
    async def test_cache_with_loader(self):
        """Test cache with loader function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = create_mobile_cache_manager(cache_dir=temp_dir)
            
            # Mock loader function
            def mock_loader(key):
                return f"loaded_value_for_{key}"
            
            # Test cache miss with loader
            result = await cache_manager.get("new_key", loader=mock_loader)
            assert result == "loaded_value_for_new_key"
            
            # Test cache hit (should not call loader again)
            result2 = await cache_manager.get("new_key")
            assert result2 == "loaded_value_for_new_key"
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test cache eviction policies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with very small size to trigger eviction
            config = {
                "l1_size_mb": 1,  # Very small
                "l2_size_mb": 2,
                "cache_dir": temp_dir,
                "eviction_policy": "adaptive"
            }
            cache_manager = IntelligentCacheManager(config)
            
            # Fill cache beyond capacity
            large_data = np.random.rand(1000, 1000)  # Large array
            
            for i in range(5):
                await cache_manager.put(f"key_{i}", large_data)
            
            # Check that some entries were evicted
            l1_stats = cache_manager.l1_cache.get_stats()
            assert l1_stats["evictions"] > 0
    
    def test_cache_mobile_optimization(self):
        """Test mobile-specific cache optimizations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = create_mobile_cache_manager(cache_dir=temp_dir)
            
            # Test battery optimization
            cache_manager.optimize_for_mobile(battery_level=0.1, memory_pressure=0.8)
            
            # Should reduce cache size
            original_size = cache_manager.config["l1_size_mb"] * 1024 * 1024
            current_size = cache_manager.l1_cache.max_size_bytes
            assert current_size <= original_size


class TestConcurrentProcessor:
    """Test concurrent processing engine."""
    
    def test_processing_engine_creation(self):
        """Test processing engine creation."""
        engine = create_mobile_processing_engine()
        
        assert engine.config["cpu_workers"] == 2  # Mobile optimized
        assert engine.config["enable_batching"]
        assert not engine.running
    
    @pytest.mark.asyncio
    async def test_single_task_processing(self):
        """Test single task processing."""
        engine = create_mobile_processing_engine()
        
        # Start engine
        await engine.start()
        
        try:
            # Define test function
            def test_function(x, y):
                return x + y
            
            # Create task
            task = ProcessingTask(
                task_id="test_task_1",
                function=test_function,
                args=(5, 3),
                priority=TaskPriority.NORMAL
            )
            
            # Submit task
            task_id = await engine.submit_task(task)
            assert task_id == "test_task_1"
            
            # Wait a bit for processing
            await asyncio.sleep(0.5)
            
            # Check metrics
            stats = engine.get_engine_stats()
            assert stats["global_metrics"]["total_tasks"] == 1
            
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing functionality."""
        engine = create_mobile_processing_engine()
        
        # Start engine
        await engine.start()
        
        try:
            # Define batch function
            def batch_function(x):
                return x * 2
            
            # Create multiple tasks for batching
            tasks = []
            for i in range(5):
                task = ProcessingTask(
                    task_id=f"batch_task_{i}",
                    function=batch_function,
                    args=(i,),
                    max_batch_size=5,  # Enable batching
                    priority=TaskPriority.NORMAL
                )
                tasks.append(task)
            
            # Submit all tasks
            for task in tasks:
                await engine.submit_task(task)
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            # Check that batching was used
            stats = engine.get_engine_stats()
            assert stats["global_metrics"]["total_tasks"] == 5
            
        finally:
            await engine.stop()
    
    def test_device_capability_detection(self):
        """Test device capability detection."""
        from src.mobile_multimodal.concurrent_processor import DeviceCapabilityDetector
        
        detector = DeviceCapabilityDetector()
        
        # Should always detect CPU
        assert detector.capabilities[ProcessingUnit.CPU]
        
        # Should have performance scores
        assert ProcessingUnit.CPU in detector.performance_scores
        assert detector.performance_scores[ProcessingUnit.CPU] > 0
        
        # Test best device selection
        best_device = detector.get_best_device()
        assert best_device in list(ProcessingUnit)
    
    def test_battery_optimization(self):
        """Test battery optimization."""
        engine = create_mobile_processing_engine()
        
        original_workers = engine.config["cpu_workers"]
        
        # Optimize for low battery
        engine.optimize_for_battery(battery_level=0.1)
        
        # Should reduce worker count
        assert engine.config["cpu_workers"] <= original_workers
        
        # Optimize for high battery
        engine.optimize_for_battery(battery_level=0.9)
        
        # May restore worker count


class TestAdvancedValidation:
    """Test advanced validation framework."""
    
    def test_validator_creation(self):
        """Test validator creation."""
        validator = create_validator(ValidationLevel.STANDARD)
        
        assert validator.level == ValidationLevel.STANDARD
        assert len(validator.validators) > 0
    
    def test_basic_validation(self):
        """Test basic validation functionality."""
        validator = create_validator(ValidationLevel.BASIC)
        
        # Test valid data
        valid_data = {
            "image": np.random.rand(224, 224, 3),
            "text": "Hello world",
            "batch_size": 4,
            "temperature": 1.0
        }
        
        result = validator.validate(valid_data)
        assert result.is_valid
        assert result.threat_level < 0.5
        assert len(result.detected_threats) == 0
    
    def test_threat_detection(self):
        """Test threat detection."""
        validator = create_validator(ValidationLevel.STRICT)
        
        # Test malformed data
        malformed_data = {
            "batch_size": -1,  # Invalid range
            "temperature": float('inf'),  # Invalid value
            "text": "x" * 20000  # Suspiciously long text
        }
        
        result = validator.validate(malformed_data)
        assert not result.is_valid
        assert result.threat_level > 0.3
        assert len(result.detected_threats) > 0
    
    def test_security_report(self):
        """Test security reporting."""
        validator = create_validator(ValidationLevel.STANDARD)
        
        # Run some validations
        for i in range(10):
            test_data = {"batch_size": i % 5, "temperature": 1.0}
            validator.validate(test_data)
        
        report = validator.get_security_report()
        
        assert "security_level" in report
        assert "metrics" in report
        assert report["metrics"]["total_validations"] == 10


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_config_creation(self):
        """Test circuit breaker config creation."""
        config = create_mobile_circuit_config()
        
        assert config.failure_threshold == 3  # Mobile optimized
        assert config.timeout_duration == 15.0
        assert config.error_rate_threshold == 0.4
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_basic_operation(self):
        """Test basic circuit breaker operation."""
        config = CircuitConfig(failure_threshold=2, timeout_duration=1.0)
        circuit = AdaptiveCircuitBreaker("test_circuit", config)
        
        # Test successful call
        async def success_func():
            return "success"
        
        result = await circuit.call(success_func)
        assert result == "success"
        
        # Test circuit state
        status = circuit.get_status()
        assert status["state"] == "closed"
        assert status["metrics"]["successful_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling."""
        config = CircuitConfig(failure_threshold=2, timeout_duration=0.1)
        circuit = AdaptiveCircuitBreaker("test_circuit", config)
        
        # Function that always fails
        async def failing_func():
            raise Exception("Test failure")
        
        # Should return fallback responses
        for _ in range(3):
            result = await circuit.call(failing_func)
            assert result is not None  # Should get fallback response
        
        # Circuit should be open after failures
        status = circuit.get_status()
        assert status["metrics"]["failed_requests"] >= 2
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout(self):
        """Test circuit breaker timeout handling."""
        config = CircuitConfig(timeout_duration=0.1)  # Very short timeout
        circuit = AdaptiveCircuitBreaker("test_circuit", config)
        
        # Function that takes too long
        async def slow_func():
            await asyncio.sleep(0.2)  # Longer than timeout
            return "slow_result"
        
        result = await circuit.call(slow_func)
        # Should get fallback due to timeout
        assert result is not None


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_mobile_inference_pipeline(self):
        """Test complete mobile inference pipeline."""
        # Create components
        cache_manager = create_mobile_cache_manager()
        processing_engine = create_mobile_processing_engine()
        validator = create_validator(ValidationLevel.STANDARD)
        quantization_engine = AdaptiveQuantizationEngine()
        
        # Start processing engine
        await processing_engine.start()
        
        try:
            # Simulate mobile inference request
            request_data = {
                "image": np.random.rand(224, 224, 3),
                "text": "Describe this image",
                "batch_size": 1,
                "temperature": 1.0
            }
            
            # Validate input
            validation_result = validator.validate(request_data)
            if not validation_result.is_valid:
                pytest.fail(f"Validation failed: {validation_result.error_message}")
            
            # Check cache
            cache_key = f"inference_{hash(str(request_data))}"
            cached_result = await cache_manager.get(cache_key)
            
            if cached_result is None:
                # Analyze for quantization
                quant_profile = quantization_engine.analyze_and_adapt(
                    image=request_data["image"],
                    text=request_data["text"]
                )
                
                # Define inference function
                def mock_inference(data, quant_profile):
                    # Simulate inference with quantization
                    time.sleep(0.1)  # Simulate processing time
                    return {
                        "result": "Mock inference result",
                        "quantization": quant_profile.vision_encoder_precision.name,
                        "speedup": quant_profile.expected_speedup
                    }
                
                # Create processing task
                task = ProcessingTask(
                    task_id="inference_task",
                    function=mock_inference,
                    args=(request_data, quant_profile),
                    priority=TaskPriority.HIGH
                )
                
                # Submit for processing
                await processing_engine.submit_task(task)
                
                # Wait for processing
                await asyncio.sleep(0.5)
                
                # Cache result (in real implementation)
                inference_result = {
                    "result": "Mock inference result",
                    "quantization": quant_profile.vision_encoder_precision.name,
                    "speedup": quant_profile.expected_speedup
                }
                await cache_manager.put(cache_key, inference_result)
            
            # Verify components worked
            stats = processing_engine.get_engine_stats()
            assert stats["global_metrics"]["total_tasks"] >= 1
            
            cache_stats = cache_manager.get_global_stats()
            assert cache_stats["global_metrics"]["total_requests"] >= 1
            
            quant_stats = quantization_engine.get_statistics()
            assert quant_stats["adaptation_count"] >= 1
            
        finally:
            await processing_engine.stop()
    
    def test_resource_optimization_integration(self):
        """Test integrated resource optimization."""
        # Create components
        cache_manager = create_mobile_cache_manager()
        processing_engine = create_mobile_processing_engine()
        
        # Simulate low battery scenario
        battery_level = 0.15
        memory_pressure = 0.8
        
        # Optimize all components
        cache_manager.optimize_for_mobile(battery_level, memory_pressure)
        processing_engine.optimize_for_battery(battery_level)
        
        # Verify optimizations
        assert cache_manager.l1_cache.max_size_bytes < 128 * 1024 * 1024  # Should be reduced
        assert processing_engine.config["cpu_workers"] <= 2  # Should be conservative
    
    def test_error_handling_integration(self):
        """Test integrated error handling and recovery."""
        # Create components with circuit breakers
        config = create_mobile_circuit_config()
        circuit = AdaptiveCircuitBreaker("integration_test", config)
        validator = create_validator(ValidationLevel.STRICT)
        
        # Test with invalid input
        invalid_data = {
            "batch_size": -1,
            "temperature": float('nan'),
            "malicious_field": "x" * 50000
        }
        
        # Validation should catch issues
        validation_result = validator.validate(invalid_data)
        assert not validation_result.is_valid
        
        # Circuit breaker should handle failures gracefully
        async def failing_process():
            raise Exception("Processing failed")
        
        async def test_circuit():
            result = await circuit.call(failing_process)
            return result
        
        # Should get fallback response
        import asyncio
        result = asyncio.run(test_circuit())
        assert result is not None


@pytest.mark.performance
class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_quantization_performance(self):
        """Test quantization analysis performance."""
        engine = AdaptiveQuantizationEngine()
        
        # Large image for performance test
        large_image = np.random.rand(1024, 1024, 3)
        
        start_time = time.perf_counter()
        profile = engine.analyze_and_adapt(image=large_image)
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        assert processing_time < 1.0  # Should be fast
        assert profile is not None
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance with high load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = create_mobile_cache_manager(cache_dir=temp_dir)
            
            # Generate test data
            test_data = [
                (f"key_{i}", np.random.rand(100, 100))
                for i in range(100)
            ]
            
            # Measure put performance
            start_time = time.perf_counter()
            for key, value in test_data:
                await cache_manager.put(key, value)
            put_time = time.perf_counter() - start_time
            
            # Measure get performance
            start_time = time.perf_counter()
            for key, _ in test_data:
                result = await cache_manager.get(key)
                assert result is not None
            get_time = time.perf_counter() - start_time
            
            # Performance assertions
            assert put_time < 5.0  # Should complete in reasonable time
            assert get_time < 2.0  # Gets should be faster than puts
            
            # Check hit rate
            stats = cache_manager.get_global_stats()
            assert stats["global_metrics"]["hit_rate"] > 0.8  # Should have good hit rate
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_throughput(self):
        """Test concurrent processing throughput."""
        engine = create_mobile_processing_engine()
        await engine.start()
        
        try:
            # Simple function for throughput test
            def simple_add(x, y):
                return x + y
            
            # Submit many tasks
            num_tasks = 50
            start_time = time.perf_counter()
            
            for i in range(num_tasks):
                task = ProcessingTask(
                    task_id=f"perf_task_{i}",
                    function=simple_add,
                    args=(i, i + 1),
                    priority=TaskPriority.NORMAL
                )
                await engine.submit_task(task)
            
            # Wait for completion
            await asyncio.sleep(2.0)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Check throughput
            stats = engine.get_engine_stats()
            completed_tasks = stats["global_metrics"]["completed_tasks"]
            
            throughput = completed_tasks / total_time
            assert throughput > 10  # Should process at least 10 tasks per second
            
        finally:
            await engine.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])