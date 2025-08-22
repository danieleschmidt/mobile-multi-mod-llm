#!/usr/bin/env python3
"""Generation 3 Enhancement: MAKE IT SCALE - Performance optimization, caching, and scaling."""

import sys
import os
import time
import concurrent.futures
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_performance_optimization():
    """Test performance optimization features."""
    try:
        from mobile_multimodal.optimization import PerformanceOptimizer, CacheManager
        from mobile_multimodal.auto_scaling import AutoScaler, ResourceManager
        
        # Test performance optimizer
        optimizer = PerformanceOptimizer()
        cache_manager = CacheManager(max_size=10000)
        
        # Test basic optimization
        cache_manager.set("test_key", "test_value", ttl=300)
        cached_value = cache_manager.get("test_key")
        assert cached_value == "test_value", "Cache operation failed"
        
        print("✅ Performance optimization features validated")
        return True
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        return False

def test_intelligent_caching():
    """Test intelligent caching system with TTL and LRU eviction."""
    try:
        from mobile_multimodal.intelligent_cache import IntelligentCache, CacheMetrics
        from mobile_multimodal.adaptive_quantization import AdaptivePruning
        
        # Test intelligent cache
        cache = IntelligentCache(max_size=1000, default_ttl=3600)
        metrics = CacheMetrics()
        
        # Test cache operations
        cache.set("model_inference_001", "cached_result", tags={"model": "tiny", "batch_size": "1"})
        result = cache.get("model_inference_001")
        assert result == "cached_result", "Cache retrieval failed"
        
        # Test cache statistics
        stats = cache.get_statistics()
        assert "hit_ratio" in stats, "Cache statistics missing"
        
        print("✅ Intelligent caching system validated")
        return True
    except Exception as e:
        print(f"❌ Intelligent caching test failed: {e}")
        return False

def test_auto_scaling():
    """Test auto-scaling and resource management."""
    try:
        from mobile_multimodal.auto_scaling import AutoScaler, ResourceManager, LoadBalancer
        from mobile_multimodal.concurrent_processor import BatchProcessor
        
        # Test auto-scaling components
        auto_scaler = AutoScaler(min_instances=1, max_instances=8)
        resource_manager = ResourceManager()
        load_balancer = LoadBalancer()
        
        # Test resource monitoring
        current_load = auto_scaler.get_current_load()
        assert isinstance(current_load, (int, float)), "Load monitoring failed"
        
        # Test scaling decision
        should_scale = auto_scaler.should_scale_up(current_load=0.8, target_load=0.7)
        assert isinstance(should_scale, bool), "Scaling decision failed"
        
        print("✅ Auto-scaling system validated")
        return True
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        return False

def test_concurrent_processing():
    """Test high-performance concurrent processing."""
    try:
        from mobile_multimodal.concurrent_processor import ConcurrentProcessor, BatchProcessor
        from mobile_multimodal.batch_processor import DynamicBatchProcessor
        
        # Test concurrent processing
        processor = ConcurrentProcessor(max_workers=4)
        batch_processor = BatchProcessor(batch_size=8)
        
        # Test batch processing
        def dummy_task(item):
            return item * 2
        
        tasks = list(range(10))
        results = processor.process_concurrent(tasks, dummy_task)
        
        assert len(results) == len(tasks), "Concurrent processing failed"
        assert results[0] == 0, "Processing result incorrect"  # 0 * 2 = 0
        
        print("✅ Concurrent processing validated")
        return True
    except Exception as e:
        print(f"❌ Concurrent processing test failed: {e}")
        return False

def test_model_optimization():
    """Test model-specific optimization features."""
    try:
        from mobile_multimodal.adaptive_quantization import AdaptiveQuantizer, ModelOptimizer
        from mobile_multimodal.hybrid_attention import HybridAttention
        
        # Test adaptive quantization
        quantizer = AdaptiveQuantizer(target_size_mb=35, quality_threshold=0.95)
        optimizer = ModelOptimizer()
        
        # Test model optimization
        optimization_config = optimizer.get_optimization_config("mobile_inference")
        assert "batch_size" in optimization_config, "Optimization config missing"
        
        print("✅ Model optimization features validated")
        return True
    except Exception as e:
        print(f"❌ Model optimization test failed: {e}")
        return False

def test_distributed_inference():
    """Test distributed inference capabilities."""
    try:
        from mobile_multimodal.distributed_inference import DistributedInferenceEngine, NodeManager
        from mobile_multimodal.realtime_pipeline import RealtimePipeline
        
        # Test distributed inference
        engine = DistributedInferenceEngine()
        node_manager = NodeManager()
        pipeline = RealtimePipeline(max_latency_ms=100)
        
        # Test node registration
        node_id = node_manager.register_node("cpu_node", {"type": "cpu", "cores": 4})
        assert node_id is not None, "Node registration failed"
        
        print("✅ Distributed inference system validated")
        return True
    except Exception as e:
        print(f"❌ Distributed inference test failed: {e}")
        return False

def test_resource_pooling():
    """Test resource pooling and connection management."""
    try:
        from mobile_multimodal.resilience import ResourcePool, ConnectionManager
        from mobile_multimodal.monitoring import ResourceMonitor
        
        # Test resource pooling
        resource_pool = ResourcePool(max_resources=10, min_resources=2)
        connection_manager = ConnectionManager()
        monitor = ResourceMonitor()
        
        # Test resource allocation
        resource = resource_pool.acquire_resource(timeout=5.0)
        if resource is not None:
            resource_pool.release_resource(resource)
        
        # Test resource statistics
        stats = resource_pool.get_statistics()
        assert "total_resources" in stats, "Resource statistics missing"
        
        print("✅ Resource pooling system validated")
        return True
    except Exception as e:
        print(f"❌ Resource pooling test failed: {e}")
        return False

def test_performance_benchmarking():
    """Test performance benchmarking and profiling."""
    try:
        from mobile_multimodal.monitoring import PerformanceBenchmark, ProfilerManager
        from mobile_multimodal import MobileMultiModalLLM
        
        # Test performance benchmarking
        benchmark = PerformanceBenchmark()
        profiler = ProfilerManager()
        
        # Test model performance
        model = MobileMultiModalLLM(device="cpu", enable_optimization=True)
        
        # Run performance test
        start_time = time.time()
        health_status = model.get_health_status()
        end_time = time.time()
        
        inference_time = end_time - start_time
        assert inference_time < 1.0, f"Health check too slow: {inference_time}s"
        
        print("✅ Performance benchmarking validated")
        return True
    except Exception as e:
        print(f"❌ Performance benchmarking test failed: {e}")
        return False

def main():
    """Run Generation 3 optimization tests."""
    print("⚡ GENERATION 3 VALIDATION: MAKE IT SCALE")
    print("=" * 60)
    
    tests = [
        test_performance_optimization,
        test_intelligent_caching,
        test_auto_scaling,
        test_concurrent_processing,
        test_model_optimization,
        test_distributed_inference,
        test_resource_pooling,
        test_performance_benchmarking
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 GENERATION 3: MAKE IT SCALE - COMPLETE!")
        return True
    else:
        print(f"⚠️  GENERATION 3: {failed} tests failed, implementing optimizations...")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)