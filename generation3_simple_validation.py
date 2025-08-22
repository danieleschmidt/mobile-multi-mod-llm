#!/usr/bin/env python3
"""Generation 3 Simple Validation: Test existing scaling components."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_existing_components():
    """Test the existing advanced components that are already implemented."""
    try:
        from mobile_multimodal.intelligent_cache import IntelligentCache
        from mobile_multimodal.auto_scaling import AutoScaler
        from mobile_multimodal.concurrent_processor import ConcurrentProcessor
        from mobile_multimodal.adaptive_quantization import AdaptiveQuantizer
        from mobile_multimodal.distributed_inference import DistributedInferenceEngine
        from mobile_multimodal.batch_processor import BatchProcessor
        from mobile_multimodal.realtime_pipeline import RealtimePipeline
        
        # Test intelligent cache
        cache = IntelligentCache(max_size=1000)
        cache.set("test_key", "test_value", ttl=3600)
        result = cache.get("test_key")
        assert result == "test_value", "Cache test failed"
        
        # Test auto scaler
        scaler = AutoScaler()
        # Test that it was created successfully
        assert scaler.current_instances >= 1, "Auto scaler test failed"
        
        # Test concurrent processor
        processor = ConcurrentProcessor(max_workers=4)
        
        # Test adaptive quantizer
        quantizer = AdaptiveQuantizer(compression_ratio=0.25)
        
        # Test distributed inference
        engine = DistributedInferenceEngine()
        
        # Test batch processor
        batch_proc = BatchProcessor(batch_size=8, max_latency_ms=100)
        
        # Test realtime pipeline
        pipeline = RealtimePipeline(max_latency_ms=50)
        
        print("✅ All existing Generation 3 components validated")
        return True
        
    except Exception as e:
        print(f"❌ Generation 3 components test failed: {e}")
        return False

def test_performance_characteristics():
    """Test performance characteristics of the system."""
    try:
        from mobile_multimodal import MobileMultiModalLLM
        from mobile_multimodal.monitoring import ResourceMonitor
        from mobile_multimodal.intelligent_cache import IntelligentCache
        
        # Test optimized model loading
        start_time = time.time()
        model = MobileMultiModalLLM(
            device="cpu", 
            enable_optimization=True,
            optimization_profile="fast"
        )
        load_time = time.time() - start_time
        
        # Test health check performance
        start_time = time.time()
        health = model.get_health_status()
        health_time = time.time() - start_time
        
        # Test cache performance
        cache = IntelligentCache(max_size=10000)
        start_time = time.time()
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
        cache_write_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(100):
            cache.get(f"key_{i}")
        cache_read_time = time.time() - start_time
        
        print(f"✅ Performance characteristics validated:")
        print(f"   - Model load time: {load_time:.3f}s")
        print(f"   - Health check time: {health_time:.3f}s")
        print(f"   - Cache write (100 items): {cache_write_time:.3f}s")
        print(f"   - Cache read (100 items): {cache_read_time:.3f}s")
        
        # Basic performance assertions
        assert load_time < 5.0, f"Model loading too slow: {load_time}s"
        assert health_time < 1.0, f"Health check too slow: {health_time}s"
        assert cache_write_time < 0.1, f"Cache writing too slow: {cache_write_time}s"
        assert cache_read_time < 0.1, f"Cache reading too slow: {cache_read_time}s"
        
        return True
        
    except Exception as e:
        print(f"❌ Performance characteristics test failed: {e}")
        return False

def test_scaling_behaviors():
    """Test that scaling-related behaviors work correctly."""
    try:
        from mobile_multimodal.concurrent_processor import ConcurrentProcessor, ThreadSafeCache
        from mobile_multimodal.intelligent_cache import IntelligentCache
        
        # Test concurrent processing with multiple workers
        processor = ConcurrentProcessor(max_workers=4, queue_size=100)
        
        def test_task(item):
            return item ** 2
        
        # Test with various batch sizes
        for batch_size in [1, 10, 50]:
            tasks = list(range(batch_size))
            start_time = time.time()
            results = processor.process_concurrent(tasks, test_task)
            process_time = time.time() - start_time
            
            assert len(results) == batch_size, f"Batch size {batch_size} failed"
            print(f"   - Batch size {batch_size}: {process_time:.3f}s")
        
        # Test cache scaling
        cache = ThreadSafeCache(max_size=1000)
        for i in range(500):
            cache.set(f"scale_test_{i}", f"value_{i}")
        
        assert cache.size() == 500, "Cache scaling failed"
        
        # Test cache overflow behavior
        for i in range(600):  # This should trigger eviction
            cache.set(f"overflow_test_{i}", f"value_{i}")
        
        assert cache.size() <= 1000, "Cache size limit not respected"
        
        print("✅ Scaling behaviors validated")
        return True
        
    except Exception as e:
        print(f"❌ Scaling behaviors test failed: {e}")
        return False

def main():
    """Run Generation 3 simple validation tests."""
    print("⚡ GENERATION 3 SIMPLE VALIDATION: MAKE IT SCALE")
    print("=" * 60)
    
    tests = [
        test_existing_components,
        test_performance_characteristics,
        test_scaling_behaviors
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
        print(f"⚠️  GENERATION 3: {failed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)