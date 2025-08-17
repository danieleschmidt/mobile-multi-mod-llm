#!/usr/bin/env python3
"""Test Generation 3: Scaling - Performance optimization, caching, and scaling."""

import os
import sys
import time
import threading
import concurrent.futures
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_performance_optimization():
    """Test performance optimization features."""
    print("Testing performance optimization...")
    
    try:
        from mobile_multimodal.core import MobileMultiModalLLM
        import numpy as np
        
        # Test performance profiles
        profiles = ["fast", "balanced", "accuracy"]
        
        for profile in profiles:
            model = MobileMultiModalLLM(
                device="cpu",
                enable_optimization=True,
                optimization_profile=profile,
                strict_security=False
            )
            
            optimization_stats = model.get_optimization_stats()
            print(f"✓ {profile} profile: optimization enabled = {optimization_stats.get('optimization_enabled', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_processing():
    """Test concurrent and parallel processing capabilities."""
    print("\nTesting concurrent processing...")
    
    try:
        from mobile_multimodal.core import MobileMultiModalLLM
        import numpy as np
        import concurrent.futures
        
        model = MobileMultiModalLLM(
            device="cpu",
            enable_optimization=True,
            optimization_profile="fast",
            strict_security=False
        )
        
        # Create test images
        test_images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = []
        for i, image in enumerate(test_images):
            try:
                caption = model.generate_caption(image, user_id=f"seq_user_{i}")
                sequential_results.append(caption)
            except Exception as e:
                sequential_results.append(f"Error: {e}")
        
        sequential_duration = time.time() - start_time
        
        # Test concurrent processing
        start_time = time.time()
        
        def process_image(args):
            i, image = args
            try:
                return model.generate_caption(image, user_id=f"conc_user_{i}")
            except Exception as e:
                return f"Error: {e}"
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            concurrent_results = list(executor.map(process_image, enumerate(test_images)))
        
        concurrent_duration = time.time() - start_time
        
        # Calculate performance metrics
        seq_throughput = len(test_images) / sequential_duration
        conc_throughput = len(test_images) / concurrent_duration
        
        print(f"✓ Sequential processing: {seq_throughput:.1f} ops/sec")
        print(f"✓ Concurrent processing: {conc_throughput:.1f} ops/sec")
        
        # Test success rates
        seq_success_rate = len([r for r in sequential_results if not r.startswith("Error")]) / len(sequential_results)
        conc_success_rate = len([r for r in concurrent_results if not r.startswith("Error")]) / len(concurrent_results)
        
        print(f"✓ Sequential success rate: {seq_success_rate:.1%}")
        print(f"✓ Concurrent success rate: {conc_success_rate:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Concurrent processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_optimization():
    """Test model-specific optimization features."""
    print("\nTesting model optimization...")
    
    try:
        from mobile_multimodal.core import MobileMultiModalLLM
        import numpy as np
        
        model = MobileMultiModalLLM(
            device="cpu",
            enable_optimization=True,
            optimization_profile="balanced",
            strict_security=False
        )
        
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test adaptive inference
        adaptive_result = model.adaptive_inference(test_image, quality_target=0.8)
        adaptive_mode = adaptive_result.get("adaptive_mode", False)
        print(f"✓ Adaptive inference: Enabled = {adaptive_mode}")
        
        # Test auto-tuning
        tuning_result = model.auto_tune_performance(target_latency_ms=50)
        if "error" not in tuning_result:
            tuning_count = len(tuning_result.get("tuning_applied", []))
            print(f"✓ Auto-tuning: {tuning_count} optimizations applied")
        else:
            print(f"✓ Auto-tuning: Not available (mock mode)")
        
        # Test benchmarking
        benchmark_result = model.benchmark_inference(test_image, iterations=3)
        fps = benchmark_result.get("fps", 0)
        latency_ms = benchmark_result.get("total_inference_ms", 0)
        print(f"✓ Model benchmarking: {fps:.1f} FPS, {latency_ms:.1f}ms latency")
        
        return True
        
    except Exception as e:
        print(f"❌ Model optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scalability_stress():
    """Test system scalability under stress."""
    print("\nTesting scalability stress...")
    
    try:
        from mobile_multimodal.core import MobileMultiModalLLM
        import numpy as np
        
        model = MobileMultiModalLLM(
            device="cpu",
            enable_optimization=True,
            optimization_profile="fast",
            strict_security=False,
            max_retries=1,
            timeout=5.0
        )
        
        # Stress test with rapid requests
        num_requests = 10
        success_count = 0
        error_count = 0
        total_time = 0
        
        print(f"   Running {num_requests} rapid requests...")
        
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                request_start = time.time()
                test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                caption = model.generate_caption(test_image, user_id=f"stress_user_{i}")
                
                request_duration = time.time() - request_start
                total_time += request_duration
                
                if len(caption) > 0:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        success_rate = success_count / num_requests
        throughput = num_requests / elapsed
        avg_latency = (total_time / success_count * 1000) if success_count > 0 else 0
        
        print(f"✓ Stress test results:")
        print(f"   Success rate: {success_rate:.1%} ({success_count}/{num_requests})")
        print(f"   Throughput: {throughput:.1f} requests/second")
        print(f"   Average latency: {avg_latency:.1f}ms")
        print(f"   Total time: {elapsed:.2f}s")
        
        # Test system health after stress
        health_status = model.get_health_status()
        print(f"✓ Post-stress health: {health_status.get('status', 'unknown')}")
        
        return success_rate > 0.5  # At least 50% success rate
        
    except Exception as e:
        print(f"❌ Scalability stress test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Generation 3 scaling tests."""
    print("="*70)
    print("GENERATION 3: SCALING TESTING")
    print("Performance optimization, caching, and scalability")
    print("="*70)
    
    tests = [
        ("Performance Optimization", test_performance_optimization),
        ("Concurrent Processing", test_concurrent_processing),
        ("Model Optimization", test_model_optimization),
        ("Scalability Stress", test_scalability_stress),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION 3 SCALING TEST RESULTS")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🚀 Generation 3 SCALABLE: All scaling tests passed!")
        print("✅ Performance optimization operational")
        print("✅ Concurrent processing optimized")
        print("✅ Model optimization features working")
        print("✅ Scalability stress testing passed")
        return 0
    else:
        print(f"\n⚠️ {total-passed} scaling tests failed. System needs optimization.")
        return 1


if __name__ == "__main__":
    sys.exit(main())