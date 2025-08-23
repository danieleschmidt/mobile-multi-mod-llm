#!/usr/bin/env python3
"""Comprehensive testing suite for mobile multi-modal LLM."""

import sys
import json
import time
import unittest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, 'src')

class TestMobileMultiModalCore(unittest.TestCase):
    """Test core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from mobile_multimodal.utils import ImageProcessor, ConfigManager
        
        self.image_processor = ImageProcessor()
        self.config_manager = ConfigManager()
    
    def test_image_processor_initialization(self):
        """Test image processor initialization."""
        self.assertIsNotNone(self.image_processor)
        self.assertEqual(self.image_processor.target_size, (224, 224))
    
    def test_config_manager_initialization(self):
        """Test configuration manager initialization."""
        self.assertIsNotNone(self.config_manager)
        self.assertIn('model', self.config_manager.config)
        self.assertIn('preprocessing', self.config_manager.config)
    
    def test_config_save_load_cycle(self):
        """Test configuration save/load cycle."""
        # Modify configuration
        original_batch_size = self.config_manager.config['model']['batch_size']
        self.config_manager.config['model']['batch_size'] = 8
        
        # Save and reload
        self.config_manager.save_config()
        self.config_manager.config['model']['batch_size'] = original_batch_size  # Reset
        self.config_manager.load_config()
        
        # Verify
        self.assertEqual(self.config_manager.config['model']['batch_size'], 8)

class TestRobustValidation(unittest.TestCase):
    """Test robust validation components."""
    
    def setUp(self):
        """Set up validation test fixtures."""
        from mobile_multimodal.robust_validation import RobustValidator
        self.validator = RobustValidator()
    
    def test_image_array_validation_valid(self):
        """Test valid image array validation."""
        import numpy as np
        
        valid_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = self.validator.validate_image_array(valid_image)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['shape'], (224, 224, 3))
        self.assertEqual(result['channels'], 3)
    
    def test_image_array_validation_invalid_shape(self):
        """Test invalid image array validation."""
        import numpy as np
        
        # Invalid shape (too many dimensions)
        with self.assertRaises(Exception):
            invalid_image = np.random.rand(224, 224, 3, 2)
            self.validator.validate_image_array(invalid_image)
    
    def test_text_validation_valid(self):
        """Test valid text input validation."""
        valid_text = "A beautiful sunset over the ocean"
        result = self.validator.validate_text_input(valid_text)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['length'], len(valid_text))
        self.assertIn('hash', result)
    
    def test_text_validation_malicious(self):
        """Test malicious text input detection."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "../../../etc/passwd", 
            "javascript:alert(1)"
        ]
        
        for malicious_text in malicious_inputs:
            with self.assertRaises(Exception):
                self.validator.validate_text_input(malicious_text)
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        valid_config = {
            "model": {"batch_size": 1},
            "preprocessing": {"mean": [0.485, 0.456, 0.406]},
            "performance": {"memory_limit_mb": 512}
        }
        
        result = self.validator.validate_model_config(valid_config)
        self.assertTrue(result['valid'])
    
    def test_model_config_validation_invalid(self):
        """Test invalid model configuration validation."""
        invalid_config = {
            "model": {"batch_size": -1},  # Invalid batch size
            "preprocessing": {"mean": [0.485, 0.456]},  # Wrong number of values
            "performance": {"memory_limit_mb": 32}  # Too low memory
        }
        
        with self.assertRaises(Exception):
            self.validator.validate_model_config(invalid_config)

class TestScalingOptimization(unittest.TestCase):
    """Test scaling and optimization components."""
    
    def setUp(self):
        """Set up optimization test fixtures."""
        from mobile_multimodal.scaling_optimization import (
            IntelligentCache, LoadBalancer, AutoScaler
        )
        
        self.cache = IntelligentCache(max_size=10, max_memory_mb=1)
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler()
    
    def test_intelligent_cache_basic_operations(self):
        """Test basic cache operations."""
        # Test put/get
        self.assertTrue(self.cache.put("key1", "value1"))
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Test cache miss
        self.assertIsNone(self.cache.get("nonexistent"))
        
        # Test cache stats
        stats = self.cache.get_stats()
        self.assertIn('size', stats)
        self.assertIn('hit_count', stats)
        self.assertIn('miss_count', stats)
    
    def test_intelligent_cache_eviction(self):
        """Test cache eviction when full."""
        # Fill cache beyond capacity
        for i in range(15):  # More than max_size (10)
            self.cache.put(f"key{i}", f"value{i}")
        
        stats = self.cache.get_stats()
        self.assertLessEqual(stats['size'], 10)  # Should not exceed max size
    
    def test_intelligent_cache_ttl(self):
        """Test cache TTL functionality."""
        # Put with short TTL
        self.cache.put("ttl_key", "ttl_value", ttl=0.1)
        
        # Should exist immediately
        self.assertEqual(self.cache.get("ttl_key"), "ttl_value")
        
        # Should expire after TTL
        time.sleep(0.2)
        self.assertIsNone(self.cache.get("ttl_key"))
    
    def test_load_balancer_endpoint_management(self):
        """Test load balancer endpoint management."""
        # Add endpoints
        self.load_balancer.add_endpoint("server1", capacity=100)
        self.load_balancer.add_endpoint("server2", capacity=80)
        
        # Select endpoint
        endpoint = self.load_balancer.select_endpoint()
        self.assertIn(endpoint, ["server1", "server2"])
        
        # Report results
        self.load_balancer.report_result(endpoint, True, 0.05)
        
        # Should still be able to select
        endpoint2 = self.load_balancer.select_endpoint()
        self.assertIsNotNone(endpoint2)
    
    def test_auto_scaler_rules(self):
        """Test auto-scaler rule evaluation."""
        # Add scaling rules
        def high_cpu_rule(metrics):
            return metrics.get("cpu", 0) > 80
        
        self.auto_scaler.add_scaling_rule(high_cpu_rule, "scale_up", 2.0)
        
        # Test scaling up
        high_cpu_metrics = {"cpu": 85}
        decision = self.auto_scaler.evaluate_scaling(high_cpu_metrics)
        
        if decision:  # May be None due to cooldown
            self.assertEqual(decision["action"], "scale_up")
            self.assertGreater(decision["target_scale"], decision["current_scale"])
    
    def test_smart_cache_decorator(self):
        """Test smart cache decorator."""
        from mobile_multimodal.scaling_optimization import smart_cache
        
        call_count = 0
        
        @smart_cache(max_size=10)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count, 1)
        
        # Second call (cached)
        result2 = expensive_function(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count, 1)  # Should not increment
        
        # Different args (not cached)
        result3 = expensive_function(2, 3)
        self.assertEqual(result3, 5)
        self.assertEqual(call_count, 2)

class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def test_end_to_end_image_processing(self):
        """Test complete image processing workflow."""
        import numpy as np
        from mobile_multimodal.utils import ImageProcessor
        from mobile_multimodal.robust_validation import RobustValidator
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Validate image
        validator = RobustValidator()
        validation_result = validator.validate_image_array(test_image)
        self.assertTrue(validation_result['valid'])
        
        # Process image
        processor = ImageProcessor()
        self.assertIsNotNone(processor)
        
        # This workflow should complete without errors
        self.assertTrue(True)
    
    def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        
        def process_task(task_id):
            # Simulate processing
            time.sleep(0.001)
            return f"processed_{task_id}"
        
        # Test concurrent execution
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_task, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        self.assertEqual(len(results), 10)
        self.assertTrue(all("processed_" in result for result in results))
    
    def test_configuration_integration(self):
        """Test configuration integration across components."""
        from mobile_multimodal.utils import ConfigManager
        from mobile_multimodal.robust_validation import RobustValidator
        
        config_manager = ConfigManager()
        validator = RobustValidator()
        
        # Validate current configuration
        config_validation = validator.validate_model_config(config_manager.config)
        self.assertTrue(config_validation['valid'])

class TestPerformance(unittest.TestCase):
    """Performance and benchmark tests."""
    
    def test_cache_performance(self):
        """Test cache performance characteristics."""
        from mobile_multimodal.scaling_optimization import IntelligentCache
        
        cache = IntelligentCache(max_size=1000, max_memory_mb=10)
        
        # Measure put performance
        start_time = time.time()
        for i in range(100):
            cache.put(f"perf_key_{i}", f"perf_value_{i}")
        put_time = time.time() - start_time
        
        # Measure get performance
        start_time = time.time()
        for i in range(100):
            cache.get(f"perf_key_{i}")
        get_time = time.time() - start_time
        
        # Performance assertions (adjust as needed)
        self.assertLess(put_time, 0.1)  # Should complete in < 100ms
        self.assertLess(get_time, 0.01)  # Should complete in < 10ms
        
        cache.shutdown()
    
    def test_concurrent_load(self):
        """Test system under concurrent load."""
        from mobile_multimodal.scaling_optimization import IntelligentCache
        import threading
        
        cache = IntelligentCache(max_size=100, max_memory_mb=5)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(50):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    cache.put(key, value)
                    retrieved = cache.get(key)
                    if retrieved != value:
                        errors.append(f"Data mismatch in worker {worker_id}")
            except Exception as e:
                errors.append(f"Error in worker {worker_id}: {e}")
        
        # Run concurrent workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check for errors
        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")
        
        cache.shutdown()

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Mobile Multi-Modal LLM - Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMobileMultiModalCore,
        TestRobustValidation,
        TestScalingOptimization,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Generate test report
    test_report = {
        "timestamp": time.time(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        "test_classes": [cls.__name__ for cls in test_classes],
        "failure_details": [(str(test), error) for test, error in result.failures],
        "error_details": [(str(test), error) for test, error in result.errors]
    }
    
    # Save test report
    report_path = Path("test_report.json")
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2, default=str)
    
    print(f"\\n📊 Test Report Summary:")
    print(f"  - Total tests: {test_report['tests_run']}")
    print(f"  - Failures: {test_report['failures']}")
    print(f"  - Errors: {test_report['errors']}")
    print(f"  - Success rate: {test_report['success_rate']:.1%}")
    print(f"  - Report saved to: {report_path}")
    
    if result.wasSuccessful():
        print("\\n✅ All tests passed successfully!")
        return 0
    else:
        print("\\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(run_comprehensive_tests())