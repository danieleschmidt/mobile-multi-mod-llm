#!/usr/bin/env python3
"""Comprehensive test suite for mobile multi-modal LLM package."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import the modules
try:
    from mobile_multimodal.core import MobileMultiModalLLM
    from mobile_multimodal.security import SecurityValidator, RateLimiter, InputSanitizer
    from mobile_multimodal.monitoring import TelemetryCollector, MetricCollector
    from mobile_multimodal.data.preprocessing import ImagePreprocessor, TextPreprocessor
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class TestMobileMultiModalCore(unittest.TestCase):
    """Test core functionality of MobileMultiModalLLM."""
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.model = MobileMultiModalLLM(device="cpu", enable_telemetry=False)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertTrue(self.model._is_initialized)
        self.assertEqual(self.model.device, "cpu")
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_generate_caption(self):
        """Test caption generation."""
        caption = self.model.generate_caption(self.test_image)
        self.assertIsInstance(caption, str)
        self.assertGreater(len(caption), 0)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_extract_text(self):
        """Test OCR text extraction."""
        text_regions = self.model.extract_text(self.test_image)
        self.assertIsInstance(text_regions, list)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_answer_question(self):
        """Test VQA functionality."""
        question = "What is in this image?"
        answer = self.model.answer_question(self.test_image, question)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_get_embeddings(self):
        """Test image embedding extraction."""
        embeddings = self.model.get_image_embeddings(self.test_image)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings.shape), 2)  # Should be (batch_size, embedding_dim)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        self.assertIsInstance(info, dict)
        self.assertIn("architecture", info)
        self.assertIn("device", info)


class TestSecurity(unittest.TestCase):
    """Test security validation functionality."""
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up security test fixtures."""
        self.validator = SecurityValidator(strict_mode=True)
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_valid_request(self):
        """Test validation of legitimate request."""
        valid_request = {
            "operation": "generate_caption",
            "image": self.test_image,
            "max_length": 50
        }
        
        result = self.validator.validate_request("test_user", valid_request)
        self.assertTrue(result["valid"])
        self.assertIn("checks", result)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_malicious_request_blocked(self):
        """Test that malicious requests are blocked."""
        malicious_request = {
            "operation": "generate_caption",
            "text": "<script>alert('xss')</script>",
            "image": self.test_image
        }
        
        result = self.validator.validate_request("test_user", malicious_request)
        # Should either block or warn about the malicious content
        self.assertTrue(not result["valid"] or len(result["warnings"]) > 0)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        rate_limiter = RateLimiter(max_requests_per_minute=3)
        
        user_id = "test_user"
        allowed_count = 0
        
        # Try to make 5 requests
        for _ in range(5):
            if rate_limiter.allow_request(user_id):
                allowed_count += 1
        
        # Should only allow 3 requests
        self.assertEqual(allowed_count, 3)
        self.assertEqual(rate_limiter.get_remaining_requests(user_id), 0)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_input_sanitization(self):
        """Test input sanitization."""
        sanitizer = InputSanitizer()
        
        dirty_data = {
            "text": "<script>alert('xss')</script>Hello World",
            "number": float('inf'),
            "array": list(range(2000)),  # Too large
        }
        
        clean_data = sanitizer.sanitize_request(dirty_data)
        self.assertIsInstance(clean_data, dict)
        
        # Text should be sanitized
        if "text" in clean_data:
            self.assertNotIn("<script>", clean_data["text"])
        
        # Infinite number should be handled
        if "number" in clean_data:
            self.assertNotEqual(clean_data["number"], float('inf'))
        
        # Large array should be truncated
        if "array" in clean_data:
            self.assertLessEqual(len(clean_data["array"]), sanitizer.max_array_size)


class TestMonitoring(unittest.TestCase):
    """Test monitoring and telemetry functionality."""
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up monitoring test fixtures."""
        self.metric_collector = MetricCollector()
        self.telemetry = TelemetryCollector(enable_system_metrics=False)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_metric_collection(self):
        """Test metric collection."""
        # Record some metrics
        self.metric_collector.record_metric("test_metric", 42.0, {"type": "test"})
        self.metric_collector.record_performance("test_op", 0.1, False)
        
        # Get recent metrics
        metrics = self.metric_collector.get_recent_metrics(5)
        self.assertGreater(len(metrics), 0)
        
        # Get performance stats
        stats = self.metric_collector.get_performance_stats("test_op")
        self.assertIsInstance(stats, dict)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_operation_tracking(self):
        """Test operation tracking in telemetry."""
        # Start an operation
        self.telemetry.record_operation_start("op1", "test_operation", "test_user")
        
        # Complete it successfully
        self.telemetry.record_operation_success("op1", duration=0.1)
        
        # Check statistics
        stats = self.telemetry.get_operation_stats("test_operation")
        self.assertEqual(stats["total_operations"], 1)
        self.assertEqual(stats["success_count"], 1)
        self.assertEqual(stats["failure_count"], 0)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_operation_failure_tracking(self):
        """Test operation failure tracking."""
        # Start and fail an operation
        self.telemetry.record_operation_start("op2", "test_operation", "test_user")
        self.telemetry.record_operation_failure("op2", "Test error")
        
        # Check statistics
        stats = self.telemetry.get_operation_stats("test_operation")
        self.assertGreater(stats["failure_count"], 0)


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functionality."""
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up preprocessing test fixtures."""
        self.image_processor = ImagePreprocessor(target_size=(224, 224))
        self.text_processor = TextPreprocessor(max_length=50)
        
        # Test data
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_texts = [
            "A red car on the road",
            "Blue sky with clouds",
            "Green trees in the park"
        ]
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_image_preprocessing(self):
        """Test image preprocessing."""
        processed = self.image_processor.process(self.test_image)
        self.assertIsInstance(processed, (np.ndarray, type(None)))
        
        if processed is not None:
            # Should be resized to target size
            if len(processed.shape) == 3:
                h, w = processed.shape[:2]
                self.assertEqual((h, w), self.image_processor.target_size)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_text_preprocessing(self):
        """Test text preprocessing."""
        # Build vocabulary
        vocab = self.text_processor.build_vocabulary(self.test_texts)
        self.assertIsInstance(vocab, dict)
        self.assertGreater(len(vocab), 0)
        
        # Process text
        processed = self.text_processor.process(self.test_texts[0])
        self.assertIsInstance(processed, list)
        
        # Should be padded to max length
        self.assertEqual(len(processed), self.text_processor.max_length)
        
        # Decode back
        decoded = self.text_processor.decode_sequence(processed)
        self.assertIsInstance(decoded, str)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up integration test fixtures."""
        self.model = MobileMultiModalLLM(
            device="cpu",
            safety_checks=True,
            enable_telemetry=True,
            enable_optimization=False  # Keep simple for testing
        )
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_end_to_end_caption_generation(self):
        """Test complete caption generation pipeline."""
        # Generate caption with security and monitoring
        caption = self.model.generate_caption(self.test_image, user_id="integration_test")
        
        # Should return a valid caption
        self.assertIsInstance(caption, str)
        self.assertGreater(len(caption), 0)
        
        # Should not contain security issues
        self.assertNotIn("<script>", caption.lower())
        self.assertNotIn("javascript:", caption.lower())
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_performance_monitoring_integration(self):
        """Test that operations are properly monitored."""
        # Perform several operations
        for i in range(3):
            self.model.generate_caption(self.test_image, user_id=f"perf_test_{i}")
        
        # Check performance metrics
        metrics = self.model.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        
        if "total_operations" in metrics:
            self.assertGreaterEqual(metrics["total_operations"], 3)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_health_monitoring(self):
        """Test health monitoring functionality."""
        health = self.model.get_health_status()
        self.assertIsInstance(health, dict)
        self.assertIn("status", health)
        self.assertIn("checks", health)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up error handling test fixtures."""
        self.model = MobileMultiModalLLM(device="cpu", safety_checks=True)
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_invalid_image_input(self):
        """Test handling of invalid image inputs."""
        # Test with None
        try:
            result = self.model.generate_caption(None)
            self.assertIn("error", result.lower())
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
        
        # Test with invalid array
        try:
            invalid_image = np.array([1, 2, 3])  # 1D array instead of image
            result = self.model.generate_caption(invalid_image)
            self.assertIn("error", result.lower())
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError))
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
    def test_invalid_text_input(self):
        """Test handling of invalid text inputs."""
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test with empty question
        result = self.model.answer_question(test_image, "")
        self.assertIsInstance(result, str)
        
        # Test with None question
        try:
            result = self.model.answer_question(test_image, None)
            self.assertIsInstance(result, str)
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError))
    
    @unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available") 
    def test_resource_limits(self):
        """Test resource limit handling."""
        # Test with very large image (if not caught by preprocessing)
        try:
            large_image = np.random.randint(0, 255, (5000, 5000, 3), dtype=np.uint8)
            result = self.model.generate_caption(large_image)
            # Should either work or return error message
            self.assertIsInstance(result, str)
        except Exception as e:
            # Should be a controlled exception
            self.assertIsInstance(e, (ValueError, RuntimeError, MemoryError))


def run_basic_test():
    """Run a basic test without unittest framework."""
    print("Running basic functionality tests...")
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required modules not available - skipping tests")
        return False
    
    try:
        # Test basic model creation
        print("Testing model creation...")
        model = MobileMultiModalLLM(device="cpu", enable_telemetry=False)
        print("✓ Model created successfully")
        
        # Test basic image processing
        print("Testing image processing...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        caption = model.generate_caption(test_image)
        print(f"✓ Caption generated: {caption[:50]}...")
        
        # Test security validation
        print("Testing security...")
        validator = SecurityValidator(strict_mode=False)
        request = {
            "operation": "generate_caption",
            "image": test_image
        }
        result = validator.validate_request("test_user", request)
        print(f"✓ Security validation: {result['valid']}")
        
        # Test monitoring
        print("Testing monitoring...")
        collector = MetricCollector()
        collector.record_metric("test", 1.0, {"status": "ok"})
        metrics = collector.get_recent_metrics(5)
        print(f"✓ Monitoring: {len(metrics)} metrics recorded")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # First try basic tests
    print("="*60)
    print("MOBILE MULTI-MODAL LLM - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    basic_success = run_basic_test()
    
    if basic_success:
        print("\n" + "="*60)
        print("RUNNING DETAILED UNIT TESTS")
        print("="*60)
        
        # Run unittest suite
        unittest.main(verbosity=2, exit=False)
    else:
        print("\n❌ Basic tests failed - skipping detailed tests")
        sys.exit(1)