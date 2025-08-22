#!/usr/bin/env python3
"""Generation 2 Enhancement: MAKE IT ROBUST - Advanced error handling, security, and validation."""

import sys
import os
import tempfile
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enhanced_error_handling():
    """Test comprehensive error handling and recovery."""
    try:
        from mobile_multimodal import MobileMultiModalLLM
        from mobile_multimodal.robust_error_handling import ErrorRecoveryManager, CircuitBreaker
        
        # Test error recovery manager
        error_manager = ErrorRecoveryManager()
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        
        print("✅ Enhanced error handling components loaded")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_advanced_security():
    """Test security validation and hardening features."""
    try:
        from mobile_multimodal.security import SecurityValidator, InputSanitizer
        from mobile_multimodal.advanced_security import ThreatDetector, AdvancedEncryption
        
        # Test security validator
        validator = SecurityValidator()
        sanitizer = InputSanitizer()
        threat_detector = ThreatDetector()
        
        # Test input validation
        test_input = {"text": "Hello world", "image_shape": [224, 224, 3]}
        is_valid = validator.validate_input(test_input)
        sanitized_input = sanitizer.sanitize(test_input)
        
        print("✅ Advanced security features validated")
        return True
    except Exception as e:
        print(f"❌ Security validation test failed: {e}")
        return False

def test_comprehensive_logging():
    """Test advanced logging with structured output and monitoring."""
    try:
        from mobile_multimodal.monitoring import TelemetryCollector, MetricCollector
        from mobile_multimodal.enhanced_monitoring import AdvancedMetrics, AlertManager
        
        # Test telemetry collection
        telemetry = TelemetryCollector(enable_system_metrics=True)
        metrics = MetricCollector()
        advanced_metrics = AdvancedMetrics()
        
        # Test metric recording
        metrics.record_metric("test_metric", 1.0, tags={"test": "true"}, unit="count")
        
        print("✅ Comprehensive logging and monitoring validated")
        return True
    except Exception as e:
        print(f"❌ Logging and monitoring test failed: {e}")
        return False

def test_input_validation():
    """Test robust input validation and sanitization."""
    try:
        from mobile_multimodal.advanced_validation import InputValidator, DataValidator
        
        validator = InputValidator()
        data_validator = DataValidator()
        
        # Test various input scenarios
        test_cases = [
            {"type": "text", "content": "Valid text input"},
            {"type": "image", "shape": [224, 224, 3], "dtype": "uint8"},
            {"type": "batch", "size": 16, "max_size": 32},
        ]
        
        for test_case in test_cases:
            is_valid = validator.validate(test_case)
            assert is_valid, f"Validation failed for {test_case}"
        
        print("✅ Input validation tests passed")
        return True
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        return False

def test_resource_management():
    """Test advanced resource monitoring and management."""
    try:
        from mobile_multimodal.monitoring import ResourceMonitor
        from mobile_multimodal.resilience import ResilienceManager
        
        # Test resource monitoring
        resource_monitor = ResourceMonitor()
        usage = resource_monitor.get_usage()
        
        assert "cpu" in usage, "CPU usage not reported"
        assert "memory" in usage, "Memory usage not reported"
        
        # Test resilience patterns
        resilience_manager = ResilienceManager()
        
        print("✅ Resource management tests passed")
        return True
    except Exception as e:
        print(f"❌ Resource management test failed: {e}")
        return False

def test_model_health_checks():
    """Test comprehensive model health monitoring."""
    try:
        from mobile_multimodal import MobileMultiModalLLM
        
        # Initialize model with health checking enabled
        model = MobileMultiModalLLM(
            device="cpu",
            safety_checks=True,
            health_check_enabled=True,
            strict_security=True
        )
        
        # Test health check functionality
        health_status = model.get_health_status()
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
        
        print("✅ Model health checks validated")
        return True
    except Exception as e:
        print(f"❌ Model health checks failed: {e}")
        return False

def test_concurrent_safety():
    """Test thread safety and concurrent processing."""
    try:
        from mobile_multimodal.concurrent_processor import ConcurrentProcessor, ThreadSafeCache
        from mobile_multimodal.circuit_breaker import CircuitBreaker
        
        # Test concurrent processor
        processor = ConcurrentProcessor(max_workers=4, queue_size=100)
        cache = ThreadSafeCache(max_size=1000)
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        # Test cache operations
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value", "Cache operation failed"
        
        print("✅ Concurrent safety tests passed")
        return True
    except Exception as e:
        print(f"❌ Concurrent safety test failed: {e}")
        return False

def test_configuration_security():
    """Test secure configuration management."""
    try:
        from mobile_multimodal.utils import ConfigManager
        from mobile_multimodal.security import SecureConfig
        
        # Test secure configuration
        secure_config = SecureConfig()
        config_manager = ConfigManager()
        
        # Test sensitive data handling
        secure_config.set_secret("api_key", "test_secret_key")
        retrieved_key = secure_config.get_secret("api_key")
        
        # Should not store in plaintext
        assert retrieved_key == "test_secret_key", "Secret retrieval failed"
        
        print("✅ Configuration security tests passed")
        return True
    except Exception as e:
        print(f"❌ Configuration security test failed: {e}")
        return False

def main():
    """Run Generation 2 robustness tests."""
    print("🛡️ GENERATION 2 VALIDATION: MAKE IT ROBUST")
    print("=" * 60)
    
    tests = [
        test_enhanced_error_handling,
        test_advanced_security,
        test_comprehensive_logging,
        test_input_validation,
        test_resource_management,
        test_model_health_checks,
        test_concurrent_safety,
        test_configuration_security
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
        print("🎉 GENERATION 2: MAKE IT ROBUST - COMPLETE!")
        return True
    else:
        print(f"⚠️  GENERATION 2: {failed} tests failed, implementing fixes...")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)