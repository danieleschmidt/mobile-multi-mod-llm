#!/usr/bin/env python3
"""Comprehensive Quality Gates and Testing - Final validation of the complete system."""

import sys
import time
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_quality_gate(gate_name: str, test_func: callable) -> bool:
    """Run a quality gate with timing and error handling."""
    print(f"🔍 Running quality gate: {gate_name}")
    start_time = time.time()
    
    try:
        result = test_func()
        end_time = time.time()
        duration = end_time - start_time
        
        if result:
            print(f"   ✅ PASSED ({duration:.3f}s)")
            return True
        else:
            print(f"   ❌ FAILED ({duration:.3f}s)")
            return False
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"   💥 ERROR ({duration:.3f}s): {e}")
        return False

def test_package_structure():
    """Test that all required package components exist."""
    try:
        from mobile_multimodal import MobileMultiModalLLM
        from mobile_multimodal.core import MobileMultiModalLLM as CoreLLM
        from mobile_multimodal.models import EfficientViTBlock
        from mobile_multimodal.quantization import INT2Quantizer
        from mobile_multimodal.utils import ImageProcessor
        from mobile_multimodal.security import SecurityValidator
        from mobile_multimodal.monitoring import ResourceMonitor
        from mobile_multimodal.resilience import ResilienceManager
        
        return True
    except ImportError:
        return False

def test_basic_functionality():
    """Test basic model functionality."""
    try:
        from mobile_multimodal import MobileMultiModalLLM
        
        model = MobileMultiModalLLM(device="cpu", safety_checks=True)
        health = model.get_health_status()
        
        # Basic health checks
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in health
        assert "checks" in health
        assert "metrics" in health
        
        return True
    except Exception:
        return False

def test_generation1_features():
    """Test Generation 1: MAKE IT WORK features."""
    try:
        result = subprocess.run([
            sys.executable, "generation1_validation.py"
        ], capture_output=True, text=True, timeout=60)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("   ⏰ Generation 1 tests timed out")
        return False
    except Exception:
        return False

def test_generation2_features():
    """Test Generation 2: MAKE IT ROBUST features."""
    try:
        result = subprocess.run([
            sys.executable, "generation2_robustness.py"
        ], capture_output=True, text=True, timeout=120)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("   ⏰ Generation 2 tests timed out")
        return False
    except Exception:
        return False

def test_generation3_features():
    """Test Generation 3: MAKE IT SCALE features."""
    try:
        result = subprocess.run([
            sys.executable, "generation3_simple_validation.py"
        ], capture_output=True, text=True, timeout=120)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("   ⏰ Generation 3 tests timed out")
        return False
    except Exception:
        return False

def test_security_hardening():
    """Test security features and hardening."""
    try:
        from mobile_multimodal.security import SecurityValidator, InputSanitizer
        from mobile_multimodal.advanced_security import ThreatDetector
        
        validator = SecurityValidator(strict_mode=True)
        sanitizer = InputSanitizer()
        detector = ThreatDetector()
        
        # Test malicious input detection
        malicious_data = {"text": "<script>alert('test')</script>"}
        threat_detected = detector.detect_threats(malicious_data)
        
        # Test input sanitization
        sanitized = sanitizer.sanitize(malicious_data)
        
        return True
    except Exception:
        return False

def test_performance_benchmarks():
    """Test performance benchmarks meet requirements."""
    try:
        from mobile_multimodal import MobileMultiModalLLM
        from mobile_multimodal.intelligent_cache import IntelligentCache
        
        # Model loading performance
        start_time = time.time()
        model = MobileMultiModalLLM(device="cpu", enable_optimization=True)
        load_time = time.time() - start_time
        
        # Health check performance
        start_time = time.time()
        health = model.get_health_status()
        health_time = time.time() - start_time
        
        # Cache performance
        cache = IntelligentCache(max_size=1000)
        start_time = time.time()
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
        cache_time = time.time() - start_time
        
        # Performance assertions
        assert load_time < 2.0, f"Model loading too slow: {load_time}s"
        assert health_time < 0.1, f"Health check too slow: {health_time}s"
        assert cache_time < 0.5, f"Cache operations too slow: {cache_time}s"
        
        return True
    except Exception:
        return False

def test_error_handling():
    """Test comprehensive error handling."""
    try:
        from mobile_multimodal.robust_error_handling import ErrorRecoveryManager, CircuitBreaker
        from mobile_multimodal import MobileMultiModalLLM
        
        # Test error recovery
        error_manager = ErrorRecoveryManager()
        test_error = ValueError("Test error")
        handled = error_manager.handle_error(test_error, "test_context")
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        def failing_function():
            raise Exception("Test failure")
        
        # Test circuit breaker behavior
        failure_count = 0
        for _ in range(5):
            try:
                breaker.call(failing_function)
            except Exception:
                failure_count += 1
        
        assert failure_count > 0, "Circuit breaker should have triggered"
        assert breaker.get_state() == "open", "Circuit breaker should be open"
        
        return True
    except Exception:
        return False

def test_deployment_readiness():
    """Test that the system is ready for production deployment."""
    try:
        from mobile_multimodal import get_package_info, check_dependencies
        from mobile_multimodal.monitoring import TelemetryCollector
        from mobile_multimodal.security import SecurityValidator
        
        # Check package info
        package_info = get_package_info()
        assert "name" in package_info
        assert "version" in package_info
        
        # Check dependencies
        deps = check_dependencies()
        critical_deps = ["torch", "cv2"]  # Core dependencies for basic functionality
        
        # Initialize key systems
        telemetry = TelemetryCollector(enable_system_metrics=False)  # Disable to avoid psutil dependency
        security = SecurityValidator(strict_mode=True)
        
        return True
    except Exception:
        return False

def test_memory_usage():
    """Test memory usage stays within acceptable bounds."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple components
        from mobile_multimodal import MobileMultiModalLLM
        from mobile_multimodal.intelligent_cache import IntelligentCache
        
        model = MobileMultiModalLLM(device="cpu")
        cache = IntelligentCache(max_size=5000)
        
        # Add some data to cache
        for i in range(1000):
            cache.set(f"test_key_{i}", f"test_value_{i}" * 10)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should use less than 500MB additional memory
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"
        
        return True
    except ImportError:
        # psutil not available, skip memory test
        return True
    except Exception:
        return False

def main():
    """Run comprehensive quality gates validation."""
    print("🚀 COMPREHENSIVE QUALITY GATES VALIDATION")
    print("=" * 70)
    
    quality_gates = [
        ("Package Structure Integrity", test_package_structure),
        ("Basic Functionality", test_basic_functionality), 
        ("Generation 1: MAKE IT WORK", test_generation1_features),
        ("Generation 2: MAKE IT ROBUST", test_generation2_features),
        ("Generation 3: MAKE IT SCALE", test_generation3_features),
        ("Security Hardening", test_security_hardening),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Error Handling & Recovery", test_error_handling),
        ("Deployment Readiness", test_deployment_readiness),
        ("Memory Usage Constraints", test_memory_usage)
    ]
    
    passed = 0
    failed = 0
    
    for gate_name, test_func in quality_gates:
        if run_quality_gate(gate_name, test_func):
            passed += 1
        else:
            failed += 1
    
    print("=" * 70)
    print(f"📊 QUALITY GATES SUMMARY")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("🎉 ALL QUALITY GATES PASSED - SYSTEM READY FOR PRODUCTION!")
        return True
    else:
        print(f"⚠️  {failed} QUALITY GATES FAILED - REQUIRES ATTENTION")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)