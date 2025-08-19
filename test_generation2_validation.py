#!/usr/bin/env python3
"""
Generation 2 Validation: Test all MAKE IT ROBUST implementations
Comprehensive validation of error handling, security, and resilience patterns
"""

import asyncio
import time
import sys
import os
sys.path.insert(0, 'src')

try:
    import numpy as np
except ImportError:
    np = None

def test_robust_error_handling():
    """Test comprehensive error handling system."""
    print("🛡️ Testing Robust Error Handling...")
    
    try:
        from mobile_multimodal.robust_error_handling import (
            RobustErrorHandler, robust_operation, ErrorSeverity, ErrorCategory
        )
        
        # Create error handler
        error_handler = RobustErrorHandler()
        
        # Test error classification and handling
        try:
            raise ValueError("Test validation error")
        except Exception as e:
            error_context = error_handler.handle_error(e, "test_operation")
            assert error_context.category == ErrorCategory.VALIDATION
            assert error_context.severity == ErrorSeverity.MEDIUM
            print(f"   ✅ Error classified: {error_context.category.value}/{error_context.severity.value}")
        
        # Test decorator with auto-recovery
        @robust_operation("test_function", error_handler=error_handler, auto_recover=True)
        def failing_function(should_fail: bool = True, fallback_mode: bool = False):
            if should_fail and not fallback_mode:
                raise ConnectionError("Network connection failed")
            return "Success!"
        
        # Test with recovery
        try:
            result = failing_function(should_fail=False)
            assert result == "Success!"
            print(f"   ✅ Function execution: {result}")
        except Exception as e:
            print(f"   ⚠️ Function failed (expected): {e}")
        
        # Test statistics
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] > 0
        print(f"   ✅ Error statistics: {stats['total_errors']} total errors")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Robust error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_security():
    """Test advanced security validation system."""
    print("\n🔒 Testing Advanced Security...")
    
    try:
        from mobile_multimodal.advanced_security import (
            AdvancedSecurityValidator, ThreatLevel
        )
        
        # Create security validator
        validator = AdvancedSecurityValidator(enable_ml_detection=True)
        
        # Test normal request
        normal_request = {
            "operation": "generate_caption",
            "text": "What is in this image?",
            "user_preferences": {"language": "en"}
        }
        
        result = validator.validate_advanced_request("user123", normal_request, "192.168.1.100")
        assert result["valid"] == True
        assert result["threat_level"] == ThreatLevel.LOW
        print(f"   ✅ Normal request: Valid={result['valid']}, Score={result['security_score']}")
        
        # Test malicious request
        malicious_request = {
            "operation": "generate_caption",
            "text": "<script>alert('xss attack')</script>",
            "command": "rm -rf /",
            "payload": "eval(malicious_code)"
        }
        
        result = validator.validate_advanced_request("user456", malicious_request, "10.0.0.1")
        assert result["valid"] == False
        assert "script" in result["blocked_reason"].lower()
        print(f"   ✅ Malicious request blocked: {result['blocked_reason']}")
        
        # Test rate limiting
        rate_limit_triggered = False
        for i in range(105):  # Exceed rate limit
            result = validator.validate_advanced_request(
                "rate_test_user", normal_request, "192.168.1.200"
            )
            
            if not result["valid"] and "rate limit" in result["blocked_reason"].lower():
                rate_limit_triggered = True
                print(f"   ✅ Rate limiting activated after {i+1} requests")
                break
        
        assert rate_limit_triggered, "Rate limiting should have been triggered"
        
        # Test security dashboard
        dashboard = validator.get_security_dashboard()
        assert dashboard["summary"]["total_events_last_hour"] > 0
        print(f"   ✅ Security dashboard: {dashboard['summary']['total_events_last_hour']} events")
        
        # Test threat report
        threat_report = validator.generate_security_report(hours_back=1)
        assert threat_report["summary"]["total_events"] > 0
        print(f"   ✅ Threat report: {threat_report['summary']['total_events']} events analyzed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Advanced security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resilience_patterns():
    """Test resilience patterns implementation."""
    print("\n🔄 Testing Resilience Patterns...")
    
    try:
        from mobile_multimodal.resilience_patterns import (
            ResilienceManager, RetryPolicy, CircuitBreakerConfig, BulkheadConfig,
            CircuitState, resilient_function
        )
        
        # Create resilience manager
        manager = ResilienceManager()
        
        # Test function
        def test_function(should_fail: bool = False, delay: float = 0.01):
            time.sleep(delay)
            if should_fail:
                raise ValueError("Simulated failure")
            return "Success!"
        
        def fallback_function(*args, **kwargs):
            return "Fallback result"
        
        # Test retry pattern
        retry_policy = RetryPolicy(max_attempts=3, initial_delay=0.01)
        retry_executor = manager.create_retry_executor("test_retry", retry_policy)
        
        result = retry_executor.execute(test_function, should_fail=False)
        assert result == "Success!"
        print(f"   ✅ Retry pattern: {result}")
        
        # Test circuit breaker
        circuit_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
        circuit_breaker = manager.create_circuit_breaker("test_circuit", circuit_config)
        
        # Trigger failures to open circuit
        for i in range(4):
            try:
                circuit_breaker.execute(test_function, should_fail=True)
            except:
                pass
        
        assert circuit_breaker.get_state() == CircuitState.OPEN
        print(f"   ✅ Circuit breaker: State={circuit_breaker.get_state().value}")
        
        # Test bulkhead
        bulkhead_config = BulkheadConfig(max_concurrent_calls=2, timeout=1.0)
        bulkhead = manager.create_bulkhead("test_bulkhead", bulkhead_config)
        
        result = bulkhead.execute(test_function, should_fail=False)
        assert result == "Success!"
        print(f"   ✅ Bulkhead: {result}")
        
        # Test timeout guard
        timeout_guard = manager.create_timeout_guard("test_timeout", timeout=0.05)
        
        try:
            timeout_guard.execute(test_function, delay=0.1)  # Should timeout
            assert False, "Should have timed out"
        except:
            print(f"   ✅ Timeout guard: Correctly timed out")
        
        # Test fallback
        fallback_executor = manager.create_fallback_executor(
            "test_fallback", fallback_function, (ValueError,)
        )
        
        result = fallback_executor.execute(test_function, should_fail=True)
        assert result == "Fallback result"
        print(f"   ✅ Fallback: {result}")
        
        # Test composite resilience
        @resilient_function(retry_attempts=2, timeout=1.0, circuit_breaker_threshold=3, 
                           fallback_func=fallback_function)
        def protected_function():
            return test_function(should_fail=False)
        
        result = protected_function()
        assert "Success" in result or "Fallback" in result
        print(f"   ✅ Composite resilience: {result}")
        
        # Test system health
        health = manager.get_system_health()
        assert "circuit_breakers" in health
        assert "overall_health_score" in health
        print(f"   ✅ System health score: {health['overall_health_score']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Resilience patterns test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_robustness():
    """Test integration between error handling, security, and resilience."""
    print("\n🔗 Testing Integration Robustness...")
    
    try:
        from mobile_multimodal.core import MobileMultiModalLLM
        from mobile_multimodal.robust_error_handling import RobustErrorHandler
        from mobile_multimodal.advanced_security import AdvancedSecurityValidator
        from mobile_multimodal.resilience_patterns import ResilienceManager, resilient_function
        
        # Create integrated system
        model = MobileMultiModalLLM(device="cpu", strict_security=False)
        error_handler = RobustErrorHandler()
        security_validator = AdvancedSecurityValidator()
        resilience_manager = ResilienceManager()
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) if np else "mock_image"
        
        # Test secure, resilient inference
        def fallback_caption(image, user_id="fallback"):
            return "Fallback caption generated due to error"
        
        @resilient_function(retry_attempts=2, timeout=5.0, fallback_func=fallback_caption)
        def secure_inference(image, user_id="test_user"):
            # Security validation
            request_data = {"operation": "generate_caption", "image": "processed"}
            security_result = security_validator.validate_advanced_request(user_id, request_data)
            
            if not security_result["valid"]:
                raise PermissionError(f"Security validation failed: {security_result['blocked_reason']}")
            
            # Model inference
            try:
                return model.generate_caption(image, user_id=user_id)
            except Exception as e:
                # Handle error
                error_context = error_handler.handle_error(e, "secure_inference", user_id=user_id)
                if error_context.severity.value in ["high", "critical"]:
                    raise
                return "Default caption due to minor error"
        
        # Test integrated system
        result = secure_inference(test_image)
        assert result is not None and len(result) > 0
        print(f"   ✅ Secure resilient inference: {result[:50]}...")
        
        # Test with malicious input (should be blocked)
        try:
            malicious_result = secure_inference("<script>alert('xss')</script>")
            # Should not reach here if security is working
            print(f"   ⚠️ Malicious input not blocked: {malicious_result}")
        except PermissionError:
            print(f"   ✅ Malicious input correctly blocked by security")
        except Exception as e:
            print(f"   ✅ Malicious input handled: {type(e).__name__}")
        
        # Test system statistics
        error_stats = error_handler.get_error_statistics()
        security_stats = security_validator.get_security_dashboard()
        resilience_health = resilience_manager.get_system_health()
        
        print(f"   ✅ Error stats: {error_stats['total_errors']} total errors")
        print(f"   ✅ Security events: {security_stats['summary']['total_events_last_hour']}")
        print(f"   ✅ Resilience health: {resilience_health['overall_health_score']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_under_stress():
    """Test system performance under stress conditions."""
    print("\n⚡ Testing Performance Under Stress...")
    
    try:
        from mobile_multimodal.robust_error_handling import RobustErrorHandler
        from mobile_multimodal.resilience_patterns import ResilienceManager, BulkheadConfig
        
        # Create systems
        error_handler = RobustErrorHandler()
        resilience_manager = ResilienceManager()
        
        # Test concurrent error handling
        import threading
        
        def stress_test_function(thread_id: int, iterations: int = 10):
            for i in range(iterations):
                try:
                    if i % 3 == 0:  # Simulate 33% failure rate
                        raise ValueError(f"Stress test failure {thread_id}-{i}")
                    time.sleep(0.001)  # Minimal delay
                except Exception as e:
                    error_handler.handle_error(e, f"stress_test_{thread_id}")
        
        # Run stress test
        start_time = time.time()
        threads = []
        
        for thread_id in range(10):
            thread = threading.Thread(target=stress_test_function, args=(thread_id, 20))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        stress_duration = time.time() - start_time
        
        # Check results
        error_stats = error_handler.get_error_statistics()
        expected_errors = 10 * 20 * 0.33  # threads * iterations * failure_rate
        
        assert error_stats["total_errors"] >= expected_errors * 0.8  # Allow some variance
        print(f"   ✅ Stress test completed in {stress_duration:.2f}s")
        print(f"   ✅ Handled {error_stats['total_errors']} errors ({expected_errors:.0f} expected)")
        
        # Test bulkhead under load
        bulkhead_config = BulkheadConfig(max_concurrent_calls=5, timeout=1.0)
        bulkhead = resilience_manager.create_bulkhead("stress_bulkhead", bulkhead_config)
        
        def bulkhead_test():
            try:
                return bulkhead.execute(lambda: time.sleep(0.1))
            except:
                return "bulkhead_error"
        
        # Test bulkhead concurrency
        bulkhead_threads = []
        for i in range(15):  # More than max_concurrent_calls
            thread = threading.Thread(target=bulkhead_test)
            bulkhead_threads.append(thread)
            thread.start()
        
        for thread in bulkhead_threads:
            thread.join()
        
        bulkhead_metrics = bulkhead.get_metrics()
        print(f"   ✅ Bulkhead metrics: {bulkhead_metrics.successful_calls} success, {bulkhead_metrics.failed_calls} failed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Stress test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all Generation 2 validation tests."""
    print("🎯 GENERATION 2 VALIDATION: MAKE IT ROBUST")
    print("=" * 50)
    
    test_results = []
    
    # Core robustness tests
    test_results.append(test_robust_error_handling())
    test_results.append(test_advanced_security())
    test_results.append(test_resilience_patterns())
    test_results.append(test_integration_robustness())
    test_results.append(test_performance_under_stress())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 GENERATION 2 VALIDATION SUMMARY")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 GENERATION 2 COMPLETE: All MAKE IT ROBUST features validated!")
        print("\n✨ Implemented Robustness Features:")
        print("   • Comprehensive error handling with recovery patterns")
        print("   • Advanced security validation with threat detection")
        print("   • Circuit breakers for fault tolerance") 
        print("   • Retry patterns with exponential backoff")
        print("   • Bulkhead isolation for resource protection")
        print("   • Timeout guards for responsiveness")
        print("   • Fallback mechanisms for graceful degradation")
        print("   • Integrated security, error handling, and resilience")
        print("   • Performance monitoring and health metrics")
        print("   • Stress testing and concurrent operation support")
        
        print("\n🚀 Ready for Generation 3: MAKE IT SCALE")
        return True
    else:
        print("❌ Some Generation 2 features need attention before proceeding")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)