#!/usr/bin/env python3
"""Basic Quality Gates - Import and structural validation without external dependencies.

This script validates that all new modules can be imported and have proper structure
without requiring external dependencies like numpy or torch.
"""

import importlib
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_module_import(module_name: str, classes_to_check: list = None) -> tuple:
    """Test module import and basic structure."""
    try:
        module = importlib.import_module(module_name)
        
        if classes_to_check:
            for class_name in classes_to_check:
                if not hasattr(module, class_name):
                    return False, f"Missing class: {class_name}"
        
        return True, f"Successfully imported {module_name}"
        
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def run_basic_quality_gates():
    """Run basic quality gates without external dependencies."""
    
    print("🚀 Mobile Multi-Modal LLM - Basic Quality Gates")
    print("=" * 60)
    
    tests = [
        {
            "name": "Adaptive Quantization",
            "module": "mobile_multimodal.adaptive_quantization",
            "classes": ["AdaptiveQuantizationEngine", "EntropyBasedStrategy", "PrecisionLevel"]
        },
        {
            "name": "Hybrid Attention",
            "module": "mobile_multimodal.hybrid_attention", 
            "classes": ["AttentionConfig", "HybridAttentionMechanism"]
        },
        {
            "name": "Edge Federated Learning",
            "module": "mobile_multimodal.edge_federated_learning",
            "classes": ["EdgeFederatedLearningCoordinator", "DeviceProfile", "DeviceClass"]
        },
        {
            "name": "Intelligent Cache",
            "module": "mobile_multimodal.intelligent_cache",
            "classes": ["IntelligentCacheManager", "CacheLevel", "EvictionPolicy"]
        },
        {
            "name": "Concurrent Processor", 
            "module": "mobile_multimodal.concurrent_processor",
            "classes": ["ConcurrentProcessingEngine", "ProcessingTask", "TaskPriority"]
        },
        {
            "name": "Advanced Validation",
            "module": "mobile_multimodal.advanced_validation",
            "classes": ["CompositeValidator", "ValidationLevel", "ThreatType"]
        },
        {
            "name": "Circuit Breaker",
            "module": "mobile_multimodal.circuit_breaker", 
            "classes": ["AdaptiveCircuitBreaker", "CircuitConfig", "CircuitState"]
        }
    ]
    
    passed = 0
    failed = 0
    total = len(tests)
    
    print(f"\n📋 Running {total} import validation tests...")
    
    for test in tests:
        print(f"\n🔍 Testing {test['name']}...")
        
        start_time = time.time()
        success, message = test_module_import(test["module"], test["classes"])
        duration = time.time() - start_time
        
        if success:
            passed += 1
            print(f"   ✅ PASS - {message} ({duration:.3f}s)")
        else:
            failed += 1
            print(f"   ❌ FAIL - {message} ({duration:.3f}s)")
    
    # Test basic functionality without numpy
    print(f"\n🔧 Testing basic functionality...")
    
    try:
        # Test enum imports
        from mobile_multimodal.adaptive_quantization import PrecisionLevel, HardwareTarget
        from mobile_multimodal.hybrid_attention import AttentionType, SparsePattern
        from mobile_multimodal.edge_federated_learning import DeviceClass, FederationStrategy
        from mobile_multimodal.intelligent_cache import CacheLevel, EvictionPolicy
        from mobile_multimodal.concurrent_processor import ProcessingUnit, TaskPriority
        from mobile_multimodal.advanced_validation import ValidationLevel, ThreatType
        from mobile_multimodal.circuit_breaker import CircuitState, FailureType
        
        print("   ✅ PASS - All enums imported successfully")
        passed += 1
        
    except Exception as e:
        print(f"   ❌ FAIL - Enum import error: {str(e)}")
        failed += 1
    
    try:
        # Test factory function imports
        from mobile_multimodal.adaptive_quantization import EntropyBasedStrategy
        from mobile_multimodal.edge_federated_learning import create_federated_coordinator
        from mobile_multimodal.intelligent_cache import create_mobile_cache_manager
        from mobile_multimodal.concurrent_processor import create_mobile_processing_engine
        from mobile_multimodal.advanced_validation import create_validator
        from mobile_multimodal.circuit_breaker import create_mobile_circuit_config
        
        print("   ✅ PASS - All factory functions imported successfully")
        passed += 1
        
    except Exception as e:
        print(f"   ❌ FAIL - Factory function import error: {str(e)}")
        failed += 1
    
    # Test basic class instantiation (without numpy)
    try:
        from mobile_multimodal.advanced_validation import ValidationLevel
        from mobile_multimodal.circuit_breaker import CircuitConfig
        from mobile_multimodal.edge_federated_learning import DeviceClass, DeviceProfile
        
        # Test basic class creation
        validation_level = ValidationLevel.STANDARD
        circuit_config = CircuitConfig()
        device_profile = DeviceProfile(
            device_id="test",
            device_class=DeviceClass.MID_RANGE,
            memory_mb=4096,
            compute_score=0.5,
            network_quality=0.8,
            battery_level=0.9,
            privacy_budget=1.0
        )
        
        print("   ✅ PASS - Basic class instantiation successful")
        passed += 1
        
    except Exception as e:
        print(f"   ❌ FAIL - Class instantiation error: {str(e)}")
        failed += 1
    
    total += 3  # Added 3 functionality tests
    
    # Summary
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   Total Tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL BASIC QUALITY GATES PASSED!")
        print(f"   All modules can be imported and have correct structure.")
        print(f"   Ready for deployment testing with full dependencies.")
        return True
    else:
        print(f"\n❌ BASIC QUALITY GATES FAILED!")
        print(f"   {failed} test(s) failed. Fix issues before proceeding.")
        return False


def main():
    """Main entry point."""
    success = run_basic_quality_gates()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())