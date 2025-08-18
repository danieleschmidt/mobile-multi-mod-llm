#!/usr/bin/env python3
"""Quick quality check for the mobile multimodal system."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def quick_quality_check():
    """Run essential quality checks."""
    print("🚀 Quick Quality Check - Mobile Multi-Modal LLM")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Basic imports
    total_checks += 1
    try:
        from mobile_multimodal.core import MobileMultiModalLLM
        from mobile_multimodal.optimization import PerformanceProfile, AutoScaler
        from mobile_multimodal.enhanced_monitoring import TelemetryCollector
        print("✅ Basic imports: PASSED")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Basic imports: FAILED ({e})")
    
    # Check 2: Core model creation
    total_checks += 1
    try:
        model = MobileMultiModalLLM(device="cpu", strict_security=False)
        info = model.get_model_info()
        assert isinstance(info, dict)
        print("✅ Core model creation: PASSED")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Core model creation: FAILED ({e})")
    
    # Check 3: Performance optimization
    total_checks += 1
    try:
        profile = PerformanceProfile(batch_size=4, num_workers=2)
        scaler = AutoScaler()
        metrics = {"avg_cpu_percent": 60.0, "memory_percent": 50.0}
        recommendations = scaler.get_scaling_recommendations(metrics)
        assert isinstance(recommendations, dict)
        print("✅ Performance optimization: PASSED")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Performance optimization: FAILED ({e})")
    
    # Check 4: Model architecture
    total_checks += 1
    try:
        from mobile_multimodal.models import NeuralArchitectureSearchSpace
        search_space = NeuralArchitectureSearchSpace.get_mobile_search_space()
        sample = NeuralArchitectureSearchSpace.sample_architecture(search_space)
        assert isinstance(sample, dict)
        print("✅ Model architecture: PASSED")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Model architecture: FAILED ({e})")
    
    # Check 5: Telemetry system
    total_checks += 1
    try:
        collector = TelemetryCollector(collection_interval=1.0, enable_file_export=False)
        metric = collector.record_operation_start("test", "test_op", "user")
        assert metric.operation_id == "test"
        print("✅ Telemetry system: PASSED")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Telemetry system: FAILED ({e})")
    
    # Summary
    success_rate = checks_passed / total_checks
    print(f"\n📊 QUALITY CHECK SUMMARY")
    print(f"Passed: {checks_passed}/{total_checks} ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("✅ QUALITY CHECK: PASSED")
        return True
    else:
        print("❌ QUALITY CHECK: FAILED")
        return False

if __name__ == "__main__":
    success = quick_quality_check()
    sys.exit(0 if success else 1)