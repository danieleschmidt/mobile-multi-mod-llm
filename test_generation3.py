#!/usr/bin/env python3
"""Test Generation 3 optimization capabilities."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mobile_multimodal.optimization import PerformanceProfile, AutoScaler

def test_generation3():
    print("🚀 Testing Generation 3: MAKE IT SCALE")
    
    try:
        # Test performance profile
        print("\n1. Testing Performance Profile...")
        profile = PerformanceProfile(
            batch_size=8,
            num_workers=4,
            cache_size_mb=512,
            enable_dynamic_batching=True
        )
        print(f"✅ Profile created: batch_size={profile.batch_size}, workers={profile.num_workers}")
        
        # Test auto-scaler
        print("\n2. Testing Auto-Scaler...")
        scaler = AutoScaler()
        
        # Test scaling recommendations
        test_metrics = {
            "avg_cpu_percent": 85.0,
            "memory_percent": 75.0,
            "avg_latency_ms": 2500,
            "error_rate": 0.03
        }
        
        recommendations = scaler.get_scaling_recommendations(test_metrics)
        print(f"✅ Scaling recommendations:")
        print(f"   Should scale: {recommendations['should_scale']}")
        print(f"   Current capacity: {recommendations['current_capacity']}")
        print(f"   Recommended capacity: {recommendations['recommended_capacity']}")
        
        if recommendations['should_scale']:
            scaling_event = scaler.apply_scaling(recommendations['recommended_capacity'])
            print(f"✅ Scaling applied: {scaling_event['scaling_ratio']:.2f}x")
        
        print("\n✅ Generation 3 optimization system working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Generation 3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generation3()
    sys.exit(0 if success else 1)