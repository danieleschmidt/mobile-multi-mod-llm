#!/usr/bin/env python3
"""Basic import and functionality test for mobile multimodal package."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that basic imports work."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import mobile_multimodal
        print(f"✅ Mobile multimodal package imported successfully")
        print(f"   Version: {mobile_multimodal.__version__}")
        print(f"   Available components: {len(mobile_multimodal.__all__)}")
    except ImportError as e:
        print(f"❌ Mobile multimodal import failed: {e}")
        return False
    
    return True


def test_package_info():
    """Test package information and metadata."""
    print("\nTesting package information...")
    
    try:
        import mobile_multimodal
        
        # Test package info
        info = mobile_multimodal.get_package_info()
        print(f"✅ Package info retrieved:")
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Features: {len(info['features'])}")
        print(f"   Supported platforms: {info['supported_platforms']}")
        
        # Test dependency check
        deps = mobile_multimodal.check_dependencies()
        available_deps = [name for name, available in deps.items() if available]
        print(f"✅ Available dependencies: {available_deps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Package info test failed: {e}")
        return False


def test_utils_functionality():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from mobile_multimodal.utils import ImageProcessor, TextTokenizer, ConfigManager
        
        # Test ImageProcessor
        processor = ImageProcessor(target_size=(224, 224))
        print(f"✅ ImageProcessor created with target size: {processor.target_size}")
        
        # Test TextTokenizer  
        tokenizer = TextTokenizer(vocab_size=1000, max_length=128)
        test_text = "This is a test sentence."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"✅ TextTokenizer: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
        
        # Test ConfigManager
        config = ConfigManager()
        input_size = config.get('model.input_size')
        print(f"✅ ConfigManager: default input size = {input_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False


def test_model_creation():
    """Test basic model creation without PyTorch."""
    print("\nTesting model creation...")
    
    try:
        from mobile_multimodal import MobileMultiModalLLM
        
        # Test creation without model file (should work without PyTorch)
        print("⚠️  PyTorch not available - testing initialization without model loading")
        
        try:
            model = MobileMultiModalLLM(model_path=None, device="cpu", safety_checks=True)
            print("❌ Model creation should have failed without PyTorch")
            return False
        except ImportError as e:
            print(f"✅ Expected ImportError caught: {e}")
            return True
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_export_utilities():
    """Test export utilities (without actual models)."""
    print("\nTesting export utilities...")
    
    try:
        from mobile_multimodal.export import optimize_for_mobile_inference
        
        # Test mobile optimization config
        android_config = optimize_for_mobile_inference("dummy_model.onnx", "android")
        ios_config = optimize_for_mobile_inference("dummy_model.onnx", "ios")
        
        print(f"✅ Android optimization config: {len(android_config)} parameters")
        print(f"✅ iOS optimization config: {len(ios_config)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Export utilities test failed: {e}")
        return False


def test_optimization_utilities():
    """Test optimization and caching utilities."""
    print("\nTesting optimization utilities...")
    
    try:
        from mobile_multimodal.optimization import AdaptiveCache, PerformanceMetrics
        
        # Test adaptive cache
        cache = AdaptiveCache(max_size=100, ttl_seconds=60)
        cache.put("test_key", "test_value")
        value = cache.get("test_key")
        stats = cache.get_stats()
        
        print(f"✅ AdaptiveCache: stored and retrieved value = '{value}'")
        print(f"   Cache stats: size={stats['size']}, hit_rate={stats['hit_rate']}")
        
        # Test performance metrics
        metrics = PerformanceMetrics(
            inference_time_ms=25.5,
            memory_usage_mb=150.0,
            cpu_usage_percent=45.0
        )
        print(f"✅ PerformanceMetrics: {metrics.inference_time_ms}ms, {metrics.memory_usage_mb}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization utilities test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running Mobile Multimodal LLM Basic Tests")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_package_info,
        test_utils_functionality,
        test_model_creation,
        test_export_utilities,
        test_optimization_utilities
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)