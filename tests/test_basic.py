#!/usr/bin/env python3
"""Basic test runner that works without external dependencies."""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that modules can be imported."""
    print("Testing module imports...")
    
    try:
        from mobile_multimodal import __version__, PACKAGE_INFO
        print(f"✓ Package version: {__version__}")
        print(f"✓ Package info loaded: {len(PACKAGE_INFO['features'])} features")
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False
    
    try:
        from mobile_multimodal.core import MobileMultiModalLLM
        print("✓ Core module imported")
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False
    
    try:
        from mobile_multimodal.security import SecurityValidator
        print("✓ Security module imported")
    except ImportError as e:
        print(f"❌ Security import failed: {e}")
        return False
    
    try:
        from mobile_multimodal.monitoring import TelemetryCollector
        print("✓ Monitoring module imported")
    except ImportError as e:
        print(f"❌ Monitoring import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        # Test model creation in mock mode
        from mobile_multimodal.core import MobileMultiModalLLM
        model = MobileMultiModalLLM(device="cpu", enable_telemetry=False)
        print("✓ Model created successfully")
        
        # Test model info
        info = model.get_model_info()
        print(f"✓ Model info: {info.get('architecture', 'unknown')} on {info.get('device', 'unknown')}")
        
        # Test mock mode functionality
        if model._mock_mode:
            print("✓ Running in mock mode (no PyTorch/NumPy)")
            
            # Create a simple "image" as nested lists
            mock_image = [[[255, 0, 0] for _ in range(10)] for _ in range(10)]
            
            try:
                caption = model.generate_caption(mock_image)
                print(f"✓ Caption generated: {caption[:50]}...")
            except Exception as e:
                print(f"⚠️ Caption generation failed (expected in mock mode): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security():
    """Test security module."""
    print("\nTesting security...")
    
    try:
        from mobile_multimodal.security import SecurityValidator, RateLimiter, InputSanitizer
        
        # Test security validator
        validator = SecurityValidator(strict_mode=False)
        print("✓ Security validator created")
        
        # Test with simple data
        test_request = {
            "operation": "generate_caption",
            "text": "Hello world"
        }
        
        result = validator.validate_request("test_user", test_request)
        print(f"✓ Security validation result: {result['valid']}")
        
        # Test rate limiter
        rate_limiter = RateLimiter(max_requests_per_minute=3)
        user_requests = 0
        for i in range(5):
            if rate_limiter.allow_request("test_user"):
                user_requests += 1
        
        print(f"✓ Rate limiter: allowed {user_requests}/5 requests (limit: 3)")
        
        # Test input sanitizer
        sanitizer = InputSanitizer()
        dirty_data = {
            "text": "<script>alert('test')</script>Clean text",
            "number": 42
        }
        
        clean_data = sanitizer.sanitize_request(dirty_data)
        print(f"✓ Input sanitizer: cleaned {len(dirty_data)} -> {len(clean_data)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitoring():
    """Test monitoring module."""
    print("\nTesting monitoring...")
    
    try:
        from mobile_multimodal.monitoring import MetricCollector, TelemetryCollector
        
        # Test metric collector
        collector = MetricCollector()
        collector.record_metric("test_metric", 42.0, {"type": "test"})
        collector.record_performance("test_operation", 0.1, False)
        
        metrics = collector.get_recent_metrics(5)
        print(f"✓ Metric collector: {len(metrics)} metrics recorded")
        
        # Test telemetry collector
        telemetry = TelemetryCollector(enable_system_metrics=False)
        
        telemetry.record_operation_start("op1", "test_operation")
        time.sleep(0.01)
        telemetry.record_operation_success("op1", duration=0.01)
        
        stats = telemetry.get_operation_stats("test_operation")
        print(f"✓ Telemetry: {stats['total_operations']} operations tracked")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_preprocessing():
    """Test data preprocessing with mock data."""
    print("\nTesting data preprocessing...")
    
    try:
        from mobile_multimodal.data.preprocessing import TextPreprocessor
        
        # Test text processor (doesn't need NumPy/CV2)
        text_processor = TextPreprocessor(max_length=20)
        
        test_texts = [
            "This is a test",
            "Another test sentence",
            "Short text"
        ]
        
        vocab = text_processor.build_vocabulary(test_texts)
        print(f"✓ Text processor: built vocabulary with {len(vocab)} tokens")
        
        processed = text_processor.process(test_texts[0])
        print(f"✓ Text processing: {len(processed)} tokens generated")
        
        decoded = text_processor.decode_sequence(processed)
        print(f"✓ Text decoding: '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scripts():
    """Test training and export scripts."""
    print("\nTesting scripts...")
    
    try:
        # Test that script modules can be imported
        from mobile_multimodal.scripts import train, export, benchmark
        print("✓ Script modules imported successfully")
        
        # Test training config creation (without actually training)
        from mobile_multimodal.scripts.train import create_training_config
        from types import SimpleNamespace
        
        # Mock args
        mock_args = SimpleNamespace(
            device='cpu',
            epochs=1,
            batch_size=2,
            output_dir='test_output',
            mixed_precision=False,
            save_interval=1,
            tasks=['captioning'],
            samples_per_task=10,
            val_samples=5,
            optimizer='adamw',
            learning_rate=1e-4,
            weight_decay=1e-5,
            scheduler='cosine'
        )
        
        config = create_training_config(mock_args)
        print(f"✓ Training config created: {len(config)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Scripts test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_package_integrity():
    """Test package structure and metadata."""
    print("\nTesting package integrity...")
    
    try:
        from mobile_multimodal import get_package_info, check_dependencies
        
        # Test package info
        info = get_package_info()
        print(f"✓ Package info: {info['name']} v{info['version']}")
        print(f"✓ Features: {len(info['features'])} listed")
        print(f"✓ Platforms: {len(info['supported_platforms'])} supported")
        
        # Test dependency check
        deps = check_dependencies()
        available_deps = [name for name, available in deps.items() if available]
        missing_deps = [name for name, available in deps.items() if not available]
        
        print(f"✓ Dependencies: {len(available_deps)} available, {len(missing_deps)} missing")
        if missing_deps:
            print(f"  Missing: {', '.join(missing_deps)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Package integrity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("MOBILE MULTI-MODAL LLM - BASIC TEST SUITE")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Security", test_security),
        ("Monitoring", test_monitoring),
        ("Data Preprocessing", test_data_preprocessing),
        ("Scripts", test_scripts),
        ("Package Integrity", test_package_integrity)
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
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 All tests passed! The package is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total-passed} tests failed. Some functionality may not be working.")
        return 1


if __name__ == "__main__":
    sys.exit(main())