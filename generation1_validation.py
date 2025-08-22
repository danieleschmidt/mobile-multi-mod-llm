#!/usr/bin/env python3
"""Generation 1 Validation: MAKE IT WORK - Basic functionality test."""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_core_imports():
    """Test that core components can be imported."""
    try:
        import mobile_multimodal
        from mobile_multimodal import MobileMultiModalLLM
        from mobile_multimodal.core import MobileMultiModalLLM as CoreModel
        
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

def test_model_initialization():
    """Test basic model initialization."""
    try:
        from mobile_multimodal import MobileMultiModalLLM
        
        # Test basic initialization
        model = MobileMultiModalLLM(device="cpu", safety_checks=True)
        
        print("✅ Model initialization successful")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def test_utility_components():
    """Test utility component imports."""
    try:
        from mobile_multimodal.utils import ImageProcessor, ModelUtils, ConfigManager
        from mobile_multimodal.quantization import INT2Quantizer, HexagonOptimizer
        from mobile_multimodal.monitoring import ResourceMonitor
        
        print("✅ Utility components imported successfully")
        return True
    except Exception as e:
        print(f"❌ Utility component import failed: {e}")
        return False

def test_basic_configuration():
    """Test basic configuration management."""
    try:
        from mobile_multimodal.utils import ConfigManager
        
        config = ConfigManager()
        # Test basic config operations
        config.set("test_key", "test_value")
        value = config.get("test_key")
        
        assert value == "test_value", "Configuration get/set failed"
        
        print("✅ Basic configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_package_metadata():
    """Test package metadata and info."""
    try:
        import mobile_multimodal
        
        # Test package info
        info = mobile_multimodal.get_package_info()
        assert "name" in info, "Package info missing name"
        assert "version" in info, "Package info missing version"
        
        # Test dependency checking
        deps = mobile_multimodal.check_dependencies()
        assert isinstance(deps, dict), "Dependencies check failed"
        
        print("✅ Package metadata test passed")
        return True
    except Exception as e:
        print(f"❌ Package metadata test failed: {e}")
        return False

def test_logging_setup():
    """Test logging configuration."""
    try:
        import mobile_multimodal
        
        logger = mobile_multimodal.setup_logging("INFO")
        assert logger is not None, "Logger setup failed"
        
        print("✅ Logging setup test passed")
        return True
    except Exception as e:
        print(f"❌ Logging setup test failed: {e}")
        return False

def main():
    """Run Generation 1 validation tests."""
    print("🚀 GENERATION 1 VALIDATION: MAKE IT WORK")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_model_initialization,
        test_utility_components,
        test_basic_configuration,
        test_package_metadata,
        test_logging_setup
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
    
    print("=" * 50)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 GENERATION 1: MAKE IT WORK - COMPLETE!")
        return True
    else:
        print("⚠️  GENERATION 1: Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)