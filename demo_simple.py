#!/usr/bin/env python3
"""Simple demo showcasing Mobile Multi-Modal LLM basic functionality.

This demonstrates the core features without requiring heavy ML dependencies.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def main():
    """Run simple demo of mobile multi-modal LLM."""
    print("🚀 Mobile Multi-Modal LLM - Simple Demo")
    print("=" * 50)
    
    try:
        # Test basic imports
        from mobile_multimodal.utils import ImageProcessor, ConfigManager
        print("✅ Core utilities imported successfully")
        
        # Test image processing utilities (without actual image dependencies)
        processor = ImageProcessor()
        print("✅ ImageProcessor instantiated")
        
        # Test config management
        config = ConfigManager()
        print("✅ ConfigManager instantiated")
        
        # Create synthetic test data
        print("\n📊 Testing with synthetic data:")
        
        # Simulate image tensor (3x224x224 RGB)
        synthetic_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"✅ Created synthetic image: {synthetic_image.shape}")
        
        # Test basic processing
        if hasattr(processor, 'normalize'):
            try:
                normalized = processor.normalize(synthetic_image)
                print(f"✅ Image normalization: {normalized.shape}")
            except Exception as e:
                print(f"⚠️  Image normalization needs implementation: {e}")
        
        # Test configuration
        config.config["model"]["quantization"] = "int2"
        config.config["model"]["batch_size"] = 1
        config.save_config()
        print("✅ Configuration saved successfully")
        
        # Reload configuration
        config.load_config()
        print(f"✅ Configuration loaded: quantization={config.config['model']['quantization']}")
        
        # Clean up
        config_path = Path(config.config_path)
        if config_path.exists():
            config_path.unlink()
        print("✅ Demo completed successfully")
        
        print("\n🎯 Next Steps:")
        print("- Install PyTorch: pip install torch")
        print("- Install transformers: pip install transformers")
        print("- Run full inference demo: python demo_inference.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit(main())