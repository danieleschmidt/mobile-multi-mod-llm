"""Mobile Multi-Modal LLM package with advanced quantization and deployment capabilities.

Tiny (<35 MB) vision-text transformer for on-device mobile AI with INT2 quantization,
multi-task capabilities (captioning, OCR, VQA, retrieval), and cross-platform deployment.
"""

from .core import MobileMultiModalLLM
from .models import (
    EfficientViTBlock, 
    EfficientSelfAttention, 
    MobileConvBlock,
    ModelProfiler,
    NeuralArchitectureSearchSpace
)
from .quantization import (
    INT2Quantizer, 
    INT2QuantizedModel, 
    HexagonOptimizer,
    QuantizationValidator
)
from .utils import (
    ImageProcessor, 
    TextTokenizer, 
    BenchmarkUtils, 
    ModelUtils,
    ConfigManager
)

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "mobile-ai@terragon.com"
__license__ = "MIT"

# Main exports
__all__ = [
    # Core functionality
    "MobileMultiModalLLM",
    
    # Model components
    "EfficientViTBlock",
    "EfficientSelfAttention", 
    "MobileConvBlock",
    "ModelProfiler",
    "NeuralArchitectureSearchSpace",
    
    # Quantization
    "INT2Quantizer",
    "INT2QuantizedModel",
    "HexagonOptimizer",
    "QuantizationValidator",
    
    # Utilities
    "ImageProcessor",
    "TextTokenizer",
    "BenchmarkUtils",
    "ModelUtils",
    "ConfigManager"
]

# Package metadata
PACKAGE_INFO = {
    "name": "mobile_multimodal",
    "version": __version__,
    "description": "Ultra-compact mobile multi-modal LLM with INT2 quantization",
    "features": [
        "Sub-35MB model size with INT2 quantization",
        "Multi-task: captioning, OCR, VQA, retrieval",
        "Hexagon NPU optimization",
        "Cross-platform: Android, iOS, Edge devices",
        "Real-time inference: 60+ FPS on modern devices",
        "Privacy-first: 100% on-device processing"
    ],
    "supported_platforms": ["Android", "iOS", "Linux", "Windows", "macOS"],
    "hardware_acceleration": ["Qualcomm Hexagon NPU", "Apple Neural Engine", "ARM Mali GPU"],
    "model_formats": ["PyTorch", "ONNX", "TensorFlow Lite", "Core ML"]
}


def get_package_info() -> dict:
    """Get comprehensive package information."""
    return PACKAGE_INFO


def check_dependencies() -> dict:
    """Check availability of optional dependencies."""
    dependencies = {
        "torch": False,
        "cv2": False,
        "onnxruntime": False,
        "tensorflow": False,
        "coremltools": False
    }
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    try:
        import cv2
        dependencies["cv2"] = True
    except ImportError:
        pass
    
    try:
        import onnxruntime
        dependencies["onnxruntime"] = True
    except ImportError:
        pass
    
    try:
        import tensorflow
        dependencies["tensorflow"] = True
    except ImportError:
        pass
    
    try:
        import coremltools
        dependencies["coremltools"] = True
    except ImportError:
        pass
    
    return dependencies


def setup_logging(level: str = "INFO"):
    """Setup logging configuration for the package."""
    import logging
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set package logger
    logger = logging.getLogger('mobile_multimodal')
    logger.info(f"Mobile Multi-Modal LLM v{__version__} initialized")
    
    # Check dependencies
    deps = check_dependencies()
    missing_deps = [name for name, available in deps.items() if not available]
    
    if missing_deps:
        logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
    else:
        logger.info("All optional dependencies available")
    
    return logger


def create_model(model_name: str = "mobile-mm-llm-base", **kwargs) -> MobileMultiModalLLM:
    """Factory function to create mobile multi-modal models."""
    return MobileMultiModalLLM.from_pretrained(model_name, **kwargs)


def load_config(config_path: str = None) -> ConfigManager:
    """Load configuration for mobile deployment."""
    return ConfigManager(config_path)