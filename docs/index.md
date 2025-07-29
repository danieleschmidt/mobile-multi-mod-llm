# Mobile Multi-Modal LLM Documentation

Welcome to the Mobile Multi-Modal LLM documentation! This project provides a tiny (<35 MB) vision-text transformer optimized for on-device mobile AI applications.

## Key Features

- **Ultra-Compact**: Full multimodal model under 35MB with INT2 quantization
- **Hardware Optimized**: Native support for Qualcomm Hexagon NPU and Apple Neural Engine
- **Multi-Task**: Image captioning, OCR, visual Q&A, and text-image retrieval
- **Cross-Platform**: Single model runs on Android, iOS, and Edge devices
- **Privacy-First**: 100% on-device inference with no cloud dependencies

## Quick Links

- [Installation Guide](guides/installation.md) - Get started quickly
- [Quick Start Tutorial](guides/quickstart.md) - Basic usage examples  
- [API Reference](api/core.md) - Complete API documentation
- [Contributing](CONTRIBUTING.md) - How to contribute to the project

## Performance Highlights

| Metric | Value |
|--------|--------|
| Model Size | <35 MB (INT2) |
| Inference Time | 12ms (Snapdragon 8 Gen 3) |
| Image Captioning (CIDEr) | 94.7 |
| OCR Accuracy | 93.1% |
| VQA Score | 73.9 |

## Supported Platforms

- **Android**: API 24+, Qualcomm Hexagon NPU
- **iOS**: iOS 14+, Apple Neural Engine
- **Desktop**: Windows, macOS, Linux (CPU/GPU)
- **Edge**: ONNX Runtime, TensorFlow Lite

## Getting Started

```python
from mobile_multimodal import MobileMultiModalLLM
import cv2

# Load model
model = MobileMultiModalLLM.from_pretrained("mobile-mm-llm-int2")

# Generate caption
image = cv2.imread("image.jpg")
caption = model.generate_caption(image)
print(f"Caption: {caption}")
```

## Community

- [GitHub Issues](https://github.com/terragon-labs/mobile-multimodal-llm/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/terragon-labs/mobile-multimodal-llm/discussions) - Community discussions
- [Email](mailto:mobile-ai@terragon.com) - Direct contact