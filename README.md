# Mobile Multi-Modal LLM

Tiny (<35MB) vision-text transformer for on-device captioning, OCR, and retrieval.

## Features
- TinyVisionEncoder: Patch embedding + 2-layer attention (~5M params)
- TinyTextDecoder: 3-layer transformer with shared vocab embedding
- CrossModalFusion: Cross-attention between vision and text features
- INT2Quantizer: 2-bit weight quantization simulation
- Benchmarking: Parameter counts and memory footprint

## Install
pip install numpy

## Usage
```python
from mmmllm.vision_encoder import TinyVisionEncoder
from mmmllm.benchmark import benchmark_model
import numpy as np

enc = TinyVisionEncoder()
img = np.random.rand(224, 224, 3)
feats = enc.forward(img)
print(feats.shape)

stats = benchmark_model()
print(f"Total params: {stats['total_params']:,}")
print(f"Memory (INT2): {stats['memory_mb_int2']:.1f} MB")
```

## Run Tests
```
~/anaconda3/bin/python3 -m pytest tests/ -v
```
