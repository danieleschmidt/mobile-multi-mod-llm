# ADR-001: INT2 Quantization for Mobile Deployment

## Status
Accepted

## Context
Mobile devices have severe memory and compute constraints. Traditional FP32 models are too large (>100MB) and slow for real-time on-device inference. We need to choose an optimal quantization strategy that balances model size, inference speed, and accuracy preservation.

## Decision
We will use INT2 quantization with Qualcomm's Hexagon NPU SDK for mobile deployment, targeting <35MB model size while maintaining >93% of original accuracy.

## Consequences

### Positive Consequences
- **Ultra-compact models**: 4-8x size reduction vs INT8
- **Hardware acceleration**: Native INT2 support on Hexagon NPU
- **Battery efficiency**: Significantly reduced power consumption
- **Real-time performance**: <15ms inference latency
- **Privacy**: Enables complete on-device processing

### Negative Consequences
- **Accuracy degradation**: 1-3% performance drop vs FP32
- **Hardware dependency**: Requires Qualcomm Hexagon NPU for optimal performance
- **Development complexity**: Custom quantization pipeline needed
- **Limited debugging**: Harder to debug quantized models

### Risks
- **Hardware fragmentation**: Not all Android devices have Hexagon NPU
- **Quantization drift**: Model accuracy may degrade over time
- **Tool maturity**: INT2 quantization tools are relatively new

## Alternatives Considered

### INT8 Quantization
- **Pros**: Better accuracy preservation, wider hardware support
- **Cons**: 2x larger models (60-70MB), slower on some hardware
- **Verdict**: Rejected due to size constraints

### FP16 Quantization  
- **Pros**: Minimal accuracy loss, good hardware support
- **Cons**: Still too large (70-80MB), limited mobile acceleration
- **Verdict**: Rejected due to size and performance constraints

### Dynamic Quantization
- **Pros**: Better accuracy than static quantization
- **Cons**: Runtime overhead, inconsistent performance
- **Verdict**: Rejected due to performance unpredictability

### Pruning + INT8
- **Pros**: Good size/accuracy tradeoff
- **Cons**: More complex pipeline, still larger than INT2
- **Verdict**: Considered for future optimization

## Implementation Notes

### Quantization Pipeline
1. **Calibration**: Use 1000 diverse samples from COCO/TextOCR
2. **QAT Training**: Quantization-aware training for 10 epochs
3. **Hardware Validation**: Verify accuracy on target devices
4. **Fallback Strategy**: INT8 model for non-Hexagon devices

### Accuracy Validation
- Maintain >90% of FP32 performance across all tasks
- Continuous monitoring for quantization drift
- Regular re-calibration with new data

### Hardware Support Matrix
- **Primary**: Qualcomm Hexagon NPU (Snapdragon 8 Gen 2+)
- **Secondary**: ARM Mali GPU with INT8 fallback
- **Fallback**: CPU-only INT8 execution

---
**Date**: 2025-01-15  
**Author**: Mobile AI Team  
**Reviewers**: Hardware Team, ML Team  
**Status Last Updated**: 2025-01-15