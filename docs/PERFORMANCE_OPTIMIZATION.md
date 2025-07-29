# Performance Optimization Guide

This document provides comprehensive guidance for optimizing the Mobile Multi-Modal LLM for various deployment scenarios.

## Overview

Performance optimization for mobile multimodal models involves multiple dimensions:
- **Model Size**: Minimizing memory footprint
- **Inference Speed**: Reducing latency for real-time applications  
- **Energy Efficiency**: Optimizing battery usage on mobile devices
- **Accuracy Preservation**: Maintaining quality while optimizing

## Hardware-Specific Optimizations

### Qualcomm Hexagon NPU (Android)

#### INT2 Quantization
```python
# Optimal quantization configuration for Hexagon
quantization_config = {
    'weight_quantization': 'int2',
    'activation_quantization': 'int8',
    'calibration_samples': 1000,
    'hardware_target': 'hexagon_v73'
}
```

#### Operator Optimization
- **Preferred Ops**: Conv2D, MatMul, Add, ReLU, BatchNorm
- **Avoid**: Complex activations (GELU, Swish), dynamic shapes
- **Memory Layout**: Use NHWC for optimal NPU utilization

#### Performance Tuning
```bash
# Export model with Hexagon optimizations
python scripts/export_models.py \
    --platform android \
    --quantization int2 \
    --hardware_target hexagon_v73 \
    --optimize_for_latency
```

### Apple Neural Engine (iOS)

#### Core ML Optimization
```python
# Neural Engine friendly configuration
coreml_config = {
    'compute_units': 'CPU_AND_NE',  # Use Neural Engine
    'precision': 'int8',  # INT8 for NE compatibility
    'batch_size': 1,      # Single inference
    'target_ios_version': '14.0'
}
```

#### Operator Constraints
- **Supported**: Conv2D, Dense, Pooling, Element-wise operations
- **Limited**: RNN layers, custom operations
- **Optimization**: Use separable convolutions, avoid branches

## Model Architecture Optimizations

### Network Pruning

#### Structured Pruning
```python
from mobile_multimodal.optimization import StructuredPruner

pruner = StructuredPruner(
    sparsity_ratio=0.5,        # 50% pruning
    importance_metric='l1',     # L1 norm importance
    granularity='channel'       # Channel-wise pruning
)

pruned_model = pruner.prune(model)
```

#### Unstructured Pruning
```python
from mobile_multimodal.optimization import UnstructuredPruner

pruner = UnstructuredPruner(
    sparsity_ratio=0.8,        # 80% sparsity
    schedule='gradual',        # Progressive pruning
    recovery_epochs=10         # Fine-tuning after pruning
)
```

### Knowledge Distillation

#### Teacher-Student Training
```python
from mobile_multimodal.distillation import KnowledgeDistiller

distiller = KnowledgeDistiller(
    teacher_model=large_model,
    student_model=mobile_model,
    temperature=4.0,           # Softmax temperature
    alpha=0.7,                 # Distillation weight
    beta=0.3                   # Task loss weight
)

distilled_model = distiller.train(train_loader, epochs=50)
```

### Neural Architecture Search (NAS)

#### Mobile-Optimized Search Space
```python
search_space = {
    'depth': [8, 12, 16],              # Number of layers
    'width': [256, 384, 512],          # Hidden dimensions
    'kernel_sizes': [3, 5, 7],         # Convolution kernels
    'attention_heads': [4, 8, 12],     # Multi-head attention
    'activation': ['relu', 'swish'],    # Activation functions
}

nas_trainer = MobileNASTrainer(
    search_space=search_space,
    hardware_target='snapdragon_8gen3',
    latency_constraint=15,  # 15ms max latency
    size_constraint=35      # 35MB max size
)
```

## Quantization Strategies

### Progressive Quantization

#### Stage 1: INT8 Quantization
```python
# Start with INT8 for stability
int8_model = quantize_model(
    model=base_model,
    quantization_type='int8',
    calibration_data=calibration_loader,
    quantization_aware_training=True
)
```

#### Stage 2: INT4 Quantization
```python
# Progress to INT4 with careful tuning
int4_model = quantize_model(
    model=int8_model,  # Use INT8 as starting point
    quantization_type='int4',
    mixed_precision=True,  # Keep sensitive layers in INT8
    accuracy_threshold=0.95  # Maintain 95% of original accuracy
)
```

#### Stage 3: INT2 Quantization (Experimental)
```python
# Extreme quantization for maximum efficiency
int2_model = quantize_model(
    model=int4_model,
    quantization_type='int2',
    custom_kernels=True,     # Use optimized kernels
    outlier_handling='clip',  # Handle extreme values
    fine_tune_epochs=20      # Extended fine-tuning
)
```

### Mixed Precision Strategy

```python
# Configure layer-wise quantization
mixed_precision_config = {
    'attention_layers': 'int8',    # Keep attention in INT8
    'classification_head': 'int8', # Keep final layers precise
    'feature_extractors': 'int4',  # Feature layers can be INT4
    'early_convolutions': 'int2'   # Early layers most quantizable
}
```

## Memory Optimization

### Gradient Checkpointing

```python
# Enable gradient checkpointing for memory efficiency
model = enable_gradient_checkpointing(
    model,
    checkpoint_ratio=0.5  # Checkpoint 50% of layers
)
```

### Dynamic Batching

```python
class DynamicBatcher:
    def __init__(self, max_memory_mb=100):
        self.max_memory = max_memory_mb
    
    def get_optimal_batch_size(self, input_shape):
        # Calculate optimal batch size based on memory constraints
        memory_per_sample = estimate_memory_usage(input_shape)
        return min(32, self.max_memory // memory_per_sample)
```

### Memory Pooling

```python
# Pre-allocate memory pools for mobile deployment
memory_pool = {
    'input_buffer': allocate_buffer(224 * 224 * 3 * 4),  # RGBA image
    'hidden_states': allocate_buffer(512 * 1024 * 2),    # Hidden layer
    'output_buffer': allocate_buffer(1000 * 4)           # Classification output
}
```

## Inference Optimization

### Operator Fusion

```python
# Fuse operations for reduced memory access
optimized_model = fuse_operations(
    model,
    fusion_patterns=[
        'conv_bn_relu',      # Conv + BatchNorm + ReLU
        'linear_relu',       # Linear + ReLU
        'attention_qkv'      # Q, K, V computation
    ]
)
```

### Graph Optimization

```python
# Apply graph-level optimizations
graph_optimizer = GraphOptimizer([
    'constant_folding',      # Pre-compute constants
    'dead_code_elimination', # Remove unused operations
    'layout_optimization',   # Optimize tensor layouts
    'operator_scheduling'    # Optimize execution order
])

optimized_graph = graph_optimizer.optimize(model.graph)
```

## Platform-Specific Tips

### Android Optimization

#### GPU Delegation (Optional)
```python
# Use GPU for larger models when NPU unavailable
tflite_config = {
    'gpu_delegate': True,
    'gpu_precision': 'fp16',
    'allow_fp16_precision_for_fp32': True
}
```

#### Thread Configuration
```python
# Optimize threading for Android
android_config = {
    'num_threads': 4,           # Use 4 CPU cores
    'thread_affinity': 'big',   # Prefer big cores
    'power_mode': 'balanced'    # Balance performance/power
}
```

### iOS Optimization

#### Compute Unit Selection
```python
# Strategic compute unit assignment
compute_strategy = {
    'vision_encoder': 'neural_engine',  # Use NE for vision
    'text_processing': 'cpu',          # CPU for text
    'fusion_layers': 'gpu'             # GPU for complex ops
}
```

#### Memory Management
```python
# iOS-specific memory handling
ios_config = {
    'memory_warnings': True,      # Handle memory warnings
    'background_mode': 'suspend', # Suspend in background
    'cache_strategy': 'lru'       # LRU cache for models
}
```

## Benchmarking and Profiling

### Performance Measurement

```python
# Comprehensive performance benchmarking
benchmark_suite = PerformanceBenchmark([
    'inference_latency',     # End-to-end latency
    'memory_usage',          # Peak memory consumption
    'energy_consumption',    # Battery usage
    'thermal_throttling',    # Temperature impact
    'accuracy_validation'    # Quality preservation
])

results = benchmark_suite.run(model, test_data)
```

### Continuous Monitoring

```python
# Set up performance monitoring in production
monitor = ProductionMonitor(
    metrics=['latency_p95', 'memory_peak', 'battery_drain'],
    alerts=[
        Alert('latency_p95 > 50ms', severity='warning'),
        Alert('memory_peak > 100MB', severity='critical')
    ]
)
```

## Best Practices Summary

### DO:
✅ **Profile First**: Always profile before optimizing  
✅ **Incremental Optimization**: Make small, measurable changes  
✅ **Hardware-Aware**: Optimize for target hardware specifically  
✅ **Accuracy Validation**: Continuously validate model quality  
✅ **Real Device Testing**: Test on actual mobile devices  

### DON'T:
❌ **Premature Optimization**: Don't optimize without measurements  
❌ **Ignore Accuracy**: Don't sacrifice quality for speed  
❌ **One-Size-Fits-All**: Don't use same optimizations for all platforms  
❌ **Skip Validation**: Don't deploy without thorough testing  
❌ **Over-Engineer**: Don't add unnecessary complexity  

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model Size | <35MB | Compressed model file |
| Inference Latency | <20ms | P95 on Snapdragon 8 Gen 3 |
| Memory Usage | <100MB | Peak RAM during inference |
| Battery Impact | <1% drain/hour | Continuous inference |
| Accuracy Drop | <5% | Compared to FP32 baseline |

## Troubleshooting Common Issues

### High Latency
- Check for unoptimized operations
- Verify hardware acceleration is enabled
- Profile memory access patterns
- Consider reducing model complexity

### Memory Issues
- Enable gradient checkpointing
- Use dynamic batching
- Implement memory pooling
- Check for memory leaks

### Accuracy Degradation
- Increase calibration dataset size
- Use mixed precision quantization
- Apply knowledge distillation
- Fine-tune after optimization

For additional support, consult the [troubleshooting guide](TROUBLESHOOTING.md) or reach out to the optimization team.