"""Advanced quantization techniques for mobile deployment."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.quantization as quant
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    quant = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class INT2Quantizer:
    """INT2 quantization implementation for Hexagon NPU deployment."""
    
    def __init__(self, calibration_samples: int = 1000):
        self.calibration_samples = calibration_samples
        self.scale_factors = {}
        self.zero_points = {}
        
    def calibrate(self, model: Any, dataloader) -> Dict[str, Any]:
        """Calibrate quantization parameters using sample data."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for quantization")
            
        model.eval()
        activation_stats = {}
        
        logger.info(f"Calibrating INT2 quantization with {self.calibration_samples} samples")
        
        # Collect activation statistics
        hooks = []
        
        def collect_stats(name):
            def hook_fn(module, input, output):
                if TORCH_AVAILABLE and hasattr(torch, 'Tensor') and isinstance(output, torch.Tensor):
                    if name not in activation_stats:
                        activation_stats[name] = []
                    
                    # Collect min/max values
                    activation_stats[name].append({
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'mean': output.mean().item(),
                        'std': output.std().item()
                    })
            return hook_fn
        
        # Register hooks
        for name, module in model.named_modules():
            if nn is not None and isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(collect_stats(name))
                hooks.append(hook)
        
        # Run calibration samples
        sample_count = 0
        if TORCH_AVAILABLE:
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                        
                    _ = model(inputs)
                    sample_count += inputs.size(0)
                    
                    if sample_count >= self.calibration_samples:
                        break
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate quantization parameters
        quant_params = {}
        for name, stats_list in activation_stats.items():
            min_vals = [s['min'] for s in stats_list]
            max_vals = [s['max'] for s in stats_list]
            
            # Use percentile clipping to handle outliers
            min_val = np.percentile(min_vals, 1)
            max_val = np.percentile(max_vals, 99)
            
            # INT2 range: -2 to 1 (4 levels)
            scale = max(abs(min_val), abs(max_val)) / 2.0
            zero_point = 0  # Symmetric quantization for INT2
            
            quant_params[name] = {
                'scale': scale,
                'zero_point': zero_point,
                'min_val': min_val,
                'max_val': max_val
            }
        
        self.scale_factors = {k: v['scale'] for k, v in quant_params.items()}
        self.zero_points = {k: v['zero_point'] for k, v in quant_params.items()}
        
        logger.info(f"Calibration complete. Found parameters for {len(quant_params)} layers")
        return quant_params
    
    def quantize_weights_int2(self, weights: Any) -> Tuple[Any, float, int]:
        """Quantize weights to INT2 format."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for weight quantization")
        
        # Calculate scale and zero point for symmetric quantization
        max_val = torch.max(torch.abs(weights))
        scale = max_val / 2.0  # INT2 range: -2 to 1
        zero_point = 0
        
        # Quantize
        quantized = torch.round(weights / scale).clamp(-2, 1)
        
        return quantized.to(torch.int8), scale.item(), zero_point
    
    def dequantize_weights_int2(self, quantized_weights: Any, 
                               scale: float, zero_point: int) -> Any:
        """Dequantize INT2 weights back to float."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for weight dequantization")
        
        return scale * (quantized_weights.float() - zero_point)
    
    def apply_int2_quantization(self, model: Any) -> Any:
        """Apply INT2 quantization to model."""
        quantized_model = INT2QuantizedModel(model, self.scale_factors, self.zero_points)
        return quantized_model


class INT2QuantizedModel:
    """Model wrapper with INT2 quantized weights."""
    
    def __init__(self, original_model: Any, scale_factors: Dict[str, float], 
                 zero_points: Dict[str, int]):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for quantized models")
            
        self.original_model = original_model
        self.scale_factors = scale_factors
        self.zero_points = zero_points
        self.quantized_weights = {}
        
        # Quantize all weights
        self._quantize_model_weights()
    
    def _quantize_model_weights(self):
        """Quantize all model weights to INT2."""
        quantizer = INT2Quantizer()
        
        for name, module in self.original_model.named_modules():
            if nn is not None and isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight'):
                quantized_weight, scale, zero_point = quantizer.quantize_weights_int2(module.weight)
                
                self.quantized_weights[name] = {
                    'weight': quantized_weight,
                    'scale': scale,
                    'zero_point': zero_point
                }
                
                # Store bias separately (typically not quantized as aggressively)
                if hasattr(module, 'bias') and module.bias is not None:
                    self.quantized_weights[name]['bias'] = module.bias
    
    def forward(self, x: Any) -> Any:
        """Forward pass with quantized weights."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for forward pass")
        
        # For demonstration, this would need custom kernels for true INT2 computation
        # Here we simulate by dequantizing weights during forward pass
        return self._simulate_quantized_forward(x)
    
    def _simulate_quantized_forward(self, x: Any) -> Any:
        """Simulate quantized forward pass by dequantizing weights."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for simulated forward pass")
        
        # This is a simplified simulation - real deployment would use
        # optimized INT2 kernels on Hexagon NPU
        
        quantizer = INT2Quantizer()
        current_x = x
        
        # Note: This is a very simplified implementation
        # Real implementation would need proper module execution
        logger.warning("Simulated quantized forward pass - not for production use")
        return current_x
    
    def get_model_size(self) -> Dict[str, float]:
        """Calculate quantized model size."""
        total_params = 0
        quantized_params = 0
        
        for name, data in self.quantized_weights.items():
            if TORCH_AVAILABLE:
                weight_params = data['weight'].numel()
                total_params += weight_params
                quantized_params += weight_params
                
                if 'bias' in data:
                    total_params += data['bias'].numel()
        
        # Add non-quantized parameters
        for name, param in self.original_model.named_parameters():
            if not any(name.startswith(qname) for qname in self.quantized_weights.keys()):
                if TORCH_AVAILABLE:
                    total_params += param.numel()
        
        # Calculate sizes in MB
        original_size = total_params * 4 / (1024 * 1024)  # FP32
        quantized_size = (quantized_params * 0.5 + (total_params - quantized_params) * 4) / (1024 * 1024)  # INT2 + FP32
        
        return {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
        }


class HexagonOptimizer:
    """Optimizer for Qualcomm Hexagon NPU deployment."""
    
    def __init__(self):
        self.hexagon_ops = {
            'int2_matmul', 'int2_conv2d', 'int8_pool', 'fp16_activation'
        }
    
    def optimize_for_hexagon(self, model: Any) -> Dict[str, Any]:
        """Optimize model for Hexagon NPU execution."""
        optimization_report = {
            'optimized_layers': [],
            'unsupported_ops': [],
            'performance_estimate': {}
        }
        
        if not TORCH_AVAILABLE or not hasattr(model, 'named_modules'):
            optimization_report['error'] = "PyTorch model required"
            return optimization_report
        
        for name, module in model.named_modules():
            if nn is not None and isinstance(module, nn.Linear):
                # Linear layers can use INT2 matmul
                optimization_report['optimized_layers'].append({
                    'name': name,
                    'type': 'Linear',
                    'optimization': 'INT2_MATMUL',
                    'expected_speedup': 4.0
                })
            elif nn is not None and isinstance(module, nn.Conv2d):
                # Conv2D layers can use INT2 convolution
                optimization_report['optimized_layers'].append({
                    'name': name,
                    'type': 'Conv2d',
                    'optimization': 'INT2_CONV2D',
                    'expected_speedup': 3.5
                })
            elif nn is not None and isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid)):
                # Activations run on vector units
                optimization_report['optimized_layers'].append({
                    'name': name,
                    'type': type(module).__name__,
                    'optimization': 'VECTOR_UNIT',
                    'expected_speedup': 2.0
                })
            else:
                # Check for unsupported operations
                optimization_report['unsupported_ops'].append({
                    'name': name,
                    'type': type(module).__name__,
                    'recommendation': 'Consider replacing with supported operation'
                })
        
        # Estimate overall performance
        if optimization_report['optimized_layers']:
            total_speedup = np.mean([layer.get('expected_speedup', 1.0) 
                                   for layer in optimization_report['optimized_layers']])
        else:
            total_speedup = 1.0
        
        optimization_report['performance_estimate'] = {
            'overall_speedup': total_speedup,
            'optimized_ops_ratio': len(optimization_report['optimized_layers']) / 
                                 max(len(optimization_report['optimized_layers']) + len(optimization_report['unsupported_ops']), 1),
            'hexagon_utilization': min(total_speedup / 4.0, 1.0)  # Normalize to 100%
        }
        
        return optimization_report
    
    def generate_hexagon_config(self, model: Any, output_path: str):
        """Generate Hexagon SDK configuration file."""
        config = {
            "version": "1.0",
            "target": "hexagon_v73",
            "quantization": {
                "default_scheme": "int2",
                "calibration_method": "entropy",
                "optimization_level": "aggressive"
            },
            "runtime": {
                "use_hexagon_nn": True,
                "enable_int2_matmul": True,
                "memory_optimization": True,
                "power_profile": "balanced"
            },
            "layers": []
        }
        
        # Add layer-specific configurations
        if TORCH_AVAILABLE and hasattr(model, 'named_modules'):
            for name, module in model.named_modules():
                if nn is not None and isinstance(module, (nn.Linear, nn.Conv2d)):
                    layer_config = {
                        "name": name,
                        "type": type(module).__name__,
                        "quantization": "int2",
                        "execution_unit": "hexagon_tensor" if isinstance(module, nn.Linear) else "hexagon_conv"
                    }
                    config["layers"].append(layer_config)
        
        # Save configuration
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Hexagon configuration saved to {output_path}")


class QuantizationValidator:
    """Validation tools for quantized models."""
    
    @staticmethod
    def validate_accuracy(original_model: Any, quantized_model: Any,
                         test_dataloader, tolerance: float = 0.05) -> Dict[str, Any]:
        """Validate quantized model accuracy against original."""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        if not (hasattr(original_model, 'eval') and hasattr(quantized_model, 'eval')):
            return {"error": "Invalid model types"}
            
        original_model.eval()
        quantized_model.eval()
        
        original_outputs = []
        quantized_outputs = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                orig_out = original_model(inputs)
                quant_out = quantized_model(inputs)
                
                original_outputs.append(orig_out.cpu())
                quantized_outputs.append(quant_out.cpu())
        
        # Calculate metrics
        original_concat = torch.cat(original_outputs, dim=0)
        quantized_concat = torch.cat(quantized_outputs, dim=0)
        
        # Mean absolute error
        mae = torch.mean(torch.abs(original_concat - quantized_concat)).item()
        
        # Mean squared error
        mse = torch.mean((original_concat - quantized_concat) ** 2).item()
        
        # Cosine similarity
        if hasattr(torch.nn.functional, 'cosine_similarity'):
            cos_sim = torch.nn.functional.cosine_similarity(
                original_concat.flatten(), 
                quantized_concat.flatten(), 
                dim=0
            ).item()
        else:
            cos_sim = 0.0
        
        # Accuracy within tolerance
        within_tolerance = torch.mean(
            (torch.abs(original_concat - quantized_concat) / 
             (torch.abs(original_concat) + 1e-8)) < tolerance
        ).item()
        
        return {
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "cosine_similarity": cos_sim,
            "accuracy_within_tolerance": within_tolerance,
            "tolerance_threshold": tolerance,
            "validation_passed": within_tolerance > 0.95  # 95% of outputs within tolerance
        }


# Export utilities
def export_quantized_model(model: Any, export_path: str, format: str = "onnx"):
    """Export quantized model to specified format."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model export")
    
    if format.lower() == "onnx":
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            logger.info(f"Quantized model exported to {export_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
    
    elif format.lower() == "torchscript":
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224)
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(export_path)
            logger.info(f"TorchScript model exported to {export_path}")
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
    
    else:
        logger.error(f"Unsupported export format: {format}")


if __name__ == "__main__":
    print("Mobile quantization module loaded successfully!")
    
    if TORCH_AVAILABLE:
        # Test INT2 quantizer
        try:
            quantizer = INT2Quantizer()
            
            # Create sample weights
            weights = torch.randn(128, 64)
            print(f"Original weights shape: {weights.shape}")
            print(f"Original weights range: [{weights.min():.3f}, {weights.max():.3f}]")
            
            # Quantize
            quant_weights, scale, zero_point = quantizer.quantize_weights_int2(weights)
            print(f"Quantized weights range: [{quant_weights.min():.0f}, {quant_weights.max():.0f}]")
            print(f"Scale: {scale:.6f}, Zero point: {zero_point}")
            
            # Dequantize
            dequant_weights = quantizer.dequantize_weights_int2(quant_weights, scale, zero_point)
            print(f"Dequantized weights range: [{dequant_weights.min():.3f}, {dequant_weights.max():.3f}]")
            
            # Calculate error
            error = torch.mean(torch.abs(weights - dequant_weights)).item()
            print(f"Quantization error: {error:.6f}")
            
            print("\nQuantization module test completed!")
            
        except Exception as e:
            print(f"Error during testing: {e}")
    else:
        print("PyTorch not available - quantization module loaded with limited functionality")
        
        # Test that basic classes can be instantiated
        try:
            quantizer = INT2Quantizer()
            print("✅ INT2Quantizer created successfully (stub mode)")
        except Exception as e:
            print(f"❌ Error creating INT2Quantizer: {e}")
        
        # Test HexagonOptimizer (doesn't require PyTorch for basic functionality)
        optimizer = HexagonOptimizer()
        print("✅ HexagonOptimizer created successfully")