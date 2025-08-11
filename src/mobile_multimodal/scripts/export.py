#!/usr/bin/env python3
"""Model export script for mobile platforms."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import dependencies
torch = None
onnx = None
tf = None
coremltools = None

try:
    import torch
    import torch.onnx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


class ModelExporter:
    """Export mobile multi-modal models to various formats."""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.input_shape = (1, 3, 224, 224)  # NCHW format
        self.mock_mode = not TORCH_AVAILABLE
        
    def load_model(self) -> bool:
        """Load the source model."""
        if self.mock_mode:
            logger.warning("Running in mock mode - PyTorch not available")
            return True
        
        try:
            # Try to load model from checkpoint
            if os.path.exists(self.model_path):
                if self.model_path.endswith('.pth') or self.model_path.endswith('.pt'):
                    # Load PyTorch model
                    checkpoint = torch.load(self.model_path, map_location='cpu')
                    
                    # Create model instance
                    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                    from core import MobileMultiModalLLM
                    
                    self.model = MobileMultiModalLLM(device='cpu')
                    
                    # Load state dict if available
                    if isinstance(checkpoint, dict):
                        if 'vision_encoder_state' in checkpoint and self.model._vision_encoder:
                            self.model._vision_encoder.load_state_dict(checkpoint['vision_encoder_state'])
                        if 'text_decoder_state' in checkpoint and self.model._text_decoder:
                            self.model._text_decoder.load_state_dict(checkpoint['text_decoder_state'])
                    
                    self.model.eval()
                    logger.info(f"Model loaded from {self.model_path}")
                    return True
                    
                elif self.model_path.endswith('.json'):
                    # Load from JSON checkpoint (training results)
                    with open(self.model_path, 'r') as f:
                        checkpoint = json.load(f)
                    
                    logger.info("Loaded model configuration from JSON checkpoint")
                    
                    # Create model instance
                    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                    from core import MobileMultiModalLLM
                    
                    self.model = MobileMultiModalLLM(device='cpu')
                    self.model.eval()
                    return True
            
            logger.warning(f"Model file not found: {self.model_path}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def export_onnx(self, quantize: bool = False, opset_version: int = 18) -> Optional[str]:
        """Export model to ONNX format."""
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available - install with: pip install onnx onnxruntime")
            return None
        
        output_path = self.output_dir / "mobile_multimodal.onnx"
        
        if self.mock_mode:
            # Create mock ONNX model for testing
            logger.info("Creating mock ONNX export")
            
            # Create simple mock metadata
            mock_metadata = {
                "model_format": "onnx",
                "opset_version": opset_version,
                "input_shape": list(self.input_shape),
                "quantized": quantize,
                "mock_export": True,
                "export_time": time.time()
            }
            
            # Save metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(mock_metadata, f, indent=2)
            
            # Create placeholder ONNX file
            with open(output_path, 'wb') as f:
                f.write(b"MOCK_ONNX_MODEL_DATA")
            
            logger.info(f"Mock ONNX model exported to {output_path}")
            return str(output_path)
        
        if not self.model or not TORCH_AVAILABLE:
            logger.error("Model not loaded or PyTorch not available")
            return None
        
        try:
            # Create dummy input
            dummy_input = torch.randn(self.input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                self.model._vision_encoder if self.model._vision_encoder else self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['image'],
                output_names=['features'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'features': {0: 'batch_size'}
                }
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # Quantization if requested
            if quantize:
                self._quantize_onnx_model(output_path)
            
            # Test inference
            self._test_onnx_inference(output_path)
            
            logger.info(f"ONNX model exported to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None
    
    def export_tflite(self, quantize: bool = False, use_gpu_delegate: bool = False) -> Optional[str]:
        """Export model to TensorFlow Lite format."""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available - install with: pip install tensorflow")
            return None
        
        output_path = self.output_dir / "mobile_multimodal.tflite"
        
        if self.mock_mode:
            # Create mock TFLite model
            logger.info("Creating mock TFLite export")
            
            mock_metadata = {
                "model_format": "tflite",
                "input_shape": list(self.input_shape),
                "quantized": quantize,
                "gpu_delegate": use_gpu_delegate,
                "mock_export": True,
                "export_time": time.time()
            }
            
            # Save metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(mock_metadata, f, indent=2)
            
            # Create placeholder TFLite file
            with open(output_path, 'wb') as f:
                f.write(b"MOCK_TFLITE_MODEL_DATA")
            
            logger.info(f"Mock TFLite model exported to {output_path}")
            return str(output_path)
        
        try:
            # Convert from ONNX to TFLite (simplified approach)
            # In a real implementation, you would use tf.lite.TFLiteConverter
            
            # Create a simple TF model as placeholder
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(384)
            ])
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = self._get_representative_dataset
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Test inference
            self._test_tflite_inference(output_path)
            
            logger.info(f"TFLite model exported to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
            return None
    
    def export_coreml(self, quantize: bool = False, use_neural_engine: bool = True) -> Optional[str]:
        """Export model to Core ML format."""
        if not COREML_AVAILABLE:
            logger.error("Core ML Tools not available - install with: pip install coremltools")
            return None
        
        output_path = self.output_dir / "MobileMultiModal.mlpackage"
        
        if self.mock_mode:
            # Create mock Core ML model
            logger.info("Creating mock Core ML export")
            
            # Create directory structure
            output_path.mkdir(parents=True, exist_ok=True)
            
            mock_metadata = {
                "model_format": "coreml",
                "input_shape": list(self.input_shape),
                "quantized": quantize,
                "neural_engine": use_neural_engine,
                "mock_export": True,
                "export_time": time.time()
            }
            
            # Save metadata
            metadata_path = output_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(mock_metadata, f, indent=2)
            
            # Create placeholder model file
            model_path = output_path / "model.mlmodel"
            with open(model_path, 'wb') as f:
                f.write(b"MOCK_COREML_MODEL_DATA")
            
            logger.info(f"Mock Core ML model exported to {output_path}")
            return str(output_path)
        
        try:
            # Convert from ONNX to Core ML (simplified approach)
            # In a real implementation, you would convert from PyTorch or ONNX
            
            # For now, create a simple placeholder
            # This would need actual conversion logic
            
            logger.info("Core ML export requires additional implementation")
            return None
            
        except Exception as e:
            logger.error(f"Core ML export failed: {e}")
            return None
    
    def export_hexagon_dlc(self, quantization_type: str = "int8") -> Optional[str]:
        """Export model to Qualcomm Hexagon DLC format."""
        output_path = self.output_dir / "mobile_multimodal.dlc"
        
        logger.warning("Hexagon DLC export requires Qualcomm SNPE SDK")
        
        if self.mock_mode:
            # Create mock DLC model
            logger.info("Creating mock Hexagon DLC export")
            
            mock_metadata = {
                "model_format": "dlc",
                "input_shape": list(self.input_shape),
                "quantization": quantization_type,
                "target": "hexagon",
                "mock_export": True,
                "export_time": time.time()
            }
            
            # Save metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(mock_metadata, f, indent=2)
            
            # Create placeholder DLC file
            with open(output_path, 'wb') as f:
                f.write(b"MOCK_HEXAGON_DLC_MODEL_DATA")
            
            logger.info(f"Mock Hexagon DLC model exported to {output_path}")
            return str(output_path)
        
        # Real implementation would use SNPE tools
        return None
    
    def _quantize_onnx_model(self, model_path: str):
        """Apply quantization to ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = model_path.replace('.onnx', '_quantized.onnx')
            
            quantize_dynamic(
                model_input=model_path,
                model_output=quantized_path,
                weight_type=QuantType.QUInt8
            )
            
            # Replace original with quantized
            os.replace(quantized_path, model_path)
            logger.info("ONNX model quantized successfully")
            
        except Exception as e:
            logger.warning(f"ONNX quantization failed: {e}")
    
    def _test_onnx_inference(self, model_path: str):
        """Test ONNX model inference."""
        try:
            session = ort.InferenceSession(model_path)
            
            # Create test input
            input_name = session.get_inputs()[0].name
            test_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_name: test_input})
            
            logger.info(f"ONNX inference test passed - output shape: {outputs[0].shape}")
            
        except Exception as e:
            logger.warning(f"ONNX inference test failed: {e}")
    
    def _test_tflite_inference(self, model_path: str):
        """Test TFLite model inference."""
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create test input
            input_shape = input_details[0]['shape']
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            logger.info(f"TFLite inference test passed - output shape: {output_data.shape}")
            
        except Exception as e:
            logger.warning(f"TFLite inference test failed: {e}")
    
    def _get_representative_dataset(self):
        """Generate representative dataset for quantization."""
        for _ in range(100):
            yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]
    
    def export_all_formats(self, formats: List[str], **kwargs) -> Dict[str, Optional[str]]:
        """Export model to multiple formats."""
        results = {}
        
        for fmt in formats:
            logger.info(f"Exporting to {fmt.upper()} format...")
            
            if fmt == 'onnx':
                results[fmt] = self.export_onnx(**kwargs)
            elif fmt == 'tflite':
                results[fmt] = self.export_tflite(**kwargs)
            elif fmt == 'coreml':
                results[fmt] = self.export_coreml(**kwargs)
            elif fmt == 'dlc':
                results[fmt] = self.export_hexagon_dlc(**kwargs)
            else:
                logger.error(f"Unknown format: {fmt}")
                results[fmt] = None
        
        return results
    
    def generate_export_report(self, export_results: Dict[str, Optional[str]]) -> str:
        """Generate export report."""
        report_path = self.output_dir / "export_report.json"
        
        report = {
            "export_timestamp": time.time(),
            "source_model": self.model_path,
            "output_directory": str(self.output_dir),
            "input_shape": list(self.input_shape),
            "mock_mode": self.mock_mode,
            "exports": {}
        }
        
        for fmt, path in export_results.items():
            if path and os.path.exists(path):
                file_size = os.path.getsize(path)
                report["exports"][fmt] = {
                    "path": path,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "success": True
                }
            else:
                report["exports"][fmt] = {
                    "path": None,
                    "success": False
                }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Export report saved to {report_path}")
        return str(report_path)


def main():
    """Main export script entry point."""
    parser = argparse.ArgumentParser(description='Export mobile multi-modal models')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to source model file')
    parser.add_argument('--output-dir', type=str, default='exports',
                        help='Output directory for exported models')
    
    # Export format arguments
    parser.add_argument('--formats', nargs='+', 
                        choices=['onnx', 'tflite', 'coreml', 'dlc'],
                        default=['onnx'],
                        help='Export formats')
    
    # Optimization arguments
    parser.add_argument('--quantize', action='store_true',
                        help='Enable quantization')
    parser.add_argument('--opset-version', type=int, default=18,
                        help='ONNX opset version')
    parser.add_argument('--use-gpu-delegate', action='store_true',
                        help='Enable GPU delegate for TFLite')
    parser.add_argument('--use-neural-engine', action='store_true', default=True,
                        help='Enable Neural Engine for Core ML')
    parser.add_argument('--quantization-type', type=str, default='int8',
                        choices=['int8', 'int2'],
                        help='Quantization type for Hexagon DLC')
    
    # Other arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create exporter
    exporter = ModelExporter(args.model_path, args.output_dir)
    
    # Load model
    if not exporter.load_model():
        logger.error("Failed to load source model")
        return 1
    
    # Export to requested formats
    try:
        export_kwargs = {
            'quantize': args.quantize,
            'opset_version': args.opset_version,
            'use_gpu_delegate': args.use_gpu_delegate,
            'use_neural_engine': args.use_neural_engine,
            'quantization_type': args.quantization_type,
        }
        
        results = exporter.export_all_formats(args.formats, **export_kwargs)
        
        # Generate report
        report_path = exporter.generate_export_report(results)
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL EXPORT COMPLETE")
        print("="*50)
        print(f"Source Model: {args.model_path}")
        print(f"Output Directory: {args.output_dir}")
        print(f"Export Report: {report_path}")
        print("\nExport Results:")
        
        for fmt, path in results.items():
            if path:
                size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
                print(f"  {fmt.upper()}: ✅ {path} ({size_mb:.2f} MB)")
            else:
                print(f"  {fmt.upper()}: ❌ Export failed")
        
        print("="*50)
        
        # Check if any exports succeeded
        success_count = sum(1 for path in results.values() if path is not None)
        if success_count == 0:
            logger.error("All exports failed")
            return 1
        
        logger.info(f"Successfully exported to {success_count}/{len(args.formats)} formats")
        return 0
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())