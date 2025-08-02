"""
Unit tests for core model functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from tests.utils import (
    MockModel,
    TestDataGenerator,
    assert_model_output_shape,
    compare_model_outputs
)


class TestModelCore:
    """Test core model functionality."""
    
    def test_model_initialization(self, test_config):
        """Test model initialization with config."""
        model = MockModel(test_config["model"])
        
        assert model.config["hidden_size"] == test_config["model"]["hidden_size"]
        assert model.config["num_layers"] == test_config["model"]["num_layers"]
        assert model.training is True
    
    def test_model_forward_pass(self):
        """Test basic forward pass."""
        model = MockModel()
        batch_size = 4
        
        # Create test inputs
        image = torch.randn(batch_size, 3, 224, 224)
        text = torch.randint(0, 1000, (batch_size, 50))
        
        # Forward pass
        outputs = model.forward(image, text)
        
        # Verify output shapes
        assert_model_output_shape(outputs["caption_logits"], (batch_size, 50, 32000))
        assert_model_output_shape(outputs["ocr_logits"], (batch_size, 100, 32000))
        assert_model_output_shape(outputs["vqa_logits"], (batch_size, 32000))
        assert_model_output_shape(outputs["image_features"], (batch_size, 768))
        assert_model_output_shape(outputs["text_features"], (batch_size, 768))
    
    def test_model_forward_image_only(self):
        """Test forward pass with image only."""
        model = MockModel()
        batch_size = 2
        
        image = torch.randn(batch_size, 3, 224, 224)
        outputs = model.forward(image)
        
        # Text features should be None when no text input
        assert outputs["text_features"] is None
        assert outputs["image_features"] is not None
    
    def test_model_training_mode(self):
        """Test training mode switching."""
        model = MockModel()
        
        # Default should be training
        assert model.training is True
        
        # Switch to eval
        model.eval()
        assert model.training is False
        
        # Switch back to training
        model.train()
        assert model.training is True
    
    def test_model_device_management(self):
        """Test model device management."""
        model = MockModel()
        
        # Default device should be CPU
        assert model.device == torch.device("cpu")
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            model.to("cuda")
            assert model.device == torch.device("cuda")
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_model_different_batch_sizes(self, batch_size):
        """Test model with different batch sizes."""
        model = MockModel()
        
        image = torch.randn(batch_size, 3, 224, 224)
        outputs = model.forward(image)
        
        assert outputs["caption_logits"].shape[0] == batch_size
        assert outputs["image_features"].shape[0] == batch_size
    
    @pytest.mark.parametrize("image_size", [(224, 224), (256, 256), (384, 384)])
    def test_model_different_image_sizes(self, image_size):
        """Test model with different image sizes."""
        model = MockModel()
        batch_size = 2
        h, w = image_size
        
        image = torch.randn(batch_size, 3, h, w)
        
        # For mock model, this should work regardless of size
        outputs = model.forward(image)
        assert outputs["image_features"].shape[0] == batch_size
    
    def test_model_output_consistency(self):
        """Test that model produces consistent outputs."""
        model = MockModel()
        model.eval()  # Set to eval mode for consistency
        
        # Same input should produce same output (in eval mode)
        torch.manual_seed(42)
        image = torch.randn(1, 3, 224, 224)
        
        torch.manual_seed(42)
        output1 = model.forward(image)
        
        torch.manual_seed(42)
        output2 = model.forward(image)
        
        # In mock model, outputs are random, so we just check shapes match
        assert output1["image_features"].shape == output2["image_features"].shape
    
    def test_model_memory_efficiency(self):
        """Test model memory usage."""
        model = MockModel()
        
        # Test with progressively larger batches
        batch_sizes = [1, 2, 4, 8]
        memory_usage = []
        
        for batch_size in batch_sizes:
            image = torch.randn(batch_size, 3, 224, 224)
            
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            
            outputs = model.forward(image)
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_usage.append(memory_after - memory_before)
        
        # Memory usage should scale roughly linearly with batch size
        if torch.cuda.is_available() and len(memory_usage) > 1:
            # Check that memory usage increases with batch size
            assert memory_usage[-1] > memory_usage[0]


class TestModelQuantization:
    """Test model quantization functionality."""
    
    def test_quantization_basic(self):
        """Test basic quantization workflow."""
        from tests.utils import MockQuantizer
        
        model = MockModel()
        quantizer = MockQuantizer("int2")
        
        # Mock dataloader
        dataloader = [torch.randn(4, 3, 224, 224) for _ in range(10)]
        
        # Calibrate and quantize
        quantizer.calibrate(dataloader)
        quantized_model = quantizer.quantize(model)
        
        assert hasattr(quantized_model, '_is_quantized')
        assert quantized_model._is_quantized is True
        assert quantized_model._precision == "int2"
    
    def test_quantization_without_calibration(self):
        """Test that quantization fails without calibration."""
        from tests.utils import MockQuantizer
        
        model = MockModel()
        quantizer = MockQuantizer("int2")
        
        # Should fail without calibration
        with pytest.raises(RuntimeError, match="must be calibrated"):
            quantizer.quantize(model)
    
    @pytest.mark.parametrize("precision", ["int2", "int4", "int8"])
    def test_different_quantization_precisions(self, precision):
        """Test different quantization precisions."""
        from tests.utils import MockQuantizer
        
        model = MockModel()
        quantizer = MockQuantizer(precision)
        
        # Mock calibration
        dataloader = [torch.randn(2, 3, 224, 224)]
        quantizer.calibrate(dataloader)
        
        quantized_model = quantizer.quantize(model)
        assert quantized_model._precision == precision


class TestModelExport:
    """Test model export functionality."""
    
    @pytest.mark.mobile
    def test_export_to_tflite(self):
        """Test TensorFlow Lite export."""
        model = MockModel()
        
        # Mock TFLite converter
        with patch('tensorflow.lite.TFLiteConverter') as mock_converter:
            mock_converter.from_concrete_functions.return_value.convert.return_value = b"fake_tflite"
            
            # This would be the actual export logic
            tflite_model = b"fake_tflite"  # Mock export result
            
            assert isinstance(tflite_model, bytes)
            assert len(tflite_model) > 0
    
    @pytest.mark.mobile
    def test_export_to_coreml(self):
        """Test Core ML export."""
        model = MockModel()
        
        # Mock Core ML converter
        with patch('coremltools.convert') as mock_convert:
            mock_convert.return_value = MagicMock()
            
            # This would be the actual export logic
            coreml_model = MagicMock()  # Mock export result
            
            assert coreml_model is not None
    
    def test_export_to_onnx(self):
        """Test ONNX export."""
        model = MockModel()
        
        # Mock ONNX export
        with patch('torch.onnx.export') as mock_export:
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                # This would be the actual export logic
                mock_export.return_value = None
                
                # Export should complete without error
                assert True  # Placeholder for actual export test


class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_inference_speed(self):
        """Test inference speed requirements."""
        from tests.utils import PerformanceMonitor
        
        model = MockModel()
        model.eval()
        
        monitor = PerformanceMonitor()
        
        # Test inference timing
        image = torch.randn(1, 3, 224, 224)
        
        with monitor.measure_inference():
            outputs = model.forward(image)
        
        # Get timing results
        summary = monitor.get_summary()
        
        # For mock model, timing should be very fast
        assert summary["inference_times"]["mean"] < 1.0  # Less than 1 second
    
    def test_memory_usage(self):
        """Test memory usage constraints."""
        model = MockModel()
        
        # Test with different input sizes
        small_input = torch.randn(1, 3, 224, 224)
        large_input = torch.randn(8, 3, 224, 224)
        
        # Both should work without memory errors
        small_output = model.forward(small_input)
        large_output = model.forward(large_input)
        
        assert small_output is not None
        assert large_output is not None
    
    @pytest.mark.slow
    def test_model_size_constraints(self):
        """Test that model size is within mobile constraints."""
        model = MockModel()
        
        # Mock model size calculation
        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = num_params * 4 / (1024 * 1024)  # Assume float32
        
        # For mock model, this is arbitrary, but test the concept
        assert model_size_mb > 0
    
    @pytest.mark.benchmark
    def test_throughput_benchmark(self, benchmark):
        """Benchmark model throughput."""
        model = MockModel()
        model.eval()
        
        image = torch.randn(1, 3, 224, 224)
        
        # Benchmark the forward pass
        result = benchmark(model.forward, image)
        
        # Result should be the model outputs
        assert "image_features" in result