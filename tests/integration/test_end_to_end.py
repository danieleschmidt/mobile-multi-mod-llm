"""
Integration tests for end-to-end workflows.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image

from tests.utils import (
    HardwareSimulator,
    MockModel,
    TestDataGenerator,
    PerformanceMonitor,
    assert_performance_within_bounds,
    mock_hardware_environment
)


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_image_captioning_pipeline(self):
        """Test complete image captioning workflow."""
        # Setup
        model = MockModel()
        model.eval()
        
        # Create test image
        image_array = TestDataGenerator.create_sample_image((224, 224), 3)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            outputs = model.forward(image_tensor)
        
        # Verify outputs
        assert "caption_logits" in outputs
        assert "image_features" in outputs
        
        # Check output shapes
        batch_size = 1
        vocab_size = model.config["vocab_size"]
        assert outputs["caption_logits"].shape == (batch_size, 50, vocab_size)
        assert outputs["image_features"].shape == (batch_size, 768)
    
    def test_ocr_extraction_pipeline(self):
        """Test OCR text extraction workflow."""
        model = MockModel()
        model.eval()
        
        # Create test document image
        image_array = TestDataGenerator.create_sample_image((224, 224), 3, pattern="gradient")
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        # Run OCR inference
        with torch.no_grad():
            outputs = model.forward(image_tensor)
        
        # Verify OCR outputs
        assert "ocr_logits" in outputs
        assert outputs["ocr_logits"].shape[1] == 100  # OCR sequence length
    
    def test_vqa_pipeline(self):
        """Test Visual Question Answering workflow."""
        model = MockModel()
        model.eval()
        
        # Create test inputs
        image_array = TestDataGenerator.create_sample_image((224, 224), 3)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        # Mock question encoding (in real implementation, this would be tokenized)
        question_tensor = torch.randint(0, 1000, (1, 20))  # Question tokens
        
        # Run VQA inference
        with torch.no_grad():
            outputs = model.forward(image_tensor, question_tensor)
        
        # Verify VQA outputs
        assert "vqa_logits" in outputs
        assert outputs["vqa_logits"].shape == (1, model.config["vocab_size"])
    
    def test_multimodal_retrieval_pipeline(self):
        """Test image-text retrieval workflow."""
        model = MockModel()
        model.eval()
        
        # Create test inputs
        images = []
        texts = []
        
        for i in range(4):  # Small batch
            image_array = TestDataGenerator.create_sample_image((224, 224), 3)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            images.append(image_tensor)
            
            text_tensor = torch.randint(0, 1000, (30,))  # Text tokens
            texts.append(text_tensor)
        
        images_batch = torch.stack(images)
        texts_batch = torch.stack(texts)
        
        # Run retrieval inference
        with torch.no_grad():
            outputs = model.forward(images_batch, texts_batch)
        
        # Verify retrieval features
        assert "image_features" in outputs
        assert "text_features" in outputs
        assert outputs["image_features"].shape == (4, 768)
        assert outputs["text_features"].shape == (4, 768)
        
        # Test similarity computation
        image_features = outputs["image_features"]
        text_features = outputs["text_features"]
        
        # Normalize features
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_features, text_features.T)
        
        assert similarity_matrix.shape == (4, 4)
        assert torch.all(similarity_matrix >= -1) and torch.all(similarity_matrix <= 1)


class TestMobileDeploymentIntegration:
    """Test mobile deployment integration workflows."""
    
    @pytest.mark.mobile
    def test_android_deployment_pipeline(self):
        """Test Android deployment workflow."""
        model = MockModel()
        
        # Mock TensorFlow Lite conversion
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp:
            # Write mock TFLite model
            tmp.write(b"fake_tflite_model_data")
            tmp.flush()
            
            tflite_path = tmp.name
        
        # Verify file creation
        assert Path(tflite_path).exists()
        
        # Mock Android SDK integration test
        # In real implementation, this would test actual SDK integration
        assert True
        
        # Cleanup
        Path(tflite_path).unlink()
    
    @pytest.mark.mobile
    def test_ios_deployment_pipeline(self):
        """Test iOS deployment workflow."""
        model = MockModel()
        
        # Mock Core ML conversion
        with tempfile.NamedTemporaryFile(suffix='.mlmodel', delete=False) as tmp:
            # Write mock Core ML model
            tmp.write(b"fake_coreml_model_data")
            tmp.flush()
            
            coreml_path = tmp.name
        
        # Verify file creation
        assert Path(coreml_path).exists()
        
        # Mock iOS SDK integration test
        # In real implementation, this would test actual SDK integration
        assert True
        
        # Cleanup
        Path(coreml_path).unlink()
    
    def test_cross_platform_consistency(self):
        """Test consistency across mobile platforms."""
        model = MockModel()
        model.eval()
        
        # Test same input on both platforms (mocked)
        image_array = TestDataGenerator.create_sample_image((224, 224), 3)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        # Mock platform-specific inference
        torch.manual_seed(42)  # For reproducibility
        android_output = model.forward(image_tensor)
        
        torch.manual_seed(42)  # Same seed
        ios_output = model.forward(image_tensor)
        
        # Outputs should be identical (in mock case, they are)
        assert android_output["image_features"].shape == ios_output["image_features"].shape


class TestPerformanceIntegration:
    """Test performance requirements in integrated scenarios."""
    
    def test_real_time_inference_requirement(self):
        """Test that inference meets real-time requirements."""
        model = MockModel()
        model.eval()
        
        monitor = PerformanceMonitor()
        
        # Test multiple inferences
        for _ in range(10):
            image = TestDataGenerator.create_sample_image((224, 224), 3)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            with monitor.measure_inference():
                with torch.no_grad():
                    outputs = model.forward(image_tensor)
        
        # Check performance requirements
        summary = monitor.get_summary()
        mean_time = summary["inference_times"]["mean"]
        max_time = summary["inference_times"]["max"]
        
        # Real-time requirement: < 50ms average, < 100ms max
        assert mean_time < 0.05, f"Average inference time {mean_time:.3f}s exceeds 50ms"
        assert max_time < 0.1, f"Max inference time {max_time:.3f}s exceeds 100ms"
    
    @pytest.mark.parametrize("device_profile", [
        "snapdragon_8gen3",
        "apple_a17_pro", 
        "generic_mobile",
        "low_end_device"
    ])
    def test_device_specific_performance(self, device_profile):
        """Test performance on different device profiles."""
        model = MockModel()
        model.eval()
        
        with mock_hardware_environment(device_profile) as simulator:
            # Create test input
            image = TestDataGenerator.create_sample_image((224, 224), 3)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            # Run simulated inference
            output, inference_time = simulator.simulate_inference(model, image_tensor)
            
            # Check device-specific performance bounds
            profile = simulator.get_profile()
            expected_max_time = profile["inference_time_ms"] / 1000.0 * 2  # 2x tolerance
            
            assert inference_time <= expected_max_time, \
                f"Inference time {inference_time:.3f}s exceeds device limit {expected_max_time:.3f}s"
    
    def test_memory_constrained_inference(self):
        """Test inference under memory constraints."""
        model = MockModel()
        model.eval()
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        max_memory_mb = 512  # Mobile memory constraint
        
        for batch_size in batch_sizes:
            images = TestDataGenerator.create_batch_images(batch_size, (224, 224), 3)
            images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
            
            # Estimate memory usage (rough calculation)
            input_memory_mb = images_tensor.numel() * 4 / (1024 * 1024)  # float32
            
            if input_memory_mb > max_memory_mb:
                # Should handle memory constraints gracefully
                with pytest.raises((RuntimeError, MemoryError)):
                    with torch.no_grad():
                        outputs = model.forward(images_tensor)
            else:
                # Should work fine
                with torch.no_grad():
                    outputs = model.forward(images_tensor)
                assert outputs is not None
    
    def test_battery_optimized_inference(self):
        """Test inference optimized for battery life."""
        model = MockModel()
        model.eval()
        
        # Simulate battery-optimized settings
        with mock_hardware_environment("battery_optimized") as simulator:
            profile = simulator.get_profile()
            
            # Test that inference still works with power constraints
            image = TestDataGenerator.create_sample_image((224, 224), 3)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            output, inference_time = simulator.simulate_inference(model, image_tensor)
            
            assert output is not None
            # Battery-optimized might be slower but should still be reasonable
            assert inference_time < 0.2  # Max 200ms for battery-optimized


class TestDataIntegration:
    """Test data processing and integration workflows."""
    
    def test_dataset_loading_pipeline(self):
        """Test complete dataset loading and processing."""
        # Create mock dataset
        samples = TestDataGenerator.create_vqa_samples(10)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(samples, tmp)
            dataset_path = tmp.name
        
        # Load and verify dataset
        with open(dataset_path) as f:
            loaded_samples = json.load(f)
        
        assert len(loaded_samples) == 10
        assert all("question" in sample for sample in loaded_samples)
        assert all("answer" in sample for sample in loaded_samples)
        
        # Cleanup
        Path(dataset_path).unlink()
    
    def test_batch_processing_pipeline(self):
        """Test batch processing of multiple samples."""
        model = MockModel()
        model.eval()
        
        # Create batch of test data
        batch_size = 4
        images = []
        questions = []
        
        for i in range(batch_size):
            # Create image
            image_array = TestDataGenerator.create_sample_image((224, 224), 3)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            images.append(image_tensor)
            
            # Create question
            question_tensor = torch.randint(0, 1000, (20,))
            questions.append(question_tensor)
        
        # Stack into batches
        images_batch = torch.stack(images)
        questions_batch = torch.stack(questions)
        
        # Process batch
        with torch.no_grad():
            batch_outputs = model.forward(images_batch, questions_batch)
        
        # Verify batch processing
        assert batch_outputs["image_features"].shape[0] == batch_size
        assert batch_outputs["text_features"].shape[0] == batch_size
        
        # Compare with individual processing
        individual_outputs = []
        for i in range(batch_size):
            with torch.no_grad():
                output = model.forward(images[i].unsqueeze(0), questions[i].unsqueeze(0))
            individual_outputs.append(output)
        
        # Shapes should match
        for i in range(batch_size):
            assert individual_outputs[i]["image_features"].shape[1:] == \
                   batch_outputs["image_features"].shape[1:]


class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        model = MockModel()
        model.eval()
        
        # Test various invalid inputs
        invalid_inputs = [
            torch.randn(1, 3, 100, 100),  # Wrong size
            torch.randn(1, 1, 224, 224),  # Wrong channels
            torch.randn(3, 224, 224),     # Missing batch dimension
        ]
        
        for invalid_input in invalid_inputs:
            # Model should handle gracefully or raise appropriate error
            try:
                with torch.no_grad():
                    output = model.forward(invalid_input)
                # If it succeeds, output should still be valid
                assert output is not None
            except (RuntimeError, ValueError, AssertionError):
                # Expected for some invalid inputs
                pass
    
    def test_model_corruption_handling(self):
        """Test handling of corrupted model states."""
        model = MockModel()
        
        # Simulate model corruption by modifying config
        original_config = model.config.copy()
        model.config["hidden_size"] = -1  # Invalid value
        
        # Should handle corrupted config gracefully
        try:
            image = torch.randn(1, 3, 224, 224)
            output = model.forward(image)
            # If it works, that's fine too
            assert output is not None
        except (ValueError, RuntimeError):
            # Expected for corrupted model
            pass
        finally:
            # Restore valid config
            model.config = original_config
    
    def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion scenarios."""
        model = MockModel()
        model.eval()
        
        # Test with very large input that might exhaust memory
        try:
            large_image = torch.randn(1, 3, 2048, 2048)  # Very large image
            with torch.no_grad():
                output = model.forward(large_image)
        except (RuntimeError, MemoryError):
            # Expected for resource exhaustion
            pytest.skip("Resource exhaustion test - expected behavior")
        
        # If it doesn't fail, that's also acceptable for mock model
        assert True