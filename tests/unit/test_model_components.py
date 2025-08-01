"""Unit tests for model components."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from tests.fixtures import create_sample_model, ModelFixture


class TestVisionEncoder:
    """Test vision encoder component."""
    
    def test_vision_encoder_forward(self):
        """Test vision encoder forward pass."""
        model = create_sample_model()
        
        # Test with single image
        test_input = torch.randn(1, 3, 224, 224)
        output = model.vision_encoder(test_input)
        
        assert output.shape == (1, 768)  # Should output hidden_size features
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_vision_encoder_batch_processing(self):
        """Test vision encoder with different batch sizes."""
        model = create_sample_model()
        
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 3, 224, 224)
            output = model.vision_encoder(test_input)
            
            assert output.shape == (batch_size, 768)
            assert not torch.isnan(output).any()
    
    def test_vision_encoder_different_input_sizes(self):
        """Test vision encoder with different input resolutions."""
        model = create_sample_model()
        
        # Note: This model uses adaptive pooling, so it should handle different sizes
        input_sizes = [(224, 224), (256, 256), (112, 112)]
        
        for height, width in input_sizes:
            test_input = torch.randn(1, 3, height, width)
            output = model.vision_encoder(test_input)
            
            # Output should always be the same size due to adaptive pooling
            assert output.shape == (1, 768)


class TestTextEncoder:
    """Test text encoder component."""
    
    def test_text_encoder_forward(self):
        """Test text encoder forward pass."""
        model = create_sample_model()
        
        # Test with tokenized text
        vocab_size = 32000
        seq_length = 50
        test_input = torch.randint(0, vocab_size, (1, seq_length))
        
        output = model.text_encoder(test_input)
        
        assert output.shape == (1, seq_length, 768)
        assert not torch.isnan(output).any()
    
    def test_text_encoder_different_sequence_lengths(self):
        """Test text encoder with different sequence lengths."""
        model = create_sample_model()
        vocab_size = 32000
        
        seq_lengths = [10, 50, 128]
        for seq_length in seq_lengths:
            test_input = torch.randint(0, vocab_size, (1, seq_length))
            output = model.text_encoder(test_input)
            
            assert output.shape == (1, seq_length, 768)
    
    def test_text_encoder_batch_processing(self):
        """Test text encoder with different batch sizes."""
        model = create_sample_model()
        vocab_size = 32000
        seq_length = 50
        
        batch_sizes = [1, 4, 8]
        for batch_size in batch_sizes:
            test_input = torch.randint(0, vocab_size, (batch_size, seq_length))
            output = model.text_encoder(test_input)
            
            assert output.shape == (batch_size, seq_length, 768)


class TestMultiTaskHeads:
    """Test multi-task decoder heads."""
    
    def test_captioning_head(self):
        """Test captioning head output."""
        model = create_sample_model()
        
        # Test captioning task
        test_image = torch.randn(1, 3, 224, 224)
        output = model(test_image, task="captioning")
        
        assert output.shape == (1, 32000)  # vocab_size
        assert not torch.isnan(output).any()
    
    def test_ocr_head(self):
        """Test OCR head output."""
        model = create_sample_model()
        
        # Test OCR task
        test_image = torch.randn(1, 3, 224, 224)
        output = model(test_image, task="ocr")
        
        assert output.shape == (1, 32000)  # vocab_size
        assert not torch.isnan(output).any()
    
    def test_vqa_head(self):
        """Test VQA head output."""
        model = create_sample_model()
        
        # Test VQA task with text
        test_image = torch.randn(1, 3, 224, 224)
        test_text = torch.randint(0, 32000, (1, 20))
        output = model(test_image, text=test_text, task="vqa")
        
        assert output.shape == (1, 32000)  # vocab_size
        assert not torch.isnan(output).any()
    
    def test_retrieval_head(self):
        """Test retrieval head output."""
        model = create_sample_model()
        
        # Test retrieval task
        test_image = torch.randn(1, 3, 224, 224)
        output = model(test_image, task="retrieval")
        
        assert output.shape == (1, 768)  # hidden_size for embeddings
        assert not torch.isnan(output).any()
    
    def test_invalid_task(self):
        """Test handling of invalid task."""
        model = create_sample_model()
        
        test_image = torch.randn(1, 3, 224, 224)
        
        with pytest.raises(ValueError):
            model(test_image, task="invalid_task")


class TestModelUtilities:
    """Test model utility functions."""
    
    def test_generate_caption_interface(self):
        """Test generate_caption interface."""
        model = create_sample_model()
        
        # Test with numpy array
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        caption = model.generate_caption(test_image)
        
        assert isinstance(caption, str)
        assert len(caption) > 0
    
    def test_extract_text_interface(self):
        """Test extract_text interface."""
        model = create_sample_model()
        
        # Test with numpy array
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        ocr_results = model.extract_text(test_image)
        
        assert isinstance(ocr_results, list)
        assert len(ocr_results) > 0
        
        # Check structure of OCR results
        for result in ocr_results:
            assert 'text' in result
            assert 'bbox' in result
            assert isinstance(result['text'], str)
            assert isinstance(result['bbox'], list)
            assert len(result['bbox']) == 4  # [x1, y1, x2, y2]
    
    def test_answer_question_interface(self):
        """Test answer_question interface."""
        model = create_sample_model()
        
        # Test with numpy array and question
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        question = "What is in this image?"
        answer = model.answer_question(test_image, question)
        
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestModelSaving:
    """Test model saving and loading."""
    
    def test_save_model_checkpoint(self, temp_model_dir):
        """Test saving model checkpoint."""
        model = create_sample_model()
        checkpoint_path = temp_model_dir / "test_checkpoint.pth"
        
        # Save checkpoint
        ModelFixture.save_mock_checkpoint(model, checkpoint_path)
        
        # Verify file exists
        assert checkpoint_path.exists()
        
        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        assert 'model_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'loss' in checkpoint
        assert 'accuracy' in checkpoint
        assert 'config' in checkpoint
    
    def test_load_model_checkpoint(self, temp_model_dir):
        """Test loading model checkpoint."""
        model = create_sample_model()
        checkpoint_path = temp_model_dir / "test_checkpoint.pth"
        
        # Save checkpoint first
        ModelFixture.save_mock_checkpoint(model, checkpoint_path)
        
        # Create new model and load checkpoint
        new_model = create_sample_model()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Compare model parameters
        for param1, param2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(param1, param2, atol=1e-6)


class TestModelConfiguration:
    """Test model configuration and initialization."""
    
    def test_model_initialization_with_different_configs(self):
        """Test model initialization with various configurations."""
        configs = [
            {"hidden_size": 512, "vocab_size": 16000},
            {"hidden_size": 1024, "vocab_size": 50000},
            {"hidden_size": 256, "vocab_size": 8000},
        ]
        
        for config in configs:
            model = ModelFixture.create_mock_model(**config)
            
            # Test vision encoder output
            test_image = torch.randn(1, 3, 224, 224)
            vision_output = model.vision_encoder(test_image)
            assert vision_output.shape == (1, config["hidden_size"])
            
            # Test text encoder output
            test_text = torch.randint(0, config["vocab_size"], (1, 20))
            text_output = model.text_encoder(test_text)
            assert text_output.shape == (1, 20, config["hidden_size"])
    
    def test_model_parameter_count(self):
        """Test model parameter count is reasonable for mobile deployment."""
        model = create_sample_model()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should be reasonable for mobile (these are mock numbers)
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable initially
        
        # Calculate approximate model size (in MB, assuming fp32)
        model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        
        # Before quantization, should be reasonable size
        assert model_size_mb < 500  # Should be less than 500MB before quantization
    
    def test_model_device_compatibility(self):
        """Test model works on different devices."""
        model = create_sample_model()
        
        # Test CPU
        model.to('cpu')
        test_input = torch.randn(1, 3, 224, 224)
        output = model(test_input, task="captioning")
        assert output.device.type == 'cpu'
        
        # Only test GPU if available
        if torch.cuda.is_available():
            model.to('cuda')
            test_input = test_input.to('cuda')
            output = model(test_input, task="captioning")
            assert output.device.type == 'cuda'