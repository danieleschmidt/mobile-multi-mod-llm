"""Model fixtures for testing Mobile Multi-Modal LLM."""

import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class MockMobileMultiModalLLM(nn.Module):
    """Mock model for testing purposes."""
    
    def __init__(self, hidden_size: int = 768, vocab_size: int = 32000):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_size)
        )
        
        # Text encoder
        self.text_encoder = nn.Embedding(vocab_size, hidden_size)
        
        # Multi-task heads
        self.captioning_head = nn.Linear(hidden_size, vocab_size)
        self.ocr_head = nn.Linear(hidden_size, vocab_size)
        self.vqa_head = nn.Linear(hidden_size, vocab_size)
        self.retrieval_head = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, image: torch.Tensor, text: torch.Tensor = None, task: str = "captioning") -> torch.Tensor:
        """Forward pass with task-specific outputs."""
        # Process image
        image_features = self.vision_encoder(image)
        
        # Combine with text if provided
        if text is not None:
            text_features = self.text_encoder(text).mean(dim=1)
            features = image_features + text_features
        else:
            features = image_features
        
        # Task-specific head
        if task == "captioning":
            return self.captioning_head(features)
        elif task == "ocr":
            return self.ocr_head(features)
        elif task == "vqa":
            return self.vqa_head(features)
        elif task == "retrieval":
            return self.retrieval_head(features)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def generate_caption(self, image: np.ndarray) -> str:
        """Mock caption generation."""
        return f"A mock caption for image with shape {image.shape}"
    
    def extract_text(self, image: np.ndarray) -> list:
        """Mock text extraction."""
        return [{"text": "Mock extracted text", "bbox": [10, 10, 100, 50]}]
    
    def answer_question(self, image: np.ndarray, question: str) -> str:
        """Mock visual question answering."""
        return f"Mock answer to: {question}"


class ModelFixture:
    """Fixture class for model-related test utilities."""
    
    @staticmethod
    def create_mock_model(hidden_size: int = 768, vocab_size: int = 32000) -> MockMobileMultiModalLLM:
        """Create a mock model for testing."""
        return MockMobileMultiModalLLM(hidden_size, vocab_size)
    
    @staticmethod
    def create_sample_weights(hidden_size: int = 768, vocab_size: int = 32000) -> Dict[str, torch.Tensor]:
        """Create sample model weights."""
        return {
            "vision_encoder.0.weight": torch.randn(64, 3, 7, 7),
            "vision_encoder.0.bias": torch.randn(64),
            "vision_encoder.5.weight": torch.randn(768, 64 * 7 * 7),
            "vision_encoder.5.bias": torch.randn(768),
            "text_encoder.weight": torch.randn(vocab_size, hidden_size),
            "captioning_head.weight": torch.randn(vocab_size, hidden_size),
            "captioning_head.bias": torch.randn(vocab_size),
        }
    
    @staticmethod
    def save_mock_checkpoint(model: nn.Module, path: Path) -> None:
        """Save a mock model checkpoint."""
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "loss": 0.5,
            "accuracy": 0.85,
            "config": {
                "hidden_size": 768,
                "vocab_size": 32000,
                "architecture": "mobile_multimodal"
            }
        }, path)
    
    @staticmethod
    def create_quantized_model_mock() -> MagicMock:
        """Create a mock quantized model."""
        mock_model = MagicMock()
        mock_model.size = 35 * 1024 * 1024  # 35MB
        mock_model.precision = "int2"
        mock_model.accuracy_loss = 0.02  # 2% accuracy loss
        mock_model.inference_time_ms = 12
        return mock_model


def create_sample_model() -> MockMobileMultiModalLLM:
    """Factory function to create a sample model."""
    return ModelFixture.create_mock_model()


def create_model_checkpoint(temp_dir: Path) -> Tuple[Path, MockMobileMultiModalLLM]:
    """Create a temporary model checkpoint for testing."""
    model = create_sample_model()
    checkpoint_path = temp_dir / "test_checkpoint.pth"
    ModelFixture.save_mock_checkpoint(model, checkpoint_path)
    return checkpoint_path, model


def mock_model_training_step():
    """Mock a single training step."""
    return {
        "loss": np.random.uniform(0.1, 1.0),
        "accuracy": np.random.uniform(0.7, 0.95), 
        "learning_rate": 1e-4,
        "step_time": np.random.uniform(0.1, 0.5)
    }


def mock_model_evaluation():
    """Mock model evaluation results."""
    return {
        "test_loss": 0.3,
        "test_accuracy": 0.89,
        "captioning_cider": 94.7,
        "ocr_accuracy": 93.1,
        "vqa_score": 73.9,
        "retrieval_map": 89.2,
        "inference_time_ms": 12,
        "memory_usage_mb": 145
    }