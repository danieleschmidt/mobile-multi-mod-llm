"""
Test utilities and helper functions for Mobile Multi-Modal LLM tests.
"""

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image


class TestDataGenerator:
    """Generate synthetic test data for various testing scenarios."""
    
    @staticmethod
    def create_sample_image(
        size: Tuple[int, int] = (224, 224),
        channels: int = 3,
        dtype: str = "uint8",
        pattern: str = "random"
    ) -> np.ndarray:
        """Create a sample image with specified properties."""
        if pattern == "random":
            if dtype == "uint8":
                return np.random.randint(0, 256, (*size, channels), dtype=np.uint8)
            else:
                return np.random.rand(*size, channels).astype(dtype)
        elif pattern == "gradient":
            # Create gradient pattern
            h, w = size
            gradient = np.linspace(0, 255, w, dtype=np.uint8)
            image = np.tile(gradient, (h, 1))
            if channels == 3:
                return np.stack([image, image // 2, image // 4], axis=-1)
            return image[..., np.newaxis] if channels == 1 else image
        elif pattern == "checkerboard":
            # Create checkerboard pattern
            h, w = size
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            checkerboard = ((x // 16) + (y // 16)) % 2 * 255
            if channels == 3:
                return np.stack([checkerboard] * 3, axis=-1).astype(np.uint8)
            return checkerboard[..., np.newaxis].astype(np.uint8)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    @staticmethod
    def create_batch_images(
        batch_size: int,
        size: Tuple[int, int] = (224, 224),
        channels: int = 3
    ) -> np.ndarray:
        """Create a batch of sample images."""
        return np.stack([
            TestDataGenerator.create_sample_image(size, channels)
            for _ in range(batch_size)
        ])
    
    @staticmethod
    def create_text_samples(count: int = 10) -> List[str]:
        """Create sample text for testing."""
        templates = [
            "A photo of a {} in the {}",
            "This image shows a {} with {}",
            "You can see a {} that is {}",
            "The {} appears to be {} in this picture",
            "Here is a {} which looks {} today"
        ]
        
        objects = ["cat", "dog", "car", "house", "tree", "person", "bird", "flower"]
        adjectives = ["red", "blue", "large", "small", "beautiful", "old", "new", "bright"]
        locations = ["park", "garden", "street", "room", "field", "forest", "beach", "city"]
        
        samples = []
        for i in range(count):
            template = templates[i % len(templates)]
            if "{}" in template:
                obj = objects[i % len(objects)]
                if template.count("{}") == 2:
                    adj_or_loc = adjectives[i % len(adjectives)] if i % 2 == 0 else locations[i % len(locations)]
                    text = template.format(obj, adj_or_loc)
                else:
                    text = template.format(obj)
            else:
                text = template
            samples.append(text)
        
        return samples
    
    @staticmethod
    def create_vqa_samples(count: int = 10) -> List[Dict[str, Any]]:
        """Create VQA question-answer pairs."""
        questions = [
            "What color is the main object?",
            "How many objects are in the image?", 
            "What is the weather like?",
            "Is this indoors or outdoors?",
            "What time of day is it?",
            "What is the person doing?",
            "What animal is this?",
            "What type of vehicle is shown?",
            "What material is the object made of?",
            "Is the object moving or stationary?"
        ]
        
        answers = [
            ["red", "blue", "green", "yellow", "black", "white"],
            ["one", "two", "three", "four", "five", "many"],
            ["sunny", "cloudy", "rainy", "snowy", "foggy"],
            ["indoors", "outdoors"],
            ["morning", "afternoon", "evening", "night"],
            ["walking", "running", "sitting", "standing", "eating"],
            ["cat", "dog", "bird", "horse", "cow", "sheep"],
            ["car", "truck", "bus", "motorcycle", "bicycle"],
            ["metal", "wood", "plastic", "glass", "fabric"],
            ["moving", "stationary"]
        ]
        
        samples = []
        for i in range(count):
            question = questions[i % len(questions)]
            answer_set = answers[i % len(answers)]
            answer = answer_set[i % len(answer_set)]
            
            samples.append({
                "question": question,
                "answer": answer,
                "image_id": f"test_image_{i:03d}",
                "question_id": i
            })
        
        return samples


class MockModel:
    """Mock model for testing without actual model weights."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.training = True
        self.device = torch.device("cpu")
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            "hidden_size": 768,
            "num_layers": 12,
            "vocab_size": 32000,
            "image_size": 224,
            "num_tasks": 4
        }
    
    def forward(self, image: torch.Tensor, text: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Mock forward pass."""
        batch_size = image.shape[0]
        hidden_size = self.config["hidden_size"]
        vocab_size = self.config["vocab_size"]
        
        # Mock outputs for different tasks
        outputs = {
            "caption_logits": torch.randn(batch_size, 50, vocab_size),  # Caption generation
            "ocr_logits": torch.randn(batch_size, 100, vocab_size),     # OCR text
            "vqa_logits": torch.randn(batch_size, vocab_size),          # VQA answer
            "image_features": torch.randn(batch_size, hidden_size),     # Image embeddings
            "text_features": torch.randn(batch_size, hidden_size) if text is not None else None
        }
        
        return outputs
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.training = False
        return self
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        return self
    
    def parameters(self):
        """Mock parameters."""
        return [torch.randn(100, 100, requires_grad=True)]


class MockQuantizer:
    """Mock quantizer for testing quantization workflows."""
    
    def __init__(self, precision: str = "int2"):
        self.precision = precision
        self.calibrated = False
    
    def calibrate(self, dataloader):
        """Mock calibration process."""
        time.sleep(0.1)  # Simulate calibration time
        self.calibrated = True
    
    def quantize(self, model) -> "MockQuantizer":
        """Mock quantization process."""
        if not self.calibrated:
            raise RuntimeError("Model must be calibrated before quantization")
        
        # Return a mock quantized model
        quantized_model = MockModel(model.config)
        quantized_model._is_quantized = True
        quantized_model._precision = self.precision
        return quantized_model


class HardwareSimulator:
    """Simulate different hardware environments for testing."""
    
    def __init__(self, device_profile: str = "snapdragon_8gen3"):
        self.device_profile = device_profile
        self.profiles = {
            "snapdragon_8gen3": {
                "memory_mb": 8192,
                "compute_gflops": 100,
                "has_npu": True,
                "inference_time_ms": 12
            },
            "apple_a17_pro": {
                "memory_mb": 8192,
                "compute_gflops": 120,
                "has_neural_engine": True,
                "inference_time_ms": 8
            },
            "generic_mobile": {
                "memory_mb": 4096,
                "compute_gflops": 50,
                "has_npu": False,
                "inference_time_ms": 25
            },
            "low_end_device": {
                "memory_mb": 2048,
                "compute_gflops": 20,
                "has_npu": False,
                "inference_time_ms": 100
            }
        }
    
    def get_profile(self) -> Dict[str, Any]:
        """Get current device profile."""
        return self.profiles.get(self.device_profile, self.profiles["generic_mobile"])
    
    def simulate_inference(self, model, inputs) -> Tuple[Any, float]:
        """Simulate inference with realistic timing."""
        profile = self.get_profile()
        base_time = profile["inference_time_ms"] / 1000.0
        
        # Add some noise to timing
        actual_time = base_time * (0.8 + 0.4 * np.random.random())
        
        # Simulate memory constraints
        if hasattr(inputs, 'shape'):
            input_size_mb = np.prod(inputs.shape) * 4 / (1024 * 1024)  # Assume float32
            if input_size_mb > profile["memory_mb"] * 0.8:
                raise RuntimeError(f"Input too large for device memory: {input_size_mb}MB > {profile['memory_mb']}MB")
        
        # Mock the actual inference
        time.sleep(actual_time)
        return model.forward(inputs), actual_time


class PerformanceMonitor:
    """Monitor performance metrics during testing."""
    
    def __init__(self):
        self.metrics = {
            "inference_times": [],
            "memory_usage": [],
            "accuracy_scores": [],
            "model_sizes": []
        }
    
    @contextmanager
    def measure_inference(self):
        """Context manager to measure inference time."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self.metrics["inference_times"].append(end_time - start_time)
    
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage."""
        self.metrics["memory_usage"].append(memory_mb)
    
    def record_accuracy(self, accuracy: float):
        """Record accuracy score."""
        self.metrics["accuracy_scores"].append(accuracy)
    
    def record_model_size(self, size_mb: float):
        """Record model size."""
        self.metrics["model_sizes"].append(size_mb)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
            else:
                summary[metric_name] = None
        return summary


def create_temp_model_file(content: bytes, suffix: str = ".pth") -> str:
    """Create a temporary model file with given content."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(content)
    except:
        os.close(fd)
        raise
    return path


def assert_model_output_shape(output: torch.Tensor, expected_shape: Tuple[int, ...]):
    """Assert that model output has expected shape."""
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


def assert_performance_within_bounds(
    inference_time: float,
    max_time: float,
    model_size: float,
    max_size: float
):
    """Assert that performance metrics are within acceptable bounds."""
    assert inference_time <= max_time, f"Inference time {inference_time:.3f}s exceeds limit {max_time}s"
    assert model_size <= max_size, f"Model size {model_size:.1f}MB exceeds limit {max_size}MB"


def compare_model_outputs(output1: torch.Tensor, output2: torch.Tensor, tolerance: float = 1e-3):
    """Compare two model outputs with tolerance."""
    diff = torch.abs(output1 - output2).max().item()
    assert diff <= tolerance, f"Model outputs differ by {diff:.6f}, exceeding tolerance {tolerance}"


def load_test_config(config_name: str) -> Dict[str, Any]:
    """Load test configuration from fixtures."""
    config_path = Path(__file__).parent.parent / "fixtures" / "configs" / f"{config_name}.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        # Return default config if file doesn't exist
        return {
            "model": {"hidden_size": 768, "num_layers": 12},
            "training": {"batch_size": 32, "learning_rate": 1e-4}
        }


@contextmanager
def mock_hardware_environment(device_profile: str):
    """Context manager to mock specific hardware environment."""
    simulator = HardwareSimulator(device_profile)
    profile = simulator.get_profile()
    
    with patch('torch.cuda.is_available', return_value=profile.get('has_gpu', False)), \
         patch('torch.cuda.get_device_properties') as mock_props:
        
        # Mock GPU properties if available
        if profile.get('has_gpu', False):
            mock_props.return_value.total_memory = profile['memory_mb'] * 1024 * 1024
        
        yield simulator


def skip_if_no_gpu():
    """Decorator to skip test if no GPU is available."""
    import pytest
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")


def skip_if_no_mobile_sdk():
    """Decorator to skip test if mobile SDKs are not available."""
    import pytest
    try:
        import tensorflow as tf  # noqa
        import coremltools as ct  # noqa
        return lambda func: func
    except ImportError:
        return pytest.mark.skip(reason="Mobile SDKs not available")


class TestMetrics:
    """Collect and validate test metrics."""
    
    def __init__(self):
        self.data = {}
    
    def record(self, key: str, value: Union[float, int, str]):
        """Record a metric value."""
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
    
    def assert_within_range(self, key: str, min_val: float, max_val: float):
        """Assert all values for a key are within range."""
        if key not in self.data:
            raise ValueError(f"No data recorded for key: {key}")
        
        values = self.data[key]
        for value in values:
            assert min_val <= value <= max_val, \
                f"{key} value {value} not in range [{min_val}, {max_val}]"
    
    def get_average(self, key: str) -> float:
        """Get average value for a key."""
        if key not in self.data:
            raise ValueError(f"No data recorded for key: {key}")
        
        return sum(self.data[key]) / len(self.data[key])