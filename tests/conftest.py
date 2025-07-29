"""
Test configuration and fixtures for Mobile Multi-Modal LLM.
Provides shared test infrastructure and utilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np
from PIL import Image


# Test data fixtures
@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    return Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )


@pytest.fixture
def sample_batch_images(sample_image) -> list[Image.Image]:
    """Create a batch of sample images."""
    return [sample_image.copy() for _ in range(4)]


@pytest.fixture
def sample_text() -> str:
    """Sample text for multimodal testing."""
    return "A sample text for testing multimodal capabilities."


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Create temporary directory for model artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model_weights() -> Dict[str, torch.Tensor]:
    """Mock model weights for testing."""
    return {
        "encoder.weight": torch.randn(768, 3072),
        "decoder.weight": torch.randn(512, 768),
        "classifier.bias": torch.randn(1000),
    }


# Hardware and environment fixtures
@pytest.fixture
def mock_gpu_available(monkeypatch) -> None:
    """Mock GPU availability for testing."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


@pytest.fixture
def mock_mobile_sdk(monkeypatch) -> MagicMock:
    """Mock mobile SDK for testing mobile exports."""
    mock_sdk = MagicMock()
    mock_sdk.export_tflite.return_value = b"fake_tflite_model"
    mock_sdk.export_coreml.return_value = b"fake_coreml_model"
    return mock_sdk


# Configuration fixtures
@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Standard test configuration."""
    return {
        "model": {
            "architecture": "mobile_multimodal",
            "hidden_size": 768,
            "num_layers": 12,
            "vocab_size": 32000,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "max_epochs": 100,
        },
        "quantization": {
            "enabled": True,
            "precision": "int2",
            "calibration_samples": 1000,
        },
    }


# Performance testing fixtures
@pytest.fixture
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        "warmup_rounds": 10,
        "test_rounds": 100,
        "timeout_seconds": 30,
        "memory_limit_mb": 512,
    }


# Security testing fixtures
@pytest.fixture
def security_test_inputs() -> Dict[str, Any]:
    """Security test inputs including adversarial examples."""
    return {
        "clean_input": np.random.rand(224, 224, 3).astype(np.float32),
        "adversarial_input": np.random.rand(224, 224, 3).astype(np.float32),
        "malformed_input": np.array([]),
        "oversized_input": np.random.rand(2048, 2048, 3).astype(np.float32),
    }


# Test markers and skipif conditions
@pytest.fixture(autouse=True)
def skip_gpu_tests_if_unavailable(request):
    """Skip GPU tests if no GPU is available."""
    if request.node.get_closest_marker("gpu"):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")


@pytest.fixture(autouse=True)
def skip_mobile_tests_if_no_sdk(request):
    """Skip mobile tests if SDKs not available."""
    if request.node.get_closest_marker("mobile"):
        try:
            import tensorflow as tf  # noqa
            import coremltools as ct  # noqa
        except ImportError:
            pytest.skip("Mobile SDKs not available")


# Test data management
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_dataset(test_data_dir) -> Generator[Path, None, None]:
    """Create mock dataset for testing."""
    dataset_dir = test_data_dir / "mock_dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    # Create sample files
    (dataset_dir / "train.json").write_text('{"samples": []}')
    (dataset_dir / "val.json").write_text('{"samples": []}')
    
    yield dataset_dir
    
    # Cleanup
    import shutil
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)


# Logging and monitoring fixtures
@pytest.fixture
def capture_logs(caplog):
    """Capture logs during tests."""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


# Environment setup
def pytest_configure(config):
    """Configure pytest environment."""
    # Set test environment variables
    os.environ["TESTING"] = "1"
    os.environ["TORCH_HOME"] = "/tmp/torch_models"
    
    # Disable CUDA for testing by default
    if not os.environ.get("ENABLE_CUDA_TESTS"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to tests without other markers
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that might be slow
        if "benchmark" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)


# Custom pytest plugins
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", 
        action="store_true", 
        default=False, 
        help="Run slow tests"
    )
    parser.addoption(
        "--run-gpu", 
        action="store_true", 
        default=False, 
        help="Run GPU tests"
    )
    parser.addoption(
        "--run-mobile", 
        action="store_true", 
        default=False, 
        help="Run mobile SDK tests"
    )


def pytest_runtest_setup(item):
    """Setup for individual tests."""
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("Need --run-slow option to run")
    if "gpu" in item.keywords and not item.config.getoption("--run-gpu"):
        pytest.skip("Need --run-gpu option to run")
    if "mobile" in item.keywords and not item.config.getoption("--run-mobile"):
        pytest.skip("Need --run-mobile option to run")