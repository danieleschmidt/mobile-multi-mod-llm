"""
Test configuration and constants for Mobile Multi-Modal LLM tests.
"""

import os
from pathlib import Path

# Test environment configuration
TEST_ENV = {
    "TESTING": True,
    "LOG_LEVEL": "DEBUG",
    "DISABLE_WANDB": True,
    "DISABLE_MLFLOW": True,
    "TORCH_HOME": "/tmp/torch_models",
    "TRANSFORMERS_CACHE": "/tmp/transformers_cache",
}

# Performance test thresholds
PERFORMANCE_THRESHOLDS = {
    "max_inference_time_ms": 50,
    "max_model_size_mb": 35,
    "min_accuracy_threshold": 0.90,
    "max_memory_usage_mb": 512,
    "max_startup_time_ms": 1000,
}

# Device profiles for testing
DEVICE_PROFILES = {
    "flagship": {
        "name": "Flagship Device (Snapdragon 8 Gen 3)",
        "memory_mb": 8192,
        "compute_gflops": 100,
        "has_npu": True,
        "max_inference_time_ms": 12,
        "target_fps": 60,
    },
    "premium": {
        "name": "Premium Device (Apple A17 Pro)",
        "memory_mb": 8192,
        "compute_gflops": 120,
        "has_neural_engine": True,
        "max_inference_time_ms": 8,
        "target_fps": 60,
    },
    "mid_range": {
        "name": "Mid-range Device",
        "memory_mb": 6144,
        "compute_gflops": 75,
        "has_npu": True,
        "max_inference_time_ms": 25,
        "target_fps": 30,
    },
    "budget": {
        "name": "Budget Device",
        "memory_mb": 4096,
        "compute_gflops": 40,
        "has_npu": False,
        "max_inference_time_ms": 50,
        "target_fps": 15,
    },
    "low_end": {
        "name": "Low-end Device",
        "memory_mb": 2048,
        "compute_gflops": 20,
        "has_npu": False,
        "max_inference_time_ms": 100,
        "target_fps": 10,
    },
}

# Test data configuration
TEST_DATA_CONFIG = {
    "image_sizes": [(224, 224), (256, 256), (384, 384)],
    "batch_sizes": [1, 2, 4, 8, 16],
    "sequence_lengths": [10, 20, 50, 100],
    "num_samples": {
        "unit": 10,
        "integration": 100,
        "performance": 1000,
        "stress": 10000,
    },
}

# Model configuration for testing
TEST_MODEL_CONFIG = {
    "hidden_size": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "vocab_size": 32000,
    "image_size": 224,
    "patch_size": 16,
    "num_tasks": 4,
    "quantization": {
        "enabled": True,
        "precision": "int2",
        "calibration_samples": 1000,
    },
}

# Benchmark configuration
BENCHMARK_CONFIG = {
    "warmup_rounds": 10,
    "measurement_rounds": 100,
    "timeout_seconds": 30,
    "memory_limit_mb": 512,
    "cpu_limit_percent": 80,
    "statistical_confidence": 0.95,
}

# Security test configuration
SECURITY_TEST_CONFIG = {
    "adversarial_epsilons": [0.01, 0.05, 0.1, 0.2],
    "attack_methods": ["fgsm", "pgd", "c&w"],
    "robustness_thresholds": {
        "min_clean_accuracy": 0.90,
        "max_accuracy_drop": 0.10,
        "max_perturbation_norm": 0.1,
    },
}

# Chaos engineering configuration
CHAOS_CONFIG = {
    "failure_scenarios": [
        "memory_pressure",
        "thermal_throttling",
        "network_interruption",
        "storage_full",
        "process_killed",
        "hardware_fault",
    ],
    "failure_probabilities": {
        "low": 0.01,
        "medium": 0.05,
        "high": 0.10,
    },
    "recovery_timeouts": {
        "fast": 1.0,
        "normal": 5.0,
        "slow": 30.0,
    },
}

# Test markers and their descriptions
TEST_MARKERS = {
    "unit": "Unit tests - fast, isolated tests",
    "integration": "Integration tests - test component interactions",
    "e2e": "End-to-end tests - full workflow tests",
    "performance": "Performance tests - measure speed and resources",
    "benchmark": "Benchmark tests - detailed performance measurement",
    "security": "Security tests - vulnerability and attack testing",
    "chaos": "Chaos engineering tests - failure resilience",
    "mobile": "Mobile-specific tests - require mobile SDKs",
    "gpu": "GPU tests - require CUDA availability",
    "slow": "Slow tests - take significant time to run",
    "network": "Network tests - require internet connectivity",
    "hardware": "Hardware tests - require specific hardware",
}

# Test file patterns
TEST_PATTERNS = {
    "unit": "tests/unit/test_*.py",
    "integration": "tests/integration/test_*.py",
    "e2e": "tests/e2e/test_*.py",
    "performance": "tests/performance/test_*.py",
    "benchmark": "tests/benchmarks/test_*.py",
    "security": "tests/security/test_*.py",
    "chaos": "tests/chaos/test_*.py",
}

# CI/CD test configuration
CI_CONFIG = {
    "fast_test_timeout": 300,  # 5 minutes
    "full_test_timeout": 3600,  # 1 hour
    "parallel_workers": 4,
    "retry_attempts": 3,
    "artifact_retention_days": 30,
    "coverage_threshold": 80,
    "flake_threshold": 5,  # Max flaky test failures
}

# Mock data paths
MOCK_DATA_PATHS = {
    "images": Path(__file__).parent / "fixtures" / "images",
    "text": Path(__file__).parent / "fixtures" / "text",
    "models": Path(__file__).parent / "fixtures" / "models",
    "datasets": Path(__file__).parent / "fixtures" / "datasets",
    "configs": Path(__file__).parent / "fixtures" / "configs",
}

# Quality gates
QUALITY_GATES = {
    "test_coverage": {
        "minimum": 80,
        "target": 90,
        "branches": 75,
    },
    "performance": {
        "max_regression": 0.05,  # 5% performance regression
        "memory_limit": 1.2,  # 20% memory increase
    },
    "security": {
        "max_vulnerabilities": 0,
        "min_security_score": 8.0,
    },
    "maintainability": {
        "max_complexity": 10,
        "min_documentation": 80,
    },
}

# Test report configuration
REPORT_CONFIG = {
    "html_report": True,
    "xml_report": True,
    "json_report": True,
    "junit_xml": True,
    "coverage_formats": ["html", "xml", "term"],
    "performance_formats": ["json", "html"],
    "security_formats": ["json", "sarif"],
}

# Environment-specific overrides
def get_test_config():
    """Get test configuration with environment-specific overrides."""
    config = {
        "env": TEST_ENV,
        "performance": PERFORMANCE_THRESHOLDS,
        "devices": DEVICE_PROFILES,
        "data": TEST_DATA_CONFIG,
        "model": TEST_MODEL_CONFIG,
        "benchmark": BENCHMARK_CONFIG,
        "security": SECURITY_TEST_CONFIG,
        "chaos": CHAOS_CONFIG,
        "markers": TEST_MARKERS,
        "patterns": TEST_PATTERNS,
        "ci": CI_CONFIG,
        "mock_paths": MOCK_DATA_PATHS,
        "quality": QUALITY_GATES,
        "reporting": REPORT_CONFIG,
    }
    
    # Apply environment-specific overrides
    if os.getenv("CI"):
        # CI environment adjustments
        config["benchmark"]["measurement_rounds"] = 50  # Faster in CI
        config["data"]["num_samples"]["performance"] = 100  # Smaller datasets
        config["chaos"]["failure_probabilities"]["high"] = 0.05  # Less chaos in CI
    
    if os.getenv("GITHUB_ACTIONS"):
        # GitHub Actions specific adjustments
        config["ci"]["parallel_workers"] = 2  # Limited resources
        config["performance"]["max_inference_time_ms"] = 100  # More lenient in CI
    
    if os.getenv("TESTING_MODE") == "fast":
        # Fast testing mode
        config["benchmark"]["measurement_rounds"] = 10
        config["data"]["num_samples"] = {k: min(v, 10) for k, v in config["data"]["num_samples"].items()}
    
    return config


# Utility functions
def is_gpu_available():
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_mobile_sdk_available():
    """Check if mobile SDKs are available for testing."""
    try:
        import tensorflow as tf  # noqa
        import coremltools as ct  # noqa
        return True
    except ImportError:
        return False


def get_test_device():
    """Get appropriate test device based on environment."""
    if is_gpu_available():
        return "cuda"
    else:
        return "cpu"


def should_run_slow_tests():
    """Determine if slow tests should be run."""
    return os.getenv("RUN_SLOW_TESTS", "false").lower() == "true"


def should_run_hardware_tests():
    """Determine if hardware-specific tests should be run."""
    return os.getenv("RUN_HARDWARE_TESTS", "false").lower() == "true"


def get_test_data_size():
    """Get test data size based on environment."""
    size = os.getenv("TEST_DATA_SIZE", "small").lower()
    if size not in ["small", "medium", "large"]:
        size = "small"
    return size