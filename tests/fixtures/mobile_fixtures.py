"""Mobile platform fixtures for testing Mobile Multi-Modal LLM."""

import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch
import numpy as np


class MobileFixture:
    """Fixture class for mobile platform testing utilities."""
    
    @staticmethod
    def mock_tflite_interpreter():
        """Mock TensorFlow Lite interpreter."""
        mock_interpreter = MagicMock()
        
        # Mock input/output details
        mock_interpreter.get_input_details.return_value = [{
            'name': 'input_image',
            'index': 0,
            'shape': [1, 224, 224, 3],
            'dtype': np.float32
        }]
        
        mock_interpreter.get_output_details.return_value = [{
            'name': 'output_logits',
            'index': 0,
            'shape': [1, 1000],
            'dtype': np.float32
        }]
        
        # Mock inference
        def mock_invoke():
            # Simulate processing time
            import time
            time.sleep(0.01)  # 10ms inference time
        
        mock_interpreter.invoke = mock_invoke
        mock_interpreter.get_tensor.return_value = np.random.randn(1, 1000).astype(np.float32)
        
        return mock_interpreter
    
    @staticmethod
    def mock_coreml_model():
        """Mock Core ML model."""
        mock_model = MagicMock()
        
        # Mock model specification
        mock_model.spec.description.input = [
            MagicMock(name='image', type=MagicMock(imageType=MagicMock(width=224, height=224)))
        ]
        mock_model.spec.description.output = [
            MagicMock(name='logits', type=MagicMock(multiArrayType=MagicMock(shape=[1, 1000])))
        ]
        
        # Mock prediction
        def mock_predict(inputs):
            import time
            time.sleep(0.008)  # 8ms inference time
            return {'logits': np.random.randn(1, 1000).astype(np.float32)}
        
        mock_model.predict = mock_predict
        return mock_model
    
    @staticmethod
    def mock_hexagon_runtime():
        """Mock Qualcomm Hexagon NPU runtime."""
        mock_runtime = MagicMock()
        
        # Mock model loading
        mock_runtime.load_model.return_value = True
        mock_runtime.is_model_loaded.return_value = True
        
        # Mock inference
        def mock_execute(input_data):
            # Simulate INT2 quantized inference
            import time
            time.sleep(0.005)  # 5ms inference time on NPU
            return np.random.randint(0, 4, (1, 1000), dtype=np.int8)  # INT2 output
        
        mock_runtime.execute = mock_execute
        mock_runtime.get_performance_stats.return_value = {
            'inference_time_ms': 5,
            'memory_usage_mb': 35,
            'power_consumption_mw': 120
        }
        
        return mock_runtime
    
    @staticmethod
    def create_mobile_test_config() -> Dict[str, Any]:
        """Create mobile testing configuration."""
        return {
            'android': {
                'min_api_level': 24,
                'target_api_level': 34,
                'arch': ['arm64-v8a', 'armeabi-v7a'],
                'tflite_version': '2.15.0',
                'hexagon_sdk_version': '5.5.0'
            },
            'ios': {
                'min_deployment_target': '14.0',
                'target_deployment_target': '17.0',
                'arch': ['arm64'],
                'coreml_version': '7.0',
                'neural_engine': True
            },
            'performance_targets': {
                'inference_time_ms': {
                    'android_flagship': 12,
                    'android_midrange': 25,
                    'ios_flagship': 8,
                    'ios_midrange': 15
                },
                'memory_usage_mb': {
                    'max_runtime': 150,
                    'model_size': 35
                },
                'battery_consumption': {
                    'max_per_inference_mah': 0.1,
                    'thermal_throttling_temp': 42
                }
            }
        }
    
    @staticmethod
    def mock_device_profiles() -> List[Dict[str, Any]]:
        """Mock mobile device profiles for testing."""
        return [
            # Android devices
            {
                'name': 'Pixel 8 Pro',
                'platform': 'android',
                'chipset': 'Tensor G3',
                'npu': 'Google TPU',
                'ram_gb': 12,
                'api_level': 34,
                'expected_performance': {
                    'inference_time_ms': 8,
                    'memory_usage_mb': 140
                }
            },
            {
                'name': 'Galaxy S24 Ultra',
                'platform': 'android', 
                'chipset': 'Snapdragon 8 Gen 3',
                'npu': 'Hexagon NPU',
                'ram_gb': 12,
                'api_level': 34,
                'expected_performance': {
                    'inference_time_ms': 5,
                    'memory_usage_mb': 135
                }
            },
            {
                'name': 'OnePlus 11',
                'platform': 'android',
                'chipset': 'Snapdragon 8 Gen 2', 
                'npu': 'Hexagon NPU',
                'ram_gb': 8,
                'api_level': 33,
                'expected_performance': {
                    'inference_time_ms': 12,
                    'memory_usage_mb': 145
                }
            },
            # iOS devices
            {
                'name': 'iPhone 15 Pro Max',
                'platform': 'ios',
                'chipset': 'A17 Pro',
                'npu': 'Neural Engine',
                'ram_gb': 8,
                'ios_version': '17.0',
                'expected_performance': {
                    'inference_time_ms': 6,
                    'memory_usage_mb': 130
                }
            },
            {
                'name': 'iPhone 14',
                'platform': 'ios',
                'chipset': 'A15 Bionic',
                'npu': 'Neural Engine',
                'ram_gb': 6,
                'ios_version': '16.0',
                'expected_performance': {
                    'inference_time_ms': 15,
                    'memory_usage_mb': 140
                }
            },
            {
                'name': 'iPad Pro M2',
                'platform': 'ios',
                'chipset': 'M2',
                'npu': 'Neural Engine',
                'ram_gb': 16,
                'ios_version': '16.0',
                'expected_performance': {
                    'inference_time_ms': 4,
                    'memory_usage_mb': 120
                }
            }
        ]
    
    @staticmethod
    def mock_quantization_results() -> Dict[str, Any]:
        """Mock quantization analysis results."""
        return {
            'original_model': {
                'size_mb': 140.2,
                'precision': 'fp32',
                'accuracy': {
                    'captioning_cider': 96.8,
                    'ocr_accuracy': 94.5,
                    'vqa_score': 75.2,
                    'retrieval_map': 91.1
                }
            },
            'quantized_models': {
                'int8': {
                    'size_mb': 35.1,
                    'precision': 'int8',
                    'accuracy': {
                        'captioning_cider': 95.9,
                        'ocr_accuracy': 93.8,
                        'vqa_score': 74.1,
                        'retrieval_map': 89.8
                    },
                    'accuracy_loss': {
                        'captioning_cider': 0.9,
                        'ocr_accuracy': 0.7,
                        'vqa_score': 1.1,
                        'retrieval_map': 1.3
                    }
                },
                'int4': {
                    'size_mb': 17.6,
                    'precision': 'int4',
                    'accuracy': {
                        'captioning_cider': 94.2,
                        'ocr_accuracy': 92.1,
                        'vqa_score': 72.8,
                        'retrieval_map': 87.5
                    },
                    'accuracy_loss': {
                        'captioning_cider': 2.6,
                        'ocr_accuracy': 2.4,
                        'vqa_score': 2.4,
                        'retrieval_map': 3.6
                    }
                },
                'int2': {
                    'size_mb': 8.8,
                    'precision': 'int2',
                    'accuracy': {
                        'captioning_cider': 91.5,
                        'ocr_accuracy': 89.7,
                        'vqa_score': 69.3,
                        'retrieval_map': 83.2
                    },
                    'accuracy_loss': {
                        'captioning_cider': 5.3,
                        'ocr_accuracy': 4.8,
                        'vqa_score': 5.9,
                        'retrieval_map': 7.9
                    }
                }
            },
            'recommended_precision': 'int8',
            'analysis': {
                'best_size_accuracy_tradeoff': 'int8',
                'acceptable_accuracy_loss': True,
                'mobile_deployment_ready': True
            }
        }


def mock_mobile_runtime():
    """Factory function to create mobile runtime mocks."""
    return {
        'tflite': MobileFixture.mock_tflite_interpreter(),
        'coreml': MobileFixture.mock_coreml_model(),
        'hexagon': MobileFixture.mock_hexagon_runtime()
    }


def create_mobile_benchmark_suite() -> Dict[str, Any]:
    """Create a comprehensive mobile benchmark suite."""
    return {
        'devices': MobileFixture.mock_device_profiles(),
        'test_cases': [
            {
                'name': 'single_image_inference',
                'description': 'Test inference on single image',
                'input_shape': [1, 224, 224, 3],
                'expected_output_shape': [1, 1000],
                'timeout_ms': 100
            },
            {
                'name': 'batch_inference',
                'description': 'Test batch inference',
                'input_shape': [4, 224, 224, 3],
                'expected_output_shape': [4, 1000],
                'timeout_ms': 400
            },
            {
                'name': 'memory_pressure',
                'description': 'Test under memory pressure',
                'input_shape': [1, 224, 224, 3],
                'memory_limit_mb': 100,
                'expected_behavior': 'graceful_degradation'
            },
            {
                'name': 'thermal_throttling',
                'description': 'Test under thermal throttling',
                'input_shape': [1, 224, 224, 3],
                'temperature_celsius': 45,
                'expected_behavior': 'performance_scaling'
            }
        ],
        'performance_metrics': [
            'inference_time_ms',
            'memory_usage_mb',
            'cpu_utilization_percent',
            'battery_consumption_mah',
            'thermal_impact_celsius'
        ]
    }