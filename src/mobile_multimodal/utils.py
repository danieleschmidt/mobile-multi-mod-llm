"""Utility functions for mobile multi-modal LLM operations."""

import hashlib
import json
import logging
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

logger = logging.getLogger(__name__)

# Security and validation constants
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TEXT_LENGTH = 10000
ALLOWED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
SECURE_RANDOM = secrets.SystemRandom()
MIN_IMAGE_DIMENSION = 16  # Minimum image size
MAX_IMAGE_DIMENSION = 4096  # Maximum image size


class ImageProcessor:
    """Image preprocessing utilities for mobile models."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
    
    def validate_image_path(self, image_path: str) -> bool:
        """Validate image file path and format."""
        try:
            path = Path(image_path)
            
            # Check if file exists
            if not path.exists():
                logger.error(f"Image file does not exist: {image_path}")
                return False
            
            # Check file size
            if path.stat().st_size > MAX_IMAGE_SIZE:
                logger.error(f"Image file too large: {path.stat().st_size} bytes")
                return False
            
            # Check file extension
            if path.suffix.lower() not in ALLOWED_IMAGE_FORMATS:
                logger.error(f"Unsupported image format: {path.suffix}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image path {image_path}: {e}")
            return False
    
    def sanitize_image_input(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Sanitize and validate image input."""
        if image is None:
            return None
        
        # Check data type
        if not isinstance(image, np.ndarray):
            logger.error("Image must be numpy array")
            return None
        
        # Check dimensions
        if len(image.shape) not in [2, 3]:
            logger.error(f"Invalid image dimensions: {image.shape}")
            return None
        
        h, w = image.shape[:2]
        if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
            logger.error(f"Image too small: {h}x{w}")
            return None
        
        if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
            logger.error(f"Image too large: {h}x{w}")
            return None
        
        # Ensure uint8 data type
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Ensure 3 channels for color images
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            logger.error(f"Invalid number of channels: {image.shape[2]}")
            return None
        
        return image
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file path."""
        if cv2 is None:
            raise ImportError("OpenCV is required for image loading")
            
        # Validate input path
        if not self.validate_image_path(image_path):
            return None
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Sanitize loaded image
            image = self.sanitize_image_input(image)
            if image is None:
                return None
            
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def resize_image(self, image: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """Resize image to target size."""
        if cv2 is None:
            raise ImportError("OpenCV is required for image resizing")
            
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        if maintain_aspect:
            # Calculate scale to maintain aspect ratio
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to target size
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            
            padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            
            return padded
        else:
            # Direct resize without maintaining aspect ratio
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization."""
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image = (image - self.imagenet_mean) / self.imagenet_std
        
        return image
    
    def preprocess_image(self, image: Union[str, np.ndarray], 
                        maintain_aspect: bool = True) -> Optional[np.ndarray]:
        """Complete preprocessing pipeline with security validation."""
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = self.load_image(image)
                if image is None:
                    return None
            else:
                # Sanitize numpy array input
                image = self.sanitize_image_input(image)
                if image is None:
                    return None
            
            # Resize image
            resized = self.resize_image(image, maintain_aspect)
            
            # Normalize
            normalized = self.normalize_image(resized)
            
            # Final validation
            if normalized is None or np.isnan(normalized).any() or np.isinf(normalized).any():
                logger.error("Preprocessing resulted in invalid values")
                return None
            
            return normalized
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def augment_image(self, image: np.ndarray, augmentations: List[str] = None) -> np.ndarray:
        """Apply data augmentations."""
        if cv2 is None or augmentations is None:
            return image
            
        augmented = image.copy()
        
        for aug in augmentations:
            if aug == "horizontal_flip" and np.random.rand() > 0.5:
                augmented = cv2.flip(augmented, 1)
            
            elif aug == "rotation" and np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)
                h, w = augmented.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented = cv2.warpAffine(augmented, M, (w, h))
            
            elif aug == "brightness" and np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)
            
            elif aug == "contrast" and np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                mean = augmented.mean()
                augmented = np.clip((augmented - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        return augmented
    
    def compute_image_hash(self, image: np.ndarray) -> str:
        """Compute hash of image for caching."""
        if image is None:
            return "none"
        try:
            # Use image shape and a sample of pixels for hash
            shape_str = str(image.shape)
            sample_pixels = image[::10, ::10].flatten()[:100]  # Sample pixels
            content = shape_str.encode() + sample_pixels.tobytes()
            return hashlib.sha256(content).hexdigest()[:16]  # Short hash
        except:
            return "hash_error"


class TextTokenizer:
    """Simple text tokenizer for mobile deployment."""
    
    def __init__(self, vocab_size: int = 32000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[UNK]": 3,
            "[MASK]": 4
        }
        
        # Initialize with special tokens
        for token, token_id in self.special_tokens.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from text corpus."""
        word_freq = {}
        
        # Count word frequencies
        for text in texts:
            words = self._simple_tokenize(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and add to vocabulary
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        current_id = len(self.special_tokens)
        for word, freq in sorted_words:
            if freq >= min_freq and current_id < self.vocab_size:
                if word not in self.token_to_id:
                    self.token_to_id[word] = current_id
                    self.id_to_token[current_id] = word
                    current_id += 1
        
        logger.info(f"Built vocabulary with {len(self.token_to_id)} tokens")
    
    def validate_text_input(self, text: str) -> bool:
        """Validate text input for security and length constraints."""
        if not isinstance(text, str):
            logger.error("Text input must be string")
            return False
        
        if len(text) > MAX_TEXT_LENGTH:
            logger.error(f"Text too long: {len(text)} characters")
            return False
        
        # Check for potential injection patterns
        suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(', '__import__']
        text_lower = text.lower()
        
        for pattern in suspicious_patterns:
            if pattern in text_lower:
                logger.warning(f"Suspicious pattern detected in text: {pattern}")
                return False
        
        return True
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text input by removing/escaping dangerous content."""
        if not self.validate_text_input(text):
            return ""
        
        # Remove control characters except whitespace
        import re
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with input validation."""
        # Sanitize input
        text = self.sanitize_text(text)
        if not text:
            return []
        
        # Basic preprocessing
        text = text.lower().strip()
        
        # Remove punctuation and split
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Additional length validation per word
        words = [word for word in words if len(word) <= 50]  # Max word length
        
        return words
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs with input validation."""
        if not self.validate_text_input(text):
            return [self.special_tokens["[UNK]"]]
        
        words = self._simple_tokenize(text)
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens["[BOS]"])
        
        for word in words:
            if word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
            else:
                token_ids.append(self.special_tokens["[UNK]"])
        
        if add_special_tokens:
            token_ids.append(self.special_tokens["[EOS]"])
        
        # Truncate or pad to max length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.special_tokens["[PAD]"]] * (self.max_length - len(token_ids)))
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        words = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                if skip_special_tokens and token in self.special_tokens:
                    continue
                    
                words.append(token)
        
        return " ".join(words)
    
    def save_vocab(self, path: str):
        """Save vocabulary to file with secure writing."""
        try:
            # Validate path
            path_obj = Path(path)
            if path_obj.suffix not in ['.json', '.txt']:
                raise ValueError("Vocabulary must be saved as .json or .txt file")
            
            vocab_data = {
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
                "vocab_size": self.vocab_size,
                "max_length": self.max_length,
                "special_tokens": self.special_tokens,
                "version": "1.0",
                "created_at": time.time()
            }
            
            # Write to temporary file first, then rename for atomicity
            temp_path = path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, indent=2, ensure_ascii=True)
            
            # Atomic rename
            os.rename(temp_path, path)
            
            logger.info(f"Vocabulary saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save vocabulary: {e}")
            # Clean up temp file if it exists
            temp_path = path + ".tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def load_vocab(self, path: str):
        """Load vocabulary from file with validation."""
        try:
            # Validate path
            if not os.path.exists(path):
                raise FileNotFoundError(f"Vocabulary file not found: {path}")
            
            # Check file size for safety
            if os.path.getsize(path) > 100 * 1024 * 1024:  # 100MB limit
                raise ValueError("Vocabulary file too large")
            
            with open(path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # Validate required fields
            required_fields = ["token_to_id", "id_to_token", "vocab_size", "max_length", "special_tokens"]
            for field in required_fields:
                if field not in vocab_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate data types and ranges
            if not isinstance(vocab_data["vocab_size"], int) or vocab_data["vocab_size"] <= 0:
                raise ValueError("Invalid vocab_size")
            
            if not isinstance(vocab_data["max_length"], int) or vocab_data["max_length"] <= 0:
                raise ValueError("Invalid max_length")
            
            self.token_to_id = vocab_data["token_to_id"]
            self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
            self.vocab_size = vocab_data["vocab_size"]
            self.max_length = vocab_data["max_length"]
            self.special_tokens = vocab_data["special_tokens"]
            
            logger.info(f"Vocabulary loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary from {path}: {e}")
            # Reset to safe defaults
            self.__init__(self.vocab_size, self.max_length)


class BenchmarkUtils:
    """Utilities for performance benchmarking."""
    
    @staticmethod
    def measure_inference_time(model_func, inputs, iterations: int = 100, 
                             warmup: int = 10) -> Dict[str, float]:
        """Measure inference time statistics."""
        # Warmup runs
        for _ in range(warmup):
            _ = model_func(inputs)
        
        # Benchmark runs
        times = []
        for _ in range(iterations):
            start_time = time.time()
            _ = model_func(inputs)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "fps": 1000.0 / float(np.mean(times)),
            "iterations": iterations
        }
    
    @staticmethod
    def measure_memory_usage(model_func, inputs) -> Dict[str, float]:
        """Measure memory usage during inference with error handling."""
        try:
            import psutil
            import gc
            
            # Input validation
            if not callable(model_func):
                raise ValueError("model_func must be callable")
            
            if inputs is None:
                raise ValueError("inputs cannot be None")
            
            # Clear cache
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Measure before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            gpu_memory_before = 0
            if torch is not None and torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # Run inference with timeout protection
            start_time = time.time()
            try:
                result = model_func(inputs)
                inference_time = time.time() - start_time
                
                # Sanity check on inference time (>30s might indicate hanging)
                if inference_time > 30:
                    logger.warning(f"Inference took unusually long: {inference_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Model inference failed during memory measurement: {e}")
                return {"error": f"Inference failed: {str(e)}"}
            
            # Measure after
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            gpu_memory_after = 0
            
            if torch is not None and torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            cpu_memory_diff = max(0, memory_after - memory_before)  # Ensure non-negative
            gpu_memory_diff = max(0, gpu_memory_after - gpu_memory_before)
            
            return {
                "cpu_memory_mb": cpu_memory_diff,
                "gpu_memory_mb": gpu_memory_diff,
                "total_memory_mb": cpu_memory_diff + gpu_memory_diff,
                "inference_time_s": inference_time,
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after
            }
            
        except ImportError:
            return {"error": "psutil required for memory measurement"}
        except Exception as e:
            logger.error(f"Memory measurement failed: {e}")
            return {"error": f"Memory measurement failed: {str(e)}"}
    
    @staticmethod
    def profile_model_layers(model, sample_input, iterations: int = 10) -> Dict[str, Any]:
        """Profile individual model layers."""
        if torch is None:
            return {"error": "PyTorch not available"}
            
        layer_times = {}
        layer_memory = {}
        
        def profile_hook(name):
            def hook_fn(module, input, output):
                # Simple timing (not very accurate but gives relative comparison)
                start_time = time.time()
                
                # Dummy operation to trigger computation
                if isinstance(output, torch.Tensor):
                    _ = output.sum()
                    
                end_time = time.time()
                
                if name not in layer_times:
                    layer_times[name] = []
                layer_times[name].append((end_time - start_time) * 1000)
                
            return hook_fn
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(profile_hook(name))
                hooks.append(hook)
        
        # Run profiling
        model.eval()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(sample_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate statistics
        layer_stats = {}
        for name, times in layer_times.items():
            times_array = np.array(times)
            layer_stats[name] = {
                "mean_ms": float(np.mean(times_array)),
                "std_ms": float(np.std(times_array)),
                "total_time_ms": float(np.sum(times_array)),
                "relative_time_percent": 0.0  # Will be calculated below
            }
        
        # Calculate relative percentages
        total_time = sum(stats["total_time_ms"] for stats in layer_stats.values())
        for stats in layer_stats.values():
            stats["relative_time_percent"] = (stats["total_time_ms"] / total_time) * 100
        
        return {
            "layer_profiles": layer_stats,
            "total_inference_time_ms": total_time / iterations,
            "iterations": iterations
        }


class ModelUtils:
    """General model utilities."""
    
    @staticmethod
    def validate_model_path(model_path: str) -> bool:
        """Validate model file path and basic properties."""
        try:
            path = Path(model_path)
            
            # Check if file exists
            if not path.exists():
                logger.error(f"Model file does not exist: {model_path}")
                return False
            
            # Check file extension
            allowed_extensions = {'.pth', '.pt', '.onnx', '.tflite', '.pb', '.mlmodel'}
            if path.suffix.lower() not in allowed_extensions:
                logger.error(f"Unsupported model format: {path.suffix}")
                return False
            
            # Check file size (reasonable limits)
            file_size = path.stat().st_size
            if file_size > 500 * 1024 * 1024:  # 500MB limit
                logger.error(f"Model file too large: {file_size} bytes")
                return False
            
            if file_size < 1024:  # 1KB minimum
                logger.error(f"Model file suspiciously small: {file_size} bytes")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating model path {model_path}: {e}")
            return False
    
    @staticmethod
    def calculate_model_hash(model_path: str) -> str:
        """Calculate hash of model file for verification."""
        if not ModelUtils.validate_model_path(model_path):
            return ""
        
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(model_path, "rb") as f:
                while chunk := f.read(8192):  # Larger chunk size for efficiency
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {model_path}: {e}")
            return ""
    
    @staticmethod
    def validate_model_architecture(model, expected_layers: List[str]) -> Dict[str, bool]:
        """Validate that model has expected architecture."""
        if torch is None:
            return {"error": "PyTorch not available"}
            
        model_layers = [name for name, _ in model.named_modules()]
        validation_results = {}
        
        for expected_layer in expected_layers:
            validation_results[expected_layer] = any(
                expected_layer in layer_name for layer_name in model_layers
            )
        
        return validation_results
    
    @staticmethod
    def get_model_summary(model) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        if torch is None:
            return {"error": "PyTorch not available"}
            
        summary = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "layer_types": {},
            "layer_count": 0,
            "model_depth": 0
        }
        
        # Count parameters and analyze layers
        for name, param in model.named_parameters():
            summary["total_parameters"] += param.numel()
            if param.requires_grad:
                summary["trainable_parameters"] += param.numel()
        
        # Analyze layer types
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layer_type = type(module).__name__
                summary["layer_types"][layer_type] = summary["layer_types"].get(layer_type, 0) + 1
                summary["layer_count"] += 1
                
                # Estimate depth based on name
                depth = name.count('.')
                summary["model_depth"] = max(summary["model_depth"], depth)
        
        # Calculate model size estimates
        summary["model_size_mb"] = summary["total_parameters"] * 4 / (1024 * 1024)  # FP32
        summary["model_size_int8_mb"] = summary["total_parameters"] / (1024 * 1024)  # INT8
        summary["model_size_int2_mb"] = summary["total_parameters"] * 0.25 / (1024 * 1024)  # INT2
        
        return summary


class ConfigManager:
    """Configuration management for mobile deployment."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "mobile_config.json"
        self.config = self._load_default_config()
        
        if os.path.exists(self.config_path):
            self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "model": {
                "input_size": [224, 224],
                "batch_size": 1,
                "precision": "fp32",
                "quantization": "none"
            },
            "preprocessing": {
                "normalize": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize_method": "bilinear"
            },
            "performance": {
                "use_gpu": False,
                "num_threads": 1,
                "memory_limit_mb": 512
            },
            "deployment": {
                "target_platform": "android",
                "optimization_level": "balanced",
                "enable_profiling": False
            }
        }
    
    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Merge with default config
            self._deep_update(self.config, loaded_config)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value


# Test and example usage
if __name__ == "__main__":
    # Test image processor
    if cv2 is not None:
        processor = ImageProcessor()
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"Test image shape: {test_image.shape}")
        
        # Process image
        processed = processor.preprocess_image(test_image)
        if processed is not None:
            print(f"Processed image shape: {processed.shape}")
            print(f"Processed image range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Test tokenizer
    tokenizer = TextTokenizer(vocab_size=1000)
    test_texts = [
        "This is a test sentence.",
        "Another example text for testing.",
        "Mobile AI models are efficient."
    ]
    
    tokenizer.build_vocab(test_texts)
    
    # Test encoding/decoding
    test_text = "This is a test"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded[:10]}...")  # Show first 10 tokens
    print(f"Decoded: {decoded}")
    
    # Test config manager
    config_manager = ConfigManager()
    print(f"\nDefault input size: {config_manager.get('model.input_size')}")
    
    config_manager.set('model.batch_size', 4)
    print(f"Updated batch size: {config_manager.get('model.batch_size')}")
    
    print("\nMobile utilities module loaded successfully!")