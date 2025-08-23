"""Robust validation and error handling for mobile multi-modal LLM."""

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error."""
    pass

class SecurityValidationError(ValidationError):
    """Security-specific validation error."""
    pass

class ModelValidationError(ValidationError):
    """Model-specific validation error."""
    pass

class RobustValidator:
    """Comprehensive validation for mobile AI models and inputs."""
    
    # Security constraints
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_IMAGE_DIMENSION = 8192
    MIN_IMAGE_DIMENSION = 8
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    MAX_TEXT_LENGTH = 50000
    
    def __init__(self):
        self.validation_cache = {}
        self.security_logger = self._setup_security_logging()
    
    def _setup_security_logging(self) -> logging.Logger:
        """Setup dedicated security logging."""
        sec_logger = logging.getLogger('mobile_multimodal.security')
        sec_logger.setLevel(logging.INFO)
        
        # Create file handler for security events
        if not sec_logger.handlers:
            handler = logging.FileHandler('security_audit.log')
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            sec_logger.addHandler(handler)
        
        return sec_logger
    
    def validate_image_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Comprehensive image file validation with security checks."""
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise ValidationError(f"Image file does not exist: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            self.security_logger.warning(
                f"Large file detected: {file_path} ({file_size} bytes)"
            )
            raise SecurityValidationError(f"File too large: {file_size} bytes")
        
        if file_size == 0:
            raise ValidationError(f"Empty file: {file_path}")
        
        # Check file extension
        if file_path.suffix.lower() not in self.ALLOWED_IMAGE_EXTENSIONS:
            self.security_logger.warning(
                f"Suspicious file extension: {file_path.suffix}"
            )
            raise SecurityValidationError(f"Invalid file extension: {file_path.suffix}")
        
        # Calculate file hash for integrity
        file_hash = self._calculate_file_hash(file_path)
        
        # Basic header validation (magic bytes)
        is_valid_format = self._validate_image_header(file_path)
        if not is_valid_format:
            self.security_logger.error(
                f"Invalid image format detected: {file_path}"
            )
            raise SecurityValidationError("Invalid image file format")
        
        return {
            "path": str(file_path),
            "size_bytes": file_size,
            "extension": file_path.suffix,
            "hash": file_hash,
            "valid": True,
            "timestamp": time.time()
        }
    
    def validate_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """Validate numpy image array."""
        if not isinstance(image, np.ndarray):
            raise ValidationError(f"Expected numpy array, got {type(image)}")
        
        # Check dimensions
        if len(image.shape) not in [2, 3]:
            raise ValidationError(f"Invalid image dimensions: {image.shape}")
        
        if len(image.shape) == 3:
            height, width, channels = image.shape
            if channels not in [1, 3, 4]:
                raise ValidationError(f"Invalid number of channels: {channels}")
        else:
            height, width = image.shape
            channels = 1
        
        # Check size constraints
        if height < self.MIN_IMAGE_DIMENSION or width < self.MIN_IMAGE_DIMENSION:
            raise ValidationError(f"Image too small: {width}x{height}")
        
        if height > self.MAX_IMAGE_DIMENSION or width > self.MAX_IMAGE_DIMENSION:
            self.security_logger.warning(f"Large image detected: {width}x{height}")
            raise SecurityValidationError(f"Image too large: {width}x{height}")
        
        # Check data type and range
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValidationError(f"Invalid image dtype: {image.dtype}")
        
        if image.dtype == np.uint8:
            if image.min() < 0 or image.max() > 255:
                raise ValidationError("Invalid uint8 image range")
        elif image.dtype in [np.float32, np.float64]:
            if image.min() < -10 or image.max() > 10:
                self.security_logger.warning("Unusual float image range detected")
        
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "channels": channels,
            "size_mb": image.nbytes / (1024 * 1024),
            "valid": True
        }
    
    def validate_text_input(self, text: str) -> Dict[str, Any]:
        """Validate text input with security checks."""
        if not isinstance(text, str):
            raise ValidationError(f"Expected string, got {type(text)}")
        
        # Check length
        if len(text) > self.MAX_TEXT_LENGTH:
            self.security_logger.warning(f"Long text input detected: {len(text)} chars")
            raise SecurityValidationError(f"Text too long: {len(text)} characters")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            '<?php', '<script>', 'javascript:', 'data:',
            '../', '..\\\\', '<iframe>', '<object>'
        ]
        
        text_lower = text.lower()
        for pattern in suspicious_patterns:
            if pattern in text_lower:
                self.security_logger.error(f"Suspicious pattern detected: {pattern}")
                raise SecurityValidationError(f"Potentially malicious content detected")
        
        # Character validation
        if len(text.encode('utf-8')) > len(text) * 4:
            self.security_logger.warning("Unusual character encoding detected")
        
        return {
            "length": len(text),
            "encoding_size": len(text.encode('utf-8')),
            "valid": True,
            "hash": hashlib.sha256(text.encode()).hexdigest()[:16]
        }
    
    def validate_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration parameters."""
        required_fields = ['model', 'preprocessing', 'performance']
        
        for field in required_fields:
            if field not in config:
                raise ModelValidationError(f"Missing required config field: {field}")
        
        # Validate model parameters
        model_config = config['model']
        if 'batch_size' in model_config:
            batch_size = model_config['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 64:
                raise ModelValidationError(f"Invalid batch_size: {batch_size}")
        
        # Validate preprocessing parameters
        prep_config = config['preprocessing']
        if 'mean' in prep_config:
            mean = prep_config['mean']
            if not isinstance(mean, list) or len(mean) != 3:
                raise ModelValidationError(f"Invalid mean values: {mean}")
            
            if not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in mean):
                raise ModelValidationError(f"Mean values out of range: {mean}")
        
        # Validate performance parameters
        perf_config = config['performance']
        if 'memory_limit_mb' in perf_config:
            memory_limit = perf_config['memory_limit_mb']
            if not isinstance(memory_limit, int) or memory_limit < 64:
                raise ModelValidationError(f"Invalid memory limit: {memory_limit}")
        
        return {"valid": True, "validated_fields": len(config)}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _validate_image_header(self, file_path: Path) -> bool:
        """Validate image file header (magic bytes)."""
        magic_bytes = {
            b'\\xff\\xd8\\xff': 'JPEG',
            b'\\x89PNG\\r\\n\\x1a\\n': 'PNG',
            b'BM': 'BMP',
            b'RIFF': 'WEBP'
        }
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            for magic, format_name in magic_bytes.items():
                if header.startswith(magic):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error reading file header: {e}")
            return False
    
    def validate_deployment_environment(self) -> Dict[str, Any]:
        """Validate deployment environment and system resources."""
        import psutil
        import platform
        
        # System info
        system_info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version()
        }
        
        # Memory check
        memory = psutil.virtual_memory()
        if memory.available < 512 * 1024 * 1024:  # 512MB
            logger.warning(f"Low available memory: {memory.available / 1024**3:.2f}GB")
        
        # Disk space check
        disk = psutil.disk_usage('.')
        if disk.free < 1024 * 1024 * 1024:  # 1GB
            logger.warning(f"Low disk space: {disk.free / 1024**3:.2f}GB")
        
        # CPU info
        cpu_count = psutil.cpu_count()
        
        validation_result = {
            "system": system_info,
            "memory_gb": memory.total / 1024**3,
            "available_memory_gb": memory.available / 1024**3,
            "disk_free_gb": disk.free / 1024**3,
            "cpu_count": cpu_count,
            "valid": memory.available >= 256 * 1024 * 1024,  # Minimum 256MB
            "warnings": []
        }
        
        if memory.available < 512 * 1024 * 1024:
            validation_result["warnings"].append("Low available memory")
        
        if disk.free < 1024 * 1024 * 1024:
            validation_result["warnings"].append("Low disk space")
        
        return validation_result

class CircuitBreaker:
    """Circuit breaker pattern for robust error handling."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = 0

class RetryManager:
    """Intelligent retry manager with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def retry(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception