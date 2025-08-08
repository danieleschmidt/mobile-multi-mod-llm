"""Security validation and protection for mobile multi-modal models."""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

logger = logging.getLogger(__name__)
security_logger = logging.getLogger(f"{__name__}.security")

# Security constants
MAX_REQUEST_RATE = 100  # requests per minute
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
BLOCKED_PATTERNS = [
    b'<script', b'javascript:', b'vbscript:', b'data:text/html',
    b'<?php', b'<%', b'${', b'eval(', b'exec('
]
SESSION_TIMEOUT = 3600  # 1 hour


class SecurityValidator:
    """Comprehensive security validation for model inputs."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize security validator.
        
        Args:
            strict_mode: Enable strict security checks
        """
        self.strict_mode = strict_mode
        self.rate_limiter = RateLimiter(MAX_REQUEST_RATE)
        self.input_sanitizer = InputSanitizer()
        self.crypto = CryptoUtils()
        
        # Security metrics
        self._blocked_requests = 0
        self._suspicious_patterns = 0
        self._rate_limited_requests = 0
        
        logger.info(f"Security validator initialized (strict_mode={strict_mode})")
    
    def validate_request(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive request validation.
        
        Args:
            user_id: User identifier for rate limiting
            request_data: Request data to validate
            
        Returns:
            Validation result with security assessment
            
        Raises:
            SecurityError: If request fails security checks
        """
        validation_result = {
            "valid": True,
            "checks": {},
            "warnings": [],
            "blocked_reason": None,
            "sanitized_data": None
        }
        
        try:
            # Rate limiting check
            if not self.rate_limiter.allow_request(user_id):
                validation_result["valid"] = False
                validation_result["blocked_reason"] = "rate_limit_exceeded"
                validation_result["checks"]["rate_limit"] = False
                self._rate_limited_requests += 1
                security_logger.warning(f"Rate limit exceeded for user {user_id}")
                return validation_result
            
            validation_result["checks"]["rate_limit"] = True
            
            # Input sanitization and validation
            try:
                sanitized = self.input_sanitizer.sanitize_request(request_data)
                validation_result["sanitized_data"] = sanitized
                validation_result["checks"]["input_sanitization"] = True
            except SecurityError as e:
                validation_result["valid"] = False
                validation_result["blocked_reason"] = f"input_validation_failed: {e}"
                validation_result["checks"]["input_sanitization"] = False
                self._blocked_requests += 1
                security_logger.error(f"Input validation failed: {e}")
                return validation_result
            
            # Content security checks
            if "image" in request_data:
                image_security = self._validate_image_security(request_data["image"])
                validation_result["checks"]["image_security"] = image_security["valid"]
                if not image_security["valid"]:
                    validation_result["valid"] = False
                    validation_result["blocked_reason"] = f"image_security: {image_security['reason']}"
                    return validation_result
                validation_result["warnings"].extend(image_security.get("warnings", []))
            
            if "text" in request_data:
                text_security = self._validate_text_security(request_data["text"])
                validation_result["checks"]["text_security"] = text_security["valid"]
                if not text_security["valid"]:
                    validation_result["valid"] = False
                    validation_result["blocked_reason"] = f"text_security: {text_security['reason']}"
                    return validation_result
                validation_result["warnings"].extend(text_security.get("warnings", []))
            
            # Request size validation
            request_size = len(json.dumps(request_data, default=str).encode())
            validation_result["checks"]["request_size"] = request_size < MAX_FILE_SIZE
            if request_size >= MAX_FILE_SIZE:
                validation_result["valid"] = False
                validation_result["blocked_reason"] = f"request_too_large: {request_size} bytes"
                return validation_result
            
            # Pattern detection
            suspicious_patterns = self._detect_suspicious_patterns(request_data)
            validation_result["checks"]["pattern_detection"] = len(suspicious_patterns) == 0
            if suspicious_patterns:
                validation_result["warnings"].extend(suspicious_patterns)
                if self.strict_mode:
                    validation_result["valid"] = False
                    validation_result["blocked_reason"] = f"suspicious_patterns: {suspicious_patterns}"
                    return validation_result
                self._suspicious_patterns += len(suspicious_patterns)
            
            # Metadata validation
            metadata_checks = self._validate_metadata(request_data)
            validation_result["checks"]["metadata"] = metadata_checks["valid"]
            validation_result["warnings"].extend(metadata_checks.get("warnings", []))
            
            logger.debug(f"Request validation completed for user {user_id}")
            return validation_result
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["blocked_reason"] = f"validation_error: {e}"
            security_logger.error(f"Security validation error: {e}", exc_info=True)
            return validation_result
    
    def _validate_image_security(self, image_data: Union[np.ndarray, str, bytes]) -> Dict[str, Any]:
        """Validate image data for security threats."""
        result = {"valid": True, "warnings": []}
        
        try:
            if isinstance(image_data, str):
                # File path validation
                if not self._is_safe_path(image_data):
                    result["valid"] = False
                    result["reason"] = "unsafe_file_path"
                    return result
                
                # Extension validation
                ext = Path(image_data).suffix.lower()
                if ext not in ALLOWED_EXTENSIONS:
                    result["valid"] = False
                    result["reason"] = f"unsupported_extension: {ext}"
                    return result
            
            elif isinstance(image_data, np.ndarray):
                # Array validation
                if image_data.size > 100 * 1024 * 1024:  # > 100M pixels
                    result["valid"] = False
                    result["reason"] = "image_too_large"
                    return result
                
                # Check for suspicious patterns in pixel data
                if self._has_suspicious_pixel_patterns(image_data):
                    result["warnings"].append("suspicious_pixel_patterns")
            
            elif isinstance(image_data, bytes):
                # Binary data validation
                if len(image_data) > MAX_FILE_SIZE:
                    result["valid"] = False
                    result["reason"] = "file_too_large"
                    return result
                
                # Check for embedded malicious content
                if self._has_embedded_threats(image_data):
                    result["valid"] = False
                    result["reason"] = "embedded_threats_detected"
                    return result
            
            return result
            
        except Exception as e:
            result["valid"] = False
            result["reason"] = f"validation_error: {e}"
            return result
    
    def _validate_text_security(self, text_data: str) -> Dict[str, Any]:
        """Validate text data for security threats."""
        result = {"valid": True, "warnings": []}
        
        try:
            if len(text_data) > 10000:  # Max 10K characters
                result["valid"] = False
                result["reason"] = "text_too_long"
                return result
            
            # Check for injection patterns
            text_lower = text_data.lower()
            injection_patterns = [
                'script>', 'javascript:', 'vbscript:', 'onload=', 'onerror=',
                '<?php', '<%', 'exec(', 'eval(', 'system(', 'shell_exec('
            ]
            
            for pattern in injection_patterns:
                if pattern in text_lower:
                    if self.strict_mode:
                        result["valid"] = False
                        result["reason"] = f"injection_pattern: {pattern}"
                        return result
                    else:
                        result["warnings"].append(f"potential_injection: {pattern}")
            
            # Check encoding
            try:
                text_data.encode('utf-8')
            except UnicodeEncodeError:
                result["warnings"].append("encoding_issues")
            
            return result
            
        except Exception as e:
            result["valid"] = False
            result["reason"] = f"validation_error: {e}"
            return result
    
    def _is_safe_path(self, path: str) -> bool:
        """Check if file path is safe."""
        try:
            # Resolve path and check for traversal
            resolved = Path(path).resolve()
            
            # Check for path traversal attempts
            if '..' in str(path) or path.startswith('/'):
                return False
            
            # Check for suspicious characters
            suspicious_chars = ['<', '>', '|', '&', ';', '$', '`']
            if any(char in path for char in suspicious_chars):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _has_suspicious_pixel_patterns(self, image: np.ndarray) -> bool:
        """Check for suspicious patterns in image pixels."""
        try:
            # Check for completely uniform images (potential steganography)
            if np.std(image) < 1e-6:
                return True
            
            # Check for unusual value distributions
            unique_values = len(np.unique(image))
            total_pixels = image.size
            if unique_values / total_pixels > 0.9:  # Too many unique values
                return True
            
            return False
            
        except Exception:
            return False
    
    def _has_embedded_threats(self, data: bytes) -> bool:
        """Check for embedded threats in binary data."""
        try:
            # Check for blocked patterns
            for pattern in BLOCKED_PATTERNS:
                if pattern in data:
                    return True
            
            # Check for executable headers
            executable_headers = [
                b'MZ',  # PE/EXE
                b'\x7fELF',  # ELF
                b'\xfe\xed\xfa',  # Mach-O
                b'#!/bin/',  # Shell script
            ]
            
            for header in executable_headers:
                if data.startswith(header):
                    return True
            
            return False
            
        except Exception:
            return True  # Assume threat on error
    
    def _detect_suspicious_patterns(self, data: Dict[str, Any]) -> List[str]:
        """Detect suspicious patterns in request data."""
        patterns = []
        
        try:
            data_str = json.dumps(data, default=str).lower()
            
            # SQL injection patterns
            sql_patterns = ['union select', 'drop table', 'exec(', '--', ';--']
            for pattern in sql_patterns:
                if pattern in data_str:
                    patterns.append(f"sql_injection: {pattern}")
            
            # Command injection patterns
            cmd_patterns = ['&&', '||', '$(', '`', '|', ';']
            for pattern in cmd_patterns:
                if pattern in data_str:
                    patterns.append(f"command_injection: {pattern}")
            
            # XSS patterns
            xss_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
            for pattern in xss_patterns:
                if pattern in data_str:
                    patterns.append(f"xss: {pattern}")
            
        except Exception as e:
            patterns.append(f"pattern_detection_error: {e}")
        
        return patterns
    
    def _validate_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request metadata."""
        result = {"valid": True, "warnings": []}
        
        try:
            # Check for excessive metadata
            if len(data) > 50:  # Too many fields
                result["warnings"].append("excessive_metadata_fields")
            
            # Check for suspicious field names
            suspicious_fields = ['__', 'eval', 'exec', 'system', 'admin', 'root']
            for field in data.keys():
                if any(sus in str(field).lower() for sus in suspicious_fields):
                    result["warnings"].append(f"suspicious_field: {field}")
            
        except Exception as e:
            result["warnings"].append(f"metadata_validation_error: {e}")
        
        return result
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics summary."""
        return {
            "blocked_requests": self._blocked_requests,
            "suspicious_patterns": self._suspicious_patterns,
            "rate_limited_requests": self._rate_limited_requests,
            "total_processed": (
                self._blocked_requests + 
                self._rate_limited_requests + 
                self.rate_limiter.get_total_requests()
            )
        }


class RateLimiter:
    """Token bucket rate limiter for request throttling."""
    
    def __init__(self, max_requests_per_minute: int = MAX_REQUEST_RATE):
        self.max_requests = max_requests_per_minute
        self.window_size = 60  # 1 minute window
        self.user_requests = {}  # user_id -> list of timestamps
        self._total_requests = 0
    
    def allow_request(self, user_id: str) -> bool:
        """Check if request is allowed for user."""
        current_time = time.time()
        
        # Initialize user if not exists
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        # Clean old requests
        user_requests = self.user_requests[user_id]
        cutoff_time = current_time - self.window_size
        self.user_requests[user_id] = [
            req_time for req_time in user_requests 
            if req_time > cutoff_time
        ]
        
        # Check rate limit
        if len(self.user_requests[user_id]) >= self.max_requests:
            return False
        
        # Allow request
        self.user_requests[user_id].append(current_time)
        self._total_requests += 1
        return True
    
    def get_total_requests(self) -> int:
        """Get total number of processed requests."""
        return self._total_requests


class InputSanitizer:
    """Input sanitization and normalization."""
    
    def sanitize_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data."""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = self._sanitize_string(str(key))
            
            # Sanitize value based on type
            if isinstance(value, str):
                clean_value = self._sanitize_string(value)
            elif isinstance(value, (int, float)):
                clean_value = self._sanitize_number(value)
            elif isinstance(value, dict):
                clean_value = self.sanitize_request(value)
            elif isinstance(value, list):
                clean_value = [self._sanitize_value(v) for v in value]
            else:
                clean_value = value  # Keep as-is for other types
            
            sanitized[clean_key] = clean_value
        
        return sanitized
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize unicode
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception:
            pass
        
        # Remove control characters except whitespace
        text = ''.join(char for char in text if ord(char) >= 32 or char.isspace())
        
        # Limit length
        if len(text) > 10000:
            text = text[:10000]
        
        return text.strip()
    
    def _sanitize_number(self, num: Union[int, float]) -> Union[int, float]:
        """Sanitize numeric input."""
        # Check for NaN and infinity
        if isinstance(num, float):
            if np.isnan(num) or np.isinf(num):
                return 0.0
        
        # Limit range
        if isinstance(num, int):
            return max(-2**31, min(2**31-1, num))
        else:
            return max(-1e10, min(1e10, num))
    
    def _sanitize_value(self, value: Any) -> Any:
        """Generic value sanitization."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, (int, float)):
            return self._sanitize_number(value)
        elif isinstance(value, dict):
            return self.sanitize_request(value)
        else:
            return value


class CryptoUtils:
    """Cryptographic utilities for secure operations."""
    
    def __init__(self):
        self.secret_key = self._get_secret_key()
    
    def _get_secret_key(self) -> bytes:
        """Get or generate secret key."""
        key_file = Path.home() / '.mobile_multimodal_key'
        
        if key_file.exists():
            try:
                return key_file.read_bytes()[:32]  # 256-bit key
            except Exception:
                pass
        
        # Generate new key
        key = secrets.token_bytes(32)
        try:
            key_file.write_bytes(key)
            key_file.chmod(0o600)  # Owner read/write only
        except Exception:
            pass  # Use in-memory key if can't save
        
        return key
    
    def hash_data(self, data: Union[str, bytes]) -> str:
        """Generate secure hash of data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    def sign_data(self, data: Union[str, bytes]) -> str:
        """Generate HMAC signature for data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        signature = hmac.new(self.secret_key, data, hashlib.sha256)
        return signature.hexdigest()
    
    def verify_signature(self, data: Union[str, bytes], signature: str) -> bool:
        """Verify HMAC signature."""
        expected = self.sign_data(data)
        return hmac.compare_digest(expected, signature)


class SecurityError(Exception):
    """Security-related error."""
    pass


# Example usage and testing
if __name__ == "__main__":
    print("Testing security validation...")
    
    # Test security validator
    validator = SecurityValidator(strict_mode=True)
    
    # Test safe request
    safe_request = {
        "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "text": "Generate a caption for this image",
        "user_id": "test_user"
    }
    
    result = validator.validate_request("test_user", safe_request)
    print(f"Safe request validation: {result['valid']}")
    
    # Test suspicious request
    suspicious_request = {
        "text": "'; DROP TABLE users; --",
        "script": "<script>alert('xss')</script>"
    }
    
    result = validator.validate_request("test_user", suspicious_request)
    print(f"Suspicious request validation: {result['valid']}")
    print(f"Blocked reason: {result['blocked_reason']}")
    
    # Test rate limiting
    for i in range(102):  # Exceed rate limit
        validator.rate_limiter.allow_request("heavy_user")
    
    result = validator.validate_request("heavy_user", safe_request)
    print(f"Rate limited request: {result['valid']}")
    
    # Test crypto utils
    crypto = CryptoUtils()
    data = "test data"
    signature = crypto.sign_data(data)
    is_valid = crypto.verify_signature(data, signature)
    print(f"Signature validation: {is_valid}")
    
    print("Security validation tests completed!")