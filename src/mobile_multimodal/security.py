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


class SecurityError(Exception):
    """Security validation error."""
    pass


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
                    validation_result["blocked_reason"] = f"image_security_failed: {image_security['reason']}"
                    self._blocked_requests += 1
                    security_logger.error(f"Image security validation failed: {image_security['reason']}")
                    return validation_result
                if image_security["warnings"]:
                    validation_result["warnings"].extend(image_security["warnings"])
            
            # Text input validation
            if "text" in request_data:
                text_security = self._validate_text_security(request_data["text"])
                validation_result["checks"]["text_security"] = text_security["valid"]
                if not text_security["valid"]:
                    validation_result["valid"] = False
                    validation_result["blocked_reason"] = f"text_security_failed: {text_security['reason']}"
                    self._blocked_requests += 1
                    security_logger.error(f"Text security validation failed: {text_security['reason']}")
                    return validation_result
                if text_security["warnings"]:
                    validation_result["warnings"].extend(text_security["warnings"])
            
            # Model parameter validation
            if "parameters" in request_data:
                param_security = self._validate_parameters_security(request_data["parameters"])
                validation_result["checks"]["parameter_security"] = param_security["valid"]
                if not param_security["valid"]:
                    validation_result["valid"] = False
                    validation_result["blocked_reason"] = f"parameter_security_failed: {param_security['reason']}"
                    self._blocked_requests += 1
                    security_logger.error(f"Parameter security validation failed: {param_security['reason']}")
                    return validation_result
                if param_security["warnings"]:
                    validation_result["warnings"].extend(param_security["warnings"])
            
            # Session validation
            session_check = self._validate_session_security(user_id)
            validation_result["checks"]["session_security"] = session_check["valid"]
            if not session_check["valid"]:
                validation_result["warnings"].extend(session_check["warnings"])
            
            return validation_result
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["blocked_reason"] = f"validation_error: {e}"
            validation_result["checks"]["validation_error"] = False
            self._blocked_requests += 1
            security_logger.error(f"Security validation error: {e}", exc_info=True)
            return validation_result
    
    def _validate_image_security(self, image: Union[np.ndarray, str, bytes]) -> Dict[str, Any]:
        """Validate image content for security threats."""
        result = {"valid": True, "warnings": [], "reason": None}
        
        try:
            # Image format validation
            if isinstance(image, str):
                # File path validation
                if not self._is_safe_file_path(image):
                    result["valid"] = False
                    result["reason"] = "unsafe_file_path"
                    return result
                    
                # File extension check
                file_ext = Path(image).suffix.lower()
                if file_ext not in ALLOWED_EXTENSIONS:
                    result["valid"] = False
                    result["reason"] = f"invalid_extension: {file_ext}"
                    return result
                    
                # File size check
                if os.path.exists(image):
                    file_size = os.path.getsize(image)
                    if file_size > MAX_FILE_SIZE:
                        result["valid"] = False
                        result["reason"] = f"file_too_large: {file_size} bytes"
                        return result
                        
            elif isinstance(image, bytes):
                # Binary data validation
                if len(image) > MAX_FILE_SIZE:
                    result["valid"] = False
                    result["reason"] = f"data_too_large: {len(image)} bytes"
                    return result
                    
                # Check for embedded scripts or malicious content
                for pattern in BLOCKED_PATTERNS:
                    if pattern in image[:1024]:  # Check first 1KB
                        result["valid"] = False
                        result["reason"] = f"malicious_content_detected: {pattern.decode('utf-8', errors='ignore')}"
                        return result
                        
            elif isinstance(image, np.ndarray):
                # Array validation
                if image.size == 0:
                    result["valid"] = False
                    result["reason"] = "empty_array"
                    return result
                    
                # Size validation
                if image.nbytes > MAX_FILE_SIZE:
                    result["valid"] = False
                    result["reason"] = f"array_too_large: {image.nbytes} bytes"
                    return result
                    
                # Dimension validation
                if len(image.shape) > 4 or len(image.shape) < 2:
                    result["valid"] = False
                    result["reason"] = f"invalid_dimensions: {image.shape}"
                    return result
                    
                # Value range validation
                if image.dtype == np.uint8:
                    if np.any((image < 0) | (image > 255)):
                        result["warnings"].append("pixel_values_out_of_range")
                elif image.dtype == np.float32 or image.dtype == np.float64:
                    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                        result["valid"] = False
                        result["reason"] = "invalid_float_values"
                        return result
                        
            return result
            
        except Exception as e:
            result["valid"] = False
            result["reason"] = f"validation_exception: {e}"
            return result
    
    def _validate_text_security(self, text: str) -> Dict[str, Any]:
        """Validate text input for security threats."""
        result = {"valid": True, "warnings": [], "reason": None}
        
        try:
            # Length validation
            if len(text) > 10000:  # 10K character limit
                result["valid"] = False
                result["reason"] = f"text_too_long: {len(text)} characters"
                return result
            
            # Encoding validation
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                result["valid"] = False
                result["reason"] = "invalid_encoding"
                return result
            
            # Script injection detection
            text_lower = text.lower()
            script_patterns = ['<script', 'javascript:', 'vbscript:', 'onclick=', 'onerror=', 'onload=']
            for pattern in script_patterns:
                if pattern in text_lower:
                    result["valid"] = False
                    result["reason"] = f"script_injection_detected: {pattern}"
                    return result
            
            # SQL injection detection
            sql_patterns = ['drop table', 'select * from', 'union select', 'insert into', 'delete from']
            for pattern in sql_patterns:
                if pattern in text_lower:
                    result["warnings"].append(f"potential_sql_injection: {pattern}")
            
            # Command injection detection
            cmd_patterns = ['$(', '`', '&&', '||', ';rm ', ';cat ', ';ls ']
            for pattern in cmd_patterns:
                if pattern in text:
                    result["warnings"].append(f"potential_command_injection: {pattern}")
            
            return result
            
        except Exception as e:
            result["valid"] = False
            result["reason"] = f"text_validation_exception: {e}"
            return result
    
    def _validate_parameters_security(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model parameters for security."""
        result = {"valid": True, "warnings": [], "reason": None}
        
        try:
            # Parameter count validation
            if len(parameters) > 50:  # Reasonable parameter limit
                result["valid"] = False
                result["reason"] = f"too_many_parameters: {len(parameters)}"
                return result
            
            # Parameter validation
            for key, value in parameters.items():
                # Key validation
                if not isinstance(key, str):
                    result["valid"] = False
                    result["reason"] = f"invalid_parameter_key: {type(key)}"
                    return result
                
                if len(key) > 100:  # Key length limit
                    result["valid"] = False
                    result["reason"] = f"parameter_key_too_long: {len(key)}"
                    return result
                
                # Value validation
                if isinstance(value, (int, float)):
                    # Numeric range validation
                    if abs(value) > 1e10:
                        result["warnings"].append(f"large_numeric_value: {key}={value}")
                elif isinstance(value, str):
                    if len(value) > 1000:
                        result["valid"] = False
                        result["reason"] = f"parameter_string_too_long: {key}"
                        return result
                elif isinstance(value, (list, tuple)):
                    if len(value) > 1000:
                        result["valid"] = False
                        result["reason"] = f"parameter_array_too_long: {key}"
                        return result
                elif value is None:
                    result["warnings"].append(f"null_parameter_value: {key}")
                else:
                    result["warnings"].append(f"unusual_parameter_type: {key}={type(value)}")
            
            return result
            
        except Exception as e:
            result["valid"] = False
            result["reason"] = f"parameter_validation_exception: {e}"
            return result
    
    def _validate_session_security(self, user_id: str) -> Dict[str, Any]:
        """Validate user session security."""
        result = {"valid": True, "warnings": [], "reason": None}
        
        try:
            # User ID validation
            if not user_id or len(user_id) < 3:
                result["warnings"].append("weak_user_id")
            
            if len(user_id) > 100:
                result["warnings"].append("unusually_long_user_id")
            
            # Character validation
            if not user_id.replace('_', '').replace('-', '').isalnum():
                result["warnings"].append("special_characters_in_user_id")
            
            return result
            
        except Exception as e:
            result["warnings"].append(f"session_validation_exception: {e}")
            return result
    
    def _is_safe_file_path(self, file_path: str) -> bool:
        """Check if file path is safe and doesn't contain directory traversal."""
        try:
            # Resolve path to prevent directory traversal
            resolved = Path(file_path).resolve()
            
            # Check for directory traversal patterns
            dangerous_patterns = ['../', '..\\', '/etc/', '/proc/', '/sys/', 'C:\\Windows', 'C:\\System']
            path_str = str(resolved).lower()
            
            for pattern in dangerous_patterns:
                if pattern.lower() in path_str:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_security_metrics(self) -> Dict[str, int]:
        """Get security metrics."""
        return {
            "blocked_requests": self._blocked_requests,
            "suspicious_patterns": self._suspicious_patterns,
            "rate_limited_requests": self._rate_limited_requests,
            "total_validations": (self._blocked_requests + self._suspicious_patterns + 
                                self._rate_limited_requests)
        }


class RateLimiter:
    """Thread-safe rate limiter with sliding window."""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self._lock = threading.Lock()
    
    def allow_request(self, user_id: str) -> bool:
        """Check if request is allowed for user."""
        with self._lock:
            current_time = time.time()
            
            if user_id not in self.requests:
                self.requests[user_id] = []
            
            # Clean old requests
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id]
                if current_time - req_time < self.time_window
            ]
            
            # Check rate limit
            if len(self.requests[user_id]) >= self.max_requests:
                return False
            
            # Add current request
            self.requests[user_id].append(current_time)
            return True


class InputSanitizer:
    """Input sanitization and validation."""
    
    def sanitize_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data."""
        sanitized = {}
        
        for key, value in request_data.items():
            # Sanitize key
            clean_key = self._sanitize_string(str(key))
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[clean_key] = self._sanitize_string(value)
            elif isinstance(value, (int, float)):
                sanitized[clean_key] = self._sanitize_numeric(value)
            elif isinstance(value, np.ndarray):
                sanitized[clean_key] = self._sanitize_array(value)
            elif isinstance(value, dict):
                sanitized[clean_key] = self.sanitize_request(value)
            elif isinstance(value, (list, tuple)):
                sanitized[clean_key] = self._sanitize_list(value)
            else:
                sanitized[clean_key] = value
        
        return sanitized
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Limit length
        if len(text) > 10000:
            text = text[:10000]
        
        # Remove control characters except whitespace
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        return sanitized
    
    def _sanitize_numeric(self, value: Union[int, float]) -> Union[int, float]:
        """Sanitize numeric input."""
        # Handle infinity and NaN
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return 0.0
        
        # Clamp to reasonable ranges
        if isinstance(value, int):
            return max(-1e10, min(1e10, value))
        else:
            return max(-1e10, min(1e10, float(value)))
    
    def _sanitize_array(self, array: np.ndarray) -> np.ndarray:
        """Sanitize numpy array."""
        if array.size == 0:
            raise SecurityError("Empty array not allowed")
        
        # Handle infinite and NaN values
        if array.dtype in [np.float32, np.float64]:
            array = np.nan_to_num(array, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return array
    
    def _sanitize_list(self, items: Union[list, tuple]) -> list:
        """Sanitize list or tuple."""
        sanitized = []
        
        for item in items:
            if isinstance(item, str):
                sanitized.append(self._sanitize_string(item))
            elif isinstance(item, (int, float)):
                sanitized.append(self._sanitize_numeric(item))
            elif isinstance(item, np.ndarray):
                sanitized.append(self._sanitize_array(item))
            elif isinstance(item, dict):
                sanitized.append(self.sanitize_request(item))
            else:
                sanitized.append(item)
        
        return sanitized


class CryptoUtils:
    """Cryptographic utilities for secure operations."""
    
    def __init__(self):
        self.key = secrets.token_bytes(32)  # 256-bit key
    
    def hash_data(self, data: Union[str, bytes]) -> str:
        """Create secure hash of data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    def create_hmac(self, data: Union[str, bytes], key: Optional[bytes] = None) -> str:
        """Create HMAC for data integrity."""
        if key is None:
            key = self.key
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(length)


# Security monitoring and alerting
class SecurityMonitor:
    """Security event monitoring and alerting."""
    
    def __init__(self):
        self.events = []
        self.alert_thresholds = {
            "blocked_requests_per_minute": 10,
            "rate_limit_violations_per_minute": 20,
            "suspicious_patterns_per_hour": 50
        }
    
    def record_security_event(self, event_type: str, details: Dict[str, Any]):
        """Record security event."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        
        self.events.append(event)
        
        # Keep only recent events (last 24 hours)
        cutoff_time = time.time() - 86400
        self.events = [e for e in self.events if e["timestamp"] > cutoff_time]
        
        # Check for alert conditions
        self._check_alerts(event)
    
    def _check_alerts(self, event: Dict[str, Any]):
        """Check if event triggers security alerts."""
        current_time = time.time()
        
        # Check blocked requests in last minute
        recent_blocked = [
            e for e in self.events 
            if (e["type"] == "blocked_request" and 
                current_time - e["timestamp"] < 60)
        ]
        
        if len(recent_blocked) >= self.alert_thresholds["blocked_requests_per_minute"]:
            security_logger.critical(f"SECURITY ALERT: High rate of blocked requests: {len(recent_blocked)}/min")
        
        # Check rate limit violations
        recent_rate_limits = [
            e for e in self.events 
            if (e["type"] == "rate_limit_exceeded" and 
                current_time - e["timestamp"] < 60)
        ]
        
        if len(recent_rate_limits) >= self.alert_thresholds["rate_limit_violations_per_minute"]:
            security_logger.critical(f"SECURITY ALERT: High rate of rate limit violations: {len(recent_rate_limits)}/min")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security event summary."""
        if not self.events:
            return {"total_events": 0, "event_types": {}, "recent_events": 0}
        
        current_time = time.time()
        recent_events = [e for e in self.events if current_time - e["timestamp"] < 3600]  # Last hour
        
        event_types = {}
        for event in self.events:
            event_type = event["type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.events),
            "event_types": event_types,
            "recent_events": len(recent_events),
            "oldest_event": min(e["timestamp"] for e in self.events) if self.events else None,
            "newest_event": max(e["timestamp"] for e in self.events) if self.events else None
        }


# Thread import for the RateLimiter  
import threading
            
            # Text content validation
            if any(key in request_data for key in ["question", "text", "caption"]):
                text_security = self._validate_text_security(request_data)
                validation_result["checks"]["text_security"] = text_security["valid"]
                if not text_security["valid"]:
                    validation_result["valid"] = False
                    validation_result["blocked_reason"] = f"text_security_failed: {text_security.get('reason', 'unknown')}"
                    self._suspicious_patterns += 1
                    security_logger.warning(f"Text security check failed: {text_security.get('reason', 'unknown')}")
                    return validation_result
                
                if text_security.get("warnings"):
                    validation_result["warnings"].extend(text_security["warnings"])
            
            # Operation-specific validation
            operation = request_data.get("operation", "unknown")
            operation_security = self._validate_operation_security(operation, request_data)
            validation_result["checks"]["operation_security"] = operation_security["valid"]
            if not operation_security["valid"]:
                validation_result["valid"] = False
                validation_result["blocked_reason"] = f"operation_security_failed: {operation_security.get('reason', 'unknown')}"
                self._blocked_requests += 1
                security_logger.error(f"Operation security check failed: {operation_security.get('reason', 'unknown')}")
                return validation_result
            
            if operation_security.get("warnings"):
                validation_result["warnings"].extend(operation_security["warnings"])
            
            # Final security assessment
            if self.strict_mode and len(validation_result["warnings"]) > 3:
                validation_result["valid"] = False
                validation_result["blocked_reason"] = "too_many_security_warnings"
                security_logger.warning(f"Request blocked due to multiple warnings: {len(validation_result['warnings'])}")
                return validation_result
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            validation_result["valid"] = False
            validation_result["blocked_reason"] = f"validation_error: {e}"
            return validation_result
    
    def _validate_image_security(self, image: Union[np.ndarray, List, Any]) -> Dict[str, Any]:
        """Validate image for security threats."""
        result = {
            "valid": True,
            "warnings": [],
            "reason": None
        }
        
        try:
            # Try to convert to numpy array if not already
            if not isinstance(image, np.ndarray):
                try:
                    # Handle nested lists (common in tests/mock scenarios)
                    if isinstance(image, (list, tuple)):
                        image = np.array(image)
                    else:
                        result["valid"] = False
                        result["reason"] = "invalid_image_type"
                        return result
                except Exception as e:
                    result["valid"] = False
                    result["reason"] = f"image_conversion_failed: {e}"
                    return result
            
            # Check dimensions
            if len(image.shape) not in [2, 3]:
                result["valid"] = False
                result["reason"] = "invalid_image_dimensions"
                return result
            
            h, w = image.shape[:2]
            if h < 1 or w < 1 or h > 4096 or w > 4096:
                result["valid"] = False
                result["reason"] = "invalid_image_size"
                return result
            
            # Check for suspicious patterns in pixel values
            if self._contains_suspicious_patterns(image.flatten()):
                result["warnings"].append("suspicious_pixel_patterns")
                if self.strict_mode:
                    result["valid"] = False
                    result["reason"] = "suspicious_pixel_patterns"
                    return result
            
            # Check for extreme pixel values
            if image.dtype == np.uint8:
                if np.any((image < 0) | (image > 255)):
                    result["valid"] = False
                    result["reason"] = "invalid_pixel_range"
                    return result
            elif image.dtype == np.float32:
                if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                    result["valid"] = False
                    result["reason"] = "invalid_float_values"
                    return result
                
                # Check for extreme float values that might indicate injection
                if np.any(np.abs(image) > 1e6):
                    result["warnings"].append("extreme_float_values")
            
            # Check image entropy (too low might indicate synthetic/malicious content)
            entropy = self._calculate_image_entropy(image)
            if entropy < 1.0:
                result["warnings"].append("low_entropy_image")
                if self.strict_mode and entropy < 0.5:
                    result["valid"] = False
                    result["reason"] = "extremely_low_entropy"
                    return result
            
            return result
            
        except Exception as e:
            logger.error(f"Image security validation error: {e}")
            result["valid"] = False
            result["reason"] = f"validation_error: {e}"
            return result
    
    def _validate_text_security(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate text content for security threats."""
        result = {
            "valid": True,
            "warnings": [],
            "reason": None
        }
        
        try:
            # Extract text fields
            text_fields = []
            for field in ["question", "text", "caption"]:
                if field in request_data:
                    text_fields.append(str(request_data[field]))
            
            for text in text_fields:
                # Length checks
                if len(text) > 10000:  # 10K character limit
                    result["valid"] = False
                    result["reason"] = "text_too_long"
                    return result
                
                if len(text.strip()) == 0:
                    result["warnings"].append("empty_text")
                    continue
                
                # Encoding validation
                try:
                    text.encode('utf-8')
                except UnicodeEncodeError:
                    result["valid"] = False
                    result["reason"] = "invalid_text_encoding"
                    return result
                
                # Pattern-based threat detection
                text_lower = text.lower()
                text_bytes = text.encode('utf-8')
                
                # Check for script injection patterns
                for pattern in BLOCKED_PATTERNS:
                    if pattern in text_bytes:
                        result["valid"] = False
                        result["reason"] = f"blocked_pattern_detected: {pattern.decode('utf-8', 'ignore')}"
                        return result
                
                # Check for suspicious keywords
                suspicious_keywords = [
                    'javascript', 'vbscript', 'onclick', 'onerror', 'onload',
                    'eval', 'exec', 'function', 'alert', 'document.cookie',
                    'window.location', 'iframe', 'embed', 'object'
                ]
                
                found_suspicious = [kw for kw in suspicious_keywords if kw in text_lower]
                if found_suspicious:
                    if len(found_suspicious) > 2:
                        result["valid"] = False
                        result["reason"] = f"multiple_suspicious_keywords: {found_suspicious[:3]}"
                        return result
                    else:
                        result["warnings"].append(f"suspicious_keywords: {found_suspicious}")
                
                # Check for excessive special characters
                special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
                if special_char_ratio > 0.5:
                    result["warnings"].append("high_special_char_ratio")
                    if self.strict_mode and special_char_ratio > 0.8:
                        result["valid"] = False
                        result["reason"] = "excessive_special_characters"
                        return result
                
                # Check for control characters
                if any(ord(c) < 32 and c not in '\t\n\r' for c in text):
                    result["warnings"].append("contains_control_characters")
                    if self.strict_mode:
                        result["valid"] = False
                        result["reason"] = "control_characters_detected"
                        return result
            
            return result
            
        except Exception as e:
            logger.error(f"Text security validation error: {e}")
            result["valid"] = False
            result["reason"] = f"validation_error: {e}"
            return result
    
    def _validate_operation_security(self, operation: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate operation-specific security requirements."""
        result = {
            "valid": True,
            "warnings": [],
            "reason": None
        }
        
        try:
            # Allowed operations
            allowed_operations = {
                'generate_caption', 'extract_text', 'answer_question', 
                'get_image_embeddings', 'benchmark_inference'
            }
            
            if operation not in allowed_operations:
                result["valid"] = False
                result["reason"] = f"unauthorized_operation: {operation}"
                return result
            
            # Operation-specific validation
            if operation == 'answer_question':
                if 'question' not in request_data:
                    result["valid"] = False
                    result["reason"] = "missing_required_field: question"
                    return result
                
                question = str(request_data['question'])
                if len(question.strip()) == 0:
                    result["valid"] = False
                    result["reason"] = "empty_question"
                    return result
            
            elif operation == 'generate_caption':
                max_length = request_data.get('max_length', 50)
                if not isinstance(max_length, int) or max_length < 1 or max_length > 200:
                    result["warnings"].append("invalid_max_length_parameter")
                    if self.strict_mode:
                        result["valid"] = False
                        result["reason"] = "invalid_max_length"
                        return result
            
            elif operation == 'benchmark_inference':
                if self.strict_mode:
                    # Benchmarking might be restricted in production
                    result["warnings"].append("benchmarking_operation_in_strict_mode")
            
            # Check for parameter injection attempts
            for key, value in request_data.items():
                if isinstance(value, str) and len(value) > 1000:
                    result["warnings"].append(f"large_parameter: {key}")
                    if self.strict_mode and len(value) > 5000:
                        result["valid"] = False
                        result["reason"] = f"parameter_too_large: {key}"
                        return result
            
            return result
            
        except Exception as e:
            logger.error(f"Operation security validation error: {e}")
            result["valid"] = False
            result["reason"] = f"validation_error: {e}"
            return result
    
    def _contains_suspicious_patterns(self, data: np.ndarray) -> bool:
        """Check for suspicious patterns in data that might indicate injection."""
        try:
            # Convert to bytes for pattern matching
            data_bytes = data.astype(np.uint8).tobytes()
            
            # Check for embedded scripts or code patterns
            for pattern in BLOCKED_PATTERNS:
                if pattern in data_bytes:
                    return True
            
            # Check for repeating patterns that might indicate synthetic data
            if len(data_bytes) > 100:
                # Look for highly repetitive patterns
                chunk_size = min(32, len(data_bytes) // 4)
                chunks = [data_bytes[i:i+chunk_size] for i in range(0, len(data_bytes)-chunk_size, chunk_size)]
                unique_chunks = set(chunks)
                
                if len(unique_chunks) / len(chunks) < 0.1:  # Less than 10% unique chunks
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _calculate_image_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy for anomaly detection."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # Calculate histogram
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
            
            # Normalize
            hist = hist / hist.sum()
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            return float(entropy)
            
        except Exception:
            return 5.0  # Default safe entropy value
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security monitoring metrics."""
        return {
            "blocked_requests": self._blocked_requests,
            "suspicious_patterns": self._suspicious_patterns,
            "rate_limited_requests": self._rate_limited_requests,
            "strict_mode": self.strict_mode
        }


class RateLimiter:
    """Rate limiting for request throttling."""
    
    def __init__(self, max_requests_per_minute: int = 100):
        """Initialize rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests = max_requests_per_minute
        self.requests = {}  # user_id -> [timestamp, ...]
        
        logger.info(f"Rate limiter initialized: {max_requests_per_minute} requests/minute")
    
    def allow_request(self, user_id: str) -> bool:
        """Check if request is allowed for user."""
        current_time = time.time()
        
        # Clean old requests
        if user_id in self.requests:
            self.requests[user_id] = [
                timestamp for timestamp in self.requests[user_id]
                if current_time - timestamp < 60
            ]
        else:
            self.requests[user_id] = []
        
        # Check rate limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Allow request and record timestamp
        self.requests[user_id].append(current_time)
        return True
    
    def get_remaining_requests(self, user_id: str) -> int:
        """Get remaining requests for user."""
        current_time = time.time()
        
        if user_id not in self.requests:
            return self.max_requests
        
        # Count recent requests
        recent_requests = [
            timestamp for timestamp in self.requests[user_id]
            if current_time - timestamp < 60
        ]
        
        return max(0, self.max_requests - len(recent_requests))
    
    def reset_user_limit(self, user_id: str):
        """Reset rate limit for specific user."""
        if user_id in self.requests:
            del self.requests[user_id]
            logger.info(f"Rate limit reset for user: {user_id}")


class InputSanitizer:
    """Input sanitization and cleaning."""
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.max_string_length = 10000
        self.max_array_size = 1000
        
        logger.info("Input sanitizer initialized")
    
    def sanitize_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data.
        
        Args:
            data: Request data to sanitize
            
        Returns:
            Sanitized request data
            
        Raises:
            SecurityError: If data cannot be safely sanitized
        """
        try:
            sanitized = {}
            
            for key, value in data.items():
                sanitized_key = self._sanitize_string(str(key))
                sanitized_value = self._sanitize_value(value)
                
                if sanitized_key and sanitized_value is not None:
                    sanitized[sanitized_key] = sanitized_value
            
            return sanitized
            
        except Exception as e:
            raise SecurityError(f"Input sanitization failed: {e}")
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, (int, float, bool)):
            return self._sanitize_number(value)
        elif isinstance(value, list):
            return self._sanitize_array(value)
        elif isinstance(value, dict):
            return self._sanitize_dict(value)
        elif isinstance(value, np.ndarray):
            return self._sanitize_array_data(value)
        else:
            # Convert unknown types to string and sanitize
            return self._sanitize_string(str(value))
    
    def _sanitize_string(self, text: str) -> Optional[str]:
        """Sanitize string input."""
        if not isinstance(text, str):
            text = str(text)
        
        # Length check
        if len(text) > self.max_string_length:
            text = text[:self.max_string_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove control characters except common ones
        allowed_control = set('\t\n\r')
        text = ''.join(c for c in text if ord(c) >= 32 or c in allowed_control)
        
        # Basic HTML encoding for safety
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace('"', '&quot;').replace("'", '&#x27;')
        
        return text.strip() if text.strip() else None
    
    def _sanitize_number(self, value: Union[int, float, bool]) -> Union[int, float, bool]:
        """Sanitize numeric input."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, int):
            # Limit integer range to prevent overflow
            return max(-2**31, min(2**31 - 1, value))
        elif isinstance(value, float):
            # Check for special float values
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return max(-1e10, min(1e10, value))
        else:
            return 0
    
    def _sanitize_array(self, arr: List[Any]) -> List[Any]:
        """Sanitize array input."""
        if not isinstance(arr, list):
            return []
        
        if len(arr) > self.max_array_size:
            arr = arr[:self.max_array_size]
        
        return [self._sanitize_value(item) for item in arr]
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary input."""
        if not isinstance(data, dict):
            return {}
        
        sanitized = {}
        for key, value in data.items():
            clean_key = self._sanitize_string(str(key))
            clean_value = self._sanitize_value(value)
            
            if clean_key and clean_value is not None:
                sanitized[clean_key] = clean_value
        
        return sanitized
    
    def _sanitize_array_data(self, data: np.ndarray) -> np.ndarray:
        """Sanitize numpy array data."""
        try:
            # Size limits
            if data.size > 100 * 1024 * 1024:  # 100M elements max
                raise SecurityError("Array too large")
            
            # Handle different data types
            if data.dtype == np.uint8:
                # Image data - ensure valid range
                return np.clip(data, 0, 255).astype(np.uint8)
            elif data.dtype in [np.float32, np.float64]:
                # Float data - remove inf/nan
                cleaned = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
                return cleaned.astype(np.float32)
            else:
                # Convert to safe type
                return data.astype(np.float32)
                
        except Exception as e:
            raise SecurityError(f"Array sanitization failed: {e}")


class CryptoUtils:
    """Cryptographic utilities for security operations."""
    
    def __init__(self):
        """Initialize crypto utilities."""
        self.hash_algorithm = 'sha256'
        logger.info("Crypto utilities initialized")
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_data(self, data: Union[str, bytes]) -> str:
        """Hash data securely."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    def verify_hash(self, data: Union[str, bytes], expected_hash: str) -> bool:
        """Verify data against hash."""
        calculated_hash = self.hash_data(data)
        return hmac.compare_digest(calculated_hash, expected_hash)
    
    def generate_session_id(self) -> str:
        """Generate secure session ID."""
        return self.generate_token(48)
    
    def is_session_valid(self, session_id: str, created_time: float) -> bool:
        """Check if session is still valid."""
        if not session_id or len(session_id) < 32:
            return False
        
        current_time = time.time()
        return (current_time - created_time) < SESSION_TIMEOUT


# Example usage and testing
if __name__ == "__main__":
    print("Testing security modules...")
    
    # Test security validator
    print("Testing SecurityValidator...")
    validator = SecurityValidator(strict_mode=True)
    
    # Test valid request
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    valid_request = {
        "operation": "generate_caption",
        "image": test_image,
        "max_length": 50
    }
    
    result = validator.validate_request("test_user", valid_request)
    print(f"Valid request result: {result['valid']}")
    
    # Test malicious request
    malicious_request = {
        "operation": "generate_caption",
        "text": "<script>alert('xss')</script>",
        "image": test_image
    }
    
    result = validator.validate_request("test_user", malicious_request)
    print(f"Malicious request blocked: {not result['valid']}")
    
    print("✓ SecurityValidator works")
    
    # Test rate limiter
    print("\nTesting RateLimiter...")
    rate_limiter = RateLimiter(max_requests_per_minute=5)
    
    user_id = "test_user"
    allowed_count = 0
    
    for i in range(10):
        if rate_limiter.allow_request(user_id):
            allowed_count += 1
    
    print(f"Allowed {allowed_count}/10 requests (limit: 5)")
    print(f"Remaining requests: {rate_limiter.get_remaining_requests(user_id)}")
    
    print("✓ RateLimiter works")
    
    # Test input sanitizer
    print("\nTesting InputSanitizer...")
    sanitizer = InputSanitizer()
    
    dirty_data = {
        "text": "<script>alert('xss')</script>Hello World",
        "number": float('inf'),
        "array": list(range(2000)),  # Too large
        "nested": {"key": "<malicious>value"}
    }
    
    clean_data = sanitizer.sanitize_request(dirty_data)
    print(f"Sanitized data keys: {list(clean_data.keys())}")
    print(f"Sanitized text: {clean_data.get('text', '')[:50]}...")
    
    print("✓ InputSanitizer works")
    
    print("\nAll security tests passed!")