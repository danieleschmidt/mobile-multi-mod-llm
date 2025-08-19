"""Enhanced security module for mobile multi-modal LLM with comprehensive validation."""

import hashlib
import hmac
import logging
import re
import secrets
import time
import threading
from typing import Any, Dict, List, Optional, Union

# Enhanced logging setup
logger = logging.getLogger(__name__)
security_logger = logging.getLogger(f"{__name__}.security")

# Security constants
MAX_REQUEST_SIZE_MB = 50
MAX_TEXT_LENGTH = 1000
MAX_IMAGE_DIMENSION = 4096
BLOCKED_PATTERNS = [
    r'<script[^>]*>.*?</script>',
    r'javascript:',
    r'on\w+\s*=',
    r'eval\s*\(',
    r'exec\s*\(',
]

class SecurityError(Exception):
    """Security validation error."""
    pass

class SecurityValidator:
    """Comprehensive security validation for mobile multi-modal operations."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.blocked_requests = 0
        self.suspicious_patterns = 0
        self.rate_limiter = RateLimiter()
        self.input_sanitizer = InputSanitizer()
        self.crypto_utils = CryptoUtils()
        self.security_monitor = SecurityMonitor()
        
        security_logger.info(f"SecurityValidator initialized (strict_mode={strict_mode})")
    
    def validate_request(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming request for security compliance."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "blocked_reason": None,
            "checks": {}
        }
        
        try:
            # Rate limiting check
            if not self.rate_limiter.allow_request(user_id):
                validation_result["valid"] = False
                validation_result["blocked_reason"] = "rate_limit_exceeded"
                self.security_monitor.record_security_event("rate_limit_exceeded", {"user_id": user_id})
                return validation_result
            
            # Input size validation
            if not self._validate_request_size(request_data):
                validation_result["valid"] = False
                validation_result["blocked_reason"] = "request_size_exceeded"
                return validation_result
            
            # Content validation
            if "image" in request_data:
                image_result = self._validate_image_security(request_data["image"])
                validation_result["checks"]["image_security"] = image_result["valid"]
                if not image_result["valid"]:
                    validation_result["valid"] = False
                    validation_result["blocked_reason"] = f"image_security: {image_result['reason']}"
                    return validation_result
            
            # Text validation  
            text_fields = ["question", "text", "caption"]
            for field in text_fields:
                if field in request_data:
                    text_result = self._validate_text_security(request_data[field])
                    validation_result["checks"][f"{field}_security"] = text_result["valid"]
                    if not text_result["valid"]:
                        validation_result["valid"] = False
                        validation_result["blocked_reason"] = f"text_security: {text_result['reason']}"
                        return validation_result
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            validation_result["valid"] = False
            validation_result["blocked_reason"] = "validation_error"
            return validation_result
    
    def _validate_request_size(self, request_data: Dict[str, Any]) -> bool:
        """Validate request size limits."""
        try:
            # Estimate request size
            size_estimate = len(str(request_data))
            if "image" in request_data and hasattr(request_data["image"], "nbytes"):
                size_estimate += request_data["image"].nbytes
            
            return size_estimate < MAX_REQUEST_SIZE_MB * 1024 * 1024
        except:
            return False
    
    def _validate_image_security(self, image) -> Dict[str, Any]:
        """Validate image security."""
        try:
            if image is None:
                return {"valid": False, "reason": "null_image"}
            
            # Check if it's a numpy array
            if hasattr(image, "shape"):
                h, w = image.shape[:2]
                if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
                    return {"valid": False, "reason": "image_too_large"}
                
                if h < 8 or w < 8:
                    return {"valid": False, "reason": "image_too_small"}
            
            return {"valid": True, "reason": None}
            
        except Exception as e:
            return {"valid": False, "reason": f"validation_error: {e}"}
    
    def _validate_text_security(self, text: str) -> Dict[str, Any]:
        """Validate text input security."""
        try:
            if not isinstance(text, str):
                return {"valid": False, "reason": "invalid_text_type"}
            
            if len(text) > MAX_TEXT_LENGTH:
                return {"valid": False, "reason": "text_too_long"}
            
            # Check for malicious patterns
            for pattern in BLOCKED_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    return {"valid": False, "reason": "malicious_pattern_detected"}
            
            return {"valid": True, "reason": None}
            
        except Exception as e:
            return {"valid": False, "reason": f"validation_error: {e}"}


class RateLimiter:
    """Rate limiting for request throttling."""
    
    def __init__(self, max_requests_per_minute: int = 100):
        self.max_requests = max_requests_per_minute
        self.requests = {}  # user_id -> [timestamp, ...]
        self.lock = threading.Lock()
    
    def allow_request(self, user_id: str) -> bool:
        """Check if request is allowed for user."""
        with self.lock:
            current_time = time.time()
            
            # Clean old requests
            if user_id in self.requests:
                self.requests[user_id] = [
                    timestamp for timestamp in self.requests[user_id]
                    if current_time - timestamp < 60
                ]
            else:
                self.requests[user_id] = []
            
            # Check if under limit
            if len(self.requests[user_id]) >= self.max_requests:
                return False
            
            # Add current request
            self.requests[user_id].append(current_time)
            return True


class InputSanitizer:
    """Input sanitization utilities."""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input."""
        if not isinstance(text, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&"\']', '', text)
        
        # Limit length
        return sanitized[:MAX_TEXT_LENGTH]
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename for security."""
        if not isinstance(filename, str):
            return False
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check for reasonable length and characters
        if len(filename) > 255 or not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            return False
        
        return True


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


class SecurityMonitor:
    """Security event monitoring and alerting."""
    
    def __init__(self):
        self.events = []
        self.alert_thresholds = {
            "blocked_requests_per_minute": 10,
            "rate_limit_violations_per_minute": 20,
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
        self._check_alerts()
    
    def _check_alerts(self):
        """Check if events trigger security alerts."""
        current_time = time.time()
        
        # Check blocked requests in last minute
        recent_blocked = [
            e for e in self.events 
            if (e["type"] == "blocked_request" and 
                current_time - e["timestamp"] < 60)
        ]
        
        if len(recent_blocked) >= self.alert_thresholds["blocked_requests_per_minute"]:
            security_logger.critical(f"SECURITY ALERT: High rate of blocked requests: {len(recent_blocked)}/min")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security event summary."""
        if not self.events:
            return {"total_events": 0, "event_types": {}}
        
        event_types = {}
        for event in self.events:
            event_type = event["type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.events),
            "event_types": event_types,
        }


if __name__ == "__main__":
    # Test security module
    print("Testing enhanced security module...")
    
    validator = SecurityValidator(strict_mode=True)
    
    # Test valid request
    test_request = {
        "operation": "generate_caption",
        "text": "What is in this image?"
    }
    
    result = validator.validate_request("test_user", test_request)
    print(f"Valid request result: {result}")
    
    # Test invalid request
    malicious_request = {
        "text": "<script>alert('xss')</script>",
        "operation": "generate_caption"
    }
    
    result = validator.validate_request("test_user", malicious_request)
    print(f"Malicious request result: {result}")
    
    print("✅ Enhanced security module working correctly!")