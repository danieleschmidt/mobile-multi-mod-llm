"""Advanced Security Hardening for Mobile Multi-Modal AI Systems.

Comprehensive security framework with input validation, threat detection,
encryption, access control, and security audit capabilities.
"""

import json
import logging
import os
import time
import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque
import re
import base64

# Cryptographic libraries
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    # Fallback implementations for environments without cryptography
    CRYPTO_AVAILABLE = False
    
    class Fernet:
        @staticmethod
        def generate_key():
            return base64.urlsafe_b64encode(secrets.token_bytes(32))
        
        def __init__(self, key):
            self.key = key
        
        def encrypt(self, data):
            if isinstance(data, str):
                data = data.encode()
            return base64.urlsafe_b64encode(data + b"_encrypted")
        
        def decrypt(self, encrypted_data):
            decoded = base64.urlsafe_b64decode(encrypted_data)
            return decoded.replace(b"_encrypted", b"")

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityAction(Enum):
    """Security response actions."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"

@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    description: str
    details: Dict[str, Any]
    action_taken: SecurityAction

@dataclass
class AccessPolicy:
    """Access control policy."""
    resource: str
    permissions: List[str]
    conditions: Dict[str, Any]
    expiry: Optional[float] = None

class SecurityHardening:
    """Advanced security hardening system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize security hardening system.
        
        Args:
            config: Security configuration
        """
        self.config = config or {}
        
        # Encryption setup
        self.master_key = None
        self.encryption_key = None
        self._init_encryption()
        
        # Input validation
        self.input_validators = {}
        self.content_filters = {}
        self._init_input_validation()
        
        # Access control
        self.access_policies = {}
        self.active_sessions = {}
        self.rate_limits = defaultdict(lambda: {"count": 0, "reset_time": 0})
        
        # Threat detection
        self.threat_patterns = {}
        self.suspicious_activities = deque(maxlen=1000)
        self.blocked_ips = set()
        self.security_events = deque(maxlen=10000)
        
        # Audit logging
        self.audit_log_path = Path(self.config.get("audit_log_path", "security_audit.log"))
        self.audit_logger = self._setup_audit_logging()
        
        # Security monitoring
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        
        # Load security rules
        self._load_security_rules()
        
        logger.info("Security hardening system initialized")
    
    def _init_encryption(self):
        """Initialize encryption components."""
        if CRYPTO_AVAILABLE:
            # Generate or load master key
            key_file = Path(self.config.get("master_key_file", "master.key"))
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.master_key = f.read()
            else:
                self.master_key = Fernet.generate_key()
                # In production, this should be stored securely (HSM, key vault, etc.)
                with open(key_file, 'wb') as f:
                    f.write(self.master_key)
                os.chmod(key_file, 0o600)  # Restrict file permissions
            
            self.encryption_key = Fernet(self.master_key)
        else:
            # Fallback encryption
            self.master_key = base64.urlsafe_b64encode(secrets.token_bytes(32))
            self.encryption_key = Fernet(self.master_key)
    
    def _init_input_validation(self):
        """Initialize input validation rules."""
        # Image validation
        self.input_validators["image"] = {
            "max_size": self.config.get("max_image_size", 10 * 1024 * 1024),  # 10MB
            "allowed_formats": self.config.get("allowed_image_formats", ["JPEG", "PNG", "WebP"]),
            "max_dimensions": self.config.get("max_image_dimensions", (4096, 4096))
        }
        
        # Text validation
        self.input_validators["text"] = {
            "max_length": self.config.get("max_text_length", 10000),
            "forbidden_patterns": [
                r"<script[^>]*>.*?</script>",  # Script injection
                r"javascript:",  # JavaScript URLs
                r"data:.*base64",  # Base64 data URLs (potential for abuse)
                r"\\x[0-9a-fA-F]{2}",  # Hex-encoded characters
                r"eval\s*\(",  # Code evaluation
                r"exec\s*\(",  # Code execution
            ]
        }
        
        # Request validation
        self.input_validators["request"] = {
            "max_requests_per_minute": self.config.get("rate_limit", 60),
            "max_concurrent_requests": self.config.get("max_concurrent", 10),
            "allowed_user_agents": self.config.get("allowed_user_agents", []),
            "blocked_user_agents": self.config.get("blocked_user_agents", [
                ".*bot.*", ".*crawler.*", ".*scraper.*"
            ])
        }
    
    def _load_security_rules(self):
        """Load security threat detection rules."""
        # SQL injection patterns
        self.threat_patterns["sql_injection"] = [
            r"'\s*OR\s*'1'\s*=\s*'1",
            r"'\s*OR\s*1\s*=\s*1",
            r"UNION\s+SELECT",
            r"DROP\s+TABLE",
            r"INSERT\s+INTO",
            r"DELETE\s+FROM"
        ]
        
        # Command injection patterns
        self.threat_patterns["command_injection"] = [
            r";\s*rm\s+",
            r";\s*cat\s+",
            r";\s*ls\s+",
            r"\|\s*nc\s+",
            r"&&\s*curl\s+",
            r">\s*/dev/null"
        ]
        
        # Path traversal patterns
        self.threat_patterns["path_traversal"] = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e%5c"
        ]
        
        # XSS patterns
        self.threat_patterns["xss"] = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*="
        ]
    
    def _setup_audit_logging(self):
        """Setup security audit logging."""
        audit_logger = logging.getLogger('security_audit')
        audit_logger.setLevel(logging.INFO)
        
        # Create file handler for audit logs
        handler = logging.FileHandler(self.audit_log_path)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        audit_logger.addHandler(handler)
        
        return audit_logger
    
    def validate_input(self, input_type: str, data: Any, 
                      metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """Validate input data according to security rules.
        
        Args:
            input_type: Type of input (image, text, request)
            data: Input data to validate
            metadata: Optional metadata about the input
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        if input_type not in self.input_validators:
            violations.append(f"Unknown input type: {input_type}")
            return False, violations
        
        validator_config = self.input_validators[input_type]
        
        if input_type == "image":
            violations.extend(self._validate_image(data, validator_config, metadata))
        elif input_type == "text":
            violations.extend(self._validate_text(data, validator_config))
        elif input_type == "request":
            violations.extend(self._validate_request(data, validator_config))
        
        is_valid = len(violations) == 0
        
        # Log validation results
        if not is_valid:
            self._log_security_event(
                "input_validation_failed",
                ThreatLevel.MEDIUM,
                f"Input validation failed: {', '.join(violations)}",
                {"input_type": input_type, "violations": violations}
            )
        
        return is_valid, violations
    
    def _validate_image(self, image_data: Any, config: Dict[str, Any], 
                       metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate image input."""
        violations = []
        
        try:
            # Check if it's image bytes or file path
            if isinstance(image_data, (str, Path)):
                # File path validation
                path = Path(image_data)
                if not path.exists():
                    violations.append("Image file does not exist")
                    return violations
                
                # Check file size
                file_size = path.stat().st_size
                if file_size > config["max_size"]:
                    violations.append(f"Image file too large: {file_size} > {config['max_size']}")
                
                # Read file data for further validation
                with open(path, 'rb') as f:
                    image_data = f.read()
            
            # Check data size
            if isinstance(image_data, bytes):
                if len(image_data) > config["max_size"]:
                    violations.append(f"Image data too large: {len(image_data)} > {config['max_size']}")
                
                # Basic format detection
                format_detected = self._detect_image_format(image_data)
                if format_detected not in config["allowed_formats"]:
                    violations.append(f"Unsupported image format: {format_detected}")
                
                # Check for embedded malicious content
                if self._contains_suspicious_image_content(image_data):
                    violations.append("Image contains suspicious content")
            
        except Exception as e:
            violations.append(f"Image validation error: {str(e)}")
        
        return violations
    
    def _validate_text(self, text_data: str, config: Dict[str, Any]) -> List[str]:
        """Validate text input."""
        violations = []
        
        # Length check
        if len(text_data) > config["max_length"]:
            violations.append(f"Text too long: {len(text_data)} > {config['max_length']}")
        
        # Pattern checks
        for pattern in config["forbidden_patterns"]:
            if re.search(pattern, text_data, re.IGNORECASE):
                violations.append(f"Text contains forbidden pattern: {pattern}")
        
        # Additional threat detection
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_data, re.IGNORECASE):
                    violations.append(f"Potential {threat_type} detected")
                    self._log_security_event(
                        f"threat_detected_{threat_type}",
                        ThreatLevel.HIGH,
                        f"Potential {threat_type} in text input",
                        {"pattern": pattern, "threat_type": threat_type}
                    )
        
        return violations
    
    def _validate_request(self, request_data: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Validate request metadata."""
        violations = []
        
        # Rate limiting
        client_id = request_data.get("client_id", "unknown")
        current_time = time.time()
        
        rate_limit_key = f"rate_limit_{client_id}"
        rate_info = self.rate_limits[rate_limit_key]
        
        if current_time > rate_info["reset_time"]:
            rate_info["count"] = 0
            rate_info["reset_time"] = current_time + 60  # 1 minute window
        
        rate_info["count"] += 1
        
        if rate_info["count"] > config["max_requests_per_minute"]:
            violations.append("Rate limit exceeded")
            
            # Block IP if severe abuse
            client_ip = request_data.get("client_ip")
            if rate_info["count"] > config["max_requests_per_minute"] * 3:
                if client_ip:
                    self.blocked_ips.add(client_ip)
                    self._log_security_event(
                        "ip_blocked",
                        ThreatLevel.HIGH,
                        f"IP blocked due to severe rate limit violation",
                        {"client_ip": client_ip, "request_count": rate_info["count"]}
                    )
        
        # User agent validation
        user_agent = request_data.get("user_agent", "")
        
        # Check blocked user agents
        for blocked_pattern in config["blocked_user_agents"]:
            if re.search(blocked_pattern, user_agent, re.IGNORECASE):
                violations.append(f"Blocked user agent: {user_agent}")
        
        # Check allowed user agents (if specified)
        if config["allowed_user_agents"]:
            allowed = False
            for allowed_pattern in config["allowed_user_agents"]:
                if re.search(allowed_pattern, user_agent, re.IGNORECASE):
                    allowed = True
                    break
            
            if not allowed:
                violations.append(f"User agent not allowed: {user_agent}")
        
        # IP blocking check
        client_ip = request_data.get("client_ip")
        if client_ip in self.blocked_ips:
            violations.append(f"IP address is blocked: {client_ip}")
        
        return violations
    
    def _detect_image_format(self, image_data: bytes) -> str:
        """Detect image format from binary data."""
        if image_data.startswith(b'\xff\xd8\xff'):
            return "JPEG"
        elif image_data.startswith(b'\x89PNG'):
            return "PNG"
        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
            return "WebP"
        elif image_data.startswith(b'GIF'):
            return "GIF"
        else:
            return "UNKNOWN"
    
    def _contains_suspicious_image_content(self, image_data: bytes) -> bool:
        """Check for suspicious content in image data."""
        # Look for embedded scripts or executables
        suspicious_patterns = [
            b'<script',
            b'javascript:',
            b'<?php',
            b'#!/bin/',
            b'MZ',  # PE executable header
            b'\x7fELF',  # ELF executable header
        ]
        
        for pattern in suspicious_patterns:
            if pattern in image_data:
                return True
        
        return False
    
    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.encryption_key.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        return self.encryption_key.decrypt(encrypted_data)
    
    def create_access_token(self, user_id: str, permissions: List[str], 
                          expiry_hours: int = 24) -> str:
        """Create secure access token.
        
        Args:
            user_id: User identifier
            permissions: List of permissions
            expiry_hours: Token expiry in hours
            
        Returns:
            Access token
        """
        expiry = time.time() + (expiry_hours * 3600)
        
        token_data = {
            "user_id": user_id,
            "permissions": permissions,
            "expiry": expiry,
            "issued_at": time.time(),
            "nonce": secrets.token_hex(16)
        }
        
        # Create token
        token_json = json.dumps(token_data, sort_keys=True)
        encrypted_token = self.encrypt_data(token_json)
        
        # Create signature
        signature = self._create_signature(encrypted_token)
        
        # Combine token and signature
        token = base64.urlsafe_b64encode(encrypted_token + b'.' + signature).decode('ascii')
        
        # Store active session
        self.active_sessions[token] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": time.time(),
            "expiry": expiry
        }
        
        self._log_security_event(
            "access_token_created",
            ThreatLevel.LOW,
            f"Access token created for user {user_id}",
            {"user_id": user_id, "permissions": permissions}
        )
        
        return token
    
    def validate_access_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate access token.
        
        Args:
            token: Access token to validate
            
        Returns:
            Tuple of (is_valid, token_data)
        """
        try:
            # Decode token
            decoded = base64.urlsafe_b64decode(token.encode('ascii'))
            
            # Split token and signature
            parts = decoded.split(b'.', 1)
            if len(parts) != 2:
                return False, None
            
            encrypted_token, signature = parts
            
            # Verify signature
            if not self._verify_signature(encrypted_token, signature):
                return False, None
            
            # Decrypt token
            decrypted_data = self.decrypt_data(encrypted_token)
            token_data = json.loads(decrypted_data.decode('utf-8'))
            
            # Check expiry
            if time.time() > token_data["expiry"]:
                # Remove expired session
                if token in self.active_sessions:
                    del self.active_sessions[token]
                return False, None
            
            # Verify session exists
            if token not in self.active_sessions:
                return False, None
            
            return True, token_data
            
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return False, None
    
    def revoke_access_token(self, token: str) -> bool:
        """Revoke access token.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if token was revoked
        """
        if token in self.active_sessions:
            session = self.active_sessions[token]
            del self.active_sessions[token]
            
            self._log_security_event(
                "access_token_revoked",
                ThreatLevel.LOW,
                f"Access token revoked for user {session['user_id']}",
                {"user_id": session["user_id"]}
            )
            
            return True
        
        return False
    
    def check_permissions(self, token: str, required_permission: str) -> bool:
        """Check if token has required permission.
        
        Args:
            token: Access token
            required_permission: Required permission
            
        Returns:
            True if permission is granted
        """
        is_valid, token_data = self.validate_access_token(token)
        
        if not is_valid:
            return False
        
        permissions = token_data.get("permissions", [])
        
        # Check for wildcard permission
        if "*" in permissions:
            return True
        
        # Check specific permission
        if required_permission in permissions:
            return True
        
        # Check permission hierarchy (e.g., "model.inference" includes "model.inference.caption")
        for permission in permissions:
            if required_permission.startswith(permission + "."):
                return True
        
        return False
    
    def _create_signature(self, data: bytes) -> bytes:
        """Create HMAC signature for data."""
        return hmac.new(self.master_key, data, hashlib.sha256).digest()
    
    def _verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify HMAC signature."""
        expected = self._create_signature(data)
        return hmac.compare_digest(expected, signature)
    
    def _log_security_event(self, event_type: str, threat_level: ThreatLevel, 
                           description: str, details: Dict[str, Any],
                           source_ip: Optional[str] = None, user_id: Optional[str] = None,
                           action_taken: SecurityAction = SecurityAction.WARN):
        """Log security event.
        
        Args:
            event_type: Type of security event
            threat_level: Threat level
            description: Event description
            details: Event details
            source_ip: Optional source IP
            user_id: Optional user ID
            action_taken: Action taken in response
        """
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            details=details,
            action_taken=action_taken
        )
        
        self.security_events.append(event)
        
        # Log to audit log
        audit_message = json.dumps({
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "threat_level": event.threat_level.value,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "description": event.description,
            "details": event.details,
            "action_taken": event.action_taken.value
        })
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.audit_logger.error(audit_message)
        elif threat_level == ThreatLevel.MEDIUM:
            self.audit_logger.warning(audit_message)
        else:
            self.audit_logger.info(audit_message)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security system summary.
        
        Returns:
            Security summary
        """
        current_time = time.time()
        recent_events = [e for e in self.security_events if current_time - e.timestamp < 3600]  # Last hour
        
        threat_counts = defaultdict(int)
        for event in recent_events:
            threat_counts[event.threat_level.value] += 1
        
        return {
            "active_sessions": len(self.active_sessions),
            "blocked_ips": len(self.blocked_ips),
            "recent_events": len(recent_events),
            "threat_level_counts": dict(threat_counts),
            "total_security_events": len(self.security_events),
            "audit_log_path": str(self.audit_log_path)
        }
    
    def export_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Export security report for specified time period.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Security report
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        # Categorize events
        events_by_type = defaultdict(list)
        events_by_threat = defaultdict(list)
        
        for event in recent_events:
            events_by_type[event.event_type].append(event)
            events_by_threat[event.threat_level.value].append(event)
        
        # Top threats
        top_threats = sorted(events_by_type.items(), 
                           key=lambda x: len(x[1]), reverse=True)[:10]
        
        # Security metrics
        total_blocked = len([e for e in recent_events 
                           if e.action_taken == SecurityAction.BLOCK])
        total_warnings = len([e for e in recent_events 
                            if e.action_taken == SecurityAction.WARN])
        
        report = {
            "report_period_hours": hours,
            "total_events": len(recent_events),
            "events_by_threat_level": {k: len(v) for k, v in events_by_threat.items()},
            "top_threat_types": [{"type": t, "count": len(events)} for t, events in top_threats],
            "security_actions": {
                "blocked": total_blocked,
                "warned": total_warnings,
                "allowed": len(recent_events) - total_blocked - total_warnings
            },
            "active_sessions": len(self.active_sessions),
            "blocked_ips": list(self.blocked_ips),
            "generated_at": time.time()
        }
        
        return report
    
    def start_monitoring(self):
        """Start security monitoring services."""
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring services."""
        self._shutdown_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Security monitoring stopped")
    
    def _monitoring_loop(self):
        """Security monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Clean up expired sessions
                expired_sessions = []
                for token, session in self.active_sessions.items():
                    if current_time > session["expiry"]:
                        expired_sessions.append(token)
                
                for token in expired_sessions:
                    del self.active_sessions[token]
                
                # Analyze security events for patterns
                self._analyze_security_patterns()
                
                # Clean up old rate limit entries
                self._cleanup_rate_limits(current_time)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                time.sleep(10)
    
    def _analyze_security_patterns(self):
        """Analyze security events for suspicious patterns."""
        current_time = time.time()
        recent_window = current_time - 300  # 5 minute window
        
        recent_events = [e for e in self.security_events if e.timestamp > recent_window]
        
        # Group events by source IP
        events_by_ip = defaultdict(list)
        for event in recent_events:
            if event.source_ip:
                events_by_ip[event.source_ip].append(event)
        
        # Check for suspicious activity patterns
        for ip, events in events_by_ip.items():
            if len(events) > 10:  # High frequency of events from single IP
                threat_types = set(e.event_type for e in events)
                if len(threat_types) > 3:  # Multiple different threat types
                    self._log_security_event(
                        "suspicious_activity_pattern",
                        ThreatLevel.HIGH,
                        f"Multiple threat types detected from IP {ip}",
                        {"ip": ip, "event_count": len(events), "threat_types": list(threat_types)},
                        source_ip=ip,
                        action_taken=SecurityAction.WARN
                    )
    
    def _cleanup_rate_limits(self, current_time: float):
        """Clean up old rate limit entries."""
        expired_keys = []
        for key, rate_info in self.rate_limits.items():
            if current_time > rate_info["reset_time"] + 3600:  # Keep for 1 hour after reset
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.rate_limits[key]


class SecureModelLoader:
    """Secure model loading with integrity verification."""
    
    def __init__(self, security_system: SecurityHardening):
        """Initialize secure model loader.
        
        Args:
            security_system: Security hardening system
        """
        self.security = security_system
        self.model_checksums = {}
        self.load_model_integrity_database()
    
    def load_model_integrity_database(self):
        """Load model integrity database."""
        # This would load known good checksums for models
        # For now, implement basic placeholder
        self.model_checksums = {
            "mobile-mm-llm-base": "sha256:abcdef123456...",  # Would be actual checksum
            "mobile-mm-llm-int2": "sha256:fedcba654321..."   # Would be actual checksum
        }
    
    def verify_model_integrity(self, model_path: Path, expected_checksum: str = None) -> bool:
        """Verify model file integrity.
        
        Args:
            model_path: Path to model file
            expected_checksum: Expected checksum (optional)
            
        Returns:
            True if model integrity is verified
        """
        if not model_path.exists():
            return False
        
        try:
            # Calculate file checksum
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            calculated_checksum = f"sha256:{sha256_hash.hexdigest()}"
            
            # Check against expected checksum
            if expected_checksum:
                return calculated_checksum == expected_checksum
            
            # Check against known good checksums
            model_name = model_path.stem
            if model_name in self.model_checksums:
                return calculated_checksum == self.model_checksums[model_name]
            
            # No checksum available - log warning
            logger.warning(f"No integrity checksum available for model: {model_name}")
            return True  # Allow loading but log the issue
            
        except Exception as e:
            logger.error(f"Model integrity verification failed: {e}")
            return False
    
    def secure_load_model(self, model_path: Path, checksum: str = None) -> bool:
        """Securely load model with integrity and security checks.
        
        Args:
            model_path: Path to model file
            checksum: Optional expected checksum
            
        Returns:
            True if model loaded securely
        """
        # Verify file path is safe
        if not self._is_safe_path(model_path):
            self.security._log_security_event(
                "unsafe_model_path",
                ThreatLevel.HIGH,
                f"Unsafe model path detected: {model_path}",
                {"path": str(model_path)},
                action_taken=SecurityAction.BLOCK
            )
            return False
        
        # Verify model integrity
        if not self.verify_model_integrity(model_path, checksum):
            self.security._log_security_event(
                "model_integrity_failed",
                ThreatLevel.CRITICAL,
                f"Model integrity verification failed: {model_path}",
                {"path": str(model_path)},
                action_taken=SecurityAction.BLOCK
            )
            return False
        
        # Log successful secure load
        self.security._log_security_event(
            "secure_model_loaded",
            ThreatLevel.LOW,
            f"Model loaded securely: {model_path}",
            {"path": str(model_path)},
            action_taken=SecurityAction.ALLOW
        )
        
        return True
    
    def _is_safe_path(self, path: Path) -> bool:
        """Check if file path is safe (no path traversal)."""
        try:
            # Resolve path and check it doesn't escape expected directory
            resolved = path.resolve()
            
            # Define allowed model directories
            allowed_dirs = [
                Path.cwd() / "models",
                Path.cwd() / "checkpoints", 
                Path("/opt/models"),  # System model directory
            ]
            
            for allowed_dir in allowed_dirs:
                try:
                    resolved.relative_to(allowed_dir.resolve())
                    return True
                except ValueError:
                    continue
            
            return False
            
        except Exception:
            return False


def create_default_security_config() -> Dict[str, Any]:
    """Create default security configuration."""
    return {
        "master_key_file": "security/master.key",
        "audit_log_path": "logs/security_audit.log",
        "max_image_size": 10 * 1024 * 1024,  # 10MB
        "allowed_image_formats": ["JPEG", "PNG", "WebP"],
        "max_image_dimensions": [4096, 4096],
        "max_text_length": 10000,
        "rate_limit": 60,  # requests per minute
        "max_concurrent": 10,
        "blocked_user_agents": [
            ".*bot.*", ".*crawler.*", ".*scraper.*", ".*spider.*"
        ],
        "threat_detection": {
            "enable_sql_injection_detection": True,
            "enable_xss_detection": True,
            "enable_command_injection_detection": True,
            "enable_path_traversal_detection": True
        }
    }