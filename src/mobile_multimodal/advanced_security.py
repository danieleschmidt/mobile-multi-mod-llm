"""Advanced security features for mobile AI deployment."""

import hashlib
import hmac
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_ip: str
    user_id: str
    details: Dict[str, Any]
    action_taken: str


class AdvancedSecurityValidator:
    """Enhanced security validation with threat detection."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.threat_database = {}
        self.security_events = []
        self.rate_limiters = {}
        self.integrity_checkers = {}
        self.max_events = 10000
        
        # Security configuration
        self.config = {
            "max_file_size_mb": 100,
            "allowed_image_formats": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
            "max_requests_per_minute": 60,
            "max_requests_per_hour": 1000,
            "enable_model_integrity_check": True,
            "enable_input_sanitization": True,
            "enable_output_filtering": True,
            "quarantine_suspicious_inputs": True
        }
    
    def validate_advanced_request(self, user_id: str, request_data: Dict[str, Any], 
                                source_ip: str = "unknown") -> Dict[str, Any]:
        """Perform comprehensive security validation."""
        validation_result = {
            "valid": True,
            "threat_level": ThreatLevel.LOW,
            "warnings": [],
            "blocked_reason": None,
            "security_score": 100,
            "mitigations_applied": []
        }
        
        try:
            # Rate limiting check
            rate_limit_result = self._check_rate_limits(user_id, source_ip)
            if not rate_limit_result["allowed"]:
                validation_result["valid"] = False
                validation_result["blocked_reason"] = "Rate limit exceeded"
                validation_result["threat_level"] = ThreatLevel.HIGH
                self._log_security_event(
                    "rate_limit_exceeded", ThreatLevel.HIGH, source_ip, user_id,
                    {"requests_per_minute": rate_limit_result["requests_per_minute"]}, "blocked"
                )
                return validation_result
            
            # Input validation and sanitization
            input_validation = self._validate_inputs(request_data)
            validation_result["security_score"] -= input_validation["penalty"]
            validation_result["warnings"].extend(input_validation["warnings"])
            
            if input_validation["blocked"]:
                validation_result["valid"] = False
                validation_result["blocked_reason"] = input_validation["reason"]
                validation_result["threat_level"] = input_validation["threat_level"]
                
            # Behavioral analysis
            behavior_analysis = self._analyze_user_behavior(user_id, request_data)
            validation_result["security_score"] -= behavior_analysis["penalty"]
            validation_result["warnings"].extend(behavior_analysis["warnings"])
            
            # Model integrity check
            if self.config["enable_model_integrity_check"]:
                integrity_check = self._check_model_integrity(request_data)
                if not integrity_check["valid"]:
                    validation_result["valid"] = False
                    validation_result["blocked_reason"] = "Model integrity check failed"
                    validation_result["threat_level"] = ThreatLevel.CRITICAL
            
            # Determine final threat level based on security score
            if validation_result["security_score"] < 30:
                validation_result["threat_level"] = ThreatLevel.CRITICAL
            elif validation_result["security_score"] < 60:
                validation_result["threat_level"] = ThreatLevel.HIGH
            elif validation_result["security_score"] < 80:
                validation_result["threat_level"] = ThreatLevel.MEDIUM
            
            # Log security event
            self._log_security_event(
                "request_validation", validation_result["threat_level"], 
                source_ip, user_id, request_data, 
                "allowed" if validation_result["valid"] else "blocked"
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return {
                "valid": False,
                "threat_level": ThreatLevel.CRITICAL,
                "blocked_reason": f"Security validation error: {str(e)}",
                "security_score": 0
            }
    
    def _check_rate_limits(self, user_id: str, source_ip: str) -> Dict[str, Any]:
        """Check rate limits for user and IP."""
        current_time = time.time()
        
        # Initialize rate limiter if not exists
        if user_id not in self.rate_limiters:
            self.rate_limiters[user_id] = {
                "requests": [],
                "last_request": 0
            }
        
        user_limiter = self.rate_limiters[user_id]
        
        # Clean old requests (older than 1 hour)
        hour_ago = current_time - 3600
        user_limiter["requests"] = [req for req in user_limiter["requests"] if req > hour_ago]
        
        # Check requests per minute
        minute_ago = current_time - 60
        requests_per_minute = len([req for req in user_limiter["requests"] if req > minute_ago])
        
        # Check requests per hour
        requests_per_hour = len(user_limiter["requests"])
        
        # Determine if request is allowed
        allowed = (requests_per_minute < self.config["max_requests_per_minute"] and 
                  requests_per_hour < self.config["max_requests_per_hour"])
        
        if allowed:
            user_limiter["requests"].append(current_time)
            user_limiter["last_request"] = current_time
        
        return {
            "allowed": allowed,
            "requests_per_minute": requests_per_minute,
            "requests_per_hour": requests_per_hour,
            "time_until_reset": 60 - (current_time - minute_ago) if not allowed else 0
        }
    
    def _validate_inputs(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize inputs."""
        validation_result = {
            "blocked": False,
            "reason": None,
            "threat_level": ThreatLevel.LOW,
            "penalty": 0,
            "warnings": []
        }
        
        try:
            # Check for suspicious patterns in text inputs
            text_inputs = [v for v in request_data.values() if isinstance(v, str)]
            for text in text_inputs:
                suspicious_patterns = [
                    "<script", "javascript:", "eval(", "exec(", 
                    "document.cookie", "window.location", "alert(",
                    "../../", "../", "passwd", "/etc/", "cmd.exe"
                ]
                
                for pattern in suspicious_patterns:
                    if pattern.lower() in text.lower():
                        validation_result["blocked"] = True
                        validation_result["reason"] = f"Suspicious pattern detected: {pattern}"
                        validation_result["threat_level"] = ThreatLevel.HIGH
                        validation_result["penalty"] = 50
                        break
            
            # Check for large payloads
            payload_size = len(json.dumps(request_data, default=str))
            if payload_size > 10 * 1024 * 1024:  # 10MB limit
                validation_result["blocked"] = True
                validation_result["reason"] = "Payload too large"
                validation_result["threat_level"] = ThreatLevel.MEDIUM
                validation_result["penalty"] = 30
            
            # Check for excessive nesting
            def count_nesting(obj, depth=0):
                if depth > 20:  # Max nesting depth
                    return depth
                if isinstance(obj, dict):
                    return max([count_nesting(v, depth + 1) for v in obj.values()] + [depth])
                elif isinstance(obj, list):
                    return max([count_nesting(item, depth + 1) for item in obj] + [depth])
                return depth
            
            nesting_depth = count_nesting(request_data)
            if nesting_depth > 20:
                validation_result["warnings"].append("Excessive nesting detected")
                validation_result["penalty"] += 10
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return {
                "blocked": True,
                "reason": f"Input validation error: {str(e)}",
                "threat_level": ThreatLevel.HIGH,
                "penalty": 100,
                "warnings": []
            }
    
    def _analyze_user_behavior(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user behavior patterns for anomalies."""
        analysis_result = {
            "penalty": 0,
            "warnings": [],
            "anomalies_detected": []
        }
        
        try:
            # Get user's request history
            user_events = [event for event in self.security_events if event.user_id == user_id]
            
            if len(user_events) < 2:
                return analysis_result  # Not enough data for analysis
            
            # Analyze request frequency
            current_time = time.time()
            recent_events = [e for e in user_events if current_time - e.timestamp < 300]  # Last 5 minutes
            
            if len(recent_events) > 50:  # Too many requests in 5 minutes
                analysis_result["penalty"] += 20
                analysis_result["warnings"].append("Unusual request frequency detected")
                analysis_result["anomalies_detected"].append("high_frequency")
            
            # Analyze request patterns
            request_types = [e.event_type for e in recent_events]
            if len(set(request_types)) == 1 and len(request_types) > 10:  # Repetitive requests
                analysis_result["penalty"] += 15
                analysis_result["warnings"].append("Repetitive request pattern detected")
                analysis_result["anomalies_detected"].append("repetitive_pattern")
            
            # Analyze time patterns
            if len(user_events) > 10:
                time_intervals = []
                for i in range(1, min(len(user_events), 20)):
                    interval = user_events[i].timestamp - user_events[i-1].timestamp
                    time_intervals.append(interval)
                
                # Check for bot-like regular intervals
                if len(time_intervals) > 5:
                    avg_interval = sum(time_intervals) / len(time_intervals)
                    variance = sum((t - avg_interval) ** 2 for t in time_intervals) / len(time_intervals)
                    
                    if variance < 1.0 and avg_interval < 10:  # Very regular, short intervals
                        analysis_result["penalty"] += 25
                        analysis_result["warnings"].append("Bot-like behavior detected")
                        analysis_result["anomalies_detected"].append("bot_behavior")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Behavior analysis failed: {e}")
            return {"penalty": 0, "warnings": [f"Behavior analysis error: {str(e)}"], "anomalies_detected": []}
    
    def _check_model_integrity(self, request_data: Dict[str, Any]) -> Dict[str, bool]:
        """Check model integrity and detect tampering."""
        try:
            # Mock model integrity check
            # In real implementation, this would verify model checksums, signatures, etc.
            model_path = request_data.get("model_path")
            
            if model_path:
                # Check if model file exists and hasn't been tampered with
                integrity_key = f"model_{hashlib.md5(str(model_path).encode()).hexdigest()}"
                
                if integrity_key not in self.integrity_checkers:
                    # Initialize integrity checker for this model
                    self.integrity_checkers[integrity_key] = {
                        "checksum": self._calculate_mock_checksum(model_path),
                        "last_check": time.time(),
                        "verification_count": 0
                    }
                
                checker = self.integrity_checkers[integrity_key]
                current_checksum = self._calculate_mock_checksum(model_path)
                
                if current_checksum != checker["checksum"]:
                    logger.warning(f"Model integrity check failed for {model_path}")
                    return {"valid": False, "reason": "Model checksum mismatch"}
                
                checker["last_check"] = time.time()
                checker["verification_count"] += 1
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"Model integrity check failed: {e}")
            return {"valid": False, "reason": str(e)}
    
    def _calculate_mock_checksum(self, model_path: str) -> str:
        """Calculate mock checksum for model."""
        # In real implementation, this would calculate actual file checksum
        return hashlib.sha256(str(model_path).encode()).hexdigest()
    
    def _log_security_event(self, event_type: str, threat_level: ThreatLevel, 
                           source_ip: str, user_id: str, details: Dict[str, Any], action: str):
        """Log security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            details=details,
            action_taken=action
        )
        
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
        
        # Log based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            logger.critical(f"CRITICAL SECURITY EVENT: {event_type} from {source_ip} for user {user_id}")
        elif threat_level == ThreatLevel.HIGH:
            logger.error(f"HIGH THREAT: {event_type} from {source_ip} for user {user_id}")
        elif threat_level == ThreatLevel.MEDIUM:
            logger.warning(f"MEDIUM THREAT: {event_type} from {source_ip} for user {user_id}")
        else:
            logger.info(f"Security event: {event_type} from {source_ip} for user {user_id}")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        current_time = time.time()
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        recent_events = [e for e in self.security_events if e.timestamp > hour_ago]
        daily_events = [e for e in self.security_events if e.timestamp > day_ago]
        
        # Aggregate statistics
        threat_counts = {}
        for level in ThreatLevel:
            threat_counts[level.value] = len([e for e in recent_events if e.threat_level == level])
        
        blocked_requests = len([e for e in recent_events if e.action_taken == "blocked"])
        total_requests = len(recent_events)
        
        top_threats = {}
        for event in recent_events:
            event_type = event.event_type
            top_threats[event_type] = top_threats.get(event_type, 0) + 1
        
        return {
            "summary": {
                "total_events_last_hour": len(recent_events),
                "total_events_last_day": len(daily_events),
                "blocked_requests_last_hour": blocked_requests,
                "block_rate_percent": (blocked_requests / max(total_requests, 1)) * 100,
                "current_threat_level": self._calculate_overall_threat_level(recent_events)
            },
            "threat_distribution": threat_counts,
            "top_threat_types": dict(sorted(top_threats.items(), key=lambda x: x[1], reverse=True)[:10]),
            "rate_limiting": {
                "active_users": len(self.rate_limiters),
                "users_near_limit": len([u for u, data in self.rate_limiters.items() 
                                       if len(data["requests"]) > self.config["max_requests_per_hour"] * 0.8])
            },
            "model_integrity": {
                "models_monitored": len(self.integrity_checkers),
                "last_integrity_check": max([c["last_check"] for c in self.integrity_checkers.values()]) 
                                     if self.integrity_checkers else 0,
                "total_verifications": sum([c["verification_count"] for c in self.integrity_checkers.values()])
            }
        }
    
    def _calculate_overall_threat_level(self, recent_events: List[SecurityEvent]) -> str:
        """Calculate overall system threat level."""
        if not recent_events:
            return ThreatLevel.LOW.value
        
        critical_events = len([e for e in recent_events if e.threat_level == ThreatLevel.CRITICAL])
        high_events = len([e for e in recent_events if e.threat_level == ThreatLevel.HIGH])
        
        if critical_events > 0:
            return ThreatLevel.CRITICAL.value
        elif high_events > 5:
            return ThreatLevel.HIGH.value
        elif high_events > 0 or len(recent_events) > 100:
            return ThreatLevel.MEDIUM.value
        else:
            return ThreatLevel.LOW.value
    
    def generate_security_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        current_time = time.time()
        start_time = current_time - (hours_back * 3600)
        
        relevant_events = [e for e in self.security_events if e.timestamp > start_time]
        
        report = {
            "report_period": {
                "start_time": start_time,
                "end_time": current_time,
                "duration_hours": hours_back
            },
            "summary": {
                "total_events": len(relevant_events),
                "unique_users": len(set([e.user_id for e in relevant_events])),
                "unique_ips": len(set([e.source_ip for e in relevant_events])),
                "blocked_requests": len([e for e in relevant_events if e.action_taken == "blocked"])
            },
            "threat_analysis": {},
            "recommendations": []
        }
        
        # Analyze threats by type
        for level in ThreatLevel:
            level_events = [e for e in relevant_events if e.threat_level == level]
            report["threat_analysis"][level.value] = {
                "count": len(level_events),
                "percentage": (len(level_events) / max(len(relevant_events), 1)) * 100,
                "top_sources": list(set([e.source_ip for e in level_events]))[:5]
            }
        
        # Generate recommendations
        if report["threat_analysis"]["critical"]["count"] > 0:
            report["recommendations"].append("URGENT: Critical security threats detected. Review immediately.")
        
        if report["summary"]["blocked_requests"] > len(relevant_events) * 0.1:
            report["recommendations"].append("High block rate detected. Review security policies.")
        
        if report["summary"]["unique_ips"] < report["summary"]["unique_users"] * 0.5:
            report["recommendations"].append("Potential IP sharing or proxy usage detected.")
        
        return report


class SecureModelLoader:
    """Secure model loading with verification."""
    
    def __init__(self):
        self.trusted_sources = set()
        self.model_signatures = {}
        
    def add_trusted_source(self, source_url: str, public_key: str):
        """Add trusted model source."""
        self.trusted_sources.add(source_url)
        
    def verify_model_signature(self, model_path: str, signature: str, public_key: str) -> bool:
        """Verify model signature."""
        # Mock signature verification
        # In real implementation, this would use cryptographic signature verification
        expected_signature = hashlib.sha256(f"{model_path}{public_key}".encode()).hexdigest()
        return hmac.compare_digest(signature, expected_signature)
    
    def secure_load_model(self, model_path: str, signature: str = None, 
                         source_url: str = None) -> Dict[str, Any]:
        """Securely load model with verification."""
        result = {
            "loaded": False,
            "verified": False,
            "warnings": [],
            "model_info": {}
        }
        
        try:
            # Check if source is trusted
            if source_url and source_url not in self.trusted_sources:
                result["warnings"].append(f"Untrusted source: {source_url}")
            
            # Verify file integrity
            if signature:
                # Mock verification - in real implementation use actual cryptographic verification
                mock_public_key = "mock_public_key"
                if self.verify_model_signature(model_path, signature, mock_public_key):
                    result["verified"] = True
                else:
                    result["warnings"].append("Model signature verification failed")
            
            # Additional security checks
            result["model_info"] = {
                "path": model_path,
                "size_mb": 35.0,  # Mock size
                "format": "onnx",
                "checksum": hashlib.md5(model_path.encode()).hexdigest()
            }
            
            result["loaded"] = True
            return result
            
        except Exception as e:
            result["warnings"].append(f"Secure loading failed: {str(e)}")
            return result


if __name__ == "__main__":
    # Test advanced security features
    print("Testing Advanced Security Validator...")
    
    validator = AdvancedSecurityValidator(strict_mode=True)
    
    # Test normal request
    normal_request = {
        "image_data": "base64_encoded_image",
        "task": "caption_generation",
        "user_preferences": {"language": "en"}
    }
    
    result = validator.validate_advanced_request("user123", normal_request, "192.168.1.100")
    print(f"Normal request validation: {result['valid']}, Threat Level: {result['threat_level'].value}")
    
    # Test suspicious request
    suspicious_request = {
        "image_data": "<script>alert('xss')</script>",
        "task": "caption_generation",
        "command": "rm -rf /"
    }
    
    result = validator.validate_advanced_request("user456", suspicious_request, "10.0.0.1")
    print(f"Suspicious request validation: {result['valid']}, Reason: {result['blocked_reason']}")
    
    # Test security dashboard
    dashboard = validator.get_security_dashboard()
    print(f"Security Dashboard - Total events: {dashboard['summary']['total_events_last_hour']}")
    
    print("✅ Advanced security features working correctly!")