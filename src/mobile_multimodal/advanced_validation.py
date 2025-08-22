"""Advanced Validation Framework - Comprehensive input validation and security checks.

This module implements production-grade validation with:
1. Multi-layer input sanitization and validation
2. ML-specific security checks (adversarial input detection)
3. Resource consumption monitoring and limits
4. Data integrity verification with cryptographic hashes
5. Real-time threat detection and response
"""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation security levels."""
    BASIC = "basic"          # Basic type and range checks
    STANDARD = "standard"    # Standard security validation
    STRICT = "strict"        # Strict security with ML-specific checks
    PARANOID = "paranoid"    # Maximum security with deep analysis


class ThreatType(Enum):
    """Types of security threats."""
    ADVERSARIAL_INPUT = "adversarial_input"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    PRIVACY_ATTACK = "privacy_attack"
    MALFORMED_INPUT = "malformed_input"


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    threat_level: float  # 0.0-1.0, higher is more threatening
    detected_threats: List[ThreatType]
    validation_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    total_validations: int = 0
    threats_detected: int = 0
    false_positives: int = 0
    avg_validation_time: float = 0.0
    resource_violations: int = 0
    blocked_requests: int = 0


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.validation_count = 0
        self.detection_count = 0
        
    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Perform validation on input data."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if validator is enabled."""
        return self.enabled
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "name": self.name,
            "validation_count": self.validation_count,
            "detection_count": self.detection_count,
            "detection_rate": self.detection_count / max(self.validation_count, 1),
            "enabled": self.enabled
        }


class TypeValidator(BaseValidator):
    """Validates data types and basic structure."""
    
    def __init__(self, expected_types: Dict[str, type]):
        super().__init__("type_validator")
        self.expected_types = expected_types
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate data types."""
        start_time = time.perf_counter()
        self.validation_count += 1
        
        threats = []
        threat_level = 0.0
        error_message = None
        
        try:
            if isinstance(data, dict):
                for key, expected_type in self.expected_types.items():
                    if key in data:
                        value = data[key]
                        if not isinstance(value, expected_type):
                            threats.append(ThreatType.MALFORMED_INPUT)
                            threat_level = max(threat_level, 0.3)
                            error_message = f"Invalid type for {key}: expected {expected_type}, got {type(value)}"
                            
                        # Check for suspicious values
                        if isinstance(value, (int, float)) and (np.isinf(value) or np.isnan(value)):
                            threats.append(ThreatType.MALFORMED_INPUT)
                            threat_level = max(threat_level, 0.5)
                            error_message = f"Invalid numeric value for {key}: {value}"
                            
            elif hasattr(data, 'shape') and hasattr(data, 'dtype'):
                # NumPy array or tensor validation
                if np.any(np.isinf(data)) or np.any(np.isnan(data)):
                    threats.append(ThreatType.MALFORMED_INPUT)
                    threat_level = max(threat_level, 0.4)
                    error_message = "Array contains invalid values (inf/nan)"
                    
        except Exception as e:
            threats.append(ThreatType.MALFORMED_INPUT)
            threat_level = 1.0
            error_message = f"Type validation error: {str(e)}"
        
        validation_time = time.perf_counter() - start_time
        
        if threats:
            self.detection_count += 1
            
        return ValidationResult(
            is_valid=len(threats) == 0,
            threat_level=threat_level,
            detected_threats=threats,
            validation_time=validation_time,
            error_message=error_message
        )


class RangeValidator(BaseValidator):
    """Validates numerical ranges and bounds."""
    
    def __init__(self, ranges: Dict[str, Tuple[float, float]]):
        super().__init__("range_validator")
        self.ranges = ranges  # field_name -> (min_val, max_val)
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate numerical ranges."""
        start_time = time.perf_counter()
        self.validation_count += 1
        
        threats = []
        threat_level = 0.0
        error_message = None
        
        try:
            if isinstance(data, dict):
                for field, (min_val, max_val) in self.ranges.items():
                    if field in data:
                        value = data[field]
                        if isinstance(value, (int, float)):
                            if value < min_val or value > max_val:
                                threats.append(ThreatType.MALFORMED_INPUT)
                                threat_level = max(threat_level, 0.4)
                                error_message = f"Value {value} for {field} outside range [{min_val}, {max_val}]"
                                
            elif hasattr(data, 'min') and hasattr(data, 'max'):
                # Array-like data
                data_min, data_max = data.min(), data.max()
                if 'array' in self.ranges:
                    range_min, range_max = self.ranges['array']
                    if data_min < range_min or data_max > range_max:
                        threats.append(ThreatType.MALFORMED_INPUT)
                        threat_level = max(threat_level, 0.3)
                        error_message = f"Array values [{data_min}, {data_max}] outside expected range [{range_min}, {range_max}]"
                        
        except Exception as e:
            threats.append(ThreatType.MALFORMED_INPUT)
            threat_level = 0.5
            error_message = f"Range validation error: {str(e)}"
        
        validation_time = time.perf_counter() - start_time
        
        if threats:
            self.detection_count += 1
            
        return ValidationResult(
            is_valid=len(threats) == 0,
            threat_level=threat_level,
            detected_threats=threats,
            validation_time=validation_time,
            error_message=error_message
        )


class AdversarialDetector(BaseValidator):
    """Detects adversarial inputs using statistical analysis."""
    
    def __init__(self, sensitivity: float = 0.1):
        super().__init__("adversarial_detector")
        self.sensitivity = sensitivity
        self.baseline_stats = {}
        self.detection_history = []
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Detect adversarial patterns in input data."""
        start_time = time.perf_counter()
        self.validation_count += 1
        
        threats = []
        threat_level = 0.0
        error_message = None
        metadata = {}
        
        try:
            if hasattr(data, 'shape') and len(data.shape) >= 2:
                # Image-like data
                threat_level, detected_threats = self._analyze_image_adversarial(data)
                threats.extend(detected_threats)
                
            elif isinstance(data, str):
                # Text data
                threat_level, detected_threats = self._analyze_text_adversarial(data)
                threats.extend(detected_threats)
                
            elif isinstance(data, (list, np.ndarray)) and len(data) > 0:
                # Sequence data
                threat_level, detected_threats = self._analyze_sequence_adversarial(data)
                threats.extend(detected_threats)
                
        except Exception as e:
            threats.append(ThreatType.ADVERSARIAL_INPUT)
            threat_level = 0.3
            error_message = f"Adversarial detection error: {str(e)}"
        
        validation_time = time.perf_counter() - start_time
        
        # Record detection for analysis
        detection_record = {
            "timestamp": time.time(),
            "threat_level": threat_level,
            "threats": [t.value for t in threats],
            "data_shape": getattr(data, 'shape', None),
            "data_type": type(data).__name__
        }
        self.detection_history.append(detection_record)
        
        # Keep only last 1000 detections
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        if threats:
            self.detection_count += 1
            
        return ValidationResult(
            is_valid=threat_level < self.sensitivity,
            threat_level=threat_level,
            detected_threats=threats,
            validation_time=validation_time,
            error_message=error_message,
            metadata=metadata
        )
    
    def _analyze_image_adversarial(self, image: np.ndarray) -> Tuple[float, List[ThreatType]]:
        """Analyze image for adversarial patterns."""
        threats = []
        threat_level = 0.0
        
        # Check for unusual statistical properties
        try:
            # Gradient magnitude analysis
            if len(image.shape) >= 2:
                grad_x = np.gradient(image, axis=-2)
                grad_y = np.gradient(image, axis=-1)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # High gradient variance might indicate adversarial noise
                grad_var = np.var(gradient_magnitude)
                if grad_var > 0.1:  # Threshold based on typical image statistics
                    threats.append(ThreatType.ADVERSARIAL_INPUT)
                    threat_level = max(threat_level, 0.6)
                    
            # Frequency domain analysis
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] in [1, 3]):
                # Convert to grayscale if needed
                if len(image.shape) == 3:
                    gray = np.mean(image, axis=-1)
                else:
                    gray = image
                    
                # FFT analysis for high-frequency noise
                fft = np.fft.fft2(gray)
                fft_magnitude = np.abs(fft)
                high_freq_energy = np.sum(fft_magnitude[fft_magnitude.shape[0]//4:, fft_magnitude.shape[1]//4:])
                total_energy = np.sum(fft_magnitude)
                
                if high_freq_energy / total_energy > 0.15:  # Unusually high high-frequency content
                    threats.append(ThreatType.ADVERSARIAL_INPUT)
                    threat_level = max(threat_level, 0.4)
                    
        except Exception:
            # If analysis fails, treat as low-level threat
            threat_level = max(threat_level, 0.1)
            
        return threat_level, threats
    
    def _analyze_text_adversarial(self, text: str) -> Tuple[float, List[ThreatType]]:
        """Analyze text for adversarial patterns."""
        threats = []
        threat_level = 0.0
        
        # Check for suspicious patterns
        if len(text) > 10000:  # Unusually long text
            threats.append(ThreatType.RESOURCE_EXHAUSTION)
            threat_level = max(threat_level, 0.5)
            
        # Check for unusual character patterns
        unusual_chars = sum(1 for c in text if ord(c) > 127 and c not in 'àáâãäåæçèéêëìíîïñòóôõöøùúûüý')
        if unusual_chars / max(len(text), 1) > 0.1:
            threats.append(ThreatType.ADVERSARIAL_INPUT)
            threat_level = max(threat_level, 0.3)
            
        # Check for repeated patterns (potential confusion attacks)
        words = text.split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = 1 - (unique_words / len(words))
            if repetition_ratio > 0.8:
                threats.append(ThreatType.ADVERSARIAL_INPUT)
                threat_level = max(threat_level, 0.4)
                
        return threat_level, threats
    
    def _analyze_sequence_adversarial(self, sequence: Union[list, np.ndarray]) -> Tuple[float, List[ThreatType]]:
        """Analyze sequence data for adversarial patterns."""
        threats = []
        threat_level = 0.0
        
        try:
            seq_array = np.array(sequence) if not isinstance(sequence, np.ndarray) else sequence
            
            # Check for unusual statistical properties
            if len(seq_array) > 5:
                # Variance analysis
                seq_var = np.var(seq_array)
                seq_mean = np.mean(seq_array)
                
                # Coefficient of variation
                if seq_mean != 0:
                    cv = np.sqrt(seq_var) / abs(seq_mean)
                    if cv > 10:  # Very high variability
                        threats.append(ThreatType.ADVERSARIAL_INPUT)
                        threat_level = max(threat_level, 0.3)
                        
                # Check for periodic patterns that might indicate synthetic data
                if len(seq_array) > 20:
                    autocorr = np.correlate(seq_array, seq_array, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    autocorr = autocorr[1:min(len(autocorr), 10)]  # Check first 10 lags
                    
                    if np.any(autocorr > 0.9 * autocorr[0]):  # High autocorrelation
                        threats.append(ThreatType.ADVERSARIAL_INPUT)
                        threat_level = max(threat_level, 0.2)
                        
        except Exception:
            threat_level = max(threat_level, 0.1)
            
        return threat_level, threats


class ResourceMonitor(BaseValidator):
    """Monitors resource consumption and prevents exhaustion attacks."""
    
    def __init__(self, max_memory_mb: float = 512, max_computation_time: float = 5.0):
        super().__init__("resource_monitor")
        self.max_memory_mb = max_memory_mb
        self.max_computation_time = max_computation_time
        self.resource_usage_history = []
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Monitor resource consumption."""
        start_time = time.perf_counter()
        self.validation_count += 1
        
        threats = []
        threat_level = 0.0
        error_message = None
        metadata = {}
        
        try:
            # Estimate memory usage
            estimated_memory = self._estimate_memory_usage(data)
            metadata["estimated_memory_mb"] = estimated_memory
            
            if estimated_memory > self.max_memory_mb:
                threats.append(ThreatType.RESOURCE_EXHAUSTION)
                threat_level = max(threat_level, 0.8)
                error_message = f"Estimated memory usage {estimated_memory:.1f}MB exceeds limit {self.max_memory_mb}MB"
                
            # Check computation time (for validation itself)
            validation_time = time.perf_counter() - start_time
            if validation_time > self.max_computation_time:
                threats.append(ThreatType.RESOURCE_EXHAUSTION)
                threat_level = max(threat_level, 0.6)
                error_message = f"Validation time {validation_time:.3f}s exceeds limit {self.max_computation_time}s"
                
            # Record resource usage
            usage_record = {
                "timestamp": time.time(),
                "memory_mb": estimated_memory,
                "validation_time": validation_time,
                "data_type": type(data).__name__
            }
            self.resource_usage_history.append(usage_record)
            
            # Keep only last 500 records
            if len(self.resource_usage_history) > 500:
                self.resource_usage_history = self.resource_usage_history[-500:]
                
        except Exception as e:
            threats.append(ThreatType.RESOURCE_EXHAUSTION)
            threat_level = 0.3
            error_message = f"Resource monitoring error: {str(e)}"
        
        validation_time = time.perf_counter() - start_time
        
        if threats:
            self.detection_count += 1
            
        return ValidationResult(
            is_valid=len(threats) == 0,
            threat_level=threat_level,
            detected_threats=threats,
            validation_time=validation_time,
            error_message=error_message,
            metadata=metadata
        )
    
    def _estimate_memory_usage(self, data: Any) -> float:
        """Estimate memory usage in MB."""
        try:
            if hasattr(data, 'nbytes'):
                # NumPy array or similar
                return data.nbytes / (1024 * 1024)
            elif TORCH_AVAILABLE and torch.is_tensor(data):
                # PyTorch tensor
                return data.element_size() * data.nelement() / (1024 * 1024)
            elif isinstance(data, (list, tuple)):
                # Estimate list/tuple memory
                if len(data) > 0:
                    sample_size = len(str(data[0]))
                    return len(data) * sample_size * 8 / (1024 * 1024)  # Rough estimate
            elif isinstance(data, str):
                return len(data.encode('utf-8')) / (1024 * 1024)
            elif isinstance(data, dict):
                return len(str(data).encode('utf-8')) / (1024 * 1024)
            else:
                return 0.1  # Default small size
        except Exception:
            return 1.0  # Conservative estimate


class IntegrityValidator(BaseValidator):
    """Validates data integrity using cryptographic hashes."""
    
    def __init__(self):
        super().__init__("integrity_validator")
        self.hash_cache = {}
        
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate data integrity."""
        start_time = time.perf_counter()
        self.validation_count += 1
        
        threats = []
        threat_level = 0.0
        error_message = None
        metadata = {}
        
        try:
            # Compute hash of data
            data_hash = self._compute_hash(data)
            metadata["data_hash"] = data_hash
            
            # Check for expected hash if provided in context
            if context and "expected_hash" in context:
                expected_hash = context["expected_hash"]
                if data_hash != expected_hash:
                    threats.append(ThreatType.DATA_POISONING)
                    threat_level = max(threat_level, 0.9)
                    error_message = f"Data integrity check failed: hash mismatch"
                    
            # Check for known malicious hashes
            if data_hash in self._get_malicious_hash_blacklist():
                threats.append(ThreatType.DATA_POISONING)
                threat_level = 1.0
                error_message = "Data matches known malicious content"
                
        except Exception as e:
            threats.append(ThreatType.MALFORMED_INPUT)
            threat_level = 0.2
            error_message = f"Integrity validation error: {str(e)}"
        
        validation_time = time.perf_counter() - start_time
        
        if threats:
            self.detection_count += 1
            
        return ValidationResult(
            is_valid=len(threats) == 0,
            threat_level=threat_level,
            detected_threats=threats,
            validation_time=validation_time,
            error_message=error_message,
            metadata=metadata
        )
    
    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        hasher = hashlib.sha256()
        
        if isinstance(data, str):
            hasher.update(data.encode('utf-8'))
        elif isinstance(data, bytes):
            hasher.update(data)
        elif hasattr(data, 'tobytes'):
            # NumPy array or similar
            hasher.update(data.tobytes())
        elif TORCH_AVAILABLE and torch.is_tensor(data):
            hasher.update(data.cpu().numpy().tobytes())
        else:
            # Fallback: convert to string and hash
            hasher.update(str(data).encode('utf-8'))
            
        return hasher.hexdigest()
    
    def _get_malicious_hash_blacklist(self) -> Set[str]:
        """Get blacklist of known malicious content hashes."""
        # In practice, this would be loaded from a security database
        return {
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # Empty string (example)
            # Add more known malicious hashes here
        }


class InputValidator:
    """Simple input validator for basic validation needs."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        logger.info(f"InputValidator initialized with strict_mode={strict_mode}")
    
    def validate(self, data: Any, validation_type: str = "general") -> bool:
        """Validate input data with basic checks."""
        try:
            if data is None:
                return False
            
            # Basic validation logic
            if isinstance(data, dict):
                if data.get("type") == "text" and "content" not in data:
                    return False
                if data.get("type") == "image" and "shape" not in data:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

class DataValidator:
    """Data-specific validation for model inputs."""
    
    def __init__(self):
        self.max_image_size = (4096, 4096)
        self.max_text_length = 10000
        logger.info("DataValidator initialized")
    
    def validate_image_data(self, image_data: Any) -> bool:
        """Validate image data format and constraints."""
        try:
            if hasattr(image_data, 'shape'):
                shape = image_data.shape
                return len(shape) == 3 and shape[2] in [1, 3, 4]
            elif isinstance(image_data, dict) and "shape" in image_data:
                shape = image_data["shape"]
                return isinstance(shape, list) and len(shape) == 3
            return False
        except Exception:
            return False

class CompositeValidator:
    """Composite validator that runs multiple validation checks."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level
        self.validators = []
        self.security_metrics = SecurityMetrics()
        self.blocked_requests = []
        
        self._setup_validators()
        
    def _setup_validators(self):
        """Setup validators based on security level."""
        # Basic validators (always enabled)
        self.validators.append(TypeValidator({
            "image": np.ndarray,
            "text": str,
            "batch_size": int,
            "temperature": float
        }))
        
        self.validators.append(RangeValidator({
            "batch_size": (1, 64),
            "temperature": (0.1, 2.0),
            "array": (-10.0, 10.0)  # General array value range
        }))
        
        if self.level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            self.validators.append(ResourceMonitor(max_memory_mb=512, max_computation_time=5.0))
            self.validators.append(IntegrityValidator())
            
        if self.level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            self.validators.append(AdversarialDetector(sensitivity=0.1))
            
        if self.level == ValidationLevel.PARANOID:
            # Additional paranoid-level validators
            self.validators.append(AdversarialDetector(sensitivity=0.05))  # More sensitive
            self.validators.append(ResourceMonitor(max_memory_mb=256, max_computation_time=2.0))  # Stricter limits
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Run all validation checks."""
        start_time = time.perf_counter()
        self.security_metrics.total_validations += 1
        
        all_threats = []
        max_threat_level = 0.0
        all_errors = []
        combined_metadata = {}
        total_validation_time = 0.0
        
        for validator in self.validators:
            if not validator.is_enabled():
                continue
                
            try:
                result = validator.validate(data, context)
                
                if result.detected_threats:
                    all_threats.extend(result.detected_threats)
                    max_threat_level = max(max_threat_level, result.threat_level)
                    
                if result.error_message:
                    all_errors.append(f"{validator.name}: {result.error_message}")
                    
                if result.metadata:
                    combined_metadata[validator.name] = result.metadata
                    
                total_validation_time += result.validation_time
                
            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {str(e)}")
                all_threats.append(ThreatType.MALFORMED_INPUT)
                max_threat_level = max(max_threat_level, 0.3)
                all_errors.append(f"{validator.name}: Validation failed - {str(e)}")
        
        # Remove duplicate threats
        unique_threats = list(set(all_threats))
        
        # Determine if input is valid
        is_valid = len(unique_threats) == 0 and max_threat_level < 0.5
        
        # Update metrics
        if unique_threats:
            self.security_metrics.threats_detected += 1
            
        if not is_valid:
            self.security_metrics.blocked_requests += 1
            
            # Record blocked request for analysis
            blocked_record = {
                "timestamp": time.time(),
                "threat_level": max_threat_level,
                "threats": [t.value for t in unique_threats],
                "errors": all_errors,
                "data_type": type(data).__name__
            }
            self.blocked_requests.append(blocked_record)
            
            # Keep only last 100 blocked requests
            if len(self.blocked_requests) > 100:
                self.blocked_requests = self.blocked_requests[-100:]
        
        validation_time = time.perf_counter() - start_time
        self.security_metrics.avg_validation_time = (
            (self.security_metrics.avg_validation_time * (self.security_metrics.total_validations - 1) + 
             validation_time) / self.security_metrics.total_validations
        )
        
        return ValidationResult(
            is_valid=is_valid,
            threat_level=max_threat_level,
            detected_threats=unique_threats,
            validation_time=validation_time,
            error_message="; ".join(all_errors) if all_errors else None,
            metadata=combined_metadata
        )
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        validator_stats = [v.get_statistics() for v in self.validators]
        
        # Threat type distribution
        threat_distribution = {}
        for record in self.blocked_requests:
            for threat in record["threats"]:
                threat_distribution[threat] = threat_distribution.get(threat, 0) + 1
        
        return {
            "security_level": self.level.value,
            "metrics": {
                "total_validations": self.security_metrics.total_validations,
                "threats_detected": self.security_metrics.threats_detected,
                "blocked_requests": self.security_metrics.blocked_requests,
                "avg_validation_time": self.security_metrics.avg_validation_time,
                "threat_detection_rate": (self.security_metrics.threats_detected / 
                                        max(self.security_metrics.total_validations, 1))
            },
            "validator_statistics": validator_stats,
            "threat_distribution": threat_distribution,
            "recent_blocked_requests": len(self.blocked_requests),
            "active_validators": len([v for v in self.validators if v.is_enabled()])
        }
    
    def update_security_level(self, new_level: ValidationLevel):
        """Update security validation level."""
        self.level = new_level
        self.validators.clear()
        self._setup_validators()
        logger.info(f"Updated validation security level to {new_level.value}")
    
    def enable_validator(self, validator_name: str):
        """Enable specific validator."""
        for validator in self.validators:
            if validator.name == validator_name:
                validator.enabled = True
                logger.info(f"Enabled validator: {validator_name}")
                return
        logger.warning(f"Validator not found: {validator_name}")
    
    def disable_validator(self, validator_name: str):
        """Disable specific validator."""
        for validator in self.validators:
            if validator.name == validator_name:
                validator.enabled = False
                logger.info(f"Disabled validator: {validator_name}")
                return
        logger.warning(f"Validator not found: {validator_name}")


# Factory functions
def create_validator(level: ValidationLevel = ValidationLevel.STANDARD) -> CompositeValidator:
    """Create a composite validator with specified security level."""
    return CompositeValidator(level)


def create_mobile_validator() -> CompositeValidator:
    """Create a validator optimized for mobile deployment."""
    validator = CompositeValidator(ValidationLevel.STANDARD)
    
    # Adjust resource limits for mobile
    for v in validator.validators:
        if isinstance(v, ResourceMonitor):
            v.max_memory_mb = 256  # Lower memory limit for mobile
            v.max_computation_time = 2.0  # Faster timeout for mobile
            
    return validator


# Export classes and functions
__all__ = [
    "ValidationLevel", "ThreatType", "ValidationResult", "SecurityMetrics",
    "BaseValidator", "TypeValidator", "RangeValidator", "AdversarialDetector",
    "ResourceMonitor", "IntegrityValidator", "CompositeValidator",
    "create_validator", "create_mobile_validator"
]