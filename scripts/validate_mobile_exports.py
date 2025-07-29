#!/usr/bin/env python3
"""
Mobile Model Export Validator

Validates that mobile model exports meet quality and performance standards
for deployment on Android and iOS devices.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import onnx
    import torch
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


class MobileExportValidator:
    """Validator for mobile model exports."""
    
    def __init__(self):
        self.issues: List[Dict] = []
        self.model_size_limits = {
            'android': 50 * 1024 * 1024,  # 50MB
            'ios': 35 * 1024 * 1024,      # 35MB
            'edge': 25 * 1024 * 1024      # 25MB
        }
    
    def validate_exports(self) -> Dict:
        """Validate all mobile model exports."""
        models_dir = Path('models')
        if not models_dir.exists():
            self._add_issue('missing_models_dir', 'Models directory not found', 'high')
            return self._generate_report()
        
        # Check for required export formats
        required_formats = {
            'android': ['.tflite', '.dlc'],
            'ios': ['.mlmodel', '.mlpackage'],
            'onnx': ['.onnx']
        }
        
        for platform, formats in required_formats.items():
            self._validate_platform_exports(models_dir, platform, formats)
        
        return self._generate_report()
    
    def _validate_platform_exports(self, models_dir: Path, platform: str, 
                                 formats: List[str]) -> None:
        """Validate exports for a specific platform."""
        found_formats = []
        
        for format_ext in formats:
            model_files = list(models_dir.glob(f'*{format_ext}'))
            if model_files:
                found_formats.append(format_ext)
                for model_file in model_files:
                    self._validate_model_file(model_file, platform)
            else:
                self._add_issue(
                    'missing_format',
                    f"No {format_ext} models found for {platform}",
                    'medium'
                )
        
        if not found_formats:
            self._add_issue(
                'no_platform_exports',
                f"No model exports found for {platform}",
                'high'
            )
    
    def _validate_model_file(self, model_file: Path, platform: str) -> None:
        """Validate a specific model file."""
        file_size = model_file.stat().st_size
        
        # Check file size limits
        if platform in self.model_size_limits:
            size_limit = self.model_size_limits[platform]
            if file_size > size_limit:
                self._add_issue(
                    'size_limit_exceeded',
                    f"{model_file.name} ({file_size / 1024 / 1024:.1f}MB) exceeds "
                    f"{platform} limit ({size_limit / 1024 / 1024:.1f}MB)",
                    'high'
                )
        
        # Validate file integrity based on format
        if model_file.suffix == '.onnx' and VALIDATION_AVAILABLE:
            self._validate_onnx_model(model_file)
        elif model_file.suffix == '.tflite':
            self._validate_tflite_model(model_file)
        elif model_file.suffix in ['.mlmodel', '.mlpackage']:
            self._validate_coreml_model(model_file)
    
    def _validate_onnx_model(self, model_file: Path) -> None:
        """Validate ONNX model."""
        try:
            model = onnx.load(str(model_file))
            onnx.checker.check_model(model)
            
            # Check for mobile-friendly operations
            unsupported_ops = self._check_mobile_ops(model)
            if unsupported_ops:
                self._add_issue(
                    'unsupported_operations',
                    f"{model_file.name} contains potentially unsupported ops: {unsupported_ops}",
                    'medium'
                )
                
        except Exception as e:
            self._add_issue(
                'invalid_onnx',
                f"Invalid ONNX model {model_file.name}: {e}",
                'high'
            )
    
    def _check_mobile_ops(self, model) -> List[str]:
        """Check for mobile-unfriendly operations."""
        unsupported_ops = []
        mobile_unfriendly = {
            'Loop', 'If', 'Scan', 'SequenceAt', 'SequenceConstruct',
            'NonMaxSuppression', 'RoiAlign', 'SpaceToDepth'
        }
        
        for node in model.graph.node:
            if node.op_type in mobile_unfriendly:
                unsupported_ops.append(node.op_type)
        
        return list(set(unsupported_ops))
    
    def _validate_tflite_model(self, model_file: Path) -> None:
        """Validate TensorFlow Lite model."""
        try:
            # Basic file validation
            with open(model_file, 'rb') as f:
                model_data = f.read()
                
            # Check TFLite magic number
            if not model_data.startswith(b'TFL3'):
                self._add_issue(
                    'invalid_tflite',
                    f"Invalid TFLite magic number in {model_file.name}",
                    'high'
                )
                
        except Exception as e:
            self._add_issue(
                'tflite_read_error',
                f"Cannot read TFLite model {model_file.name}: {e}",
                'high'
            )
    
    def _validate_coreml_model(self, model_file: Path) -> None:
        """Validate Core ML model."""
        if not model_file.exists():
            self._add_issue(
                'missing_coreml',
                f"Core ML model not found: {model_file.name}",
                'high'
            )
            return
        
        # For .mlpackage, check directory structure
        if model_file.suffix == '.mlpackage':
            required_files = ['Manifest.json', 'Data', 'Metadata']
            for req_file in required_files:
                if not (model_file / req_file).exists():
                    self._add_issue(
                        'incomplete_mlpackage',
                        f"Missing {req_file} in {model_file.name}",
                        'medium'
                    )
    
    def _add_issue(self, issue_type: str, message: str, severity: str) -> None:
        """Add a validation issue."""
        self.issues.append({
            'type': issue_type,
            'message': message,
            'severity': severity
        })
    
    def _generate_report(self) -> Dict:
        """Generate validation report."""
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for issue in self.issues:
            severity_counts[issue['severity']] += 1
        
        return {
            'total_issues': len(self.issues),
            'severity_breakdown': severity_counts,
            'issues': self.issues,
            'validation_passed': severity_counts['high'] == 0
        }


def main():
    """Main validation function."""
    if not VALIDATION_AVAILABLE:
        print("⚠️  Model validation libraries not available")
        print("Install with: pip install onnx torch")
        return
    
    validator = MobileExportValidator()
    report = validator.validate_exports()
    
    # Save report
    report_path = Path('reports/mobile-export-validation.json')
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print("📱 Mobile Export Validation Results")
    print(f"Total Issues: {report['total_issues']}")
    print(f"High: {report['severity_breakdown']['high']}")
    print(f"Medium: {report['severity_breakdown']['medium']}")
    print(f"Validation: {'✅ PASSED' if report['validation_passed'] else '❌ FAILED'}")
    
    if report['issues']:
        print("\n🚨 Issues Found:")
        for issue in report['issues']:
            print(f"  {issue['severity'].upper()}: {issue['message']}")
    
    # Exit with error if validation failed
    if not report['validation_passed']:
        sys.exit(1)


if __name__ == "__main__":
    main()