#!/usr/bin/env python3
"""
Quantization Accuracy Drift Checker

Monitors and validates that quantized models maintain acceptable accuracy
compared to their full-precision counterparts.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class QuantizationDriftChecker:
    """Checks for accuracy drift in quantized models."""
    
    def __init__(self):
        self.drift_thresholds = {
            'int8': 0.02,   # 2% max accuracy drop for INT8
            'int4': 0.05,   # 5% max accuracy drop for INT4
            'int2': 0.10,   # 10% max accuracy drop for INT2
        }
        self.issues: List[Dict] = []
    
    def check_quantization_drift(self) -> Dict:
        """Check for quantization accuracy drift."""
        results_dir = Path('quantization_results')
        if not results_dir.exists():
            self._add_issue(
                'missing_results',
                'Quantization results directory not found',
                'medium'
            )
            return self._generate_report()
        
        # Look for accuracy comparison files
        comparison_files = list(results_dir.glob('*_accuracy_comparison.json'))
        if not comparison_files:
            self._add_issue(
                'no_comparisons',
                'No accuracy comparison files found',
                'medium'
            )
            return self._generate_report()
        
        for comparison_file in comparison_files:
            self._analyze_comparison_file(comparison_file)
        
        return self._generate_report()
    
    def _analyze_comparison_file(self, comparison_file: Path) -> None:
        """Analyze a single accuracy comparison file."""
        try:
            with open(comparison_file, 'r') as f:
                data = json.load(f)
            
            base_accuracy = data.get('base_model_accuracy')
            quantized_results = data.get('quantized_models', {})
            
            if base_accuracy is None:
                self._add_issue(
                    'missing_base_accuracy',
                    f'No base model accuracy in {comparison_file.name}',
                    'medium'
                )
                return
            
            for quant_type, quant_data in quantized_results.items():
                self._check_model_drift(
                    comparison_file.name, 
                    quant_type, 
                    base_accuracy, 
                    quant_data
                )
                
        except json.JSONDecodeError:
            self._add_issue(
                'invalid_json',
                f'Invalid JSON in {comparison_file.name}',
                'high'
            )
        except Exception as e:
            self._add_issue(
                'analysis_error',
                f'Error analyzing {comparison_file.name}: {e}',
                'medium'
            )
    
    def _check_model_drift(self, file_name: str, quant_type: str, 
                          base_accuracy: float, quant_data: Dict) -> None:
        """Check drift for a specific quantized model."""
        quant_accuracy = quant_data.get('accuracy')
        if quant_accuracy is None:
            self._add_issue(
                'missing_quant_accuracy',
                f'No accuracy data for {quant_type} in {file_name}',
                'medium'
            )
            return
        
        # Calculate accuracy drop
        accuracy_drop = base_accuracy - quant_accuracy
        relative_drop = accuracy_drop / base_accuracy
        
        # Get threshold for this quantization type
        threshold = self._get_threshold(quant_type)
        
        if relative_drop > threshold:
            self._add_issue(
                'accuracy_drift',
                f'{quant_type} model in {file_name} has {relative_drop:.1%} '
                f'accuracy drop (threshold: {threshold:.1%})',
                'high'
            )
        elif relative_drop > threshold * 0.8:  # 80% of threshold = warning
            self._add_issue(
                'accuracy_warning',
                f'{quant_type} model in {file_name} approaching drift threshold '
                f'({relative_drop:.1%} vs {threshold:.1%})',
                'medium'
            )
        
        # Check for other quality metrics
        self._check_additional_metrics(file_name, quant_type, quant_data)
    
    def _get_threshold(self, quant_type: str) -> float:
        """Get accuracy threshold for quantization type."""
        # Extract quantization level from type name
        if 'int2' in quant_type.lower():
            return self.drift_thresholds['int2']
        elif 'int4' in quant_type.lower():
            return self.drift_thresholds['int4']
        elif 'int8' in quant_type.lower():
            return self.drift_thresholds['int8']
        else:
            # Default to most restrictive
            return self.drift_thresholds['int8']
    
    def _check_additional_metrics(self, file_name: str, quant_type: str, 
                                quant_data: Dict) -> None:
        """Check additional quality metrics beyond accuracy."""
        # Check model size
        model_size = quant_data.get('model_size_mb')
        if model_size and model_size > 50:  # 50MB threshold
            self._add_issue(
                'size_warning',
                f'{quant_type} model in {file_name} is large ({model_size:.1f}MB)',
                'low'
            )
        
        # Check inference time
        inference_time = quant_data.get('inference_time_ms')
        expected_speedup = self._get_expected_speedup(quant_type)
        
        if inference_time and expected_speedup:
            base_time = quant_data.get('base_inference_time_ms')
            if base_time:
                actual_speedup = base_time / inference_time
                if actual_speedup < expected_speedup * 0.7:  # 70% of expected
                    self._add_issue(
                        'performance_warning',
                        f'{quant_type} model in {file_name} not achieving expected '
                        f'speedup ({actual_speedup:.1f}x vs {expected_speedup:.1f}x)',
                        'medium'
                    )
    
    def _get_expected_speedup(self, quant_type: str) -> Optional[float]:
        """Get expected inference speedup for quantization type."""
        speedup_factors = {
            'int8': 2.0,
            'int4': 3.0,
            'int2': 4.0
        }
        
        for key, speedup in speedup_factors.items():
            if key in quant_type.lower():
                return speedup
        return None
    
    def _add_issue(self, issue_type: str, message: str, severity: str) -> None:
        """Add a drift check issue."""
        self.issues.append({
            'type': issue_type,
            'message': message,
            'severity': severity
        })
    
    def _generate_report(self) -> Dict:
        """Generate drift check report."""
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for issue in self.issues:
            severity_counts[issue['severity']] += 1
        
        return {
            'total_issues': len(self.issues),
            'severity_breakdown': severity_counts,
            'issues': self.issues,
            'drift_check_passed': severity_counts['high'] == 0
        }


def main():
    """Main drift checking function."""
    checker = QuantizationDriftChecker()
    report = checker.check_quantization_drift()
    
    # Save report
    report_path = Path('reports/quantization-drift-check.json')
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print("⚖️  Quantization Drift Check Results")
    print(f"Total Issues: {report['total_issues']}")
    print(f"High: {report['severity_breakdown']['high']}")
    print(f"Medium: {report['severity_breakdown']['medium']}")
    print(f"Low: {report['severity_breakdown']['low']}")
    print(f"Check: {'✅ PASSED' if report['drift_check_passed'] else '❌ FAILED'}")
    
    if report['issues']:
        print("\n🚨 Issues Found:")
        for issue in report['issues']:
            print(f"  {issue['severity'].upper()}: {issue['message']}")
    
    # Exit with error if high-severity issues found
    if not report['drift_check_passed']:
        sys.exit(1)


if __name__ == "__main__":
    main()