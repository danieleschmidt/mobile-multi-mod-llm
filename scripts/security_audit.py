#!/usr/bin/env python3
"""
Advanced Security Audit Script for Mobile Multi-Modal LLM

This script performs comprehensive security analysis beyond standard tools,
focusing on AI/ML specific security concerns and mobile deployment risks.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class SecurityAuditor:
    """Advanced security auditor for ML/mobile applications."""
    
    def __init__(self):
        self.issues: List[Dict] = []
        self.high_risk_patterns = {
            'hardcoded_credentials': [
                'password', 'secret', 'key', 'token', 'api_key',
                'private_key', 'auth_token', 'access_token'
            ],
            'unsafe_ml_ops': [
                'pickle.loads', 'torch.load', 'joblib.load',
                'eval(', 'exec(', '__import__'
            ],
            'mobile_security_risks': [
                'open(', 'file(', 'subprocess.call', 'os.system',
                'shell=True'
            ]
        }
    
    def audit_file(self, file_path: Path) -> None:
        """Audit a single Python file for security issues."""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            self._check_imports(tree, file_path)
            self._check_string_literals(tree, file_path, content)
            self._check_function_calls(tree, file_path)
            
        except Exception as e:
            self._add_issue(
                'parse_error',
                f"Failed to parse {file_path}: {e}",
                str(file_path),
                0,
                'medium'
            )
    
    def _check_imports(self, tree: ast.AST, file_path: Path) -> None:
        """Check for risky imports."""
        risky_imports = {
            'pickle': 'Pickle deserialization can execute arbitrary code',
            'marshal': 'Marshal can execute arbitrary code during loading',
            'subprocess': 'Subprocess calls can lead to command injection',
            'os': 'OS module provides system access - use carefully'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in risky_imports:
                        self._add_issue(
                            'risky_import',
                            f"Risky import: {alias.name} - {risky_imports[alias.name]}",
                            str(file_path),
                            node.lineno,
                            'medium'
                        )
    
    def _check_string_literals(self, tree: ast.AST, file_path: Path, content: str) -> None:
        """Check string literals for embedded secrets."""
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self._analyze_string_for_secrets(
                    node.value, str(file_path), node.lineno, lines
                )
    
    def _analyze_string_for_secrets(self, value: str, file_path: str, 
                                  line_no: int, lines: List[str]) -> None:
        """Analyze string for potential secrets."""
        value_lower = value.lower()
        
        # Check for potential API keys or tokens
        if len(value) > 20 and any(char.isalnum() for char in value):
            if any(pattern in value_lower for pattern in self.high_risk_patterns['hardcoded_credentials']):
                context = lines[line_no - 1] if line_no <= len(lines) else ""
                if '=' in context or ':' in context:
                    self._add_issue(
                        'potential_hardcoded_secret',
                        f"Potential hardcoded secret in string literal",
                        file_path,
                        line_no,
                        'high'
                    )
        
        # Check for SQL injection patterns
        sql_patterns = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP']
        if any(pattern in value.upper() for pattern in sql_patterns) and '%' in value:
            self._add_issue(
                'sql_injection_risk',
                "Potential SQL injection vulnerability - use parameterized queries",
                file_path,
                line_no,
                'high'
            )
    
    def _check_function_calls(self, tree: ast.AST, file_path: Path) -> None:
        """Check for risky function calls."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name:
                    self._analyze_function_call(func_name, str(file_path), node.lineno)
    
    def _get_function_name(self, node: ast.Call) -> str:
        """Extract function name from call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return f"{self._get_attr_chain(node.func)}"
        return ""
    
    def _get_attr_chain(self, node: ast.Attribute) -> str:
        """Get full attribute chain (e.g., torch.load)."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attr_chain(node.value)}.{node.attr}"
        return node.attr
    
    def _analyze_function_call(self, func_name: str, file_path: str, line_no: int) -> None:
        """Analyze function call for security risks."""
        # Check for unsafe ML operations
        if any(pattern in func_name for pattern in self.high_risk_patterns['unsafe_ml_ops']):
            self._add_issue(
                'unsafe_ml_operation',
                f"Unsafe ML operation: {func_name} - can execute arbitrary code",
                file_path,
                line_no,
                'high'
            )
        
        # Check for mobile security risks
        if any(pattern in func_name for pattern in self.high_risk_patterns['mobile_security_risks']):
            self._add_issue(
                'mobile_security_risk',
                f"Mobile security risk: {func_name} - potential system access",
                file_path,
                line_no,
                'medium'
            )
    
    def _add_issue(self, issue_type: str, message: str, file_path: str, 
                   line_no: int, severity: str) -> None:
        """Add a security issue to the list."""
        self.issues.append({
            'type': issue_type,
            'message': message,
            'file': file_path,
            'line': line_no,
            'severity': severity
        })
    
    def generate_report(self) -> Dict:
        """Generate security audit report."""
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for issue in self.issues:
            severity_counts[issue['severity']] += 1
        
        return {
            'total_issues': len(self.issues),
            'severity_breakdown': severity_counts,
            'issues': self.issues,
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get security recommendations based on found issues."""
        recommendations = []
        
        if any(issue['type'] == 'potential_hardcoded_secret' for issue in self.issues):
            recommendations.append(
                "Use environment variables or secure key management for secrets"
            )
        
        if any(issue['type'] == 'unsafe_ml_operation' for issue in self.issues):
            recommendations.append(
                "Replace unsafe serialization with secure alternatives like safetensors"
            )
        
        if any(issue['type'] == 'mobile_security_risk' for issue in self.issues):
            recommendations.append(
                "Minimize system access in mobile deployments - use sandboxed operations"
            )
        
        return recommendations


def main():
    """Main security audit function."""
    auditor = SecurityAuditor()
    
    # Audit Python files in key directories
    for directory in ['src', 'tests', 'scripts']:
        if Path(directory).exists():
            for py_file in Path(directory).rglob('*.py'):
                auditor.audit_file(py_file)
    
    # Generate and save report
    report = auditor.generate_report()
    
    # Save detailed report
    report_path = Path('reports/security-audit-detailed.json')
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("🔒 Advanced Security Audit Results")
    print(f"Total Issues: {report['total_issues']}")
    print(f"High: {report['severity_breakdown']['high']}")
    print(f"Medium: {report['severity_breakdown']['medium']}")
    print(f"Low: {report['severity_breakdown']['low']}")
    
    if report['issues']:
        print("\n🚨 Issues Found:")
        for issue in report['issues'][:5]:  # Show first 5 issues
            print(f"  {issue['severity'].upper()}: {issue['message']} ({issue['file']}:{issue['line']})")
        
        if len(report['issues']) > 5:
            print(f"  ... and {len(report['issues']) - 5} more issues")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Exit with error code if high-severity issues found
    if report['severity_breakdown']['high'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()