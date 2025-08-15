#!/usr/bin/env python3
"""Advanced security scanner for Self-Healing Pipeline Guard."""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import logging


class SecurityScanner:
    """Advanced security scanner for Python codebases."""
    
    def __init__(self, project_root: str):
        """Initialize security scanner.
        
        Args:
            project_root: Root directory of the project to scan
        """
        self.project_root = Path(project_root)
        self.findings: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Security patterns to detect
        self.security_patterns = {
            'hardcoded_secrets': [
                r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{4,}["\']',
                r'(?i)(secret|token|key)\s*=\s*["\'][^"\']{8,}["\']',
                r'(?i)(api[_-]?key)\s*=\s*["\'][^"\']{8,}["\']',
                r'(?i)(auth[_-]?token)\s*=\s*["\'][^"\']{8,}["\']',
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'][^"\']*%[^"\']*["\']',
                r'execute\s*\(\s*["\'][^"\']*\+[^"\']*["\']',
                r'execute\s*\(\s*["\'][^"\']*\.format\(',
                r'cursor\.execute\s*\([^?]*%',
            ],
            'command_injection': [
                r'os\.system\s*\(',
                r'subprocess\.(run|call|Popen)\s*\([^)]*shell\s*=\s*True',
                r'eval\s*\(',
                r'exec\s*\(',
            ],
            'path_traversal': [
                r'open\s*\([^)]*\.\./.*["\']',
                r'file\s*=.*\.\./.*["\']',
            ],
            'unsafe_deserialization': [
                r'pickle\.loads?\s*\(',
                r'yaml\.load\s*\(',
                r'marshal\.loads?\s*\(',
            ]
        }
        
        # Allowed patterns (to reduce false positives)
        self.allowed_patterns = [
            r'example.*password',
            r'test.*password',
            r'placeholder.*secret',
            r'dummy.*key',
            r'#.*password',  # Comments
            r'""".*password.*"""',  # Docstrings
        ]
    
    def scan_project(self) -> Dict[str, Any]:
        """Scan the entire project for security issues.
        
        Returns:
            Security scan results
        """
        self.logger.info(f"Starting security scan of {self.project_root}")
        
        # Scan Python files
        python_files = list(self.project_root.rglob("*.py"))
        self.logger.info(f"Scanning {len(python_files)} Python files")
        
        for file_path in python_files:
            self._scan_file(file_path)
        
        # Additional scans
        self._scan_dependencies()
        self._scan_permissions()
        self._scan_configuration_files()
        
        return self._generate_report()
    
    def _scan_file(self, file_path: Path):
        """Scan a single Python file for security issues.
        
        Args:
            file_path: Path to the Python file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for various security patterns
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        if not self._is_allowed_match(match.group(0)):
                            self._add_finding(
                                category=category,
                                file_path=file_path,
                                line_number=content[:match.start()].count('\n') + 1,
                                match=match.group(0),
                                severity=self._get_severity(category)
                            )
            
            # AST-based checks
            try:
                tree = ast.parse(content)
                self._scan_ast(tree, file_path)
            except SyntaxError:
                self.logger.warning(f"Could not parse {file_path} - syntax error")
                
        except Exception as e:
            self.logger.error(f"Error scanning {file_path}: {e}")
    
    def _scan_ast(self, tree: ast.AST, file_path: Path):
        """Scan AST for security issues.
        
        Args:
            tree: AST tree
            file_path: Path to the file
        """
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                
                if func_name in ['eval', 'exec']:
                    self._add_finding(
                        category='dangerous_functions',
                        file_path=file_path,
                        line_number=node.lineno,
                        match=f"Call to {func_name}",
                        severity='high'
                    )
                
                elif func_name == 'open' and len(node.args) > 1:
                    # Check for unsafe file modes
                    if (isinstance(node.args[1], ast.Constant) and 
                        isinstance(node.args[1].value, str) and 
                        'w' in node.args[1].value):
                        # File writing detected - check if path is dynamic
                        if isinstance(node.args[0], ast.Name):
                            self._add_finding(
                                category='file_operations',
                                file_path=file_path,
                                line_number=node.lineno,
                                match="Dynamic file write operation",
                                severity='medium'
                            )
            
            # Check for imports of dangerous modules
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'subprocess', 'pickle']:
                        # Note: These are often necessary, so low severity
                        pass
    
    def _scan_dependencies(self):
        """Scan dependencies for known vulnerabilities."""
        requirements_files = [
            self.project_root / "requirements.txt",
            self.project_root / "requirements-prod.txt",
            self.project_root / "pyproject.toml"
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                self._scan_requirements_file(req_file)
    
    def _scan_requirements_file(self, req_file: Path):
        """Scan requirements file for vulnerable packages.
        
        Args:
            req_file: Path to requirements file
        """
        try:
            with open(req_file, 'r') as f:
                content = f.read()
            
            # Look for packages without version constraints
            loose_pins = re.finditer(r'^([a-zA-Z0-9_-]+)(?:>=|>|==)?$', content, re.MULTILINE)
            for match in loose_pins:
                self._add_finding(
                    category='dependency_security',
                    file_path=req_file,
                    line_number=content[:match.start()].count('\n') + 1,
                    match=f"Unpinned dependency: {match.group(1)}",
                    severity='low'
                )
                
        except Exception as e:
            self.logger.error(f"Error scanning requirements {req_file}: {e}")
    
    def _scan_permissions(self):
        """Scan file permissions for security issues."""
        sensitive_files = [
            "*.key", "*.pem", "*.crt", "*.p12", "*.pfx",
            ".env", ".env.*", "config/*.conf"
        ]
        
        for pattern in sensitive_files:
            for file_path in self.project_root.rglob(pattern):
                try:
                    stat = file_path.stat()
                    # Check if file is world-readable (on Unix systems)
                    if hasattr(stat, 'st_mode') and (stat.st_mode & 0o044):
                        self._add_finding(
                            category='file_permissions',
                            file_path=file_path,
                            line_number=0,
                            match="Sensitive file is world-readable",
                            severity='medium'
                        )
                except Exception:
                    pass
    
    def _scan_configuration_files(self):
        """Scan configuration files for security issues."""
        config_files = list(self.project_root.rglob("*.json")) + \
                      list(self.project_root.rglob("*.yaml")) + \
                      list(self.project_root.rglob("*.yml")) + \
                      list(self.project_root.rglob("*.toml"))
        
        for config_file in config_files:
            self._scan_config_file(config_file)
    
    def _scan_config_file(self, config_file: Path):
        """Scan a configuration file for security issues.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Look for potential secrets in config files
            secret_patterns = [
                r'(?i)(password|secret|key|token)\s*[:=]\s*["\'][^"\']{4,}["\']',
                r'(?i)(database_url|db_url)\s*[:=]\s*["\'][^"\']*://[^"\']*["\']',
            ]
            
            for pattern in secret_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    if not self._is_allowed_match(match.group(0)):
                        self._add_finding(
                            category='config_secrets',
                            file_path=config_file,
                            line_number=content[:match.start()].count('\n') + 1,
                            match=match.group(0),
                            severity='high'
                        )
                        
        except Exception as e:
            self.logger.error(f"Error scanning config file {config_file}: {e}")
    
    def _get_function_name(self, func_node: ast.AST) -> str:
        """Get function name from AST node.
        
        Args:
            func_node: AST function node
            
        Returns:
            Function name
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        else:
            return "unknown"
    
    def _is_allowed_match(self, match: str) -> bool:
        """Check if a match is in the allowed patterns.
        
        Args:
            match: The matched string
            
        Returns:
            True if match is allowed (false positive)
        """
        for pattern in self.allowed_patterns:
            if re.search(pattern, match, re.IGNORECASE):
                return True
        return False
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for a security category.
        
        Args:
            category: Security category
            
        Returns:
            Severity level
        """
        severity_map = {
            'hardcoded_secrets': 'high',
            'sql_injection': 'critical',
            'command_injection': 'critical',
            'path_traversal': 'high',
            'unsafe_deserialization': 'high',
            'dangerous_functions': 'high',
            'file_operations': 'medium',
            'dependency_security': 'low',
            'file_permissions': 'medium',
            'config_secrets': 'high',
        }
        return severity_map.get(category, 'medium')
    
    def _add_finding(self, category: str, file_path: Path, line_number: int,
                    match: str, severity: str):
        """Add a security finding.
        
        Args:
            category: Security category
            file_path: Path to the file
            line_number: Line number
            match: Matched content
            severity: Severity level
        """
        finding = {
            'category': category,
            'file_path': str(file_path.relative_to(self.project_root)),
            'line_number': line_number,
            'match': match[:100],  # Truncate long matches
            'severity': severity,
            'timestamp': os.path.getmtime(file_path)
        }
        self.findings.append(finding)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate security scan report.
        
        Returns:
            Security scan report
        """
        # Group findings by category
        by_category = {}
        by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for finding in self.findings:
            category = finding['category']
            severity = finding['severity']
            
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(finding)
            by_severity[severity] += 1
        
        # Calculate risk score
        risk_score = (
            by_severity['critical'] * 10 +
            by_severity['high'] * 5 +
            by_severity['medium'] * 2 +
            by_severity['low'] * 1
        )
        
        # Determine overall security level
        if by_severity['critical'] > 0:
            security_level = 'CRITICAL'
        elif by_severity['high'] > 0:
            security_level = 'HIGH_RISK'
        elif by_severity['medium'] > 0:
            security_level = 'MEDIUM_RISK'
        elif by_severity['low'] > 0:
            security_level = 'LOW_RISK'
        else:
            security_level = 'SECURE'
        
        return {
            'scan_summary': {
                'total_findings': len(self.findings),
                'by_severity': by_severity,
                'by_category': {k: len(v) for k, v in by_category.items()},
                'risk_score': risk_score,
                'security_level': security_level,
                'files_scanned': len(list(self.project_root.rglob("*.py")))
            },
            'findings': self.findings,
            'recommendations': self._generate_recommendations(by_category, by_severity)
        }
    
    def _generate_recommendations(self, by_category: Dict, by_severity: Dict) -> List[str]:
        """Generate security recommendations.
        
        Args:
            by_category: Findings grouped by category
            by_severity: Findings count by severity
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if 'hardcoded_secrets' in by_category:
            recommendations.append(
                "Remove hardcoded secrets and use environment variables or secure vaults"
            )
        
        if 'sql_injection' in by_category:
            recommendations.append(
                "Use parameterized queries to prevent SQL injection attacks"
            )
        
        if 'command_injection' in by_category:
            recommendations.append(
                "Avoid shell=True in subprocess calls and validate all inputs"
            )
        
        if 'dangerous_functions' in by_category:
            recommendations.append(
                "Replace eval()/exec() calls with safer alternatives"
            )
        
        if 'dependency_security' in by_category:
            recommendations.append(
                "Pin dependency versions and regularly update packages"
            )
        
        if by_severity['critical'] > 0 or by_severity['high'] > 0:
            recommendations.append(
                "Address critical and high severity issues immediately"
            )
        
        recommendations.extend([
            "Implement security code review processes",
            "Add security linting to CI/CD pipeline",
            "Regular security scans and penetration testing",
            "Implement proper logging and monitoring for security events"
        ])
        
        return recommendations


def main():
    """CLI for security scanner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Scanner for Pipeline Guard")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", help="Output file for scan results (JSON)")
    parser.add_argument("--format", choices=['json', 'text'], default='text', 
                       help="Output format")
    parser.add_argument("--severity", choices=['low', 'medium', 'high', 'critical'],
                       help="Minimum severity level to report")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run security scan
    scanner = SecurityScanner(args.project_root)
    report = scanner.scan_project()
    
    # Filter by severity if specified
    if args.severity:
        severity_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        min_severity = severity_order[args.severity]
        
        filtered_findings = [
            finding for finding in report['findings']
            if severity_order.get(finding['severity'], 0) >= min_severity
        ]
        report['findings'] = filtered_findings
        report['scan_summary']['total_findings'] = len(filtered_findings)
    
    # Output results
    if args.format == 'json':
        output = json.dumps(report, indent=2)
    else:
        output = format_text_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Security scan results saved to {args.output}")
    else:
        print(output)


def format_text_report(report: Dict[str, Any]) -> str:
    """Format security report as text.
    
    Args:
        report: Security scan report
        
    Returns:
        Formatted text report
    """
    output = []
    summary = report['scan_summary']
    
    output.append("🔒 SECURITY SCAN REPORT")
    output.append("=" * 50)
    output.append(f"Security Level: {summary['security_level']}")
    output.append(f"Risk Score: {summary['risk_score']}")
    output.append(f"Total Findings: {summary['total_findings']}")
    output.append(f"Files Scanned: {summary['files_scanned']}")
    
    output.append("\nFindings by Severity:")
    for severity, count in summary['by_severity'].items():
        if count > 0:
            output.append(f"  {severity.upper()}: {count}")
    
    output.append("\nFindings by Category:")
    for category, count in summary['by_category'].items():
        output.append(f"  {category}: {count}")
    
    # Show sample findings
    if report['findings']:
        output.append("\nSample Findings:")
        for finding in report['findings'][:10]:  # Show first 10
            output.append(
                f"  [{finding['severity'].upper()}] {finding['category']} "
                f"in {finding['file_path']}:{finding['line_number']}"
            )
            output.append(f"    {finding['match']}")
    
    # Show recommendations
    output.append("\nRecommendations:")
    for rec in report['recommendations']:
        output.append(f"  - {rec}")
    
    return '\n'.join(output)


if __name__ == "__main__":
    main()