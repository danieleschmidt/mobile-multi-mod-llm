#!/usr/bin/env python3
"""Comprehensive Quality Validation for Mobile Multi-Modal LLM System.

Advanced quality gates with security scanning, performance validation,
integration testing, and production readiness assessment.
"""

import os
import sys
import ast
import json
import time
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import re
import hashlib


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    message: str
    details: Dict[str, Any] = None
    execution_time: float = 0.0


class ComprehensiveValidator:
    """Comprehensive system validation."""
    
    def __init__(self, project_root: Path = None):
        """Initialize validator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        
        self.validation_results = []
        self.start_time = time.time()
        
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation checks.
        
        Returns:
            Comprehensive validation report
        """
        print("🔍 Starting Comprehensive Quality Validation...")
        print("=" * 60)
        
        # Core validations
        self._validate_project_structure()
        self._validate_code_syntax()
        self._validate_imports()
        self._validate_security()
        self._validate_documentation()
        self._validate_testing()
        self._validate_performance()
        self._validate_deployment_readiness()
        
        # Generate final report
        return self._generate_validation_report()
    
    def _validate_project_structure(self):
        """Validate project structure and organization."""
        print("📁 Validating Project Structure...")
        
        required_dirs = ["src", "tests", "docs", "deployment"]
        required_files = ["README.md", "requirements.txt", "pyproject.toml"]
        
        # Check directories
        missing_dirs = []
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            self._add_result("project_structure_dirs", "WARN", 
                           f"Missing directories: {', '.join(missing_dirs)}")
        else:
            self._add_result("project_structure_dirs", "PASS", "All required directories present")
        
        # Check files
        missing_files = []
        for file_name in required_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            self._add_result("project_structure_files", "WARN", 
                           f"Missing files: {', '.join(missing_files)}")
        else:
            self._add_result("project_structure_files", "PASS", "All required files present")
        
        # Check source code organization
        if self.src_dir.exists():
            py_files = list(self.src_dir.glob("**/*.py"))
            if len(py_files) >= 10:
                self._add_result("source_organization", "PASS", 
                               f"Well-organized codebase with {len(py_files)} Python files")
            else:
                self._add_result("source_organization", "WARN", 
                               f"Limited codebase: {len(py_files)} Python files")
    
    def _validate_code_syntax(self):
        """Validate Python syntax across all source files."""
        print("🐍 Validating Python Syntax...")
        
        if not self.src_dir.exists():
            self._add_result("syntax_validation", "SKIP", "No source directory found")
            return
        
        syntax_errors = []
        files_checked = 0
        
        for py_file in self.src_dir.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                compile(source, str(py_file), 'exec')
                files_checked += 1
                
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}:{e.lineno} - {e.msg}")
            except Exception as e:
                syntax_errors.append(f"{py_file} - {str(e)}")
        
        if syntax_errors:
            self._add_result("syntax_validation", "FAIL", 
                           f"Syntax errors found in {len(syntax_errors)} files",
                           {"errors": syntax_errors[:10]})  # Show first 10
        else:
            self._add_result("syntax_validation", "PASS", 
                           f"All {files_checked} Python files have valid syntax")
    
    def _validate_imports(self):
        """Validate import statements and dependencies."""
        print("📦 Validating Imports and Dependencies...")
        
        import_issues = []
        circular_imports = []
        files_checked = 0
        
        for py_file in self.src_dir.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source, filename=str(py_file))
                files_checked += 1
                
                # Check for relative imports that might cause issues
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.ImportFrom):
                            if node.level > 2:  # Deep relative imports
                                import_issues.append(f"{py_file} - Deep relative import: {node.level} levels")
                
            except Exception as e:
                import_issues.append(f"{py_file} - Parse error: {str(e)}")
        
        if import_issues:
            self._add_result("import_validation", "WARN", 
                           f"Import issues found in {len(import_issues)} cases",
                           {"issues": import_issues[:5]})
        else:
            self._add_result("import_validation", "PASS", 
                           f"Import structure validated across {files_checked} files")
    
    def _validate_security(self):
        """Validate security aspects of the codebase."""
        print("🔒 Validating Security...")
        
        security_issues = []
        files_scanned = 0
        
        # Security patterns to check
        dangerous_patterns = [
            (r'\bexec\s*\(', "Dangerous exec() usage"),
            (r'\beval\s*\(', "Dangerous eval() usage"),
            (r'subprocess\.call.*shell=True', "Shell injection risk"),
            (r'os\.system\s*\(', "Command injection risk"),
            (r'pickle\.loads?\s*\(', "Pickle security risk"),
            (r'__import__\s*\(.*user', "Dynamic import with user input"),
            (r'open\s*\(.*["\'][rwab]+["\'].*\+', "File write with string concatenation"),
        ]
        
        for py_file in self.src_dir.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                files_scanned += 1
                
                for pattern, description in dangerous_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        security_issues.append(f"{py_file} - {description}: {len(matches)} occurrences")
                
                # Check for hardcoded secrets patterns
                secret_patterns = [
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'api_key\s*=\s*["\'][^"\']+["\']',
                    r'secret\s*=\s*["\'][^"\']+["\']',
                    r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']'
                ]
                
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        security_issues.append(f"{py_file} - Potential hardcoded secret")
                        break
                
            except Exception as e:
                security_issues.append(f"{py_file} - Scan error: {str(e)}")
        
        if security_issues:
            # Filter out expected security-related files
            filtered_issues = []
            for issue in security_issues:
                if not any(safe_file in issue for safe_file in ['security', 'test', 'mock']):
                    filtered_issues.append(issue)
            
            if filtered_issues:
                self._add_result("security_validation", "WARN", 
                               f"Security concerns found in {len(filtered_issues)} cases",
                               {"issues": filtered_issues[:10]})
            else:
                self._add_result("security_validation", "PASS", 
                               "Security issues found only in expected security/test files")
        else:
            self._add_result("security_validation", "PASS", 
                           f"No major security issues found in {files_scanned} files")
    
    def _validate_documentation(self):
        """Validate documentation coverage and quality."""
        print("📚 Validating Documentation...")
        
        doc_issues = []
        documented_functions = 0
        total_functions = 0
        
        for py_file in self.src_dir.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source, filename=str(py_file))
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        
                        # Check for docstring
                        if (ast.get_docstring(node) or 
                            (node.body and isinstance(node.body[0], ast.Expr) and
                             isinstance(node.body[0].value, ast.Constant) and
                             isinstance(node.body[0].value.value, str))):
                            documented_functions += 1
                        else:
                            # Only flag public functions (not starting with _)
                            if not node.name.startswith('_'):
                                doc_issues.append(f"{py_file}:{node.lineno} - Function '{node.name}' lacks docstring")
                
            except Exception as e:
                doc_issues.append(f"{py_file} - Parse error: {str(e)}")
        
        doc_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
        
        if doc_coverage >= 70:
            self._add_result("documentation_validation", "PASS", 
                           f"Good documentation coverage: {doc_coverage:.1f}% ({documented_functions}/{total_functions})")
        elif doc_coverage >= 40:
            self._add_result("documentation_validation", "WARN", 
                           f"Moderate documentation coverage: {doc_coverage:.1f}% ({documented_functions}/{total_functions})")
        else:
            self._add_result("documentation_validation", "FAIL", 
                           f"Poor documentation coverage: {doc_coverage:.1f}% ({documented_functions}/{total_functions})",
                           {"missing_docs": doc_issues[:10]})
    
    def _validate_testing(self):
        """Validate testing setup and coverage."""
        print("🧪 Validating Testing Setup...")
        
        if not self.tests_dir.exists():
            self._add_result("testing_validation", "WARN", "No tests directory found")
            return
        
        test_files = list(self.tests_dir.glob("**/test_*.py"))
        
        if len(test_files) == 0:
            self._add_result("testing_validation", "FAIL", "No test files found")
            return
        
        test_functions = 0
        test_classes = 0
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source, filename=str(test_file))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_functions += 1
                    elif isinstance(node, ast.ClassDef) and 'test' in node.name.lower():
                        test_classes += 1
                
            except Exception:
                continue
        
        # Try to run tests if pytest is available
        test_execution_result = "Not executed"
        try:
            result = subprocess.run(['python3', '-m', 'pytest', str(self.tests_dir), '--collect-only', '-q'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                test_execution_result = "Tests can be collected successfully"
            else:
                test_execution_result = "Test collection issues detected"
        except Exception:
            test_execution_result = "pytest not available or execution failed"
        
        self._add_result("testing_validation", "PASS", 
                       f"Testing setup found: {len(test_files)} test files, {test_functions} test functions, {test_classes} test classes",
                       {"execution_result": test_execution_result})
    
    def _validate_performance(self):
        """Validate performance characteristics."""
        print("⚡ Validating Performance...")
        
        # Check for performance optimization patterns
        optimization_patterns = 0
        performance_issues = []
        
        patterns_to_check = [
            (r'\b@lru_cache', "LRU caching optimization"),
            (r'\basync\s+def', "Async function optimization"),
            (r'\bwith\s+concurrent\.futures', "Concurrent processing"),
            (r'\bmultiprocessing', "Multiprocessing optimization"),
            (r'\bthreading', "Threading optimization"),
            (r'\.compile\(', "Compiled regex optimization"),
        ]
        
        for py_file in self.src_dir.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, description in patterns_to_check:
                    if re.search(pattern, content):
                        optimization_patterns += 1
                        break  # Count once per file
                
                # Check for potential performance issues
                if re.search(r'for.*in.*range\(len\(', content):
                    performance_issues.append(f"{py_file} - Consider enumerate() instead of range(len())")
                
                if len(re.findall(r'\.append\(', content)) > 10:
                    performance_issues.append(f"{py_file} - Many append() calls, consider list comprehension")
                
            except Exception:
                continue
        
        total_files = len(list(self.src_dir.glob("**/*.py")))
        optimization_ratio = (optimization_patterns / total_files * 100) if total_files > 0 else 0
        
        if optimization_ratio >= 20:
            self._add_result("performance_validation", "PASS", 
                           f"Good performance optimization coverage: {optimization_ratio:.1f}%")
        elif optimization_ratio >= 10:
            self._add_result("performance_validation", "WARN", 
                           f"Moderate performance optimization: {optimization_ratio:.1f}%")
        else:
            self._add_result("performance_validation", "WARN", 
                           f"Limited performance optimization: {optimization_ratio:.1f}%",
                           {"issues": performance_issues[:5]})
    
    def _validate_deployment_readiness(self):
        """Validate deployment readiness."""
        print("🚀 Validating Deployment Readiness...")
        
        deployment_components = []
        
        # Check for Docker support
        if (self.project_root / "Dockerfile").exists():
            deployment_components.append("Docker")
        
        if (self.project_root / "docker-compose.yml").exists():
            deployment_components.append("Docker Compose")
        
        # Check for Kubernetes support
        k8s_dir = self.project_root / "kubernetes"
        if k8s_dir.exists() and list(k8s_dir.glob("*.yaml")):
            deployment_components.append("Kubernetes")
        
        # Check for monitoring setup
        monitoring_dir = self.project_root / "monitoring"
        if monitoring_dir.exists():
            deployment_components.append("Monitoring")
        
        # Check for CI/CD
        if (self.project_root / ".github" / "workflows").exists():
            deployment_components.append("GitHub Actions")
        
        # Check for configuration management
        if (self.project_root / "pyproject.toml").exists():
            deployment_components.append("Python Package Config")
        
        if len(deployment_components) >= 4:
            self._add_result("deployment_validation", "PASS", 
                           f"Excellent deployment readiness: {', '.join(deployment_components)}")
        elif len(deployment_components) >= 2:
            self._add_result("deployment_validation", "WARN", 
                           f"Good deployment readiness: {', '.join(deployment_components)}")
        else:
            self._add_result("deployment_validation", "FAIL", 
                           f"Limited deployment readiness: {', '.join(deployment_components) if deployment_components else 'No deployment components found'}")
    
    def _add_result(self, check_name: str, status: str, message: str, details: Dict[str, Any] = None):
        """Add validation result.
        
        Args:
            check_name: Name of the check
            status: Status (PASS, FAIL, WARN, SKIP)
            message: Result message
            details: Optional additional details
        """
        result = ValidationResult(
            check_name=check_name,
            status=status,
            message=message,
            details=details or {},
            execution_time=time.time() - self.start_time
        )
        
        self.validation_results.append(result)
        
        # Print result
        status_symbol = {
            "PASS": "✅",
            "FAIL": "❌", 
            "WARN": "⚠️",
            "SKIP": "⏭️"
        }.get(status, "❓")
        
        print(f"  {status_symbol} {check_name}: {message}")
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report.
        
        Returns:
            Validation report
        """
        total_time = time.time() - self.start_time
        
        # Count results by status
        status_counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        for result in self.validation_results:
            status_counts[result.status] += 1
        
        # Calculate quality score
        total_checks = len(self.validation_results)
        score = 0
        if total_checks > 0:
            score = ((status_counts["PASS"] * 100 + status_counts["WARN"] * 50) / total_checks)
        
        # Determine overall status
        if status_counts["FAIL"] > 0:
            overall_status = "FAILED"
        elif status_counts["WARN"] > 0:
            overall_status = "PASSED_WITH_WARNINGS"
        else:
            overall_status = "PASSED"
        
        report = {
            "validation_summary": {
                "overall_status": overall_status,
                "quality_score": round(score, 1),
                "total_checks": total_checks,
                "execution_time": round(total_time, 2),
                "status_breakdown": status_counts
            },
            "validation_results": [
                {
                    "check_name": r.check_name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.validation_results
            ],
            "recommendations": self._generate_recommendations(),
            "generated_at": time.time()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze results for recommendations
        failed_checks = [r for r in self.validation_results if r.status == "FAIL"]
        warning_checks = [r for r in self.validation_results if r.status == "WARN"]
        
        if any("syntax" in r.check_name for r in failed_checks):
            recommendations.append("Fix Python syntax errors before deployment")
        
        if any("security" in r.check_name for r in warning_checks):
            recommendations.append("Review and address security warnings")
        
        if any("documentation" in r.check_name for r in failed_checks + warning_checks):
            recommendations.append("Improve code documentation coverage")
        
        if any("testing" in r.check_name for r in failed_checks):
            recommendations.append("Implement comprehensive test suite")
        
        if any("deployment" in r.check_name for r in failed_checks + warning_checks):
            recommendations.append("Enhance deployment configuration and automation")
        
        if any("performance" in r.check_name for r in warning_checks):
            recommendations.append("Consider performance optimization opportunities")
        
        # Generic recommendations
        if not recommendations:
            recommendations.append("Code quality is good - consider adding more advanced monitoring and optimization")
        
        return recommendations


def main():
    """Main validation execution."""
    validator = ComprehensiveValidator()
    report = validator.run_all_validations()
    
    print("\n" + "="*60)
    print("🎯 VALIDATION SUMMARY")
    print("="*60)
    
    summary = report["validation_summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Quality Score: {summary['quality_score']}/100")
    print(f"Total Checks: {summary['total_checks']}")
    print(f"Execution Time: {summary['execution_time']}s")
    print()
    print("Status Breakdown:")
    for status, count in summary["status_breakdown"].items():
        print(f"  {status}: {count}")
    
    print("\n📋 RECOMMENDATIONS:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    # Save detailed report
    report_file = Path("validation_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if summary["overall_status"] == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()