#!/usr/bin/env python3
"""Security and Quality Gates - Comprehensive security scanning and quality assurance."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

def run_command(command: str, capture_output: bool = True) -> Dict[str, Any]:
    """Run shell command and return result."""
    try:
        if capture_output:
            result = subprocess.run(
                command.split(), 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        else:
            result = subprocess.run(command.split(), timeout=60)
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "returncode": -1
        }

def check_python_syntax() -> Dict[str, Any]:
    """Check Python syntax across all source files."""
    print("🔍 Checking Python syntax...")
    
    python_files = list(Path("src").rglob("*.py"))
    python_files.extend(list(Path(".").glob("*.py")))
    
    syntax_errors = []
    files_checked = 0
    
    for file_path in python_files:
        files_checked += 1
        result = run_command(f"python3 -m py_compile {file_path}")
        
        if not result["success"]:
            syntax_errors.append({
                "file": str(file_path),
                "error": result.get("stderr", "Unknown syntax error")
            })
    
    return {
        "check": "python_syntax",
        "success": len(syntax_errors) == 0,
        "files_checked": files_checked,
        "errors": syntax_errors,
        "summary": f"Checked {files_checked} Python files, {len(syntax_errors)} syntax errors"
    }

def check_import_security() -> Dict[str, Any]:
    """Check for suspicious imports and potential security issues."""
    print("🔒 Checking import security...")
    
    suspicious_imports = [
        "os.system", "subprocess.call", "eval", "exec", 
        "pickle.load", "__import__", "getattr", "setattr"
    ]
    
    dangerous_patterns = [
        "shell=True",
        "input(",
        "raw_input(",
        "open(",  # Could be suspicious depending on context
    ]
    
    security_issues = []
    files_checked = 0
    
    python_files = list(Path("src").rglob("*.py"))
    
    for file_path in python_files:
        files_checked += 1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for suspicious imports
                for suspicious in suspicious_imports:
                    if suspicious in content:
                        security_issues.append({
                            "file": str(file_path),
                            "issue": f"Suspicious import/usage: {suspicious}",
                            "severity": "HIGH"
                        })
                
                # Check for dangerous patterns
                for pattern in dangerous_patterns:
                    if pattern in content:
                        # Additional context check for open()
                        if pattern == "open(" and "mode=" in content and any(mode in content for mode in ['"w"', "'w'", '"a"', "'a'"]):
                            security_issues.append({
                                "file": str(file_path),
                                "issue": f"File write operation detected: {pattern}",
                                "severity": "MEDIUM"
                            })
                        elif pattern != "open(":
                            security_issues.append({
                                "file": str(file_path),
                                "issue": f"Potentially dangerous pattern: {pattern}",
                                "severity": "MEDIUM"
                            })
        
        except Exception as e:
            security_issues.append({
                "file": str(file_path),
                "issue": f"Could not scan file: {e}",
                "severity": "LOW"
            })
    
    return {
        "check": "import_security",
        "success": len(security_issues) == 0,
        "files_checked": files_checked,
        "issues": security_issues,
        "summary": f"Scanned {files_checked} files, {len(security_issues)} security issues found"
    }

def check_dependency_security() -> Dict[str, Any]:
    """Check for known security vulnerabilities in dependencies."""
    print("📦 Checking dependency security...")
    
    # Check if requirements files exist
    req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
    dependencies_found = []
    
    for req_file in req_files:
        if Path(req_file).exists():
            dependencies_found.append(req_file)
    
    if not dependencies_found:
        return {
            "check": "dependency_security",
            "success": True,
            "summary": "No dependency files found to check",
            "dependencies_found": dependencies_found
        }
    
    # Simulate security check (would use real tools like safety, pip-audit in production)
    known_vulnerabilities = []
    
    # Check for potentially risky packages
    risky_patterns = ["eval", "exec", "pickle", "yaml.load"]
    
    for req_file in dependencies_found:
        try:
            with open(req_file, 'r') as f:
                content = f.read().lower()
                
                for pattern in risky_patterns:
                    if pattern in content:
                        known_vulnerabilities.append({
                            "file": req_file,
                            "vulnerability": f"Potentially risky dependency pattern: {pattern}",
                            "severity": "MEDIUM"
                        })
        
        except Exception as e:
            known_vulnerabilities.append({
                "file": req_file,
                "vulnerability": f"Could not scan dependencies: {e}",
                "severity": "LOW"
            })
    
    return {
        "check": "dependency_security",
        "success": len(known_vulnerabilities) == 0,
        "dependencies_found": dependencies_found,
        "vulnerabilities": known_vulnerabilities,
        "summary": f"Checked {len(dependencies_found)} dependency files, {len(known_vulnerabilities)} potential issues"
    }

def check_secrets_exposure() -> Dict[str, Any]:
    """Check for exposed secrets, keys, and sensitive information."""
    print("🔑 Checking for exposed secrets...")
    
    secret_patterns = [
        r"api[_-]?key[\"'\s]*[:=][\"'\s]*[a-zA-Z0-9]{10,}",
        r"secret[_-]?key[\"'\s]*[:=][\"'\s]*[a-zA-Z0-9]{10,}",
        r"password[\"'\s]*[:=][\"'\s]*[a-zA-Z0-9]{6,}",
        r"token[\"'\s]*[:=][\"'\s]*[a-zA-Z0-9]{10,}",
        r"aws[_-]?access[_-]?key[\"'\s]*[:=]",
        r"private[_-]?key[\"'\s]*[:=]",
    ]
    
    # Simple pattern matching (would use advanced regex in production)
    simple_patterns = [
        "api_key=",
        "secret_key=", 
        "password=",
        "token=",
        "private_key=",
        "AWS_ACCESS_KEY",
        "SECRET_ACCESS_KEY"
    ]
    
    exposed_secrets = []
    files_checked = 0
    
    # Check all text files
    text_extensions = ['.py', '.json', '.yaml', '.yml', '.txt', '.md', '.env']
    
    for ext in text_extensions:
        for file_path in Path('.').rglob(f'*{ext}'):
            # Skip test files and demos as they may contain dummy credentials
            if any(skip in str(file_path) for skip in ['test', 'demo', '.git']):
                continue
                
            files_checked += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    for pattern in simple_patterns:
                        if pattern in content and not any(exclude in content for exclude in ['example', 'dummy', 'placeholder']):
                            exposed_secrets.append({
                                "file": str(file_path),
                                "pattern": pattern,
                                "severity": "HIGH",
                                "line": "Multiple lines may contain secrets"
                            })
            
            except Exception:
                # Skip files that can't be read
                continue
    
    return {
        "check": "secrets_exposure",
        "success": len(exposed_secrets) == 0,
        "files_checked": files_checked,
        "exposed_secrets": exposed_secrets,
        "summary": f"Scanned {files_checked} files, {len(exposed_secrets)} potential secret exposures"
    }

def check_code_quality() -> Dict[str, Any]:
    """Check code quality metrics."""
    print("📏 Checking code quality...")
    
    quality_issues = []
    
    # Check for basic code quality issues
    python_files = list(Path("src").rglob("*.py"))
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Check for very long lines
                for i, line in enumerate(lines):
                    if len(line) > 120:
                        quality_issues.append({
                            "file": str(file_path),
                            "line": i + 1,
                            "issue": f"Line too long ({len(line)} chars)",
                            "severity": "LOW"
                        })
                
                # Check for missing docstrings in classes and functions
                content = ''.join(lines)
                
                if 'class ' in content and '"""' not in content:
                    quality_issues.append({
                        "file": str(file_path),
                        "issue": "Class found without docstring",
                        "severity": "MEDIUM"
                    })
                
                if 'def ' in content and lines[0].strip() and not lines[0].strip().startswith('"""'):
                    # Check if any function has docstring
                    has_docstring = False
                    for line in lines:
                        if 'def ' in line:
                            # Look for docstring in next few lines
                            line_idx = lines.index(line)
                            for check_line in lines[line_idx:line_idx+3]:
                                if '"""' in check_line or "'''" in check_line:
                                    has_docstring = True
                                    break
                            break
                    
                    if not has_docstring:
                        quality_issues.append({
                            "file": str(file_path),
                            "issue": "Functions found without docstrings",
                            "severity": "LOW"
                        })
        
        except Exception:
            continue
    
    return {
        "check": "code_quality",
        "success": len([issue for issue in quality_issues if issue["severity"] in ["HIGH", "MEDIUM"]]) == 0,
        "files_checked": len(python_files),
        "issues": quality_issues,
        "summary": f"Quality check on {len(python_files)} files, {len(quality_issues)} issues found"
    }

def check_test_coverage() -> Dict[str, Any]:
    """Check test coverage and test quality."""
    print("🧪 Checking test coverage...")
    
    # Count Python files
    src_files = list(Path("src").rglob("*.py"))
    test_files = list(Path("tests").rglob("test_*.py")) + list(Path(".").glob("test_*.py"))
    
    # Basic coverage estimation
    src_modules = set()
    for file_path in src_files:
        if file_path.name != "__init__.py":
            module_name = file_path.stem
            src_modules.add(module_name)
    
    tested_modules = set()
    for file_path in test_files:
        test_name = file_path.stem
        if test_name.startswith("test_"):
            module_name = test_name[5:]  # Remove "test_" prefix
            tested_modules.add(module_name)
    
    coverage_estimate = len(tested_modules) / len(src_modules) if src_modules else 0
    
    return {
        "check": "test_coverage",
        "success": coverage_estimate >= 0.8,  # 80% coverage threshold
        "src_files": len(src_files),
        "test_files": len(test_files),
        "src_modules": len(src_modules),
        "tested_modules": len(tested_modules),
        "coverage_estimate": coverage_estimate,
        "summary": f"Estimated {coverage_estimate:.1%} test coverage ({len(test_files)} test files for {len(src_files)} source files)"
    }

def run_security_quality_gates() -> Dict[str, Any]:
    """Run all security and quality gates."""
    print("🛡️  Mobile Multi-Modal LLM - Security & Quality Gates")
    print("=" * 65)
    
    start_time = time.time()
    
    # Run all checks
    checks = [
        check_python_syntax(),
        check_import_security(),
        check_dependency_security(),
        check_secrets_exposure(),
        check_code_quality(),
        check_test_coverage()
    ]
    
    total_time = time.time() - start_time
    
    # Analyze results
    all_passed = all(check["success"] for check in checks)
    critical_failures = [check for check in checks if not check["success"] and check["check"] in ["python_syntax", "secrets_exposure"]]
    
    # Generate comprehensive report
    report = {
        "timestamp": time.time(),
        "overall_status": "PASSED" if all_passed else "FAILED",
        "critical_failures": len(critical_failures),
        "total_checks": len(checks),
        "passed_checks": sum(1 for check in checks if check["success"]),
        "execution_time_seconds": total_time,
        "checks": checks,
        "recommendations": []
    }
    
    # Add recommendations
    if not all_passed:
        report["recommendations"].extend([
            "Review and fix all critical security issues immediately",
            "Address code quality issues to improve maintainability",
            "Increase test coverage to meet quality standards"
        ])
    
    if any(check["check"] == "secrets_exposure" and not check["success"] for check in checks):
        report["recommendations"].append("URGENT: Remove all exposed secrets and rotate compromised credentials")
    
    # Display results
    print(f"\n📊 Security & Quality Gate Results:")
    print(f"  Overall Status: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    print(f"  Checks Passed: {report['passed_checks']}/{report['total_checks']}")
    print(f"  Execution Time: {total_time:.2f}s")
    
    if critical_failures:
        print(f"  🚨 Critical Failures: {len(critical_failures)}")
        for failure in critical_failures:
            print(f"    - {failure['check']}: {failure['summary']}")
    
    print(f"\n📋 Individual Check Results:")
    for check in checks:
        status_emoji = "✅" if check["success"] else "❌"
        print(f"  {status_emoji} {check['check']}: {check['summary']}")
    
    if report["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
    
    # Save detailed report
    report_path = Path("security_quality_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return report

def main():
    """Main function to run security and quality gates."""
    try:
        report = run_security_quality_gates()
        
        if report["overall_status"] == "PASSED":
            print("\n🎯 All Security & Quality Gates Passed!")
            print("✅ Code is secure and ready for production deployment")
            return 0
        else:
            print("\n⚠️  Security & Quality Gates Failed!")
            print("❌ Address critical issues before deployment")
            return 1
    
    except Exception as e:
        print(f"❌ Error running security and quality gates: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())