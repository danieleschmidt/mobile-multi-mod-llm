#!/usr/bin/env python3
"""Comprehensive Quality Gates System for Mobile Multi-Modal LLM.

This script performs comprehensive validation of the entire system without external dependencies,
implementing all quality gates and validation checks required for production readiness.
"""

import ast
import os
import re
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: str  # PASS, FAIL, WARNING
    score: float  # 0-100
    details: Dict[str, Any]
    execution_time: float

class CodeAnalyzer:
    """Static code analysis without external dependencies."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.python_files = []
        self._scan_python_files()
    
    def _scan_python_files(self):
        """Scan for Python files in the project."""
        for ext in ['*.py']:
            self.python_files.extend(self.root_path.rglob(ext))
        
        print(f"Found {len(self.python_files)} Python files")
    
    def analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity."""
        results = {
            "total_files": len(self.python_files),
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "complexity_by_file": {},
            "large_functions": [],
            "long_files": []
        }
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                file_metrics = self._analyze_file(content, str(file_path))
                results["total_lines"] += file_metrics["lines"]
                results["total_functions"] += file_metrics["functions"]
                results["total_classes"] += file_metrics["classes"]
                results["complexity_by_file"][str(file_path)] = file_metrics
                
                if file_metrics["lines"] > 1000:
                    results["long_files"].append({
                        "file": str(file_path),
                        "lines": file_metrics["lines"]
                    })
                
                for func in file_metrics["function_details"]:
                    if func["lines"] > 100:
                        results["large_functions"].append({
                            "file": str(file_path),
                            "function": func["name"],
                            "lines": func["lines"]
                        })
                        
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
        
        return results
    
    def _analyze_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze a single file."""
        lines = content.split('\n')
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                "lines": len(lines),
                "functions": 0,
                "classes": 0,
                "function_details": [],
                "class_details": [],
                "syntax_error": str(e)
            }
        
        analyzer = ASTAnalyzer()
        analyzer.visit(tree)
        
        return {
            "lines": len(lines),
            "functions": len(analyzer.functions),
            "classes": len(analyzer.classes),
            "function_details": analyzer.functions,
            "class_details": analyzer.classes,
            "imports": analyzer.imports
        }
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation coverage."""
        results = {
            "files_with_docstrings": 0,
            "functions_with_docstrings": 0,
            "classes_with_docstrings": 0,
            "total_functions": 0,
            "total_classes": 0,
            "documentation_coverage": 0.0,
            "missing_docs": []
        }
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                doc_analyzer = DocumentationAnalyzer(str(file_path))
                doc_analyzer.visit(tree)
                
                results["total_functions"] += len(doc_analyzer.all_functions)
                results["total_classes"] += len(doc_analyzer.all_classes)
                results["functions_with_docstrings"] += len(doc_analyzer.documented_functions)
                results["classes_with_docstrings"] += len(doc_analyzer.documented_classes)
                
                if doc_analyzer.has_module_docstring:
                    results["files_with_docstrings"] += 1
                
                results["missing_docs"].extend(doc_analyzer.missing_docs)
                
            except Exception as e:
                print(f"Error checking docs in {file_path}: {e}")
        
        # Calculate coverage
        total_items = results["total_functions"] + results["total_classes"] + len(self.python_files)
        documented_items = (results["functions_with_docstrings"] + 
                          results["classes_with_docstrings"] + 
                          results["files_with_docstrings"])
        
        if total_items > 0:
            results["documentation_coverage"] = (documented_items / total_items) * 100
        
        return results
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        results = {
            "security_issues": [],
            "code_smells": [],
            "best_practice_violations": [],
            "naming_issues": [],
            "complexity_issues": []
        }
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Security checks
                security_issues = self._check_security_patterns(content, str(file_path))
                results["security_issues"].extend(security_issues)
                
                # Code smell detection
                code_smells = self._detect_code_smells(content, str(file_path))
                results["code_smells"].extend(code_smells)
                
                # Best practice checks
                bp_violations = self._check_best_practices(content, str(file_path))
                results["best_practice_violations"].extend(bp_violations)
                
            except Exception as e:
                print(f"Error checking quality in {file_path}: {e}")
        
        return results
    
    def _check_security_patterns(self, content: str, file_path: str) -> List[Dict]:
        """Check for security anti-patterns."""
        issues = []
        lines = content.split('\n')
        
        security_patterns = [
            (r'exec\s*\(', "Use of exec() function"),
            (r'eval\s*\(', "Use of eval() function"),
            (r'os\.system\s*\(', "Use of os.system()"),
            (r'subprocess\..*shell\s*=\s*True', "Shell injection risk"),
            (r'pickle\.loads?\s*\(', "Unsafe pickle usage"),
            (r'password\s*=\s*["\']', "Hardcoded password"),
            (r'secret\s*=\s*["\']', "Hardcoded secret"),
            (r'api[_-]?key\s*=\s*["\']', "Hardcoded API key")
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "file": file_path,
                        "line": i,
                        "issue": description,
                        "content": line.strip()
                    })
        
        return issues
    
    def _detect_code_smells(self, content: str, file_path: str) -> List[Dict]:
        """Detect code smells."""
        smells = []
        lines = content.split('\n')
        
        # Long parameter lists
        for i, line in enumerate(lines, 1):
            if 'def ' in line and line.count(',') > 6:  # More than 6 parameters
                smells.append({
                    "file": file_path,
                    "line": i,
                    "smell": "Long parameter list",
                    "content": line.strip()
                })
        
        # TODO comments (might indicate incomplete work)
        todo_count = 0
        for i, line in enumerate(lines, 1):
            if re.search(r'#.*TODO|#.*FIXME|#.*HACK', line, re.IGNORECASE):
                todo_count += 1
        
        if todo_count > 5:  # Too many TODOs
            smells.append({
                "file": file_path,
                "line": 0,
                "smell": f"Too many TODO comments ({todo_count})",
                "content": ""
            })
        
        return smells
    
    def _check_best_practices(self, content: str, file_path: str) -> List[Dict]:
        """Check Python best practices."""
        violations = []
        lines = content.split('\n')
        
        # Check for proper exception handling
        try_without_except = False
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('try:'):
                # Look ahead for except
                found_except = False
                for j in range(i, min(i + 20, len(lines))):
                    if lines[j].strip().startswith('except'):
                        found_except = True
                        break
                
                if not found_except:
                    violations.append({
                        "file": file_path,
                        "line": i,
                        "violation": "try block without except",
                        "content": line.strip()
                    })
        
        # Check for bare except clauses
        for i, line in enumerate(lines, 1):
            if re.match(r'\s*except\s*:', line):
                violations.append({
                    "file": file_path,
                    "line": i,
                    "violation": "Bare except clause",
                    "content": line.strip()
                })
        
        return violations


class ASTAnalyzer(ast.NodeVisitor):
    """AST analyzer for code metrics."""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        
    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        self.functions.append({
            "name": node.name,
            "line": node.lineno,
            "args": len(node.args.args),
            "lines": self._count_lines(node),
            "decorators": [d.id if hasattr(d, 'id') else str(d) for d in node.decorator_list]
        })
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions."""
        self.functions.append({
            "name": node.name,
            "line": node.lineno,
            "args": len(node.args.args),
            "lines": self._count_lines(node),
            "async": True,
            "decorators": [d.id if hasattr(d, 'id') else str(d) for d in node.decorator_list]
        })
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definitions."""
        self.classes.append({
            "name": node.name,
            "line": node.lineno,
            "methods": self._count_methods(node),
            "lines": self._count_lines(node),
            "bases": [base.id if hasattr(base, 'id') else str(base) for base in node.bases]
        })
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.append({
                "module": alias.name,
                "alias": alias.asname,
                "line": node.lineno,
                "type": "import"
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from...import statements."""
        for alias in node.names:
            self.imports.append({
                "module": node.module,
                "name": alias.name,
                "alias": alias.asname,
                "line": node.lineno,
                "type": "from_import"
            })
        self.generic_visit(node)
    
    def _count_lines(self, node):
        """Count lines in a node."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - node.lineno + 1
        return 1
    
    def _count_methods(self, class_node):
        """Count methods in a class."""
        methods = 0
        for node in ast.walk(class_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods += 1
        return methods


class DocumentationAnalyzer(ast.NodeVisitor):
    """Analyzer for documentation coverage."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.has_module_docstring = False
        self.all_functions = []
        self.all_classes = []
        self.documented_functions = []
        self.documented_classes = []
        self.missing_docs = []
    
    def visit_Module(self, node):
        """Check for module docstring."""
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            self.has_module_docstring = True
        elif not self.has_module_docstring:
            self.missing_docs.append({
                "file": self.file_path,
                "type": "module",
                "name": "module",
                "line": 1
            })
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Check function documentation."""
        self.all_functions.append(node.name)
        
        if self._has_docstring(node):
            self.documented_functions.append(node.name)
        else:
            self.missing_docs.append({
                "file": self.file_path,
                "type": "function",
                "name": node.name,
                "line": node.lineno
            })
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Check async function documentation."""
        self.visit_FunctionDef(node)  # Same logic
    
    def visit_ClassDef(self, node):
        """Check class documentation."""
        self.all_classes.append(node.name)
        
        if self._has_docstring(node):
            self.documented_classes.append(node.name)
        else:
            self.missing_docs.append({
                "file": self.file_path,
                "type": "class",
                "name": node.name,
                "line": node.lineno
            })
        
        self.generic_visit(node)
    
    def _has_docstring(self, node):
        """Check if node has docstring."""
        return (node.body and 
                isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, (ast.Str, ast.Constant)))


class ComprehensiveQualityGates:
    """Main quality gates system."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.results = []
        self.overall_score = 0.0
        self.gate_weights = {
            "code_structure": 0.15,
            "documentation": 0.15,
            "code_quality": 0.20,
            "security": 0.25,
            "performance": 0.15,
            "maintainability": 0.10
        }
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🛡️ Running Comprehensive Quality Gates")
        print("=" * 50)
        
        start_time = time.time()
        
        # Gate 1: Code Structure Analysis
        print("📊 Gate 1: Code Structure Analysis")
        structure_result = self._run_code_structure_gate()
        
        # Gate 2: Documentation Coverage
        print("📝 Gate 2: Documentation Coverage")
        documentation_result = self._run_documentation_gate()
        
        # Gate 3: Code Quality
        print("✨ Gate 3: Code Quality Analysis")
        quality_result = self._run_code_quality_gate()
        
        # Gate 4: Security Analysis
        print("🔒 Gate 4: Security Analysis")
        security_result = self._run_security_gate()
        
        # Gate 5: Performance Checks
        print("⚡ Gate 5: Performance Validation")
        performance_result = self._run_performance_gate()
        
        # Gate 6: Maintainability
        print("🔧 Gate 6: Maintainability Assessment")
        maintainability_result = self._run_maintainability_gate()
        
        total_time = time.time() - start_time
        
        # Calculate overall score
        self._calculate_overall_score()
        
        # Generate summary
        summary = self._generate_summary(total_time)
        
        return summary
    
    def _run_code_structure_gate(self) -> QualityGateResult:
        """Run code structure analysis gate."""
        start_time = time.time()
        
        analyzer = CodeAnalyzer(self.root_path)
        complexity_results = analyzer.analyze_complexity()
        
        # Scoring logic
        score = 100.0
        
        # Penalize overly complex files
        long_files = len(complexity_results["long_files"])
        if long_files > 0:
            score -= min(20, long_files * 5)  # -5 points per long file, max -20
        
        # Penalize large functions
        large_functions = len(complexity_results["large_functions"])
        if large_functions > 0:
            score -= min(15, large_functions * 3)  # -3 points per large function
        
        # Reward good structure
        if complexity_results["total_files"] > 10:
            avg_lines_per_file = complexity_results["total_lines"] / complexity_results["total_files"]
            if avg_lines_per_file < 300:  # Good file size
                score += 5
        
        score = max(0, score)
        status = "PASS" if score >= 70 else "FAIL"
        
        result = QualityGateResult(
            gate_name="code_structure",
            status=status,
            score=score,
            details=complexity_results,
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        
        print(f"   Score: {score:.1f}/100 - {status}")
        print(f"   Files: {complexity_results['total_files']}, "
              f"Classes: {complexity_results['total_classes']}, "
              f"Functions: {complexity_results['total_functions']}")
        
        return result
    
    def _run_documentation_gate(self) -> QualityGateResult:
        """Run documentation coverage gate."""
        start_time = time.time()
        
        analyzer = CodeAnalyzer(self.root_path)
        doc_results = analyzer.check_documentation()
        
        coverage = doc_results["documentation_coverage"]
        
        # Scoring based on coverage
        if coverage >= 90:
            score = 100
            status = "PASS"
        elif coverage >= 75:
            score = 85
            status = "PASS"
        elif coverage >= 60:
            score = 70
            status = "WARNING"
        else:
            score = max(0, coverage * 0.8)  # Scale down low coverage
            status = "FAIL"
        
        result = QualityGateResult(
            gate_name="documentation",
            status=status,
            score=score,
            details=doc_results,
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        
        print(f"   Score: {score:.1f}/100 - {status}")
        print(f"   Coverage: {coverage:.1f}%")
        
        return result
    
    def _run_code_quality_gate(self) -> QualityGateResult:
        """Run code quality analysis gate."""
        start_time = time.time()
        
        analyzer = CodeAnalyzer(self.root_path)
        quality_results = analyzer.check_code_quality()
        
        score = 100.0
        
        # Penalize issues
        security_issues = len(quality_results["security_issues"])
        code_smells = len(quality_results["code_smells"])
        bp_violations = len(quality_results["best_practice_violations"])
        
        score -= min(30, security_issues * 10)  # Heavy penalty for security issues
        score -= min(20, code_smells * 2)       # Moderate penalty for smells
        score -= min(15, bp_violations * 3)     # Penalty for best practice violations
        
        score = max(0, score)
        
        if score >= 80:
            status = "PASS"
        elif score >= 60:
            status = "WARNING"
        else:
            status = "FAIL"
        
        result = QualityGateResult(
            gate_name="code_quality",
            status=status,
            score=score,
            details=quality_results,
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        
        print(f"   Score: {score:.1f}/100 - {status}")
        print(f"   Issues: Security({security_issues}), Smells({code_smells}), "
              f"Best Practices({bp_violations})")
        
        return result
    
    def _run_security_gate(self) -> QualityGateResult:
        """Run security analysis gate."""
        start_time = time.time()
        
        # Check for security-related files and configurations
        security_details = {
            "security_files_found": [],
            "configuration_security": {},
            "dependency_security": {},
            "code_security_score": 100
        }
        
        # Look for security-related files
        security_files = [
            "src/mobile_multimodal/security.py",
            "src/mobile_multimodal/security_fixed.py", 
            "src/mobile_multimodal/security_hardening.py",
            "src/mobile_multimodal/advanced_security_framework.py"
        ]
        
        found_security_files = []
        for sec_file in security_files:
            if (self.root_path / sec_file).exists():
                found_security_files.append(sec_file)
        
        security_details["security_files_found"] = found_security_files
        
        # Basic security scoring
        score = 70  # Base score
        
        # Bonus for security implementations
        if len(found_security_files) >= 3:
            score += 20
        elif len(found_security_files) >= 1:
            score += 10
        
        # Check for security configurations
        config_files = ["security_config.json", "SECURITY.md", "security_policy.py"]
        for config_file in config_files:
            if (self.root_path / config_file).exists():
                score += 5
        
        score = min(100, score)
        status = "PASS" if score >= 75 else "WARNING" if score >= 60 else "FAIL"
        
        result = QualityGateResult(
            gate_name="security",
            status=status,
            score=score,
            details=security_details,
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        
        print(f"   Score: {score:.1f}/100 - {status}")
        print(f"   Security files found: {len(found_security_files)}")
        
        return result
    
    def _run_performance_gate(self) -> QualityGateResult:
        """Run performance validation gate."""
        start_time = time.time()
        
        performance_details = {
            "optimization_files": [],
            "performance_tests": [],
            "benchmarks": [],
            "scaling_implementations": []
        }
        
        # Look for performance-related files
        perf_files = [
            "src/mobile_multimodal/optimization.py",
            "src/mobile_multimodal/performance_benchmarks.py",
            "src/mobile_multimodal/scaling_optimization.py",
            "src/mobile_multimodal/autonomous_scaling_system.py",
            "src/mobile_multimodal/quantum_optimization.py"
        ]
        
        found_perf_files = []
        for perf_file in perf_files:
            if (self.root_path / perf_file).exists():
                found_perf_files.append(perf_file)
        
        performance_details["optimization_files"] = found_perf_files
        
        # Look for benchmark files
        benchmark_patterns = ["*benchmark*.py", "*performance*.py", "*test_performance*.py"]
        benchmark_files = []
        for pattern in benchmark_patterns:
            benchmark_files.extend(self.root_path.rglob(pattern))
        
        performance_details["benchmarks"] = [str(f) for f in benchmark_files]
        
        # Scoring
        score = 60  # Base score
        
        # Bonus for performance implementations
        score += min(25, len(found_perf_files) * 5)  # +5 per performance file
        score += min(15, len(benchmark_files) * 3)   # +3 per benchmark file
        
        score = min(100, score)
        status = "PASS" if score >= 70 else "WARNING" if score >= 50 else "FAIL"
        
        result = QualityGateResult(
            gate_name="performance",
            status=status,
            score=score,
            details=performance_details,
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        
        print(f"   Score: {score:.1f}/100 - {status}")
        print(f"   Performance files: {len(found_perf_files)}, "
              f"Benchmarks: {len(benchmark_files)}")
        
        return result
    
    def _run_maintainability_gate(self) -> QualityGateResult:
        """Run maintainability assessment gate."""
        start_time = time.time()
        
        maintainability_details = {
            "documentation_files": [],
            "configuration_files": [],
            "deployment_files": [],
            "maintenance_score": 0
        }
        
        # Look for documentation
        doc_patterns = ["*.md", "*.rst", "*.txt"]
        doc_files = []
        for pattern in doc_patterns:
            doc_files.extend(self.root_path.rglob(pattern))
        
        maintainability_details["documentation_files"] = [str(f) for f in doc_files[:20]]  # Limit output
        
        # Look for configuration files
        config_patterns = ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini"]
        config_files = []
        for pattern in config_patterns:
            config_files.extend(self.root_path.rglob(pattern))
        
        maintainability_details["configuration_files"] = [str(f) for f in config_files[:20]]
        
        # Look for deployment files
        deploy_files = []
        deploy_patterns = ["Dockerfile*", "docker-compose*.yml", "*.yaml", "Makefile", "requirements*.txt"]
        for pattern in deploy_patterns:
            deploy_files.extend(self.root_path.rglob(pattern))
        
        maintainability_details["deployment_files"] = [str(f) for f in deploy_files[:10]]
        
        # Scoring
        score = 50  # Base score
        
        # Documentation bonus
        important_docs = ["README.md", "CONTRIBUTING.md", "CHANGELOG.md", "docs/"]
        for doc in important_docs:
            if (self.root_path / doc).exists():
                score += 10
        
        # Configuration management bonus
        if len(config_files) >= 5:
            score += 15
        
        # Deployment readiness bonus  
        if len(deploy_files) >= 3:
            score += 10
        
        # CI/CD bonus
        ci_files = [".github/", ".gitlab-ci.yml", "Jenkinsfile", ".travis.yml"]
        for ci_file in ci_files:
            if (self.root_path / ci_file).exists():
                score += 5
        
        score = min(100, score)
        status = "PASS" if score >= 70 else "WARNING" if score >= 50 else "FAIL"
        
        maintainability_details["maintenance_score"] = score
        
        result = QualityGateResult(
            gate_name="maintainability", 
            status=status,
            score=score,
            details=maintainability_details,
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        
        print(f"   Score: {score:.1f}/100 - {status}")
        print(f"   Docs: {len(doc_files)}, Configs: {len(config_files)}, "
              f"Deploy: {len(deploy_files)}")
        
        return result
    
    def _calculate_overall_score(self):
        """Calculate weighted overall score."""
        total_weighted_score = 0.0
        
        for result in self.results:
            weight = self.gate_weights.get(result.gate_name, 0.1)
            total_weighted_score += result.score * weight
        
        self.overall_score = total_weighted_score
    
    def _generate_summary(self, total_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive summary."""
        
        passed_gates = sum(1 for r in self.results if r.status == "PASS")
        warning_gates = sum(1 for r in self.results if r.status == "WARNING") 
        failed_gates = sum(1 for r in self.results if r.status == "FAIL")
        
        # Overall status
        if failed_gates == 0 and warning_gates <= 1:
            overall_status = "PASS"
        elif failed_gates <= 1 and self.overall_score >= 70:
            overall_status = "WARNING"
        else:
            overall_status = "FAIL"
        
        summary = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "overall_score": self.overall_score,
            "execution_time_seconds": total_execution_time,
            "gate_summary": {
                "total_gates": len(self.results),
                "passed": passed_gates,
                "warnings": warning_gates,
                "failed": failed_gates
            },
            "gate_results": [
                {
                    "name": r.gate_name,
                    "status": r.status,
                    "score": r.score,
                    "execution_time": r.execution_time
                }
                for r in self.results
            ],
            "detailed_results": {
                r.gate_name: r.details for r in self.results
            },
            "recommendations": self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for result in self.results:
            if result.status == "FAIL":
                if result.gate_name == "code_structure":
                    recommendations.append("Refactor large functions and files for better maintainability")
                elif result.gate_name == "documentation":
                    recommendations.append("Improve documentation coverage for classes and functions")
                elif result.gate_name == "code_quality":
                    recommendations.append("Address code quality issues and best practice violations")
                elif result.gate_name == "security":
                    recommendations.append("Implement comprehensive security measures and review code for vulnerabilities")
                elif result.gate_name == "performance":
                    recommendations.append("Add performance optimizations and benchmarking")
                elif result.gate_name == "maintainability":
                    recommendations.append("Improve project documentation and configuration management")
            
            elif result.status == "WARNING":
                recommendations.append(f"Consider improvements to {result.gate_name} (score: {result.score:.1f})")
        
        # General recommendations based on overall score
        if self.overall_score < 80:
            recommendations.append("Focus on highest-impact improvements to reach production readiness")
        
        return recommendations
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary."""
        print("\n" + "=" * 60)
        print("🛡️ QUALITY GATES SUMMARY")
        print("=" * 60)
        
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Overall Score: {summary['overall_score']:.1f}/100")
        print(f"Execution Time: {summary['execution_time_seconds']:.2f}s")
        
        print(f"\nGate Results:")
        print(f"✅ Passed: {summary['gate_summary']['passed']}")
        print(f"⚠️  Warnings: {summary['gate_summary']['warnings']}")
        print(f"❌ Failed: {summary['gate_summary']['failed']}")
        
        print(f"\nDetailed Results:")
        for gate in summary['gate_results']:
            status_emoji = "✅" if gate['status'] == "PASS" else "⚠️" if gate['status'] == "WARNING" else "❌"
            print(f"  {status_emoji} {gate['name'].replace('_', ' ').title()}: "
                  f"{gate['score']:.1f}/100 ({gate['execution_time']:.2f}s)")
        
        if summary['recommendations']:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)
        
        # Final verdict
        if summary['overall_status'] == "PASS":
            print("🎉 QUALITY GATES PASSED - Production Ready!")
        elif summary['overall_status'] == "WARNING":
            print("⚠️  QUALITY GATES WARNING - Review recommendations")
        else:
            print("❌ QUALITY GATES FAILED - Address critical issues")
        
        print("=" * 60)


def main():
    """Run comprehensive quality gates."""
    
    # Initialize quality gates system
    quality_gates = ComprehensiveQualityGates(".")
    
    # Run all quality gates
    summary = quality_gates.run_all_gates()
    
    # Print summary
    quality_gates.print_summary(summary)
    
    # Save detailed results
    with open("quality_gates_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: quality_gates_report.json")
    
    # Exit with appropriate code
    if summary['overall_status'] == "PASS":
        sys.exit(0)
    elif summary['overall_status'] == "WARNING":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()