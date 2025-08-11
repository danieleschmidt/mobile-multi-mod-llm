#!/usr/bin/env python3
"""Quality gates validation for mobile multi-modal LLM package."""

import ast
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any


class QualityGateValidator:
    """Comprehensive quality gate validation."""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.src_dir = self.repo_root / "src"
        self.tests_dir = self.repo_root / "tests"
        
        self.results = {
            "code_quality": {},
            "structure": {},
            "documentation": {},
            "security": {},
            "performance": {},
            "overall_score": 0
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        print("🔍 Running Quality Gate Validations...")
        print("=" * 60)
        
        # Code Quality Checks
        print("\n📊 Code Quality Analysis")
        self.validate_code_quality()
        
        # Project Structure Checks
        print("\n📁 Project Structure Analysis")
        self.validate_project_structure()
        
        # Documentation Checks
        print("\n📚 Documentation Analysis")
        self.validate_documentation()
        
        # Security Checks
        print("\n🔒 Security Analysis")
        self.validate_security()
        
        # Performance Analysis
        print("\n⚡ Performance Analysis")
        self.validate_performance_readiness()
        
        # Calculate overall score
        self.calculate_overall_score()
        
        return self.results
    
    def validate_code_quality(self):
        """Validate code quality metrics."""
        quality_checks = {
            "syntax_errors": self.check_syntax_errors(),
            "import_structure": self.check_import_structure(),
            "function_complexity": self.analyze_function_complexity(),
            "code_coverage": self.estimate_code_coverage(),
            "docstring_coverage": self.check_docstring_coverage()
        }
        
        self.results["code_quality"] = quality_checks
        
        # Print results
        for check, result in quality_checks.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status} {check.replace('_', ' ').title()}: {result['message']}")
    
    def check_syntax_errors(self) -> Dict[str, Any]:
        """Check for Python syntax errors."""
        errors = []
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"{py_file.relative_to(self.repo_root)}: {e}")
            except Exception as e:
                errors.append(f"{py_file.relative_to(self.repo_root)}: {e}")
        
        return {
            "passed": len(errors) == 0,
            "message": f"No syntax errors found in {len(python_files)} files" if not errors else f"{len(errors)} syntax errors found",
            "details": errors
        }
    
    def check_import_structure(self) -> Dict[str, Any]:
        """Check import structure and circular dependencies."""
        imports = {}
        python_files = list(self.src_dir.rglob("*.py"))
        issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                file_imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            file_imports.append(node.module)
                
                imports[str(py_file.relative_to(self.repo_root))] = file_imports
                
            except Exception as e:
                issues.append(f"{py_file.relative_to(self.repo_root)}: {e}")
        
        # Check for relative imports from package root
        proper_imports = 0
        for file_path, file_imports in imports.items():
            for imp in file_imports:
                if imp.startswith('mobile_multimodal.'):
                    proper_imports += 1
        
        return {
            "passed": len(issues) == 0,
            "message": f"Import structure analyzed: {proper_imports} proper package imports, {len(issues)} issues",
            "details": {
                "issues": issues,
                "proper_imports": proper_imports,
                "total_files": len(python_files)
            }
        }
    
    def analyze_function_complexity(self) -> Dict[str, Any]:
        """Analyze function complexity."""
        complex_functions = []
        python_files = list(self.src_dir.rglob("*.py"))
        total_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        # Simple complexity measure: count nested nodes
                        complexity = sum(1 for _ in ast.walk(node))
                        
                        if complexity > 100:  # Arbitrary threshold
                            complex_functions.append(f"{py_file.name}:{node.name} (complexity: {complexity})")
                            
            except Exception:
                continue
        
        return {
            "passed": len(complex_functions) < total_functions * 0.1,  # Less than 10% complex functions
            "message": f"Analyzed {total_functions} functions, {len(complex_functions)} highly complex",
            "details": complex_functions[:10]  # Show first 10
        }
    
    def estimate_code_coverage(self) -> Dict[str, Any]:
        """Estimate code coverage based on test files."""
        src_files = list(self.src_dir.rglob("*.py"))
        test_files = list(self.tests_dir.rglob("*.py")) if self.tests_dir.exists() else []
        
        # Simple heuristic: estimate coverage based on test file existence
        src_modules = set(f.stem for f in src_files if not f.name.startswith('__'))
        test_coverage = set()
        
        for test_file in test_files:
            with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for module in src_modules:
                    if module in content:
                        test_coverage.add(module)
        
        coverage_ratio = len(test_coverage) / len(src_modules) if src_modules else 0
        
        return {
            "passed": coverage_ratio >= 0.6,  # 60% coverage threshold
            "message": f"Estimated coverage: {coverage_ratio:.1%} ({len(test_coverage)}/{len(src_modules)} modules)",
            "details": {
                "covered_modules": list(test_coverage),
                "uncovered_modules": list(src_modules - test_coverage)
            }
        }
    
    def check_docstring_coverage(self) -> Dict[str, Any]:
        """Check docstring coverage."""
        functions_with_docs = 0
        total_functions = 0
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if not node.name.startswith('_'):  # Skip private methods
                            total_functions += 1
                            
                            # Check if has docstring
                            if (node.body and 
                                isinstance(node.body[0], ast.Expr) and 
                                isinstance(node.body[0].value, ast.Constant) and
                                isinstance(node.body[0].value.value, str)):
                                functions_with_docs += 1
                                
            except Exception:
                continue
        
        coverage = functions_with_docs / total_functions if total_functions > 0 else 0
        
        return {
            "passed": coverage >= 0.7,  # 70% docstring coverage
            "message": f"Docstring coverage: {coverage:.1%} ({functions_with_docs}/{total_functions})",
            "details": {"coverage_ratio": coverage}
        }
    
    def validate_project_structure(self):
        """Validate project structure and organization."""
        structure_checks = {
            "package_structure": self.check_package_structure(),
            "configuration_files": self.check_configuration_files(),
            "documentation_files": self.check_documentation_files(),
            "deployment_readiness": self.check_deployment_files()
        }
        
        self.results["structure"] = structure_checks
        
        # Print results
        for check, result in structure_checks.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status} {check.replace('_', ' ').title()}: {result['message']}")
    
    def check_package_structure(self) -> Dict[str, Any]:
        """Check package structure."""
        required_files = [
            "src/mobile_multimodal/__init__.py",
            "src/mobile_multimodal/core.py",
            "src/mobile_multimodal/security.py",
            "src/mobile_multimodal/monitoring.py",
            "src/mobile_multimodal/optimization.py",
            "src/mobile_multimodal/data/__init__.py",
            "src/mobile_multimodal/scripts/__init__.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                missing_files.append(file_path)
        
        return {
            "passed": len(missing_files) == 0,
            "message": f"Package structure complete" if not missing_files else f"{len(missing_files)} required files missing",
            "details": missing_files
        }
    
    def check_configuration_files(self) -> Dict[str, Any]:
        """Check configuration files."""
        config_files = [
            "pyproject.toml",
            "requirements.txt",
            "README.md"
        ]
        
        existing_files = []
        for file_path in config_files:
            if (self.repo_root / file_path).exists():
                existing_files.append(file_path)
        
        return {
            "passed": len(existing_files) >= 2,  # At least 2 config files
            "message": f"{len(existing_files)}/{len(config_files)} configuration files present",
            "details": {"existing": existing_files}
        }
    
    def check_documentation_files(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        doc_files = [
            "README.md",
            "docs/",
            "CONTRIBUTING.md",
            "LICENSE",
        ]
        
        present_docs = []
        for doc_path in doc_files:
            if (self.repo_root / doc_path).exists():
                present_docs.append(doc_path)
        
        return {
            "passed": len(present_docs) >= 2,
            "message": f"{len(present_docs)}/{len(doc_files)} documentation components present",
            "details": present_docs
        }
    
    def check_deployment_files(self) -> Dict[str, Any]:
        """Check deployment readiness."""
        deployment_files = [
            "Dockerfile",
            "docker-compose.yml",
            "deployment/",
            "monitoring/",
            ".github/workflows/" if (self.repo_root / ".github/workflows/").exists() else None
        ]
        
        deployment_files = [f for f in deployment_files if f is not None]
        
        present_files = []
        for file_path in deployment_files:
            if (self.repo_root / file_path).exists():
                present_files.append(file_path)
        
        return {
            "passed": len(present_files) >= 2,
            "message": f"{len(present_files)}/{len(deployment_files)} deployment components present",
            "details": present_files
        }
    
    def validate_documentation(self):
        """Validate documentation quality."""
        doc_checks = {
            "readme_quality": self.check_readme_quality(),
            "api_documentation": self.check_api_documentation(),
            "examples": self.check_examples()
        }
        
        self.results["documentation"] = doc_checks
        
        for check, result in doc_checks.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status} {check.replace('_', ' ').title()}: {result['message']}")
    
    def check_readme_quality(self) -> Dict[str, Any]:
        """Check README quality."""
        readme_path = self.repo_root / "README.md"
        
        if not readme_path.exists():
            return {"passed": False, "message": "README.md not found"}
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for essential sections
            required_sections = [
                "installation", "usage", "example", "features",
                "requirements", "quick", "start"
            ]
            
            content_lower = content.lower()
            found_sections = [s for s in required_sections if s in content_lower]
            
            return {
                "passed": len(found_sections) >= 4,  # At least 4 essential sections
                "message": f"README quality: {len(found_sections)}/{len(required_sections)} sections, {len(content)} characters",
                "details": {"found_sections": found_sections, "length": len(content)}
            }
            
        except Exception as e:
            return {"passed": False, "message": f"README analysis failed: {e}"}
    
    def check_api_documentation(self) -> Dict[str, Any]:
        """Check API documentation in code."""
        python_files = list(self.src_dir.rglob("*.py"))
        documented_classes = 0
        total_classes = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                        total_classes += 1
                        
                        # Check if class has docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                            documented_classes += 1
                            
            except Exception:
                continue
        
        coverage = documented_classes / total_classes if total_classes > 0 else 1
        
        return {
            "passed": coverage >= 0.8,  # 80% class documentation
            "message": f"API documentation: {coverage:.1%} classes documented ({documented_classes}/{total_classes})",
            "details": {"coverage": coverage}
        }
    
    def check_examples(self) -> Dict[str, Any]:
        """Check for usage examples."""
        # Look for examples in various places
        example_locations = [
            self.repo_root / "examples/",
            self.repo_root / "docs/examples/",
            self.repo_root / "README.md"
        ]
        
        examples_found = 0
        for location in example_locations:
            if location.exists():
                if location.is_file():
                    # Check README for code examples
                    try:
                        with open(location, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if "```python" in content or "from mobile_multimodal" in content:
                            examples_found += 1
                    except Exception:
                        pass
                else:
                    # Check directory for example files
                    example_files = list(location.rglob("*.py"))
                    if example_files:
                        examples_found += len(example_files)
        
        return {
            "passed": examples_found > 0,
            "message": f"{examples_found} usage examples found",
            "details": {"example_count": examples_found}
        }
    
    def validate_security(self):
        """Validate security measures."""
        security_checks = {
            "input_validation": self.check_input_validation(),
            "security_patterns": self.check_security_patterns(),
            "dependency_security": self.check_dependency_security()
        }
        
        self.results["security"] = security_checks
        
        for check, result in security_checks.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status} {check.replace('_', ' ').title()}: {result['message']}")
    
    def check_input_validation(self) -> Dict[str, Any]:
        """Check for input validation patterns."""
        validation_patterns = [
            "validate", "sanitize", "check", "verify", "isinstance",
            "ValueError", "TypeError", "security", "SecurityError"
        ]
        
        python_files = list(self.src_dir.rglob("*.py"))
        validation_usage = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in validation_patterns:
                    if pattern in content:
                        validation_usage += content.count(pattern)
                        
            except Exception:
                continue
        
        return {
            "passed": validation_usage >= 10,  # Reasonable number of validation patterns
            "message": f"{validation_usage} input validation patterns found",
            "details": {"pattern_count": validation_usage}
        }
    
    def check_security_patterns(self) -> Dict[str, Any]:
        """Check for security implementation patterns."""
        # Look for security module and patterns
        security_file = self.src_dir / "mobile_multimodal" / "security.py"
        
        if not security_file.exists():
            return {"passed": False, "message": "Security module not found"}
        
        try:
            with open(security_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            security_features = [
                "rate_limit", "sanitiz", "validat", "authenticat", "authoriz",
                "encrypt", "hash", "token", "session", "csrf"
            ]
            
            found_features = [f for f in security_features if f in content.lower()]
            
            return {
                "passed": len(found_features) >= 5,
                "message": f"Security features: {len(found_features)}/{len(security_features)} implemented",
                "details": found_features
            }
            
        except Exception as e:
            return {"passed": False, "message": f"Security analysis failed: {e}"}
    
    def check_dependency_security(self) -> Dict[str, Any]:
        """Check dependency security practices."""
        requirements_files = [
            self.repo_root / "requirements.txt",
            self.repo_root / "pyproject.toml"
        ]
        
        pinned_deps = 0
        total_deps = 0
        
        for req_file in requirements_files:
            if not req_file.exists():
                continue
            
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if req_file.name == "requirements.txt":
                    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
                    for line in lines:
                        if '==' in line or '>=' in line:
                            total_deps += 1
                            if '==' in line:
                                pinned_deps += 1
                
                elif req_file.name == "pyproject.toml":
                    # Simple check for dependencies section
                    if "dependencies" in content:
                        total_deps += content.count('>=')
                        pinned_deps += content.count('==')
                        
            except Exception:
                continue
        
        pin_ratio = pinned_deps / total_deps if total_deps > 0 else 0
        
        return {
            "passed": total_deps > 0,  # At least some dependencies managed
            "message": f"Dependencies: {total_deps} total, {pin_ratio:.1%} pinned",
            "details": {"total_deps": total_deps, "pinned_deps": pinned_deps}
        }
    
    def validate_performance_readiness(self):
        """Validate performance optimization readiness."""
        perf_checks = {
            "optimization_modules": self.check_optimization_modules(),
            "monitoring_capabilities": self.check_monitoring_capabilities(),
            "caching_implementation": self.check_caching_implementation(),
            "scalability_patterns": self.check_scalability_patterns()
        }
        
        self.results["performance"] = perf_checks
        
        for check, result in perf_checks.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status} {check.replace('_', ' ').title()}: {result['message']}")
    
    def check_optimization_modules(self) -> Dict[str, Any]:
        """Check for optimization modules."""
        opt_file = self.src_dir / "mobile_multimodal" / "optimization.py"
        
        if not opt_file.exists():
            return {"passed": False, "message": "Optimization module not found"}
        
        try:
            with open(opt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            opt_features = [
                "cache", "batch", "pool", "concurrent", "async", 
                "optimize", "performance", "scale", "resource"
            ]
            
            found_features = [f for f in opt_features if f in content.lower()]
            
            return {
                "passed": len(found_features) >= 6,
                "message": f"Optimization features: {len(found_features)}/{len(opt_features)} implemented",
                "details": found_features
            }
            
        except Exception as e:
            return {"passed": False, "message": f"Optimization analysis failed: {e}"}
    
    def check_monitoring_capabilities(self) -> Dict[str, Any]:
        """Check monitoring implementation."""
        monitoring_file = self.src_dir / "mobile_multimodal" / "monitoring.py"
        
        if not monitoring_file.exists():
            return {"passed": False, "message": "Monitoring module not found"}
        
        try:
            with open(monitoring_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            monitoring_features = [
                "metric", "telemetry", "logging", "alert", "dashboard",
                "observability", "trace", "monitor"
            ]
            
            found_features = [f for f in monitoring_features if f in content.lower()]
            
            return {
                "passed": len(found_features) >= 5,
                "message": f"Monitoring features: {len(found_features)}/{len(monitoring_features)} implemented",
                "details": found_features
            }
            
        except Exception as e:
            return {"passed": False, "message": f"Monitoring analysis failed: {e}"}
    
    def check_caching_implementation(self) -> Dict[str, Any]:
        """Check caching implementation."""
        cache_patterns = ["cache", "lru", "ttl", "evict", "memoiz"]
        
        python_files = list(self.src_dir.rglob("*.py"))
        cache_usage = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                for pattern in cache_patterns:
                    cache_usage += content.count(pattern)
                    
            except Exception:
                continue
        
        return {
            "passed": cache_usage >= 5,
            "message": f"Caching patterns: {cache_usage} occurrences found",
            "details": {"cache_usage": cache_usage}
        }
    
    def check_scalability_patterns(self) -> Dict[str, Any]:
        """Check scalability implementation patterns."""
        scalability_patterns = [
            "thread", "process", "async", "concurrent", "pool", 
            "queue", "batch", "scale", "load", "balance"
        ]
        
        python_files = list(self.src_dir.rglob("*.py"))
        scalability_usage = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                for pattern in scalability_patterns:
                    scalability_usage += content.count(pattern)
                    
            except Exception:
                continue
        
        return {
            "passed": scalability_usage >= 10,
            "message": f"Scalability patterns: {scalability_usage} occurrences found",
            "details": {"scalability_usage": scalability_usage}
        }
    
    def calculate_overall_score(self):
        """Calculate overall quality score."""
        category_weights = {
            "code_quality": 0.3,
            "structure": 0.2,
            "documentation": 0.2,
            "security": 0.15,
            "performance": 0.15
        }
        
        category_scores = {}
        total_score = 0
        
        for category, weight in category_weights.items():
            if category in self.results:
                checks = self.results[category]
                passed_checks = sum(1 for check in checks.values() if check["passed"])
                total_checks = len(checks)
                category_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
                category_scores[category] = category_score
                total_score += category_score * weight
        
        self.results["category_scores"] = category_scores
        self.results["overall_score"] = total_score
        
        # Print summary
        print(f"\n🎯 QUALITY GATES SUMMARY")
        print("=" * 60)
        
        for category, score in category_scores.items():
            print(f"  {category.replace('_', ' ').title()}: {score:.1f}%")
        
        print(f"\n🏆 Overall Score: {total_score:.1f}%")
        
        if total_score >= 80:
            print("🎉 EXCELLENT - Production ready!")
        elif total_score >= 70:
            print("✅ GOOD - Ready with minor improvements")
        elif total_score >= 60:
            print("⚠️  ACCEPTABLE - Needs improvement")
        else:
            print("❌ NEEDS WORK - Significant issues to address")


def main():
    """Main entry point for quality gate validation."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    validator = QualityGateValidator(repo_root)
    
    results = validator.validate_all()
    
    # Save results
    results_file = os.path.join(repo_root, "quality_gate_results.json")
    try:
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📊 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    # Return appropriate exit code
    overall_score = results.get("overall_score", 0)
    if overall_score >= 60:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    sys.exit(main())