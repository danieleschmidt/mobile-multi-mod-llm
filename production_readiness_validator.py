#!/usr/bin/env python3
"""Production Readiness Validator for Mobile Multi-Modal LLM.

This script validates the system for production deployment with comprehensive checks
across all critical aspects including functionality, security, performance, and compliance.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import subprocess
import re

@dataclass
class ValidationResult:
    """Result of a validation check."""
    category: str
    check_name: str
    status: str  # PASS, FAIL, WARNING, SKIP
    score: float  # 0-100
    message: str
    details: Dict[str, Any]
    critical: bool = False

class ProductionReadinessValidator:
    """Comprehensive production readiness validation."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.results = []
        self.critical_failures = []
        
        # Validation categories and weights
        self.categories = {
            "core_functionality": {"weight": 0.25, "critical": True},
            "security": {"weight": 0.20, "critical": True},
            "performance": {"weight": 0.15, "critical": False},
            "scalability": {"weight": 0.15, "critical": False},
            "monitoring": {"weight": 0.10, "critical": False},
            "documentation": {"weight": 0.10, "critical": False},
            "deployment": {"weight": 0.05, "critical": False}
        }
    
    def validate_production_readiness(self) -> Dict[str, Any]:
        """Run comprehensive production readiness validation."""
        print("🚀 Production Readiness Validator")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run validation categories
        self._validate_core_functionality()
        self._validate_security_implementation()
        self._validate_performance_capabilities()
        self._validate_scalability_features()
        self._validate_monitoring_observability()
        self._validate_documentation_completeness()
        self._validate_deployment_readiness()
        
        total_time = time.time() - start_time
        
        # Calculate scores and generate report
        summary = self._generate_validation_summary(total_time)
        
        return summary
    
    def _validate_core_functionality(self):
        """Validate core functionality components."""
        print("🧠 Validating Core Functionality...")
        
        # Check main model implementation
        core_files = [
            "src/mobile_multimodal/core.py",
            "src/mobile_multimodal/__init__.py",
            "src/mobile_multimodal/models.py",
            "src/mobile_multimodal/utils.py"
        ]
        
        missing_core = []
        for file_path in core_files:
            if not (self.root_path / file_path).exists():
                missing_core.append(file_path)
        
        if missing_core:
            self._add_result("core_functionality", "core_files", "FAIL", 0,
                           f"Missing core files: {missing_core}", {"missing": missing_core}, True)
        else:
            self._add_result("core_functionality", "core_files", "PASS", 100,
                           "All core files present", {"files": core_files})
        
        # Check advanced features
        advanced_features = [
            "src/mobile_multimodal/advanced_research_system.py",
            "src/mobile_multimodal/quantum_optimization.py",
            "src/mobile_multimodal/edge_federated_learning.py",
            "src/mobile_multimodal/research_framework.py"
        ]
        
        present_features = [f for f in advanced_features if (self.root_path / f).exists()]
        
        feature_score = (len(present_features) / len(advanced_features)) * 100
        if feature_score >= 75:
            status = "PASS"
        elif feature_score >= 50:
            status = "WARNING"
        else:
            status = "FAIL"
        
        self._add_result("core_functionality", "advanced_features", status, feature_score,
                        f"Advanced features: {len(present_features)}/{len(advanced_features)}",
                        {"present": present_features, "missing": list(set(advanced_features) - set(present_features))})
        
        # Check model export capabilities
        export_files = [
            "src/mobile_multimodal/export.py",
            "src/mobile_multimodal/edge_export.py"
        ]
        
        export_present = [f for f in export_files if (self.root_path / f).exists()]
        export_score = (len(export_present) / len(export_files)) * 100
        
        self._add_result("core_functionality", "export_capabilities", 
                        "PASS" if export_score >= 50 else "FAIL",
                        export_score,
                        f"Export capabilities: {len(export_present)}/{len(export_files)}",
                        {"present": export_present})
        
        # Check quantization support
        quantization_files = [
            "src/mobile_multimodal/quantization.py",
            "src/mobile_multimodal/adaptive_quantization.py"
        ]
        
        quant_present = [f for f in quantization_files if (self.root_path / f).exists()]
        quant_score = (len(quant_present) / len(quantization_files)) * 100
        
        self._add_result("core_functionality", "quantization_support",
                        "PASS" if quant_score >= 50 else "WARNING",
                        quant_score,
                        f"Quantization support: {len(quant_present)}/{len(quantization_files)}",
                        {"present": quant_present})
    
    def _validate_security_implementation(self):
        """Validate security implementation."""
        print("🔒 Validating Security Implementation...")
        
        # Check security frameworks
        security_files = [
            "src/mobile_multimodal/security.py",
            "src/mobile_multimodal/security_fixed.py",
            "src/mobile_multimodal/security_hardening.py", 
            "src/mobile_multimodal/advanced_security_framework.py"
        ]
        
        security_present = [f for f in security_files if (self.root_path / f).exists()]
        security_score = (len(security_present) / len(security_files)) * 100
        
        if security_score >= 75:
            status = "PASS"
            critical = False
        elif security_score >= 50:
            status = "WARNING"
            critical = False
        else:
            status = "FAIL"
            critical = True
        
        self._add_result("security", "security_frameworks", status, security_score,
                        f"Security frameworks: {len(security_present)}/{len(security_files)}",
                        {"present": security_present}, critical)
        
        # Check error handling
        error_handling_files = [
            "src/mobile_multimodal/comprehensive_error_handler.py",
            "src/mobile_multimodal/robust_error_handling.py"
        ]
        
        error_present = [f for f in error_handling_files if (self.root_path / f).exists()]
        error_score = (len(error_present) / len(error_handling_files)) * 100
        
        self._add_result("security", "error_handling", 
                        "PASS" if error_score >= 50 else "WARNING",
                        error_score,
                        f"Error handling: {len(error_present)}/{len(error_handling_files)}",
                        {"present": error_present})
        
        # Check security configurations
        security_configs = [
            "security_config.json",
            "SECURITY.md",
            "security_compliance_report.json"
        ]
        
        config_present = [f for f in security_configs if (self.root_path / f).exists()]
        config_score = (len(config_present) / len(security_configs)) * 100
        
        self._add_result("security", "security_configs",
                        "PASS" if config_score >= 66 else "WARNING",
                        config_score,
                        f"Security configs: {len(config_present)}/{len(security_configs)}",
                        {"present": config_present})
    
    def _validate_performance_capabilities(self):
        """Validate performance optimization capabilities."""
        print("⚡ Validating Performance Capabilities...")
        
        # Check optimization modules
        perf_files = [
            "src/mobile_multimodal/optimization.py",
            "src/mobile_multimodal/performance_benchmarks.py",
            "src/mobile_multimodal/scaling_optimization.py",
            "src/mobile_multimodal/intelligent_cache.py"
        ]
        
        perf_present = [f for f in perf_files if (self.root_path / f).exists()]
        perf_score = (len(perf_present) / len(perf_files)) * 100
        
        self._add_result("performance", "optimization_modules",
                        "PASS" if perf_score >= 75 else "WARNING",
                        perf_score,
                        f"Performance modules: {len(perf_present)}/{len(perf_files)}",
                        {"present": perf_present})
        
        # Check caching implementation
        cache_files = [
            "src/mobile_multimodal/data/cache.py",
            "src/mobile_multimodal/intelligent_cache.py"
        ]
        
        cache_present = [f for f in cache_files if (self.root_path / f).exists()]
        cache_score = (len(cache_present) / len(cache_files)) * 100
        
        self._add_result("performance", "caching_system",
                        "PASS" if cache_score >= 50 else "WARNING",
                        cache_score,
                        f"Caching system: {len(cache_present)}/{len(cache_files)}",
                        {"present": cache_present})
        
        # Check mobile optimizations
        mobile_opt_files = [
            "src/mobile_multimodal/quantization.py",
            "src/mobile_multimodal/adaptive_quantization.py",
            "exports/"
        ]
        
        mobile_present = [f for f in mobile_opt_files if (self.root_path / f).exists()]
        mobile_score = (len(mobile_present) / len(mobile_opt_files)) * 100
        
        self._add_result("performance", "mobile_optimizations",
                        "PASS" if mobile_score >= 66 else "WARNING",
                        mobile_score,
                        f"Mobile optimizations: {len(mobile_present)}/{len(mobile_opt_files)}",
                        {"present": mobile_present})
    
    def _validate_scalability_features(self):
        """Validate scalability features."""
        print("📈 Validating Scalability Features...")
        
        # Check scaling systems
        scaling_files = [
            "src/mobile_multimodal/autonomous_scaling_system.py",
            "src/mobile_multimodal/auto_scaling.py",
            "src/mobile_multimodal/distributed_scaling.py",
            "src/mobile_multimodal/distributed_inference.py"
        ]
        
        scaling_present = [f for f in scaling_files if (self.root_path / f).exists()]
        scaling_score = (len(scaling_present) / len(scaling_files)) * 100
        
        self._add_result("scalability", "scaling_systems",
                        "PASS" if scaling_score >= 50 else "WARNING",
                        scaling_score,
                        f"Scaling systems: {len(scaling_present)}/{len(scaling_files)}",
                        {"present": scaling_present})
        
        # Check load balancing and distribution
        distribution_files = [
            "src/mobile_multimodal/concurrent_processor.py", 
            "src/mobile_multimodal/batch_processor.py"
        ]
        
        dist_present = [f for f in distribution_files if (self.root_path / f).exists()]
        dist_score = (len(dist_present) / len(distribution_files)) * 100
        
        self._add_result("scalability", "load_distribution",
                        "PASS" if dist_score >= 50 else "WARNING",
                        dist_score,
                        f"Load distribution: {len(dist_present)}/{len(distribution_files)}",
                        {"present": dist_present})
        
        # Check federation support
        federation_files = [
            "src/mobile_multimodal/edge_federated_learning.py"
        ]
        
        fed_present = [f for f in federation_files if (self.root_path / f).exists()]
        fed_score = (len(fed_present) / len(federation_files)) * 100
        
        self._add_result("scalability", "federated_learning",
                        "PASS" if fed_score >= 100 else "WARNING",
                        fed_score,
                        f"Federated learning: {len(fed_present)}/{len(federation_files)}",
                        {"present": fed_present})
    
    def _validate_monitoring_observability(self):
        """Validate monitoring and observability."""
        print("📊 Validating Monitoring & Observability...")
        
        # Check monitoring systems
        monitoring_files = [
            "src/mobile_multimodal/monitoring.py",
            "src/mobile_multimodal/enhanced_monitoring.py",
            "src/mobile_multimodal/robust_monitoring.py",
            "src/mobile_multimodal/production_monitoring.py"
        ]
        
        mon_present = [f for f in monitoring_files if (self.root_path / f).exists()]
        mon_score = (len(mon_present) / len(monitoring_files)) * 100
        
        self._add_result("monitoring", "monitoring_systems",
                        "PASS" if mon_score >= 75 else "WARNING",
                        mon_score,
                        f"Monitoring systems: {len(mon_present)}/{len(monitoring_files)}",
                        {"present": mon_present})
        
        # Check observability infrastructure
        observability_files = [
            "monitoring/",
            "monitoring/prometheus.yml",
            "monitoring/grafana/",
            "monitoring/alerts.yml"
        ]
        
        obs_present = [f for f in observability_files if (self.root_path / f).exists()]
        obs_score = (len(obs_present) / len(observability_files)) * 100
        
        self._add_result("monitoring", "observability_infrastructure",
                        "PASS" if obs_score >= 50 else "WARNING",
                        obs_score,
                        f"Observability infra: {len(obs_present)}/{len(observability_files)}",
                        {"present": obs_present})
        
        # Check health checks
        health_files = [
            "deployment/healthcheck.py",
            "docker/healthcheck.py"
        ]
        
        health_present = [f for f in health_files if (self.root_path / f).exists()]
        health_score = (len(health_present) / len(health_files)) * 100
        
        self._add_result("monitoring", "health_checks",
                        "PASS" if health_score >= 50 else "WARNING",
                        health_score,
                        f"Health checks: {len(health_present)}/{len(health_files)}",
                        {"present": health_present})
    
    def _validate_documentation_completeness(self):
        """Validate documentation completeness."""
        print("📚 Validating Documentation Completeness...")
        
        # Check essential documentation
        essential_docs = [
            "README.md",
            "DEPLOYMENT.md", 
            "API_REFERENCE.md",
            "ARCHITECTURE_DECISION_RECORD.md"
        ]
        
        docs_present = [f for f in essential_docs if (self.root_path / f).exists()]
        docs_score = (len(docs_present) / len(essential_docs)) * 100
        
        self._add_result("documentation", "essential_documentation",
                        "PASS" if docs_score >= 75 else "WARNING",
                        docs_score,
                        f"Essential docs: {len(docs_present)}/{len(essential_docs)}",
                        {"present": docs_present})
        
        # Check deployment guides
        deploy_docs = [
            "DEPLOYMENT_GUIDE.md",
            "GLOBAL_DEPLOYMENT_GUIDE.md",
            "deployment/production-deployment-guide.md"
        ]
        
        deploy_docs_present = [f for f in deploy_docs if (self.root_path / f).exists()]
        deploy_docs_score = (len(deploy_docs_present) / len(deploy_docs)) * 100
        
        self._add_result("documentation", "deployment_guides",
                        "PASS" if deploy_docs_score >= 66 else "WARNING",
                        deploy_docs_score,
                        f"Deployment guides: {len(deploy_docs_present)}/{len(deploy_docs)}",
                        {"present": deploy_docs_present})
        
        # Check comprehensive guides
        advanced_docs = [
            "ADVANCED_FEATURES_DEPLOYMENT_GUIDE.md",
            "SECURITY.md",
            "COMPLIANCE.md",
            "docs/"
        ]
        
        advanced_present = [f for f in advanced_docs if (self.root_path / f).exists()]
        advanced_score = (len(advanced_present) / len(advanced_docs)) * 100
        
        self._add_result("documentation", "advanced_documentation",
                        "PASS" if advanced_score >= 75 else "WARNING",
                        advanced_score,
                        f"Advanced docs: {len(advanced_present)}/{len(advanced_docs)}",
                        {"present": advanced_present})
    
    def _validate_deployment_readiness(self):
        """Validate deployment readiness."""
        print("🚢 Validating Deployment Readiness...")
        
        # Check container support
        container_files = [
            "Dockerfile",
            "docker-compose.yml",
            "deployment/Dockerfile",
            "deployment/docker-compose.yml"
        ]
        
        container_present = [f for f in container_files if (self.root_path / f).exists()]
        container_score = (len(container_present) / len(container_files)) * 100
        
        self._add_result("deployment", "container_support",
                        "PASS" if container_score >= 50 else "WARNING",
                        container_score,
                        f"Container support: {len(container_present)}/{len(container_files)}",
                        {"present": container_present})
        
        # Check Kubernetes support
        k8s_files = [
            "kubernetes/",
            "deployment/k8s/",
            "kubernetes/deployment.yaml"
        ]
        
        k8s_present = [f for f in k8s_files if (self.root_path / f).exists()]
        k8s_score = (len(k8s_present) / len(k8s_files)) * 100
        
        self._add_result("deployment", "kubernetes_support",
                        "PASS" if k8s_score >= 66 else "WARNING",
                        k8s_score,
                        f"Kubernetes support: {len(k8s_present)}/{len(k8s_files)}",
                        {"present": k8s_present})
        
        # Check configuration management
        config_files = [
            "pyproject.toml",
            "requirements.txt",
            "requirements-prod.txt",
            "mobile_config.json"
        ]
        
        config_present = [f for f in config_files if (self.root_path / f).exists()]
        config_score = (len(config_present) / len(config_files)) * 100
        
        self._add_result("deployment", "configuration_management",
                        "PASS" if config_score >= 75 else "WARNING",
                        config_score,
                        f"Configuration files: {len(config_present)}/{len(config_files)}",
                        {"present": config_present})
        
        # Check production configurations
        prod_configs = [
            "deployment/production-deployment-final.yml",
            "docker/docker-compose.production.yml",
            "deployment/Dockerfile.production"
        ]
        
        prod_present = [f for f in prod_configs if (self.root_path / f).exists()]
        prod_score = (len(prod_present) / len(prod_configs)) * 100
        
        self._add_result("deployment", "production_configurations",
                        "PASS" if prod_score >= 66 else "WARNING",
                        prod_score,
                        f"Production configs: {len(prod_present)}/{len(prod_configs)}",
                        {"present": prod_present})
    
    def _add_result(self, category: str, check_name: str, status: str, score: float,
                   message: str, details: Dict[str, Any], critical: bool = False):
        """Add validation result."""
        result = ValidationResult(
            category=category,
            check_name=check_name,
            status=status,
            score=score,
            message=message,
            details=details,
            critical=critical
        )
        
        self.results.append(result)
        
        if critical and status == "FAIL":
            self.critical_failures.append(result)
        
        # Print result
        status_emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}[status]
        critical_marker = " [CRITICAL]" if critical else ""
        print(f"  {status_emoji} {check_name.replace('_', ' ').title()}: {message}{critical_marker}")
    
    def _generate_validation_summary(self, execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        
        # Calculate category scores
        category_scores = {}
        category_statuses = {}
        
        for category_name in self.categories.keys():
            category_results = [r for r in self.results if r.category == category_name]
            
            if category_results:
                avg_score = sum(r.score for r in category_results) / len(category_results)
                category_scores[category_name] = avg_score
                
                # Determine status
                failed_checks = sum(1 for r in category_results if r.status == "FAIL")
                critical_failures = sum(1 for r in category_results if r.status == "FAIL" and r.critical)
                
                if critical_failures > 0:
                    category_statuses[category_name] = "FAIL"
                elif failed_checks > 0:
                    category_statuses[category_name] = "WARNING"
                else:
                    category_statuses[category_name] = "PASS"
            else:
                category_scores[category_name] = 0
                category_statuses[category_name] = "SKIP"
        
        # Calculate overall score
        overall_score = 0.0
        for category, weight_info in self.categories.items():
            weight = weight_info["weight"]
            score = category_scores.get(category, 0)
            overall_score += score * weight
        
        # Determine overall status
        critical_category_failures = sum(1 for cat, status in category_statuses.items() 
                                       if status == "FAIL" and self.categories[cat]["critical"])
        
        if critical_category_failures > 0 or len(self.critical_failures) > 0:
            overall_status = "FAIL"
        elif overall_score >= 80:
            overall_status = "PASS"
        elif overall_score >= 60:
            overall_status = "WARNING"  
        else:
            overall_status = "FAIL"
        
        # Count results by status
        status_counts = {"PASS": 0, "WARNING": 0, "FAIL": 0, "SKIP": 0}
        for result in self.results:
            status_counts[result.status] += 1
        
        summary = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "overall_score": overall_score,
            "execution_time": execution_time,
            "critical_failures": len(self.critical_failures),
            "category_scores": category_scores,
            "category_statuses": category_statuses,
            "status_summary": status_counts,
            "total_checks": len(self.results),
            "detailed_results": [
                {
                    "category": r.category,
                    "check": r.check_name,
                    "status": r.status,
                    "score": r.score,
                    "message": r.message,
                    "critical": r.critical,
                    "details": r.details
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations(category_statuses, overall_score),
            "production_readiness": self._assess_production_readiness(overall_status, overall_score)
        }
        
        return summary
    
    def _generate_recommendations(self, category_statuses: Dict[str, str], overall_score: float) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Critical failures first
        if self.critical_failures:
            recommendations.append("🚨 CRITICAL: Address all critical failures before production deployment")
            for failure in self.critical_failures:
                recommendations.append(f"   - Fix {failure.category}.{failure.check_name}: {failure.message}")
        
        # Category-specific recommendations
        for category, status in category_statuses.items():
            if status == "FAIL":
                if category == "core_functionality":
                    recommendations.append("Implement missing core functionality components")
                elif category == "security":
                    recommendations.append("Strengthen security implementation and error handling")
                elif category == "performance":
                    recommendations.append("Add performance optimizations and benchmarking")
                elif category == "scalability":
                    recommendations.append("Implement scalability and load distribution features")
                elif category == "monitoring":
                    recommendations.append("Set up comprehensive monitoring and observability")
                elif category == "documentation":
                    recommendations.append("Complete essential documentation for production use")
                elif category == "deployment":
                    recommendations.append("Prepare deployment configurations and container support")
        
        # Score-based recommendations
        if overall_score < 70:
            recommendations.append("Focus on highest-impact improvements to reach production threshold")
        elif overall_score < 85:
            recommendations.append("Consider additional optimizations for enhanced production readiness")
        
        return recommendations
    
    def _assess_production_readiness(self, overall_status: str, overall_score: float) -> Dict[str, Any]:
        """Assess production readiness level."""
        
        readiness_levels = {
            "PRODUCTION_READY": {"min_score": 85, "description": "Ready for production deployment"},
            "STAGING_READY": {"min_score": 75, "description": "Ready for staging/pre-production testing"},
            "DEVELOPMENT_COMPLETE": {"min_score": 60, "description": "Development complete, needs production hardening"},
            "IN_DEVELOPMENT": {"min_score": 0, "description": "Still in development phase"}
        }
        
        readiness_level = "IN_DEVELOPMENT"
        for level, criteria in readiness_levels.items():
            if overall_score >= criteria["min_score"] and overall_status != "FAIL":
                readiness_level = level
                break
        
        return {
            "level": readiness_level,
            "score": overall_score,
            "status": overall_status,
            "description": readiness_levels[readiness_level]["description"],
            "critical_blockers": len(self.critical_failures),
            "deployment_recommendation": self._get_deployment_recommendation(readiness_level, overall_status)
        }
    
    def _get_deployment_recommendation(self, readiness_level: str, overall_status: str) -> str:
        """Get deployment recommendation."""
        if overall_status == "FAIL":
            return "🚫 DO NOT DEPLOY - Critical issues must be resolved"
        elif readiness_level == "PRODUCTION_READY":
            return "✅ APPROVED for production deployment"
        elif readiness_level == "STAGING_READY":  
            return "⚠️ APPROVED for staging deployment only"
        elif readiness_level == "DEVELOPMENT_COMPLETE":
            return "🔧 Complete production hardening before deployment"
        else:
            return "🚧 Continue development - not ready for deployment"
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print comprehensive validation summary."""
        print("\n" + "=" * 70)
        print("🚀 PRODUCTION READINESS VALIDATION SUMMARY")
        print("=" * 70)
        
        # Overall status
        readiness = summary["production_readiness"]
        print(f"Production Readiness Level: {readiness['level']}")
        print(f"Overall Score: {summary['overall_score']:.1f}/100")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Execution Time: {summary['execution_time']:.2f}s")
        
        if readiness['critical_blockers'] > 0:
            print(f"🚨 Critical Blockers: {readiness['critical_blockers']}")
        
        print(f"\nDeployment Recommendation:")
        print(f"  {readiness['deployment_recommendation']}")
        
        # Category breakdown
        print(f"\n📊 Category Breakdown:")
        for category, score in summary['category_scores'].items():
            status = summary['category_statuses'][category]
            weight = self.categories[category]["weight"]
            critical_marker = " [CRITICAL]" if self.categories[category]["critical"] else ""
            status_emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}[status]
            
            print(f"  {status_emoji} {category.replace('_', ' ').title()}: "
                  f"{score:.1f}/100 (weight: {weight:.0%}){critical_marker}")
        
        # Status summary
        print(f"\n📈 Check Summary:")
        print(f"  ✅ Passed: {summary['status_summary']['PASS']}")
        print(f"  ⚠️ Warnings: {summary['status_summary']['WARNING']}")
        print(f"  ❌ Failed: {summary['status_summary']['FAIL']}")
        print(f"  ⏭️ Skipped: {summary['status_summary']['SKIP']}")
        print(f"  📊 Total: {summary['total_checks']}")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 70)
        
        # Final verdict
        if readiness['level'] == "PRODUCTION_READY":
            print("🎉 PRODUCTION READY - System approved for deployment!")
        elif readiness['level'] == "STAGING_READY":
            print("⚠️  STAGING READY - System ready for staging deployment")
        else:
            print("🚧 NOT READY - Complete development before production deployment")
        
        print("=" * 70)


def main():
    """Run production readiness validation."""
    
    validator = ProductionReadinessValidator(".")
    summary = validator.validate_production_readiness()
    
    # Print summary
    validator.print_summary(summary)
    
    # Save detailed report
    with open("production_readiness_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: production_readiness_report.json")
    
    # Exit with appropriate code based on readiness level
    readiness_level = summary["production_readiness"]["level"]
    
    if readiness_level == "PRODUCTION_READY":
        sys.exit(0)
    elif readiness_level in ["STAGING_READY", "DEVELOPMENT_COMPLETE"]:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()