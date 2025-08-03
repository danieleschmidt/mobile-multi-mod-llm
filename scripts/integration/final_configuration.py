#!/usr/bin/env python3
"""
Final Configuration Script for Mobile Multi-Modal LLM SDLC

This script applies final configurations and optimizations to ensure
the repository meets excellence-level SDLC maturity standards.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalConfigurationManager:
    """Manages final configuration and optimization of SDLC components."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the final configuration manager."""
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config_applied = []
        self.errors = []
        
    def apply_all_configurations(self) -> Dict[str, Any]:
        """Apply all final configurations."""
        logger.info("Starting final SDLC configuration...")
        
        result = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "configurations_applied": [],
            "errors": [],
            "status": "unknown"
        }
        
        configurations = [
            ("Git configuration", self._configure_git),
            ("IDE settings", self._configure_ide_settings),
            ("Environment templates", self._configure_environment_templates),
            ("Automation scripts", self._configure_automation_scripts),
            ("Documentation templates", self._configure_documentation_templates),
            ("Quality gates", self._configure_quality_gates),
            ("Security configurations", self._configure_security_settings),
            ("Mobile optimization", self._configure_mobile_optimization),
            ("Monitoring dashboards", self._configure_monitoring_dashboards),
            ("Performance baselines", self._configure_performance_baselines)
        ]
        
        for config_name, config_func in configurations:
            try:
                logger.info(f"Applying {config_name}...")
                config_result = config_func()
                if config_result.get("success", False):
                    result["configurations_applied"].append({
                        "name": config_name,
                        "status": "success",
                        "details": config_result
                    })
                    self.config_applied.append(config_name)
                else:
                    result["errors"].append({
                        "name": config_name,
                        "error": config_result.get("error", "Unknown error")
                    })
                    self.errors.append(config_name)
                    
            except Exception as e:
                logger.error(f"Error applying {config_name}: {e}")
                result["errors"].append({
                    "name": config_name,
                    "error": str(e)
                })
                self.errors.append(config_name)
        
        # Determine overall status
        if not result["errors"]:
            result["status"] = "success"
        elif len(result["configurations_applied"]) > len(result["errors"]):
            result["status"] = "partial_success"
        else:
            result["status"] = "failed"
        
        logger.info(f"Final configuration completed with status: {result['status']}")
        return result
    
    def _configure_git(self) -> Dict[str, Any]:
        """Configure Git settings and hooks."""
        try:
            # Create .gitattributes for better handling of different file types
            gitattributes_content = """# Auto detect text files and perform LF normalization
* text=auto

# Ensure shell scripts always have LF line endings
*.sh text eol=lf

# Ensure Python files have consistent line endings
*.py text eol=lf

# Handle binary files
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.mov binary
*.mp4 binary
*.mp3 binary
*.tflite binary
*.onnx binary
*.mlpackage binary

# Documentation
*.md text
*.txt text
*.rst text

# Configuration files
*.json text
*.yaml text
*.yml text
*.toml text
*.ini text

# Jupyter notebooks (treat as text but with special handling)
*.ipynb text

# Archives
*.7z binary
*.gz binary
*.tar binary
*.zip binary

# Docker
Dockerfile text
*.dockerfile text

# GitHub
.gitignore text
.gitattributes text
"""
            
            gitattributes_path = self.project_root / ".gitattributes"
            with open(gitattributes_path, 'w') as f:
                f.write(gitattributes_content)
            
            # Create git hooks directory if it doesn't exist
            hooks_dir = self.project_root / ".git" / "hooks"
            if hooks_dir.exists():
                # Create pre-commit hook
                pre_commit_hook = hooks_dir / "pre-commit"
                if not pre_commit_hook.exists():
                    hook_content = """#!/bin/sh
# Pre-commit hook for Mobile Multi-Modal LLM

echo "Running pre-commit checks..."

# Run pre-commit if available
if command -v pre-commit >/dev/null 2>&1; then
    pre-commit run --all-files
    if [ $? -ne 0 ]; then
        echo "Pre-commit checks failed. Please fix the issues and try again."
        exit 1
    fi
fi

# Check for large files
large_files=$(find . -size +50M -not -path "./.git/*" -not -path "./monitoring/data/*")
if [ -n "$large_files" ]; then
    echo "Large files detected (>50MB):"
    echo "$large_files"
    echo "Please ensure these files should be committed or add them to .gitignore"
    exit 1
fi

echo "Pre-commit checks passed!"
"""
                    with open(pre_commit_hook, 'w') as f:
                        f.write(hook_content)
                    pre_commit_hook.chmod(0o755)
            
            return {"success": True, "details": "Git configuration applied"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _configure_ide_settings(self) -> Dict[str, Any]:
        """Configure IDE settings and extensions."""
        try:
            # VS Code settings
            vscode_dir = self.project_root / ".vscode"
            vscode_dir.mkdir(exist_ok=True)
            
            # Settings.json
            settings = {
                "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
                "python.linting.enabled": True,
                "python.linting.pylintEnabled": False,
                "python.linting.flake8Enabled": True,
                "python.linting.mypyEnabled": True,
                "python.formatting.provider": "black",
                "python.sortImports.args": ["--profile", "black"],
                "editor.formatOnSave": True,
                "editor.codeActionsOnSave": {
                    "source.organizeImports": True
                },
                "files.exclude": {
                    "**/__pycache__": True,
                    "**/.pytest_cache": True,
                    "**/.mypy_cache": True,
                    "**/node_modules": True,
                    "**/.coverage": True,
                    "**/htmlcov": True
                },
                "python.testing.pytestEnabled": True,
                "python.testing.unittestEnabled": False,
                "python.testing.pytestArgs": [
                    "tests"
                ],
                "jupyter.askForKernelRestart": False,
                "notebook.cellToolbarLocation": {
                    "default": "right",
                    "jupyter-notebook": "left"
                }
            }
            
            settings_path = vscode_dir / "settings.json"
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            # Extensions.json
            extensions = {
                "recommendations": [
                    "ms-python.python",
                    "ms-python.vscode-pylance",
                    "ms-python.black-formatter",
                    "ms-python.isort",
                    "ms-python.flake8",
                    "ms-toolsai.jupyter",
                    "ms-vscode.vscode-typescript-next",
                    "bradlc.vscode-tailwindcss",
                    "esbenp.prettier-vscode",
                    "ms-vscode.vscode-json",
                    "redhat.vscode-yaml",
                    "ms-vscode-remote.remote-containers",
                    "github.vscode-pull-request-github",
                    "github.copilot",
                    "ms-vsliveshare.vsliveshare",
                    "donjayamanne.githistory",
                    "eamodio.gitlens",
                    "gruntfuggly.todo-tree",
                    "streetsidesoftware.code-spell-checker"
                ]
            }
            
            extensions_path = vscode_dir / "extensions.json"
            with open(extensions_path, 'w') as f:
                json.dump(extensions, f, indent=2)
            
            # Launch configuration for debugging
            launch = {
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "Python: Current File",
                        "type": "python",
                        "request": "launch",
                        "program": "${file}",
                        "console": "integratedTerminal",
                        "cwd": "${workspaceFolder}"
                    },
                    {
                        "name": "Python: Module",
                        "type": "python",
                        "request": "launch",
                        "module": "mobile_multimodal",
                        "console": "integratedTerminal",
                        "cwd": "${workspaceFolder}"
                    },
                    {
                        "name": "Python: pytest",
                        "type": "python",
                        "request": "launch",
                        "module": "pytest",
                        "args": ["${workspaceFolder}/tests"],
                        "console": "integratedTerminal",
                        "cwd": "${workspaceFolder}"
                    }
                ]
            }
            
            launch_path = vscode_dir / "launch.json"
            with open(launch_path, 'w') as f:
                json.dump(launch, f, indent=2)
            
            return {"success": True, "details": "IDE settings configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _configure_environment_templates(self) -> Dict[str, Any]:
        """Configure environment templates and examples."""
        try:
            # Enhanced .env.example
            env_example_content = """# Mobile Multi-Modal LLM Environment Configuration

# =============================================================================
# Core Application Settings
# =============================================================================
APP_NAME="Mobile Multi-Modal LLM"
APP_VERSION="1.0.0"
DEBUG=false
LOG_LEVEL=INFO

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_PATH="./models"
DEFAULT_MODEL_NAME="multimodal_v1"
MODEL_CACHE_SIZE=512
QUANTIZATION_ENABLED=true
DEFAULT_QUANTIZATION="int2"

# =============================================================================
# Mobile Export Settings
# =============================================================================
ANDROID_TARGET_API=34
IOS_TARGET_VERSION=14.0
MAX_MODEL_SIZE_MB=35
EXPORT_VALIDATION_ENABLED=true

# =============================================================================
# Performance Settings
# =============================================================================
MAX_INFERENCE_TIME_MS=100
MIN_ACCURACY_THRESHOLD=0.90
PERFORMANCE_MONITORING_ENABLED=true
BENCHMARK_ITERATIONS=10

# =============================================================================
# Development Settings
# =============================================================================
DEV_MODE=false
HOT_RELOAD=false
PROFILING_ENABLED=false
TELEMETRY_ENABLED=true

# =============================================================================
# Testing Configuration
# =============================================================================
TEST_DATA_PATH="./tests/data"
PARALLEL_TESTING=true
COVERAGE_THRESHOLD=85
INTEGRATION_TESTS_ENABLED=true

# =============================================================================
# CI/CD Configuration
# =============================================================================
CI_ENVIRONMENT=false
AUTO_DEPLOY_ENABLED=false
DEPLOYMENT_TARGET="staging"

# =============================================================================
# Security Settings
# =============================================================================
SECURITY_SCANNING_ENABLED=true
VULNERABILITY_CHECKS=true
SECRET_SCANNING=true
DEPENDENCY_CHECKS=true

# =============================================================================
# Monitoring and Observability
# =============================================================================
PROMETHEUS_ENABLED=false
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"
HEALTH_CHECK_INTERVAL=300
METRICS_COLLECTION_ENABLED=true

# =============================================================================
# Notification Settings
# =============================================================================
SLACK_WEBHOOK_URL=""
TEAMS_WEBHOOK_URL=""
EMAIL_NOTIFICATIONS=false
NOTIFICATION_RECIPIENTS=""

# =============================================================================
# GitHub Integration
# =============================================================================
GITHUB_TOKEN=""
GITHUB_REPOSITORY="danieleschmidt/mobile-multi-mod-llm"
AUTO_PR_UPDATES=false

# =============================================================================
# Container Settings
# =============================================================================
DOCKER_REGISTRY="ghcr.io"
CONTAINER_TAG="latest"
RESOURCE_LIMITS_ENABLED=true
MEMORY_LIMIT="2Gi"
CPU_LIMIT="1000m"

# =============================================================================
# Database and Storage
# =============================================================================
DATA_STORAGE_PATH="./data"
CACHE_STORAGE_PATH="./cache"
LOG_STORAGE_PATH="./logs"
BACKUP_ENABLED=false
RETENTION_DAYS=30

# =============================================================================
# External Services
# =============================================================================
# Add your external service configurations here
"""
            
            env_example_path = self.project_root / ".env.example"
            with open(env_example_path, 'w') as f:
                f.write(env_example_content)
            
            return {"success": True, "details": "Environment templates configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _configure_automation_scripts(self) -> Dict[str, Any]:
        """Configure automation scripts permissions and shortcuts."""
        try:
            scripts_dir = self.project_root / "scripts"
            
            # Make all Python scripts executable
            for script_path in scripts_dir.rglob("*.py"):
                if script_path.is_file():
                    script_path.chmod(0o755)
            
            # Make shell scripts executable
            for script_path in scripts_dir.rglob("*.sh"):
                if script_path.is_file():
                    script_path.chmod(0o755)
            
            # Create convenience scripts in project root
            convenience_scripts = {
                "dev-setup.sh": """#!/bin/bash
# Quick development environment setup
set -e
echo "Setting up development environment..."
bash scripts/setup_dev_environment.sh
echo "Development environment ready!"
""",
                "run-tests.sh": """#!/bin/bash
# Run all tests with coverage
set -e
echo "Running comprehensive test suite..."
pytest tests/ --cov=src --cov-report=html --cov-report=term
echo "Tests completed!"
""",
                "mobile-export.sh": """#!/bin/bash
# Export models for all mobile platforms
set -e
echo "Exporting models for mobile platforms..."
python scripts/export_models.py --platform all --validate
echo "Mobile export completed!"
""",
                "health-check.sh": """#!/bin/bash
# Run repository health check
set -e
echo "Running repository health check..."
python scripts/automation/repository_health_monitor.py --dashboard health-dashboard.html
echo "Health check completed! Dashboard: health-dashboard.html"
"""
            }
            
            for script_name, script_content in convenience_scripts.items():
                script_path = self.project_root / script_name
                with open(script_path, 'w') as f:
                    f.write(script_content)
                script_path.chmod(0o755)
            
            return {"success": True, "details": "Automation scripts configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _configure_documentation_templates(self) -> Dict[str, Any]:
        """Configure documentation templates and structure."""
        try:
            # Create documentation templates
            templates_dir = self.project_root / "docs" / "templates"
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Pull Request template
            pr_template = """## Summary
Brief description of the changes made in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Mobile Impact
- [ ] Android compatibility verified
- [ ] iOS compatibility verified
- [ ] ONNX export validated
- [ ] Model size within limits (<35MB)
- [ ] Performance benchmarks passed

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Performance testing completed

## Security
- [ ] Security scan passed
- [ ] No sensitive data exposed
- [ ] Dependencies updated
- [ ] Vulnerability scan clean

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Breaking changes documented
- [ ] Mobile exports validated

## Additional Notes
Any additional information, dependencies, or context needed for reviewers.
"""
            
            pr_template_path = self.project_root / ".github" / "pull_request_template.md"
            pr_template_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pr_template_path, 'w') as f:
                f.write(pr_template)
            
            # Issue templates
            issue_templates_dir = self.project_root / ".github" / "ISSUE_TEMPLATE"
            issue_templates_dir.mkdir(parents=True, exist_ok=True)
            
            bug_template = """---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Screenshots
If applicable, add screenshots to help explain your problem.

## Environment
- OS: [e.g. iOS, Android, Windows, macOS, Linux]
- Python Version: [e.g. 3.11]
- Model Version: [e.g. v1.0.0]
- Platform: [e.g. TensorFlow Lite, Core ML, ONNX]

## Additional Context
Add any other context about the problem here.
"""
            
            with open(issue_templates_dir / "bug_report.md", 'w') as f:
                f.write(bug_template)
            
            feature_template = """---
name: Feature Request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

## Feature Description
A clear and concise description of what you want to happen.

## Problem Statement
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

## Proposed Solution
A clear and concise description of what you want to happen.

## Mobile Impact
- [ ] Affects Android platform
- [ ] Affects iOS platform
- [ ] Affects ONNX compatibility
- [ ] Impacts model size
- [ ] Affects performance

## Alternative Solutions
A clear and concise description of any alternative solutions or features you've considered.

## Additional Context
Add any other context or screenshots about the feature request here.
"""
            
            with open(issue_templates_dir / "feature_request.md", 'w') as f:
                f.write(feature_template)
            
            return {"success": True, "details": "Documentation templates configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _configure_quality_gates(self) -> Dict[str, Any]:
        """Configure quality gates and validation rules."""
        try:
            # Create quality gate configuration
            quality_config = {
                "code_quality": {
                    "min_coverage": 85,
                    "max_complexity": 10,
                    "max_file_length": 500,
                    "max_function_length": 50
                },
                "security": {
                    "max_critical_vulnerabilities": 0,
                    "max_high_vulnerabilities": 2,
                    "dependency_check_required": True,
                    "secret_scan_required": True
                },
                "performance": {
                    "max_inference_time_ms": 100,
                    "min_accuracy": 0.90,
                    "max_model_size_mb": 35,
                    "benchmark_required": True
                },
                "mobile": {
                    "android_compatibility_required": True,
                    "ios_compatibility_required": True,
                    "onnx_compatibility_required": True,
                    "cross_platform_validation": True
                }
            }
            
            quality_config_path = self.project_root / ".github" / "quality-gates.yml"
            with open(quality_config_path, 'w') as f:
                yaml.dump(quality_config, f, default_flow_style=False)
            
            return {"success": True, "details": "Quality gates configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _configure_security_settings(self) -> Dict[str, Any]:
        """Configure security settings and policies."""
        try:
            # Security policy
            security_policy = """# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability, please report it to us as follows:

1. **Do not** create a public GitHub issue for security vulnerabilities
2. Send an email to security@terragon-labs.com with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 24 hours and provide a detailed response within 72 hours.

## Security Measures

- All dependencies are regularly scanned for vulnerabilities
- Code is scanned using multiple security tools (Bandit, Safety, etc.)
- Mobile models are signed and validated
- Sensitive data is never logged or exposed
- Regular security audits are performed

## Responsible Disclosure

We follow responsible disclosure practices and will:
- Acknowledge your report within 24 hours
- Provide a timeline for addressing the vulnerability
- Keep you informed of progress
- Credit you in our security advisories (if desired)
"""
            
            security_policy_path = self.project_root / "SECURITY.md"
            with open(security_policy_path, 'w') as f:
                f.write(security_policy)
            
            return {"success": True, "details": "Security settings configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _configure_mobile_optimization(self) -> Dict[str, Any]:
        """Configure mobile-specific optimizations."""
        try:
            # Mobile optimization configuration
            mobile_config = {
                "optimization": {
                    "quantization": {
                        "default": "int2",
                        "options": ["int2", "int4", "int8", "fp16"],
                        "platform_specific": {
                            "android": "int2",
                            "ios": "int2",
                            "onnx": "int4"
                        }
                    },
                    "model_compression": {
                        "enabled": True,
                        "algorithms": ["pruning", "quantization", "distillation"]
                    },
                    "hardware_acceleration": {
                        "android": ["gpu", "npu", "dsp"],
                        "ios": ["neural_engine", "gpu"],
                        "general": ["cpu", "gpu"]
                    }
                },
                "validation": {
                    "size_limits": {
                        "android": "35MB",
                        "ios": "35MB",
                        "onnx": "40MB"
                    },
                    "performance_targets": {
                        "inference_time_ms": 100,
                        "memory_usage_mb": 512,
                        "accuracy_threshold": 0.90
                    }
                }
            }
            
            mobile_config_path = self.project_root / "configs" / "mobile_optimization.yml"
            mobile_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mobile_config_path, 'w') as f:
                yaml.dump(mobile_config, f, default_flow_style=False)
            
            return {"success": True, "details": "Mobile optimization configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _configure_monitoring_dashboards(self) -> Dict[str, Any]:
        """Configure monitoring dashboards and alerts."""
        try:
            # Create dashboard configuration
            dashboard_config = {
                "dashboards": {
                    "system_health": {
                        "panels": [
                            "cpu_usage", "memory_usage", "disk_usage",
                            "network_io", "health_score"
                        ]
                    },
                    "application_performance": {
                        "panels": [
                            "inference_time", "throughput", "accuracy",
                            "error_rate", "response_time"
                        ]
                    },
                    "mobile_metrics": {
                        "panels": [
                            "export_success_rate", "model_size_trend",
                            "platform_compatibility", "validation_results"
                        ]
                    }
                },
                "alerts": {
                    "health_score_critical": {
                        "condition": "health_score < 70",
                        "severity": "critical",
                        "channels": ["slack", "email"]
                    },
                    "inference_time_high": {
                        "condition": "inference_time > 100ms",
                        "severity": "warning",
                        "channels": ["slack"]
                    }
                }
            }
            
            dashboard_config_path = self.project_root / "monitoring" / "dashboards.yml"
            dashboard_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dashboard_config_path, 'w') as f:
                yaml.dump(dashboard_config, f, default_flow_style=False)
            
            return {"success": True, "details": "Monitoring dashboards configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _configure_performance_baselines(self) -> Dict[str, Any]:
        """Configure performance baselines and benchmarks."""
        try:
            # Performance baseline configuration
            baselines = {
                "inference_performance": {
                    "android_tflite": {
                        "target_time_ms": 80,
                        "acceptable_time_ms": 100,
                        "memory_usage_mb": 256
                    },
                    "ios_coreml": {
                        "target_time_ms": 60,
                        "acceptable_time_ms": 80,
                        "memory_usage_mb": 200
                    },
                    "onnx_runtime": {
                        "target_time_ms": 70,
                        "acceptable_time_ms": 90,
                        "memory_usage_mb": 300
                    }
                },
                "accuracy_baselines": {
                    "minimum_accuracy": 0.90,
                    "target_accuracy": 0.95,
                    "cross_platform_variance": 0.02
                },
                "resource_usage": {
                    "build_time_minutes": 15,
                    "test_time_minutes": 10,
                    "export_time_minutes": 5
                }
            }
            
            baselines_path = self.project_root / "configs" / "performance_baselines.yml"
            baselines_path.parent.mkdir(parents=True, exist_ok=True)
            with open(baselines_path, 'w') as f:
                yaml.dump(baselines, f, default_flow_style=False)
            
            return {"success": True, "details": "Performance baselines configured"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    """Main entry point for final configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final SDLC Configuration")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_root = Path(args.project_root) if args.project_root else None
    manager = FinalConfigurationManager(project_root)
    
    result = manager.apply_all_configurations()
    
    print(f"\nFinal Configuration Results:")
    print(f"Status: {result['status'].upper()}")
    print(f"Configurations Applied: {len(result['configurations_applied'])}")
    print(f"Errors: {len(result['errors'])}")
    
    if result['configurations_applied']:
        print("\nSuccessful Configurations:")
        for config in result['configurations_applied']:
            print(f"  ✅ {config['name']}")
    
    if result['errors']:
        print("\nFailed Configurations:")
        for error in result['errors']:
            print(f"  ❌ {error['name']}: {error['error']}")
    
    # Exit with appropriate code
    if result['status'] == 'success':
        sys.exit(0)
    elif result['status'] == 'partial_success':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()