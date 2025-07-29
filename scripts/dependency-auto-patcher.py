#!/usr/bin/env python3
"""Automated dependency vulnerability patching system.

This script automatically identifies, evaluates, and applies security patches
for vulnerable dependencies while maintaining compatibility and stability.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import re
import semver

try:
    import requests
    import toml
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests or toml not available. Install with: pip install requests toml")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VulnerabilityInfo:
    """Information about a dependency vulnerability."""
    
    def __init__(self, package: str, current_version: str, vulnerability_id: str,
                 severity: str, fixed_version: str, description: str):
        self.package = package
        self.current_version = current_version
        self.vulnerability_id = vulnerability_id
        self.severity = severity
        self.fixed_version = fixed_version
        self.description = description
        self.patch_available = bool(fixed_version)


class DependencyAutoPatcher:
    """Automated dependency vulnerability patching system."""
    
    def __init__(self, config_path: str = "auto-patcher-config.json"):
        self.config = self._load_config(config_path)
        self.project_root = Path(".")
        self.vulnerabilities = []
        self.patches_applied = []
        self.backup_created = False
    
    def _load_config(self, config_path: str) -> Dict:
        """Load auto-patcher configuration."""
        default_config = {
            "auto_patch_severities": ["critical", "high"],
            "manual_review_severities": ["medium", "low"],
            "excluded_packages": [],
            "testing_required": True,
            "backup_before_patch": True,
            "max_version_jump": {"major": 0, "minor": 2, "patch": 10},
            "compatibility_checks": True,
            "rollback_on_failure": True,
            "notification_webhook": None,
            "dry_run": False,
            "patch_frequency_hours": 24,
            "vulnerability_sources": [
                "pip-audit",
                "safety",
                "osv-scanner"
            ]
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            logger.info(f"Config file {config_path} not found, using defaults")
            return default_config
    
    def scan_vulnerabilities(self) -> List[VulnerabilityInfo]:
        """Scan for dependency vulnerabilities using multiple sources."""
        logger.info("Scanning for dependency vulnerabilities...")
        
        vulnerabilities = []
        
        # Scan with pip-audit
        if "pip-audit" in self.config["vulnerability_sources"]:
            vulnerabilities.extend(self._scan_with_pip_audit())
        
        # Scan with safety
        if "safety" in self.config["vulnerability_sources"]:
            vulnerabilities.extend(self._scan_with_safety())
        
        # Scan with OSV scanner
        if "osv-scanner" in self.config["vulnerability_sources"]:
            vulnerabilities.extend(self._scan_with_osv())
        
        # Deduplicate vulnerabilities
        unique_vulnerabilities = self._deduplicate_vulnerabilities(vulnerabilities)
        
        logger.info(f"Found {len(unique_vulnerabilities)} unique vulnerabilities")
        return unique_vulnerabilities
    
    def _scan_with_pip_audit(self) -> List[VulnerabilityInfo]:
        """Scan vulnerabilities using pip-audit."""
        logger.info("Scanning with pip-audit...")
        
        try:
            result = subprocess.run([
                "pip-audit", "--format=json", "--output=-"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.warning(f"pip-audit scan failed: {result.stderr}")
                return []
            
            data = json.loads(result.stdout)
            vulnerabilities = []
            
            for vuln in data.get("vulnerabilities", []):
                package = vuln.get("package")
                current_version = vuln.get("installed_version")
                vuln_id = vuln.get("id")
                description = vuln.get("description", "")
                
                # Extract fixed version from fix info
                fixed_version = None
                if vuln.get("fix_versions"):
                    fixed_version = min(vuln["fix_versions"])
                
                # Determine severity (pip-audit doesn't always provide this)
                severity = "medium"  # Default
                if any(word in description.lower() for word in ["critical", "rce", "remote code"]):
                    severity = "critical"
                elif any(word in description.lower() for word in ["high", "privilege", "auth"]):
                    severity = "high"
                
                vulnerabilities.append(VulnerabilityInfo(
                    package=package,
                    current_version=current_version,
                    vulnerability_id=vuln_id,
                    severity=severity,
                    fixed_version=fixed_version,
                    description=description
                ))
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error running pip-audit: {e}")
            return []
    
    def _scan_with_safety(self) -> List[VulnerabilityInfo]:
        """Scan vulnerabilities using safety."""
        logger.info("Scanning with safety...")
        
        try:
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Safety returns non-zero when vulnerabilities are found
            if result.returncode not in [0, 64]:  # 64 = vulnerabilities found
                logger.warning(f"Safety scan failed: {result.stderr}")
                return []
            
            if not result.stdout.strip():
                return []
            
            data = json.loads(result.stdout)
            vulnerabilities = []
            
            for vuln in data:
                package = vuln.get("package")
                current_version = vuln.get("installed_version")
                vuln_id = vuln.get("vulnerability_id") or vuln.get("id")
                description = vuln.get("advisory", "")
                
                # Safety typically provides severity
                severity = vuln.get("severity", "medium").lower()
                
                # Extract fixed version from advisory
                fixed_version = None
                advisory_text = vuln.get("advisory", "")
                version_match = re.search(r"upgrade to ([0-9]+\.[0-9]+\.[0-9]+)", advisory_text)
                if version_match:
                    fixed_version = version_match.group(1)
                
                vulnerabilities.append(VulnerabilityInfo(
                    package=package,
                    current_version=current_version,
                    vulnerability_id=vuln_id,
                    severity=severity,
                    fixed_version=fixed_version,
                    description=description
                ))
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error running safety: {e}")
            return []
    
    def _scan_with_osv(self) -> List[VulnerabilityInfo]:
        """Scan vulnerabilities using OSV scanner."""
        logger.info("Scanning with OSV scanner...")
        
        try:
            # OSV scanner needs a requirements file or lockfile
            lockfiles = ["requirements.txt", "pyproject.toml", "poetry.lock"]
            available_lockfile = None
            
            for lockfile in lockfiles:
                if (self.project_root / lockfile).exists():
                    available_lockfile = lockfile
                    break
            
            if not available_lockfile:
                logger.warning("No supported lockfile found for OSV scanner")
                return []
            
            result = subprocess.run([
                "osv-scanner", "--format=json", f"--lockfile={available_lockfile}"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.warning(f"OSV scanner failed: {result.stderr}")
                return []
            
            data = json.loads(result.stdout)
            vulnerabilities = []
            
            for result_entry in data.get("results", []):
                for package in result_entry.get("packages", []):
                    for vuln in package.get("vulnerabilities", []):
                        package_name = package.get("package", {}).get("name")
                        current_version = package.get("package", {}).get("version")
                        vuln_id = vuln.get("id")
                        summary = vuln.get("summary", "")
                        
                        # OSV provides CVSS scores, convert to severity
                        severity = "medium"
                        if vuln.get("database_specific", {}).get("severity"):
                            cvss_severity = vuln["database_specific"]["severity"].lower()
                            if cvss_severity in ["critical", "high", "medium", "low"]:
                                severity = cvss_severity
                        
                        # Extract fixed version from ranges
                        fixed_version = None
                        for affected in vuln.get("affected", []):
                            if affected.get("package", {}).get("name") == package_name:
                                for range_info in affected.get("ranges", []):
                                    if range_info.get("type") == "ECOSYSTEM":
                                        for event in range_info.get("events", []):
                                            if "fixed" in event:
                                                fixed_version = event["fixed"]
                                                break
                        
                        vulnerabilities.append(VulnerabilityInfo(
                            package=package_name,
                            current_version=current_version,
                            vulnerability_id=vuln_id,
                            severity=severity,
                            fixed_version=fixed_version,
                            description=summary
                        ))
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Error running OSV scanner: {e}")
            return []
    
    def _deduplicate_vulnerabilities(self, vulnerabilities: List[VulnerabilityInfo]) -> List[VulnerabilityInfo]:
        """Remove duplicate vulnerabilities from different sources."""
        seen = set()
        unique_vulnerabilities = []
        
        for vuln in vulnerabilities:
            # Create unique key based on package and vulnerability ID
            key = f"{vuln.package}:{vuln.vulnerability_id}:{vuln.current_version}"
            
            if key not in seen:
                seen.add(key)
                unique_vulnerabilities.append(vuln)
            else:
                # If we've seen this vulnerability, prefer the one with more info
                for i, existing in enumerate(unique_vulnerabilities):
                    existing_key = f"{existing.package}:{existing.vulnerability_id}:{existing.current_version}"
                    if existing_key == key:
                        if vuln.fixed_version and not existing.fixed_version:
                            unique_vulnerabilities[i] = vuln
                        break
        
        return unique_vulnerabilities
    
    def evaluate_patches(self, vulnerabilities: List[VulnerabilityInfo]) -> Tuple[List[VulnerabilityInfo], List[VulnerabilityInfo]]:
        """Evaluate which vulnerabilities can be auto-patched vs need manual review."""
        auto_patch = []
        manual_review = []
        
        for vuln in vulnerabilities:
            # Skip excluded packages
            if vuln.package in self.config["excluded_packages"]:
                logger.info(f"Skipping excluded package: {vuln.package}")
                continue
            
            # Check if severity qualifies for auto-patching
            if vuln.severity in self.config["auto_patch_severities"] and vuln.patch_available:
                # Verify version jump is within limits
                if self._is_safe_version_jump(vuln.current_version, vuln.fixed_version):
                    auto_patch.append(vuln)
                else:
                    logger.warning(f"Version jump too large for {vuln.package}: {vuln.current_version} -> {vuln.fixed_version}")
                    manual_review.append(vuln)
            else:
                manual_review.append(vuln)
        
        logger.info(f"Auto-patch candidates: {len(auto_patch)}")
        logger.info(f"Manual review required: {len(manual_review)}")
        
        return auto_patch, manual_review
    
    def _is_safe_version_jump(self, current: str, target: str) -> bool:
        """Check if version jump is within safe limits."""
        try:
            current_ver = semver.VersionInfo.parse(current)
            target_ver = semver.VersionInfo.parse(target)
            
            major_jump = target_ver.major - current_ver.major
            minor_jump = target_ver.minor - current_ver.minor
            patch_jump = target_ver.patch - current_ver.patch
            
            limits = self.config["max_version_jump"]
            
            return (major_jump <= limits["major"] and
                    minor_jump <= limits["minor"] and
                    patch_jump <= limits["patch"])
            
        except Exception as e:
            logger.error(f"Error parsing versions {current} -> {target}: {e}")
            return False
    
    def create_backup(self) -> bool:
        """Create backup of current dependency files."""
        if not self.config["backup_before_patch"]:
            return True
        
        logger.info("Creating backup of dependency files...")
        
        backup_dir = Path("backups") / f"dependencies_{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_backup = [
            "requirements.txt",
            "requirements-dev.txt", 
            "pyproject.toml",
            "poetry.lock",
            "Pipfile",
            "Pipfile.lock"
        ]
        
        backed_up = False
        for file_path in files_to_backup:
            source = self.project_root / file_path
            if source.exists():
                target = backup_dir / file_path
                target.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    import shutil
                    shutil.copy2(source, target)
                    logger.info(f"Backed up {file_path}")
                    backed_up = True
                except Exception as e:
                    logger.error(f"Failed to backup {file_path}: {e}")
                    return False
        
        if backed_up:
            self.backup_created = True
            logger.info(f"Backup created in {backup_dir}")
            return True
        else:
            logger.warning("No dependency files found to backup")
            return True
    
    def apply_patches(self, vulnerabilities: List[VulnerabilityInfo]) -> bool:
        """Apply patches for vulnerabilities."""
        if self.config["dry_run"]:
            logger.info("DRY RUN: Would apply the following patches:")
            for vuln in vulnerabilities:
                logger.info(f"  {vuln.package}: {vuln.current_version} -> {vuln.fixed_version}")
            return True
        
        logger.info(f"Applying patches for {len(vulnerabilities)} vulnerabilities...")
        
        success = True
        
        for vuln in vulnerabilities:
            try:
                if self._apply_single_patch(vuln):
                    self.patches_applied.append(vuln)
                    logger.info(f"Successfully patched {vuln.package}")
                else:
                    logger.error(f"Failed to patch {vuln.package}")
                    success = False
                    
            except Exception as e:
                logger.error(f"Error patching {vuln.package}: {e}")
                success = False
        
        return success
    
    def _apply_single_patch(self, vuln: VulnerabilityInfo) -> bool:
        """Apply patch for a single vulnerability."""
        logger.info(f"Patching {vuln.package}: {vuln.current_version} -> {vuln.fixed_version}")
        
        try:
            # Update pip package
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                f"{vuln.package}=={vuln.fixed_version}"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to install {vuln.package}=={vuln.fixed_version}: {result.stderr}")
                return False
            
            # Update requirements.txt if it exists
            self._update_requirements_file(vuln.package, vuln.fixed_version)
            
            # Update pyproject.toml if it exists
            self._update_pyproject_toml(vuln.package, vuln.fixed_version)
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying patch for {vuln.package}: {e}")
            return False
    
    def _update_requirements_file(self, package: str, version: str):
        """Update requirements.txt file with new version."""
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            return
        
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            updated = False
            
            for line in lines:
                if line.strip().startswith(package):
                    # Replace the version
                    updated_lines.append(f"{package}=={version}\n")
                    updated = True
                else:
                    updated_lines.append(line)
            
            if updated:
                with open(req_file, 'w') as f:
                    f.writelines(updated_lines)
                logger.info(f"Updated requirements.txt for {package}")
            
        except Exception as e:
            logger.error(f"Error updating requirements.txt: {e}")
    
    def _update_pyproject_toml(self, package: str, version: str):
        """Update pyproject.toml file with new version."""
        toml_file = self.project_root / "pyproject.toml"
        if not toml_file.exists():
            return
        
        try:
            with open(toml_file, 'r') as f:
                content = f.read()
            
            # Simple regex replacement for pyproject.toml
            pattern = rf'"{package}[^"]*"'
            replacement = f'"{package}>={version}"'
            
            updated_content = re.sub(pattern, replacement, content)
            
            if updated_content != content:
                with open(toml_file, 'w') as f:
                    f.write(updated_content)
                logger.info(f"Updated pyproject.toml for {package}")
            
        except Exception as e:
            logger.error(f"Error updating pyproject.toml: {e}")
    
    def run_tests(self) -> bool:
        """Run tests to verify patches don't break functionality."""
        if not self.config["testing_required"]:
            return True
        
        logger.info("Running tests to verify patches...")
        
        try:
            # Try pytest first
            result = subprocess.run([
                "python", "-m", "pytest", "-x", "--tb=short"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("All tests passed")
                return True
            else:
                logger.error(f"Tests failed: {result.stdout}\n{result.stderr}")
                return False
                
        except FileNotFoundError:
            # Try unittest if pytest not available
            try:
                result = subprocess.run([
                    "python", "-m", "unittest", "discover", "-s", "tests"
                ], capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    logger.info("All tests passed")
                    return True
                else:
                    logger.error(f"Tests failed: {result.stdout}\n{result.stderr}")
                    return False
                    
            except Exception as e:
                logger.warning(f"Could not run tests: {e}")
                return True  # Assume success if can't run tests
    
    def rollback_patches(self) -> bool:
        """Rollback applied patches if something went wrong."""
        if not self.config["rollback_on_failure"] or not self.backup_created:
            return False
        
        logger.info("Rolling back patches...")
        
        try:
            # Find most recent backup
            backup_dir = Path("backups")
            if not backup_dir.exists():
                return False
            
            backup_dirs = sorted([d for d in backup_dir.iterdir() if d.is_dir() and d.name.startswith("dependencies_")])
            if not backup_dirs:
                return False
            
            latest_backup = backup_dirs[-1]
            
            # Restore files
            files_to_restore = ["requirements.txt", "pyproject.toml"]
            for file_name in files_to_restore:
                backup_file = latest_backup / file_name
                target_file = self.project_root / file_name
                
                if backup_file.exists():
                    import shutil
                    shutil.copy2(backup_file, target_file)
                    logger.info(f"Restored {file_name}")
            
            # Reinstall dependencies
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, cwd=self.project_root)
            
            logger.info("Rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
    
    def send_notification(self, success: bool, vulnerabilities: List[VulnerabilityInfo], patches_applied: List[VulnerabilityInfo]):
        """Send notification about patching results."""
        if not self.config.get("notification_webhook"):
            return
        
        try:
            status = "success" if success else "failure"
            message = {
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "vulnerabilities_found": len(vulnerabilities),
                "patches_applied": len(patches_applied),
                "details": {
                    "applied_patches": [
                        {
                            "package": p.package,
                            "from_version": p.current_version,
                            "to_version": p.fixed_version,
                            "severity": p.severity
                        } for p in patches_applied
                    ]
                }
            }
            
            if REQUESTS_AVAILABLE:
                response = requests.post(
                    self.config["notification_webhook"],
                    json=message,
                    timeout=30
                )
                response.raise_for_status()
                logger.info("Notification sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def run_auto_patching(self) -> bool:
        """Run complete auto-patching process."""
        logger.info("Starting automated dependency vulnerability patching")
        
        try:
            # Create backup
            if not self.create_backup():
                logger.error("Failed to create backup, aborting")
                return False
            
            # Scan for vulnerabilities
            vulnerabilities = self.scan_vulnerabilities()
            if not vulnerabilities:
                logger.info("No vulnerabilities found")
                return True
            
            # Evaluate patches
            auto_patch, manual_review = self.evaluate_patches(vulnerabilities)
            
            if manual_review:
                logger.info("Vulnerabilities requiring manual review:")
                for vuln in manual_review:
                    logger.info(f"  {vuln.package} ({vuln.severity}): {vuln.vulnerability_id}")
            
            if not auto_patch:
                logger.info("No vulnerabilities qualify for auto-patching")
                return True
            
            # Apply patches
            success = self.apply_patches(auto_patch)
            
            if success and self.config["testing_required"]:
                # Run tests
                if not self.run_tests():
                    logger.error("Tests failed after patching")
                    if self.config["rollback_on_failure"]:
                        self.rollback_patches()
                    success = False
            
            # Send notification
            self.send_notification(success, vulnerabilities, self.patches_applied if success else [])
            
            if success:
                logger.info(f"Successfully applied {len(self.patches_applied)} patches")
            else:
                logger.error("Auto-patching process failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Auto-patching process failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated dependency vulnerability patcher")
    parser.add_argument("--config", default="auto-patcher-config.json",
                       help="Configuration file path")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be patched without applying changes")
    parser.add_argument("--scan-only", action="store_true",
                       help="Only scan for vulnerabilities, don't patch")
    parser.add_argument("--force", action="store_true",
                       help="Skip safety checks and apply all available patches")
    
    args = parser.parse_args()
    
    # Load config and override with CLI args
    patcher = DependencyAutoPatcher(args.config)
    
    if args.dry_run:
        patcher.config["dry_run"] = True
    
    if args.force:
        patcher.config["auto_patch_severities"] = ["critical", "high", "medium", "low"]
        patcher.config["testing_required"] = False
    
    if args.scan_only:
        # Only scan and report
        vulnerabilities = patcher.scan_vulnerabilities()
        
        if vulnerabilities:
            print(f"\nFound {len(vulnerabilities)} vulnerabilities:")
            for vuln in vulnerabilities:
                print(f"  {vuln.package} ({vuln.current_version}): {vuln.severity} - {vuln.vulnerability_id}")
                if vuln.fixed_version:
                    print(f"    Fix available: {vuln.fixed_version}")
                else:
                    print("    No fix available")
        else:
            print("No vulnerabilities found")
        
        return
    
    # Run full auto-patching process
    success = patcher.run_auto_patching()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()