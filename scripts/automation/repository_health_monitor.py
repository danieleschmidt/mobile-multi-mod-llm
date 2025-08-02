#!/usr/bin/env python3
"""
Repository health monitoring and automation system.
Monitors repository health, performs automated maintenance, and generates alerts.
"""

import json
import logging
import os
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepositoryHealthMonitor:
    """Monitors repository health and automates maintenance tasks."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the repository health monitor."""
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.parent.parent
        self.metrics_file = self.project_root / ".github" / "project-metrics.json"
        self.alerts_sent = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration for the health monitor."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        return {
            "thresholds": {
                "health_score": {
                    "critical": 70,
                    "warning": 85
                },
                "test_coverage": {
                    "critical": 80,
                    "warning": 85
                },
                "build_success_rate": {
                    "critical": 90,
                    "warning": 95
                },
                "security_vulnerabilities": {
                    "critical_max": 1,
                    "high_max": 3
                },
                "performance_degradation": {
                    "critical": 20,
                    "warning": 10
                },
                "dependency_age": {
                    "warning_days": 90,
                    "critical_days": 180
                }
            },
            "notifications": {
                "slack": {
                    "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                    "channel": "#mobile-ml-alerts"
                },
                "email": {
                    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                    "username": os.getenv("EMAIL_USERNAME"),
                    "password": os.getenv("EMAIL_PASSWORD"),
                    "recipients": os.getenv("ALERT_RECIPIENTS", "").split(",")
                },
                "github": {
                    "token": os.getenv("GITHUB_TOKEN"),
                    "repo": os.getenv("GITHUB_REPOSITORY", "danieleschmidt/mobile-multi-mod-llm")
                }
            },
            "automation": {
                "auto_fix_enabled": os.getenv("AUTO_FIX_ENABLED", "false").lower() == "true",
                "auto_update_dependencies": os.getenv("AUTO_UPDATE_DEPS", "false").lower() == "true",
                "auto_merge_minor": os.getenv("AUTO_MERGE_MINOR", "false").lower() == "true"
            }
        }
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive repository health check."""
        logger.info("Starting repository health check...")
        
        health_report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "overall_status": "healthy",
            "issues": [],
            "recommendations": [],
            "metrics": {}
        }
        
        try:
            # Load current metrics
            if self.metrics_file.exists():
                with open(self.metrics_file) as f:
                    metrics = json.load(f)
                health_report["metrics"] = metrics
            else:
                logger.warning("Metrics file not found, running metrics collection...")
                from metrics_collector import MetricsCollector
                collector = MetricsCollector()
                metrics = collector.collect_all_metrics()
                health_report["metrics"] = metrics
            
            # Analyze health indicators
            issues = []
            recommendations = []
            
            # Check overall health score
            health_score = metrics.get("health_score", {}).get("overall", 100)
            if health_score < self.config["thresholds"]["health_score"]["critical"]:
                issues.append({
                    "type": "critical",
                    "category": "overall_health",
                    "message": f"Overall health score ({health_score}) is below critical threshold ({self.config['thresholds']['health_score']['critical']})",
                    "impact": "high",
                    "auto_fixable": False
                })
            elif health_score < self.config["thresholds"]["health_score"]["warning"]:
                issues.append({
                    "type": "warning",
                    "category": "overall_health", 
                    "message": f"Overall health score ({health_score}) is below warning threshold ({self.config['thresholds']['health_score']['warning']})",
                    "impact": "medium",
                    "auto_fixable": False
                })
            
            # Check test coverage
            coverage = metrics.get("code_quality", {}).get("test_coverage", {}).get("percentage", 100)
            if coverage < self.config["thresholds"]["test_coverage"]["critical"]:
                issues.append({
                    "type": "critical",
                    "category": "code_quality",
                    "message": f"Test coverage ({coverage}%) is below critical threshold ({self.config['thresholds']['test_coverage']['critical']}%)",
                    "impact": "high",
                    "auto_fixable": False
                })
                recommendations.append("Add more unit tests to improve coverage")
            
            # Check security vulnerabilities
            security = metrics.get("security", {})
            critical_vulns = security.get("vulnerabilities", {}).get("critical", 0)
            high_vulns = security.get("vulnerabilities", {}).get("high", 0)
            
            if critical_vulns > self.config["thresholds"]["security_vulnerabilities"]["critical_max"]:
                issues.append({
                    "type": "critical",
                    "category": "security",
                    "message": f"Found {critical_vulns} critical security vulnerabilities",
                    "impact": "critical",
                    "auto_fixable": True
                })
            
            if high_vulns > self.config["thresholds"]["security_vulnerabilities"]["high_max"]:
                issues.append({
                    "type": "warning",
                    "category": "security",
                    "message": f"Found {high_vulns} high security vulnerabilities",
                    "impact": "high",
                    "auto_fixable": True
                })
            
            # Check build success rate
            build_success = metrics.get("ci_cd", {}).get("workflow_runs", {}).get("success_rate", 100)
            if build_success < self.config["thresholds"]["build_success_rate"]["critical"]:
                issues.append({
                    "type": "critical",
                    "category": "reliability",
                    "message": f"Build success rate ({build_success}%) is below critical threshold",
                    "impact": "high",
                    "auto_fixable": False
                })
            
            # Check mobile compatibility
            mobile_exports = metrics.get("mobile", {}).get("export_compatibility", {})
            failed_exports = []
            for platform, status in mobile_exports.items():
                if not status.get("success", False):
                    failed_exports.append(platform)
            
            if failed_exports:
                issues.append({
                    "type": "warning",
                    "category": "mobile_compatibility",
                    "message": f"Mobile export failures on platforms: {', '.join(failed_exports)}",
                    "impact": "medium",
                    "auto_fixable": True
                })
            
            # Check dependency health
            deps = metrics.get("dependencies", {}).get("dependency_health", {})
            outdated_percentage = deps.get("outdated_percentage", 0)
            if outdated_percentage > 20:  # More than 20% outdated
                issues.append({
                    "type": "warning",
                    "category": "dependencies",
                    "message": f"{outdated_percentage:.1f}% of dependencies are outdated",
                    "impact": "medium",
                    "auto_fixable": True
                })
                recommendations.append("Update dependencies to latest compatible versions")
            
            # Determine overall status
            critical_issues = [issue for issue in issues if issue["type"] == "critical"]
            if critical_issues:
                health_report["overall_status"] = "critical"
            elif issues:
                health_report["overall_status"] = "warning"
            
            health_report["issues"] = issues
            health_report["recommendations"] = recommendations
            
            logger.info(f"Health check completed. Status: {health_report['overall_status']}, Issues: {len(issues)}")
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            health_report["overall_status"] = "error"
            health_report["error"] = str(e)
        
        return health_report
    
    def perform_automated_fixes(self, health_report: Dict[str, Any]) -> Dict[str, Any]:
        """Perform automated fixes for identified issues."""
        if not self.config["automation"]["auto_fix_enabled"]:
            logger.info("Automated fixes disabled")
            return {"fixes_applied": [], "status": "disabled"}
        
        logger.info("Performing automated fixes...")
        
        fixes_applied = []
        
        for issue in health_report.get("issues", []):
            if not issue.get("auto_fixable", False):
                continue
            
            try:
                if issue["category"] == "security":
                    fix_result = self._fix_security_vulnerabilities()
                    if fix_result:
                        fixes_applied.append({
                            "issue": issue,
                            "fix": "Updated vulnerable dependencies",
                            "status": "success"
                        })
                
                elif issue["category"] == "dependencies":
                    fix_result = self._update_dependencies()
                    if fix_result:
                        fixes_applied.append({
                            "issue": issue,
                            "fix": "Updated outdated dependencies",
                            "status": "success"
                        })
                
                elif issue["category"] == "mobile_compatibility":
                    fix_result = self._fix_mobile_exports()
                    if fix_result:
                        fixes_applied.append({
                            "issue": issue,
                            "fix": "Regenerated mobile exports",
                            "status": "success"
                        })
                
            except Exception as e:
                logger.error(f"Error applying fix for {issue['category']}: {e}")
                fixes_applied.append({
                    "issue": issue,
                    "fix": f"Failed to apply fix: {e}",
                    "status": "failed"
                })
        
        return {
            "fixes_applied": fixes_applied,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    def _fix_security_vulnerabilities(self) -> bool:
        """Fix security vulnerabilities by updating dependencies."""
        try:
            logger.info("Fixing security vulnerabilities...")
            
            # Run safety check to get specific vulnerabilities
            import subprocess
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                vulnerabilities = json.loads(result.stdout) if result.stdout else []
                
                # Extract package names that need updating
                packages_to_update = set()
                for vuln in vulnerabilities:
                    package_name = vuln.get("package_name", "")
                    if package_name:
                        packages_to_update.add(package_name)
                
                # Update vulnerable packages
                for package in packages_to_update:
                    update_result = subprocess.run(
                        ["pip", "install", "--upgrade", package],
                        capture_output=True,
                        text=True,
                        cwd=self.project_root
                    )
                    
                    if update_result.returncode == 0:
                        logger.info(f"Updated vulnerable package: {package}")
                    else:
                        logger.warning(f"Failed to update package {package}: {update_result.stderr}")
                
                # Update requirements.txt if any packages were updated
                if packages_to_update:
                    subprocess.run(
                        ["pip", "freeze", ">", "requirements.txt"],
                        shell=True,
                        cwd=self.project_root
                    )
                
                return len(packages_to_update) > 0
            
            return True  # No vulnerabilities found
            
        except Exception as e:
            logger.error(f"Error fixing security vulnerabilities: {e}")
            return False
    
    def _update_dependencies(self) -> bool:
        """Update outdated dependencies."""
        try:
            logger.info("Updating outdated dependencies...")
            
            import subprocess
            
            # Get list of outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0 and result.stdout:
                outdated_packages = json.loads(result.stdout)
                
                updated_count = 0
                for package in outdated_packages:
                    package_name = package.get("name", "")
                    
                    # Skip critical packages that might break compatibility
                    skip_packages = {"torch", "tensorflow", "transformers"}
                    if package_name.lower() in skip_packages:
                        logger.info(f"Skipping critical package: {package_name}")
                        continue
                    
                    # Update package
                    update_result = subprocess.run(
                        ["pip", "install", "--upgrade", package_name],
                        capture_output=True,
                        text=True,
                        cwd=self.project_root
                    )
                    
                    if update_result.returncode == 0:
                        logger.info(f"Updated package: {package_name}")
                        updated_count += 1
                    else:
                        logger.warning(f"Failed to update {package_name}: {update_result.stderr}")
                
                return updated_count > 0
            
            return True  # No outdated packages
            
        except Exception as e:
            logger.error(f"Error updating dependencies: {e}")
            return False
    
    def _fix_mobile_exports(self) -> bool:
        """Fix mobile export issues by regenerating exports."""
        try:
            logger.info("Fixing mobile export issues...")
            
            import subprocess
            
            # Regenerate mobile exports
            platforms = ["android", "ios", "onnx"]
            success_count = 0
            
            for platform in platforms:
                result = subprocess.run(
                    ["python", "scripts/export_models.py", "--platform", platform, "--force"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                if result.returncode == 0:
                    logger.info(f"Successfully regenerated {platform} export")
                    success_count += 1
                else:
                    logger.warning(f"Failed to regenerate {platform} export: {result.stderr}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error fixing mobile exports: {e}")
            return False
    
    def send_notifications(self, health_report: Dict[str, Any], fixes_applied: Optional[Dict[str, Any]] = None) -> None:
        """Send notifications about repository health status."""
        logger.info("Sending health notifications...")
        
        status = health_report.get("overall_status", "unknown")
        issues = health_report.get("issues", [])
        
        # Only send notifications for critical or warning status
        if status not in ["critical", "warning"]:
            logger.info("Repository is healthy, no notifications needed")
            return
        
        # Prepare notification content
        subject = f"🚨 Repository Health Alert: {status.upper()}"
        if status == "warning":
            subject = f"⚠️ Repository Health Warning"
        
        message = self._create_notification_message(health_report, fixes_applied)
        
        # Send to different channels
        notification_results = {}
        
        # Slack notification
        if self.config["notifications"]["slack"]["webhook_url"]:
            notification_results["slack"] = self._send_slack_notification(subject, message)
        
        # Email notification
        if self.config["notifications"]["email"]["recipients"]:
            notification_results["email"] = self._send_email_notification(subject, message)
        
        # GitHub issue/comment
        if self.config["notifications"]["github"]["token"] and status == "critical":
            notification_results["github"] = self._create_github_issue(subject, message)
        
        logger.info(f"Notifications sent: {notification_results}")
    
    def _create_notification_message(self, health_report: Dict[str, Any], fixes_applied: Optional[Dict[str, Any]] = None) -> str:
        """Create notification message content."""
        lines = [
            f"# Repository Health Report",
            f"**Status:** {health_report.get('overall_status', 'unknown').upper()}",
            f"**Timestamp:** {health_report.get('timestamp', 'unknown')}",
            "",
        ]
        
        # Health score
        metrics = health_report.get("metrics", {})
        health_score = metrics.get("health_score", {})
        if health_score:
            lines.extend([
                f"**Overall Health Score:** {health_score.get('overall', 'N/A')}/100 ({health_score.get('grade', 'N/A')})",
                ""
            ])
        
        # Issues
        issues = health_report.get("issues", [])
        if issues:
            lines.append("## Issues Found")
            for issue in issues:
                icon = "🔴" if issue["type"] == "critical" else "🟡"
                lines.append(f"{icon} **{issue['category'].replace('_', ' ').title()}:** {issue['message']}")
            lines.append("")
        
        # Fixes applied
        if fixes_applied and fixes_applied.get("fixes_applied"):
            lines.append("## Automated Fixes Applied")
            for fix in fixes_applied["fixes_applied"]:
                icon = "✅" if fix["status"] == "success" else "❌"
                lines.append(f"{icon} {fix['fix']}")
            lines.append("")
        
        # Recommendations
        recommendations = health_report.get("recommendations", [])
        if recommendations:
            lines.append("## Recommendations")
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Key metrics
        lines.append("## Key Metrics")
        
        # Test coverage
        coverage = metrics.get("code_quality", {}).get("test_coverage", {}).get("percentage")
        if coverage is not None:
            lines.append(f"- Test Coverage: {coverage}%")
        
        # Security
        security = metrics.get("security", {}).get("vulnerabilities", {})
        if security:
            lines.append(f"- Security: {security.get('critical', 0)} critical, {security.get('high', 0)} high vulnerabilities")
        
        # Build success rate
        build_success = metrics.get("ci_cd", {}).get("workflow_runs", {}).get("success_rate")
        if build_success is not None:
            lines.append(f"- Build Success Rate: {build_success}%")
        
        # Mobile compatibility
        mobile_exports = metrics.get("mobile", {}).get("export_compatibility", {})
        if mobile_exports:
            working_platforms = [p for p, s in mobile_exports.items() if s.get("success")]
            lines.append(f"- Mobile Exports Working: {len(working_platforms)}/{len(mobile_exports)} platforms")
        
        return "\n".join(lines)
    
    def _send_slack_notification(self, subject: str, message: str) -> bool:
        """Send notification to Slack."""
        try:
            webhook_url = self.config["notifications"]["slack"]["webhook_url"]
            if not webhook_url:
                return False
            
            payload = {
                "text": subject,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": message
                        }
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info("Slack notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    def _send_email_notification(self, subject: str, message: str) -> bool:
        """Send email notification."""
        try:
            email_config = self.config["notifications"]["email"]
            
            if not all([email_config["username"], email_config["password"], email_config["recipients"]]):
                logger.warning("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config["username"]
            msg['To'] = ", ".join(email_config["recipients"])
            msg['Subject'] = subject
            
            # Convert markdown to HTML (simple conversion)
            html_message = message.replace("\n", "<br>").replace("**", "<b>").replace("**", "</b>")
            msg.attach(MIMEText(html_message, 'html'))
            
            # Send email
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
            
            logger.info("Email notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def _create_github_issue(self, subject: str, message: str) -> bool:
        """Create GitHub issue for critical alerts."""
        try:
            github_config = self.config["notifications"]["github"]
            token = github_config["token"]
            repo = github_config["repo"]
            
            if not token or not repo:
                return False
            
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Check if similar issue already exists
            existing_issues = requests.get(
                f"https://api.github.com/repos/{repo}/issues",
                headers=headers,
                params={"labels": "health-alert", "state": "open"}
            )
            
            if existing_issues.status_code == 200:
                open_health_issues = existing_issues.json()
                if open_health_issues:
                    logger.info("Health alert issue already exists, skipping creation")
                    return True
            
            # Create new issue
            issue_data = {
                "title": subject,
                "body": message,
                "labels": ["health-alert", "automated", "urgent"]
            }
            
            response = requests.post(
                f"https://api.github.com/repos/{repo}/issues",
                headers=headers,
                json=issue_data
            )
            
            if response.status_code == 201:
                logger.info("GitHub issue created successfully")
                return True
            else:
                logger.error(f"Failed to create GitHub issue: {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"Error creating GitHub issue: {e}")
            return False
    
    def generate_health_dashboard(self, health_report: Dict[str, Any]) -> str:
        """Generate HTML health dashboard."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Repository Health Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status-healthy { color: green; }
                .status-warning { color: orange; }
                .status-critical { color: red; }
                .metric { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .issue { margin: 5px 0; padding: 8px; background-color: #f8f9fa; border-radius: 3px; }
                .critical { border-left: 4px solid red; }
                .warning { border-left: 4px solid orange; }
            </style>
        </head>
        <body>
            <h1>Repository Health Dashboard</h1>
            <h2 class="status-{status}">{status_text}</h2>
            <p><strong>Last Updated:</strong> {timestamp}</p>
            
            <h3>Health Score</h3>
            <div class="metric">
                <strong>Overall Score:</strong> {health_score}/100 ({grade})
            </div>
            
            <h3>Issues</h3>
            {issues_html}
            
            <h3>Key Metrics</h3>
            {metrics_html}
        </body>
        </html>
        """
        
        # Prepare template variables
        status = health_report.get("overall_status", "unknown")
        status_text = status.upper()
        timestamp = health_report.get("timestamp", "unknown")
        
        metrics = health_report.get("metrics", {})
        health_score = metrics.get("health_score", {})
        
        # Issues HTML
        issues_html = ""
        for issue in health_report.get("issues", []):
            css_class = issue["type"]
            issues_html += f'<div class="issue {css_class}"><strong>{issue["category"]}:</strong> {issue["message"]}</div>'
        
        if not issues_html:
            issues_html = '<div class="issue">No issues found</div>'
        
        # Metrics HTML
        metrics_html = ""
        
        # Test coverage
        coverage = metrics.get("code_quality", {}).get("test_coverage", {}).get("percentage")
        if coverage is not None:
            metrics_html += f'<div class="metric"><strong>Test Coverage:</strong> {coverage}%</div>'
        
        # Security
        security = metrics.get("security", {}).get("vulnerabilities", {})
        if security:
            metrics_html += f'<div class="metric"><strong>Security:</strong> {security.get("critical", 0)} critical, {security.get("high", 0)} high vulnerabilities</div>'
        
        # Performance
        performance = metrics.get("performance", {}).get("inference_performance", {})
        if performance:
            metrics_html += f'<div class="metric"><strong>Inference Time:</strong> {performance.get("average_ms", "N/A")} ms average</div>'
        
        return html_template.format(
            status=status,
            status_text=status_text,
            timestamp=timestamp,
            health_score=health_score.get("overall", "N/A"),
            grade=health_score.get("grade", "N/A"),
            issues_html=issues_html,
            metrics_html=metrics_html
        )
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run complete monitoring cycle."""
        logger.info("Starting monitoring cycle...")
        
        cycle_results = {
            "start_time": datetime.utcnow().isoformat() + "Z",
            "end_time": None,
            "health_report": None,
            "fixes_applied": None,
            "notifications_sent": False,
            "status": "running"
        }
        
        try:
            # Run health check
            health_report = self.run_health_check()
            cycle_results["health_report"] = health_report
            
            # Apply automated fixes if enabled and needed
            if health_report.get("issues"):
                fixes_applied = self.perform_automated_fixes(health_report)
                cycle_results["fixes_applied"] = fixes_applied
                
                # Re-run health check if fixes were applied
                if fixes_applied.get("fixes_applied"):
                    logger.info("Re-running health check after applying fixes...")
                    health_report = self.run_health_check()
                    cycle_results["health_report"] = health_report
            
            # Send notifications if needed
            status = health_report.get("overall_status", "unknown")
            if status in ["critical", "warning"]:
                self.send_notifications(health_report, cycle_results.get("fixes_applied"))
                cycle_results["notifications_sent"] = True
            
            cycle_results["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Error during monitoring cycle: {e}")
            cycle_results["status"] = "error"
            cycle_results["error"] = str(e)
        
        cycle_results["end_time"] = datetime.utcnow().isoformat() + "Z"
        
        # Save monitoring results
        monitoring_file = self.project_root / ".github" / "monitoring-results.json"
        try:
            with open(monitoring_file, 'w') as f:
                json.dump(cycle_results, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save monitoring results: {e}")
        
        logger.info(f"Monitoring cycle completed with status: {cycle_results['status']}")
        return cycle_results


def main():
    """Main entry point for repository health monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository health monitoring")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--dashboard", help="Generate HTML dashboard to file")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (continuous monitoring)")
    parser.add_argument("--interval", type=int, default=3600, help="Monitoring interval in seconds (for daemon mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize monitor
    monitor = RepositoryHealthMonitor(args.config)
    
    if args.daemon:
        logger.info(f"Starting daemon mode with {args.interval}s interval...")
        
        while True:
            try:
                cycle_results = monitor.run_monitoring_cycle()
                
                # Generate dashboard if requested
                if args.dashboard and cycle_results.get("health_report"):
                    dashboard_html = monitor.generate_health_dashboard(cycle_results["health_report"])
                    with open(args.dashboard, 'w') as f:
                        f.write(dashboard_html)
                    logger.info(f"Dashboard generated: {args.dashboard}")
                
                # Wait for next cycle
                logger.info(f"Sleeping for {args.interval} seconds...")
                time.sleep(args.interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in daemon loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    else:
        # Single run
        cycle_results = monitor.run_monitoring_cycle()
        
        # Generate dashboard if requested
        if args.dashboard and cycle_results.get("health_report"):
            dashboard_html = monitor.generate_health_dashboard(cycle_results["health_report"])
            with open(args.dashboard, 'w') as f:
                f.write(dashboard_html)
            logger.info(f"Dashboard generated: {args.dashboard}")
        
        # Print summary
        health_report = cycle_results.get("health_report", {})
        status = health_report.get("overall_status", "unknown")
        issues_count = len(health_report.get("issues", []))
        
        print(f"\nRepository Health Status: {status.upper()}")
        print(f"Issues Found: {issues_count}")
        
        if issues_count > 0:
            print("\nIssues:")
            for issue in health_report.get("issues", []):
                print(f"  - {issue['type'].upper()}: {issue['message']}")
        
        # Exit with appropriate code
        if status == "critical":
            exit(2)
        elif status == "warning":
            exit(1)
        else:
            exit(0)


if __name__ == "__main__":
    main()