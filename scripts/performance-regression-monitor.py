#!/usr/bin/env python3
"""Performance regression monitoring and alerting system.

This script runs continuous performance monitoring and alerts on regressions.
Designed to integrate with CI/CD pipelines and monitoring systems.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Continuous performance monitoring system."""
    
    def __init__(self, config_path: str = "performance-monitor.json"):
        self.config = self._load_config(config_path)
        self.baseline_path = Path("tests/performance/baselines.json")
        self.results_path = Path("tests/performance/results")
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration."""
        default_config = {
            "regression_threshold": 0.15,
            "monitoring_interval_hours": 6,
            "alert_webhook": None,
            "slack_webhook": None,
            "email_alerts": [],
            "performance_targets": {
                "image_captioning": {"latency_ms": 15.0, "throughput_fps": 60.0},
                "ocr_extraction": {"latency_ms": 20.0, "throughput_fps": 50.0},
                "vqa_inference": {"latency_ms": 18.0, "throughput_fps": 55.0}
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            logger.info(f"Config file {config_path} not found, using defaults")
            return default_config
    
    def run_performance_tests(self) -> Dict:
        """Run performance regression tests and return results."""
        logger.info("Running performance regression tests...")
        
        try:
            # Run pytest with performance tests
            result = subprocess.run([
                "python", "-m", "pytest", 
                "tests/performance/test_regression.py",
                "-v", "--tb=short", "--json-report",
                f"--json-report-file={self.results_path}/latest_results.json"
            ], capture_output=True, text=True, cwd=".")
            
            # Load test results
            results_file = self.results_path / "latest_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    return json.load(f)
            else:
                logger.error("Test results file not found")
                return {"tests": [], "summary": {"failed": 1}}
                
        except Exception as e:
            logger.error(f"Failed to run performance tests: {e}")
            return {"tests": [], "summary": {"failed": 1}}
    
    def check_performance_targets(self, results: Dict) -> List[str]:
        """Check if performance meets targets."""
        violations = []
        
        # Load current baselines
        if not self.baseline_path.exists():
            logger.warning("No performance baselines found")
            return violations
        
        with open(self.baseline_path, 'r') as f:
            baselines = json.load(f)
        
        for test_name, targets in self.config["performance_targets"].items():
            if test_name in baselines:
                baseline = baselines[test_name]
                
                # Check latency target
                if baseline["latency_ms"] > targets["latency_ms"]:
                    violations.append(
                        f"{test_name}: Latency {baseline['latency_ms']:.1f}ms "
                        f"exceeds target {targets['latency_ms']:.1f}ms"
                    )
                
                # Check throughput target
                if baseline["throughput_fps"] < targets["throughput_fps"]:
                    violations.append(
                        f"{test_name}: Throughput {baseline['throughput_fps']:.1f}fps "
                        f"below target {targets['throughput_fps']:.1f}fps"
                    )
        
        return violations
    
    def send_alert(self, message: str, severity: str = "warning"):
        """Send performance alert via configured channels."""
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "message": message,
            "project": "mobile-multimodal-llm",
            "component": "performance-monitoring"
        }
        
        # Slack notification
        if self.config.get("slack_webhook"):
            self._send_slack_alert(alert_data)
        
        # Generic webhook
        if self.config.get("alert_webhook"):
            self._send_webhook_alert(alert_data)
        
        # Email alerts (would integrate with email service)
        if self.config.get("email_alerts"):
            logger.info(f"Would send email alert to: {self.config['email_alerts']}")
    
    def _send_slack_alert(self, alert_data: Dict):
        """Send Slack notification."""
        try:
            color = {"error": "danger", "warning": "warning", "info": "good"}[alert_data["severity"]]
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"Performance Alert - {alert_data['project']}",
                    "text": alert_data["message"],
                    "fields": [
                        {"title": "Severity", "value": alert_data["severity"], "short": True},
                        {"title": "Timestamp", "value": alert_data["timestamp"], "short": True}
                    ]
                }]
            }
            
            response = requests.post(self.config["slack_webhook"], json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Slack alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_webhook_alert(self, alert_data: Dict):
        """Send generic webhook alert."""
        try:
            response = requests.post(
                self.config["alert_webhook"], 
                json=alert_data, 
                timeout=10
            )
            response.raise_for_status()
            logger.info("Webhook alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def generate_performance_report(self, results: Dict) -> str:
        """Generate performance monitoring report."""
        report = ["# Performance Monitoring Report", ""]
        report.append(f"**Timestamp**: {datetime.now().isoformat()}")
        report.append(f"**Tests Run**: {len(results.get('tests', []))}")
        report.append(f"**Failed**: {results.get('summary', {}).get('failed', 0)}")
        report.append("")
        
        # Add test details
        if results.get('tests'):
            report.append("## Test Results")
            for test in results['tests']:
                status = "✅ PASS" if test.get('outcome') == 'passed' else "❌ FAIL"
                report.append(f"- {test.get('nodeid', 'Unknown')}: {status}")
                
                if test.get('outcome') == 'failed':
                    report.append(f"  - Error: {test.get('call', {}).get('longrepr', 'Unknown error')}")
        
        # Add performance targets check
        violations = self.check_performance_targets(results)
        if violations:
            report.append("\n## Performance Target Violations")
            for violation in violations:
                report.append(f"- ⚠️ {violation}")
        
        return "\n".join(report)
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle."""
        logger.info("Starting performance monitoring cycle")
        
        # Run performance tests
        results = self.run_performance_tests()
        
        # Check for failures
        failed_tests = results.get('summary', {}).get('failed', 0)
        if failed_tests > 0:
            message = f"Performance regression detected: {failed_tests} test(s) failed"
            logger.error(message)
            self.send_alert(message, severity="error")
        
        # Check performance targets
        violations = self.check_performance_targets(results)
        if violations:
            message = f"Performance targets violated: {'; '.join(violations)}"
            logger.warning(message)
            self.send_alert(message, severity="warning")
        
        # Generate and save report
        report = self.generate_performance_report(results)
        report_file = self.results_path / f"report_{int(time.time())}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Monitoring cycle completed. Report saved to {report_file}")
        return failed_tests == 0 and len(violations) == 0
    
    def run_continuous_monitoring(self):
        """Run continuous performance monitoring."""
        logger.info("Starting continuous performance monitoring")
        
        interval_seconds = self.config["monitoring_interval_hours"] * 3600
        
        while True:
            try:
                success = self.run_monitoring_cycle()
                
                if success:
                    logger.info(f"All performance checks passed. Next check in {self.config['monitoring_interval_hours']} hours")
                else:
                    logger.warning("Performance issues detected. See alerts for details")
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}")
                time.sleep(300)  # Wait 5 minutes before retry


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Performance regression monitoring")
    parser.add_argument("--config", default="performance-monitor.json", 
                       help="Configuration file path")
    parser.add_argument("--once", action="store_true", 
                       help="Run once instead of continuous monitoring")
    parser.add_argument("--save-baseline", action="store_true",
                       help="Save current results as new baseline")
    
    args = parser.parse_args()
    
    # Set environment variable for baseline saving
    if args.save_baseline:
        import os
        os.environ["SAVE_PERFORMANCE_BASELINE"] = "1"
    
    monitor = PerformanceMonitor(args.config)
    
    if args.once:
        success = monitor.run_monitoring_cycle()
        sys.exit(0 if success else 1)
    else:
        monitor.run_continuous_monitoring()


if __name__ == "__main__":
    main()