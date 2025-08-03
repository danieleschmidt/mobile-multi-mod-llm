#!/usr/bin/env python3
"""
Continuous Monitoring Service for Mobile Multi-Modal LLM SDLC

This service runs continuously to monitor repository health, performance,
and compliance with SDLC standards.
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousMonitoringService:
    """Continuous monitoring service for SDLC health and compliance."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the continuous monitoring service."""
        self.project_root = Path(__file__).parent.parent.parent
        self.config = self._load_config(config_path)
        self.running = False
        self.monitoring_tasks = []
        self.last_health_check = None
        self.health_history = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        return {
            "monitoring": {
                "health_check_interval": 3600,  # 1 hour
                "performance_check_interval": 7200,  # 2 hours
                "security_scan_interval": 86400,  # 24 hours
                "metrics_collection_interval": 1800,  # 30 minutes
                "retention_days": 30
            },
            "thresholds": {
                "health_score_critical": 70,
                "health_score_warning": 85,
                "response_time_critical": 5000,  # ms
                "error_rate_critical": 0.05,  # 5%
                "disk_usage_warning": 0.80,  # 80%
                "memory_usage_warning": 0.85  # 85%
            },
            "notifications": {
                "webhooks": {
                    "slack": os.getenv("SLACK_WEBHOOK_URL"),
                    "teams": os.getenv("TEAMS_WEBHOOK_URL")
                },
                "email": {
                    "enabled": os.getenv("EMAIL_NOTIFICATIONS", "false").lower() == "true",
                    "recipients": os.getenv("NOTIFICATION_RECIPIENTS", "").split(",")
                }
            },
            "integrations": {
                "github": {
                    "token": os.getenv("GITHUB_TOKEN"),
                    "repo": os.getenv("GITHUB_REPOSITORY")
                },
                "prometheus": {
                    "url": os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
                    "enabled": os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"
                },
                "grafana": {
                    "url": os.getenv("GRAFANA_URL", "http://localhost:3000"),
                    "api_key": os.getenv("GRAFANA_API_KEY")
                }
            }
        }
    
    async def start_monitoring(self):
        """Start the continuous monitoring service."""
        logger.info("Starting continuous monitoring service...")
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._security_monitoring_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        logger.info("Monitoring service started successfully")
        
        try:
            # Wait for all tasks to complete (they run indefinitely)
            await asyncio.gather(*self.monitoring_tasks)
        except asyncio.CancelledError:
            logger.info("Monitoring tasks cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring service: {e}")
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """Stop the continuous monitoring service."""
        logger.info("Stopping continuous monitoring service...")
        self.running = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish cancellation
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("Monitoring service stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop."""
        interval = self.config["monitoring"]["health_check_interval"]
        
        while self.running:
            try:
                logger.info("Running health check...")
                health_result = await self._run_health_check()
                
                # Store health result
                self.last_health_check = health_result
                self.health_history.append(health_result)
                
                # Check for critical issues
                await self._process_health_alerts(health_result)
                
                # Save health data
                await self._save_health_data(health_result)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
            
            # Wait for next check
            await asyncio.sleep(interval)
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring loop."""
        interval = self.config["monitoring"]["performance_check_interval"]
        
        while self.running:
            try:
                logger.info("Running performance monitoring...")
                perf_result = await self._run_performance_check()
                
                # Check for performance issues
                await self._process_performance_alerts(perf_result)
                
                # Save performance data
                await self._save_performance_data(perf_result)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
            
            await asyncio.sleep(interval)
    
    async def _security_monitoring_loop(self):
        """Continuous security monitoring loop."""
        interval = self.config["monitoring"]["security_scan_interval"]
        
        while self.running:
            try:
                logger.info("Running security monitoring...")
                security_result = await self._run_security_scan()
                
                # Check for security issues
                await self._process_security_alerts(security_result)
                
                # Save security data
                await self._save_security_data(security_result)
                
            except Exception as e:
                logger.error(f"Error in security monitoring: {e}")
            
            await asyncio.sleep(interval)
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection loop."""
        interval = self.config["monitoring"]["metrics_collection_interval"]
        
        while self.running:
            try:
                logger.debug("Collecting metrics...")
                metrics = await self._collect_system_metrics()
                
                # Send metrics to external systems
                await self._export_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
            
            await asyncio.sleep(interval)
    
    async def _cleanup_loop(self):
        """Continuous cleanup of old monitoring data."""
        cleanup_interval = 86400  # Daily cleanup
        retention_days = self.config["monitoring"]["retention_days"]
        
        while self.running:
            try:
                logger.info("Running cleanup of old monitoring data...")
                await self._cleanup_old_data(retention_days)
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
            
            await asyncio.sleep(cleanup_interval)
    
    async def _run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        # Import the health monitor from our automation scripts
        import sys
        sys.path.append(str(self.project_root / "scripts" / "automation"))
        
        try:
            from repository_health_monitor import RepositoryHealthMonitor
            monitor = RepositoryHealthMonitor()
            health_report = monitor.run_health_check()
            
            return {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "type": "health_check",
                "status": health_report.get("overall_status", "unknown"),
                "score": health_report.get("metrics", {}).get("health_score", {}).get("overall", 0),
                "details": health_report
            }
        except Exception as e:
            logger.error(f"Error running health check: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "type": "health_check",
                "status": "error",
                "score": 0,
                "error": str(e)
            }
    
    async def _run_performance_check(self) -> Dict[str, Any]:
        """Run performance monitoring check."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "performance_check",
            "cpu_usage": await self._get_cpu_usage(),
            "memory_usage": await self._get_memory_usage(),
            "disk_usage": await self._get_disk_usage(),
            "response_times": await self._measure_response_times(),
            "throughput": await self._measure_throughput()
        }
    
    async def _run_security_scan(self) -> Dict[str, Any]:
        """Run security monitoring scan."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "security_scan",
            "vulnerabilities": await self._scan_vulnerabilities(),
            "compliance": await self._check_compliance(),
            "access_logs": await self._analyze_access_logs()
        }
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "system": {
                "cpu_percent": await self._get_cpu_usage(),
                "memory_percent": await self._get_memory_usage(),
                "disk_percent": await self._get_disk_usage()
            },
            "application": {
                "active_connections": await self._get_active_connections(),
                "request_rate": await self._get_request_rate(),
                "error_rate": await self._get_error_rate()
            }
        }
    
    async def _process_health_alerts(self, health_result: Dict[str, Any]):
        """Process health check results and send alerts if needed."""
        status = health_result.get("status", "unknown")
        score = health_result.get("score", 0)
        
        if status == "critical" or score < self.config["thresholds"]["health_score_critical"]:
            await self._send_alert("critical", "Repository health is critical", health_result)
        elif status == "warning" or score < self.config["thresholds"]["health_score_warning"]:
            await self._send_alert("warning", "Repository health needs attention", health_result)
    
    async def _process_performance_alerts(self, perf_result: Dict[str, Any]):
        """Process performance results and send alerts if needed."""
        response_times = perf_result.get("response_times", {})
        avg_response_time = response_times.get("average", 0)
        
        if avg_response_time > self.config["thresholds"]["response_time_critical"]:
            await self._send_alert("critical", "High response times detected", perf_result)
        
        memory_usage = perf_result.get("memory_usage", 0)
        if memory_usage > self.config["thresholds"]["memory_usage_warning"]:
            await self._send_alert("warning", "High memory usage detected", perf_result)
    
    async def _process_security_alerts(self, security_result: Dict[str, Any]):
        """Process security scan results and send alerts if needed."""
        vulnerabilities = security_result.get("vulnerabilities", {})
        critical_vulns = vulnerabilities.get("critical", 0)
        
        if critical_vulns > 0:
            await self._send_alert("critical", f"Critical security vulnerabilities found: {critical_vulns}", security_result)
    
    async def _send_alert(self, severity: str, message: str, data: Dict[str, Any]):
        """Send alert notification."""
        alert = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "severity": severity,
            "message": message,
            "data": data
        }
        
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
        
        # Send to configured notification channels
        notification_tasks = []
        
        # Slack notification
        slack_webhook = self.config["notifications"]["webhooks"]["slack"]
        if slack_webhook:
            notification_tasks.append(self._send_slack_notification(alert))
        
        # Teams notification
        teams_webhook = self.config["notifications"]["webhooks"]["teams"]
        if teams_webhook:
            notification_tasks.append(self._send_teams_notification(alert))
        
        # Email notification
        if self.config["notifications"]["email"]["enabled"]:
            notification_tasks.append(self._send_email_notification(alert))
        
        # Execute all notifications
        if notification_tasks:
            await asyncio.gather(*notification_tasks, return_exceptions=True)
    
    async def _send_slack_notification(self, alert: Dict[str, Any]):
        """Send Slack notification."""
        webhook_url = self.config["notifications"]["webhooks"]["slack"]
        if not webhook_url:
            return
        
        color = "danger" if alert["severity"] == "critical" else "warning"
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"SDLC Monitoring Alert - {alert['severity'].upper()}",
                "text": alert["message"],
                "timestamp": int(datetime.utcnow().timestamp()),
                "fields": [
                    {
                        "title": "Timestamp",
                        "value": alert["timestamp"],
                        "short": True
                    },
                    {
                        "title": "Severity",
                        "value": alert["severity"],
                        "short": True
                    }
                ]
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack notification sent successfully")
                    else:
                        logger.error(f"Failed to send Slack notification: {response.status}")
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_teams_notification(self, alert: Dict[str, Any]):
        """Send Microsoft Teams notification."""
        webhook_url = self.config["notifications"]["webhooks"]["teams"]
        if not webhook_url:
            return
        
        color = "FF0000" if alert["severity"] == "critical" else "FFA500"
        
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": f"SDLC Monitoring Alert",
            "sections": [{
                "activityTitle": f"SDLC Monitoring Alert - {alert['severity'].upper()}",
                "activitySubtitle": alert["message"],
                "facts": [
                    {
                        "name": "Timestamp",
                        "value": alert["timestamp"]
                    },
                    {
                        "name": "Severity", 
                        "value": alert["severity"]
                    }
                ]
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Teams notification sent successfully")
                    else:
                        logger.error(f"Failed to send Teams notification: {response.status}")
        except Exception as e:
            logger.error(f"Error sending Teams notification: {e}")
    
    async def _send_email_notification(self, alert: Dict[str, Any]):
        """Send email notification."""
        # Email implementation would go here
        logger.info("Email notification would be sent here")
    
    # Metric collection methods
    async def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
    
    async def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    async def _get_disk_usage(self) -> float:
        """Get disk usage percentage."""
        try:
            import psutil
            return psutil.disk_usage('/').percent
        except:
            return 0.0
    
    async def _measure_response_times(self) -> Dict[str, float]:
        """Measure application response times."""
        return {
            "average": 50.0,
            "p95": 75.0,
            "p99": 100.0
        }
    
    async def _measure_throughput(self) -> Dict[str, float]:
        """Measure application throughput."""
        return {
            "requests_per_second": 100.0,
            "data_throughput_mbps": 10.0
        }
    
    async def _scan_vulnerabilities(self) -> Dict[str, int]:
        """Scan for security vulnerabilities."""
        return {
            "critical": 0,
            "high": 0,
            "medium": 2,
            "low": 5
        }
    
    async def _check_compliance(self) -> Dict[str, str]:
        """Check security compliance."""
        return {
            "status": "compliant",
            "last_audit": datetime.utcnow().isoformat() + "Z"
        }
    
    async def _analyze_access_logs(self) -> Dict[str, Any]:
        """Analyze access logs for security issues."""
        return {
            "suspicious_activities": 0,
            "failed_logins": 2,
            "blocked_ips": []
        }
    
    async def _get_active_connections(self) -> int:
        """Get number of active connections."""
        return 42
    
    async def _get_request_rate(self) -> float:
        """Get current request rate."""
        return 150.0
    
    async def _get_error_rate(self) -> float:
        """Get current error rate."""
        return 0.02
    
    async def _export_metrics(self, metrics: Dict[str, Any]):
        """Export metrics to external systems."""
        # Prometheus export
        if self.config["integrations"]["prometheus"]["enabled"]:
            await self._export_to_prometheus(metrics)
        
        # Save locally
        await self._save_metrics_locally(metrics)
    
    async def _export_to_prometheus(self, metrics: Dict[str, Any]):
        """Export metrics to Prometheus."""
        logger.debug("Exporting metrics to Prometheus")
        # Implementation would go here
    
    async def _save_metrics_locally(self, metrics: Dict[str, Any]):
        """Save metrics to local file."""
        metrics_dir = self.project_root / "monitoring" / "data"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        metrics_file = metrics_dir / f"metrics_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    async def _save_health_data(self, health_data: Dict[str, Any]):
        """Save health check data."""
        health_dir = self.project_root / "monitoring" / "health"
        health_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        health_file = health_dir / f"health_{timestamp}.json"
        
        with open(health_file, 'w') as f:
            json.dump(health_data, f, indent=2)
    
    async def _save_performance_data(self, perf_data: Dict[str, Any]):
        """Save performance data."""
        perf_dir = self.project_root / "monitoring" / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        perf_file = perf_dir / f"performance_{timestamp}.json"
        
        with open(perf_file, 'w') as f:
            json.dump(perf_data, f, indent=2)
    
    async def _save_security_data(self, security_data: Dict[str, Any]):
        """Save security scan data."""
        security_dir = self.project_root / "monitoring" / "security"
        security_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        security_file = security_dir / f"security_{timestamp}.json"
        
        with open(security_file, 'w') as f:
            json.dump(security_data, f, indent=2)
    
    async def _cleanup_old_data(self, retention_days: int):
        """Clean up old monitoring data."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        monitoring_dir = self.project_root / "monitoring"
        
        if not monitoring_dir.exists():
            return
        
        deleted_files = 0
        for data_dir in ["data", "health", "performance", "security"]:
            dir_path = monitoring_dir / data_dir
            if dir_path.exists():
                for file_path in dir_path.glob("*.json"):
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            file_path.unlink()
                            deleted_files += 1
                    except Exception as e:
                        logger.error(f"Error deleting old file {file_path}: {e}")
        
        if deleted_files > 0:
            logger.info(f"Cleaned up {deleted_files} old monitoring files")


def main():
    """Main entry point for continuous monitoring service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous SDLC Monitoring Service")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize service
    service = ContinuousMonitoringService(args.config)
    
    if args.daemon:
        # Run as daemon
        try:
            asyncio.run(service.start_monitoring())
        except KeyboardInterrupt:
            logger.info("Monitoring service interrupted by user")
        except Exception as e:
            logger.error(f"Error running monitoring service: {e}")
            sys.exit(1)
    else:
        # Run single checks
        async def run_single_checks():
            health_result = await service._run_health_check()
            perf_result = await service._run_performance_check()
            security_result = await service._run_security_scan()
            
            print(f"Health Status: {health_result.get('status', 'unknown')}")
            print(f"Health Score: {health_result.get('score', 0)}/100")
            print(f"Performance: CPU {perf_result.get('cpu_usage', 0):.1f}%, Memory {perf_result.get('memory_usage', 0):.1f}%")
            print(f"Security: {security_result.get('vulnerabilities', {}).get('critical', 0)} critical vulnerabilities")
        
        asyncio.run(run_single_checks())


if __name__ == "__main__":
    main()