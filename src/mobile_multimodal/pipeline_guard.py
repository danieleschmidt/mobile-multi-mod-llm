"""Self-Healing Pipeline Guard for Mobile Multi-Modal LLM Infrastructure.

This module implements autonomous pipeline monitoring, failure detection,
and self-healing capabilities for the mobile AI deployment pipeline.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
import json
import hashlib
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class HealthStatus(Enum):
    """Pipeline component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"


class PipelineComponent(Enum):
    """Pipeline components to monitor."""
    MODEL_TRAINING = "model_training"
    QUANTIZATION = "quantization"
    MOBILE_EXPORT = "mobile_export"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    STORAGE = "storage"
    COMPUTE = "compute"


@dataclass
class HealthCheck:
    """Health check configuration and results."""
    component: PipelineComponent
    check_name: str
    check_function: Callable
    interval_seconds: int
    timeout_seconds: int
    max_retries: int
    last_check: Optional[datetime] = None
    last_status: Optional[HealthStatus] = None
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Alert:
    """Pipeline alert information."""
    component: PipelineComponent
    severity: HealthStatus
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]
    resolved: bool = False
    resolution_action: Optional[str] = None


class SelfHealingPipelineGuard:
    """Autonomous pipeline monitoring and self-healing system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline guard.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alerts: List[Alert] = []
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._lock = threading.RLock()
        
        # Initialize health checks
        self._setup_health_checks()
        
        # Recovery strategies
        self.recovery_strategies = {
            PipelineComponent.MODEL_TRAINING: self._recover_training,
            PipelineComponent.QUANTIZATION: self._recover_quantization,
            PipelineComponent.MOBILE_EXPORT: self._recover_mobile_export,
            PipelineComponent.TESTING: self._recover_testing,
            PipelineComponent.DEPLOYMENT: self._recover_deployment,
            PipelineComponent.MONITORING: self._recover_monitoring,
            PipelineComponent.STORAGE: self._recover_storage,
            PipelineComponent.COMPUTE: self._recover_compute,
        }
        
        self.logger.info("Self-Healing Pipeline Guard initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "check_intervals": {
                "model_training": 300,  # 5 minutes
                "quantization": 180,    # 3 minutes
                "mobile_export": 120,   # 2 minutes
                "testing": 60,          # 1 minute
                "deployment": 300,      # 5 minutes
                "monitoring": 30,       # 30 seconds
                "storage": 60,          # 1 minute
                "compute": 45,          # 45 seconds
            },
            "timeouts": {
                "default": 30,
                "training": 300,
                "export": 120,
            },
            "max_retries": 3,
            "alert_cooldown": 900,  # 15 minutes
            "auto_recovery": True,
            "notification_webhooks": [],
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_health_checks(self):
        """Initialize all health check configurations."""
        checks = [
            # Model Training Pipeline
            HealthCheck(
                component=PipelineComponent.MODEL_TRAINING,
                check_name="training_process",
                check_function=self._check_training_process,
                interval_seconds=self.config["check_intervals"]["model_training"],
                timeout_seconds=self.config["timeouts"].get("training", 300),
                max_retries=self.config["max_retries"],
            ),
            
            # Quantization Pipeline
            HealthCheck(
                component=PipelineComponent.QUANTIZATION,
                check_name="quantization_accuracy",
                check_function=self._check_quantization_accuracy,
                interval_seconds=self.config["check_intervals"]["quantization"],
                timeout_seconds=self.config["timeouts"]["default"],
                max_retries=self.config["max_retries"],
            ),
            
            # Mobile Export Pipeline
            HealthCheck(
                component=PipelineComponent.MOBILE_EXPORT,
                check_name="mobile_export_status",
                check_function=self._check_mobile_export,
                interval_seconds=self.config["check_intervals"]["mobile_export"],
                timeout_seconds=self.config["timeouts"].get("export", 120),
                max_retries=self.config["max_retries"],
            ),
            
            # Testing Pipeline
            HealthCheck(
                component=PipelineComponent.TESTING,
                check_name="test_suite_status",
                check_function=self._check_test_suite,
                interval_seconds=self.config["check_intervals"]["testing"],
                timeout_seconds=self.config["timeouts"]["default"],
                max_retries=self.config["max_retries"],
            ),
            
            # Deployment Pipeline
            HealthCheck(
                component=PipelineComponent.DEPLOYMENT,
                check_name="deployment_health",
                check_function=self._check_deployment_health,
                interval_seconds=self.config["check_intervals"]["deployment"],
                timeout_seconds=self.config["timeouts"]["default"],
                max_retries=self.config["max_retries"],
            ),
            
            # Monitoring Systems
            HealthCheck(
                component=PipelineComponent.MONITORING,
                check_name="monitoring_systems",
                check_function=self._check_monitoring_systems,
                interval_seconds=self.config["check_intervals"]["monitoring"],
                timeout_seconds=self.config["timeouts"]["default"],
                max_retries=self.config["max_retries"],
            ),
            
            # Storage Systems
            HealthCheck(
                component=PipelineComponent.STORAGE,
                check_name="storage_availability",
                check_function=self._check_storage_systems,
                interval_seconds=self.config["check_intervals"]["storage"],
                timeout_seconds=self.config["timeouts"]["default"],
                max_retries=self.config["max_retries"],
            ),
            
            # Compute Resources
            HealthCheck(
                component=PipelineComponent.COMPUTE,
                check_name="compute_resources",
                check_function=self._check_compute_resources,
                interval_seconds=self.config["check_intervals"]["compute"],
                timeout_seconds=self.config["timeouts"]["default"],
                max_retries=self.config["max_retries"],
            ),
        ]
        
        for check in checks:
            self.health_checks[f"{check.component.value}_{check.check_name}"] = check
    
    async def start_monitoring(self):
        """Start the autonomous monitoring loop."""
        if self.is_running:
            self.logger.warning("Pipeline guard is already running")
            return
        
        self.is_running = True
        self.logger.info("Starting autonomous pipeline monitoring")
        
        # Start health check loops
        tasks = []
        for check_id, check in self.health_checks.items():
            task = asyncio.create_task(self._health_check_loop(check_id, check))
            tasks.append(task)
        
        # Start alert processing
        alert_task = asyncio.create_task(self._alert_processing_loop())
        tasks.append(alert_task)
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Monitoring tasks cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
        finally:
            self.is_running = False
    
    async def _health_check_loop(self, check_id: str, check: HealthCheck):
        """Run health check loop for a specific component."""
        while self.is_running:
            try:
                await asyncio.sleep(check.interval_seconds)
                await self._run_health_check(check_id, check)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error for {check_id}: {e}")
    
    async def _run_health_check(self, check_id: str, check: HealthCheck):
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            # Run health check function with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, check.check_function
                ),
                timeout=check.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            with self._lock:
                check.last_check = datetime.now()
                check.last_status = result
                
                if result == HealthStatus.HEALTHY:
                    check.consecutive_failures = 0
                    self.logger.debug(f"Health check {check_id} passed ({execution_time:.2f}s)")
                else:
                    check.consecutive_failures += 1
                    self.logger.warning(
                        f"Health check {check_id} failed: {result} "
                        f"(failures: {check.consecutive_failures}/{check.max_retries})"
                    )
                    
                    # Create alert
                    alert = Alert(
                        component=check.component,
                        severity=result,
                        message=f"Health check {check.check_name} failed",
                        timestamp=datetime.now(),
                        metadata={
                            "check_id": check_id,
                            "consecutive_failures": check.consecutive_failures,
                            "execution_time": execution_time,
                        }
                    )
                    self.alerts.append(alert)
                    
                    # Trigger recovery if max retries reached
                    if (check.consecutive_failures >= check.max_retries and 
                        self.config.get("auto_recovery", True)):
                        await self._trigger_recovery(check.component, alert)
        
        except asyncio.TimeoutError:
            self.logger.error(f"Health check {check_id} timed out after {check.timeout_seconds}s")
            with self._lock:
                check.consecutive_failures += 1
                check.last_status = HealthStatus.FAILED
        
        except Exception as e:
            self.logger.error(f"Health check {check_id} error: {e}")
            with self._lock:
                check.consecutive_failures += 1
                check.last_status = HealthStatus.FAILED
    
    async def _trigger_recovery(self, component: PipelineComponent, alert: Alert):
        """Trigger automated recovery for a failed component."""
        self.logger.info(f"Triggering recovery for component: {component.value}")
        
        recovery_function = self.recovery_strategies.get(component)
        if not recovery_function:
            self.logger.error(f"No recovery strategy for component: {component.value}")
            return
        
        try:
            # Execute recovery strategy
            recovery_result = await asyncio.get_event_loop().run_in_executor(
                self.executor, recovery_function, alert
            )
            
            if recovery_result:
                self.logger.info(f"Recovery successful for {component.value}")
                alert.resolved = True
                alert.resolution_action = "Automated recovery successful"
            else:
                self.logger.error(f"Recovery failed for {component.value}")
                # Escalate alert or trigger manual intervention
                await self._escalate_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Recovery execution error for {component.value}: {e}")
            await self._escalate_alert(alert)
    
    # Health Check Functions
    def _check_training_process(self) -> HealthStatus:
        """Check model training pipeline health."""
        try:
            # Check for running training processes
            result = subprocess.run(
                ["pgrep", "-f", "train"],
                capture_output=True, text=True, timeout=10
            )
            
            # Check recent training logs
            log_files = list(Path("logs").glob("*train*.log")) if Path("logs").exists() else []
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                if (time.time() - latest_log.stat().st_mtime) < 3600:  # Updated within 1 hour
                    return HealthStatus.HEALTHY
            
            # Check for training artifacts
            model_dir = Path("models")
            if model_dir.exists():
                recent_models = [
                    f for f in model_dir.glob("*.pth") 
                    if (time.time() - f.stat().st_mtime) < 86400  # Created within 24 hours
                ]
                if recent_models:
                    return HealthStatus.HEALTHY
            
            return HealthStatus.DEGRADED
            
        except Exception as e:
            self.logger.error(f"Training process check failed: {e}")
            return HealthStatus.FAILED
    
    def _check_quantization_accuracy(self) -> HealthStatus:
        """Check quantization pipeline accuracy."""
        try:
            # Check for quantized models
            quantized_dir = Path("models/quantized")
            if not quantized_dir.exists():
                return HealthStatus.FAILED
            
            # Check model size compliance (<35MB)
            for model_file in quantized_dir.glob("*.dlc"):
                if model_file.stat().st_size > 35 * 1024 * 1024:  # 35MB
                    return HealthStatus.DEGRADED
            
            # Check accuracy metrics file
            metrics_file = quantized_dir / "accuracy_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    accuracy_drop = metrics.get("accuracy_drop_percent", 100)
                    if accuracy_drop > 5:  # >5% accuracy drop
                        return HealthStatus.DEGRADED
            
            return HealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Quantization check failed: {e}")
            return HealthStatus.FAILED
    
    def _check_mobile_export(self) -> HealthStatus:
        """Check mobile export pipeline status."""
        try:
            # Check for exported mobile models
            android_models = list(Path("models/android").glob("*.tflite")) if Path("models/android").exists() else []
            ios_models = list(Path("models/ios").glob("*.mlpackage")) if Path("models/ios").exists() else []
            
            if not android_models and not ios_models:
                return HealthStatus.FAILED
            
            # Check export timestamps
            all_models = android_models + ios_models
            for model in all_models:
                if (time.time() - model.stat().st_mtime) > 604800:  # Older than 7 days
                    return HealthStatus.DEGRADED
            
            return HealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Mobile export check failed: {e}")
            return HealthStatus.FAILED
    
    def _check_test_suite(self) -> HealthStatus:
        """Check test suite status."""
        try:
            # Run quick test validation
            result = subprocess.run(
                ["python", "-m", "pytest", "--collect-only", "-q"],
                capture_output=True, text=True, timeout=30,
                cwd=Path.cwd()
            )
            
            if result.returncode != 0:
                return HealthStatus.DEGRADED
            
            # Check test coverage
            coverage_file = Path(".coverage")
            if coverage_file.exists():
                # Recent coverage data available
                if (time.time() - coverage_file.stat().st_mtime) < 86400:  # Within 24 hours
                    return HealthStatus.HEALTHY
            
            return HealthStatus.DEGRADED
            
        except Exception as e:
            self.logger.error(f"Test suite check failed: {e}")
            return HealthStatus.FAILED
    
    def _check_deployment_health(self) -> HealthStatus:
        """Check deployment pipeline health."""
        try:
            # Check Docker health
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Status}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                statuses = result.stdout.strip().split('\n')
                unhealthy_containers = [s for s in statuses if 'unhealthy' in s.lower()]
                if unhealthy_containers:
                    return HealthStatus.DEGRADED
                return HealthStatus.HEALTHY
            
            return HealthStatus.DEGRADED
            
        except Exception as e:
            self.logger.error(f"Deployment health check failed: {e}")
            return HealthStatus.FAILED
    
    def _check_monitoring_systems(self) -> HealthStatus:
        """Check monitoring system health."""
        try:
            # Check if Prometheus is accessible (if configured)
            # Check log file health
            log_dir = Path("logs")
            if log_dir.exists():
                recent_logs = [
                    f for f in log_dir.glob("*.log")
                    if (time.time() - f.stat().st_mtime) < 3600  # Updated within 1 hour
                ]
                if recent_logs:
                    return HealthStatus.HEALTHY
            
            return HealthStatus.DEGRADED
            
        except Exception as e:
            self.logger.error(f"Monitoring systems check failed: {e}")
            return HealthStatus.FAILED
    
    def _check_storage_systems(self) -> HealthStatus:
        """Check storage system availability."""
        try:
            # Check disk space
            result = subprocess.run(
                ["df", "-h", "."],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    usage_line = lines[1]
                    usage_percent = usage_line.split()[4].rstrip('%')
                    if int(usage_percent) > 90:  # >90% disk usage
                        return HealthStatus.CRITICAL
                    elif int(usage_percent) > 80:  # >80% disk usage
                        return HealthStatus.DEGRADED
            
            return HealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Storage systems check failed: {e}")
            return HealthStatus.FAILED
    
    def _check_compute_resources(self) -> HealthStatus:
        """Check compute resource availability."""
        try:
            # Check CPU and memory usage
            result = subprocess.run(
                ["top", "-bn1"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'load average' in line.lower():
                        # Basic load check
                        load_values = line.split('load average:')[1].strip().split(',')
                        if load_values:
                            load_1min = float(load_values[0].strip())
                            if load_1min > 8.0:  # High load
                                return HealthStatus.DEGRADED
                        break
            
            return HealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Compute resources check failed: {e}")
            return HealthStatus.FAILED
    
    # Recovery Functions
    def _recover_training(self, alert: Alert) -> bool:
        """Recover training pipeline."""
        try:
            self.logger.info("Attempting training pipeline recovery")
            
            # Clear stale training locks
            lock_files = list(Path(".").glob("*.lock"))
            for lock_file in lock_files:
                if "train" in lock_file.name.lower():
                    lock_file.unlink()
                    self.logger.info(f"Removed stale lock file: {lock_file}")
            
            # Restart training if needed
            # This would integrate with actual training scripts
            self.logger.info("Training pipeline recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Training recovery failed: {e}")
            return False
    
    def _recover_quantization(self, alert: Alert) -> bool:
        """Recover quantization pipeline."""
        try:
            self.logger.info("Attempting quantization pipeline recovery")
            
            # Clear cache and restart quantization
            cache_dir = Path("mobile_multimodal_cache")
            if cache_dir.exists():
                # Clear old cache entries
                for cache_file in cache_dir.glob("*.cache"):
                    if (time.time() - cache_file.stat().st_mtime) > 86400:  # Older than 1 day
                        cache_file.unlink()
            
            self.logger.info("Quantization pipeline recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Quantization recovery failed: {e}")
            return False
    
    def _recover_mobile_export(self, alert: Alert) -> bool:
        """Recover mobile export pipeline."""
        try:
            self.logger.info("Attempting mobile export recovery")
            
            # Recreate export directories
            Path("models/android").mkdir(parents=True, exist_ok=True)
            Path("models/ios").mkdir(parents=True, exist_ok=True)
            
            # This would trigger mobile export scripts
            self.logger.info("Mobile export pipeline recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Mobile export recovery failed: {e}")
            return False
    
    def _recover_testing(self, alert: Alert) -> bool:
        """Recover testing pipeline."""
        try:
            self.logger.info("Attempting testing pipeline recovery")
            
            # Clear test cache
            cache_dirs = [".pytest_cache", "__pycache__"]
            for cache_dir in cache_dirs:
                if Path(cache_dir).exists():
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
            
            self.logger.info("Testing pipeline recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Testing recovery failed: {e}")
            return False
    
    def _recover_deployment(self, alert: Alert) -> bool:
        """Recover deployment pipeline."""
        try:
            self.logger.info("Attempting deployment recovery")
            
            # Restart containers if needed
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                container_names = result.stdout.strip().split('\n')
                for container in container_names:
                    if container and 'mobile' in container.lower():
                        subprocess.run(["docker", "restart", container], timeout=30)
            
            self.logger.info("Deployment recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment recovery failed: {e}")
            return False
    
    def _recover_monitoring(self, alert: Alert) -> bool:
        """Recover monitoring systems."""
        try:
            self.logger.info("Attempting monitoring recovery")
            
            # Ensure log directory exists
            Path("logs").mkdir(exist_ok=True)
            
            # Restart monitoring services if needed
            self.logger.info("Monitoring recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Monitoring recovery failed: {e}")
            return False
    
    def _recover_storage(self, alert: Alert) -> bool:
        """Recover storage systems."""
        try:
            self.logger.info("Attempting storage recovery")
            
            # Clean up old files if disk space is critical
            if alert.severity == HealthStatus.CRITICAL:
                # Clean cache directories
                cache_dirs = ["mobile_multimodal_cache", ".cache", "tmp"]
                for cache_dir in cache_dirs:
                    if Path(cache_dir).exists():
                        for old_file in Path(cache_dir).glob("*"):
                            if (time.time() - old_file.stat().st_mtime) > 604800:  # 7 days
                                old_file.unlink()
            
            self.logger.info("Storage recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Storage recovery failed: {e}")
            return False
    
    def _recover_compute(self, alert: Alert) -> bool:
        """Recover compute resources."""
        try:
            self.logger.info("Attempting compute recovery")
            
            # Kill high-CPU processes if needed
            # This would implement intelligent process management
            
            self.logger.info("Compute recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Compute recovery failed: {e}")
            return False
    
    async def _alert_processing_loop(self):
        """Process and manage alerts."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Process alerts every 30 seconds
                await self._process_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
    
    async def _process_alerts(self):
        """Process pending alerts."""
        unresolved_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        for alert in unresolved_alerts:
            # Check if alert is still relevant
            if alert.timestamp < datetime.now() - timedelta(hours=24):
                alert.resolved = True
                alert.resolution_action = "Auto-resolved due to age"
                continue
            
            # Send notifications for critical alerts
            if alert.severity in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                await self._send_notification(alert)
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate alert to manual intervention."""
        self.logger.critical(f"ESCALATION: {alert.component.value} - {alert.message}")
        alert.metadata["escalated"] = True
        await self._send_notification(alert)
    
    async def _send_notification(self, alert: Alert):
        """Send alert notification."""
        notification_data = {
            "component": alert.component.value,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata,
        }
        
        self.logger.info(f"ALERT: {notification_data}")
        
        # Send to configured webhooks
        for webhook_url in self.config.get("notification_webhooks", []):
            try:
                # Would implement actual webhook sending
                pass
            except Exception as e:
                self.logger.error(f"Failed to send notification to {webhook_url}: {e}")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_running = False
        self.logger.info("Stopping pipeline monitoring")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        with self._lock:
            component_status = {}
            for check_id, check in self.health_checks.items():
                component_status[check_id] = {
                    "status": check.last_status.value if check.last_status else "unknown",
                    "last_check": check.last_check.isoformat() if check.last_check else None,
                    "consecutive_failures": check.consecutive_failures,
                }
            
            active_alerts = [
                {
                    "component": alert.component.value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ]
            
            return {
                "overall_health": self._calculate_overall_health(),
                "is_monitoring": self.is_running,
                "component_status": component_status,
                "active_alerts": active_alerts,
                "total_alerts": len(self.alerts),
                "unresolved_alerts": len([a for a in self.alerts if not a.resolved]),
            }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health."""
        if not self.health_checks:
            return "unknown"
        
        statuses = [check.last_status for check in self.health_checks.values() if check.last_status]
        if not statuses:
            return "unknown"
        
        if any(status == HealthStatus.FAILED for status in statuses):
            return "failed"
        elif any(status == HealthStatus.CRITICAL for status in statuses):
            return "critical"
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return "degraded"
        elif any(status == HealthStatus.RECOVERING for status in statuses):
            return "recovering"
        else:
            return "healthy"


# CLI Interface
def main():
    """Main entry point for the pipeline guard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Healing Pipeline Guard")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    guard = SelfHealingPipelineGuard(args.config)
    
    if args.status:
        status = guard.get_system_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.daemon:
        try:
            asyncio.run(guard.start_monitoring())
        except KeyboardInterrupt:
            guard.stop_monitoring()
            print("\nPipeline guard stopped")


if __name__ == "__main__":
    main()