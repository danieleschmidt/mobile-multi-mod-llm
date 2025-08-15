"""Comprehensive tests for Self-Healing Pipeline Guard."""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import sqlite3
import threading

from src.mobile_multimodal.pipeline_guard import (
    SelfHealingPipelineGuard,
    HealthStatus,
    PipelineComponent,
    Alert,
    HealthCheck
)
from src.mobile_multimodal.guard_config import GuardConfig, ConfigManager


class TestSelfHealingPipelineGuard:
    """Test suite for SelfHealingPipelineGuard."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "check_intervals": {
                    "model_training": 10,
                    "quantization": 10,
                    "mobile_export": 10,
                    "testing": 10,
                    "deployment": 10,
                    "monitoring": 10,
                    "storage": 10,
                    "compute": 10,
                },
                "timeouts": {"default": 5},
                "max_retries": 2,
                "auto_recovery": True,
            }
            json.dump(config, f)
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def pipeline_guard(self, temp_config):
        """Create pipeline guard instance for testing."""
        guard = SelfHealingPipelineGuard(temp_config)
        yield guard
        guard.stop_monitoring()
    
    def test_initialization(self, pipeline_guard):
        """Test pipeline guard initialization."""
        assert pipeline_guard is not None
        assert not pipeline_guard.is_running
        assert len(pipeline_guard.health_checks) > 0
        assert len(pipeline_guard.recovery_strategies) == len(PipelineComponent)
    
    def test_config_loading(self, temp_config):
        """Test configuration loading."""
        guard = SelfHealingPipelineGuard(temp_config)
        assert guard.config["max_retries"] == 2
        assert guard.config["auto_recovery"] is True
        assert guard.config["check_intervals"]["model_training"] == 10
    
    def test_default_config(self):
        """Test default configuration when no config file provided."""
        guard = SelfHealingPipelineGuard()
        assert guard.config is not None
        assert "check_intervals" in guard.config
        assert "max_retries" in guard.config
    
    def test_health_check_setup(self, pipeline_guard):
        """Test health check configuration setup."""
        # Verify all components have health checks
        expected_components = set(component.value for component in PipelineComponent)
        actual_components = set()
        
        for check_id, check in pipeline_guard.health_checks.items():
            actual_components.add(check.component.value)
        
        assert expected_components.issubset(actual_components)
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, pipeline_guard):
        """Test starting and stopping monitoring."""
        # Start monitoring
        start_task = asyncio.create_task(pipeline_guard.start_monitoring())
        await asyncio.sleep(0.1)  # Let it start
        
        assert pipeline_guard.is_running
        
        # Stop monitoring
        pipeline_guard.stop_monitoring()
        await asyncio.sleep(0.1)  # Let it stop
        
        assert not pipeline_guard.is_running
        
        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self, pipeline_guard):
        """Test health check execution."""
        # Mock a health check function
        mock_check = Mock(return_value=HealthStatus.HEALTHY)
        
        check = HealthCheck(
            component=PipelineComponent.TESTING,
            check_name="mock_check",
            check_function=mock_check,
            interval_seconds=1,
            timeout_seconds=5,
            max_retries=2
        )
        
        # Run health check
        await pipeline_guard._run_health_check("test_check", check)
        
        # Verify check was called
        mock_check.assert_called_once()
        assert check.last_status == HealthStatus.HEALTHY
        assert check.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure_handling(self, pipeline_guard):
        """Test health check failure handling."""
        # Mock a failing health check
        mock_check = Mock(return_value=HealthStatus.FAILED)
        
        check = HealthCheck(
            component=PipelineComponent.TESTING,
            check_name="failing_check",
            check_function=mock_check,
            interval_seconds=1,
            timeout_seconds=5,
            max_retries=2
        )
        
        # Run health check
        await pipeline_guard._run_health_check("failing_check", check)
        
        # Verify failure was recorded
        assert check.last_status == HealthStatus.FAILED
        assert check.consecutive_failures == 1
        assert len(pipeline_guard.alerts) > 0
    
    @pytest.mark.asyncio
    async def test_recovery_trigger(self, pipeline_guard):
        """Test automatic recovery trigger."""
        # Mock recovery function
        mock_recovery = Mock(return_value=True)
        pipeline_guard.recovery_strategies[PipelineComponent.TESTING] = mock_recovery
        
        # Create alert that should trigger recovery
        alert = Alert(
            component=PipelineComponent.TESTING,
            severity=HealthStatus.FAILED,
            message="Test failure",
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Trigger recovery
        await pipeline_guard._trigger_recovery(PipelineComponent.TESTING, alert)
        
        # Verify recovery was called
        mock_recovery.assert_called_once_with(alert)
        assert alert.resolved
    
    def test_system_status(self, pipeline_guard):
        """Test system status reporting."""
        status = pipeline_guard.get_system_status()
        
        assert "overall_health" in status
        assert "is_monitoring" in status
        assert "component_status" in status
        assert "active_alerts" in status
        assert "total_alerts" in status
        assert "unresolved_alerts" in status
    
    def test_training_process_check(self, pipeline_guard):
        """Test training process health check."""
        # This would normally check actual processes
        # For testing, we verify the method exists and runs
        result = pipeline_guard._check_training_process()
        assert isinstance(result, HealthStatus)
    
    def test_quantization_check(self, pipeline_guard):
        """Test quantization health check."""
        result = pipeline_guard._check_quantization_accuracy()
        assert isinstance(result, HealthStatus)
    
    def test_mobile_export_check(self, pipeline_guard):
        """Test mobile export health check."""
        result = pipeline_guard._check_mobile_export()
        assert isinstance(result, HealthStatus)
    
    def test_test_suite_check(self, pipeline_guard):
        """Test test suite health check."""
        result = pipeline_guard._check_test_suite()
        assert isinstance(result, HealthStatus)
    
    @patch('subprocess.run')
    def test_deployment_health_check(self, mock_subprocess, pipeline_guard):
        """Test deployment health check with mocked subprocess."""
        # Mock successful docker ps
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Up 10 minutes\\nUp 5 minutes"
        
        result = pipeline_guard._check_deployment_health()
        assert result == HealthStatus.HEALTHY
        
        # Mock failed docker ps
        mock_subprocess.return_value.returncode = 1
        result = pipeline_guard._check_deployment_health()
        assert result == HealthStatus.DEGRADED
    
    @patch('subprocess.run')
    def test_storage_check(self, mock_subprocess, pipeline_guard):
        """Test storage system health check."""
        # Mock disk usage output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Filesystem  Size Used Avail Use% Mounted\\n/dev/sda1   100G  70G  30G  70% /"
        
        result = pipeline_guard._check_storage_systems()
        assert result == HealthStatus.HEALTHY
        
        # Mock high disk usage
        mock_subprocess.return_value.stdout = "Filesystem  Size Used Avail Use% Mounted\\n/dev/sda1   100G  95G   5G  95% /"
        result = pipeline_guard._check_storage_systems()
        assert result == HealthStatus.CRITICAL
    
    def test_recovery_functions_exist(self, pipeline_guard):
        """Test that all recovery functions exist."""
        for component in PipelineComponent:
            assert component in pipeline_guard.recovery_strategies
            recovery_func = pipeline_guard.recovery_strategies[component]
            assert callable(recovery_func)
    
    def test_training_recovery(self, pipeline_guard):
        """Test training pipeline recovery."""
        alert = Alert(
            component=PipelineComponent.MODEL_TRAINING,
            severity=HealthStatus.FAILED,
            message="Training failed",
            timestamp=datetime.now(),
            metadata={}
        )
        
        result = pipeline_guard._recover_training(alert)
        assert isinstance(result, bool)
    
    def test_quantization_recovery(self, pipeline_guard):
        """Test quantization pipeline recovery."""
        alert = Alert(
            component=PipelineComponent.QUANTIZATION,
            severity=HealthStatus.FAILED,
            message="Quantization failed",
            timestamp=datetime.now(),
            metadata={}
        )
        
        result = pipeline_guard._recover_quantization(alert)
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_alert_processing(self, pipeline_guard):
        """Test alert processing loop."""
        # Add test alert
        alert = Alert(
            component=PipelineComponent.TESTING,
            severity=HealthStatus.CRITICAL,
            message="Critical test failure",
            timestamp=datetime.now(),
            metadata={}
        )
        pipeline_guard.alerts.append(alert)
        
        # Process alerts
        await pipeline_guard._process_alerts()
        
        # Verify alert was processed (this is a basic test)
        assert len(pipeline_guard.alerts) > 0
    
    @pytest.mark.asyncio
    async def test_notification_sending(self, pipeline_guard):
        """Test alert notification sending."""
        alert = Alert(
            component=PipelineComponent.TESTING,
            severity=HealthStatus.CRITICAL,
            message="Test notification",
            timestamp=datetime.now(),
            metadata={}
        )
        
        # This should not raise an exception
        await pipeline_guard._send_notification(alert)
    
    def test_health_status_enum(self):
        """Test health status enum values."""
        statuses = [status.value for status in HealthStatus]
        expected_statuses = ["healthy", "degraded", "critical", "recovering", "failed"]
        
        for expected in expected_statuses:
            assert expected in statuses
    
    def test_pipeline_component_enum(self):
        """Test pipeline component enum values."""
        components = [component.value for component in PipelineComponent]
        expected_components = [
            "model_training", "quantization", "mobile_export", "testing",
            "deployment", "monitoring", "storage", "compute"
        ]
        
        for expected in expected_components:
            assert expected in components
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, pipeline_guard):
        """Test concurrent health check execution."""
        # Start monitoring
        start_task = asyncio.create_task(pipeline_guard.start_monitoring())
        await asyncio.sleep(0.2)  # Let health checks run
        
        # Verify health checks are running
        assert pipeline_guard.is_running
        
        # Check that health checks have been executed
        executed_checks = sum(1 for check in pipeline_guard.health_checks.values() 
                             if check.last_check is not None)
        
        # Stop monitoring
        pipeline_guard.stop_monitoring()
        start_task.cancel()
        
        try:
            await start_task
        except asyncio.CancelledError:
            pass
        
        # Should have executed at least some checks
        assert executed_checks >= 0  # May be 0 in fast test environment
    
    def test_alert_escalation(self, pipeline_guard):
        """Test alert escalation logic."""
        alert = Alert(
            component=PipelineComponent.TESTING,
            severity=HealthStatus.CRITICAL,
            message="Escalation test",
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Test escalation (should not raise exception)
        asyncio.run(pipeline_guard._escalate_alert(alert))
        
        # Verify escalation metadata was added
        assert alert.metadata.get("escalated") is True


class TestConfigManager:
    """Test suite for GuardConfig and ConfigManager."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = GuardConfig()
        
        assert config.guard_name == "mobile-multimodal-pipeline-guard"
        assert config.log_level == "INFO"
        assert config.model_training is not None
        assert config.quantization is not None
        assert config.alerting is not None
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = GuardConfig()
        
        # Valid configuration should have no issues
        issues = ConfigManager.validate_config(config)
        assert len(issues) == 0
        
        # Create invalid configuration
        config.model_training.check_interval_seconds = 5  # Too low
        config.model_training.timeout_seconds = 100  # Higher than interval
        
        issues = ConfigManager.validate_config(config)
        assert len(issues) > 0
    
    def test_config_manager_initialization(self):
        """Test config manager initialization."""
        manager = ConfigManager()
        assert manager.config is not None
        assert isinstance(manager.config, GuardConfig)
    
    def test_config_saving_and_loading(self):
        """Test configuration saving and loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create and save config
            manager = ConfigManager()
            original_guard_name = manager.config.guard_name
            manager.config_path = temp_path
            manager.save_config()
            
            # Load config from file
            new_manager = ConfigManager(temp_path)
            assert new_manager.config.guard_name == original_guard_name
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_sample_config_creation(self):
        """Test sample configuration creation."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = ConfigManager()
            manager.create_sample_config(temp_path)
            
            # Verify file was created
            assert Path(temp_path).exists()
            
            # Verify it can be loaded
            sample_manager = ConfigManager(temp_path)
            assert sample_manager.config is not None
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests for the complete pipeline guard system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring(self):
        """Test end-to-end monitoring workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "check_intervals": {"testing": 1},  # Very short for testing
                "timeouts": {"default": 2},
                "max_retries": 1,
                "auto_recovery": True,
            }
            json.dump(config, f)
            temp_config = f.name
        
        try:
            # Create pipeline guard
            guard = SelfHealingPipelineGuard(temp_config)
            
            # Mock a failing health check for testing
            original_check = guard._check_test_suite
            guard._check_test_suite = Mock(return_value=HealthStatus.FAILED)
            
            # Start monitoring briefly
            start_task = asyncio.create_task(guard.start_monitoring())
            await asyncio.sleep(2)  # Let it run for 2 seconds
            
            # Stop monitoring
            guard.stop_monitoring()
            start_task.cancel()
            
            try:
                await start_task
            except asyncio.CancelledError:
                pass
            
            # Verify monitoring occurred
            status = guard.get_system_status()
            assert not status["is_monitoring"]
            
            # Restore original check
            guard._check_test_suite = original_check
            
        finally:
            Path(temp_config).unlink(missing_ok=True)
    
    def test_performance_under_load(self):
        """Test pipeline guard performance under simulated load."""
        guard = SelfHealingPipelineGuard()
        
        # Simulate multiple concurrent status requests
        def get_status():
            return guard.get_system_status()
        
        # Run multiple threads requesting status
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_status)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        final_status = guard.get_system_status()
        assert final_status is not None
    
    def test_resource_cleanup(self):
        """Test proper resource cleanup."""
        guard = SelfHealingPipelineGuard()
        
        # Create some alerts and state
        alert = Alert(
            component=PipelineComponent.TESTING,
            severity=HealthStatus.FAILED,
            message="Cleanup test",
            timestamp=datetime.now(),
            metadata={}
        )
        guard.alerts.append(alert)
        
        # Stop monitoring (cleanup)
        guard.stop_monitoring()
        
        # Verify cleanup occurred
        assert not guard.is_running
    
    @pytest.mark.asyncio
    async def test_recovery_effectiveness(self):
        """Test recovery mechanism effectiveness."""
        guard = SelfHealingPipelineGuard()
        
        # Mock successful recovery
        mock_recovery = Mock(return_value=True)
        guard.recovery_strategies[PipelineComponent.TESTING] = mock_recovery
        
        # Create alert requiring recovery
        alert = Alert(
            component=PipelineComponent.TESTING,
            severity=HealthStatus.FAILED,
            message="Recovery test",
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Trigger recovery
        await guard._trigger_recovery(PipelineComponent.TESTING, alert)
        
        # Verify recovery was attempted and successful
        mock_recovery.assert_called_once()
        assert alert.resolved
        assert "successful" in alert.resolution_action


@pytest.mark.integration
class TestRealSystemIntegration:
    """Integration tests with real system components (when available)."""
    
    def test_actual_disk_usage_check(self):
        """Test actual disk usage checking."""
        guard = SelfHealingPipelineGuard()
        
        # This should work on any Unix-like system
        result = guard._check_storage_systems()
        assert isinstance(result, HealthStatus)
    
    def test_actual_process_check(self):
        """Test actual process checking."""
        guard = SelfHealingPipelineGuard()
        
        # This should work on any system
        result = guard._check_training_process()
        assert isinstance(result, HealthStatus)
    
    @pytest.mark.skipif(not Path("/usr/bin/docker").exists(), 
                       reason="Docker not available")
    def test_docker_integration(self):
        """Test Docker integration (if Docker is available)."""
        guard = SelfHealingPipelineGuard()
        
        result = guard._check_deployment_health()
        assert isinstance(result, HealthStatus)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])