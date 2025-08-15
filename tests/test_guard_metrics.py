"""Comprehensive tests for Pipeline Guard Metrics system."""

import pytest
import asyncio
import sqlite3
import tempfile
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading

from src.mobile_multimodal.guard_metrics import (
    MetricsCollector,
    AnomalyDetector,
    AlertManager,
    MetricPoint,
    AlertThreshold,
    HealthStatus
)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class TestMetricsCollector:
    """Test suite for MetricsCollector."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def metrics_collector(self, temp_db):
        """Create metrics collector for testing."""
        collector = MetricsCollector(temp_db)
        yield collector
        asyncio.run(collector.stop_collection())
    
    def test_initialization(self, temp_db):
        """Test metrics collector initialization."""
        collector = MetricsCollector(temp_db)
        
        assert collector.db_path == temp_db
        assert len(collector.metrics_buffer) == 0
        assert not collector.is_running
        
        # Check database was created
        assert Path(temp_db).exists()
    
    def test_database_initialization(self, temp_db):
        """Test database schema creation."""
        collector = MetricsCollector(temp_db)
        
        # Check database structure
        conn = sqlite3.connect(temp_db)
        try:
            cursor = conn.cursor()
            
            # Check metrics table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'")
            result = cursor.fetchone()
            assert result is not None
            
            # Check indexes exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            assert 'idx_metrics_timestamp' in indexes
            assert 'idx_metrics_component_metric' in indexes
            
        finally:
            conn.close()
    
    def test_record_metric(self, metrics_collector):
        """Test metric recording."""
        # Record a metric
        metrics_collector.record_metric("test_component", "test_metric", 42.5)
        
        # Check metric was buffered
        assert len(metrics_collector.metrics_buffer) == 1
        
        metric = metrics_collector.metrics_buffer[0]
        assert metric.component == "test_component"
        assert metric.metric_name == "test_metric"
        assert metric.value == 42.5
        assert isinstance(metric.timestamp, datetime)
    
    def test_record_metric_with_labels(self, metrics_collector):
        """Test metric recording with labels."""
        labels = {"env": "test", "version": "1.0"}
        metrics_collector.record_metric("component", "metric", 100.0, labels)
        
        metric = metrics_collector.metrics_buffer[0]
        assert metric.labels == labels
    
    @pytest.mark.asyncio
    async def test_metric_collection_start_stop(self, metrics_collector):
        """Test starting and stopping metric collection."""
        # Start collection
        await metrics_collector.start_collection()
        assert metrics_collector.is_running
        
        # Stop collection
        await metrics_collector.stop_collection()
        assert not metrics_collector.is_running
    
    @pytest.mark.asyncio
    async def test_metric_flushing(self, metrics_collector):
        """Test metric flushing to database."""
        # Record some metrics
        for i in range(5):
            metrics_collector.record_metric("test", f"metric_{i}", float(i))
        
        # Flush to database
        await metrics_collector._flush_metrics()
        
        # Check buffer was cleared
        assert len(metrics_collector.metrics_buffer) == 0
        
        # Check metrics were stored in database
        conn = sqlite3.connect(metrics_collector.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM metrics")
            count = cursor.fetchone()[0]
            assert count == 5
        finally:
            conn.close()
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    async def test_system_metrics_collection(self, metrics_collector):
        """Test system metrics collection."""
        # Collect system metrics
        await metrics_collector._collect_system_metrics()
        
        # Check that system metrics were recorded
        system_metrics = [m for m in metrics_collector.metrics_buffer 
                         if m.component == "system"]
        
        assert len(system_metrics) > 0
        
        # Check for expected metric types
        metric_names = {m.metric_name for m in system_metrics}
        expected_metrics = {"cpu_usage_percent", "memory_usage_percent", 
                           "disk_usage_percent", "disk_free_gb"}
        
        assert len(expected_metrics.intersection(metric_names)) > 0
    
    def test_get_metrics(self, metrics_collector):
        """Test metrics retrieval."""
        # Store some test data
        test_metrics = [
            MetricPoint(datetime.now(), "comp1", "metric1", 10.0),
            MetricPoint(datetime.now(), "comp1", "metric2", 20.0),
            MetricPoint(datetime.now(), "comp2", "metric1", 30.0),
        ]
        
        # Manually insert into database
        conn = sqlite3.connect(metrics_collector.db_path)
        try:
            cursor = conn.cursor()
            for metric in test_metrics:
                cursor.execute('''
                    INSERT INTO metrics (timestamp, component, metric_name, value, labels)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.component,
                    metric.metric_name,
                    metric.value,
                    "{}"
                ))
            conn.commit()
        finally:
            conn.close()
        
        # Test retrieval
        all_metrics = metrics_collector.get_metrics()
        assert len(all_metrics) == 3
        
        # Test component filtering
        comp1_metrics = metrics_collector.get_metrics(component="comp1")
        assert len(comp1_metrics) == 2
        
        # Test metric name filtering
        metric1_metrics = metrics_collector.get_metrics(metric_name="metric1")
        assert len(metric1_metrics) == 2
    
    def test_get_aggregated_metrics(self, metrics_collector):
        """Test aggregated metrics calculation."""
        # Store test data
        now = datetime.now()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        conn = sqlite3.connect(metrics_collector.db_path)
        try:
            cursor = conn.cursor()
            for i, value in enumerate(values):
                timestamp = now + timedelta(minutes=i)
                cursor.execute('''
                    INSERT INTO metrics (timestamp, component, metric_name, value, labels)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp.isoformat(), "test_comp", "test_metric", value, "{}"))
            conn.commit()
        finally:
            conn.close()
        
        # Test aggregations
        start_time = now - timedelta(minutes=1)
        end_time = now + timedelta(minutes=10)
        
        avg_value = metrics_collector.get_aggregated_metrics(
            "test_comp", "test_metric", start_time, end_time, "avg"
        )
        assert avg_value == 30.0  # Average of 10,20,30,40,50
        
        min_value = metrics_collector.get_aggregated_metrics(
            "test_comp", "test_metric", start_time, end_time, "min"
        )
        assert min_value == 10.0
        
        max_value = metrics_collector.get_aggregated_metrics(
            "test_comp", "test_metric", start_time, end_time, "max"
        )
        assert max_value == 50.0
    
    def test_cleanup_old_metrics(self, metrics_collector):
        """Test cleanup of old metrics."""
        # Store old and new metrics
        old_time = datetime.now() - timedelta(days=40)
        new_time = datetime.now()
        
        conn = sqlite3.connect(metrics_collector.db_path)
        try:
            cursor = conn.cursor()
            
            # Insert old metric
            cursor.execute('''
                INSERT INTO metrics (timestamp, component, metric_name, value, labels)
                VALUES (?, ?, ?, ?, ?)
            ''', (old_time.isoformat(), "test", "old_metric", 1.0, "{}"))
            
            # Insert new metric
            cursor.execute('''
                INSERT INTO metrics (timestamp, component, metric_name, value, labels)
                VALUES (?, ?, ?, ?, ?)
            ''', (new_time.isoformat(), "test", "new_metric", 2.0, "{}"))
            
            conn.commit()
        finally:
            conn.close()
        
        # Cleanup old metrics (keep 30 days)
        metrics_collector.cleanup_old_metrics(days_to_keep=30)
        
        # Check that old metric was removed
        remaining_metrics = metrics_collector.get_metrics()
        assert len(remaining_metrics) == 1
        assert remaining_metrics[0].metric_name == "new_metric"


class TestAnomalyDetector:
    """Test suite for AnomalyDetector."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def anomaly_detector(self, temp_db):
        """Create anomaly detector for testing."""
        collector = MetricsCollector(temp_db)
        detector = AnomalyDetector(collector)
        yield detector
    
    def test_initialization(self, anomaly_detector):
        """Test anomaly detector initialization."""
        assert anomaly_detector.metrics_collector is not None
        assert len(anomaly_detector.baselines) == 0
    
    def test_baseline_calculation(self, anomaly_detector):
        """Test baseline calculation."""
        # Prepare test data
        now = datetime.now()
        values = [10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 10.5, 11.5, 9.5, 12.5]
        
        conn = sqlite3.connect(anomaly_detector.metrics_collector.db_path)
        try:
            cursor = conn.cursor()
            for i, value in enumerate(values):
                timestamp = now - timedelta(hours=i)
                cursor.execute('''
                    INSERT INTO metrics (timestamp, component, metric_name, value, labels)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp.isoformat(), "test_comp", "test_metric", value, "{}"))
            conn.commit()
        finally:
            conn.close()
        
        # Calculate baselines
        anomaly_detector.calculate_baselines(lookback_days=1)
        
        # Check baseline was calculated
        assert "test_comp" in anomaly_detector.baselines
        assert "test_metric" in anomaly_detector.baselines["test_comp"]
        
        baseline = anomaly_detector.baselines["test_comp"]["test_metric"]
        assert "mean" in baseline
        assert "stdev" in baseline
        assert baseline["sample_count"] == len(values)
    
    def test_anomaly_detection(self, anomaly_detector):
        """Test anomaly detection."""
        # Set up baseline manually
        anomaly_detector.baselines = {
            "test_comp": {
                "test_metric": {
                    "mean": 10.0,
                    "stdev": 2.0,
                    "min": 6.0,
                    "max": 14.0,
                    "median": 10.0,
                    "p95": 13.8,
                    "p99": 13.96,
                    "sample_count": 100,
                    "calculated_at": datetime.now().isoformat(),
                }
            }
        }
        
        # Test normal value
        result = anomaly_detector.detect_anomalies("test_comp", "test_metric", 10.5)
        assert not result["is_anomaly"]
        assert result["severity"] == "normal"
        
        # Test anomalous value (3+ standard deviations)
        result = anomaly_detector.detect_anomalies("test_comp", "test_metric", 17.0)
        assert result["is_anomaly"]
        assert result["severity"] == "critical"
        
        # Test warning value (2+ standard deviations)
        result = anomaly_detector.detect_anomalies("test_comp", "test_metric", 14.5)
        assert result["is_anomaly"]
        assert result["severity"] == "warning"
    
    def test_baseline_persistence(self, anomaly_detector):
        """Test saving and loading baselines."""
        # Set up test baseline
        test_baseline = {
            "test_comp": {
                "test_metric": {
                    "mean": 10.0,
                    "stdev": 2.0,
                    "calculated_at": datetime.now().isoformat(),
                }
            }
        }
        anomaly_detector.baselines = test_baseline
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save baselines
            anomaly_detector.save_baselines(temp_path)
            assert Path(temp_path).exists()
            
            # Clear baselines and load
            anomaly_detector.baselines = {}
            anomaly_detector.load_baselines(temp_path)
            
            # Check baselines were loaded
            assert "test_comp" in anomaly_detector.baselines
            assert anomaly_detector.baselines["test_comp"]["test_metric"]["mean"] == 10.0
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestAlertManager:
    """Test suite for AlertManager."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def alert_manager(self, temp_db):
        """Create alert manager for testing."""
        collector = MetricsCollector(temp_db)
        manager = AlertManager(collector)
        yield manager
    
    def test_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert alert_manager.metrics_collector is not None
        assert len(alert_manager.thresholds) > 0
        assert len(alert_manager.active_alerts) == 0
        assert len(alert_manager.alert_history) == 0
    
    def test_add_remove_threshold(self, alert_manager):
        """Test adding and removing thresholds."""
        # Add custom threshold
        threshold = AlertThreshold("custom_metric", 80.0, 95.0)
        alert_manager.add_threshold(threshold)
        
        assert "custom_metric" in alert_manager.thresholds
        assert alert_manager.thresholds["custom_metric"] == threshold
        
        # Remove threshold
        alert_manager.remove_threshold("custom_metric")
        assert "custom_metric" not in alert_manager.thresholds
    
    def test_threshold_evaluation(self, alert_manager):
        """Test threshold evaluation."""
        # Prepare test data that exceeds thresholds
        now = datetime.now()
        
        # Add metrics that exceed CPU threshold (80%)
        conn = sqlite3.connect(alert_manager.metrics_collector.db_path)
        try:
            cursor = conn.cursor()
            for i in range(5):  # 5 consecutive high values
                timestamp = now - timedelta(minutes=i)
                cursor.execute('''
                    INSERT INTO metrics (timestamp, component, metric_name, value, labels)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp.isoformat(), "test_comp", "cpu_usage_percent", 85.0, "{}"))
            conn.commit()
        finally:
            conn.close()
        
        # Evaluate thresholds
        new_alerts = alert_manager.evaluate_thresholds()
        
        # Should generate alerts for high CPU usage
        cpu_alerts = [alert for alert in new_alerts 
                     if alert["metric_name"] == "cpu_usage_percent"]
        assert len(cpu_alerts) > 0
    
    def test_alert_resolution(self, alert_manager):
        """Test manual alert resolution."""
        # Create test alert
        test_alert = {
            "id": "test_alert_123",
            "component": "test_comp",
            "metric_name": "test_metric",
            "severity": "critical",
            "message": "Test alert",
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "metadata": {},
        }
        
        alert_manager.active_alerts["test_key"] = test_alert
        alert_manager.alert_history.append(test_alert)
        
        # Resolve alert
        alert_manager.resolve_alert("test_alert_123")
        
        # Check alert was resolved
        assert len(alert_manager.active_alerts) == 0
        assert test_alert["status"] == "resolved"
        assert "resolved_at" in test_alert
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        # Add test alert
        test_alert = {
            "id": "active_test",
            "status": "active",
            "message": "Active test alert"
        }
        alert_manager.active_alerts["test_key"] = test_alert
        
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0]["id"] == "active_test"
    
    def test_alert_history(self, alert_manager):
        """Test alert history functionality."""
        # Add test alerts to history
        for i in range(15):
            alert = {
                "id": f"alert_{i}",
                "message": f"Test alert {i}",
                "timestamp": datetime.now().isoformat()
            }
            alert_manager.alert_history.append(alert)
        
        # Get limited history
        history = alert_manager.get_alert_history(limit=10)
        assert len(history) == 10
        
        # Should get most recent alerts
        assert history[-1]["id"] == "alert_14"


class TestMetricsIntegration:
    """Integration tests for the complete metrics system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_metrics_flow(self):
        """Test complete metrics collection and analysis flow."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            # Initialize components
            collector = MetricsCollector(temp_db)
            detector = AnomalyDetector(collector)
            alert_manager = AlertManager(collector)
            
            # Record some metrics
            for i in range(20):
                value = 50.0 + np.sin(i * 0.1) * 10  # Sine wave pattern
                collector.record_metric("test_system", "cpu_usage_percent", value)
            
            # Add some anomalous values
            collector.record_metric("test_system", "cpu_usage_percent", 95.0)  # High value
            collector.record_metric("test_system", "cpu_usage_percent", 98.0)  # Very high
            
            # Flush to database
            await collector._flush_metrics()
            
            # Calculate baselines
            detector.calculate_baselines(lookback_days=1)
            
            # Test anomaly detection
            normal_result = detector.detect_anomalies("test_system", "cpu_usage_percent", 55.0)
            assert not normal_result["is_anomaly"]
            
            anomaly_result = detector.detect_anomalies("test_system", "cpu_usage_percent", 100.0)
            assert anomaly_result["is_anomaly"]
            
            # Test alert evaluation
            alerts = alert_manager.evaluate_thresholds()
            
            # Should have alerts for high CPU usage
            cpu_alerts = [alert for alert in alerts 
                         if alert["metric_name"] == "cpu_usage_percent"]
            assert len(cpu_alerts) > 0
            
        finally:
            Path(temp_db).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_concurrent_metrics_operations(self):
        """Test concurrent metrics operations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            collector = MetricsCollector(temp_db)
            
            # Define concurrent operations
            async def record_metrics():
                for i in range(100):
                    collector.record_metric("concurrent_test", f"metric_{i%10}", float(i))
                    await asyncio.sleep(0.001)  # Small delay
            
            async def flush_metrics():
                for _ in range(10):
                    await collector._flush_metrics()
                    await asyncio.sleep(0.01)
            
            # Run operations concurrently
            await asyncio.gather(
                record_metrics(),
                flush_metrics(),
                return_exceptions=True
            )
            
            # Final flush
            await collector._flush_metrics()
            
            # Verify data integrity
            metrics = collector.get_metrics(limit=1000)
            assert len(metrics) > 0
            
        finally:
            Path(temp_db).unlink(missing_ok=True)
    
    def test_performance_with_large_dataset(self):
        """Test performance with large dataset."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            collector = MetricsCollector(temp_db)
            
            # Record large number of metrics
            start_time = time.time()
            for i in range(1000):
                collector.record_metric("perf_test", "test_metric", float(i))
            record_time = time.time() - start_time
            
            # Flush to database
            start_time = time.time()
            asyncio.run(collector._flush_metrics())
            flush_time = time.time() - start_time
            
            # Query metrics
            start_time = time.time()
            metrics = collector.get_metrics(limit=1000)
            query_time = time.time() - start_time
            
            # Performance assertions (should be reasonably fast)
            assert record_time < 1.0  # Recording should be fast
            assert flush_time < 5.0   # Flushing should complete in reasonable time
            assert query_time < 2.0   # Querying should be fast
            assert len(metrics) == 1000
            
        finally:
            Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])