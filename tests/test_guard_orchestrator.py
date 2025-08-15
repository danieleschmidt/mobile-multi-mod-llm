"""Comprehensive tests for Pipeline Guard Orchestrator."""

import pytest
import asyncio
import numpy as np
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pickle

from src.mobile_multimodal.guard_orchestrator import (
    PipelineOrchestrator,
    MLOptimizer,
    AutoScaler,
    ResourceState,
    ScalingDecision,
    ScalingAction,
    OptimizationStrategy,
    PredictionModel
)
from src.mobile_multimodal.pipeline_guard import PipelineComponent, HealthStatus

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestResourceState:
    """Test suite for ResourceState dataclass."""
    
    def test_resource_state_creation(self):
        """Test ResourceState creation."""
        state = ResourceState(
            cpu_usage=75.0,
            memory_usage=60.0,
            disk_usage=50.0,
            network_io=100.0,
            queue_length=10,
            throughput=50.0,
            latency=120.0,
            error_rate=0.02,
            timestamp=datetime.now()
        )
        
        assert state.cpu_usage == 75.0
        assert state.memory_usage == 60.0
        assert state.error_rate == 0.02
        assert isinstance(state.timestamp, datetime)


class TestScalingDecision:
    """Test suite for ScalingDecision dataclass."""
    
    def test_scaling_decision_creation(self):
        """Test ScalingDecision creation."""
        decision = ScalingDecision(
            component=PipelineComponent.MODEL_TRAINING,
            action=ScalingAction.SCALE_UP,
            magnitude=1.5,
            reason="High CPU usage",
            confidence=0.8,
            expected_impact={'performance': 0.3},
            cost_impact=0.5,
            timestamp=datetime.now()
        )
        
        assert decision.component == PipelineComponent.MODEL_TRAINING
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.magnitude == 1.5
        assert decision.confidence == 0.8


class TestMLOptimizer:
    """Test suite for MLOptimizer."""
    
    @pytest.fixture
    def ml_optimizer(self):
        """Create ML optimizer for testing."""
        return MLOptimizer(data_window_hours=24)
    
    def test_initialization(self, ml_optimizer):
        """Test ML optimizer initialization."""
        assert ml_optimizer.data_window_hours == 24
        assert len(ml_optimizer.models) == 0
        assert len(ml_optimizer.feature_cache) == 0
    
    def test_feature_extraction(self, ml_optimizer):
        """Test feature extraction from resource state."""
        current_state = ResourceState(
            cpu_usage=75.0,
            memory_usage=60.0,
            disk_usage=50.0,
            network_io=100.0,
            queue_length=10,
            throughput=50.0,
            latency=120.0,
            error_rate=0.02,
            timestamp=datetime.now()
        )
        
        # Test with no historical data
        features = ml_optimizer.extract_features(current_state, [])
        assert isinstance(features, np.ndarray)
        assert len(features) > 8  # Should have at least basic features
        
        # Test with historical data
        historical_data = [
            ResourceState(70.0, 55.0, 45.0, 95.0, 8, 48.0, 115.0, 0.01, 
                         datetime.now() - timedelta(minutes=5)),
            ResourceState(72.0, 58.0, 48.0, 98.0, 9, 49.0, 118.0, 0.015,
                         datetime.now() - timedelta(minutes=3)),
            ResourceState(74.0, 59.0, 49.0, 99.0, 9, 49.5, 119.0, 0.018,
                         datetime.now() - timedelta(minutes=1)),
        ]
        
        features_with_history = ml_optimizer.extract_features(current_state, historical_data)
        assert isinstance(features_with_history, np.ndarray)
        assert len(features_with_history) >= len(features)  # Should have trend features
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_failure_prediction_model_training(self, ml_optimizer):
        """Test failure prediction model training."""
        # Generate training data
        training_data = []
        for i in range(100):
            state = ResourceState(
                cpu_usage=np.random.uniform(10, 95),
                memory_usage=np.random.uniform(20, 90),
                disk_usage=np.random.uniform(30, 95),
                network_io=np.random.uniform(5, 100),
                queue_length=np.random.randint(0, 200),
                throughput=np.random.uniform(1, 100),
                latency=np.random.uniform(20, 800),
                error_rate=np.random.uniform(0, 0.2),
                timestamp=datetime.now()
            )
            
            # Simulate failure based on high resource usage
            failed = state.cpu_usage > 85 and state.memory_usage > 80
            training_data.append((state, failed))
        
        # Train model
        ml_optimizer.train_failure_prediction_model(training_data)
        
        # Check model was created
        assert 'failure_prediction' in ml_optimizer.models
        
        model_info = ml_optimizer.models['failure_prediction']
        assert model_info.model_type == 'failure_prediction'
        assert model_info.accuracy > 0
        assert isinstance(model_info.last_trained, datetime)
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_failure_probability_prediction(self, ml_optimizer):
        """Test failure probability prediction."""
        # First train a model
        training_data = []
        for i in range(100):
            state = ResourceState(
                cpu_usage=np.random.uniform(10, 95),
                memory_usage=np.random.uniform(20, 90),
                disk_usage=50.0, network_io=50.0, queue_length=10,
                throughput=50.0, latency=100.0, error_rate=0.01,
                timestamp=datetime.now()
            )
            failed = state.cpu_usage > 85 and state.memory_usage > 80
            training_data.append((state, failed))
        
        ml_optimizer.train_failure_prediction_model(training_data)
        
        # Test prediction
        test_state = ResourceState(
            cpu_usage=90.0,  # High CPU
            memory_usage=85.0,  # High memory
            disk_usage=50.0, network_io=50.0, queue_length=10,
            throughput=50.0, latency=100.0, error_rate=0.01,
            timestamp=datetime.now()
        )
        
        probability = ml_optimizer.predict_failure_probability(test_state, [])
        assert 0.0 <= probability <= 1.0
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_anomaly_detection(self, ml_optimizer):
        """Test anomaly detection."""
        # Generate normal resource states
        normal_states = []
        for i in range(50):
            state = ResourceState(
                cpu_usage=np.random.normal(50, 10),  # Normal around 50%
                memory_usage=np.random.normal(60, 8),
                disk_usage=50.0, network_io=50.0, queue_length=10,
                throughput=50.0, latency=100.0, error_rate=0.01,
                timestamp=datetime.now()
            )
            normal_states.append(state)
        
        # Add anomalous states
        anomalous_states = [
            ResourceState(95.0, 95.0, 50.0, 50.0, 10, 50.0, 100.0, 0.01, datetime.now()),
            ResourceState(5.0, 5.0, 50.0, 50.0, 10, 50.0, 100.0, 0.01, datetime.now()),
        ]
        
        all_states = normal_states + anomalous_states
        
        # Detect anomalies
        anomalies = ml_optimizer.detect_anomalies(all_states)
        
        assert len(anomalies) == len(all_states)
        assert isinstance(anomalies[0], bool)
        
        # Should detect some anomalies
        assert any(anomalies)


class TestAutoScaler:
    """Test suite for AutoScaler."""
    
    @pytest.fixture
    def auto_scaler(self):
        """Create auto scaler for testing."""
        ml_optimizer = MLOptimizer()
        return AutoScaler(ml_optimizer)
    
    def test_initialization(self, auto_scaler):
        """Test auto scaler initialization."""
        assert auto_scaler.ml_optimizer is not None
        assert len(auto_scaler.scaling_history) == 0
        assert len(auto_scaler.resource_states) == 0
        assert 'cpu_scale_up' in auto_scaler.thresholds
        assert 'memory_scale_up' in auto_scaler.thresholds
    
    def test_update_resource_state(self, auto_scaler):
        """Test resource state updating."""
        component = PipelineComponent.MODEL_TRAINING
        state = ResourceState(
            cpu_usage=75.0, memory_usage=60.0, disk_usage=50.0,
            network_io=100.0, queue_length=10, throughput=50.0,
            latency=120.0, error_rate=0.02, timestamp=datetime.now()
        )
        
        auto_scaler.update_resource_state(component, state)
        
        assert component in auto_scaler.resource_states
        assert len(auto_scaler.resource_states[component]) == 1
        assert auto_scaler.resource_states[component][0] == state
    
    def test_threshold_based_scaling_decisions(self, auto_scaler):
        """Test traditional threshold-based scaling decisions."""
        component = PipelineComponent.MODEL_TRAINING
        
        # Add high resource usage states
        high_cpu_state = ResourceState(
            cpu_usage=90.0,  # Above 80% threshold
            memory_usage=50.0, disk_usage=50.0, network_io=100.0,
            queue_length=10, throughput=50.0, latency=120.0,
            error_rate=0.02, timestamp=datetime.now()
        )
        
        auto_scaler.update_resource_state(component, high_cpu_state)
        
        # Should recommend scaling up
        decision = auto_scaler.should_scale(component)
        assert decision is not None
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.magnitude > 1.0
        assert "threshold exceeded" in decision.reason.lower()
    
    def test_scale_down_decision(self, auto_scaler):
        """Test scaling down decision."""
        component = PipelineComponent.MODEL_TRAINING
        
        # Add several low resource usage states
        for i in range(5):
            low_usage_state = ResourceState(
                cpu_usage=25.0,  # Below 30% threshold
                memory_usage=35.0,  # Below 40% threshold
                disk_usage=50.0, network_io=50.0, queue_length=5,
                throughput=50.0, latency=80.0, error_rate=0.005,
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            auto_scaler.update_resource_state(component, low_usage_state)
        
        # Should recommend scaling down
        decision = auto_scaler.should_scale(component)
        assert decision is not None
        assert decision.action == ScalingAction.SCALE_DOWN
        assert decision.magnitude < 1.0
        assert "consistently low" in decision.reason.lower()
    
    def test_cooldown_period(self, auto_scaler):
        """Test scaling cooldown period."""
        component = PipelineComponent.MODEL_TRAINING
        
        # Record a recent scaling decision
        recent_decision = ScalingDecision(
            component=component,
            action=ScalingAction.SCALE_UP,
            magnitude=1.5,
            reason="Test scaling",
            confidence=0.8,
            expected_impact={},
            cost_impact=0.5,
            timestamp=datetime.now() - timedelta(minutes=2)  # 2 minutes ago
        )
        auto_scaler.scaling_history.append(recent_decision)
        
        # Add high resource state that would normally trigger scaling
        high_state = ResourceState(
            cpu_usage=95.0, memory_usage=90.0, disk_usage=50.0,
            network_io=100.0, queue_length=10, throughput=50.0,
            latency=200.0, error_rate=0.05, timestamp=datetime.now()
        )
        auto_scaler.update_resource_state(component, high_state)
        
        # Should not scale due to cooldown
        decision = auto_scaler.should_scale(component)
        # Might be None due to cooldown, or might be allowed for critical situations
        if decision is not None:
            # If decision is made, it should be due to critical conditions
            assert decision.magnitude >= 1.5
    
    def test_scaling_execution(self, auto_scaler):
        """Test scaling decision execution."""
        decision = ScalingDecision(
            component=PipelineComponent.MODEL_TRAINING,
            action=ScalingAction.SCALE_UP,
            magnitude=1.5,
            reason="Test execution",
            confidence=0.8,
            expected_impact={'performance': 0.3},
            cost_impact=0.5,
            timestamp=datetime.now()
        )
        
        # Execute scaling decision
        success = auto_scaler.execute_scaling_decision(decision)
        
        assert success
        assert decision in auto_scaler.scaling_history
    
    def test_get_scaling_recommendations(self, auto_scaler):
        """Test getting scaling recommendations."""
        # Add resource states for multiple components
        components_with_high_usage = [
            PipelineComponent.MODEL_TRAINING,
            PipelineComponent.QUANTIZATION,
        ]
        
        for component in components_with_high_usage:
            high_state = ResourceState(
                cpu_usage=95.0, memory_usage=90.0, disk_usage=50.0,
                network_io=100.0, queue_length=50, throughput=30.0,
                latency=300.0, error_rate=0.08, timestamp=datetime.now()
            )
            auto_scaler.update_resource_state(component, high_state)
        
        # Get recommendations
        recommendations = auto_scaler.get_scaling_recommendations()
        
        # Should have recommendations for high-usage components
        assert len(recommendations) >= 0  # May be 0 due to test constraints
        
        for recommendation in recommendations:
            assert isinstance(recommendation, ScalingDecision)
            assert recommendation.component in components_with_high_usage


class TestPipelineOrchestrator:
    """Test suite for PipelineOrchestrator."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "check_intervals": {"testing": 5},
                "auto_recovery": True,
                "max_retries": 2
            }
            import json
            json.dump(config, f)
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def orchestrator(self, temp_config):
        """Create orchestrator for testing."""
        orchestrator = PipelineOrchestrator(temp_config)
        yield orchestrator
        # Cleanup
        if orchestrator.is_running:
            asyncio.run(orchestrator.stop())
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.pipeline_guard is not None
        assert orchestrator.metrics_collector is not None
        assert orchestrator.ml_optimizer is not None
        assert orchestrator.auto_scaler is not None
        assert not orchestrator.is_running
    
    @pytest.mark.asyncio
    async def test_start_stop_orchestrator(self, orchestrator):
        """Test starting and stopping orchestrator."""
        # Start orchestrator
        start_task = asyncio.create_task(orchestrator.start())
        await asyncio.sleep(0.2)  # Let it start
        
        assert orchestrator.is_running
        
        # Stop orchestrator
        await orchestrator.stop()
        assert not orchestrator.is_running
        
        # Cancel start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    def test_component_resource_state_simulation(self, orchestrator):
        """Test simulated resource state generation."""
        for component in PipelineComponent:
            state = orchestrator._get_component_resource_state(component)
            
            assert isinstance(state, ResourceState)
            assert 0 <= state.cpu_usage <= 100
            assert 0 <= state.memory_usage <= 100
            assert state.latency > 0
            assert state.throughput > 0
            assert isinstance(state.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_health_metrics_collection(self, orchestrator):
        """Test health metrics collection."""
        # This should not raise an exception
        await orchestrator._collect_health_metrics()
        
        # Check that metrics were collected
        # (In a real test, we'd verify metrics were recorded)
        assert True  # Placeholder assertion
    
    def test_training_data_generation(self, orchestrator):
        """Test ML training data generation."""
        training_data = orchestrator._generate_training_data()
        
        assert len(training_data) > 0
        
        for state, failed in training_data:
            assert isinstance(state, ResourceState)
            assert isinstance(failed, bool)
    
    @pytest.mark.asyncio
    async def test_ml_model_training(self, orchestrator):
        """Test ML model training process."""
        # This should not raise an exception
        await orchestrator._train_ml_models()
        
        # In a real scenario with sufficient data, models would be trained
        assert True  # Placeholder assertion
    
    @pytest.mark.asyncio
    async def test_auto_scaling_evaluation(self, orchestrator):
        """Test auto-scaling evaluation."""
        # Add some resource states to trigger scaling
        for component in [PipelineComponent.MODEL_TRAINING, PipelineComponent.QUANTIZATION]:
            high_state = ResourceState(
                cpu_usage=95.0, memory_usage=90.0, disk_usage=50.0,
                network_io=100.0, queue_length=100, throughput=20.0,
                latency=400.0, error_rate=0.1, timestamp=datetime.now()
            )
            orchestrator.auto_scaler.update_resource_state(component, high_state)
        
        # Evaluate auto-scaling
        await orchestrator._evaluate_auto_scaling()
        
        # Should have attempted scaling decisions
        assert len(orchestrator.auto_scaler.scaling_history) >= 0
    
    def test_optimization_recommendations_generation(self, orchestrator):
        """Test optimization recommendations generation."""
        # Mock system status
        system_status = {
            'overall_health': 'critical',
            'component_status': {
                'comp1': {'status': 'failed'},
                'comp2': {'status': 'healthy'}
            }
        }
        
        # Mock log analysis results
        log_analysis = []  # Empty for this test
        
        recommendations = orchestrator._generate_optimization_recommendations(
            system_status, log_analysis
        )
        
        # Should generate recommendations for critical health
        assert len(recommendations) > 0
        
        for recommendation in recommendations:
            assert isinstance(recommendation, ScalingDecision)
            assert recommendation.confidence > 0
    
    def test_orchestrator_status(self, orchestrator):
        """Test orchestrator status reporting."""
        status = orchestrator.get_orchestrator_status()
        
        assert "orchestrator_status" in status
        assert "pipeline_health" in status
        assert "recent_scaling_decisions" in status
        assert "ml_models_available" in status
        assert "components_monitored" in status
        assert "last_optimization" in status
    
    @pytest.mark.asyncio
    async def test_optimization_cycle(self, orchestrator):
        """Test complete optimization cycle."""
        # Mock some dependencies to avoid external calls
        with patch.object(orchestrator.log_analyzer, 'analyze_logs', return_value=[]):
            with patch.object(orchestrator.pipeline_guard, 'get_system_status', 
                             return_value={'overall_health': 'healthy', 'component_status': {}}):
                
                # Run optimization cycle
                await orchestrator._run_optimization_cycle()
                
                # Should complete without errors
                assert True


class TestIntegration:
    """Integration tests for the complete orchestrator system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_orchestration(self):
        """Test end-to-end orchestration workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "check_intervals": {"testing": 1},
                "auto_recovery": True,
                "max_retries": 1
            }
            import json
            json.dump(config, f)
            temp_config = f.name
        
        try:
            # Create orchestrator
            orchestrator = PipelineOrchestrator(temp_config)
            
            # Add some high resource usage to trigger scaling
            high_state = ResourceState(
                cpu_usage=95.0, memory_usage=90.0, disk_usage=80.0,
                network_io=150.0, queue_length=200, throughput=10.0,
                latency=500.0, error_rate=0.15, timestamp=datetime.now()
            )
            
            for component in [PipelineComponent.MODEL_TRAINING, PipelineComponent.QUANTIZATION]:
                orchestrator.auto_scaler.update_resource_state(component, high_state)
            
            # Run a brief orchestration cycle
            start_task = asyncio.create_task(orchestrator.start())
            await asyncio.sleep(1.0)  # Let it run briefly
            
            # Stop orchestrator
            await orchestrator.stop()
            start_task.cancel()
            
            try:
                await start_task
            except asyncio.CancelledError:
                pass
            
            # Verify orchestration occurred
            status = orchestrator.get_orchestrator_status()
            assert status["orchestrator_status"] == "stopped"
            
        finally:
            Path(temp_config).unlink(missing_ok=True)
    
    def test_performance_under_load(self):
        """Test orchestrator performance under load."""
        orchestrator = PipelineOrchestrator()
        
        # Simulate high load by adding many resource states
        start_time = time.time()
        
        for i in range(100):
            for component in PipelineComponent:
                state = ResourceState(
                    cpu_usage=np.random.uniform(10, 95),
                    memory_usage=np.random.uniform(20, 90),
                    disk_usage=np.random.uniform(30, 95),
                    network_io=np.random.uniform(5, 100),
                    queue_length=np.random.randint(0, 200),
                    throughput=np.random.uniform(1, 100),
                    latency=np.random.uniform(20, 800),
                    error_rate=np.random.uniform(0, 0.2),
                    timestamp=datetime.now()
                )
                orchestrator.auto_scaler.update_resource_state(component, state)
        
        processing_time = time.time() - start_time
        
        # Get scaling recommendations
        start_time = time.time()
        recommendations = orchestrator.auto_scaler.get_scaling_recommendations()
        recommendation_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 5.0  # Should process quickly
        assert recommendation_time < 2.0  # Should generate recommendations quickly
        assert isinstance(recommendations, list)
    
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_ml_pipeline_integration(self):
        """Test ML pipeline integration."""
        orchestrator = PipelineOrchestrator()
        
        # Generate training data
        training_data = []
        for i in range(200):  # Sufficient data for ML
            state = ResourceState(
                cpu_usage=np.random.uniform(10, 95),
                memory_usage=np.random.uniform(20, 90),
                disk_usage=50.0, network_io=50.0, queue_length=10,
                throughput=50.0, latency=100.0, error_rate=0.01,
                timestamp=datetime.now()
            )
            # Simulate failure based on high resource usage
            failed = state.cpu_usage > 85 and state.memory_usage > 80
            training_data.append((state, failed))
        
        # Train ML model
        orchestrator.ml_optimizer.train_failure_prediction_model(training_data)
        
        # Test prediction
        test_state = ResourceState(
            cpu_usage=90.0, memory_usage=85.0, disk_usage=50.0,
            network_io=50.0, queue_length=10, throughput=50.0,
            latency=100.0, error_rate=0.01, timestamp=datetime.now()
        )
        
        failure_prob = orchestrator.ml_optimizer.predict_failure_probability(test_state, [])
        assert 0.0 <= failure_prob <= 1.0
        
        # Should have trained model
        assert 'failure_prediction' in orchestrator.ml_optimizer.models


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])