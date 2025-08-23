"""Test suite for research validation and benchmarking frameworks."""

import pytest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

try:
    from src.mobile_multimodal.research_framework import (
        ResearchFramework, ExperimentConfig, BenchmarkSuite
    )
    from src.mobile_multimodal.performance_benchmarks import (
        PerformanceBenchmarks, BenchmarkConfig, HardwareSpec, CompetitiveBenchmarking
    )
except ImportError:
    # Fallback for CI environments
    ResearchFramework = Mock
    ExperimentConfig = Mock
    BenchmarkSuite = Mock
    PerformanceBenchmarks = Mock
    BenchmarkConfig = Mock
    HardwareSpec = Mock
    CompetitiveBenchmarking = Mock

class MockModel:
    """Mock model for testing."""
    
    def __init__(self, name="mock_model"):
        self.name = name
        self._inference_count = 0
    
    def generate_caption(self, image):
        """Mock caption generation."""
        self._inference_count += 1
        return {
            "caption": f"Mock caption {self._inference_count}",
            "confidence": 0.85 + (np.random.random() * 0.1),
            "metrics": {
                "accuracy": 0.92,
                "bleu_score": 0.78,
                "processing_time": 0.05 + (np.random.random() * 0.02)
            }
        }
    
    def generate_caption_batch(self, images):
        """Mock batch caption generation."""
        return [self.generate_caption(img) for img in images]
    
    def __call__(self, input_data):
        """Make model callable."""
        return self.generate_caption(input_data)

class TestResearchFramework:
    """Test suite for research framework."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.framework = ResearchFramework(experiment_dir=self.temp_dir)
        
        # Create test data
        self.test_data = [f"test_sample_{i}" for i in range(10)]
        
        # Create mock models
        self.baseline_model = MockModel("baseline_model")
        self.novel_model = MockModel("novel_model")
    
    def test_experiment_creation(self):
        """Test experiment creation."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment for validation",
            baseline_models=["baseline_model"],
            novel_models=["novel_model"],
            datasets=["test_dataset"],
            metrics=["accuracy", "bleu_score"],
            num_runs=3
        )
        
        experiment_id = self.framework.create_experiment(config)
        
        assert experiment_id is not None
        assert len(experiment_id) == 12  # MD5 hash truncated to 12 chars
        assert experiment_id in self.framework.active_experiments
        assert experiment_id in self.framework.results_db
        
        # Verify experiment directory creation
        exp_dir = Path(self.temp_dir) / experiment_id
        assert exp_dir.exists()
        assert (exp_dir / "config.json").exists()
    
    def test_single_experiment_run(self):
        """Test single experiment run."""
        config = ExperimentConfig(
            name="single_run_test",
            description="Test single experiment run",
            baseline_models=["baseline"],
            novel_models=["novel"],
            datasets=["test"],
            metrics=["accuracy", "bleu_score"],
            num_runs=1,
            random_seed=42
        )
        
        def mock_inference(sample):
            return {
                "result": f"processed_{sample}",
                "metrics": {
                    "accuracy": 0.85,
                    "bleu_score": 0.72
                }
            }
        
        result = self.framework._run_single_experiment(
            "test_run", mock_inference, self.test_data[:5], config
        )
        
        assert result["run_id"] == "test_run"
        assert "metrics" in result
        assert "accuracy" in result["metrics"]
        assert "bleu_score" in result["metrics"]
        assert "runtime_metrics" in result
        assert result["runtime_metrics"]["total_samples"] == 5
        assert result["runtime_metrics"]["error_rate"] == 0.0
    
    def test_comparative_study(self):
        """Test comparative study execution."""
        config = ExperimentConfig(
            name="comparative_test",
            description="Test comparative study",
            baseline_models=["baseline"],
            novel_models=["novel"],
            datasets=["test"],
            metrics=["accuracy", "bleu_score"],
            num_runs=3
        )
        
        experiment_id = self.framework.create_experiment(config)
        
        def baseline_inference(sample):
            return {
                "result": f"baseline_{sample}",
                "metrics": {
                    "accuracy": 0.80 + (np.random.random() * 0.05),
                    "bleu_score": 0.70 + (np.random.random() * 0.05)
                }
            }
        
        def novel_inference(sample):
            return {
                "result": f"novel_{sample}",
                "metrics": {
                    "accuracy": 0.85 + (np.random.random() * 0.05),
                    "bleu_score": 0.75 + (np.random.random() * 0.05)
                }
            }
        
        results = self.framework.run_comparative_study(
            experiment_id, baseline_inference, novel_inference, self.test_data[:5]
        )
        
        assert "experiment_id" in results
        assert "comparative_results" in results
        assert "statistical_tests" in results
        assert "metadata" in results
        
        # Check baseline and novel results
        assert "baseline" in results["comparative_results"]
        assert "novel" in results["comparative_results"]
        
        baseline_results = results["comparative_results"]["baseline"]
        novel_results = results["comparative_results"]["novel"]
        
        assert baseline_results["num_runs"] == 3
        assert novel_results["num_runs"] == 3
        assert "metrics" in baseline_results
        assert "metrics" in novel_results
        
        # Check statistical tests
        for metric in config.metrics:
            if metric in results["statistical_tests"]:
                stat_test = results["statistical_tests"][metric]
                assert "baseline_n" in stat_test
                assert "novel_n" in stat_test
                assert "improvement" in stat_test
    
    def test_research_report_generation(self):
        """Test research report generation."""
        config = ExperimentConfig(
            name="report_test",
            description="Test report generation",
            baseline_models=["baseline"],
            novel_models=["novel"],
            datasets=["test"],
            metrics=["accuracy"],
            num_runs=2
        )
        
        experiment_id = self.framework.create_experiment(config)
        
        # Mock comparative study results
        mock_results = {
            "experiment_id": experiment_id,
            "comparative_results": {
                "baseline": {
                    "num_runs": 2,
                    "metrics": {"accuracy": {"mean": 0.80, "std": 0.02}}
                },
                "novel": {
                    "num_runs": 2,
                    "metrics": {"accuracy": {"mean": 0.85, "std": 0.02}}
                }
            },
            "statistical_tests": {
                "accuracy": {
                    "baseline_mean": 0.80,
                    "novel_mean": 0.85,
                    "improvement": 6.25,
                    "tests": {
                        "mann_whitney": {"p_value": 0.03, "significant": True}
                    }
                }
            },
            "metadata": {"test_data_size": 5, "num_runs": 2}
        }
        
        # Save mock results
        self.framework._save_experiment_results(experiment_id, mock_results)
        
        # Generate report
        report = self.framework.generate_research_report(experiment_id)
        
        assert "experiment_info" in report
        assert "methodology" in report
        assert "statistical_analysis" in report
        assert "publication_ready" in report
        
        exp_info = report["experiment_info"]
        assert exp_info["id"] == experiment_id
        assert exp_info["name"] == "report_test"
        
        methodology = report["methodology"]
        assert methodology["baseline_models"] == ["baseline"]
        assert methodology["novel_models"] == ["novel"]
        assert methodology["num_runs"] == 2
    
    def test_experiment_registry_persistence(self):
        """Test experiment registry persistence."""
        config = ExperimentConfig(
            name="persistence_test",
            description="Test persistence",
            baseline_models=["baseline"],
            novel_models=["novel"], 
            datasets=["test"],
            metrics=["accuracy"]
        )
        
        experiment_id = self.framework.create_experiment(config)
        
        # Create new framework instance to test loading
        new_framework = ResearchFramework(experiment_dir=self.temp_dir)
        
        assert experiment_id in new_framework.results_db
        assert new_framework.results_db[experiment_id]["config"]["name"] == "persistence_test"

class TestPerformanceBenchmarks:
    """Test suite for performance benchmarking."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmarks = PerformanceBenchmarks(results_dir=self.temp_dir)
        self.model = MockModel()
        self.test_data = [f"test_input_{i}" for i in range(20)]
    
    def test_hardware_detection(self):
        """Test hardware detection."""
        assert "current" in self.benchmarks.hardware_specs
        
        current_hw = self.benchmarks.hardware_specs["current"]
        assert hasattr(current_hw, 'name')
        assert hasattr(current_hw, 'cpu_cores')
        assert hasattr(current_hw, 'memory_gb')
        assert current_hw.cpu_cores > 0
    
    def test_benchmark_registration(self):
        """Test benchmark registration."""
        def custom_benchmark(model, hardware_spec, config, test_data):
            return {"metrics": {"custom_metric": 1.0}}
        
        self.benchmarks.register_benchmark("custom_test", custom_benchmark)
        
        assert "custom_test" in self.benchmarks.benchmark_registry
    
    def test_single_inference_benchmark(self):
        """Test single inference latency benchmark."""
        config = BenchmarkConfig(
            name="latency_test",
            description="Test latency benchmarking",
            benchmark_type="latency",
            hardware_targets=["current"],
            batch_sizes=[1],
            sequence_lengths=[224],
            num_iterations=5,
            warmup_iterations=2
        )
        
        hardware_spec = self.benchmarks.hardware_specs["current"]
        
        result = self.benchmarks._benchmark_single_inference(
            self.model, hardware_spec, config, self.test_data
        )
        
        assert "metrics" in result
        assert "detailed_metrics" in result
        
        metrics = result["metrics"]
        assert "mean_latency_ms" in metrics
        assert "p95_latency_ms" in metrics
        assert "std_latency_ms" in metrics
        
        assert metrics["mean_latency_ms"] > 0
        assert metrics["p95_latency_ms"] >= metrics["mean_latency_ms"]
    
    def test_batch_inference_benchmark(self):
        """Test batch inference benchmark."""
        config = BenchmarkConfig(
            name="batch_test",
            description="Test batch benchmarking", 
            benchmark_type="throughput",
            hardware_targets=["current"],
            batch_sizes=[1, 2, 4],
            sequence_lengths=[224],
            num_iterations=3
        )
        
        hardware_spec = self.benchmarks.hardware_specs["current"]
        
        result = self.benchmarks._benchmark_batch_inference(
            self.model, hardware_spec, config, self.test_data
        )
        
        assert "metrics" in result
        metrics = result["metrics"]
        
        # Should have results for each batch size
        for batch_size in config.batch_sizes:
            batch_key = f"batch_size_{batch_size}"
            if batch_key in metrics:
                assert "mean_batch_latency_ms" in metrics[batch_key]
                assert "mean_throughput_sps" in metrics[batch_key]
    
    def test_memory_usage_benchmark(self):
        """Test memory usage benchmark."""
        config = BenchmarkConfig(
            name="memory_test",
            description="Test memory benchmarking",
            benchmark_type="memory",
            hardware_targets=["current"],
            batch_sizes=[1],
            sequence_lengths=[224],
            num_iterations=10
        )
        
        hardware_spec = self.benchmarks.hardware_specs["current"]
        
        result = self.benchmarks._benchmark_memory_usage(
            self.model, hardware_spec, config, self.test_data
        )
        
        assert "metrics" in result
        metrics = result["metrics"]
        
        assert "baseline_memory_mb" in metrics
        assert "peak_memory_mb" in metrics
        assert "memory_overhead_mb" in metrics
        assert metrics["peak_memory_mb"] >= metrics["baseline_memory_mb"]
    
    def test_benchmark_suite_execution(self):
        """Test full benchmark suite execution."""
        config = BenchmarkConfig(
            name="full_suite_test",
            description="Test full benchmark suite",
            benchmark_type="all",
            hardware_targets=["current"],
            batch_sizes=[1, 2],
            sequence_lengths=[224],
            num_iterations=3,
            warmup_iterations=1
        )
        
        results = self.benchmarks.run_benchmark_suite(
            self.model, config, self.test_data
        )
        
        assert len(results) > 0
        
        # Check that we have successful results
        successful_results = [r for r in results if r.success]
        assert len(successful_results) > 0
        
        # Verify result structure
        for result in successful_results:
            assert hasattr(result, 'benchmark_id')
            assert hasattr(result, 'model_name')
            assert hasattr(result, 'hardware_spec')
            assert hasattr(result, 'metrics')
            assert hasattr(result, 'timestamp')
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        config = BenchmarkConfig(
            name="report_test",
            description="Test report generation",
            benchmark_type="latency", 
            hardware_targets=["current"],
            batch_sizes=[1],
            sequence_lengths=[224],
            num_iterations=3
        )
        
        # Run benchmark suite
        results = self.benchmarks.run_benchmark_suite(
            self.model, config, self.test_data
        )
        
        # Extract benchmark ID from first result
        if results:
            benchmark_id = results[0].benchmark_id.split('_')[0]
            
            report = self.benchmarks.generate_performance_report(benchmark_id)
            
            assert "benchmark_id" in report
            assert "summary" in report
            assert "performance_metrics" in report
            assert "recommendations" in report
            
            summary = report["summary"]
            assert "total_benchmarks" in summary
            assert "successful_benchmarks" in summary
            assert "success_rate" in summary

class TestCompetitiveBenchmarking:
    """Test suite for competitive benchmarking."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.performance_benchmarks = PerformanceBenchmarks(results_dir=self.temp_dir)
        self.competitive = CompetitiveBenchmarking(self.performance_benchmarks)
        
        # Create test models
        self.target_model = MockModel("target_model")
        self.baseline_model_1 = MockModel("baseline_1")
        self.baseline_model_2 = MockModel("baseline_2")
        
        self.test_data = [f"test_{i}" for i in range(10)]
    
    def test_baseline_registration(self):
        """Test baseline model registration."""
        self.competitive.register_baseline(
            "baseline_1", self.baseline_model_1, "First baseline model"
        )
        
        assert "baseline_1" in self.competitive.baseline_models
        
        baseline_info = self.competitive.baseline_models["baseline_1"]
        assert baseline_info["model"] == self.baseline_model_1
        assert baseline_info["description"] == "First baseline model"
    
    def test_competitive_benchmark_execution(self):
        """Test competitive benchmark execution."""
        # Register baselines
        self.competitive.register_baseline("baseline_1", self.baseline_model_1)
        self.competitive.register_baseline("baseline_2", self.baseline_model_2)
        
        config = BenchmarkConfig(
            name="competitive_test",
            description="Test competitive benchmarking",
            benchmark_type="latency",
            hardware_targets=["current"],
            batch_sizes=[1],
            sequence_lengths=[224],
            num_iterations=3
        )
        
        # Mock the benchmark suite results
        with patch.object(self.performance_benchmarks, 'run_benchmark_suite') as mock_benchmark:
            # Mock successful benchmark results
            mock_result = Mock()
            mock_result.success = True
            mock_result.metrics = {
                "mean_latency_ms": 50.0,
                "throughput_sps": 20.0
            }
            mock_benchmark.return_value = [mock_result]
            
            results = self.competitive.run_competitive_benchmark(
                self.target_model, config, self.test_data
            )
            
            assert "competitive_id" in results
            assert "target_model" in results
            assert "baselines" in results
            assert "target_results" in results
            assert "comparisons" in results
            assert "summary" in results
            
            # Should have called benchmark suite for target + baselines
            assert mock_benchmark.call_count == 3  # target + 2 baselines
    
    def test_results_comparison(self):
        """Test results comparison logic."""
        # Mock benchmark results
        target_result = Mock()
        target_result.success = True
        target_result.metrics = {
            "mean_latency_ms": 40.0,
            "throughput_sps": 25.0
        }
        
        baseline_result = Mock()
        baseline_result.success = True  
        baseline_result.metrics = {
            "mean_latency_ms": 50.0,
            "throughput_sps": 20.0
        }
        
        comparison = self.competitive._compare_results(
            [target_result], [baseline_result]
        )
        
        assert "metrics_comparison" in comparison
        assert "improvements" in comparison
        assert "summary" in comparison
        
        # Check latency comparison (lower is better, so -20% is improvement)
        if "mean_latency_ms" in comparison["metrics_comparison"]:
            latency_comp = comparison["metrics_comparison"]["mean_latency_ms"]
            assert latency_comp["target_value"] == 40.0
            assert latency_comp["baseline_value"] == 50.0
            assert latency_comp["improvement_percent"] < 0  # Improvement in latency
        
        # Check throughput comparison (higher is better, so +25% is improvement)
        if "throughput_sps" in comparison["metrics_comparison"]:
            throughput_comp = comparison["metrics_comparison"]["throughput_sps"]
            assert throughput_comp["target_value"] == 25.0
            assert throughput_comp["baseline_value"] == 20.0
            assert throughput_comp["improvement_percent"] > 0  # Improvement in throughput

class TestBenchmarkSuite:
    """Test suite for benchmark suite integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.research_framework = ResearchFramework(experiment_dir=self.temp_dir)
        self.benchmark_suite = BenchmarkSuite(self.research_framework)
        
        self.model = MockModel()
        self.test_data = [f"data_{i}" for i in range(5)]
    
    def test_benchmark_registration(self):
        """Test benchmark registration in suite."""
        def custom_benchmark(model, sample):
            return {"accuracy": 0.9, "latency": 0.05}
        
        def baseline_benchmark(sample):
            return {"accuracy": 0.8, "latency": 0.06}
        
        self.benchmark_suite.register_benchmark(
            "custom_test", custom_benchmark, baseline_benchmark
        )
        
        assert "custom_test" in self.benchmark_suite.benchmark_registry
        
        benchmark_info = self.benchmark_suite.benchmark_registry["custom_test"]
        assert benchmark_info["function"] == custom_benchmark
        assert benchmark_info["baseline"] == baseline_benchmark
    
    def test_benchmark_execution(self):
        """Test benchmark execution through suite."""
        def accuracy_benchmark(model, sample):
            result = model.generate_caption(sample)
            return {"accuracy": result["metrics"]["accuracy"]}
        
        self.benchmark_suite.register_benchmark("accuracy_test", accuracy_benchmark)
        
        config = ExperimentConfig(
            name="suite_test",
            description="Test benchmark suite",
            baseline_models=["baseline"],
            novel_models=["novel"],
            datasets=["test"],
            metrics=["accuracy"],
            num_runs=2
        )
        
        results = self.benchmark_suite.run_benchmark(
            "accuracy_test", self.model, self.test_data, config
        )
        
        # Should return experiment results
        assert "run_id" in results
        assert "metrics" in results

class TestIntegrationValidation:
    """Integration tests for research validation pipeline."""
    
    def setup_method(self):
        """Setup integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.research_framework = ResearchFramework(experiment_dir=self.temp_dir)
        self.performance_benchmarks = PerformanceBenchmarks(results_dir=self.temp_dir)
        self.competitive_benchmarking = CompetitiveBenchmarking(self.performance_benchmarks)
        
        # Create test models
        self.target_model = MockModel("advanced_model") 
        self.baseline_model = MockModel("baseline_model")
        
        self.test_data = [f"sample_{i}" for i in range(15)]
    
    def test_end_to_end_research_pipeline(self):
        """Test complete research pipeline."""
        # 1. Register baseline for competitive analysis
        self.competitive_benchmarking.register_baseline(
            "standard_baseline", self.baseline_model, "Standard baseline model"
        )
        
        # 2. Create research experiment
        research_config = ExperimentConfig(
            name="end_to_end_validation",
            description="Complete research validation pipeline",
            baseline_models=["standard_baseline"],
            novel_models=["advanced_model"], 
            datasets=["validation_set"],
            metrics=["accuracy", "bleu_score"],
            num_runs=3,
            significance_level=0.05
        )
        
        experiment_id = self.research_framework.create_experiment(research_config)
        
        # 3. Define inference functions
        def baseline_inference(sample):
            return self.baseline_model.generate_caption(sample)
        
        def novel_inference(sample):
            return self.target_model.generate_caption(sample)
        
        # 4. Run comparative study
        research_results = self.research_framework.run_comparative_study(
            experiment_id, baseline_inference, novel_inference, self.test_data
        )
        
        # 5. Generate research report
        research_report = self.research_framework.generate_research_report(experiment_id)
        
        # 6. Run performance benchmarks
        perf_config = BenchmarkConfig(
            name="performance_validation",
            description="Performance validation for research",
            benchmark_type="latency",
            hardware_targets=["current"],
            batch_sizes=[1, 4],
            sequence_lengths=[224],
            num_iterations=10
        )
        
        perf_results = self.performance_benchmarks.run_benchmark_suite(
            self.target_model, perf_config, self.test_data
        )
        
        # 7. Run competitive benchmark
        competitive_results = self.competitive_benchmarking.run_competitive_benchmark(
            self.target_model, perf_config, self.test_data
        )
        
        # Validation assertions
        assert research_results is not None
        assert "comparative_results" in research_results
        assert "statistical_tests" in research_results
        
        assert research_report is not None
        assert "experiment_info" in research_report
        assert "methodology" in research_report
        assert "statistical_analysis" in research_report
        
        assert len(perf_results) > 0
        assert any(r.success for r in perf_results)
        
        assert competitive_results is not None
        assert "comparisons" in competitive_results
        assert "summary" in competitive_results
        
        # Verify files were created
        exp_dir = Path(self.temp_dir) / experiment_id
        assert exp_dir.exists()
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "results.json").exists()
        assert (exp_dir / "research_report.json").exists()
    
    def test_statistical_significance_validation(self):
        """Test statistical significance validation."""
        # Create experiment with controlled data
        config = ExperimentConfig(
            name="significance_test",
            description="Test statistical significance",
            baseline_models=["baseline"],
            novel_models=["novel"],
            datasets=["test"], 
            metrics=["accuracy"],
            num_runs=5,
            significance_level=0.05
        )
        
        experiment_id = self.research_framework.create_experiment(config)
        
        # Create controlled inference functions with known differences
        def baseline_inference(sample):
            # Baseline with lower, more variable performance
            accuracy = 0.75 + (np.random.normal(0, 0.05))
            return {"metrics": {"accuracy": max(0, min(1, accuracy))}}
        
        def novel_inference(sample):
            # Novel approach with higher, more consistent performance
            accuracy = 0.85 + (np.random.normal(0, 0.02))
            return {"metrics": {"accuracy": max(0, min(1, accuracy))}}
        
        results = self.research_framework.run_comparative_study(
            experiment_id, baseline_inference, novel_inference, self.test_data
        )
        
        # Verify statistical tests were conducted
        assert "statistical_tests" in results
        if "accuracy" in results["statistical_tests"]:
            stat_test = results["statistical_tests"]["accuracy"]
            assert "baseline_mean" in stat_test
            assert "novel_mean" in stat_test
            assert "improvement" in stat_test
            assert "tests" in stat_test
            
            # Novel approach should show improvement
            assert stat_test["novel_mean"] > stat_test["baseline_mean"]
            assert stat_test["improvement"] > 0
    
    @pytest.mark.slow
    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        # This test would be marked as slow and run in extended test suites
        
        # Create baseline performance benchmark
        baseline_config = BenchmarkConfig(
            name="regression_baseline",
            description="Baseline for regression detection",
            benchmark_type="latency",
            hardware_targets=["current"],
            batch_sizes=[1],
            sequence_lengths=[224],
            num_iterations=20
        )
        
        # Simulate fast baseline model
        fast_model = MockModel("fast_baseline")
        fast_results = self.performance_benchmarks.run_benchmark_suite(
            fast_model, baseline_config, self.test_data
        )
        
        # Simulate slower model (regression)
        class SlowModel(MockModel):
            def generate_caption(self, image):
                time.sleep(0.01)  # Add artificial delay
                return super().generate_caption(image)
        
        slow_model = SlowModel("slow_model")
        slow_results = self.performance_benchmarks.run_benchmark_suite(
            slow_model, baseline_config, self.test_data
        )
        
        # Verify regression detection
        fast_latencies = []
        slow_latencies = []
        
        for result in fast_results:
            if result.success and "mean_latency_ms" in result.metrics:
                fast_latencies.append(result.metrics["mean_latency_ms"])
        
        for result in slow_results:
            if result.success and "mean_latency_ms" in result.metrics:
                slow_latencies.append(result.metrics["mean_latency_ms"])
        
        if fast_latencies and slow_latencies:
            avg_fast = np.mean(fast_latencies)
            avg_slow = np.mean(slow_latencies)
            
            # Slow model should have higher latency (regression)
            assert avg_slow > avg_fast, f"Expected regression: {avg_slow} > {avg_fast}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])