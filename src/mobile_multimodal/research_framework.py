"""Research Framework for Mobile Multi-Modal LLM Experiments.

Advanced research framework for conducting controlled experiments, baseline comparisons,
statistical validation, and academic publication preparation.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import pickle
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import threading
import queue

# Statistical and research libraries
try:
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    stats = None
    plt = None
    sns = None

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for research experiments."""
    name: str
    description: str
    baseline_models: List[str]
    novel_models: List[str] 
    datasets: List[str]
    metrics: List[str]
    num_runs: int = 5
    significance_level: float = 0.05
    random_seed: int = 42
    hardware_targets: List[str] = None
    
    def __post_init__(self):
        if self.hardware_targets is None:
            self.hardware_targets = ["snapdragon_8gen3", "apple_a17", "generic_arm"]

@dataclass
class ExperimentResult:
    """Results from a single experimental run."""
    experiment_id: str
    model_name: str
    dataset: str
    metrics: Dict[str, float]
    runtime_metrics: Dict[str, float]
    timestamp: float
    hardware_info: Dict[str, Any]
    model_size_mb: float
    
class ResearchFramework:
    """Advanced research framework for mobile AI experiments."""
    
    def __init__(self, experiment_dir: str = "experiments"):
        """Initialize research framework.
        
        Args:
            experiment_dir: Directory to store experiment results
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.results_db = {}
        self.active_experiments = {}
        
        # Setup experiment tracking
        self.experiment_registry = self.experiment_dir / "experiment_registry.json"
        self._load_experiment_registry()
        
        # Statistical analysis components
        self.significance_tests = {
            "wilcoxon": self._wilcoxon_test,
            "mannwhitney": self._mannwhitney_test,
            "ttest": self._ttest_test,
            "bootstrap": self._bootstrap_test
        }
        
        logger.info(f"Research framework initialized with experiment dir: {experiment_dir}")
    
    def _load_experiment_registry(self):
        """Load existing experiment registry."""
        if self.experiment_registry.exists():
            with open(self.experiment_registry, 'r') as f:
                self.results_db = json.load(f)
        else:
            self.results_db = {}
    
    def _save_experiment_registry(self):
        """Save experiment registry to disk."""
        with open(self.experiment_registry, 'w') as f:
            json.dump(self.results_db, f, indent=2)
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new research experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        experiment_id = self._generate_experiment_id(config)
        
        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "name": config.name,
                "description": config.description,
                "baseline_models": config.baseline_models,
                "novel_models": config.novel_models,
                "datasets": config.datasets,
                "metrics": config.metrics,
                "num_runs": config.num_runs,
                "significance_level": config.significance_level,
                "random_seed": config.random_seed,
                "hardware_targets": config.hardware_targets
            }, f, indent=2)
        
        # Initialize experiment in registry
        self.results_db[experiment_id] = {
            "config": config.__dict__,
            "status": "created",
            "created_at": time.time(),
            "results": []
        }
        
        self._save_experiment_registry()
        self.active_experiments[experiment_id] = config
        
        logger.info(f"Created experiment {experiment_id}: {config.name}")
        return experiment_id
    
    def _generate_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID."""
        content = f"{config.name}_{config.description}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def run_comparative_study(self, experiment_id: str, 
                            baseline_inference_fn: callable,
                            novel_inference_fn: callable,
                            test_data: List[Any]) -> Dict[str, Any]:
        """Run comparative study between baseline and novel approaches.
        
        Args:
            experiment_id: Experiment identifier
            baseline_inference_fn: Function for baseline inference
            novel_inference_fn: Function for novel inference
            test_data: Test dataset
            
        Returns:
            Comparative results with statistical significance
        """
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.active_experiments[experiment_id]
        results = {
            "experiment_id": experiment_id,
            "comparative_results": {},
            "statistical_tests": {},
            "metadata": {
                "test_data_size": len(test_data),
                "num_runs": config.num_runs,
                "timestamp": time.time()
            }
        }
        
        logger.info(f"Starting comparative study for experiment {experiment_id}")
        
        # Run baseline experiments
        baseline_results = []
        for run in range(config.num_runs):
            logger.info(f"Running baseline experiment {run + 1}/{config.num_runs}")
            run_results = self._run_single_experiment(
                f"baseline_run_{run}", baseline_inference_fn, test_data, config
            )
            baseline_results.append(run_results)
        
        # Run novel approach experiments
        novel_results = []
        for run in range(config.num_runs):
            logger.info(f"Running novel experiment {run + 1}/{config.num_runs}")
            run_results = self._run_single_experiment(
                f"novel_run_{run}", novel_inference_fn, test_data, config
            )
            novel_results.append(run_results)
        
        # Aggregate results
        baseline_aggregated = self._aggregate_results(baseline_results, "baseline")
        novel_aggregated = self._aggregate_results(novel_results, "novel")
        
        results["comparative_results"]["baseline"] = baseline_aggregated
        results["comparative_results"]["novel"] = novel_aggregated
        
        # Perform statistical tests
        for metric in config.metrics:
            baseline_values = [r["metrics"][metric] for r in baseline_results if metric in r["metrics"]]
            novel_values = [r["metrics"][metric] for r in novel_results if metric in r["metrics"]]
            
            if len(baseline_values) >= 3 and len(novel_values) >= 3:
                stat_results = self._perform_statistical_tests(
                    baseline_values, novel_values, metric, config.significance_level
                )
                results["statistical_tests"][metric] = stat_results
        
        # Save results
        self._save_experiment_results(experiment_id, results)
        
        logger.info(f"Comparative study completed for experiment {experiment_id}")
        return results
    
    def _run_single_experiment(self, run_id: str, inference_fn: callable, 
                             test_data: List[Any], config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experimental trial."""
        start_time = time.time()
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed + hash(run_id) % 1000)
        
        # Initialize metrics collection
        metrics = defaultdict(list)
        runtime_metrics = {
            "total_samples": len(test_data),
            "processing_times": [],
            "memory_usage": [],
            "errors": 0
        }
        
        # Process test data
        for i, sample in enumerate(test_data):
            try:
                sample_start = time.time()
                
                # Run inference
                result = inference_fn(sample)
                
                sample_time = time.time() - sample_start
                runtime_metrics["processing_times"].append(sample_time)
                
                # Collect metrics if result contains them
                if isinstance(result, dict) and "metrics" in result:
                    for metric_name, value in result["metrics"].items():
                        if metric_name in config.metrics:
                            metrics[metric_name].append(value)
                
            except Exception as e:
                logger.warning(f"Error in sample {i}: {e}")
                runtime_metrics["errors"] += 1
        
        # Aggregate metrics
        aggregated_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                aggregated_metrics[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
        
        # Calculate runtime statistics
        processing_times = runtime_metrics["processing_times"]
        runtime_stats = {
            "mean_processing_time": np.mean(processing_times) if processing_times else 0,
            "std_processing_time": np.std(processing_times) if processing_times else 0,
            "total_time": time.time() - start_time,
            "throughput": len(test_data) / (time.time() - start_time),
            "error_rate": runtime_metrics["errors"] / len(test_data)
        }
        
        return {
            "run_id": run_id,
            "metrics": {k: v["mean"] for k, v in aggregated_metrics.items()},
            "detailed_metrics": aggregated_metrics,
            "runtime_metrics": runtime_stats,
            "timestamp": start_time
        }
    
    def _aggregate_results(self, results: List[Dict], model_type: str) -> Dict[str, Any]:
        """Aggregate results across multiple runs."""
        if not results:
            return {}
        
        # Aggregate metrics
        all_metrics = defaultdict(list)
        all_runtime = defaultdict(list)
        
        for result in results:
            for metric, value in result["metrics"].items():
                all_metrics[metric].append(value)
            for metric, value in result["runtime_metrics"].items():
                if isinstance(value, (int, float)):
                    all_runtime[metric].append(value)
        
        aggregated = {
            "model_type": model_type,
            "num_runs": len(results),
            "metrics": {},
            "runtime_metrics": {}
        }
        
        # Statistical aggregation for metrics
        for metric, values in all_metrics.items():
            aggregated["metrics"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
                "values": values  # Keep raw values for statistical tests
            }
        
        # Statistical aggregation for runtime metrics
        for metric, values in all_runtime.items():
            aggregated["runtime_metrics"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        return aggregated
    
    def _perform_statistical_tests(self, baseline_values: List[float], 
                                 novel_values: List[float], 
                                 metric_name: str,
                                 significance_level: float) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        if not stats:
            logger.warning("scipy.stats not available, skipping statistical tests")
            return {"error": "scipy.stats not available"}
        
        test_results = {
            "metric": metric_name,
            "baseline_n": len(baseline_values),
            "novel_n": len(novel_values),
            "baseline_mean": np.mean(baseline_values),
            "novel_mean": np.mean(novel_values),
            "improvement": (np.mean(novel_values) - np.mean(baseline_values)) / np.mean(baseline_values) * 100,
            "significance_level": significance_level,
            "tests": {}
        }
        
        # Mann-Whitney U test (non-parametric)
        try:
            statistic, p_value = stats.mannwhitneyu(novel_values, baseline_values, alternative='two-sided')
            test_results["tests"]["mann_whitney"] = {
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < significance_level,
                "effect_size": self._calculate_effect_size(baseline_values, novel_values)
            }
        except Exception as e:
            logger.warning(f"Mann-Whitney U test failed: {e}")
        
        # Wilcoxon signed-rank test (if paired data)
        if len(baseline_values) == len(novel_values):
            try:
                statistic, p_value = stats.wilcoxon(novel_values, baseline_values, alternative='two-sided')
                test_results["tests"]["wilcoxon"] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "significant": p_value < significance_level
                }
            except Exception as e:
                logger.warning(f"Wilcoxon test failed: {e}")
        
        # Bootstrap confidence interval
        try:
            diff_ci = self._bootstrap_confidence_interval(baseline_values, novel_values)
            test_results["tests"]["bootstrap"] = {
                "confidence_interval": diff_ci,
                "significant": not (diff_ci[0] <= 0 <= diff_ci[1])  # 0 not in CI
            }
        except Exception as e:
            logger.warning(f"Bootstrap test failed: {e}")
        
        return test_results
    
    def _calculate_effect_size(self, baseline: List[float], novel: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline) + 
                             (len(novel) - 1) * np.var(novel)) / 
                            (len(baseline) + len(novel) - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(novel) - np.mean(baseline)) / pooled_std
    
    def _bootstrap_confidence_interval(self, baseline: List[float], novel: List[float], 
                                     n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for difference in means."""
        baseline_array = np.array(baseline)
        novel_array = np.array(novel)
        
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            baseline_sample = np.random.choice(baseline_array, size=len(baseline_array), replace=True)
            novel_sample = np.random.choice(novel_array, size=len(novel_array), replace=True)
            diff = np.mean(novel_sample) - np.mean(baseline_sample)
            bootstrap_diffs.append(diff)
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return (np.percentile(bootstrap_diffs, lower_percentile),
                np.percentile(bootstrap_diffs, upper_percentile))
    
    def _save_experiment_results(self, experiment_id: str, results: Dict[str, Any]):
        """Save experiment results to disk."""
        exp_dir = self.experiment_dir / experiment_id
        results_file = exp_dir / "results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Update registry
        if experiment_id in self.results_db:
            self.results_db[experiment_id]["results"].append(results)
            self.results_db[experiment_id]["status"] = "completed"
            self.results_db[experiment_id]["completed_at"] = time.time()
        
        self._save_experiment_registry()
    
    def generate_research_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate comprehensive research report for publication."""
        if experiment_id not in self.results_db:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp_data = self.results_db[experiment_id]
        exp_dir = self.experiment_dir / experiment_id
        
        # Load detailed results
        results_file = exp_dir / "results.json"
        if not results_file.exists():
            raise ValueError(f"Results file not found for experiment {experiment_id}")
        
        with open(results_file, 'r') as f:
            detailed_results = json.load(f)
        
        report = {
            "experiment_info": {
                "id": experiment_id,
                "name": exp_data["config"]["name"],
                "description": exp_data["config"]["description"],
                "created_at": exp_data["created_at"],
                "completed_at": exp_data.get("completed_at"),
                "duration": exp_data.get("completed_at", time.time()) - exp_data["created_at"]
            },
            "methodology": {
                "baseline_models": exp_data["config"]["baseline_models"],
                "novel_models": exp_data["config"]["novel_models"],
                "datasets": exp_data["config"]["datasets"],
                "metrics": exp_data["config"]["metrics"],
                "num_runs": exp_data["config"]["num_runs"],
                "significance_level": exp_data["config"]["significance_level"]
            },
            "results_summary": {},
            "statistical_analysis": {},
            "publication_ready": {
                "reproducibility_info": self._generate_reproducibility_info(exp_data),
                "code_availability": self._check_code_availability(exp_dir),
                "data_availability": self._check_data_availability(exp_dir)
            }
        }
        
        # Process statistical results
        if "statistical_tests" in detailed_results:
            report["statistical_analysis"] = detailed_results["statistical_tests"]
            report["results_summary"]["significant_improvements"] = []
            
            for metric, test_results in detailed_results["statistical_tests"].items():
                if any(test["significant"] for test in test_results["tests"].values() if isinstance(test, dict)):
                    report["results_summary"]["significant_improvements"].append({
                        "metric": metric,
                        "improvement_percent": test_results["improvement"],
                        "baseline_mean": test_results["baseline_mean"],
                        "novel_mean": test_results["novel_mean"]
                    })
        
        # Generate visualizations if matplotlib available
        if plt:
            self._generate_research_plots(experiment_id, detailed_results, exp_dir)
            report["visualizations"] = ["comparison_plot.png", "statistical_significance.png"]
        
        # Save report
        report_file = exp_dir / "research_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Research report generated for experiment {experiment_id}")
        return report
    
    def _generate_reproducibility_info(self, exp_data: Dict) -> Dict[str, Any]:
        """Generate reproducibility information for research publication."""
        return {
            "random_seed": exp_data["config"]["random_seed"],
            "framework_version": "1.0.0",
            "python_version": "3.10+",
            "hardware_requirements": exp_data["config"]["hardware_targets"],
            "dependencies": [
                "torch>=2.3.0",
                "numpy>=1.24.0",
                "scipy>=1.10.0"
            ]
        }
    
    def _check_code_availability(self, exp_dir: Path) -> bool:
        """Check if code is available and documented."""
        required_files = ["config.json", "results.json"]
        return all((exp_dir / f).exists() for f in required_files)
    
    def _check_data_availability(self, exp_dir: Path) -> bool:
        """Check if experimental data is properly stored."""
        return (exp_dir / "results.json").exists()
    
    def _generate_research_plots(self, experiment_id: str, results: Dict, exp_dir: Path):
        """Generate publication-ready plots."""
        if not plt:
            return
        
        try:
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Research Results: {experiment_id}', fontsize=14)
            
            # Plot 1: Metric comparison
            if "comparative_results" in results:
                baseline = results["comparative_results"].get("baseline", {})
                novel = results["comparative_results"].get("novel", {})
                
                if "metrics" in baseline and "metrics" in novel:
                    metrics = list(baseline["metrics"].keys())
                    baseline_means = [baseline["metrics"][m]["mean"] for m in metrics]
                    novel_means = [novel["metrics"][m]["mean"] for m in metrics]
                    
                    x = np.arange(len(metrics))
                    width = 0.35
                    
                    axes[0, 0].bar(x - width/2, baseline_means, width, label='Baseline', alpha=0.8)
                    axes[0, 0].bar(x + width/2, novel_means, width, label='Novel', alpha=0.8)
                    axes[0, 0].set_xlabel('Metrics')
                    axes[0, 0].set_ylabel('Values')
                    axes[0, 0].set_title('Performance Comparison')
                    axes[0, 0].set_xticks(x)
                    axes[0, 0].set_xticklabels(metrics, rotation=45)
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
            
            # Save plots
            plt.tight_layout()
            plt.savefig(exp_dir / "comparison_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        experiments = []
        for exp_id, exp_data in self.results_db.items():
            experiments.append({
                "id": exp_id,
                "name": exp_data["config"]["name"],
                "status": exp_data["status"],
                "created_at": exp_data["created_at"],
                "num_results": len(exp_data.get("results", []))
            })
        return experiments


class BenchmarkSuite:
    """Comprehensive benchmarking suite for mobile AI models."""
    
    def __init__(self, research_framework: ResearchFramework):
        """Initialize benchmark suite.
        
        Args:
            research_framework: Research framework instance
        """
        self.research_framework = research_framework
        self.benchmark_registry = {}
        
    def register_benchmark(self, name: str, benchmark_fn: callable, 
                         baseline_fn: callable = None):
        """Register a new benchmark.
        
        Args:
            name: Benchmark name
            benchmark_fn: Function to run benchmark
            baseline_fn: Optional baseline function for comparison
        """
        self.benchmark_registry[name] = {
            "function": benchmark_fn,
            "baseline": baseline_fn,
            "registered_at": time.time()
        }
        
        logger.info(f"Registered benchmark: {name}")
    
    def run_benchmark(self, name: str, model: Any, test_data: List[Any], 
                     config: ExperimentConfig) -> Dict[str, Any]:
        """Run a specific benchmark.
        
        Args:
            name: Benchmark name
            model: Model to benchmark
            test_data: Test dataset
            config: Experiment configuration
            
        Returns:
            Benchmark results
        """
        if name not in self.benchmark_registry:
            raise ValueError(f"Benchmark {name} not registered")
        
        benchmark_info = self.benchmark_registry[name]
        benchmark_fn = benchmark_info["function"]
        
        logger.info(f"Running benchmark: {name}")
        
        # Create inference function for the model
        def inference_fn(sample):
            return benchmark_fn(model, sample)
        
        # Create baseline inference if available
        baseline_fn = None
        if benchmark_info["baseline"]:
            def baseline_inference_fn(sample):
                return benchmark_info["baseline"](sample)
            baseline_fn = baseline_inference_fn
        
        # Run experiment
        if baseline_fn:
            experiment_id = self.research_framework.create_experiment(config)
            results = self.research_framework.run_comparative_study(
                experiment_id, baseline_fn, inference_fn, test_data
            )
        else:
            # Run single model benchmark
            results = self.research_framework._run_single_experiment(
                f"benchmark_{name}", inference_fn, test_data, config
            )
        
        return results