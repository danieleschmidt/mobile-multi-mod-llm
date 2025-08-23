"""Advanced Performance Benchmarking for Mobile Multi-Modal Models.

Comprehensive benchmarking suite with hardware-specific optimizations,
real-world performance analysis, and competitive baseline comparisons.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import multiprocessing as mp
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    name: str
    description: str
    benchmark_type: str  # latency, throughput, accuracy, memory, power
    hardware_targets: List[str]
    batch_sizes: List[int]
    sequence_lengths: List[int]
    num_iterations: int = 100
    warmup_iterations: int = 10
    timeout_seconds: float = 300.0
    measure_memory: bool = True
    measure_power: bool = False
    parallel_workers: int = 1

@dataclass
class HardwareSpec:
    """Hardware specification for benchmarks."""
    name: str
    cpu_model: str
    cpu_cores: int
    cpu_frequency_ghz: float
    memory_gb: int
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[int] = None
    npu_available: bool = False
    npu_model: Optional[str] = None
    thermal_design_power: Optional[int] = None

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    benchmark_id: str
    model_name: str
    hardware_spec: HardwareSpec
    config: BenchmarkConfig
    metrics: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    timestamp: float
    duration: float
    success: bool
    error_message: Optional[str] = None

class PerformanceBenchmarks:
    """Advanced performance benchmarking system."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """Initialize performance benchmarking system.
        
        Args:
            results_dir: Directory to store benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize hardware detection
        self.hardware_specs = {}
        self._detect_hardware()
        
        # Benchmark registry
        self.benchmark_registry = {}
        self._register_default_benchmarks()
        
        # Results database
        self.results_db = {}
        self._load_results_db()
        
        logger.info(f"Performance benchmarking system initialized")
    
    def _detect_hardware(self):
        """Detect current hardware specifications."""
        import platform
        import psutil
        
        # CPU information
        cpu_info = {
            "model": platform.processor() or "Unknown",
            "cores": mp.cpu_count(),
            "frequency": 0.0
        }
        
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info["frequency"] = cpu_freq.current / 1000.0  # Convert MHz to GHz
        except:
            pass
        
        # Memory information
        try:
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.total / (1024**3)
        except:
            memory_gb = 0
        
        # GPU detection
        gpu_model = None
        gpu_memory = None
        
        try:
            # Try NVIDIA GPU detection
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(', ')
                    if len(parts) >= 2:
                        gpu_model = parts[0].strip()
                        try:
                            gpu_memory = int(parts[1].strip()) / 1024  # Convert MB to GB
                        except ValueError:
                            pass
        except:
            pass
        
        # Create hardware spec
        current_spec = HardwareSpec(
            name="current_system",
            cpu_model=cpu_info["model"],
            cpu_cores=cpu_info["cores"],
            cpu_frequency_ghz=cpu_info["frequency"],
            memory_gb=memory_gb,
            gpu_model=gpu_model,
            gpu_memory_gb=gpu_memory
        )
        
        self.hardware_specs["current"] = current_spec
        logger.info(f"Detected hardware: {cpu_info['cores']} cores, {memory_gb:.1f}GB RAM")
        
        if gpu_model:
            logger.info(f"Detected GPU: {gpu_model} ({gpu_memory:.1f}GB)")
    
    def _register_default_benchmarks(self):
        """Register default benchmark suites."""
        
        # Latency benchmarks
        self.register_benchmark("single_inference_latency", self._benchmark_single_inference)
        self.register_benchmark("batch_inference_latency", self._benchmark_batch_inference)
        self.register_benchmark("cold_start_latency", self._benchmark_cold_start)
        
        # Throughput benchmarks  
        self.register_benchmark("max_throughput", self._benchmark_max_throughput)
        self.register_benchmark("sustainable_throughput", self._benchmark_sustainable_throughput)
        
        # Memory benchmarks
        self.register_benchmark("memory_usage", self._benchmark_memory_usage)
        self.register_benchmark("memory_efficiency", self._benchmark_memory_efficiency)
        
        # Accuracy benchmarks
        self.register_benchmark("accuracy_vs_speed", self._benchmark_accuracy_speed_tradeoff)
        
        logger.info("Registered default benchmarks")
    
    def register_benchmark(self, name: str, benchmark_fn: callable):
        """Register a custom benchmark.
        
        Args:
            name: Benchmark name
            benchmark_fn: Benchmark function
        """
        self.benchmark_registry[name] = benchmark_fn
        logger.info(f"Registered benchmark: {name}")
    
    def run_benchmark_suite(self, model: Any, benchmark_config: BenchmarkConfig,
                          test_data: List[Any]) -> List[BenchmarkResult]:
        """Run complete benchmark suite.
        
        Args:
            model: Model to benchmark
            benchmark_config: Benchmark configuration
            test_data: Test dataset
            
        Returns:
            List of benchmark results
        """
        results = []
        benchmark_id = self._generate_benchmark_id(benchmark_config)
        
        logger.info(f"Starting benchmark suite: {benchmark_config.name}")
        
        # Run benchmarks for each hardware target
        for hardware_target in benchmark_config.hardware_targets:
            if hardware_target not in self.hardware_specs:
                logger.warning(f"Hardware target {hardware_target} not available")
                continue
                
            hardware_spec = self.hardware_specs[hardware_target]
            
            # Run each registered benchmark
            for benchmark_name in self.benchmark_registry:
                if benchmark_config.benchmark_type != "all" and \
                   benchmark_config.benchmark_type not in benchmark_name:
                    continue
                
                try:
                    result = self._run_single_benchmark(
                        benchmark_id, benchmark_name, model, hardware_spec, 
                        benchmark_config, test_data
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Benchmark {benchmark_name} failed: {e}")
                    error_result = BenchmarkResult(
                        benchmark_id=f"{benchmark_id}_{benchmark_name}",
                        model_name=getattr(model, 'name', 'unknown'),
                        hardware_spec=hardware_spec,
                        config=benchmark_config,
                        metrics={},
                        detailed_metrics={},
                        timestamp=time.time(),
                        duration=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)
        
        # Save results
        self._save_benchmark_results(benchmark_id, results)
        
        logger.info(f"Completed benchmark suite: {benchmark_config.name}")
        return results
    
    def _generate_benchmark_id(self, config: BenchmarkConfig) -> str:
        """Generate unique benchmark ID."""
        content = f"{config.name}_{config.description}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _run_single_benchmark(self, benchmark_id: str, benchmark_name: str,
                            model: Any, hardware_spec: HardwareSpec,
                            config: BenchmarkConfig, test_data: List[Any]) -> BenchmarkResult:
        """Run a single benchmark."""
        logger.info(f"Running benchmark: {benchmark_name}")
        
        start_time = time.time()
        benchmark_fn = self.benchmark_registry[benchmark_name]
        
        try:
            # Run benchmark with timeout
            result_metrics = self._run_with_timeout(
                benchmark_fn, config.timeout_seconds, 
                model, hardware_spec, config, test_data
            )
            
            duration = time.time() - start_time
            
            result = BenchmarkResult(
                benchmark_id=f"{benchmark_id}_{benchmark_name}",
                model_name=getattr(model, 'name', 'unknown'),
                hardware_spec=hardware_spec,
                config=config,
                metrics=result_metrics.get('metrics', {}),
                detailed_metrics=result_metrics.get('detailed_metrics', {}),
                timestamp=start_time,
                duration=duration,
                success=True
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = BenchmarkResult(
                benchmark_id=f"{benchmark_id}_{benchmark_name}",
                model_name=getattr(model, 'unknown'),
                hardware_spec=hardware_spec,
                config=config,
                metrics={},
                detailed_metrics={},
                timestamp=start_time,
                duration=duration,
                success=False,
                error_message=str(e)
            )
        
        return result
    
    def _run_with_timeout(self, func: callable, timeout: float, *args, **kwargs) -> Any:
        """Run function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Benchmark timed out after {timeout} seconds")
        
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _benchmark_single_inference(self, model: Any, hardware_spec: HardwareSpec,
                                  config: BenchmarkConfig, test_data: List[Any]) -> Dict[str, Any]:
        """Benchmark single inference latency."""
        latencies = []
        memory_usage = []
        
        # Warmup
        for _ in range(config.warmup_iterations):
            if test_data:
                try:
                    _ = self._run_inference(model, test_data[0])
                except:
                    pass
        
        # Measure single inference times
        for i in range(min(config.num_iterations, len(test_data))):
            sample = test_data[i % len(test_data)]
            
            # Memory before inference
            mem_before = self._get_memory_usage()
            
            start_time = time.perf_counter()
            try:
                _ = self._run_inference(model, sample)
                end_time = time.perf_counter()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
                # Memory after inference
                mem_after = self._get_memory_usage()
                memory_usage.append(mem_after - mem_before)
                
            except Exception as e:
                logger.warning(f"Inference failed: {e}")
                continue
        
        if not latencies:
            raise RuntimeError("No successful inferences")
        
        return {
            'metrics': {
                'mean_latency_ms': np.mean(latencies),
                'p50_latency_ms': np.percentile(latencies, 50),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'std_latency_ms': np.std(latencies),
                'mean_memory_mb': np.mean(memory_usage) if memory_usage else 0
            },
            'detailed_metrics': {
                'all_latencies': latencies,
                'all_memory_usage': memory_usage,
                'successful_inferences': len(latencies),
                'total_attempts': config.num_iterations
            }
        }
    
    def _benchmark_batch_inference(self, model: Any, hardware_spec: HardwareSpec,
                                 config: BenchmarkConfig, test_data: List[Any]) -> Dict[str, Any]:
        """Benchmark batch inference performance."""
        results = {}
        
        for batch_size in config.batch_sizes:
            if batch_size > len(test_data):
                continue
                
            batch_latencies = []
            throughputs = []
            
            # Create batches
            num_batches = min(config.num_iterations, len(test_data) // batch_size)
            
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                batch = test_data[batch_start:batch_end]
                
                start_time = time.perf_counter()
                try:
                    _ = self._run_batch_inference(model, batch)
                    end_time = time.perf_counter()
                    
                    batch_latency = (end_time - start_time) * 1000  # ms
                    throughput = batch_size / (end_time - start_time)  # samples/sec
                    
                    batch_latencies.append(batch_latency)
                    throughputs.append(throughput)
                    
                except Exception as e:
                    logger.warning(f"Batch inference failed: {e}")
                    continue
            
            if batch_latencies:
                results[f'batch_size_{batch_size}'] = {
                    'mean_batch_latency_ms': np.mean(batch_latencies),
                    'mean_throughput_sps': np.mean(throughputs),
                    'p95_batch_latency_ms': np.percentile(batch_latencies, 95),
                    'max_throughput_sps': np.max(throughputs)
                }
        
        return {
            'metrics': results,
            'detailed_metrics': {
                'batch_sizes_tested': config.batch_sizes,
                'successful_batches': sum(1 for v in results.values() if v)
            }
        }
    
    def _benchmark_cold_start(self, model: Any, hardware_spec: HardwareSpec,
                            config: BenchmarkConfig, test_data: List[Any]) -> Dict[str, Any]:
        """Benchmark cold start latency."""
        cold_start_times = []
        
        for i in range(min(10, config.num_iterations)):  # Limit cold starts
            # Simulate cold start by reinitializing model components
            start_time = time.perf_counter()
            
            try:
                # Cold start simulation
                if hasattr(model, 'reset_cache'):
                    model.reset_cache()
                
                # First inference after reset
                if test_data:
                    _ = self._run_inference(model, test_data[0])
                
                end_time = time.perf_counter()
                cold_start_time = (end_time - start_time) * 1000  # ms
                cold_start_times.append(cold_start_time)
                
            except Exception as e:
                logger.warning(f"Cold start benchmark failed: {e}")
                continue
        
        if not cold_start_times:
            raise RuntimeError("No successful cold starts")
        
        return {
            'metrics': {
                'mean_cold_start_ms': np.mean(cold_start_times),
                'p95_cold_start_ms': np.percentile(cold_start_times, 95),
                'max_cold_start_ms': np.max(cold_start_times)
            },
            'detailed_metrics': {
                'all_cold_start_times': cold_start_times,
                'successful_cold_starts': len(cold_start_times)
            }
        }
    
    def _benchmark_max_throughput(self, model: Any, hardware_spec: HardwareSpec,
                                config: BenchmarkConfig, test_data: List[Any]) -> Dict[str, Any]:
        """Benchmark maximum sustainable throughput."""
        max_throughput = 0
        optimal_batch_size = 1
        
        # Test different batch sizes to find optimal throughput
        for batch_size in [1, 2, 4, 8, 16, 32]:
            if batch_size > len(test_data):
                continue
            
            # Test throughput for this batch size
            throughputs = []
            
            for _ in range(10):  # Multiple measurements
                batch = test_data[:batch_size]
                
                start_time = time.perf_counter()
                try:
                    _ = self._run_batch_inference(model, batch)
                    end_time = time.perf_counter()
                    
                    throughput = batch_size / (end_time - start_time)
                    throughputs.append(throughput)
                    
                except Exception as e:
                    logger.warning(f"Throughput test failed: {e}")
                    break
            
            if throughputs:
                avg_throughput = np.mean(throughputs)
                if avg_throughput > max_throughput:
                    max_throughput = avg_throughput
                    optimal_batch_size = batch_size
        
        return {
            'metrics': {
                'max_throughput_sps': max_throughput,
                'optimal_batch_size': optimal_batch_size
            },
            'detailed_metrics': {
                'batch_sizes_tested': [1, 2, 4, 8, 16, 32],
                'throughput_achieved': max_throughput > 0
            }
        }
    
    def _benchmark_sustainable_throughput(self, model: Any, hardware_spec: HardwareSpec,
                                        config: BenchmarkConfig, test_data: List[Any]) -> Dict[str, Any]:
        """Benchmark sustainable throughput over extended period."""
        duration_seconds = 60  # 1 minute sustained test
        batch_size = 4
        
        start_time = time.time()
        total_samples = 0
        throughputs = []
        
        while time.time() - start_time < duration_seconds:
            batch = test_data[:batch_size]
            
            batch_start = time.perf_counter()
            try:
                _ = self._run_batch_inference(model, batch)
                batch_end = time.perf_counter()
                
                batch_throughput = batch_size / (batch_end - batch_start)
                throughputs.append(batch_throughput)
                total_samples += batch_size
                
            except Exception as e:
                logger.warning(f"Sustainable throughput test failed: {e}")
                break
        
        actual_duration = time.time() - start_time
        
        return {
            'metrics': {
                'sustained_throughput_sps': total_samples / actual_duration,
                'mean_throughput_sps': np.mean(throughputs) if throughputs else 0,
                'min_throughput_sps': np.min(throughputs) if throughputs else 0,
                'throughput_std': np.std(throughputs) if throughputs else 0,
                'test_duration_seconds': actual_duration
            },
            'detailed_metrics': {
                'total_samples_processed': total_samples,
                'batch_size': batch_size,
                'throughput_measurements': len(throughputs)
            }
        }
    
    def _benchmark_memory_usage(self, model: Any, hardware_spec: HardwareSpec,
                              config: BenchmarkConfig, test_data: List[Any]) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        # Baseline memory
        baseline_memory = self._get_memory_usage()
        
        # Memory during inference
        inference_memories = []
        peak_memory = baseline_memory
        
        for i in range(min(50, len(test_data))):
            sample = test_data[i]
            
            try:
                mem_before = self._get_memory_usage()
                _ = self._run_inference(model, sample)
                mem_after = self._get_memory_usage()
                
                inference_memories.append(mem_after)
                peak_memory = max(peak_memory, mem_after)
                
            except Exception as e:
                logger.warning(f"Memory benchmark failed: {e}")
                continue
        
        # Memory efficiency calculation
        memory_overhead = peak_memory - baseline_memory
        
        return {
            'metrics': {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'memory_overhead_mb': memory_overhead,
                'mean_inference_memory_mb': np.mean(inference_memories) if inference_memories else baseline_memory
            },
            'detailed_metrics': {
                'memory_measurements': len(inference_memories),
                'memory_timeline': inference_memories
            }
        }
    
    def _benchmark_memory_efficiency(self, model: Any, hardware_spec: HardwareSpec,
                                   config: BenchmarkConfig, test_data: List[Any]) -> Dict[str, Any]:
        """Benchmark memory efficiency across different scenarios."""
        scenarios = {
            'single_inference': 1,
            'small_batch': 4,
            'medium_batch': 16,
            'large_batch': 32
        }
        
        efficiency_results = {}
        
        for scenario_name, batch_size in scenarios.items():
            if batch_size > len(test_data):
                continue
            
            # Measure memory for this scenario
            baseline = self._get_memory_usage()
            
            try:
                batch = test_data[:batch_size]
                _ = self._run_batch_inference(model, batch)
                peak = self._get_memory_usage()
                
                memory_per_sample = (peak - baseline) / batch_size
                efficiency_results[scenario_name] = {
                    'memory_per_sample_mb': memory_per_sample,
                    'total_memory_mb': peak - baseline,
                    'batch_size': batch_size
                }
                
            except Exception as e:
                logger.warning(f"Memory efficiency test failed for {scenario_name}: {e}")
                continue
        
        return {
            'metrics': efficiency_results,
            'detailed_metrics': {
                'scenarios_tested': list(scenarios.keys()),
                'successful_scenarios': len(efficiency_results)
            }
        }
    
    def _benchmark_accuracy_speed_tradeoff(self, model: Any, hardware_spec: HardwareSpec,
                                         config: BenchmarkConfig, test_data: List[Any]) -> Dict[str, Any]:
        """Benchmark accuracy vs speed tradeoffs."""
        # This would need model-specific accuracy measurement
        # For now, return basic metrics
        
        speed_metrics = []
        accuracy_proxies = []  # Confidence scores as accuracy proxy
        
        for i in range(min(100, len(test_data))):
            sample = test_data[i]
            
            start_time = time.perf_counter()
            try:
                result = self._run_inference(model, sample)
                end_time = time.perf_counter()
                
                speed = 1000 / ((end_time - start_time) * 1000)  # Inferences per second
                speed_metrics.append(speed)
                
                # Extract confidence if available
                if isinstance(result, dict) and 'confidence' in result:
                    accuracy_proxies.append(result['confidence'])
                else:
                    accuracy_proxies.append(1.0)  # Default confidence
                    
            except Exception as e:
                logger.warning(f"Accuracy-speed benchmark failed: {e}")
                continue
        
        return {
            'metrics': {
                'mean_speed_ips': np.mean(speed_metrics) if speed_metrics else 0,
                'mean_confidence': np.mean(accuracy_proxies) if accuracy_proxies else 0,
                'speed_accuracy_correlation': np.corrcoef(speed_metrics, accuracy_proxies)[0, 1] if len(speed_metrics) > 1 else 0
            },
            'detailed_metrics': {
                'speed_measurements': len(speed_metrics),
                'accuracy_measurements': len(accuracy_proxies)
            }
        }
    
    def _run_inference(self, model: Any, sample: Any) -> Any:
        """Run single inference."""
        if hasattr(model, 'generate_caption'):
            return model.generate_caption(sample)
        elif hasattr(model, '__call__'):
            return model(sample)
        else:
            # Mock inference for testing
            time.sleep(0.001)  # Simulate 1ms inference
            return {"result": "mock_output", "confidence": 0.95}
    
    def _run_batch_inference(self, model: Any, batch: List[Any]) -> List[Any]:
        """Run batch inference."""
        if hasattr(model, 'generate_caption_batch'):
            return model.generate_caption_batch(batch)
        else:
            # Fallback to individual inferences
            return [self._run_inference(model, sample) for sample in batch]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        except ImportError:
            return 0.0
    
    def _save_benchmark_results(self, benchmark_id: str, results: List[BenchmarkResult]):
        """Save benchmark results to disk."""
        results_file = self.results_dir / f"benchmark_{benchmark_id}.json"
        
        # Convert results to JSON-serializable format
        json_results = []
        for result in results:
            json_results.append({
                "benchmark_id": result.benchmark_id,
                "model_name": result.model_name,
                "hardware_spec": {
                    "name": result.hardware_spec.name,
                    "cpu_model": result.hardware_spec.cpu_model,
                    "cpu_cores": result.hardware_spec.cpu_cores,
                    "cpu_frequency_ghz": result.hardware_spec.cpu_frequency_ghz,
                    "memory_gb": result.hardware_spec.memory_gb,
                    "gpu_model": result.hardware_spec.gpu_model,
                    "gpu_memory_gb": result.hardware_spec.gpu_memory_gb
                },
                "metrics": result.metrics,
                "detailed_metrics": result.detailed_metrics,
                "timestamp": result.timestamp,
                "duration": result.duration,
                "success": result.success,
                "error_message": result.error_message
            })
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Update results database
        if benchmark_id not in self.results_db:
            self.results_db[benchmark_id] = []
        self.results_db[benchmark_id].extend(json_results)
        
        logger.info(f"Saved benchmark results to {results_file}")
    
    def _load_results_db(self):
        """Load existing benchmark results."""
        for results_file in self.results_dir.glob("benchmark_*.json"):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract benchmark ID from filename
                benchmark_id = results_file.stem.replace("benchmark_", "")
                self.results_db[benchmark_id] = results
                
            except Exception as e:
                logger.warning(f"Failed to load results from {results_file}: {e}")
    
    def generate_performance_report(self, benchmark_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if benchmark_id not in self.results_db:
            raise ValueError(f"Benchmark results not found for ID: {benchmark_id}")
        
        results = self.results_db[benchmark_id]
        
        # Aggregate metrics
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        report = {
            "benchmark_id": benchmark_id,
            "summary": {
                "total_benchmarks": len(results),
                "successful_benchmarks": len(successful_results),
                "failed_benchmarks": len(failed_results),
                "success_rate": len(successful_results) / len(results) if results else 0
            },
            "performance_metrics": {},
            "hardware_breakdown": {},
            "recommendations": []
        }
        
        # Aggregate performance metrics
        all_metrics = defaultdict(list)
        for result in successful_results:
            for metric_name, metric_value in result["metrics"].items():
                if isinstance(metric_value, (int, float)):
                    all_metrics[metric_name].append(metric_value)
                elif isinstance(metric_value, dict):
                    for sub_metric, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            all_metrics[f"{metric_name}_{sub_metric}"].append(sub_value)
        
        # Calculate aggregated statistics
        for metric_name, values in all_metrics.items():
            if values:
                report["performance_metrics"][metric_name] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "p95": np.percentile(values, 95)
                }
        
        # Hardware breakdown
        hardware_groups = defaultdict(list)
        for result in successful_results:
            hw_name = result["hardware_spec"]["name"]
            hardware_groups[hw_name].append(result)
        
        for hw_name, hw_results in hardware_groups.items():
            hw_metrics = defaultdict(list)
            for result in hw_results:
                for metric_name, metric_value in result["metrics"].items():
                    if isinstance(metric_value, (int, float)):
                        hw_metrics[metric_name].append(metric_value)
            
            hw_summary = {}
            for metric_name, values in hw_metrics.items():
                if values:
                    hw_summary[metric_name] = {
                        "mean": np.mean(values),
                        "count": len(values)
                    }
            
            report["hardware_breakdown"][hw_name] = hw_summary
        
        # Generate recommendations
        recommendations = []
        
        # Latency recommendations
        if "mean_latency_ms" in report["performance_metrics"]:
            latency = report["performance_metrics"]["mean_latency_ms"]["mean"]
            if latency > 100:
                recommendations.append("High latency detected. Consider model quantization or hardware acceleration.")
            elif latency < 10:
                recommendations.append("Excellent latency performance. Consider increasing model complexity if accuracy needs improvement.")
        
        # Throughput recommendations
        if "max_throughput_sps" in report["performance_metrics"]:
            throughput = report["performance_metrics"]["max_throughput_sps"]["mean"]
            if throughput < 10:
                recommendations.append("Low throughput detected. Consider batch processing optimization or model simplification.")
        
        # Memory recommendations
        if "memory_overhead_mb" in report["performance_metrics"]:
            memory = report["performance_metrics"]["memory_overhead_mb"]["mean"]
            if memory > 500:
                recommendations.append("High memory usage detected. Consider model compression or memory optimization techniques.")
        
        report["recommendations"] = recommendations
        
        # Save report
        report_file = self.results_dir / f"performance_report_{benchmark_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report


class CompetitiveBenchmarking:
    """Competitive benchmarking against baseline models."""
    
    def __init__(self, performance_benchmarks: PerformanceBenchmarks):
        """Initialize competitive benchmarking.
        
        Args:
            performance_benchmarks: Performance benchmarking system
        """
        self.performance_benchmarks = performance_benchmarks
        self.baseline_models = {}
        self.competitive_results = {}
    
    def register_baseline(self, name: str, model: Any, description: str = ""):
        """Register a baseline model for comparison.
        
        Args:
            name: Baseline model name
            model: Baseline model instance
            description: Model description
        """
        self.baseline_models[name] = {
            "model": model,
            "description": description,
            "registered_at": time.time()
        }
        
        logger.info(f"Registered baseline model: {name}")
    
    def run_competitive_benchmark(self, target_model: Any, 
                                benchmark_config: BenchmarkConfig,
                                test_data: List[Any]) -> Dict[str, Any]:
        """Run competitive benchmark against all baselines.
        
        Args:
            target_model: Model to benchmark
            benchmark_config: Benchmark configuration
            test_data: Test dataset
            
        Returns:
            Competitive benchmark results
        """
        logger.info("Starting competitive benchmark")
        
        competitive_id = f"competitive_{int(time.time())}"
        results = {
            "competitive_id": competitive_id,
            "target_model": getattr(target_model, 'name', 'target_model'),
            "baselines": {},
            "target_results": {},
            "comparisons": {},
            "summary": {}
        }
        
        # Benchmark target model
        logger.info("Benchmarking target model")
        target_results = self.performance_benchmarks.run_benchmark_suite(
            target_model, benchmark_config, test_data
        )
        results["target_results"] = target_results
        
        # Benchmark baseline models
        for baseline_name, baseline_info in self.baseline_models.items():
            logger.info(f"Benchmarking baseline: {baseline_name}")
            
            try:
                baseline_results = self.performance_benchmarks.run_benchmark_suite(
                    baseline_info["model"], benchmark_config, test_data
                )
                results["baselines"][baseline_name] = baseline_results
                
                # Generate comparison
                comparison = self._compare_results(target_results, baseline_results)
                results["comparisons"][baseline_name] = comparison
                
            except Exception as e:
                logger.error(f"Failed to benchmark baseline {baseline_name}: {e}")
                continue
        
        # Generate competitive summary
        results["summary"] = self._generate_competitive_summary(results)
        
        # Save competitive results
        self._save_competitive_results(competitive_id, results)
        
        logger.info("Completed competitive benchmark")
        return results
    
    def _compare_results(self, target_results: List[BenchmarkResult], 
                        baseline_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compare target results with baseline results."""
        comparison = {
            "metrics_comparison": {},
            "improvements": {},
            "regressions": {},
            "summary": {
                "total_improvements": 0,
                "total_regressions": 0,
                "net_improvement_score": 0.0
            }
        }
        
        # Group results by benchmark type
        target_metrics = self._aggregate_benchmark_metrics(target_results)
        baseline_metrics = self._aggregate_benchmark_metrics(baseline_results)
        
        # Compare each metric
        for metric_name in set(target_metrics.keys()) | set(baseline_metrics.keys()):
            if metric_name in target_metrics and metric_name in baseline_metrics:
                target_value = target_metrics[metric_name]
                baseline_value = baseline_metrics[metric_name]
                
                # Calculate improvement (positive is better)
                if baseline_value != 0:
                    improvement_percent = ((target_value - baseline_value) / abs(baseline_value)) * 100
                else:
                    improvement_percent = 0.0
                
                comparison["metrics_comparison"][metric_name] = {
                    "target_value": target_value,
                    "baseline_value": baseline_value,
                    "improvement_percent": improvement_percent,
                    "is_improvement": improvement_percent > 0
                }
                
                # Categorize improvements and regressions
                if improvement_percent > 5:  # Significant improvement threshold
                    comparison["improvements"][metric_name] = improvement_percent
                    comparison["summary"]["total_improvements"] += 1
                elif improvement_percent < -5:  # Significant regression threshold
                    comparison["regressions"][metric_name] = improvement_percent
                    comparison["summary"]["total_regressions"] += 1
                
                # Add to net improvement score
                comparison["summary"]["net_improvement_score"] += improvement_percent
        
        return comparison
    
    def _aggregate_benchmark_metrics(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """Aggregate metrics from benchmark results."""
        aggregated = {}
        metric_counts = defaultdict(int)
        
        for result in results:
            if result.success:
                for metric_name, metric_value in result.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if metric_name not in aggregated:
                            aggregated[metric_name] = 0
                        aggregated[metric_name] += metric_value
                        metric_counts[metric_name] += 1
        
        # Calculate averages
        for metric_name, total in aggregated.items():
            if metric_counts[metric_name] > 0:
                aggregated[metric_name] = total / metric_counts[metric_name]
        
        return aggregated
    
    def _generate_competitive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate competitive summary."""
        summary = {
            "overall_performance": "unknown",
            "best_metrics": [],
            "worst_metrics": [],
            "competitive_ranking": {},
            "recommendations": []
        }
        
        if not results["comparisons"]:
            return summary
        
        # Analyze overall performance
        total_improvements = 0
        total_regressions = 0
        
        for baseline_name, comparison in results["comparisons"].items():
            total_improvements += comparison["summary"]["total_improvements"]
            total_regressions += comparison["summary"]["total_regressions"]
        
        if total_improvements > total_regressions:
            summary["overall_performance"] = "better"
        elif total_improvements < total_regressions:
            summary["overall_performance"] = "worse"
        else:
            summary["overall_performance"] = "similar"
        
        # Find consistently best and worst metrics
        metric_improvements = defaultdict(list)
        
        for baseline_name, comparison in results["comparisons"].items():
            for metric_name, improvement in comparison["improvements"].items():
                metric_improvements[metric_name].append(improvement)
        
        # Metrics that show improvement across most baselines
        for metric_name, improvements in metric_improvements.items():
            if len(improvements) >= len(results["comparisons"]) * 0.7:  # 70% of baselines
                avg_improvement = np.mean(improvements)
                summary["best_metrics"].append({
                    "metric": metric_name,
                    "avg_improvement": avg_improvement
                })
        
        # Generate recommendations
        recommendations = []
        
        if summary["overall_performance"] == "better":
            recommendations.append("Model shows strong competitive performance. Consider publication of results.")
        elif summary["overall_performance"] == "worse":
            recommendations.append("Model underperforms baselines. Focus on optimization and model improvements.")
        else:
            recommendations.append("Model performance is similar to baselines. Look for specific use case advantages.")
        
        if summary["best_metrics"]:
            best_metric = max(summary["best_metrics"], key=lambda x: x["avg_improvement"])
            recommendations.append(f"Highlight {best_metric['metric']} performance as key advantage.")
        
        summary["recommendations"] = recommendations
        
        return summary
    
    def _save_competitive_results(self, competitive_id: str, results: Dict[str, Any]):
        """Save competitive benchmark results."""
        results_file = self.performance_benchmarks.results_dir / f"competitive_{competitive_id}.json"
        
        # Convert BenchmarkResult objects to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if key in ["target_results", "baselines"] and isinstance(value, list):
                json_results[key] = [self._benchmark_result_to_dict(r) for r in value]
            elif key == "baselines" and isinstance(value, dict):
                json_results[key] = {k: [self._benchmark_result_to_dict(r) for r in v] 
                                   for k, v in value.items()}
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        self.competitive_results[competitive_id] = results
        logger.info(f"Saved competitive results to {results_file}")
    
    def _benchmark_result_to_dict(self, result):
        """Convert BenchmarkResult to dictionary."""
        if isinstance(result, BenchmarkResult):
            return {
                "benchmark_id": result.benchmark_id,
                "model_name": result.model_name,
                "metrics": result.metrics,
                "detailed_metrics": result.detailed_metrics,
                "timestamp": result.timestamp,
                "duration": result.duration,
                "success": result.success,
                "error_message": result.error_message
            }
        return result