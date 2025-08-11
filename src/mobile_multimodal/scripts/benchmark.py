#!/usr/bin/env python3
"""Benchmarking script for mobile multi-modal models."""

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SystemProfiler:
    """Profile system resources and hardware."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }
        
        if PSUTIL_AVAILABLE:
            # Memory information
            memory = psutil.virtual_memory()
            info.update({
                "total_memory_gb": round(memory.total / (1024**3), 2),
                "available_memory_gb": round(memory.available / (1024**3), 2),
                "memory_usage_percent": memory.percent,
            })
            
            # CPU information
            info.update({
                "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "cpu_usage_percent": psutil.cpu_percent(interval=1),
            })
        
        # GPU information
        if TORCH_AVAILABLE:
            info.update({
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            })
            
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        
        return info


class ModelBenchmarker:
    """Benchmark mobile multi-modal models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._validate_device(device)
        self.model = None
        self.mock_mode = not TORCH_AVAILABLE
        
        # Benchmark configuration
        self.warmup_iterations = 10
        self.benchmark_iterations = 100
        self.batch_sizes = [1, 4, 8, 16]
        self.image_sizes = [(224, 224), (320, 320), (512, 512)]
        
        # Results storage
        self.results = {
            "system_info": SystemProfiler.get_system_info(),
            "model_path": model_path,
            "device": self.device,
            "mock_mode": self.mock_mode,
            "benchmarks": {}
        }
    
    def _validate_device(self, device: str) -> str:
        """Validate and select optimal device."""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self) -> bool:
        """Load model for benchmarking."""
        if self.mock_mode:
            logger.warning("Running in mock mode - PyTorch not available")
            return True
        
        try:
            # Load model using the core module
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from core import MobileMultiModalLLM
            
            self.model = MobileMultiModalLLM(
                model_path=self.model_path,
                device=self.device,
                enable_optimization=True,
                optimization_profile="fast"
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def benchmark_inference(self, batch_size: int = 1, image_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
        """Benchmark inference performance."""
        logger.info(f"Benchmarking inference - batch_size: {batch_size}, image_size: {image_size}")
        
        # Create test images
        test_images = []
        for _ in range(batch_size):
            image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            test_images.append(image)
        
        # Warmup
        logger.info("Warming up...")
        for i in range(self.warmup_iterations):
            if self.mock_mode:
                time.sleep(0.01)  # Mock inference time
            else:
                try:
                    _ = self.model.generate_caption(test_images[0])
                except Exception as e:
                    logger.warning(f"Warmup iteration {i} failed: {e}")
        
        # Benchmark
        logger.info(f"Running {self.benchmark_iterations} benchmark iterations...")
        
        # Metrics
        latencies = []
        memory_usage = []
        cpu_usage = []
        
        for i in range(self.benchmark_iterations):
            # Measure system resources before
            start_memory = psutil.virtual_memory().used / (1024**2) if PSUTIL_AVAILABLE else 0
            start_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0
            
            # Inference timing
            start_time = time.time()
            
            if self.mock_mode:
                # Mock inference with realistic timing
                base_latency = 0.05  # 50ms base
                size_factor = (image_size[0] * image_size[1]) / (224 * 224)
                batch_factor = batch_size
                mock_latency = base_latency * size_factor * batch_factor
                time.sleep(mock_latency)
                result = f"Mock caption for batch {i}"
            else:
                try:
                    if batch_size == 1:
                        result = self.model.generate_caption(test_images[0])
                    else:
                        # Process batch (simplified)
                        results = []
                        for img in test_images:
                            results.append(self.model.generate_caption(img))
                        result = results[0]
                except Exception as e:
                    logger.warning(f"Benchmark iteration {i} failed: {e}")
                    continue
            
            end_time = time.time()
            
            # Measure system resources after
            end_memory = psutil.virtual_memory().used / (1024**2) if PSUTIL_AVAILABLE else 0
            end_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0
            
            # Record metrics
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            memory_usage.append(max(0, end_memory - start_memory))
            cpu_usage.append(max(0, end_cpu - start_cpu))
            
            if i % 20 == 0:
                logger.info(f"Iteration {i}: {latency_ms:.2f}ms")
        
        # Calculate statistics
        if not latencies:
            logger.error("No successful benchmark iterations")
            return {}
        
        stats = {
            "batch_size": batch_size,
            "image_size": image_size,
            "iterations": len(latencies),
            "latency": {
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
            },
            "throughput": {
                "fps": 1000.0 / np.mean(latencies),
                "images_per_second": batch_size * 1000.0 / np.mean(latencies),
            }
        }
        
        if PSUTIL_AVAILABLE:
            stats.update({
                "memory_mb": {
                    "mean": float(np.mean(memory_usage)),
                    "max": float(np.max(memory_usage)),
                },
                "cpu_percent": {
                    "mean": float(np.mean(cpu_usage)),
                    "max": float(np.max(cpu_usage)),
                }
            })
        
        return stats
    
    def benchmark_model_loading(self) -> Dict[str, Any]:
        """Benchmark model loading time."""
        logger.info("Benchmarking model loading...")
        
        loading_times = []
        
        for i in range(5):  # Load model 5 times
            start_time = time.time()
            
            if self.mock_mode:
                time.sleep(0.5)  # Mock loading time
            else:
                try:
                    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                    from core import MobileMultiModalLLM
                    
                    temp_model = MobileMultiModalLLM(
                        model_path=self.model_path,
                        device=self.device
                    )
                    del temp_model
                    
                except Exception as e:
                    logger.warning(f"Loading benchmark {i} failed: {e}")
                    continue
            
            end_time = time.time()
            loading_times.append((end_time - start_time) * 1000)
        
        if not loading_times:
            return {"error": "No successful loading attempts"}
        
        return {
            "mean_ms": float(np.mean(loading_times)),
            "std_ms": float(np.std(loading_times)),
            "min_ms": float(np.min(loading_times)),
            "max_ms": float(np.max(loading_times)),
        }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}
        
        logger.info("Benchmarking memory usage...")
        
        # Baseline memory
        baseline_memory = psutil.virtual_memory().used / (1024**2)
        
        # Load model and measure
        model_loaded_memory = baseline_memory
        if not self.mock_mode:
            try:
                # This would measure actual model loading memory
                model_loaded_memory = psutil.virtual_memory().used / (1024**2)
            except Exception as e:
                logger.warning(f"Memory measurement failed: {e}")
        else:
            model_loaded_memory = baseline_memory + 50  # Mock 50MB usage
        
        # Run inference and measure peak
        peak_memory = model_loaded_memory
        if self.model:
            for _ in range(10):
                try:
                    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    _ = self.model.generate_caption(test_image)
                    current_memory = psutil.virtual_memory().used / (1024**2)
                    peak_memory = max(peak_memory, current_memory)
                except Exception as e:
                    logger.warning(f"Memory benchmark iteration failed: {e}")
        else:
            peak_memory = model_loaded_memory + 20  # Mock additional 20MB
        
        return {
            "baseline_mb": float(baseline_memory),
            "model_loaded_mb": float(model_loaded_memory),
            "peak_inference_mb": float(peak_memory),
            "model_overhead_mb": float(model_loaded_memory - baseline_memory),
            "inference_overhead_mb": float(peak_memory - model_loaded_memory),
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        logger.info("Starting comprehensive benchmark suite...")
        
        # Load model
        if not self.load_model():
            logger.error("Failed to load model for benchmarking")
            return {"error": "Model loading failed"}
        
        # Model loading benchmark
        self.results["benchmarks"]["model_loading"] = self.benchmark_model_loading()
        
        # Memory usage benchmark
        self.results["benchmarks"]["memory_usage"] = self.benchmark_memory_usage()
        
        # Inference benchmarks
        self.results["benchmarks"]["inference"] = {}
        
        for batch_size in self.batch_sizes:
            for image_size in self.image_sizes:
                key = f"batch_{batch_size}_size_{image_size[0]}x{image_size[1]}"
                
                try:
                    benchmark_result = self.benchmark_inference(batch_size, image_size)
                    if benchmark_result:
                        self.results["benchmarks"]["inference"][key] = benchmark_result
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {key}: {e}")
                    self.results["benchmarks"]["inference"][key] = {"error": str(e)}
        
        # Add timestamp
        self.results["timestamp"] = time.time()
        self.results["benchmark_duration"] = time.time() - (self.results.get("start_time", time.time()))
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Benchmark results saved to {output_path}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("MOBILE MULTI-MODAL LLM BENCHMARK RESULTS")
        print("="*60)
        
        # System info
        system_info = self.results["system_info"]
        print(f"System: {system_info.get('platform', 'Unknown')}")
        print(f"CPU: {system_info.get('processor', 'Unknown')} ({system_info.get('cpu_count', 'Unknown')} cores)")
        print(f"Memory: {system_info.get('total_memory_gb', 'Unknown')} GB")
        print(f"Device: {self.device}")
        print(f"Mock Mode: {self.mock_mode}")
        
        # Model loading
        if "model_loading" in self.results["benchmarks"]:
            loading = self.results["benchmarks"]["model_loading"]
            if "mean_ms" in loading:
                print(f"\nModel Loading: {loading['mean_ms']:.1f}ms (±{loading['std_ms']:.1f}ms)")
        
        # Memory usage
        if "memory_usage" in self.results["benchmarks"]:
            memory = self.results["benchmarks"]["memory_usage"]
            if "model_overhead_mb" in memory:
                print(f"Memory Usage: {memory['model_overhead_mb']:.1f}MB model + {memory['inference_overhead_mb']:.1f}MB inference")
        
        # Inference performance
        print("\nInference Performance:")
        print("-" * 40)
        if "inference" in self.results["benchmarks"]:
            for key, benchmark in self.results["benchmarks"]["inference"].items():
                if "error" in benchmark:
                    print(f"{key}: ERROR - {benchmark['error']}")
                elif "latency" in benchmark:
                    latency = benchmark["latency"]
                    throughput = benchmark["throughput"]
                    print(f"{key}:")
                    print(f"  Latency: {latency['mean_ms']:.1f}ms (p95: {latency['p95_ms']:.1f}ms)")
                    print(f"  Throughput: {throughput['fps']:.1f} FPS")
        
        print("="*60)


def main():
    """Main benchmark script entry point."""
    parser = argparse.ArgumentParser(description='Benchmark mobile multi-modal models')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model file')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device for benchmarking')
    
    # Benchmark configuration
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8],
                        help='Batch sizes to benchmark')
    parser.add_argument('--image-sizes', nargs='+', default=['224x224', '320x320'],
                        help='Image sizes to benchmark (format: WIDTHxHEIGHT)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Parse image sizes
    image_sizes = []
    for size_str in args.image_sizes:
        try:
            w, h = map(int, size_str.split('x'))
            image_sizes.append((w, h))
        except ValueError:
            logger.error(f"Invalid image size format: {size_str}")
            return 1
    
    # Create benchmarker
    benchmarker = ModelBenchmarker(args.model_path, args.device)
    
    # Configure benchmark parameters
    benchmarker.batch_sizes = args.batch_sizes
    benchmarker.image_sizes = image_sizes
    benchmarker.benchmark_iterations = args.iterations
    benchmarker.warmup_iterations = args.warmup
    
    # Run benchmarks
    try:
        logger.info("Starting comprehensive benchmark...")
        results = benchmarker.run_full_benchmark()
        
        # Save results
        benchmarker.save_results(args.output)
        
        # Print summary
        benchmarker.print_summary()
        
        logger.info("Benchmark completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())