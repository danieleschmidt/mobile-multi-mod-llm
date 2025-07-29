#!/usr/bin/env python3
"""Chaos testing runner for comprehensive resilience validation.

This script orchestrates chaos engineering tests and provides detailed
reporting on system resilience characteristics.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChaosTestRunner:
    """Orchestrates chaos engineering test execution and reporting."""
    
    def __init__(self, config_path: str = "chaos-config.json"):
        self.config = self._load_config(config_path)
        self.results_path = Path("tests/chaos/results")
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load chaos testing configuration."""
        default_config = {
            "test_scenarios": [
                "memory_resilience",
                "cpu_resilience", 
                "network_resilience",
                "io_resilience",
                "failure_recovery"
            ],
            "stress_levels": ["low", "medium", "high"],
            "duration_minutes": 10,
            "parallel_executions": 4,
            "failure_threshold": 0.2,  # 20% failure rate acceptable
            "timeout_seconds": 300,
            "system_monitoring": True,
            "detailed_reporting": True
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            logger.info(f"Config file {config_path} not found, using defaults")
            return default_config
    
    def run_chaos_test_scenario(self, scenario: str, stress_level: str = "medium") -> Dict:
        """Run a specific chaos test scenario."""
        logger.info(f"Running chaos test scenario: {scenario} (stress level: {stress_level})")
        
        # Map scenarios to test files/classes
        scenario_mapping = {
            "memory_resilience": "tests/chaos/test_resilience.py::TestMemoryResilience",
            "cpu_resilience": "tests/chaos/test_resilience.py::TestCPUResilience",
            "network_resilience": "tests/chaos/test_resilience.py::TestNetworkResilience",
            "io_resilience": "tests/chaos/test_resilience.py::TestIOResilience",
            "failure_recovery": "tests/chaos/test_resilience.py::TestFailureRecovery",
            "chaos_scenarios": "tests/chaos/test_resilience.py::TestChaosScenarios"
        }
        
        test_target = scenario_mapping.get(scenario, scenario)
        
        # Set environment variables for stress level
        env_vars = {
            "CHAOS_STRESS_LEVEL": stress_level,
            "CHAOS_DURATION": str(self.config["duration_minutes"] * 60),
            "CHAOS_PARALLEL": str(self.config["parallel_executions"])
        }
        
        try:
            # Run pytest with chaos tests
            cmd = [
                "python", "-m", "pytest",
                test_target,
                "-v", "--tb=short",
                f"--timeout={self.config['timeout_seconds']}",
                "--json-report",
                f"--json-report-file={self.results_path}/scenario_{scenario}_{stress_level}.json"
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=env_vars,
                cwd="."
            )
            end_time = time.time()
            
            # Load test results
            results_file = self.results_path / f"scenario_{scenario}_{stress_level}.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    test_results = json.load(f)
            else:
                test_results = {"tests": [], "summary": {"failed": 1}}
            
            # Add execution metadata
            test_results["execution_metadata"] = {
                "scenario": scenario,
                "stress_level": stress_level,
                "duration_seconds": end_time - start_time,
                "return_code": result.returncode,
                "stdout": result.stdout[-1000:],  # Last 1000 chars
                "stderr": result.stderr[-1000:] if result.stderr else ""
            }
            
            return test_results
            
        except Exception as e:
            logger.error(f"Failed to run chaos scenario {scenario}: {e}")
            return {
                "tests": [],
                "summary": {"failed": 1, "error": str(e)},
                "execution_metadata": {
                    "scenario": scenario,
                    "stress_level": stress_level,
                    "error": str(e)
                }
            }
    
    def monitor_system_resources(self, duration_seconds: int = 60) -> Dict:
        """Monitor system resources during chaos testing."""
        logger.info(f"Monitoring system resources for {duration_seconds} seconds")
        
        measurements = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            measurement = {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "disk_io_read_mb": psutil.disk_io_counters().read_bytes / 1024 / 1024 if psutil.disk_io_counters() else 0,
                "disk_io_write_mb": psutil.disk_io_counters().write_bytes / 1024 / 1024 if psutil.disk_io_counters() else 0,
                "network_sent_mb": psutil.net_io_counters().bytes_sent / 1024 / 1024,
                "network_recv_mb": psutil.net_io_counters().bytes_recv / 1024 / 1024
            }
            measurements.append(measurement)
            time.sleep(1)
        
        # Calculate statistics
        if measurements:
            stats = {
                "duration_seconds": duration_seconds,
                "measurement_count": len(measurements),
                "cpu_stats": {
                    "avg": sum(m["cpu_percent"] for m in measurements) / len(measurements),
                    "max": max(m["cpu_percent"] for m in measurements),
                    "min": min(m["cpu_percent"] for m in measurements)
                },
                "memory_stats": {
                    "avg_percent": sum(m["memory_percent"] for m in measurements) / len(measurements),
                    "max_percent": max(m["memory_percent"] for m in measurements),
                    "min_available_mb": min(m["memory_available_mb"] for m in measurements)
                },
                "measurements": measurements
            }
        else:
            stats = {"error": "No measurements collected"}
        
        return stats
    
    def analyze_resilience_metrics(self, all_results: List[Dict]) -> Dict:
        """Analyze resilience metrics across all chaos test results."""
        logger.info("Analyzing resilience metrics")
        
        total_tests = 0
        total_failures = 0
        scenario_results = {}
        
        for result in all_results:
            scenario = result.get("execution_metadata", {}).get("scenario", "unknown")
            stress_level = result.get("execution_metadata", {}).get("stress_level", "unknown")
            
            test_count = len(result.get("tests", []))
            failure_count = result.get("summary", {}).get("failed", 0)
            
            total_tests += test_count
            total_failures += failure_count
            
            scenario_key = f"{scenario}_{stress_level}"
            scenario_results[scenario_key] = {
                "test_count": test_count,
                "failure_count": failure_count,
                "success_rate": (test_count - failure_count) / test_count if test_count > 0 else 0,
                "duration": result.get("execution_metadata", {}).get("duration_seconds", 0)
            }
        
        overall_success_rate = (total_tests - total_failures) / total_tests if total_tests > 0 else 0
        
        # Resilience score calculation (0-100)
        resilience_score = min(100, overall_success_rate * 100)
        
        # Adjust score based on stress level performance
        stress_penalty = 0
        for scenario, metrics in scenario_results.items():
            if "high" in scenario and metrics["success_rate"] < 0.8:
                stress_penalty += 10
            elif "medium" in scenario and metrics["success_rate"] < 0.9:
                stress_penalty += 5
        
        resilience_score = max(0, resilience_score - stress_penalty)
        
        return {
            "overall_resilience_score": resilience_score,
            "total_tests": total_tests,
            "total_failures": total_failures,
            "overall_success_rate": overall_success_rate,
            "scenario_breakdown": scenario_results,
            "resilience_grade": self._get_resilience_grade(resilience_score),
            "recommendations": self._generate_recommendations(scenario_results, resilience_score)
        }
    
    def _get_resilience_grade(self, score: float) -> str:
        """Convert resilience score to letter grade."""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Fair)"
        elif score >= 60:
            return "D (Poor)"
        else:
            return "F (Failing)"
    
    def _generate_recommendations(self, scenario_results: Dict, score: float) -> List[str]:
        """Generate resilience improvement recommendations."""
        recommendations = []
        
        # Analyze scenario performance
        for scenario, metrics in scenario_results.items():
            if metrics["success_rate"] < 0.8:
                if "memory" in scenario:
                    recommendations.append("Improve memory management and garbage collection")
                elif "cpu" in scenario:
                    recommendations.append("Optimize CPU-intensive operations and add throttling")
                elif "network" in scenario:
                    recommendations.append("Implement robust retry mechanisms and circuit breakers")
                elif "io" in scenario:
                    recommendations.append("Add I/O error handling and file corruption recovery")
                elif "failure" in scenario:
                    recommendations.append("Enhance failure detection and recovery procedures")
        
        # Overall score recommendations
        if score < 70:
            recommendations.append("Implement comprehensive health checks and monitoring")
            recommendations.append("Add graceful degradation mechanisms")
        
        if score < 50:
            recommendations.append("Critical: System requires significant resilience improvements")
            recommendations.append("Consider implementing bulkhead and circuit breaker patterns")
        
        return recommendations
    
    def generate_chaos_report(self, all_results: List[Dict], system_stats: Dict, metrics: Dict) -> str:
        """Generate comprehensive chaos engineering report."""
        report = ["# Chaos Engineering Test Report", ""]
        report.append(f"**Generated**: {datetime.now().isoformat()}")
        report.append(f"**Test Duration**: {self.config['duration_minutes']} minutes per scenario")
        report.append(f"**Stress Levels Tested**: {', '.join(self.config['stress_levels'])}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"**Overall Resilience Score**: {metrics['overall_resilience_score']:.1f}/100 ({metrics['resilience_grade']})")
        report.append(f"**Total Tests Executed**: {metrics['total_tests']}")
        report.append(f"**Overall Success Rate**: {metrics['overall_success_rate']:.1%}")
        report.append("")
        
        # Scenario Breakdown
        report.append("## Scenario Results")
        for scenario, results in metrics['scenario_breakdown'].items():
            status = "✅ PASS" if results['success_rate'] >= 0.8 else "❌ FAIL"
            report.append(f"### {scenario.replace('_', ' ').title()} {status}")
            report.append(f"- **Success Rate**: {results['success_rate']:.1%}")
            report.append(f"- **Tests Run**: {results['test_count']}")
            report.append(f"- **Failures**: {results['failure_count']}")
            report.append(f"- **Duration**: {results['duration']:.1f}s")
            report.append("")
        
        # System Resource Impact
        if system_stats and "cpu_stats" in system_stats:
            report.append("## System Resource Impact")
            report.append(f"**CPU Usage**: Avg {system_stats['cpu_stats']['avg']:.1f}%, Max {system_stats['cpu_stats']['max']:.1f}%")
            report.append(f"**Memory Usage**: Avg {system_stats['memory_stats']['avg_percent']:.1f}%, Max {system_stats['memory_stats']['max_percent']:.1f}%")
            report.append(f"**Minimum Available Memory**: {system_stats['memory_stats']['min_available_mb']:.0f}MB")
            report.append("")
        
        # Recommendations
        if metrics['recommendations']:
            report.append("## Recommendations")
            for rec in metrics['recommendations']:
                report.append(f"- {rec}")
            report.append("")
        
        # Detailed Test Results
        report.append("## Detailed Test Results")
        for result in all_results:
            scenario = result.get("execution_metadata", {}).get("scenario", "unknown")
            stress_level = result.get("execution_metadata", {}).get("stress_level", "unknown")
            
            report.append(f"### {scenario} ({stress_level})")
            
            if result.get("tests"):
                for test in result["tests"]:
                    status = "✅" if test.get("outcome") == "passed" else "❌"
                    test_name = test.get("nodeid", "Unknown").split("::")[-1]
                    report.append(f"- {status} {test_name}")
                    
                    if test.get("outcome") == "failed":
                        error = test.get("call", {}).get("longrepr", "Unknown error")
                        report.append(f"  - Error: {error[:200]}...")
            report.append("")
        
        return "\n".join(report)
    
    def run_comprehensive_chaos_testing(self) -> bool:
        """Run comprehensive chaos engineering test suite."""
        logger.info("Starting comprehensive chaos engineering tests")
        
        all_results = []
        system_stats = None
        
        try:
            # Start system monitoring
            if self.config["system_monitoring"]:
                import threading
                
                monitor_duration = len(self.config["test_scenarios"]) * len(self.config["stress_levels"]) * self.config["duration_minutes"] * 60
                
                def monitor_resources():
                    nonlocal system_stats
                    system_stats = self.monitor_system_resources(monitor_duration)
                
                monitor_thread = threading.Thread(target=monitor_resources)
                monitor_thread.start()
            
            # Run all chaos test scenarios
            for scenario in self.config["test_scenarios"]:
                for stress_level in self.config["stress_levels"]:
                    result = self.run_chaos_test_scenario(scenario, stress_level)
                    all_results.append(result)
            
            # Wait for monitoring to complete
            if self.config["system_monitoring"]:
                monitor_thread.join()
            
            # Analyze results
            metrics = self.analyze_resilience_metrics(all_results)
            
            # Generate report
            report = self.generate_chaos_report(all_results, system_stats, metrics)
            
            # Save report
            report_file = self.results_path / f"chaos_report_{int(time.time())}.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Chaos testing completed. Report saved to {report_file}")
            logger.info(f"Resilience Score: {metrics['overall_resilience_score']:.1f}/100 ({metrics['resilience_grade']})")
            
            # Return success if score meets threshold
            return metrics['overall_resilience_score'] >= 70
            
        except Exception as e:
            logger.error(f"Chaos testing failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chaos engineering test runner")
    parser.add_argument("--config", default="chaos-config.json",
                       help="Configuration file path")
    parser.add_argument("--scenario", 
                       help="Run specific scenario only")
    parser.add_argument("--stress-level", choices=["low", "medium", "high"],
                       default="medium", help="Stress level for testing")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report from existing results")
    
    args = parser.parse_args()
    
    runner = ChaosTestRunner(args.config)
    
    if args.report_only:
        # Load existing results and generate report
        logger.info("Generating report from existing results")
        # Implementation would load existing results
        sys.exit(0)
    
    if args.scenario:
        # Run single scenario
        result = runner.run_chaos_test_scenario(args.scenario, args.stress_level)
        success = result.get("summary", {}).get("failed", 1) == 0
        sys.exit(0 if success else 1)
    else:
        # Run comprehensive testing
        success = runner.run_comprehensive_chaos_testing()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()