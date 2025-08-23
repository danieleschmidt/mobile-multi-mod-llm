#!/usr/bin/env python3
"""Simple Monitoring Demo - Basic observability without threading."""

import sys
import json
import time
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def main():
    """Run simple monitoring demonstration."""
    print("📊 Mobile Multi-Modal LLM - Simple Monitoring Demo")
    print("=" * 60)
    
    try:
        from mobile_multimodal.robust_monitoring import (
            MetricsCollector, HealthChecker, performance_trace
        )
        print("✅ Monitoring components loaded")
        
        # Initialize metrics collector
        metrics = MetricsCollector()
        print("✅ Metrics collector initialized")
        
        # Simulate model operations with monitoring
        print("\\n🤖 Simulating model operations...")
        
        # Image processing simulation
        with performance_trace(metrics, "image_processing"):
            print("  📸 Processing image...")
            time.sleep(0.01)  # Quick simulation
            metrics.record_histogram("inference.latency", random.uniform(10, 50))
            metrics.increment_counter("images.processed")
        
        # Text processing simulation  
        with performance_trace(metrics, "text_processing"):
            print("  📝 Processing text...")
            time.sleep(0.01)  # Quick simulation
            metrics.record_histogram("inference.latency", random.uniform(5, 25))
            metrics.increment_counter("texts.processed")
        
        # OCR simulation
        with performance_trace(metrics, "ocr_extraction"):
            print("  🔍 Extracting text from image...")
            time.sleep(0.01)  # Quick simulation
            metrics.record_histogram("inference.latency", random.uniform(15, 60))
            metrics.increment_counter("ocr.extractions")
        
        # Simulate some errors
        try:
            with performance_trace(metrics, "error_simulation"):
                print("  ❌ Simulating error...")
                raise Exception("Simulated processing error")
        except Exception:
            metrics.increment_counter("errors.processing")
            print("  ✅ Error properly tracked")
        
        # Record some custom metrics
        print("\\n📈 Recording custom metrics...")
        
        # Model performance metrics
        metrics.set_gauge("model.accuracy", 0.937)
        metrics.set_gauge("model.confidence", 0.892)
        metrics.record_histogram("batch.size", random.randint(1, 8))
        
        # Resource usage metrics (simulated)
        metrics.set_gauge("gpu.utilization", random.uniform(60, 90))
        metrics.set_gauge("memory.model", random.uniform(200, 400))
        
        print("✅ Custom metrics recorded")
        
        # Initialize health checker
        health_checker = HealthChecker()
        
        def memory_check():
            return True  # Simulate healthy memory
        
        def disk_check():
            return True  # Simulate healthy disk
        
        health_checker.register_check("memory", memory_check)
        health_checker.register_check("disk", disk_check)
        
        # Check health status
        print("\\n🏥 Running health checks...")
        health_results = health_checker.run_all_checks()
        
        print(f"  Overall Status: {health_results['overall_status']}")
        for check_name, check_result in health_results['checks'].items():
            status_emoji = "✅" if check_result['status'] == "HEALTHY" else "❌"
            print(f"  {status_emoji} {check_name}: {check_result['status']}")
        
        # Get metrics summary
        print("\\n📊 Generating metrics summary...")
        metrics_summary = metrics.get_metrics_summary()
        
        # Display key metrics
        print("\\n📈 Key Metrics Summary:")
        print(f"  - Images processed: {metrics_summary['counters'].get('images.processed', 0)}")
        print(f"  - Texts processed: {metrics_summary['counters'].get('texts.processed', 0)}")
        print(f"  - OCR extractions: {metrics_summary['counters'].get('ocr.extractions', 0)}")
        print(f"  - Processing errors: {metrics_summary['counters'].get('errors.processing', 0)}")
        
        # Display gauge metrics
        print(f"  - Model accuracy: {metrics_summary['gauges'].get('model.accuracy', 0):.3f}")
        print(f"  - Model confidence: {metrics_summary['gauges'].get('model.confidence', 0):.3f}")
        
        # Display performance statistics
        if 'inference.latency' in metrics_summary['histograms']:
            latency_stats = metrics_summary['histograms']['inference.latency']
            print(f"  - Latency P50: {latency_stats['p50']:.1f}ms")
            print(f"  - Latency P95: {latency_stats['p95']:.1f}ms")
            print(f"  - Latency P99: {latency_stats['p99']:.1f}ms")
        
        # Save metrics summary
        summary_path = Path("metrics_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2, default=str)
        
        print(f"✅ Metrics summary saved to {summary_path}")
        
        # Display recent traces
        print("\\n🔍 Recent Operation Traces:")
        for trace in metrics_summary['recent_traces']:
            success_emoji = "✅" if trace['success'] else "❌"
            print(f"  {success_emoji} {trace['operation']}: {trace['duration_ms']:.1f}ms")
        
        print("\\n🎯 Simple Monitoring Demo Complete!")
        print("✅ All monitoring components working correctly")
        print("✅ Metrics collection and health checks operational")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during monitoring demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())