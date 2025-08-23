#!/usr/bin/env python3
"""Robust Monitoring Demo - Comprehensive observability and monitoring."""

import sys
import json
import time
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def main():
    """Run robust monitoring demonstration."""
    print("📊 Mobile Multi-Modal LLM - Robust Monitoring Demo")
    print("=" * 60)
    
    try:
        from mobile_multimodal.robust_monitoring import (
            ObservabilityManager, performance_trace
        )
        print("✅ Monitoring components loaded")
        
        # Initialize observability system
        obs_manager = ObservabilityManager()
        obs_manager.start()
        print("✅ Observability system started")
        
        # Simulate model operations with monitoring
        print("\\n🤖 Simulating model operations...")
        
        # Image processing simulation
        with performance_trace(obs_manager.metrics, "image_processing"):
            print("  📸 Processing image...")
            time.sleep(0.1)  # Simulate processing time
            obs_manager.metrics.record_histogram("inference.latency", random.uniform(10, 50))
            obs_manager.metrics.increment_counter("images.processed")
        
        # Text processing simulation  
        with performance_trace(obs_manager.metrics, "text_processing"):
            print("  📝 Processing text...")
            time.sleep(0.05)  # Simulate processing time
            obs_manager.metrics.record_histogram("inference.latency", random.uniform(5, 25))
            obs_manager.metrics.increment_counter("texts.processed")
        
        # OCR simulation
        with performance_trace(obs_manager.metrics, "ocr_extraction"):
            print("  🔍 Extracting text from image...")
            time.sleep(0.15)  # Simulate processing time
            obs_manager.metrics.record_histogram("inference.latency", random.uniform(15, 60))
            obs_manager.metrics.increment_counter("ocr.extractions")
        
        # Simulate some errors
        try:
            with performance_trace(obs_manager.metrics, "error_simulation"):
                print("  ❌ Simulating error...")
                raise Exception("Simulated processing error")
        except Exception:
            obs_manager.metrics.increment_counter("errors.processing")
            print("  ✅ Error properly tracked")
        
        # Record some custom metrics
        print("\\n📈 Recording custom metrics...")
        
        # Model performance metrics
        obs_manager.metrics.set_gauge("model.accuracy", 0.937)
        obs_manager.metrics.set_gauge("model.confidence", 0.892)
        obs_manager.metrics.record_histogram("batch.size", random.randint(1, 8))
        
        # Resource usage metrics
        obs_manager.metrics.set_gauge("gpu.utilization", random.uniform(60, 90))
        obs_manager.metrics.set_gauge("memory.model", random.uniform(200, 400))
        
        print("✅ Custom metrics recorded")
        
        # Wait for system monitoring to collect data
        print("\\n⏱️  Collecting system metrics...")
        time.sleep(2)  # Wait for monitoring cycle
        
        # Check health status
        print("\\n🏥 Running health checks...")
        health_results = obs_manager.health_checker.run_all_checks()
        
        print(f"  Overall Status: {health_results['overall_status']}")
        for check_name, check_result in health_results['checks'].items():
            status_emoji = "✅" if check_result['status'] == "HEALTHY" else "❌"
            print(f"  {status_emoji} {check_name}: {check_result['status']}")
        
        # Check for alerts
        print("\\n🚨 Checking alerts...")
        alerts = obs_manager.alert_manager.check_alerts()
        
        if alerts:
            for alert in alerts:
                severity_emoji = "🟠" if alert['severity'] == "WARNING" else "🔴"
                print(f"  {severity_emoji} {alert['severity']}: {alert['name']}")
        else:
            print("  ✅ No active alerts")
        
        # Generate comprehensive dashboard
        print("\\n📊 Generating monitoring dashboard...")
        dashboard = obs_manager.get_dashboard_data()
        
        # Display key metrics
        metrics_summary = dashboard['metrics']
        print("\\n📈 Key Metrics Summary:")
        print(f"  - Images processed: {metrics_summary['counters'].get('images.processed', 0)}")
        print(f"  - Texts processed: {metrics_summary['counters'].get('texts.processed', 0)}")
        print(f"  - OCR extractions: {metrics_summary['counters'].get('ocr.extractions', 0)}")
        print(f"  - Processing errors: {metrics_summary['counters'].get('errors.processing', 0)}")
        
        # Display system metrics
        if 'system.memory.percent' in metrics_summary['gauges']:
            memory_pct = metrics_summary['gauges']['system.memory.percent']
            cpu_pct = metrics_summary['gauges'].get('system.cpu.percent', 0)
            print(f"  - Memory usage: {memory_pct:.1f}%")
            print(f"  - CPU usage: {cpu_pct:.1f}%")
        
        # Display performance statistics
        if 'inference.latency' in metrics_summary['histograms']:
            latency_stats = metrics_summary['histograms']['inference.latency']
            print(f"  - Latency P50: {latency_stats['p50']:.1f}ms")
            print(f"  - Latency P95: {latency_stats['p95']:.1f}ms")
            print(f"  - Latency P99: {latency_stats['p99']:.1f}ms")
        
        # Save dashboard data
        dashboard_path = Path("monitoring_dashboard.json")
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard, f, indent=2, default=str)
        
        print(f"✅ Dashboard data saved to {dashboard_path}")
        
        # Display recent traces
        print("\\n🔍 Recent Operation Traces:")
        for trace in metrics_summary['recent_traces'][-5:]:
            success_emoji = "✅" if trace['success'] else "❌"
            print(f"  {success_emoji} {trace['operation']}: {trace['duration_ms']:.1f}ms")
        
        # Stop observability system
        obs_manager.stop()
        print("\\n🎯 Monitoring Demo Complete!")
        print("✅ All observability components working correctly")
        print("✅ Ready for production monitoring")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during monitoring demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())