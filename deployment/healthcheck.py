#!/usr/bin/env python3
"""Production health check script for Mobile Multi-Modal LLM."""

import sys
import time
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from mobile_multimodal import MobileMultiModalLLM
    from mobile_multimodal.monitoring import SystemMetrics
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def check_system_health() -> dict:
    """Comprehensive system health check."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {},
        "metrics": {}
    }
    
    try:
        # Check 1: Model initialization
        try:
            model = MobileMultiModalLLM(health_check_enabled=False)
            health_status["checks"]["model_init"] = model._is_initialized
        except Exception as e:
            health_status["checks"]["model_init"] = False
            health_status["status"] = "unhealthy"
            logging.error(f"Model initialization failed: {e}")
        
        # Check 2: Basic inference capability
        try:
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            caption = model.generate_caption(test_image, max_length=20)
            health_status["checks"]["inference"] = len(caption) > 0
        except Exception as e:
            health_status["checks"]["inference"] = False
            health_status["status"] = "unhealthy"
            logging.error(f"Inference test failed: {e}")
        
        # Check 3: System resources
        try:
            sys_metrics = SystemMetrics()
            metrics = sys_metrics.collect()
            
            # CPU check
            cpu_ok = metrics.get("cpu_percent", 0) < 90
            health_status["checks"]["cpu"] = cpu_ok
            health_status["metrics"]["cpu_percent"] = metrics.get("cpu_percent", 0)
            
            # Memory check  
            memory_ok = metrics.get("memory_percent", 0) < 90
            health_status["checks"]["memory"] = memory_ok
            health_status["metrics"]["memory_percent"] = metrics.get("memory_percent", 0)
            
            # Disk check
            disk_ok = metrics.get("disk_percent", 0) < 95
            health_status["checks"]["disk"] = disk_ok
            health_status["metrics"]["disk_percent"] = metrics.get("disk_percent", 0)
            
            if not (cpu_ok and memory_ok and disk_ok):
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["checks"]["system_resources"] = False
            logging.error(f"System resource check failed: {e}")
        
        # Check 4: Cache directory
        try:
            cache_dir = Path("/app/cache")
            cache_writable = cache_dir.exists() and cache_dir.is_dir()
            if cache_writable:
                # Test write
                test_file = cache_dir / "health_test.tmp"
                test_file.write_text("test")
                test_file.unlink()
            
            health_status["checks"]["cache_directory"] = cache_writable
        except Exception as e:
            health_status["checks"]["cache_directory"] = False
            logging.error(f"Cache directory check failed: {e}")
        
        # Check 5: Log directory
        try:
            log_dir = Path("/app/logs")
            log_writable = log_dir.exists() and log_dir.is_dir()
            health_status["checks"]["log_directory"] = log_writable
        except Exception as e:
            health_status["checks"]["log_directory"] = False
            logging.error(f"Log directory check failed: {e}")
        
        # Overall health determination
        failed_checks = [k for k, v in health_status["checks"].items() if not v]
        critical_checks = ["model_init", "inference"]
        
        critical_failures = [c for c in failed_checks if c in critical_checks]
        if critical_failures:
            health_status["status"] = "unhealthy"
            health_status["critical_failures"] = critical_failures
        elif failed_checks:
            health_status["status"] = "degraded" 
            health_status["failed_checks"] = failed_checks
        
        return health_status
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e),
            "checks": {},
            "metrics": {}
        }

def main():
    """Main health check function."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Run health check
        health = check_system_health()
        
        # Print health status (for debugging)
        print(json.dumps(health, indent=2))
        
        # Exit with appropriate code
        if health["status"] == "healthy":
            sys.exit(0)  # Success
        elif health["status"] == "degraded":
            sys.exit(1)  # Warning - container stays up but marked unhealthy
        else:
            sys.exit(2)  # Failure - container should be restarted
            
    except Exception as e:
        print(f"Health check error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()