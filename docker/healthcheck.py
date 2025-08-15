#!/usr/bin/env python3
"""Health check script for Self-Healing Pipeline Guard container."""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def check_file_system():
    """Check file system health."""
    try:
        # Check if required directories exist and are writable
        required_dirs = ['/app/logs', '/app/data', '/app/models']
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                return False, f"Directory {dir_path} does not exist"
            
            if not os.access(dir_path, os.W_OK):
                return False, f"Directory {dir_path} is not writable"
        
        # Check disk space
        stat = os.statvfs('/app')
        free_space = stat.f_bavail * stat.f_frsize
        free_space_mb = free_space / (1024 * 1024)
        
        if free_space_mb < 100:  # Less than 100MB free
            return False, f"Low disk space: {free_space_mb:.1f}MB remaining"
        
        return True, "File system healthy"
        
    except Exception as e:
        return False, f"File system check failed: {e}"


def check_database():
    """Check database connectivity."""
    try:
        sys.path.insert(0, '/app')
        
        # Test database connection
        import sqlite3
        db_path = '/app/data/pipeline_guard.db'
        
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path, timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            if not tables:
                return False, "Database exists but has no tables"
            
            return True, f"Database healthy with {len(tables)} tables"
        else:
            return False, "Database file does not exist"
            
    except Exception as e:
        return False, f"Database check failed: {e}"


def check_application():
    """Check application health."""
    try:
        sys.path.insert(0, '/app')
        
        # Test core module imports
        from src.mobile_multimodal.pipeline_guard import SelfHealingPipelineGuard
        
        # Create minimal guard instance
        guard = SelfHealingPipelineGuard()
        
        # Test basic functionality
        status = guard.get_system_status()
        
        if status and 'overall_health' in status:
            return True, f"Application healthy: {status['overall_health']}"
        else:
            return False, "Application status check returned invalid data"
            
    except ImportError as e:
        return False, f"Module import failed: {e}"
    except Exception as e:
        return False, f"Application check failed: {e}"


def check_http_endpoint():
    """Check HTTP endpoint if available."""
    try:
        # Try to connect to health endpoint
        health_url = "http://localhost:8080/health"
        
        req = urllib.request.Request(health_url)
        req.add_header('User-Agent', 'Pipeline-Guard-HealthCheck/1.0')
        
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                return True, f"HTTP endpoint healthy: {data.get('status', 'unknown')}"
            else:
                return False, f"HTTP endpoint returned status {response.status}"
                
    except urllib.error.URLError:
        # HTTP endpoint may not be available, which is okay
        return True, "HTTP endpoint not available (expected for some configurations)"
    except Exception as e:
        return False, f"HTTP endpoint check failed: {e}"


def check_process_health():
    """Check process health indicators."""
    try:
        # Check memory usage
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        memory_kb = int(line.split()[1])
                        memory_mb = memory_kb / 1024
                        
                        if memory_mb > 1000:  # More than 1GB
                            return False, f"High memory usage: {memory_mb:.1f}MB"
                        break
        except:
            pass  # /proc may not be available in all environments
        
        # Check if process is responsive
        start_time = time.time()
        
        # Simple CPU-bound operation to test responsiveness
        result = sum(i * i for i in range(1000))
        
        elapsed = time.time() - start_time
        if elapsed > 1.0:  # Took more than 1 second
            return False, f"Process unresponsive: {elapsed:.2f}s for simple operation"
        
        return True, f"Process healthy (response time: {elapsed:.3f}s)"
        
    except Exception as e:
        return False, f"Process health check failed: {e}"


def run_health_checks():
    """Run all health checks and return overall status."""
    checks = [
        ("File System", check_file_system),
        ("Database", check_database),
        ("Application", check_application),
        ("HTTP Endpoint", check_http_endpoint),
        ("Process Health", check_process_health),
    ]
    
    results = []
    overall_healthy = True
    
    for check_name, check_func in checks:
        try:
            healthy, message = check_func()
            results.append({
                'check': check_name,
                'healthy': healthy,
                'message': message
            })
            
            if not healthy:
                overall_healthy = False
                
        except Exception as e:
            results.append({
                'check': check_name,
                'healthy': False,
                'message': f"Check failed with exception: {e}"
            })
            overall_healthy = False
    
    return overall_healthy, results


def main():
    """Main health check function."""
    try:
        overall_healthy, results = run_health_checks()
        
        # Create health check report
        report = {
            'timestamp': time.time(),
            'healthy': overall_healthy,
            'checks': results
        }
        
        # Output format based on environment
        if os.getenv('HEALTH_CHECK_VERBOSE', 'false').lower() == 'true':
            print(json.dumps(report, indent=2))
        else:
            # Concise output for container logs
            if overall_healthy:
                print("✅ Health check passed")
            else:
                failed_checks = [r['check'] for r in results if not r['healthy']]
                print(f"❌ Health check failed: {', '.join(failed_checks)}")
        
        # Save health check results for monitoring
        try:
            health_file = '/app/data/health_status.json'
            os.makedirs(os.path.dirname(health_file), exist_ok=True)
            with open(health_file, 'w') as f:
                json.dump(report, f, indent=2)
        except:
            pass  # Non-critical if we can't save
        
        # Exit with appropriate code
        sys.exit(0 if overall_healthy else 1)
        
    except Exception as e:
        print(f"❌ Health check system error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()