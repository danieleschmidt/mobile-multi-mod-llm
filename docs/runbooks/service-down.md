# Runbook: Service Down

**Alert**: `ModelServiceDown`  
**Severity**: Critical  
**Description**: The Mobile Multi-Modal LLM service is not responding to health checks.

## Immediate Actions (First 5 minutes)

### 1. Verify the Alert
```bash
# Check if service is actually down
curl -f http://app-prod:8080/health
curl -f http://app-prod:8080/alive

# Check service status in orchestrator
docker ps | grep mobile-multimodal
# OR for Kubernetes
kubectl get pods -l app=mobile-multimodal
```

### 2. Check Recent Changes
```bash
# Check recent deployments
git log --oneline -10
docker images mobile-multimodal:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}"

# Check if maintenance window is active
# Refer to maintenance schedule in monitoring/health-checks.yml
```

### 3. Quick Restart Attempt
```bash
# Docker Compose
docker-compose restart app-prod

# Kubernetes
kubectl rollout restart deployment/mobile-multimodal

# Wait 2-3 minutes and verify
curl -f http://app-prod:8080/health
```

## Investigation Steps

### 1. Check Application Logs
```bash
# Docker Compose
docker-compose logs --tail=100 app-prod

# Kubernetes  
kubectl logs -l app=mobile-multimodal --tail=100

# Look for:
# - OutOfMemoryError
# - Model loading failures
# - Database connection errors
# - Configuration errors
```

### 2. Check System Resources
```bash
# Memory usage
free -h
docker stats --no-stream

# Disk space
df -h

# Check if system is under pressure
dmesg | grep -i "killed process"  # OOM killer
```

### 3. Check Dependencies
```bash
# Database connectivity
pg_isready -h postgres -p 5432 -U mmllm

# Redis connectivity  
redis-cli -h redis -p 6379 ping

# Model files
ls -la /app/models/
du -sh /app/models/*
```

### 4. Check Network Issues
```bash
# Port availability
netstat -tlnp | grep :8080
ss -tlnp | grep :8080

# DNS resolution
nslookup app-prod
nslookup postgres
nslookup redis
```

## Resolution Steps

### Scenario 1: Out of Memory
```bash
# Check memory usage
docker stats --no-stream app-prod

# If memory usage > 90%:
# 1. Scale up resources
docker-compose up --scale app-prod=2

# 2. OR increase memory limits
# Edit docker-compose.yml:
#   mem_limit: 4g
#   mem_reservation: 2g

# 3. Restart with new limits
docker-compose up -d app-prod
```

### Scenario 2: Model Loading Failure
```bash
# Check model files
ls -la /app/models/
md5sum /app/models/*.pth

# Re-download models if corrupted
python scripts/download_models.py --model int2_quantized --force

# Restart service
docker-compose restart app-prod
```

### Scenario 3: Database Connection Issues
```bash
# Check database status
docker-compose ps postgres
docker-compose logs postgres

# If database is down
docker-compose up -d postgres

# Wait for database to be ready
until pg_isready -h postgres -p 5432 -U mmllm; do
  echo "Waiting for database..."
  sleep 2
done

# Restart application
docker-compose restart app-prod
```

### Scenario 4: Configuration Issues
```bash
# Check environment variables
docker-compose exec app-prod env | grep -E "(POSTGRES|REDIS|MODEL)"

# Validate configuration
python -c "
import os
import yaml
config = yaml.safe_load(open('config/production.yml'))
print('Config validation:', config)
"

# Fix configuration and restart
docker-compose restart app-prod
```

### Scenario 5: Port Conflicts
```bash
# Check what's using port 8080
lsof -i :8080
netstat -tlnp | grep :8080

# If port is in use by another process
# Kill the conflicting process or change port
# Update docker-compose.yml port mapping

docker-compose up -d app-prod
```

## Validation Steps

### 1. Health Check Verification
```bash
# Basic health check
curl -f http://app-prod:8080/health
# Expected: 200 OK with {"status": "healthy"}

# Readiness check
curl -f http://app-prod:8080/ready
# Expected: 200 OK with dependency status

# Model inference test
curl -X POST http://app-prod:8080/model/inference \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_test_image", "task": "captioning"}'
# Expected: 200 OK with inference result
```

### 2. Monitoring Verification
```bash
# Check Prometheus metrics
curl -s http://prometheus:9090/api/v1/query?query=up{job="mobile-multimodal-app"}
# Expected: value should be 1

# Check application metrics
curl -s http://app-prod:8080/metrics | grep model_inference_requests_total
```

### 3. Load Testing
```bash
# Light load test to ensure stability
for i in {1..10}; do
  curl -f http://app-prod:8080/health &
done
wait

# All requests should return 200
```

## Prevention Measures

### 1. Improve Health Checks
```yaml
# Add to docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### 2. Resource Monitoring
```bash
# Set up resource alerts
# Add to monitoring/alerts.yml:
# - alert: HighMemoryUsage (already exists)
# - alert: HighCPUUsage (already exists)
# - alert: DiskSpaceLow (already exists)
```

### 3. Automated Recovery
```bash
# Add restart policy to docker-compose.yml
restart: unless-stopped

# OR implement circuit breaker pattern in application
```

### 4. Dependency Health Checks
```python
# Add dependency checks to /health endpoint
# Check database, redis, model files, etc.
```

## Escalation Criteria

**Escalate to Engineering Manager if:**
- Service has been down for more than 15 minutes
- Multiple restart attempts have failed
- Root cause indicates larger infrastructure issues
- Data loss or corruption is suspected

**Escalate to Security Team if:**
- Service outage appears to be security-related
- Unusual access patterns in logs
- Potential DDoS or attack patterns

## Post-Incident Actions

### 1. Document the Incident
```bash
# Create incident report
# - Timeline of events
# - Root cause analysis
# - Actions taken
# - Time to resolution
```

### 2. Update Monitoring
```bash
# Add new alerts if gaps were identified
# Adjust alert thresholds if needed
# Add new health checks
```

### 3. Improve Runbook
```bash
# Update this runbook with lessons learned
# Add new scenarios encountered
# Improve prevention measures
```

### 4. Schedule Review
```bash
# Schedule post-incident review meeting
# Include all stakeholders
# Focus on prevention improvements
```

## Related Runbooks
- [High Error Rate](./high-error-rate.md)
- [High Memory Usage](./high-memory.md) 
- [Database Issues](./database-issues.md)
- [Model Load Failure](./model-load-failure.md)

---
**Last Updated**: January 2025  
**Runbook Version**: 1.0  
**Owner**: ML Engineering Team