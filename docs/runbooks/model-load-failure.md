# Runbook: Model Load Failure

**Alert**: `ModelLoadFailure`  
**Severity**: Critical  
**Description**: The model failed to load during service startup or runtime.

## Immediate Actions (First 3 minutes)

### 1. Check Alert Details
```bash
# Get specific error from Prometheus
curl -s "http://prometheus:9090/api/v1/query?query=model_load_failures_total" | jq '.'

# Check recent application logs
docker-compose logs --since=10m app-prod | grep -i "model\|load\|error"
```

### 2. Verify Model Files
```bash
# Check if model files exist
ls -la /app/models/
ls -la /app/models/*.pth /app/models/*.tflite /app/models/*.mlpackage

# Check file permissions
stat /app/models/best_model.pth
```

### 3. Check Available Resources
```bash
# Check memory availability
free -h
docker stats --no-stream app-prod

# Check disk space
df -h /app/models/
```

## Investigation Steps

### 1. Analyze Error Messages
```bash
# Get detailed error logs
docker-compose logs --since=30m app-prod | grep -A 10 -B 10 -i "model.*load.*fail"

# Common error patterns to look for:
# - "FileNotFoundError"
# - "RuntimeError: CUDA out of memory"
# - "OSError: [Errno 28] No space left on device"
# - "ValueError: The model file is corrupted"
# - "ImportError: No module named"
```

### 2. Validate Model Files
```bash
# Check model file integrity
python -c "
import torch
try:
    model = torch.load('/app/models/best_model.pth', map_location='cpu')
    print('Model loaded successfully')
    print('Model keys:', list(model.keys()))
except Exception as e:
    print('Model load error:', str(e))
"

# Check file size and checksums
ls -lh /app/models/
md5sum /app/models/*.pth
```

### 3. Test Model Loading Manually
```bash
# Try loading model manually
docker-compose exec app-prod python -c "
from mobile_multimodal import MobileMultiModalLLM
try:
    model = MobileMultiModalLLM.from_pretrained('mobile-mm-llm-int2')
    print('Model loaded successfully')
except Exception as e:
    print('Error:', str(e))
    import traceback
    traceback.print_exc()
"
```

### 4. Check Dependencies
```bash
# Verify required packages
docker-compose exec app-prod pip list | grep -E "(torch|tensorflow|onnx|coreml)"

# Check Python path
docker-compose exec app-prod python -c "import sys; print('\n'.join(sys.path))"

# Check environment variables
docker-compose exec app-prod env | grep -E "(MODEL|TORCH|PATH)"
```

## Resolution Steps

### Scenario 1: Model File Missing or Corrupted
```bash
# Download fresh model files
python scripts/download_models.py --model int2_quantized --force

# Or restore from backup
cp /backup/models/best_model.pth /app/models/

# Verify file integrity
python -c "
import torch
model = torch.load('/app/models/best_model.pth', map_location='cpu')
print('Model validation successful')
"

# Restart service
docker-compose restart app-prod
```

### Scenario 2: Out of Memory During Load
```bash
# Check current memory usage
docker stats --no-stream app-prod

# Increase memory limits
# Edit docker-compose.yml:
services:
  app-prod:
    mem_limit: 4g  # Increase from current limit
    mem_reservation: 2g

# Enable memory-efficient loading
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Restart with new configuration
docker-compose up -d app-prod
```

### Scenario 3: CUDA/GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Check CUDA version compatibility
docker-compose exec app-prod python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('PyTorch version:', torch.__version__)
"

# If GPU issues, force CPU loading
export CUDA_VISIBLE_DEVICES=""
docker-compose restart app-prod

# Or update model loading to use CPU
# Edit application configuration:
# device: "cpu"
```

### Scenario 4: Permission Issues
```bash
# Check file ownership and permissions
ls -la /app/models/
stat /app/models/best_model.pth

# Fix permissions if needed
chown -R appuser:appuser /app/models/
chmod 644 /app/models/*.pth

# Restart service
docker-compose restart app-prod
```

### Scenario 5: Model Format/Version Issues
```bash
# Check model format compatibility
python -c "
import torch
checkpoint = torch.load('/app/models/best_model.pth', map_location='cpu')
print('Model format version:', checkpoint.get('version', 'unknown'))
print('PyTorch version used:', checkpoint.get('pytorch_version', 'unknown'))
print('Model architecture:', checkpoint.get('architecture', 'unknown'))
"

# Convert model if needed
python scripts/convert_model.py \
  --input /app/models/best_model.pth \
  --output /app/models/best_model_converted.pth \
  --target-version current

# Update configuration to use converted model
```

### Scenario 6: Missing Dependencies
```bash
# Install missing dependencies
docker-compose exec app-prod pip install -r requirements.txt

# Or rebuild container with dependencies
docker-compose build --no-cache app-prod
docker-compose up -d app-prod
```

### Scenario 7: Configuration Issues
```bash
# Check model configuration
cat config/model.yml

# Validate configuration
python -c "
import yaml
with open('config/model.yml') as f:
    config = yaml.safe_load(f)
print('Model config:', config)
"

# Fix configuration file
# Update model_path, architecture, etc.

# Restart service
docker-compose restart app-prod
```

## Validation Steps

### 1. Verify Model Loading
```bash
# Check health endpoint
curl -f http://app-prod:8080/health
# Should return: {"status": "healthy", "model_loaded": true}

# Check model-specific endpoint
curl -f http://app-prod:8080/model/health
# Should return model status and capabilities
```

### 2. Test Model Inference
```bash
# Simple inference test
curl -X POST http://app-prod:8080/model/inference \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    "task": "captioning"
  }'

# Should return valid inference result
```

### 3. Check Application Metrics
```bash
# Verify model metrics are being reported
curl -s http://app-prod:8080/metrics | grep model_load
curl -s http://app-prod:8080/metrics | grep model_inference
```

### 4. Monitor for Stability
```bash
# Watch logs for continued issues
docker-compose logs -f app-prod | grep -i model

# Monitor memory usage over time
watch -n 5 'docker stats --no-stream app-prod'
```

## Prevention Measures

### 1. Model File Validation
```python
# Add to application startup
def validate_model_file(model_path):
    """Validate model file before loading."""
    import os
    import torch
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if os.path.getsize(model_path) == 0:
        raise ValueError(f"Model file is empty: {model_path}")
    
    try:
        # Quick validation without full load
        torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception as e:
        raise ValueError(f"Model file is corrupted: {e}")
```

### 2. Graceful Degradation
```python
# Implement fallback models
class ModelManager:
    def __init__(self):
        self.primary_model = None
        self.fallback_model = None
    
    def load_models(self):
        try:
            self.primary_model = self.load_primary_model()
        except Exception as e:
            logger.error(f"Primary model load failed: {e}")
            self.fallback_model = self.load_fallback_model()
```

### 3. Health Check Enhancement
```python
# Add model-specific health checks
@app.route('/model/health')
def model_health():
    """Detailed model health check."""
    status = {
        'model_loaded': model_manager.is_loaded(),
        'model_type': model_manager.get_model_type(),
        'memory_usage_mb': get_memory_usage(),
        'last_inference_time': model_manager.last_inference_time,
        'inference_count': model_manager.inference_count
    }
    return jsonify(status)
```

### 4. Monitoring Enhancements
```yaml
# Add to monitoring/alerts.yml
- alert: ModelLoadTime
  expr: model_load_duration_seconds > 60
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "Model taking too long to load"
    description: "Model load time is {{ $value }}s, exceeding 60s threshold."

- alert: ModelMemoryUsage
  expr: model_memory_usage_bytes > 2e9  # 2GB
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "Model using excessive memory"
    description: "Model memory usage is {{ $value | humanizeBytes }}."
```

### 5. Automated Recovery
```bash
# Add to docker-compose.yml
healthcheck:
  test: ["CMD", "python", "-c", "
    from mobile_multimodal import MobileMultiModalLLM;
    MobileMultiModalLLM.health_check()
  "]
  interval: 60s
  timeout: 30s
  retries: 3
  start_period: 120s

restart: unless-stopped
```

## Escalation Criteria

**Escalate to ML Team Lead if:**
- Model loading continues to fail after file restoration
- Multiple different models are failing to load
- Issue affects model accuracy or capabilities

**Escalate to Infrastructure Team if:**
- Issue appears to be hardware-related (GPU, memory)
- Multiple services are affected
- Persistent storage issues

**Emergency Escalation if:**
- Production service has been down for >10 minutes
- No fallback model is available
- Data corruption is suspected

## Post-Incident Actions

### 1. Root Cause Analysis
```bash
# Document the specific cause
# - File corruption source
# - Configuration drift
# - Resource exhaustion cause
# - Dependency issues
```

### 2. Improve Monitoring
```bash
# Add model file integrity checks
# Monitor model loading performance
# Add alerts for model file changes
```

### 3. Update Documentation
```bash
# Update model deployment procedures
# Enhance troubleshooting guides
# Update backup/restore procedures
```

### 4. Test Recovery Procedures
```bash
# Test model backup/restore process
# Validate fallback model functionality
# Test automated recovery scripts
```

## Related Runbooks
- [Service Down](./service-down.md)
- [High Memory Usage](./high-memory.md)
- [Accuracy Drop](./accuracy-drop.md)
- [Quantization Issues](./quantization-issues.md)

## Emergency Commands Cheat Sheet

```bash
# Quick diagnostics
docker-compose logs --tail=50 app-prod | grep -i model
ls -la /app/models/
free -h && df -h

# Quick fixes
docker-compose restart app-prod
python scripts/download_models.py --force
chown -R appuser:appuser /app/models/

# Health checks
curl -f http://app-prod:8080/health
curl -f http://app-prod:8080/model/health
```

---
**Last Updated**: January 2025  
**Runbook Version**: 1.0  
**Owner**: ML Engineering Team