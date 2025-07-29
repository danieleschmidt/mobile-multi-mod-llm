# Deployment Guide

This guide covers deployment strategies for the Mobile Multi-Modal LLM project across different platforms and environments.

## Package Distribution

### PyPI Release Process

1. **Version Management**
   ```bash
   # Update version in pyproject.toml
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. **Automated Release**
   - GitHub Actions automatically builds and publishes to PyPI
   - Manual release: `make build` then `twine upload dist/*`

3. **Installation Verification**
   ```bash
   pip install mobile-multimodal-llm
   python -c "from mobile_multimodal import MobileMultiModalLLM; print('Success')"
   ```

## Mobile Application Deployment

### Android Deployment

#### Play Store Release
```bash
cd mobile-app-android
./gradlew bundleRelease
```

**Requirements**:
- Signed AAB bundle
- Play Console developer account
- App signing certificate

#### Direct APK Distribution
```bash
./gradlew assembleRelease
```

**Considerations**:
- Enable "Install from unknown sources"
- Code signing for production
- ProGuard/R8 obfuscation

### iOS Deployment

#### App Store Release
```bash
cd mobile-app-ios
xcodebuild -workspace MultiModalDemo.xcworkspace \
           -scheme MultiModalDemo \
           -configuration Release \
           archive -archivePath build/MultiModalDemo.xcarchive
```

**Requirements**:
- Apple Developer account
- Distribution certificate
- App Store provisioning profile

#### Enterprise Distribution
- Enterprise certificate required
- Internal distribution only
- No App Store review needed

## Model Deployment

### Model Zoo Setup

#### Hugging Face Hub
```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="models/mobile-mm-llm-int2.tflite",
    path_in_repo="mobile-mm-llm-int2.tflite",
    repo_id="terragon-labs/mobile-multimodal-llm",
    repo_type="model"
)
```

#### Model Versioning
- Semantic versioning (v1.0.0)
- Hardware-specific variants
- Quantization level tags
- Performance benchmarks

### Edge Deployment

#### Docker Containers
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
RUN pip install -e .

EXPOSE 8000
CMD ["python", "-m", "mobile_multimodal.serve"]
```

#### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mobile-mm-llm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mobile-mm-llm
  template:
    metadata:
      labels:
        app: mobile-mm-llm
    spec:
      containers:
      - name: mobile-mm-llm
        image: terragon/mobile-mm-llm:latest
        ports:
        - containerPort: 8000
```

## Cloud Deployment

### Serverless Functions

#### AWS Lambda
```python
import json
from mobile_multimodal import MobileMultiModalLLM

model = None

def lambda_handler(event, context):
    global model
    if model is None:
        model = MobileMultiModalLLM.from_pretrained("mobile-mm-llm-int2")
    
    # Process request
    result = model.generate_caption(event['image'])
    
    return {
        'statusCode': 200,
        'body': json.dumps({'caption': result})
    }
```

#### Google Cloud Functions
```python
import functions_framework
from mobile_multimodal import MobileMultiModalLLM

@functions_framework.http
def process_image(request):
    model = MobileMultiModalLLM.from_pretrained("mobile-mm-llm-int2")
    # Process request
    return {'caption': result}
```

### Container Orchestration

#### Docker Compose
```yaml
version: '3.8'
services:
  mobile-mm-llm:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/mobile-mm-llm-int2
    volumes:
      - ./models:/models
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## Performance Optimization

### Model Optimization

#### Quantization Pipeline
```bash
# INT8 quantization
python tools/quantize.py --model base --target int8

# INT4 quantization (experimental)
python tools/quantize.py --model base --target int4 --calibration data/calib

# ONNX optimization
python tools/optimize_onnx.py --model model.onnx --optimize-level 99
```

#### Hardware-Specific Optimization
```bash
# Snapdragon optimization
python tools/optimize_hexagon.py --model model.tflite

# Apple Silicon optimization
python tools/optimize_coreml.py --model model.mlpackage --use-ane
```

### Deployment Monitoring

#### Metrics Collection
- Inference latency (p50, p95, p99)
- Memory usage patterns
- Error rates and types
- Model accuracy drift

#### Alerting Setup
```yaml
# Prometheus alerting rules
groups:
- name: mobile-mm-llm
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, inference_duration_seconds) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High inference latency detected"
```

## Security Considerations

### Model Security
- **Model signing**: Verify model integrity
- **Encrypted storage**: Protect model weights
- **Access control**: Limit model download permissions
- **Audit logging**: Track model usage

### Application Security
- **Certificate pinning**: Secure API communications
- **Input validation**: Sanitize all inputs
- **Rate limiting**: Prevent abuse
- **Error handling**: Don't expose internals

### Infrastructure Security
- **Network isolation**: VPC/subnet configuration
- **Secrets management**: Use vault systems
- **Regular updates**: Security patches
- **Monitoring**: Security event detection

## Scaling Strategies

### Horizontal Scaling
- **Load balancing**: Distribute requests
- **Auto-scaling**: Dynamic resource adjustment
- **Multi-region**: Geographic distribution
- **CDN**: Model weight distribution

### Vertical Scaling
- **GPU acceleration**: CUDA/ROCm support
- **Memory optimization**: Reduce footprint
- **CPU optimization**: SIMD instructions
- **Batch processing**: Improved throughput

## Rollback Procedures

### Model Rollback
```bash
# Quick model version rollback
kubectl set image deployment/mobile-mm-llm \
  mobile-mm-llm=terragon/mobile-mm-llm:v1.0.0

# Verify rollback
kubectl rollout status deployment/mobile-mm-llm
```

### Application Rollback
```bash
# Git-based rollback
git revert <commit-hash>
git push origin main

# Infrastructure rollback
terraform plan -target=module.mobile_mm_llm
terraform apply -target=module.mobile_mm_llm
```

## Disaster Recovery

### Backup Strategies
- **Model weights**: S3/GCS backup with versioning
- **Configuration**: Git-based infrastructure as code
- **Data**: Regular database backups
- **Monitoring**: Backup alert systems

### Recovery Procedures
1. **Assess impact**: Determine scope of failure
2. **Isolate**: Stop affected services
3. **Restore**: Deploy known good version
4. **Verify**: Test functionality
5. **Monitor**: Watch for issues
6. **Post-mortem**: Document lessons learned

## Environment Management

### Development
- **Local testing**: Docker Compose setup
- **Feature branches**: Isolated development
- **Mock services**: Simulate dependencies
- **Debug tools**: Profiling and tracing

### Staging
- **Production-like**: Same infrastructure
- **Data sanitization**: Anonymized datasets
- **Load testing**: Performance validation
- **Integration testing**: End-to-end scenarios

### Production
- **Blue-green deployment**: Zero-downtime updates
- **Canary releases**: Gradual rollout
- **Health checks**: Automated monitoring
- **Incident response**: On-call procedures

## Compliance & Governance

### Model Governance
- **Version control**: Track model lineage
- **Approval process**: Review before deployment
- **A/B testing**: Validate improvements
- **Rollback criteria**: Define failure conditions

### Data Governance
- **Privacy compliance**: GDPR, CCPA requirements
- **Data retention**: Lifecycle management
- **Access auditing**: Track data usage
- **Anonymization**: Protect user privacy

## Cost Optimization

### Resource Management
- **Right-sizing**: Match resources to demand
- **Reserved instances**: Long-term commitments
- **Spot instances**: Cost-effective computing
- **Auto-shutdown**: Development environments

### Model Efficiency
- **Quantization**: Reduce model size
- **Pruning**: Remove unnecessary parameters
- **Knowledge distillation**: Smaller models
- **Batch optimization**: Improve throughput