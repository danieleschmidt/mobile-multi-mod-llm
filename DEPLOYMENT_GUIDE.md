# Self-Healing Pipeline Guard - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Self-Healing Pipeline Guard in various environments, from development to production-scale Kubernetes clusters.

## Quick Start

### Local Development
```bash
# 1. Clone and setup
git clone <repository-url>
cd self-healing-pipeline-guard

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. Run locally
python3 -m src.mobile_multimodal.guard_orchestrator --start
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.production.yml up -d
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/deployment.yaml
```

## Deployment Options

### 1. Local Development Environment

#### Prerequisites
- Python 3.10+
- 8GB+ RAM
- 10GB+ disk space

#### Setup
```bash
# Create virtual environment
python3 -m venv pipeline-guard-env
source pipeline-guard-env/bin/activate  # Linux/Mac
# pipeline-guard-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-prod.txt

# Initialize configuration
mkdir -p config data logs models
cp config/pipeline_guard_config_sample.yaml config/production.yaml

# Start services
python3 -m src.mobile_multimodal.guard_orchestrator --config config/production.yaml --start
```

#### Verification
```bash
# Check status
python3 -m src.mobile_multimodal.pipeline_guard --status

# View logs
tail -f logs/pipeline_guard.log

# Access monitoring
# Open http://localhost:8080/metrics in browser
```

### 2. Docker Standalone Deployment

#### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 16GB+ RAM
- 50GB+ disk space

#### Build and Deploy
```bash
# Build production image
docker build -f docker/Dockerfile.production -t pipeline-guard:production .

# Start all services
docker-compose -f docker/docker-compose.production.yml up -d

# Check service health
docker-compose -f docker/docker-compose.production.yml ps
docker-compose -f docker/docker-compose.production.yml logs pipeline-guard
```

#### Service Endpoints
- **Pipeline Guard**: http://localhost:8080
- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Redis**: localhost:6379

#### Management Commands
```bash
# View logs
docker-compose logs -f pipeline-guard

# Scale services
docker-compose up -d --scale pipeline-guard=3

# Update configuration
docker-compose restart pipeline-guard

# Backup data
docker run --rm -v pipeline_data:/source -v $(pwd)/backup:/backup alpine tar czf /backup/pipeline-data-$(date +%Y%m%d).tar.gz -C /source .

# Stop services
docker-compose down
```

### 3. Kubernetes Production Deployment

#### Prerequisites
- Kubernetes 1.21+
- kubectl configured
- Minimum 3 nodes with 16GB+ RAM each
- Persistent storage provisioner
- Ingress controller (optional)

#### Namespace Setup
```bash
# Create namespace
kubectl create namespace pipeline-system

# Set default namespace
kubectl config set-context --current --namespace=pipeline-system
```

#### Storage Configuration
```bash
# For cloud providers, ensure storage classes exist
kubectl get storageclass

# Example for AWS EKS
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com
volumeBindingMode: WaitForFirstConsumer
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
allowVolumeExpansion: true
EOF
```

#### Deploy Application
```bash
# Apply all configurations
kubectl apply -f kubernetes/deployment.yaml

# Verify deployment
kubectl get all -n pipeline-system

# Check pod status
kubectl get pods -n pipeline-system -w

# View logs
kubectl logs -l app=pipeline-guard -n pipeline-system -f
```

#### Expose Services
```bash
# Option 1: Port forwarding (development)
kubectl port-forward svc/pipeline-guard-service 8080:8080 -n pipeline-system

# Option 2: LoadBalancer (cloud)
kubectl patch svc pipeline-guard-service -p '{"spec":{"type":"LoadBalancer"}}' -n pipeline-system

# Option 3: Ingress (production)
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pipeline-guard-ingress
  namespace: pipeline-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - pipeline-guard.yourdomain.com
    secretName: pipeline-guard-tls
  rules:
  - host: pipeline-guard.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pipeline-guard-service
            port:
              number: 8080
EOF
```

#### Monitoring Setup
```bash
# Deploy Prometheus and Grafana
kubectl apply -f kubernetes/monitoring.yaml

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n pipeline-system
# Open http://localhost:3000 (admin/admin)
```

### 4. Cloud-Specific Deployments

#### AWS EKS
```bash
# Create EKS cluster
eksctl create cluster \
  --name pipeline-guard-cluster \
  --version 1.28 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

# Deploy application
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/aws-specific.yaml
```

#### Google GKE
```bash
# Create GKE cluster
gcloud container clusters create pipeline-guard-cluster \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --region us-central1

# Get credentials
gcloud container clusters get-credentials pipeline-guard-cluster --region us-central1

# Deploy application
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/gcp-specific.yaml
```

#### Azure AKS
```bash
# Create AKS cluster
az aks create \
  --resource-group pipeline-guard-rg \
  --name pipeline-guard-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys \
  --node-vm-size Standard_D4s_v3

# Get credentials
az aks get-credentials --resource-group pipeline-guard-rg --name pipeline-guard-cluster

# Deploy application
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/azure-specific.yaml
```

## Configuration Management

### Environment Variables
```bash
# Core configuration
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export CONFIG_PATH=/app/config/production.yaml

# Database configuration
export PIPELINE_GUARD_DB_PATH=/app/data/pipeline_guard.db
export PIPELINE_GUARD_LOG_PATH=/app/logs/pipeline_guard.log

# Security configuration
export REDIS_password=<SECURE_PASSWORD>-secure-password
export JWT_secret_key=<YOUR_SECRET_KEY>-jwt-secret

# Monitoring configuration
export PROMETHEUS_URL=http://prometheus:9090
export GRAFANA_URL=http://grafana:3000
```

### Configuration Files
```yaml
# config/production.yaml
guard_name: "pipeline-guard-production"
log_level: "INFO"

# Component-specific settings
model_training:
  enabled: true
  check_interval_seconds: 300
  auto_recovery: true

# Alerting configuration
alerting:
  enabled: true
  slack_webhook: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  email_recipients:
    - "admin@yourdomain.com"

# Advanced features
ml_anomaly_detection: true
predictive_scaling: true
chaos_engineering: false
```

## Security Configuration

### TLS/SSL Setup
```bash
# Generate certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout pipeline-guard.key \
  -out pipeline-guard.crt \
  -subj "/CN=pipeline-guard.yourdomain.com"

# Create Kubernetes secret
kubectl create secret tls pipeline-guard-tls \
  --cert=pipeline-guard.crt \
  --key=pipeline-guard.key \
  -n pipeline-system
```

### RBAC Configuration
```yaml
# Security policies already included in deployment.yaml
# - ServiceAccount with minimal permissions
# - Pod Security Standards
# - Network Policies
# - Security Contexts
```

### Secrets Management
```bash
# Kubernetes secrets
kubectl create secret generic pipeline-guard-secrets \
  --from-literal=redis-password=<SECURE_PASSWORD>-password \
  --from-literal=jwt-secret=your-jwt-secret \
  -n pipeline-system

# External secret management (AWS Secrets Manager)
kubectl apply -f kubernetes/external-secrets.yaml
```

## Monitoring and Observability

### Metrics Collection
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **Custom Metrics**: Application-specific metrics

### Logging
- **Structured Logging**: JSON format with correlation IDs
- **Log Aggregation**: Centralized log collection
- **Log Analysis**: Pattern detection and alerting

### Alerting Rules
```yaml
# Prometheus alerting rules
groups:
- name: pipeline-guard
  rules:
  - alert: HighErrorRate
    expr: pipeline_guard_error_rate > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
```

## Scaling and Performance

### Horizontal Scaling
```bash
# Manual scaling
kubectl scale deployment pipeline-guard --replicas=5 -n pipeline-system

# Auto-scaling (HPA already configured)
kubectl get hpa -n pipeline-system
```

### Vertical Scaling
```bash
# Update resource limits
kubectl patch deployment pipeline-guard -p '{"spec":{"template":{"spec":{"containers":[{"name":"orchestrator","resources":{"limits":{"memory":"4Gi","cpu":"2000m"}}}]}}}}' -n pipeline-system
```

### Performance Tuning
```yaml
# Resource optimization
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

# JVM tuning (if applicable)
env:
- name: JAVA_OPTS
  value: "-Xmx2g -Xms1g -XX:+UseG1GC"
```

## Backup and Disaster Recovery

### Data Backup
```bash
# Database backup
kubectl exec -it deployment/pipeline-guard -n pipeline-system -- \
  sqlite3 /app/data/pipeline_guard.db ".backup /app/data/backup_$(date +%Y%m%d).db"

# Volume backup
kubectl create job --from=cronjob/backup-job backup-manual -n pipeline-system
```

### Disaster Recovery
```bash
# Restore from backup
kubectl cp backup_20231201.db pipeline-guard-pod:/app/data/pipeline_guard.db

# Database restoration
kubectl exec -it deployment/pipeline-guard -n pipeline-system -- \
  sqlite3 /app/data/pipeline_guard.db ".restore /app/data/backup_20231201.db"
```

## Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod events
kubectl describe pod -l app=pipeline-guard -n pipeline-system

# Check logs
kubectl logs -l app=pipeline-guard -n pipeline-system --previous

# Debug with shell
kubectl exec -it deployment/pipeline-guard -n pipeline-system -- /bin/bash
```

#### Database Connection Issues
```bash
# Check database file permissions
kubectl exec -it deployment/pipeline-guard -n pipeline-system -- ls -la /app/data/

# Test database connectivity
kubectl exec -it deployment/pipeline-guard -n pipeline-system -- \
  python3 -c "from src.mobile_multimodal.guard_metrics import MetricsCollector; MetricsCollector('/app/data/pipeline_guard.db')"
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n pipeline-system

# Analyze metrics
kubectl port-forward svc/prometheus 9090:9090 -n pipeline-system
# Open http://localhost:9090
```

### Debug Commands
```bash
# Health check
kubectl exec -it deployment/pipeline-guard -n pipeline-system -- python3 healthcheck.py

# System status
kubectl exec -it deployment/pipeline-guard -n pipeline-system -- \
  python3 -m src.mobile_multimodal.pipeline_guard --status

# Configuration validation
kubectl exec -it deployment/pipeline-guard -n pipeline-system -- \
  python3 -m src.mobile_multimodal.guard_config --validate /app/config/production.yaml
```

## Maintenance

### Regular Maintenance Tasks
```bash
# Database cleanup (weekly)
kubectl create job --from=cronjob/db-cleanup cleanup-manual -n pipeline-system

# Security updates (monthly)
docker build --no-cache -f docker/Dockerfile.production -t pipeline-guard:latest .
kubectl set image deployment/pipeline-guard orchestrator=pipeline-guard:latest -n pipeline-system

# Configuration updates
kubectl patch configmap pipeline-guard-config --patch-file config-update.yaml -n pipeline-system
kubectl rollout restart deployment/pipeline-guard -n pipeline-system
```

### Health Monitoring
```bash
# Automated health checks
kubectl get deployment pipeline-guard -o jsonpath='{.status.readyReplicas}' -n pipeline-system

# Service mesh health (if using Istio)
kubectl get destinationrule,virtualservice -n pipeline-system
```

## Best Practices

### Deployment Best Practices
1. **Use Infrastructure as Code**: Version control all configurations
2. **Implement GitOps**: Automated deployment pipelines
3. **Resource Limits**: Always set resource requests and limits
4. **Health Checks**: Implement comprehensive health checks
5. **Security**: Follow security best practices and policies

### Operational Best Practices
1. **Monitoring**: Comprehensive observability stack
2. **Alerting**: Proactive alerting on key metrics
3. **Backup**: Regular automated backups
4. **Testing**: Automated testing in CI/CD pipeline
5. **Documentation**: Keep deployment documentation updated

### Performance Best Practices
1. **Resource Optimization**: Right-size resources based on usage
2. **Caching**: Implement appropriate caching strategies
3. **Load Balancing**: Distribute load effectively
4. **Auto-scaling**: Configure HPA and VPA appropriately
5. **Performance Testing**: Regular performance testing and optimization

## Support and Documentation

### Additional Resources
- [API Documentation](./API_DOCUMENTATION.md)
- [Security Guide](./SECURITY_HARDENING_REPORT.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Performance Tuning](./PERFORMANCE_GUIDE.md)

### Community Support
- GitHub Issues: Report bugs and feature requests
- Documentation: Contribute to documentation improvements
- Community Forum: Join discussions and share experiences

This deployment guide provides comprehensive instructions for deploying the Self-Healing Pipeline Guard across various environments. Choose the deployment option that best fits your infrastructure and requirements.