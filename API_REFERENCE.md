# Self-Healing Pipeline Guard API Reference

## Overview

The Self-Healing Pipeline Guard provides a comprehensive API for monitoring, managing, and controlling ML/AI pipeline infrastructure. This document covers all available APIs, endpoints, and usage examples.

## Core APIs

### Pipeline Guard API

#### `SelfHealingPipelineGuard`

Main class for pipeline monitoring and recovery.

```python
class SelfHealingPipelineGuard:
    def __init__(self, config_path: Optional[str] = None)
    async def start_monitoring(self) -> None
    def stop_monitoring(self) -> None
    def get_system_status(self) -> Dict[str, Any]
```

**Example Usage:**
```python
from src.mobile_multimodal.pipeline_guard import SelfHealingPipelineGuard

# Initialize
guard = SelfHealingPipelineGuard("config/production.yaml")

# Start monitoring
await guard.start_monitoring()

# Get status
status = guard.get_system_status()
print(f"Health: {status['overall_health']}")
```

### Metrics API

#### `MetricsCollector`

Collects and stores pipeline metrics.

```python
class MetricsCollector:
    def __init__(self, db_path: str = "pipeline_metrics.db")
    async def start_collection(self) -> None
    async def stop_collection(self) -> None
    def record_metric(self, component: str, metric_name: str, value: float, labels: Dict[str, str] = None)
    def get_metrics(self, component: str = None, metric_name: str = None, start_time: datetime = None, end_time: datetime = None, limit: int = 1000) -> List[MetricPoint]
```

**Example Usage:**
```python
from src.mobile_multimodal.guard_metrics import MetricsCollector

collector = MetricsCollector()
await collector.start_collection()

# Record metrics
collector.record_metric("training", "accuracy", 0.95, {"model": "v2.1"})
collector.record_metric("quantization", "model_size_mb", 34.2)

# Query metrics
metrics = collector.get_metrics(component="training", limit=100)
```

### Orchestration API

#### `PipelineOrchestrator`

High-level orchestration and automation.

```python
class PipelineOrchestrator:
    def __init__(self, config_path: Optional[str] = None)
    async def start(self) -> None
    async def stop(self) -> None
    def get_orchestrator_status(self) -> Dict[str, Any]
```

## Configuration API

### `ConfigManager`

Manages configuration for the pipeline guard system.

```python
class ConfigManager:
    def __init__(self, config_path: Optional[str] = None)
    def get_config(self) -> GuardConfig
    def save_config(self, config: GuardConfig = None)
    def create_sample_config(self, output_path: str = "pipeline_guard_config_sample.yaml")
    @staticmethod
    def validate_config(config: GuardConfig) -> List[str]
```

**Example Usage:**
```python
from src.mobile_multimodal.guard_config import ConfigManager

manager = ConfigManager("config/production.yaml")
config = manager.get_config()

# Validate configuration
issues = ConfigManager.validate_config(config)
if issues:
    print("Configuration issues:", issues)
```

## CLI Commands

### Pipeline Guard Commands

```bash
# Start pipeline guard daemon
python3 -m src.mobile_multimodal.pipeline_guard --config config.yaml --daemon

# Check system status
python3 -m src.mobile_multimodal.pipeline_guard --status

# Show help
python3 -m src.mobile_multimodal.pipeline_guard --help
```

### Metrics Commands

```bash
# Start metrics collection
python3 -m src.mobile_multimodal.guard_metrics --collect --db metrics.db

# Analyze metrics
python3 -m src.mobile_multimodal.guard_metrics --analyze training

# Calculate baselines
python3 -m src.mobile_multimodal.guard_metrics --baselines

# Show active alerts
python3 -m src.mobile_multimodal.guard_metrics --alerts
```

### Orchestrator Commands

```bash
# Start orchestrator
python3 -m src.mobile_multimodal.guard_orchestrator --config config.yaml --start

# Show orchestrator status
python3 -m src.mobile_multimodal.guard_orchestrator --status
```

## HTTP API Endpoints

When running with HTTP server enabled:

### Health Endpoints

```http
GET /health
```
Returns overall system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "training": "healthy",
    "quantization": "healthy",
    "export": "healthy"
  }
}
```

### Metrics Endpoints

```http
GET /metrics
```
Returns Prometheus-format metrics.

```http
GET /api/v1/metrics?component=training&limit=100
```
Returns JSON metrics data.

### Status Endpoints

```http
GET /api/v1/status
```
Returns detailed system status.

```http
GET /api/v1/alerts
```
Returns active alerts.

## Data Models

### Core Data Types

#### `HealthStatus`
```python
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"
```

#### `PipelineComponent`
```python
class PipelineComponent(Enum):
    MODEL_TRAINING = "model_training"
    QUANTIZATION = "quantization"
    MOBILE_EXPORT = "mobile_export"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    STORAGE = "storage"
    COMPUTE = "compute"
```

#### `MetricPoint`
```python
@dataclass
class MetricPoint:
    timestamp: datetime
    component: str
    metric_name: str
    value: float
    labels: Dict[str, str] = None
```

#### `Alert`
```python
@dataclass  
class Alert:
    component: PipelineComponent
    severity: HealthStatus
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]
    resolved: bool = False
    resolution_action: Optional[str] = None
```

## Error Handling

### Exception Types

```python
class PipelineGuardError(Exception):
    """Base exception for pipeline guard errors."""
    pass

class ConfigurationError(PipelineGuardError):
    """Configuration related errors."""
    pass

class MonitoringError(PipelineGuardError):
    """Monitoring related errors."""
    pass

class RecoveryError(PipelineGuardError):
    """Recovery action errors."""
    pass
```

### Error Response Format

```json
{
  "error": {
    "type": "ConfigurationError",
    "message": "Invalid configuration parameter",
    "details": {
      "parameter": "check_interval_seconds",
      "value": -1,
      "constraint": "must be positive"
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Security API

### Authentication

All API endpoints support token-based authentication:

```python
# Set authentication token
headers = {"Authorization": "Bearer your-jwt-token"}
response = requests.get("/api/v1/status", headers=headers)
```

### Rate Limiting

API endpoints are rate limited:
- **Default**: 100 requests per minute per IP
- **Metrics**: 1000 requests per minute per IP
- **Health**: 300 requests per minute per IP

### Security Scanning

```python
from scripts.security_scanner import SecurityScanner

scanner = SecurityScanner("/path/to/project")
report = scanner.scan_project()
```

## Integration Examples

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'pipeline-guard'
    static_configs:
      - targets: ['pipeline-guard:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Pipeline Guard Monitoring",
    "panels": [
      {
        "title": "Component Health",
        "type": "stat",
        "targets": [
          {
            "expr": "pipeline_guard_component_health"
          }
        ]
      }
    ]
  }
}
```

### Slack Integration

```python
# Webhook notification
webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
alert_data = {
    "text": f"Pipeline Alert: {alert.severity.value}",
    "attachments": [
        {
            "color": "danger" if alert.severity == HealthStatus.CRITICAL else "warning",
            "fields": [
                {
                    "title": "Component",
                    "value": alert.component.value,
                    "short": True
                },
                {
                    "title": "Message", 
                    "value": alert.message,
                    "short": False
                }
            ]
        }
    ]
}
```

## Advanced Usage

### Custom Health Checks

```python
def custom_health_check() -> HealthStatus:
    """Custom health check function."""
    try:
        # Your custom logic here
        if check_custom_condition():
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.DEGRADED
    except Exception:
        return HealthStatus.FAILED

# Register custom health check
guard._add_custom_health_check(
    component=PipelineComponent.CUSTOM,
    check_name="custom_check",
    check_function=custom_health_check,
    interval_seconds=60
)
```

### Custom Recovery Actions

```python
def custom_recovery(alert: Alert) -> bool:
    """Custom recovery action."""
    try:
        # Your recovery logic here
        perform_custom_recovery()
        return True
    except Exception:
        return False

# Register custom recovery
guard.recovery_strategies[PipelineComponent.CUSTOM] = custom_recovery
```

### Custom Metrics

```python
# Record custom metrics with labels
collector.record_metric(
    component="mobile_inference",
    metric_name="response_time_ms", 
    value=45.2,
    labels={
        "model_version": "v2.1",
        "device_type": "android",
        "quantization": "int2"
    }
)

# Bulk metric recording
metrics_batch = [
    MetricPoint(datetime.now(), "training", "loss", 0.023),
    MetricPoint(datetime.now(), "training", "accuracy", 0.97),
    MetricPoint(datetime.now(), "validation", "accuracy", 0.95)
]

for metric in metrics_batch:
    collector.record_metric(
        metric.component, 
        metric.metric_name, 
        metric.value
    )
```

## Best Practices

### Configuration Best Practices

1. **Environment-Specific Configs**: Use separate configs for dev/staging/prod
2. **Secret Management**: Store sensitive data in environment variables
3. **Validation**: Always validate configuration before deployment
4. **Documentation**: Document all configuration parameters

### Monitoring Best Practices

1. **Metric Naming**: Use consistent, descriptive metric names
2. **Labels**: Use labels for dimensional metrics
3. **Retention**: Configure appropriate metric retention periods
4. **Aggregation**: Use appropriate aggregation for different metric types

### Performance Best Practices

1. **Batch Operations**: Use batch operations for high-volume metrics
2. **Async Operations**: Use async/await for I/O operations
3. **Resource Limits**: Configure appropriate resource limits
4. **Caching**: Implement caching for frequently accessed data

This API reference provides comprehensive coverage of all available APIs and usage patterns for the Self-Healing Pipeline Guard system.