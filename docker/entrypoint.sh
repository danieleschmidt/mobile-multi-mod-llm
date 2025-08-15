#!/bin/bash
# Production entrypoint script for Self-Healing Pipeline Guard

set -e

# Default configuration
DEFAULT_LOG_LEVEL=${LOG_LEVEL:-INFO}
DEFAULT_CONFIG_PATH=${CONFIG_PATH:-/app/config/production.yaml}
DEFAULT_DATA_DIR=${DATA_DIR:-/app/data}
DEFAULT_MODELS_DIR=${MODELS_DIR:-/app/models}

# Environment validation
echo "🚀 Starting Self-Healing Pipeline Guard"
echo "Environment: ${ENVIRONMENT:-production}"
echo "Log Level: ${DEFAULT_LOG_LEVEL}"
echo "Config Path: ${DEFAULT_CONFIG_PATH}"

# Create necessary directories
mkdir -p "${DEFAULT_DATA_DIR}" "${DEFAULT_MODELS_DIR}" /app/logs

# Validate required environment variables
required_vars=(
    "PIPELINE_GUARD_DB_PATH"
    "PIPELINE_GUARD_LOG_PATH"
)

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo "⚠️  Warning: ${var} not set, using default"
    fi
done

# Database initialization
if [[ ! -f "${DEFAULT_DATA_DIR}/pipeline_guard.db" ]]; then
    echo "📊 Initializing database..."
    python3 -c "
from src.mobile_multimodal.guard_metrics import MetricsCollector
from src.mobile_multimodal.guard_logging import GuardLogHandler
import os

# Initialize database
db_path = os.path.join('${DEFAULT_DATA_DIR}', 'pipeline_guard.db')
collector = MetricsCollector(db_path)
print(f'✅ Database initialized at {db_path}')

# Initialize log database
log_db_path = os.path.join('${DEFAULT_DATA_DIR}', 'pipeline_logs.db')
log_handler = GuardLogHandler(log_db_path)
print(f'✅ Log database initialized at {log_db_path}')
"
fi

# Configuration validation
if [[ -f "${DEFAULT_CONFIG_PATH}" ]]; then
    echo "📋 Validating configuration..."
    python3 -c "
from src.mobile_multimodal.guard_config import ConfigManager
try:
    manager = ConfigManager('${DEFAULT_CONFIG_PATH}')
    issues = ConfigManager.validate_config(manager.get_config())
    if issues:
        print('⚠️  Configuration issues found:')
        for issue in issues:
            print(f'  - {issue}')
    else:
        print('✅ Configuration validation passed')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    exit(1)
"
else
    echo "📋 No configuration file found, using defaults"
fi

# Health check
echo "🔍 Running startup health checks..."
python3 -c "
import sys
import os
sys.path.insert(0, '/app')

try:
    # Test core imports
    from src.mobile_multimodal.pipeline_guard import SelfHealingPipelineGuard
    from src.mobile_multimodal.guard_orchestrator import PipelineOrchestrator
    print('✅ Core modules imported successfully')
    
    # Test basic functionality
    guard = SelfHealingPipelineGuard()
    status = guard.get_system_status()
    print('✅ Pipeline guard initialization successful')
    
    print('✅ All startup health checks passed')
except Exception as e:
    print(f'❌ Startup health check failed: {e}')
    sys.exit(1)
"

# Signal handling
cleanup() {
    echo "🛑 Received shutdown signal, cleaning up..."
    if [[ -n "$PIPELINE_PID" ]]; then
        kill -TERM "$PIPELINE_PID" 2>/dev/null || true
        wait "$PIPELINE_PID" 2>/dev/null || true
    fi
    echo "✅ Cleanup completed"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main execution based on command
case "$1" in
    "orchestrator")
        echo "🎯 Starting Pipeline Orchestrator..."
        python3 -m src.mobile_multimodal.guard_orchestrator --config "${DEFAULT_CONFIG_PATH}" --start &
        PIPELINE_PID=$!
        wait $PIPELINE_PID
        ;;
    "guard")
        echo "🛡️  Starting Pipeline Guard..."
        python3 -m src.mobile_multimodal.pipeline_guard --config "${DEFAULT_CONFIG_PATH}" --daemon &
        PIPELINE_PID=$!
        wait $PIPELINE_PID
        ;;
    "metrics")
        echo "📊 Starting Metrics Collector..."
        python3 -m src.mobile_multimodal.guard_metrics --collect --db "${DEFAULT_DATA_DIR}/pipeline_metrics.db" &
        PIPELINE_PID=$!
        wait $PIPELINE_PID
        ;;
    "status")
        echo "📋 Checking system status..."
        python3 -c "
from src.mobile_multimodal.pipeline_guard import SelfHealingPipelineGuard
import json
guard = SelfHealingPipelineGuard('${DEFAULT_CONFIG_PATH}')
status = guard.get_system_status()
print(json.dumps(status, indent=2, default=str))
"
        ;;
    "shell")
        echo "🐚 Starting interactive shell..."
        exec /bin/bash
        ;;
    *)
        echo "❓ Unknown command: $1"
        echo "Available commands:"
        echo "  orchestrator - Start full orchestration system"
        echo "  guard       - Start pipeline guard only"
        echo "  metrics     - Start metrics collector only"
        echo "  status      - Show system status"
        echo "  shell       - Interactive shell"
        exit 1
        ;;
esac