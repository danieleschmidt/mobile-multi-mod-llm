"""Configuration management for Self-Healing Pipeline Guard."""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class AlertingConfig:
    """Alerting configuration."""
    enabled: bool = True
    slack_webhook: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_recipients: List[str] = None
    alert_cooldown_minutes: int = 15
    escalation_timeout_minutes: int = 60
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []


@dataclass
class ComponentConfig:
    """Individual component configuration."""
    enabled: bool = True
    check_interval_seconds: int = 60
    timeout_seconds: int = 30
    max_retries: int = 3
    auto_recovery: bool = True
    critical_threshold: float = 0.8
    warning_threshold: float = 0.6


@dataclass
class StorageConfig:
    """Storage monitoring configuration."""
    disk_usage_critical: int = 90  # Percentage
    disk_usage_warning: int = 80   # Percentage
    cleanup_enabled: bool = True
    cleanup_age_days: int = 7
    backup_retention_days: int = 30


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    load_average_threshold: float = 8.0
    inference_latency_ms: int = 100
    training_timeout_hours: int = 24


@dataclass
class SecurityConfig:
    """Security monitoring configuration."""
    enabled: bool = True
    scan_interval_hours: int = 6
    vulnerability_db_update: bool = True
    secret_detection: bool = True
    compliance_checks: bool = True


@dataclass
class GuardConfig:
    """Main pipeline guard configuration."""
    # General settings
    guard_name: str = "mobile-multimodal-pipeline-guard"
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/pipeline_guard.log"
    state_file: str = "pipeline_guard_state.json"
    
    # Component configurations
    model_training: ComponentConfig = None
    quantization: ComponentConfig = None
    mobile_export: ComponentConfig = None
    testing: ComponentConfig = None
    deployment: ComponentConfig = None
    monitoring: ComponentConfig = None
    storage: ComponentConfig = None
    compute: ComponentConfig = None
    
    # Specialized configs
    alerting: AlertingConfig = None
    storage_monitoring: StorageConfig = None
    performance: PerformanceConfig = None
    security: SecurityConfig = None
    
    # Advanced features
    ml_anomaly_detection: bool = False
    predictive_scaling: bool = False
    chaos_engineering: bool = False
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.model_training is None:
            self.model_training = ComponentConfig(
                check_interval_seconds=300,  # 5 minutes
                timeout_seconds=300,         # 5 minutes
                max_retries=2
            )
        
        if self.quantization is None:
            self.quantization = ComponentConfig(
                check_interval_seconds=180,  # 3 minutes
                timeout_seconds=120,         # 2 minutes
                max_retries=3
            )
        
        if self.mobile_export is None:
            self.mobile_export = ComponentConfig(
                check_interval_seconds=120,  # 2 minutes
                timeout_seconds=180,         # 3 minutes
                max_retries=3
            )
        
        if self.testing is None:
            self.testing = ComponentConfig(
                check_interval_seconds=60,   # 1 minute
                timeout_seconds=120,         # 2 minutes
                max_retries=2
            )
        
        if self.deployment is None:
            self.deployment = ComponentConfig(
                check_interval_seconds=300,  # 5 minutes
                timeout_seconds=60,          # 1 minute
                max_retries=3
            )
        
        if self.monitoring is None:
            self.monitoring = ComponentConfig(
                check_interval_seconds=30,   # 30 seconds
                timeout_seconds=30,          # 30 seconds
                max_retries=5
            )
        
        if self.storage is None:
            self.storage = ComponentConfig(
                check_interval_seconds=60,   # 1 minute
                timeout_seconds=30,          # 30 seconds
                max_retries=2
            )
        
        if self.compute is None:
            self.compute = ComponentConfig(
                check_interval_seconds=45,   # 45 seconds
                timeout_seconds=15,          # 15 seconds
                max_retries=3
            )
        
        if self.alerting is None:
            self.alerting = AlertingConfig()
        
        if self.storage_monitoring is None:
            self.storage_monitoring = StorageConfig()
        
        if self.performance is None:
            self.performance = PerformanceConfig()
        
        if self.security is None:
            self.security = SecurityConfig()


class ConfigManager:
    """Configuration manager for the pipeline guard."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._find_default_config()
        self.config = self._load_config()
    
    def _find_default_config(self) -> Optional[str]:
        """Find default configuration file."""
        possible_paths = [
            "pipeline_guard_config.yaml",
            "pipeline_guard_config.json",
            "config/pipeline_guard.yaml",
            "config/pipeline_guard.json",
            os.path.expanduser("~/.pipeline_guard.yaml"),
            "/etc/pipeline_guard/config.yaml",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def _load_config(self) -> GuardConfig:
        """Load configuration from file or create default."""
        if not self.config_path or not Path(self.config_path).exists():
            return GuardConfig()
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            return self._dict_to_config(config_data)
        
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            return GuardConfig()
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> GuardConfig:
        """Convert dictionary to GuardConfig object."""
        # Extract nested configurations
        component_configs = {}
        for component in ['model_training', 'quantization', 'mobile_export', 
                         'testing', 'deployment', 'monitoring', 'storage', 'compute']:
            if component in config_data:
                component_configs[component] = ComponentConfig(**config_data[component])
        
        specialized_configs = {}
        if 'alerting' in config_data:
            specialized_configs['alerting'] = AlertingConfig(**config_data['alerting'])
        if 'storage_monitoring' in config_data:
            specialized_configs['storage_monitoring'] = StorageConfig(**config_data['storage_monitoring'])
        if 'performance' in config_data:
            specialized_configs['performance'] = PerformanceConfig(**config_data['performance'])
        if 'security' in config_data:
            specialized_configs['security'] = SecurityConfig(**config_data['security'])
        
        # Create main config
        main_config = {k: v for k, v in config_data.items() 
                      if k not in ['model_training', 'quantization', 'mobile_export', 
                                  'testing', 'deployment', 'monitoring', 'storage', 'compute',
                                  'alerting', 'storage_monitoring', 'performance', 'security']}
        
        return GuardConfig(**main_config, **component_configs, **specialized_configs)
    
    def save_config(self, config: GuardConfig = None):
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        if not self.config_path:
            self.config_path = "pipeline_guard_config.yaml"
        
        config_dict = asdict(config)
        
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            print(f"Configuration saved to {self.config_path}")
        
        except Exception as e:
            print(f"Error saving config to {self.config_path}: {e}")
    
    def get_config(self) -> GuardConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def create_sample_config(self, output_path: str = "pipeline_guard_config_sample.yaml"):
        """Create a sample configuration file."""
        sample_config = GuardConfig()
        
        # Add sample values
        sample_config.alerting.slack_webhook = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        sample_config.alerting.email_recipients = ["admin@example.com", "team@example.com"]
        sample_config.alerting.email_smtp_server = "smtp.example.com:587"
        
        # Enable advanced features for sample
        sample_config.ml_anomaly_detection = True
        sample_config.predictive_scaling = True
        sample_config.chaos_engineering = False  # Keep disabled by default
        
        config_dict = asdict(sample_config)
        
        # Add comments for YAML
        config_with_comments = {
            "# Pipeline Guard Configuration": None,
            "# General Settings": None,
            "guard_name": config_dict["guard_name"],
            "log_level": config_dict["log_level"] + "  # DEBUG, INFO, WARNING, ERROR, CRITICAL",
            "log_file": config_dict["log_file"],
            "state_file": config_dict["state_file"],
            
            "\n# Component Configurations": None,
            "model_training": config_dict["model_training"],
            "quantization": config_dict["quantization"],
            "mobile_export": config_dict["mobile_export"],
            "testing": config_dict["testing"],
            "deployment": config_dict["deployment"],
            "monitoring": config_dict["monitoring"],
            "storage": config_dict["storage"],
            "compute": config_dict["compute"],
            
            "\n# Alerting Configuration": None,
            "alerting": config_dict["alerting"],
            
            "\n# Monitoring Configurations": None,
            "storage_monitoring": config_dict["storage_monitoring"],
            "performance": config_dict["performance"],
            "security": config_dict["security"],
            
            "\n# Advanced Features": None,
            "ml_anomaly_detection": config_dict["ml_anomaly_detection"],
            "predictive_scaling": config_dict["predictive_scaling"],
            "chaos_engineering": config_dict["chaos_engineering"],
        }
        
        # Remove None values (comments)
        clean_config = {k: v for k, v in config_with_comments.items() if v is not None}
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(clean_config, f, default_flow_style=False, indent=2)
            
            print(f"Sample configuration created at {output_path}")
        
        except Exception as e:
            print(f"Error creating sample config: {e}")
    
    @staticmethod
    def validate_config(config: GuardConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate intervals
        components = [
            config.model_training, config.quantization, config.mobile_export,
            config.testing, config.deployment, config.monitoring,
            config.storage, config.compute
        ]
        
        for i, component in enumerate(components):
            component_name = ['model_training', 'quantization', 'mobile_export',
                            'testing', 'deployment', 'monitoring', 'storage', 'compute'][i]
            
            if component.check_interval_seconds < 10:
                issues.append(f"{component_name}: check_interval_seconds too low (minimum 10)")
            
            if component.timeout_seconds > component.check_interval_seconds:
                issues.append(f"{component_name}: timeout_seconds exceeds check_interval_seconds")
            
            if component.max_retries < 1 or component.max_retries > 10:
                issues.append(f"{component_name}: max_retries should be between 1 and 10")
        
        # Validate storage thresholds
        if config.storage_monitoring.disk_usage_warning >= config.storage_monitoring.disk_usage_critical:
            issues.append("storage_monitoring: warning threshold must be less than critical threshold")
        
        # Validate performance thresholds
        if config.performance.cpu_threshold > 100 or config.performance.cpu_threshold < 0:
            issues.append("performance: cpu_threshold must be between 0 and 100")
        
        if config.performance.memory_threshold > 100 or config.performance.memory_threshold < 0:
            issues.append("performance: memory_threshold must be between 0 and 100")
        
        # Validate alerting
        if config.alerting.enabled:
            if not config.alerting.slack_webhook and not config.alerting.email_recipients:
                issues.append("alerting: enabled but no notification methods configured")
        
        return issues


def main():
    """CLI for configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Guard Configuration Manager")
    parser.add_argument("--create-sample", help="Create sample configuration file")
    parser.add_argument("--validate", help="Validate configuration file")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    if args.create_sample:
        manager = ConfigManager()
        manager.create_sample_config(args.create_sample)
        return
    
    if args.validate:
        manager = ConfigManager(args.validate)
        issues = ConfigManager.validate_config(manager.get_config())
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid!")
        return
    
    if args.config:
        manager = ConfigManager(args.config)
        config = manager.get_config()
        print(f"Loaded configuration from {args.config}")
        print(f"Guard name: {config.guard_name}")
        print(f"Log level: {config.log_level}")
    else:
        print("No action specified. Use --help for options.")


if __name__ == "__main__":
    main()