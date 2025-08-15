"""Advanced logging system for Pipeline Guard with structured logging and analysis."""

import json
import logging
import logging.handlers
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import gzip
import sqlite3
import threading
from enum import Enum


class LogLevel(Enum):
    """Enhanced log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    ALERT = 60  # Custom level for alerts


class LogPattern(Enum):
    """Common log patterns to detect."""
    ERROR_SPIKE = "error_spike"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_THREAT = "security_threat"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PIPELINE_FAILURE = "pipeline_failure"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    component: str
    metadata: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LogAnalysisResult:
    """Log analysis result."""
    pattern_type: LogPattern
    severity: str
    count: int
    time_window: str
    affected_components: List[str]
    sample_messages: List[str]
    recommendations: List[str]
    first_occurrence: datetime
    last_occurrence: datetime


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self):
        super().__init__()
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'hostname': self.hostname,
            'process_id': os.getpid(),
            'thread_id': threading.get_ident(),
        }
        
        # Add component information if available
        if hasattr(record, 'component'):
            log_data['component'] = record.component
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # Add metadata if available
        if hasattr(record, 'metadata'):
            log_data['metadata'] = record.metadata
        
        # Add exception information
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add file and line information in debug mode
        if record.levelno <= logging.DEBUG:
            log_data['file'] = {
                'name': record.filename,
                'line': record.lineno,
                'function': record.funcName
            }
        
        return json.dumps(log_data)


class GuardLogHandler(logging.Handler):
    """Custom log handler for pipeline guard with analysis capabilities."""
    
    def __init__(self, db_path: str = "pipeline_logs.db", max_buffer_size: int = 1000):
        """Initialize guard log handler.
        
        Args:
            db_path: Path to SQLite database for log storage
            max_buffer_size: Maximum buffer size before flushing
        """
        super().__init__()
        self.db_path = db_path
        self.max_buffer_size = max_buffer_size
        self.log_buffer = deque(maxlen=max_buffer_size)
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Analysis patterns
        self.error_patterns = {
            LogPattern.ERROR_SPIKE: [
                r'error', r'exception', r'failed', r'timeout', r'connection.*refused'
            ],
            LogPattern.PERFORMANCE_DEGRADATION: [
                r'slow', r'timeout', r'latency.*high', r'response.*time.*exceeded'
            ],
            LogPattern.SECURITY_THREAT: [
                r'unauthorized', r'authentication.*failed', r'access.*denied',
                r'suspicious.*activity', r'security.*violation'
            ],
            LogPattern.RESOURCE_EXHAUSTION: [
                r'out.*of.*memory', r'disk.*full', r'no.*space', r'resource.*exhausted'
            ],
            LogPattern.PIPELINE_FAILURE: [
                r'pipeline.*failed', r'training.*failed', r'export.*failed',
                r'quantization.*failed', r'deployment.*failed'
            ],
            LogPattern.ANOMALOUS_BEHAVIOR: [
                r'unexpected', r'anomaly', r'unusual.*pattern', r'deviation'
            ]
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        for pattern_type, patterns in self.error_patterns.items():
            self.compiled_patterns[pattern_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def _init_database(self):
        """Initialize SQLite database for log storage."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    logger_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    component TEXT,
                    correlation_id TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp 
                ON logs(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_level_component 
                ON logs(level, component)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_correlation_id 
                ON logs(correlation_id)
            ''')
            
            conn.commit()
        finally:
            conn.close()
    
    def emit(self, record):
        """Handle log record emission."""
        try:
            # Create structured log entry
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                component=getattr(record, 'component', 'unknown'),
                correlation_id=getattr(record, 'correlation_id', None),
                metadata=getattr(record, 'metadata', {})
            )
            
            with self._lock:
                self.log_buffer.append(log_entry)
                
                # Flush if buffer is full
                if len(self.log_buffer) >= self.max_buffer_size:
                    self._flush_logs()
                    
        except Exception:
            self.handleError(record)
    
    def _flush_logs(self):
        """Flush buffered logs to database."""
        if not self.log_buffer:
            return
        
        logs_to_flush = list(self.log_buffer)
        self.log_buffer.clear()
        
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                for log_entry in logs_to_flush:
                    cursor.execute('''
                        INSERT INTO logs (timestamp, level, logger_name, message, 
                                        component, correlation_id, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        log_entry.timestamp.isoformat(),
                        log_entry.level,
                        log_entry.logger_name,
                        log_entry.message,
                        log_entry.component,
                        log_entry.correlation_id,
                        json.dumps(log_entry.metadata)
                    ))
                
                conn.commit()
                
            finally:
                conn.close()
                
        except Exception as e:
            # Put logs back in buffer for retry
            self.log_buffer.extendleft(reversed(logs_to_flush))
            raise e
    
    def close(self):
        """Close handler and flush remaining logs."""
        with self._lock:
            self._flush_logs()
        super().close()


class LogAnalyzer:
    """Analyzes logs for patterns and anomalies."""
    
    def __init__(self, db_path: str = "pipeline_logs.db"):
        """Initialize log analyzer.
        
        Args:
            db_path: Path to SQLite database containing logs
        """
        self.db_path = db_path
        self.pattern_cache = {}
        self._lock = threading.RLock()
    
    def analyze_logs(self, time_window_hours: int = 1, 
                    min_occurrences: int = 5) -> List[LogAnalysisResult]:
        """Analyze logs for patterns and anomalies.
        
        Args:
            time_window_hours: Time window to analyze (hours)
            min_occurrences: Minimum occurrences to consider a pattern
            
        Returns:
            List of analysis results
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        # Get logs from time window
        logs = self._get_logs(start_time, end_time)
        
        if not logs:
            return []
        
        results = []
        
        # Analyze for each pattern type
        for pattern_type in LogPattern:
            result = self._analyze_pattern(logs, pattern_type, min_occurrences)
            if result:
                results.append(result)
        
        # Analyze error rate spikes
        error_spike_result = self._analyze_error_spikes(logs, time_window_hours)
        if error_spike_result:
            results.append(error_spike_result)
        
        return results
    
    def _get_logs(self, start_time: datetime, end_time: datetime) -> List[LogEntry]:
        """Get logs from database within time range."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, level, logger_name, message, component, 
                       correlation_id, metadata
                FROM logs 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            ''', (start_time.isoformat(), end_time.isoformat()))
            
            logs = []
            for row in cursor.fetchall():
                timestamp_str, level, logger_name, message, component, correlation_id, metadata_str = row
                metadata = json.loads(metadata_str) if metadata_str else {}
                
                logs.append(LogEntry(
                    timestamp=datetime.fromisoformat(timestamp_str),
                    level=level,
                    logger_name=logger_name,
                    message=message,
                    component=component or 'unknown',
                    correlation_id=correlation_id,
                    metadata=metadata
                ))
            
            return logs
            
        finally:
            conn.close()
    
    def _analyze_pattern(self, logs: List[LogEntry], pattern_type: LogPattern,
                        min_occurrences: int) -> Optional[LogAnalysisResult]:
        """Analyze logs for a specific pattern type."""
        patterns = {
            LogPattern.ERROR_SPIKE: [
                r'error', r'exception', r'failed', r'timeout'
            ],
            LogPattern.PERFORMANCE_DEGRADATION: [
                r'slow', r'timeout', r'latency.*high', r'response.*time'
            ],
            LogPattern.SECURITY_THREAT: [
                r'unauthorized', r'authentication.*failed', r'access.*denied'
            ],
            LogPattern.RESOURCE_EXHAUSTION: [
                r'out.*of.*memory', r'disk.*full', r'no.*space'
            ],
            LogPattern.PIPELINE_FAILURE: [
                r'pipeline.*failed', r'training.*failed', r'export.*failed'
            ],
            LogPattern.ANOMALOUS_BEHAVIOR: [
                r'unexpected', r'anomaly', r'unusual'
            ]
        }
        
        if pattern_type not in patterns:
            return None
        
        # Compile patterns
        compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in patterns[pattern_type]
        ]
        
        # Find matching logs
        matching_logs = []
        for log_entry in logs:
            for pattern in compiled_patterns:
                if pattern.search(log_entry.message):
                    matching_logs.append(log_entry)
                    break
        
        if len(matching_logs) < min_occurrences:
            return None
        
        # Analyze matches
        affected_components = set(log.component for log in matching_logs)
        sample_messages = list(set(log.message for log in matching_logs[:10]))
        
        # Determine severity
        error_count = sum(1 for log in matching_logs if log.level in ['ERROR', 'CRITICAL'])
        warning_count = sum(1 for log in matching_logs if log.level == 'WARNING')
        
        if error_count > len(matching_logs) * 0.7:  # >70% errors
            severity = "critical"
        elif error_count > len(matching_logs) * 0.3:  # >30% errors
            severity = "high"
        elif warning_count > len(matching_logs) * 0.5:  # >50% warnings
            severity = "medium"
        else:
            severity = "low"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(pattern_type, severity, affected_components)
        
        return LogAnalysisResult(
            pattern_type=pattern_type,
            severity=severity,
            count=len(matching_logs),
            time_window=f"{len(logs)} total logs analyzed",
            affected_components=list(affected_components),
            sample_messages=sample_messages,
            recommendations=recommendations,
            first_occurrence=min(log.timestamp for log in matching_logs),
            last_occurrence=max(log.timestamp for log in matching_logs)
        )
    
    def _analyze_error_spikes(self, logs: List[LogEntry], 
                             time_window_hours: int) -> Optional[LogAnalysisResult]:
        """Analyze for error rate spikes."""
        if not logs:
            return None
        
        # Count errors by time buckets (15-minute intervals)
        bucket_size_minutes = 15
        bucket_size_seconds = bucket_size_minutes * 60
        
        error_buckets = defaultdict(int)
        total_buckets = defaultdict(int)
        
        for log_entry in logs:
            if log_entry.level in ['ERROR', 'CRITICAL']:
                bucket_time = int(log_entry.timestamp.timestamp() // bucket_size_seconds)
                error_buckets[bucket_time] += 1
            
            bucket_time = int(log_entry.timestamp.timestamp() // bucket_size_seconds)
            total_buckets[bucket_time] += 1
        
        if not error_buckets:
            return None
        
        # Calculate error rates
        error_rates = []
        for bucket_time in total_buckets:
            error_count = error_buckets.get(bucket_time, 0)
            total_count = total_buckets[bucket_time]
            error_rate = error_count / total_count if total_count > 0 else 0
            error_rates.append((bucket_time, error_rate, error_count))
        
        if len(error_rates) < 2:
            return None
        
        # Calculate baseline error rate
        baseline_error_rate = sum(rate[1] for rate in error_rates) / len(error_rates)
        
        # Find spikes (error rate > 3x baseline)
        spike_threshold = max(baseline_error_rate * 3, 0.1)  # At least 10%
        spikes = [rate for rate in error_rates if rate[1] > spike_threshold]
        
        if not spikes:
            return None
        
        total_spike_errors = sum(spike[2] for spike in spikes)
        
        return LogAnalysisResult(
            pattern_type=LogPattern.ERROR_SPIKE,
            severity="high" if len(spikes) > len(error_rates) * 0.3 else "medium",
            count=total_spike_errors,
            time_window=f"{len(spikes)} spike periods in {time_window_hours}h",
            affected_components=["multiple"],
            sample_messages=[f"Error spike: {spike[1]:.1%} error rate" for spike in spikes[:5]],
            recommendations=[
                "Investigate recent changes or deployments",
                "Check system resources and dependencies",
                "Review error logs for common patterns",
                "Consider rollback if issue persists"
            ],
            first_occurrence=datetime.fromtimestamp(min(spike[0] for spike in spikes) * bucket_size_seconds),
            last_occurrence=datetime.fromtimestamp(max(spike[0] for spike in spikes) * bucket_size_seconds)
        )
    
    def _generate_recommendations(self, pattern_type: LogPattern, severity: str,
                                affected_components: Set[str]) -> List[str]:
        """Generate recommendations based on pattern analysis."""
        recommendations = []
        
        base_recommendations = {
            LogPattern.ERROR_SPIKE: [
                "Check recent deployments or configuration changes",
                "Review system resources (CPU, memory, disk)",
                "Examine error patterns for common root causes",
                "Consider implementing circuit breakers"
            ],
            LogPattern.PERFORMANCE_DEGRADATION: [
                "Profile application performance",
                "Check database query performance",
                "Monitor network latency and bandwidth",
                "Review resource allocation and scaling policies"
            ],
            LogPattern.SECURITY_THREAT: [
                "Review access logs and authentication failures",
                "Check for suspicious IP addresses or patterns",
                "Verify security configurations and policies",
                "Consider implementing additional security measures"
            ],
            LogPattern.RESOURCE_EXHAUSTION: [
                "Increase available resources (CPU, memory, disk)",
                "Implement resource cleanup procedures",
                "Review resource usage patterns",
                "Set up proactive monitoring and alerts"
            ],
            LogPattern.PIPELINE_FAILURE: [
                "Check pipeline configurations and dependencies",
                "Verify data integrity and availability",
                "Review pipeline execution logs",
                "Implement retry mechanisms and fallbacks"
            ],
            LogPattern.ANOMALOUS_BEHAVIOR: [
                "Investigate unusual patterns or behaviors",
                "Check for data quality issues",
                "Review recent changes to the system",
                "Implement anomaly detection and alerting"
            ]
        }
        
        recommendations.extend(base_recommendations.get(pattern_type, []))
        
        # Add severity-specific recommendations
        if severity == "critical":
            recommendations.extend([
                "Immediate attention required",
                "Consider emergency response procedures",
                "Escalate to on-call team if available"
            ])
        
        # Add component-specific recommendations
        if "training" in affected_components:
            recommendations.append("Check training data and model configurations")
        
        if "quantization" in affected_components:
            recommendations.append("Verify quantization parameters and accuracy thresholds")
        
        if "mobile_export" in affected_components:
            recommendations.append("Check mobile platform compatibility and export settings")
        
        return recommendations
    
    def get_log_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get log statistics for the specified time window."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Total log count
            cursor.execute('''
                SELECT COUNT(*) FROM logs 
                WHERE timestamp >= ? AND timestamp <= ?
            ''', (start_time.isoformat(), end_time.isoformat()))
            total_logs = cursor.fetchone()[0]
            
            # Log level distribution
            cursor.execute('''
                SELECT level, COUNT(*) FROM logs 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY level
            ''', (start_time.isoformat(), end_time.isoformat()))
            level_distribution = dict(cursor.fetchall())
            
            # Component distribution
            cursor.execute('''
                SELECT component, COUNT(*) FROM logs 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY component
                ORDER BY COUNT(*) DESC
                LIMIT 10
            ''', (start_time.isoformat(), end_time.isoformat()))
            component_distribution = dict(cursor.fetchall())
            
            # Error rate calculation
            error_count = level_distribution.get('ERROR', 0) + level_distribution.get('CRITICAL', 0)
            error_rate = (error_count / total_logs) * 100 if total_logs > 0 else 0
            
            return {
                "time_window_hours": time_window_hours,
                "total_logs": total_logs,
                "error_rate_percent": round(error_rate, 2),
                "level_distribution": level_distribution,
                "component_distribution": component_distribution,
                "logs_per_hour": round(total_logs / time_window_hours, 2) if time_window_hours > 0 else 0,
            }
            
        finally:
            conn.close()
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Remove old logs from database."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM logs WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            deleted_count = cursor.rowcount
            conn.commit()
            
            return deleted_count
            
        finally:
            conn.close()


def setup_logging(log_level: str = "INFO", log_file: str = "pipeline_guard.log",
                 enable_structured: bool = True, enable_analysis: bool = True) -> logging.Logger:
    """Setup comprehensive logging for pipeline guard.
    
    Args:
        log_level: Logging level
        log_file: Log file path
        enable_structured: Enable structured JSON logging
        enable_analysis: Enable log analysis capabilities
        
    Returns:
        Configured logger
    """
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger('pipeline_guard')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with standard formatting
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB files, keep 5 backups
    )
    
    if enable_structured:
        file_handler.setFormatter(StructuredFormatter())
    else:
        file_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    
    # Analysis handler
    if enable_analysis:
        analysis_handler = GuardLogHandler()
        analysis_handler.setLevel(logging.WARNING)  # Only store warnings and above
        logger.addHandler(analysis_handler)
    
    logger.info("Pipeline Guard logging system initialized")
    return logger


def main():
    """CLI for log analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Guard Log Analyzer")
    parser.add_argument("--analyze", action="store_true", help="Analyze recent logs")
    parser.add_argument("--stats", action="store_true", help="Show log statistics")
    parser.add_argument("--cleanup", type=int, help="Cleanup logs older than N days")
    parser.add_argument("--db", default="pipeline_logs.db", help="Database path")
    parser.add_argument("--hours", type=int, default=1, help="Time window in hours")
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.db)
    
    if args.analyze:
        results = analyzer.analyze_logs(time_window_hours=args.hours)
        
        if not results:
            print("No significant patterns found in logs")
            return
        
        print(f"Log Analysis Results ({args.hours}h window):")
        print("=" * 50)
        
        for result in results:
            print(f"\nPattern: {result.pattern_type.value}")
            print(f"Severity: {result.severity}")
            print(f"Occurrences: {result.count}")
            print(f"Affected Components: {', '.join(result.affected_components)}")
            print(f"Time Range: {result.first_occurrence} to {result.last_occurrence}")
            
            print("\nSample Messages:")
            for message in result.sample_messages[:3]:
                print(f"  - {message}")
            
            print("\nRecommendations:")
            for rec in result.recommendations[:3]:
                print(f"  - {rec}")
            print("-" * 30)
    
    elif args.stats:
        stats = analyzer.get_log_statistics(time_window_hours=args.hours)
        
        print(f"Log Statistics ({args.hours}h window):")
        print("=" * 30)
        print(f"Total Logs: {stats['total_logs']}")
        print(f"Error Rate: {stats['error_rate_percent']:.2f}%")
        print(f"Logs/Hour: {stats['logs_per_hour']:.1f}")
        
        print("\nLevel Distribution:")
        for level, count in stats['level_distribution'].items():
            print(f"  {level}: {count}")
        
        print("\nTop Components:")
        for component, count in list(stats['component_distribution'].items())[:5]:
            print(f"  {component}: {count}")
    
    elif args.cleanup:
        deleted = analyzer.cleanup_old_logs(args.cleanup)
        print(f"Deleted {deleted} old log entries")
    
    else:
        print("No action specified. Use --help for options.")


if __name__ == "__main__":
    main()