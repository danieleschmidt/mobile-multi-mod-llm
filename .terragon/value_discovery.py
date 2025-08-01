#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery System
Continuously harvests signals from multiple sources to identify improvement opportunities
"""

import json
import yaml
import re
import subprocess
import ast
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import os
import hashlib

from scoring_engine import ValueItem, ScoringEngine


@dataclass
class DiscoverySignal:
    """Represents a discovered signal that might indicate value opportunity."""
    source: str
    type: str
    file_path: str
    line_number: Optional[int]
    content: str
    metadata: Dict[str, Any]
    confidence: float
    timestamp: str


class ValueDiscoveryEngine:
    """Advanced value discovery system with multi-source signal harvesting."""
    
    def __init__(self, repo_path: str = ".", config_path: str = ".terragon/value-config.yaml"):
        """Initialize value discovery engine."""
        self.repo_path = Path(repo_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.scoring_engine = ScoringEngine(config_path)
        self.discovered_items = []
        self.processed_signals = set()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def discover_all_opportunities(self) -> List[ValueItem]:
        """Main discovery method - harvests from all configured sources."""
        signals = []
        
        # Harvest signals from all sources
        discovery_sources = self.config.get("discovery", {}).get("sources", [])
        
        if "gitHistory" in discovery_sources:
            signals.extend(self._harvest_git_history_signals())
        
        if "staticAnalysis" in discovery_sources:
            signals.extend(self._harvest_static_analysis_signals())
        
        if "codeComments" in discovery_sources:
            signals.extend(self._harvest_code_comment_signals())
        
        if "dependencyUpdates" in discovery_sources:
            signals.extend(self._harvest_dependency_signals())
        
        if "performanceMonitoring" in discovery_sources:
            signals.extend(self._harvest_performance_signals())
        
        if "vulnerabilityDatabases" in discovery_sources:
            signals.extend(self._harvest_security_signals())
        
        # Convert signals to value items
        value_items = self._signals_to_value_items(signals)
        
        # Score and prioritize
        scored_items = []
        for item in value_items:
            scores = self.scoring_engine.calculate_composite_score(item)
            item.metadata["scores"] = asdict(scores)
            scored_items.append(item)
        
        # Sort by composite score
        scored_items.sort(key=lambda x: x.metadata["scores"]["composite"], reverse=True)
        
        self.discovered_items = scored_items
        return scored_items
    
    def _harvest_git_history_signals(self) -> List[DiscoverySignal]:
        """Harvest signals from Git history analysis."""
        signals = []
        
        try:
            # Get recent commits
            result = subprocess.run([
                "git", "log", "--oneline", "--since=3 months ago", 
                "--grep=TODO\\|FIXME\\|HACK\\|temporary\\|quick fix"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    commit_hash, message = line.split(' ', 1)
                    signals.append(DiscoverySignal(
                        source="git_history",
                        type="commit_message",
                        file_path="",
                        line_number=None,
                        content=message,
                        metadata={"commit": commit_hash},
                        confidence=0.6,
                        timestamp=datetime.now().isoformat()
                    ))
        except subprocess.CalledProcessError:
            pass
        
        # Analyze file churn and complexity
        try:
            # Get files with high churn
            result = subprocess.run([
                "git", "log", "--name-only", "--pretty=format:", "--since=6 months ago"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            file_changes = {}
            for line in result.stdout.strip().split('\n'):
                if line and line.endswith('.py'):
                    file_changes[line] = file_changes.get(line, 0) + 1
            
            # Identify hotspots (high churn files)
            for file_path, change_count in file_changes.items():
                if change_count > 10:  # High churn threshold
                    signals.append(DiscoverySignal(
                        source="git_history",
                        type="hotspot",
                        file_path=file_path,
                        line_number=None,
                        content=f"High churn file with {change_count} changes",
                        metadata={"change_count": change_count},
                        confidence=0.8,
                        timestamp=datetime.now().isoformat()
                    ))
        except subprocess.CalledProcessError:
            pass
        
        return signals
    
    def _harvest_static_analysis_signals(self) -> List[DiscoverySignal]:
        """Harvest signals from static analysis tools."""
        signals = []
        
        # Run flake8 for code quality issues
        try:
            result = subprocess.run([
                "flake8", "src/", "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    parts = line.split(':', 3)
                    if len(parts) >= 4:
                        file_path, line_num, col, message = parts
                        signals.append(DiscoverySignal(
                            source="static_analysis",
                            type="code_quality",
                            file_path=file_path,
                            line_number=int(line_num),
                            content=message.strip(),
                            metadata={"tool": "flake8", "column": col},
                            confidence=0.7,
                            timestamp=datetime.now().isoformat()
                        ))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Run mypy for type issues
        try:
            result = subprocess.run([
                "mypy", "src/", "--show-error-codes"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            for line in result.stdout.strip().split('\n'):
                if ':' in line and 'error:' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path, line_num, message = parts
                        signals.append(DiscoverySignal(
                            source="static_analysis",
                            type="type_error",
                            file_path=file_path,
                            line_number=int(line_num) if line_num.isdigit() else None,
                            content=message.strip(),
                            metadata={"tool": "mypy"},
                            confidence=0.8,
                            timestamp=datetime.now().isoformat()
                        ))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return signals
    
    def _harvest_code_comment_signals(self) -> List[DiscoverySignal]:
        """Harvest signals from code comments and TODO markers."""
        signals = []
        
        # Pattern for TODO/FIXME/HACK comments
        patterns = self.config.get("discovery", {}).get("patterns", {})
        todo_markers = patterns.get("todoMarkers", ["TODO", "FIXME", "HACK", "XXX"])
        debt_indicators = patterns.get("debtIndicators", ["quick fix", "temporary", "workaround"])
        
        # Search Python files
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line_lower = line.lower()
                        
                        # Check for TODO markers
                        for marker in todo_markers:
                            if marker.lower() in line_lower:
                                signals.append(DiscoverySignal(
                                    source="code_comments",
                                    type="todo_marker",
                                    file_path=str(py_file.relative_to(self.repo_path)),
                                    line_number=line_num,
                                    content=line.strip(),
                                    metadata={"marker": marker},
                                    confidence=0.9,
                                    timestamp=datetime.now().isoformat()
                                ))
                        
                        # Check for debt indicators
                        for indicator in debt_indicators:
                            if indicator.lower() in line_lower:
                                signals.append(DiscoverySignal(
                                    source="code_comments",
                                    type="debt_indicator",
                                    file_path=str(py_file.relative_to(self.repo_path)),
                                    line_number=line_num,
                                    content=line.strip(),
                                    metadata={"indicator": indicator},
                                    confidence=0.7,
                                    timestamp=datetime.now().isoformat()
                                ))
            except (IOError, UnicodeDecodeError):
                continue
        
        return signals
    
    def _harvest_dependency_signals(self) -> List[DiscoverySignal]:
        """Harvest signals from dependency analysis."""
        signals = []
        
        # Check for outdated dependencies
        requirements_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
        
        for req_file in requirements_files:
            req_path = self.repo_path / req_file
            if req_path.exists():
                signals.append(DiscoverySignal(
                    source="dependency_updates",
                    type="dependency_check",
                    file_path=req_file,
                    line_number=None,
                    content=f"Check for outdated dependencies in {req_file}",
                    metadata={"file": req_file},
                    confidence=0.5,
                    timestamp=datetime.now().isoformat()
                ))
        
        # Check for security vulnerabilities with safety
        try:
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0 and result.stdout:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    for vuln in vulnerabilities:
                        signals.append(DiscoverySignal(
                            source="dependency_updates",
                            type="security_vulnerability",
                            file_path="requirements.txt",
                            line_number=None,
                            content=f"Security vulnerability in {vuln.get('package', 'unknown')}: {vuln.get('advisory', 'No details')}",
                            metadata={"vulnerability": vuln},
                            confidence=0.95,
                            timestamp=datetime.now().isoformat()
                        ))
                except json.JSONDecodeError:
                    pass
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return signals
    
    def _harvest_performance_signals(self) -> List[DiscoverySignal]:
        """Harvest signals from performance analysis."""
        signals = []
        
        # Look for performance-related files and patterns
        perf_keywords = ["benchmark", "profiler", "performance", "memory", "cpu"]
        
        # Check if performance tests exist but are failing
        perf_test_dir = self.repo_path / "tests" / "benchmarks"
        if perf_test_dir.exists():
            signals.append(DiscoverySignal(
                source="performance_monitoring",
                type="performance_test",
                file_path="tests/benchmarks/",
                line_number=None,
                content="Performance benchmarks exist - ensure they're passing and optimized",
                metadata={"test_dir": str(perf_test_dir)},
                confidence=0.6,
                timestamp=datetime.now().isoformat()
            ))
        
        # Look for performance comments in code
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line_lower = line.lower()
                        if any(keyword in line_lower for keyword in perf_keywords):
                            if "slow" in line_lower or "optimize" in line_lower:
                                signals.append(DiscoverySignal(
                                    source="performance_monitoring",
                                    type="performance_comment",
                                    file_path=str(py_file.relative_to(self.repo_path)),
                                    line_number=line_num,
                                    content=line.strip(),
                                    metadata={"keywords": perf_keywords},
                                    confidence=0.7,
                                    timestamp=datetime.now().isoformat()
                                ))
            except (IOError, UnicodeDecodeError):
                continue
        
        return signals
    
    def _harvest_security_signals(self) -> List[DiscoverySignal]:
        """Harvest signals from security analysis."""
        signals = []
        
        # Run bandit for security issues
        try:
            result = subprocess.run([
                "bandit", "-r", "src/", "-f", "json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                try:
                    bandit_results = json.loads(result.stdout)
                    for issue in bandit_results.get("results", []):
                        signals.append(DiscoverySignal(
                            source="vulnerability_databases",
                            type="security_issue",
                            file_path=issue.get("filename", ""),
                            line_number=issue.get("line_number"),
                            content=f"{issue.get('issue_text', '')} - {issue.get('test_name', '')}",
                            metadata={
                                "severity": issue.get("issue_severity"),
                                "confidence": issue.get("issue_confidence"),
                                "test_id": issue.get("test_id")
                            },
                            confidence=0.8,
                            timestamp=datetime.now().isoformat()
                        ))
                except json.JSONDecodeError:
                    pass
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return signals
    
    def _signals_to_value_items(self, signals: List[DiscoverySignal]) -> List[ValueItem]:
        """Convert discovery signals into actionable value items."""
        value_items = []
        
        # Group signals by type and file
        signal_groups = {}
        for signal in signals:
            key = f"{signal.type}_{signal.file_path}"
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        # Convert each group to a value item
        for group_key, group_signals in signal_groups.items():
            if not group_signals:
                continue
            
            primary_signal = group_signals[0]
            item_id = self._generate_item_id(primary_signal)
            
            # Skip if already processed
            if item_id in self.processed_signals:
                continue
            
            # Determine category and effort
            category, estimated_effort = self._categorize_signal_group(group_signals)
            
            # Generate title and description
            title, description = self._generate_item_details(group_signals, category)
            
            # Collect affected files
            affected_files = list(set(s.file_path for s in group_signals if s.file_path))
            
            # Generate tags
            tags = self._generate_tags(group_signals, category)
            
            value_item = ValueItem(
                id=item_id,
                title=title,
                description=description,
                category=category,
                source=primary_signal.source,
                files=affected_files,
                estimated_effort=estimated_effort,
                created_at=datetime.now().isoformat(),
                tags=tags,
                metadata={
                    "signal_count": len(group_signals),
                    "confidence": sum(s.confidence for s in group_signals) / len(group_signals),
                    "signals": [asdict(s) for s in group_signals[:3]]  # Keep first 3 for reference
                }
            )
            
            value_items.append(value_item)
            self.processed_signals.add(item_id)
        
        return value_items
    
    def _generate_item_id(self, signal: DiscoverySignal) -> str:
        """Generate unique ID for value item."""
        content = f"{signal.type}_{signal.file_path}_{signal.content}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _categorize_signal_group(self, signals: List[DiscoverySignal]) -> Tuple[str, float]:
        """Categorize signal group and estimate effort."""
        signal_types = [s.type for s in signals]
        
        # Security signals
        if any("security" in t or "vulnerability" in t for t in signal_types):
            return "security", 2.0
        
        # Performance signals
        if any("performance" in t for t in signal_types):
            return "performance", 4.0
        
        # Code quality signals
        if any("code_quality" in t or "type_error" in t for t in signal_types):
            return "code-quality", 1.5
        
        # Technical debt signals
        if any("debt" in t or "todo" in t for t in signal_types):
            return "technical-debt", 3.0
        
        # Dependency signals
        if any("dependency" in t for t in signal_types):
            return "dependencies", 1.0
        
        # Default
        return "maintenance", 2.0
    
    def _generate_item_details(self, signals: List[DiscoverySignal], category: str) -> Tuple[str, str]:
        """Generate title and description for value item."""
        primary_signal = signals[0]
        signal_count = len(signals)
        
        # Generate title based on category
        if category == "security":
            title = f"Fix security issues in {primary_signal.file_path or 'codebase'}"
        elif category == "performance":
            title = f"Optimize performance in {primary_signal.file_path or 'codebase'}"
        elif category == "code-quality":
            title = f"Improve code quality in {primary_signal.file_path or 'codebase'}"
        elif category == "technical-debt":
            title = f"Address technical debt in {primary_signal.file_path or 'codebase'}"
        elif category == "dependencies":
            title = f"Update dependencies in {primary_signal.file_path or 'project'}"
        else:
            title = f"Improve {category} in {primary_signal.file_path or 'codebase'}"
        
        # Generate description
        description_parts = []
        if signal_count > 1:
            description_parts.append(f"Found {signal_count} related issues:")
        
        for i, signal in enumerate(signals[:3]):  # Limit to first 3
            desc = signal.content
            if signal.line_number:
                desc += f" (line {signal.line_number})"
            description_parts.append(f"- {desc}")
        
        if len(signals) > 3:
            description_parts.append(f"... and {len(signals) - 3} more issues")
        
        return title, "\n".join(description_parts)
    
    def _generate_tags(self, signals: List[DiscoverySignal], category: str) -> List[str]:
        """Generate tags for value item."""
        tags = [category]
        
        # Add source-based tags
        sources = set(s.source for s in signals)
        tags.extend(sources)
        
        # Add mobile-specific tags if relevant
        mobile_keywords = ["mobile", "quantization", "inference", "compression"]
        for signal in signals:
            if any(keyword in signal.content.lower() for keyword in mobile_keywords):
                tags.extend(mobile_keywords)
                break
        
        return list(set(tags))
    
    def save_discovery_results(self, output_path: str = ".terragon/discovery_results.json") -> None:
        """Save discovery results to file."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_items": len(self.discovered_items),
            "items": [asdict(item) for item in self.discovered_items]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_next_best_value_item(self) -> Optional[ValueItem]:
        """Get the highest-scoring unprocessed value item."""
        if not self.discovered_items:
            self.discover_all_opportunities()
        
        # Filter by minimum score threshold
        min_score = self.config.get("scoring", {}).get("thresholds", {}).get("minScore", 10)
        
        for item in self.discovered_items:
            composite_score = item.metadata.get("scores", {}).get("composite", 0)
            if composite_score >= min_score:
                return item
        
        return None


def main():
    """Example usage of the value discovery system."""
    discovery_engine = ValueDiscoveryEngine()
    
    print("🔍 Starting autonomous value discovery...")
    opportunities = discovery_engine.discover_all_opportunities()
    
    print(f"\n📊 Discovered {len(opportunities)} value opportunities:")
    
    for i, item in enumerate(opportunities[:10], 1):  # Show top 10
        scores = item.metadata.get("scores", {})
        print(f"\n{i}. {item.title}")
        print(f"   Category: {item.category}")
        print(f"   Composite Score: {scores.get('composite', 0):.1f}")
        print(f"   Estimated Effort: {item.estimated_effort} hours")
        print(f"   Files: {len(item.files)} affected")
    
    # Save results
    discovery_engine.save_discovery_results()
    print(f"\n💾 Results saved to .terragon/discovery_results.json")
    
    # Get next best item
    next_item = discovery_engine.get_next_best_value_item()
    if next_item:
        print(f"\n🎯 Next Best Value Item:")
        print(f"   {next_item.title}")
        print(f"   Score: {next_item.metadata.get('scores', {}).get('composite', 0):.1f}")


if __name__ == "__main__":
    main()