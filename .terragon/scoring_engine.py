#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Scoring Engine
Implements WSJF + ICE + Technical Debt composite scoring for value prioritization
"""

import json
import yaml
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import re


@dataclass
class ValueItem:
    """Represents a discovered value opportunity."""
    id: str
    title: str
    description: str
    category: str
    source: str
    files: List[str]
    estimated_effort: float  # in hours
    created_at: str
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class ScoreComponents:
    """Individual scoring components."""
    wsjf: float
    ice: float
    technical_debt: float
    security_boost: float = 1.0
    compliance_boost: float = 1.0
    performance_boost: float = 1.0
    composite: float = 0.0


class ScoringEngine:
    """Advanced scoring engine with WSJF, ICE, and Technical Debt integration."""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        """Initialize scoring engine with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.learning_data = self._load_learning_data()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _load_learning_data(self) -> Dict[str, Any]:
        """Load historical learning data."""
        learning_path = self.config_path.parent / "learning_data.json"
        try:
            with open(learning_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"accuracy_history": [], "effort_history": []}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if file not found."""
        return {
            "scoring": {
                "weights": {
                    "maturing": {
                        "wsjf": 0.6,
                        "ice": 0.1,
                        "technicalDebt": 0.2,
                        "security": 0.1
                    }
                },
                "current_maturity": "maturing",
                "thresholds": {
                    "minScore": 10,
                    "maxRisk": 0.8,
                    "securityBoost": 2.0,
                    "complianceBoost": 1.8,
                    "performanceBoost": 1.5
                }
            }
        }
    
    def calculate_wsjf(self, item: ValueItem) -> float:
        """Calculate Weighted Shortest Job First score."""
        # User/Business Value (1-100 scale)
        user_value = self._assess_user_business_value(item)
        
        # Time Criticality (1-100 scale)  
        time_criticality = self._assess_time_criticality(item)
        
        # Risk Reduction (1-100 scale)
        risk_reduction = self._assess_risk_reduction(item)
        
        # Opportunity Enablement (1-100 scale)
        opportunity_enablement = self._assess_opportunity_enablement(item)
        
        # Cost of Delay = sum of value components
        cost_of_delay = (
            user_value * 0.4 +
            time_criticality * 0.3 +
            risk_reduction * 0.2 +
            opportunity_enablement * 0.1
        )
        
        # Job Size (effort in story points, normalized)
        job_size = max(item.estimated_effort, 0.5)  # Avoid division by zero
        
        # WSJF = Cost of Delay / Job Size
        wsjf = cost_of_delay / job_size
        
        return round(wsjf, 2)
    
    def calculate_ice(self, item: ValueItem) -> float:
        """Calculate Impact, Confidence, Ease score."""
        # Impact (1-10 scale)
        impact = self._assess_impact(item)
        
        # Confidence (1-10 scale)
        confidence = self._assess_confidence(item)
        
        # Ease (1-10 scale)
        ease = self._assess_ease(item)
        
        # ICE = Impact × Confidence × Ease
        ice = impact * confidence * ease
        
        return round(ice, 2)
    
    def calculate_technical_debt_score(self, item: ValueItem) -> float:
        """Calculate technical debt scoring."""
        # Debt Impact (maintenance hours saved)
        debt_impact = self._assess_debt_impact(item)
        
        # Debt Interest (future cost if not addressed)
        debt_interest = self._assess_debt_interest(item)
        
        # Hotspot Multiplier (based on file activity)
        hotspot_multiplier = self._assess_hotspot_multiplier(item)
        
        # Technical Debt Score
        debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        
        return round(debt_score, 2)
    
    def calculate_composite_score(self, item: ValueItem) -> ScoreComponents:
        """Calculate comprehensive composite score."""
        # Calculate individual components
        wsjf = self.calculate_wsjf(item)
        ice = self.calculate_ice(item)
        technical_debt = self.calculate_technical_debt_score(item)
        
        # Get adaptive weights
        weights = self._get_adaptive_weights()
        
        # Normalize scores to 0-100 scale
        normalized_wsjf = min(wsjf * 10, 100)  # WSJF typically 0-10
        normalized_ice = min(ice / 10, 100)     # ICE typically 0-1000
        normalized_debt = min(technical_debt, 100)  # Debt score 0-100
        
        # Calculate base composite score
        composite = (
            weights["wsjf"] * normalized_wsjf +
            weights["ice"] * normalized_ice +
            weights["technicalDebt"] * normalized_debt
        )
        
        # Apply boosts and penalties
        security_boost = self._get_security_boost(item)
        compliance_boost = self._get_compliance_boost(item)
        performance_boost = self._get_performance_boost(item)
        
        # Final composite score
        final_composite = composite * security_boost * compliance_boost * performance_boost
        
        return ScoreComponents(
            wsjf=wsjf,
            ice=ice,
            technical_debt=technical_debt,
            security_boost=security_boost,
            compliance_boost=compliance_boost,
            performance_boost=performance_boost,
            composite=round(final_composite, 2)
        )
    
    def _assess_user_business_value(self, item: ValueItem) -> float:
        """Assess user/business value component."""
        score = 30  # Base score
        
        # Category-based scoring
        if item.category in ["security", "critical-bug"]:
            score += 40
        elif item.category in ["performance", "mobile-optimization"]:
            score += 30
        elif item.category in ["feature", "enhancement"]:
            score += 20
        elif item.category in ["documentation", "refactoring"]:
            score += 10
        
        # Mobile AI specific bonuses
        mobile_keywords = ["mobile", "inference", "quantization", "compression"]
        if any(keyword in item.description.lower() for keyword in mobile_keywords):
            score += 15
        
        return min(score, 100)
    
    def _assess_time_criticality(self, item: ValueItem) -> float:
        """Assess time criticality component."""
        score = 20  # Base score
        
        # Security items are time-critical
        if item.category == "security":
            score += 50
        
        # Performance issues in production
        if "performance" in item.category and "production" in item.description.lower():
            score += 30
        
        # Dependency updates with security fixes
        if "dependency" in item.category and "security" in item.description.lower():
            score += 40
        
        return min(score, 100)
    
    def _assess_risk_reduction(self, item: ValueItem) -> float:
        """Assess risk reduction component."""
        score = 20  # Base score
        
        # Security and compliance items reduce risk
        if item.category in ["security", "compliance"]:
            score += 40
        
        # Technical debt reduction
        if "refactor" in item.description.lower() or "debt" in item.tags:
            score += 25
        
        # Test coverage improvements
        if "test" in item.category or "coverage" in item.description.lower():
            score += 20
        
        return min(score, 100)
    
    def _assess_opportunity_enablement(self, item: ValueItem) -> float:
        """Assess opportunity enablement component."""
        score = 15  # Base score
        
        # Infrastructure and tooling improvements
        if item.category in ["infrastructure", "tooling", "automation"]:
            score += 30
        
        # Platform/framework upgrades
        if "upgrade" in item.description.lower():
            score += 25
        
        # Development productivity improvements
        if "developer" in item.description.lower() or "dx" in item.tags:
            score += 20
        
        return min(score, 100)
    
    def _assess_impact(self, item: ValueItem) -> float:
        """Assess business impact (1-10 scale)."""
        # Map business value to 1-10 scale
        user_value = self._assess_user_business_value(item)
        return round(user_value / 10, 1)
    
    def _assess_confidence(self, item: ValueItem) -> float:
        """Assess execution confidence (1-10 scale)."""
        confidence = 7.0  # Base confidence
        
        # Adjust based on complexity
        if item.estimated_effort <= 2:
            confidence += 1.5  # Simple tasks
        elif item.estimated_effort >= 8:
            confidence -= 2.0  # Complex tasks
        
        # Adjust based on category
        if item.category in ["documentation", "configuration"]:
            confidence += 1.0
        elif item.category in ["architecture", "major-refactor"]:
            confidence -= 1.5
        
        # Learning from history
        historical_accuracy = self._get_historical_accuracy(item.category)
        confidence *= historical_accuracy
        
        return max(1.0, min(confidence, 10.0))
    
    def _assess_ease(self, item: ValueItem) -> float:
        """Assess implementation ease (1-10 scale)."""
        # Inverse of effort (normalized)
        if item.estimated_effort <= 1:
            ease = 9.0
        elif item.estimated_effort <= 2:
            ease = 8.0
        elif item.estimated_effort <= 4:
            ease = 6.0
        elif item.estimated_effort <= 8:
            ease = 4.0
        else:
            ease = 2.0
        
        # Adjust for dependencies
        if len(item.files) > 5:
            ease -= 1.0  # Many files affected
        
        return max(1.0, min(ease, 10.0))
    
    def _assess_debt_impact(self, item: ValueItem) -> float:
        """Assess technical debt impact."""
        impact = 20  # Base impact
        
        # High-churn files have higher impact
        if any("core" in f or "main" in f for f in item.files):
            impact += 30
        
        # Performance improvements
        if "performance" in item.category:
            impact += 25
        
        # Code quality improvements
        if item.category in ["refactoring", "cleanup"]:
            impact += 20
        
        return impact
    
    def _assess_debt_interest(self, item: ValueItem) -> float:
        """Assess future cost of not addressing debt."""
        interest = 10  # Base interest
        
        # Security debt compounds quickly
        if item.category == "security":
            interest += 40
        
        # Performance debt affects user experience
        if "performance" in item.category:
            interest += 25
        
        # Maintenance debt slows development
        if "maintenance" in item.tags:
            interest += 15
        
        return interest
    
    def _assess_hotspot_multiplier(self, item: ValueItem) -> float:
        """Assess hotspot multiplier based on file activity."""
        multiplier = 1.0
        
        # Check if files are in critical paths
        critical_paths = self.config.get("repository", {}).get("criticalPaths", [])
        for file_path in item.files:
            if any(critical in file_path for critical in critical_paths):
                multiplier += 0.5
        
        return min(multiplier, 3.0)
    
    def _get_adaptive_weights(self) -> Dict[str, float]:
        """Get adaptive weights based on repository maturity."""
        maturity = self.config["scoring"]["current_maturity"]
        return self.config["scoring"]["weights"][maturity]
    
    def _get_security_boost(self, item: ValueItem) -> float:
        """Get security priority boost."""
        if item.category == "security" or "security" in item.tags:
            return self.config["scoring"]["thresholds"]["securityBoost"]
        return 1.0
    
    def _get_compliance_boost(self, item: ValueItem) -> float:
        """Get compliance priority boost."""
        if item.category == "compliance" or "compliance" in item.tags:
            return self.config["scoring"]["thresholds"]["complianceBoost"]
        return 1.0
    
    def _get_performance_boost(self, item: ValueItem) -> float:
        """Get performance priority boost."""
        if "performance" in item.category or "performance" in item.tags:
            return self.config["scoring"]["thresholds"].get("performanceBoost", 1.5)
        return 1.0
    
    def _get_historical_accuracy(self, category: str) -> float:
        """Get historical accuracy for category."""
        history = self.learning_data.get("accuracy_history", [])
        category_history = [h for h in history if h.get("category") == category]
        
        if not category_history:
            return 1.0  # Default confidence
        
        # Calculate average accuracy
        accuracies = [h.get("accuracy", 1.0) for h in category_history[-10:]]  # Last 10
        return sum(accuracies) / len(accuracies)
    
    def update_learning_data(self, item: ValueItem, actual_effort: float, 
                           actual_impact: float) -> None:
        """Update learning data with execution results."""
        # Calculate accuracy metrics
        effort_accuracy = min(item.estimated_effort, actual_effort) / max(item.estimated_effort, actual_effort)
        
        # Predicted impact estimation (simplified)
        predicted_impact = self._assess_user_business_value(item)
        impact_accuracy = min(predicted_impact, actual_impact) / max(predicted_impact, actual_impact)
        
        # Store learning data
        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "item_id": item.id,
            "category": item.category,
            "predicted_effort": item.estimated_effort,
            "actual_effort": actual_effort,
            "effort_accuracy": effort_accuracy,
            "predicted_impact": predicted_impact,
            "actual_impact": actual_impact,
            "impact_accuracy": impact_accuracy
        }
        
        self.learning_data["accuracy_history"].append(learning_entry)
        
        # Keep only recent entries
        max_entries = self.config.get("learning", {}).get("retention", {}).get("maxHistoryEntries", 1000)
        if len(self.learning_data["accuracy_history"]) > max_entries:
            self.learning_data["accuracy_history"] = self.learning_data["accuracy_history"][-max_entries:]
        
        # Save updated learning data
        self._save_learning_data()
    
    def _save_learning_data(self) -> None:
        """Save learning data to file."""
        learning_path = self.config_path.parent / "learning_data.json"
        with open(learning_path, 'w') as f:
            json.dump(self.learning_data, f, indent=2)


def main():
    """Example usage of the scoring engine."""
    engine = ScoringEngine()
    
    # Example value item
    example_item = ValueItem(
        id="perf-001",
        title="Optimize mobile inference pipeline",
        description="Reduce inference latency for mobile quantized models by 20%",
        category="performance",
        source="static_analysis",
        files=["src/mobile_multimodal/core.py"],
        estimated_effort=6.0,
        created_at=datetime.now().isoformat(),
        tags=["mobile", "performance", "quantization"],
        metadata={"priority": "high", "complexity": "medium"}
    )
    
    # Calculate scores
    scores = engine.calculate_composite_score(example_item)
    
    print("Scoring Results:")
    print(f"WSJF Score: {scores.wsjf}")
    print(f"ICE Score: {scores.ice}")
    print(f"Technical Debt Score: {scores.technical_debt}")
    print(f"Composite Score: {scores.composite}")
    print(f"Security Boost: {scores.security_boost}x")
    print(f"Performance Boost: {scores.performance_boost}x")


if __name__ == "__main__":
    main()