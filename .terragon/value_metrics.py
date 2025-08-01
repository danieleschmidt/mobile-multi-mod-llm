#!/usr/bin/env python3
"""
Terragon Value Metrics and Learning System
Tracks value delivery, learning, and continuous improvement
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import statistics
import math


@dataclass
class ValueMetric:
    """Represents a value delivery metric."""
    timestamp: str
    item_id: str
    category: str
    predicted_value: float
    actual_value: float
    predicted_effort: float
    actual_effort: float
    success: bool
    impact_metrics: Dict[str, Any]
    learning_feedback: Dict[str, Any]


@dataclass
class LearningInsight:
    """Represents a learning insight from execution data."""
    category: str
    pattern: str
    confidence: float
    recommendation: str
    impact_score: float
    sample_size: int


class ValueMetricsEngine:
    """Advanced metrics tracking and learning system."""
    
    def __init__(self, repo_path: str = ".", config_path: str = ".terragon/value-config.yaml"):
        """Initialize value metrics engine."""
        self.repo_path = Path(repo_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.metrics_file = self.repo_path / ".terragon" / "value_metrics.json"
        self.learning_file = self.repo_path / ".terragon" / "learning_insights.json"
        self.metrics_data = self._load_metrics_data()
        self.learning_insights = self._load_learning_insights()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def _load_metrics_data(self) -> Dict[str, Any]:
        """Load existing metrics data."""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "repository_info": {
                    "name": self.repo_path.name,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                },
                "value_metrics": [],
                "summary_stats": {},
                "trends": {}
            }
    
    def _load_learning_insights(self) -> List[LearningInsight]:
        """Load learning insights."""
        try:
            with open(self.learning_file, 'r') as f:
                data = json.load(f)
                return [LearningInsight(**insight) for insight in data.get("insights", [])]
        except FileNotFoundError:
            return []
    
    def record_value_delivery(self, 
                            item_id: str,
                            category: str,
                            predicted_value: float,
                            actual_value: float,
                            predicted_effort: float,
                            actual_effort: float,
                            success: bool,
                            impact_metrics: Dict[str, Any] = None,
                            learning_feedback: Dict[str, Any] = None) -> None:
        """Record a value delivery event."""
        
        metric = ValueMetric(
            timestamp=datetime.now().isoformat(),
            item_id=item_id,
            category=category,
            predicted_value=predicted_value,
            actual_value=actual_value,
            predicted_effort=predicted_effort,
            actual_effort=actual_effort,
            success=success,
            impact_metrics=impact_metrics or {},
            learning_feedback=learning_feedback or {}
        )
        
        self.metrics_data["value_metrics"].append(asdict(metric))
        self.metrics_data["repository_info"]["last_updated"] = datetime.now().isoformat()
        
        # Update summary statistics
        self._update_summary_stats()
        
        # Update trends
        self._update_trends()
        
        # Generate learning insights
        self._generate_learning_insights()
        
        # Save data
        self._save_metrics_data()
        self._save_learning_insights()
    
    def _update_summary_stats(self) -> None:
        """Update summary statistics."""
        metrics = self.metrics_data["value_metrics"]
        
        if not metrics:
            return
        
        # Overall statistics
        total_items = len(metrics)
        successful_items = len([m for m in metrics if m["success"]])
        success_rate = successful_items / total_items if total_items > 0 else 0
        
        # Value accuracy
        value_accuracies = []
        for m in metrics:
            if m["predicted_value"] > 0 and m["actual_value"] > 0:
                accuracy = min(m["predicted_value"], m["actual_value"]) / max(m["predicted_value"], m["actual_value"])
                value_accuracies.append(accuracy)
        
        # Effort accuracy
        effort_accuracies = []
        for m in metrics:
            if m["predicted_effort"] > 0 and m["actual_effort"] > 0:
                accuracy = min(m["predicted_effort"], m["actual_effort"]) / max(m["predicted_effort"], m["actual_effort"])
                effort_accuracies.append(accuracy)
        
        # Category breakdown
        category_stats = {}
        for m in metrics:
            cat = m["category"]
            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "success_rate": 0, "avg_value": 0, "avg_effort": 0}
            
            category_stats[cat]["count"] += 1
            category_stats[cat]["success_rate"] = len([x for x in metrics if x["category"] == cat and x["success"]]) / category_stats[cat]["count"]
            category_stats[cat]["avg_value"] = statistics.mean([x["actual_value"] for x in metrics if x["category"] == cat])
            category_stats[cat]["avg_effort"] = statistics.mean([x["actual_effort"] for x in metrics if x["category"] == cat])
        
        # Update summary
        self.metrics_data["summary_stats"] = {
            "total_value_items": total_items,
            "successful_items": successful_items,
            "success_rate": round(success_rate, 3),
            "avg_value_accuracy": round(statistics.mean(value_accuracies), 3) if value_accuracies else 0,
            "avg_effort_accuracy": round(statistics.mean(effort_accuracies), 3) if effort_accuracies else 0,
            "total_value_delivered": sum(m["actual_value"] for m in metrics),
            "total_effort_hours": sum(m["actual_effort"] for m in metrics),
            "avg_value_per_hour": round(sum(m["actual_value"] for m in metrics) / sum(m["actual_effort"] for m in metrics), 2) if sum(m["actual_effort"] for m in metrics) > 0 else 0,
            "category_breakdown": category_stats,
            "last_updated": datetime.now().isoformat()
        }
    
    def _update_trends(self) -> None:
        """Update trend analysis."""
        metrics = self.metrics_data["value_metrics"]
        
        if len(metrics) < 5:  # Need minimum data for trends
            return
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m["timestamp"])
        recent_metrics = sorted_metrics[-10:]  # Last 10 items
        
        # Calculate trends
        if len(recent_metrics) >= 5:
            # Success rate trend
            recent_success = [m["success"] for m in recent_metrics]
            success_trend = sum(recent_success) / len(recent_success)
            
            # Value accuracy trend
            value_accuracies = []
            for m in recent_metrics:
                if m["predicted_value"] > 0 and m["actual_value"] > 0:
                    accuracy = min(m["predicted_value"], m["actual_value"]) / max(m["predicted_value"], m["actual_value"])
                    value_accuracies.append(accuracy)
            
            # Effort accuracy trend
            effort_accuracies = []
            for m in recent_metrics:
                if m["predicted_effort"] > 0 and m["actual_effort"] > 0:
                    accuracy = min(m["predicted_effort"], m["actual_effort"]) / max(m["predicted_effort"], m["actual_effort"])
                    effort_accuracies.append(accuracy)
            
            # Velocity trend (value per hour)
            velocity_points = []
            for m in recent_metrics:
                if m["actual_effort"] > 0:
                    velocity_points.append(m["actual_value"] / m["actual_effort"])
            
            self.metrics_data["trends"] = {
                "success_rate_trend": round(success_trend, 3),
                "value_accuracy_trend": round(statistics.mean(value_accuracies), 3) if value_accuracies else 0,
                "effort_accuracy_trend": round(statistics.mean(effort_accuracies), 3) if effort_accuracies else 0,
                "velocity_trend": round(statistics.mean(velocity_points), 2) if velocity_points else 0,
                "trend_period": "last_10_items",
                "trend_updated": datetime.now().isoformat()
            }
    
    def _generate_learning_insights(self) -> None:
        """Generate learning insights from metrics data."""
        metrics = self.metrics_data["value_metrics"]
        
        if len(metrics) < 10:  # Need minimum data for insights
            return
        
        insights = []
        
        # Category performance insights
        category_performance = {}
        for m in metrics:
            cat = m["category"]
            if cat not in category_performance:
                category_performance[cat] = {"successes": 0, "total": 0, "value": 0, "effort": 0}
            
            category_performance[cat]["total"] += 1
            if m["success"]:
                category_performance[cat]["successes"] += 1
            category_performance[cat]["value"] += m["actual_value"]
            category_performance[cat]["effort"] += m["actual_effort"]
        
        # Generate insights for each category
        for category, stats in category_performance.items():
            if stats["total"] >= 3:  # Minimum sample size
                success_rate = stats["successes"] / stats["total"]
                avg_value = stats["value"] / stats["total"]
                avg_effort = stats["effort"] / stats["total"]
                value_per_hour = avg_value / avg_effort if avg_effort > 0 else 0
                
                # High performance category
                if success_rate > 0.8 and value_per_hour > 10:
                    insights.append(LearningInsight(
                        category=category,
                        pattern="high_performance",
                        confidence=0.8,
                        recommendation=f"Prioritize {category} items for consistent high value delivery",
                        impact_score=value_per_hour,
                        sample_size=stats["total"]
                    ))
                
                # Low performance category
                elif success_rate < 0.5:
                    insights.append(LearningInsight(
                        category=category,
                        pattern="low_success_rate",
                        confidence=0.7,
                        recommendation=f"Review and improve {category} execution approach",
                        impact_score=-value_per_hour,
                        sample_size=stats["total"]
                    ))
        
        # Effort estimation insights
        effort_errors = []
        for m in metrics:
            if m["predicted_effort"] > 0 and m["actual_effort"] > 0:
                error_ratio = m["actual_effort"] / m["predicted_effort"]
                effort_errors.append((m["category"], error_ratio))
        
        if effort_errors:
            # Categories with consistent underestimation
            category_errors = {}
            for cat, error in effort_errors:
                if cat not in category_errors:
                    category_errors[cat] = []
                category_errors[cat].append(error)
            
            for cat, errors in category_errors.items():
                if len(errors) >= 3:  # Minimum sample
                    avg_error = statistics.mean(errors)
                    if avg_error > 1.5:  # Consistent underestimation
                        insights.append(LearningInsight(
                            category=cat,
                            pattern="effort_underestimation",
                            confidence=0.75,
                            recommendation=f"Increase effort estimates for {cat} by {int((avg_error - 1) * 100)}%",
                            impact_score=avg_error,
                            sample_size=len(errors)
                        ))
        
        # Update learning insights
        self.learning_insights = insights
    
    def get_value_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive value dashboard."""
        stats = self.metrics_data.get("summary_stats", {})
        trends = self.metrics_data.get("trends", {})
        
        # Recent activity
        recent_metrics = sorted(
            self.metrics_data["value_metrics"], 
            key=lambda m: m["timestamp"]
        )[-5:]
        
        # Top performing categories
        category_stats = stats.get("category_breakdown", {})
        top_categories = sorted(
            category_stats.items(),
            key=lambda x: x[1].get("avg_value", 0),
            reverse=True
        )[:3]
        
        dashboard = {
            "repository": self.metrics_data["repository_info"]["name"],
            "last_updated": datetime.now().isoformat(),
            "overall_performance": {
                "total_items": stats.get("total_value_items", 0),
                "success_rate": f"{stats.get('success_rate', 0) * 100:.1f}%",
                "total_value_delivered": stats.get("total_value_delivered", 0),
                "value_per_hour": stats.get("avg_value_per_hour", 0),
                "prediction_accuracy": {
                    "value": f"{stats.get('avg_value_accuracy', 0) * 100:.1f}%",
                    "effort": f"{stats.get('avg_effort_accuracy', 0) * 100:.1f}%"
                }
            },
            "trends": {
                "success_rate": f"{trends.get('success_rate_trend', 0) * 100:.1f}%",
                "velocity": trends.get("velocity_trend", 0),
                "prediction_improvement": {
                    "value": f"{trends.get('value_accuracy_trend', 0) * 100:.1f}%",
                    "effort": f"{trends.get('effort_accuracy_trend', 0) * 100:.1f}%"
                }
            },
            "top_categories": [
                {
                    "category": cat,
                    "avg_value": round(data["avg_value"], 1),
                    "success_rate": f"{data['success_rate'] * 100:.1f}%",
                    "count": data["count"]
                }
                for cat, data in top_categories
            ],
            "recent_activity": [
                {
                    "item_id": m["item_id"],
                    "category": m["category"],
                    "success": m["success"],
                    "value": m["actual_value"],
                    "effort": m["actual_effort"]
                }
                for m in recent_metrics
            ],
            "learning_insights": [
                {
                    "category": insight.category,
                    "pattern": insight.pattern,
                    "recommendation": insight.recommendation,
                    "confidence": f"{insight.confidence * 100:.0f}%"
                }
                for insight in self.learning_insights[:5]  # Top 5 insights
            ]
        }
        
        return dashboard
    
    def generate_backlog_report(self, opportunities: List[Dict[str, Any]]) -> str:
        """Generate backlog markdown report."""
        dashboard = self.get_value_dashboard()
        
        report = f"""# 📊 Autonomous Value Backlog

**Repository:** {dashboard['repository']}  
**Last Updated:** {dashboard['last_updated']}  
**Next Execution:** {(datetime.now() + timedelta(hours=1)).isoformat()}

## 🎯 Performance Overview

| Metric | Value | Trend |
|--------|-------|-------|
| Success Rate | {dashboard['overall_performance']['success_rate']} | {dashboard['trends']['success_rate']} |
| Total Value Delivered | {dashboard['overall_performance']['total_value_delivered']:.1f} | ↗️ |
| Value per Hour | {dashboard['overall_performance']['value_per_hour']:.1f} | {dashboard['trends']['velocity']:.1f} |
| Prediction Accuracy | Value: {dashboard['overall_performance']['prediction_accuracy']['value']}, Effort: {dashboard['overall_performance']['prediction_accuracy']['effort']} | Improving |

## 🔍 Next Best Value Item

"""
        
        if opportunities:
            next_item = opportunities[0]
            scores = next_item.get('metadata', {}).get('scores', {})
            report += f"""**[{next_item.get('id', 'N/A')}] {next_item.get('title', 'N/A')}**
- **Composite Score**: {scores.get('composite', 0):.1f}
- **WSJF**: {scores.get('wsjf', 0):.1f} | **ICE**: {scores.get('ice', 0):.1f} | **Tech Debt**: {scores.get('technical_debt', 0):.1f}
- **Estimated Effort**: {next_item.get('estimated_effort', 0)} hours
- **Category**: {next_item.get('category', 'N/A')}
- **Expected Impact**: High value delivery based on composite scoring

"""
        else:
            report += "No high-value items currently identified. Repository optimization is complete!\n\n"
        
        report += """## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, item in enumerate(opportunities[:10], 1):
            scores = item.get('metadata', {}).get('scores', {})
            report += f"| {i} | {item.get('id', 'N/A')[:8]} | {item.get('title', 'N/A')[:40]} | {scores.get('composite', 0):.1f} | {item.get('category', 'N/A')} | {item.get('estimated_effort', 0)} |\n"
        
        if not opportunities:
            report += "| - | - | No items in backlog | - | - | - |\n"
        
        report += f"""
## 📈 Value Metrics

- **Items Completed This Period**: {len(dashboard['recent_activity'])}
- **Average Cycle Time**: {statistics.mean([m['effort'] for m in dashboard['recent_activity']]) if dashboard['recent_activity'] else 0:.1f} hours
- **Value Delivered This Period**: {sum(m['value'] for m in dashboard['recent_activity']):.1f}
- **Success Rate**: {dashboard['overall_performance']['success_rate']}

## 🏆 Top Performing Categories

"""
        
        for cat_data in dashboard['top_categories']:
            report += f"- **{cat_data['category'].title()}**: {cat_data['avg_value']} avg value, {cat_data['success_rate']} success rate ({cat_data['count']} items)\n"
        
        report += """
## 🧠 Learning Insights

"""
        
        for insight in dashboard['learning_insights']:
            report += f"- **{insight['category'].title()}** ({insight['pattern']}): {insight['recommendation']} (Confidence: {insight['confidence']})\n"
        
        report += f"""
## 🔄 Continuous Discovery Stats

- **Active Signal Sources**: Git history, static analysis, code comments, dependencies, performance monitoring
- **Discovery Frequency**: Every PR merge, hourly security scans, daily comprehensive analysis
- **Learning Model**: Adaptive scoring with {len(self.metrics_data['value_metrics'])} training samples
- **Prediction Accuracy**: Value {dashboard['overall_performance']['prediction_accuracy']['value']}, Effort {dashboard['overall_performance']['prediction_accuracy']['effort']}

---

*🤖 Generated by Terragon Autonomous SDLC Engine*  
*⚡ Continuous value discovery and delivery*
"""
        
        return report
    
    def _save_metrics_data(self) -> None:
        """Save metrics data to file."""
        self.metrics_file.parent.mkdir(exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2, default=str)
    
    def _save_learning_insights(self) -> None:
        """Save learning insights to file."""
        self.learning_file.parent.mkdir(exist_ok=True)
        insights_data = {
            "last_updated": datetime.now().isoformat(),
            "total_insights": len(self.learning_insights),
            "insights": [asdict(insight) for insight in self.learning_insights]
        }
        with open(self.learning_file, 'w') as f:
            json.dump(insights_data, f, indent=2)


def main():
    """Example usage of value metrics system."""
    metrics_engine = ValueMetricsEngine()
    
    # Example: Record a value delivery
    metrics_engine.record_value_delivery(
        item_id="perf-001",
        category="performance",
        predicted_value=50.0,
        actual_value=65.0,
        predicted_effort=4.0,
        actual_effort=3.5,
        success=True,
        impact_metrics={"latency_reduction": "20%", "memory_savings": "15MB"},
        learning_feedback={"estimation_accuracy": "good", "complexity": "medium"}
    )
    
    # Generate dashboard
    dashboard = metrics_engine.get_value_dashboard()
    print("Value Dashboard:")
    print(json.dumps(dashboard, indent=2))
    
    # Generate backlog report
    sample_opportunities = [
        {
            "id": "sec-001",
            "title": "Update vulnerable dependencies",
            "category": "security",
            "estimated_effort": 2.0,
            "metadata": {"scores": {"composite": 85.2, "wsjf": 45.1, "ice": 320, "technical_debt": 25}}
        }
    ]
    
    report = metrics_engine.generate_backlog_report(sample_opportunities)
    print("\nBacklog Report:")
    print(report)


if __name__ == "__main__":
    main()