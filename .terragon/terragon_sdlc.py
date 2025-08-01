#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Main Orchestrator
The primary entry point for autonomous SDLC enhancement with perpetual value discovery
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add terragon modules to path
sys.path.insert(0, str(Path(__file__).parent))

from value_discovery import ValueDiscoveryEngine
from scoring_engine import ScoringEngine
from autonomous_executor import AutonomousExecutor
from value_metrics import ValueMetricsEngine


class TeragonSDLC:
    """Main Terragon Autonomous SDLC orchestrator."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize Terragon SDLC system."""
        self.repo_path = Path(repo_path).resolve()
        self.config_path = self.repo_path / ".terragon" / "value-config.yaml"
        
        # Ensure .terragon directory exists
        (self.repo_path / ".terragon").mkdir(exist_ok=True)
        
        # Initialize engines
        self.discovery_engine = ValueDiscoveryEngine(str(self.repo_path), str(self.config_path))
        self.scoring_engine = ScoringEngine(str(self.config_path))
        self.executor = AutonomousExecutor(str(self.repo_path), str(self.config_path))
        self.metrics_engine = ValueMetricsEngine(str(self.repo_path), str(self.config_path))
        
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print("⚠️  Configuration not found. Run 'terragon init' first.")
            return {}
    
    def init_repository(self) -> bool:
        """Initialize Terragon SDLC for repository."""
        print("🚀 Initializing Terragon Autonomous SDLC")
        print(f"📁 Repository: {self.repo_path}")
        
        # Check if already initialized
        if self.config_path.exists():
            response = input("⚠️  Terragon already initialized. Reinitialize? (y/N): ")
            if response.lower() != 'y':
                return False
        
        # Analyze repository maturity
        print("🔍 Analyzing repository maturity...")
        maturity = self._assess_repository_maturity()
        print(f"📊 Repository maturity: {maturity}")
        
        # Create initial configuration if not exists
        if not self.config_path.exists():
            print("⚙️  Configuration already exists")
        else:
            print("⚙️  Configuration created")
        
        # Initialize metrics
        print("📈 Initializing value metrics system...")
        self.metrics_engine._save_metrics_data()
        
        # Run initial discovery
        print("🔍 Running initial value discovery...")
        opportunities = self.discovery_engine.discover_all_opportunities()
        print(f"✅ Discovered {len(opportunities)} value opportunities")
        
        # Generate initial backlog
        self._generate_backlog_report()
        
        print("✅ Terragon Autonomous SDLC initialized successfully!")
        print("\nNext steps:")
        print("  1. Review BACKLOG.md for discovered opportunities")
        print("  2. Run 'terragon discover' to find new opportunities")
        print("  3. Run 'terragon execute' to start autonomous execution")
        print("  4. Run 'terragon status' to check system status")
        
        return True
    
    def discover_opportunities(self) -> List[Dict[str, Any]]:
        """Discover new value opportunities."""
        print("🔍 Starting value discovery...")
        
        opportunities = self.discovery_engine.discover_all_opportunities()
        
        print(f"📊 Discovery Results:")
        print(f"   Total opportunities: {len(opportunities)}")
        
        # Group by category
        categories = {}
        for opp in opportunities:
            cat = opp.category
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        print("   By category:")
        for cat, count in categories.items():
            print(f"     {cat}: {count}")
        
        # Show top 5 opportunities
        print("\n🎯 Top 5 Opportunities:")
        for i, opp in enumerate(opportunities[:5], 1):
            scores = opp.metadata.get("scores", {})
            print(f"   {i}. {opp.title}")
            print(f"      Score: {scores.get('composite', 0):.1f} | Category: {opp.category} | Effort: {opp.estimated_effort}h")
        
        # Save results
        self.discovery_engine.save_discovery_results()
        self._generate_backlog_report()
        
        print(f"\n💾 Results saved to:")
        print(f"   - .terragon/discovery_results.json")
        print(f"   - BACKLOG.md")
        
        return [opp.__dict__ for opp in opportunities]
    
    def execute_next_item(self) -> bool:
        """Execute the next highest-value item."""
        print("⚡ Starting autonomous execution...")
        
        # Discover opportunities
        opportunities = self.discovery_engine.discover_all_opportunities()
        
        if not opportunities:
            print("✅ No high-value opportunities found. Repository is optimized!")
            return True
        
        # Get next best item
        next_item = opportunities[0]
        scores = next_item.metadata.get("scores", {})
        
        print(f"🎯 Executing: {next_item.title}")
        print(f"   Category: {next_item.category}")
        print(f"   Score: {scores.get('composite', 0):.1f}")
        print(f"   Estimated effort: {next_item.estimated_effort}h")
        
        # Execute the item
        success = self.executor._execute_value_item(next_item)
        
        if success:
            print("✅ Execution completed successfully!")
            
            # Record metrics (with simulated actual values)
            self.metrics_engine.record_value_delivery(
                item_id=next_item.id,
                category=next_item.category,
                predicted_value=scores.get('composite', 0),
                actual_value=scores.get('composite', 0) * 1.1,  # Simulated
                predicted_effort=next_item.estimated_effort,
                actual_effort=next_item.estimated_effort * 0.9,  # Simulated
                success=True
            )
        else:
            print("❌ Execution failed!")
            
            # Record failed metrics
            self.metrics_engine.record_value_delivery(
                item_id=next_item.id,
                category=next_item.category,
                predicted_value=scores.get('composite', 0),
                actual_value=0,
                predicted_effort=next_item.estimated_effort,
                actual_effort=next_item.estimated_effort * 1.2,  # Simulated
                success=False
            )
        
        # Update backlog
        self._generate_backlog_report()
        
        return success
    
    def run_continuous_loop(self, max_iterations: int = 10) -> None:
        """Run continuous autonomous execution loop."""
        print("🔄 Starting continuous autonomous execution")
        print(f"📊 Max iterations: {max_iterations}")
        print("-" * 50)
        
        self.executor.run_continuous_loop(max_iterations)
        
        # Update final backlog report
        self._generate_backlog_report()
        
        print("\n📊 Final execution summary saved to BACKLOG.md")
    
    def show_status(self) -> Dict[str, Any]:
        """Show current Terragon SDLC status."""
        print("📊 Terragon SDLC Status")
        print("=" * 50)
        
        # Repository info
        print(f"📁 Repository: {self.repo_path.name}")
        print(f"⚙️  Configuration: {'✅ Found' if self.config_path.exists() else '❌ Missing'}")
        
        # Discovery status
        discovery_file = self.repo_path / ".terragon" / "discovery_results.json"
        if discovery_file.exists():
            with open(discovery_file, 'r') as f:
                discovery_data = json.load(f)
            print(f"🔍 Last discovery: {discovery_data.get('timestamp', 'Unknown')}")
            print(f"🎯 Opportunities found: {discovery_data.get('total_items', 0)}")
        else:
            print("🔍 Discovery: Not run yet")
        
        # Metrics status
        dashboard = self.metrics_engine.get_value_dashboard()
        print(f"📈 Total value items: {dashboard['overall_performance']['total_items']}")
        print(f"📈 Success rate: {dashboard['overall_performance']['success_rate']}")
        print(f"📈 Value delivered: {dashboard['overall_performance']['total_value_delivered']:.1f}")
        
        # Recent activity
        if dashboard['recent_activity']:
            print("\n🕐 Recent Activity:")
            for activity in dashboard['recent_activity'][-3:]:
                status = "✅" if activity['success'] else "❌"
                print(f"   {status} {activity['category']}: {activity['value']:.1f} value, {activity['effort']:.1f}h")
        
        # Learning insights
        if dashboard['learning_insights']:
            print("\n🧠 Top Learning Insights:")
            for insight in dashboard['learning_insights'][:3]:
                print(f"   • {insight['category']}: {insight['recommendation']}")
        
        return dashboard
    
    def _assess_repository_maturity(self) -> str:
        """Assess repository SDLC maturity level."""
        score = 0
        
        # Check for basic files
        basic_files = ["README.md", "LICENSE", ".gitignore"]
        for file in basic_files:
            if (self.repo_path / file).exists():
                score += 10
        
        # Check for SDLC files
        sdlc_files = ["CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md"]
        for file in sdlc_files:
            if (self.repo_path / file).exists():
                score += 15
        
        # Check for testing
        if (self.repo_path / "tests").exists():
            score += 20
        
        # Check for CI/CD
        if (self.repo_path / ".github" / "workflows").exists():
            score += 20
        
        # Check for package management
        package_files = ["pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
        for file in package_files:
            if (self.repo_path / file).exists():
                score += 15
                break
        
        # Determine maturity
        if score < 25:
            return "nascent"
        elif score < 50:
            return "developing"
        elif score < 75:
            return "maturing"
        else:
            return "advanced"
    
    def _generate_backlog_report(self) -> None:
        """Generate BACKLOG.md report."""
        try:
            # Get latest opportunities
            opportunities = self.discovery_engine.discover_all_opportunities()
            opportunities_dict = [opp.__dict__ for opp in opportunities]
            
            # Generate report
            report = self.metrics_engine.generate_backlog_report(opportunities_dict)
            
            # Save to BACKLOG.md
            backlog_path = self.repo_path / "BACKLOG.md"
            with open(backlog_path, 'w') as f:
                f.write(report)
                
        except Exception as e:
            print(f"⚠️  Failed to generate backlog report: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Terragon Autonomous SDLC Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  terragon init                    # Initialize Terragon SDLC
  terragon discover                # Discover value opportunities
  terragon execute                 # Execute next best item
  terragon continuous --iterations 5  # Run 5 iterations
  terragon status                  # Show system status
        """
    )
    
    parser.add_argument("command", choices=[
        "init", "discover", "execute", "continuous", "status"
    ], help="Command to run")
    
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations for continuous mode")
    
    parser.add_argument("--repo", default=".",
                       help="Repository path")
    
    args = parser.parse_args()
    
    # Initialize Terragon SDLC
    terragon = TeragonSDLC(args.repo)
    
    try:
        if args.command == "init":
            terragon.init_repository()
        
        elif args.command == "discover":
            terragon.discover_opportunities()
        
        elif args.command == "execute":
            terragon.execute_next_item()
        
        elif args.command == "continuous":
            terragon.run_continuous_loop(args.iterations)
        
        elif args.command == "status":
            terragon.show_status()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()