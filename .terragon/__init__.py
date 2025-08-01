"""
Terragon Autonomous SDLC Engine
Perpetual value discovery and execution for repository enhancement
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__description__ = "Autonomous SDLC enhancement with continuous value discovery"

from .value_discovery import ValueDiscoveryEngine
from .scoring_engine import ScoringEngine
from .autonomous_executor import AutonomousExecutor
from .value_metrics import ValueMetricsEngine
from .terragon_sdlc import TeragonSDLC

__all__ = [
    "ValueDiscoveryEngine",
    "ScoringEngine", 
    "AutonomousExecutor",
    "ValueMetricsEngine",
    "TeragonSDLC"
]