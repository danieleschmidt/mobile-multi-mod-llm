"""Mobile Multi-Modal LLM

Tiny (<35 MB) vision-text transformer for on-device mobile AI.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "mobile-ai@terragon.com"

from .core import MobileMultiModalLLM

__all__ = ["MobileMultiModalLLM"]