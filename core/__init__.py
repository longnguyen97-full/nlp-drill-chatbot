"""
Core Module - LawBot v8.0
==========================

Main entry point for the core module. All utilities are now organized in the utils package.
"""

# Import all utilities from the utils package for backward compatibility
from .utils import *

# Re-export all utilities for clean imports
__all__ = [
    # Re-export everything from utils
    *__import__("core.utils", fromlist=["*"]).__all__
]
