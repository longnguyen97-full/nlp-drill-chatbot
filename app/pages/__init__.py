#!/usr/bin/env python3
"""
LawBot Pages Module
==================

This module contains all the page components for the LawBot application.
Each page is designed to be modular, maintainable, and optimized.
"""

# Import and export all page modules
from . import search
from . import analysis
from . import system

__all__ = ["search", "analysis", "system"]
