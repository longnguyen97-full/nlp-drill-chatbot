#!/usr/bin/env python3
"""
LawBot Application Module - LawBot v8.1
=======================================

Main application module for LawBot with comprehensive system monitoring,
evaluation display, and legal QA capabilities.

Features:
- Interactive legal question answering
- Comprehensive system health monitoring
- Real-time evaluation results display
- Performance metrics visualization
- Multi-tier pipeline status tracking
"""

__version__ = "8.1.0"
__author__ = "LawBot Team"
__description__ = "Comprehensive Legal QA System with Advanced Monitoring"

# Import main app components
from .app import main

__all__ = [
    "main",
]
