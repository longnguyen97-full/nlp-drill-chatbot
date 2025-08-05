#!/usr/bin/env python3
"""
File Utilities - LawBot v8.0
============================

File and path utilities for project management.
"""

import sys
from pathlib import Path
from typing import List, Union
from core.logging_system import get_logger

logger = get_logger(__name__)


def create_directories(directories: List[Union[str, Path]]) -> None:
    """Create directories if they don't exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_script_paths():
    """Setup Python path for script imports"""
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def load_config():
    """Load configuration module"""
    setup_script_paths()
    try:
        import config

        return config
    except ImportError as e:
        logger.error(f"Failed to import config: {e}")
        return None
