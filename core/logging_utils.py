#!/usr/bin/env python3
"""
Logging Utilities - Module Quan Ly Logging Tong Hop
===================================================

Module nay quan ly logging cho toan bo pipeline, tao mot file log duy nhat
cho moi lan chay pipeline voi day du thong tin va khong bi cat.

Tac gia: LawBot Team
Phien ban: Unified Logging System
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import queue
import time


class UnifiedLogger:
    """
    Logger tong hop cho pipeline - tao mot file log duy nhat cho moi lan chay
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.log_file = None
        self.logger = None
        self.handlers = []
        self.is_setup = False
        self.session_id = None
        self.start_time = None

    def setup_logging(
        self, session_name: str = "pipeline", log_level: str = "INFO"
    ) -> str:
        """
        Thiet lap logging tong hop cho mot session moi

        Args:
            session_name: Ten session (se dung trong ten file log)
            log_level: Muc do logging (DEBUG, INFO, WARNING, ERROR)

        Returns:
            Duong dan den file log duoc tao
        """
        if self.is_setup:
            return str(self.log_file)

        # Tao session ID va timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()

        # Tao thu muc logs
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Ten file log
        log_filename = f"{session_name}_{self.session_id}.log"
        self.log_file = logs_dir / log_filename

        # Xoa handlers cu neu co
        for handler in self.handlers:
            if hasattr(handler, "close"):
                handler.close()
        self.handlers.clear()

        # Tao file handler voi UTF-8 encoding
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8", mode="w")
        file_handler.setLevel(getattr(logging, log_level.upper()))

        # Formatter chi tiet cho file
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        # Formatter ngan gon cho console
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

        # Luu handlers
        self.handlers = [file_handler, console_handler]

        # Cau hinh root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=self.handlers,
            force=True,
        )

        # Tao logger cho module nay
        self.logger = logging.getLogger(__name__)

        # Log thong tin bat dau session
        self.logger.info("=" * 80)
        self.logger.info(f"BAT DAU SESSION: {session_name.upper()}")
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(
            f"Thoi gian bat dau: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.logger.info(f"File log: {self.log_file}")
        self.logger.info(f"Log level: {log_level.upper()}")
        self.logger.info("=" * 80)

        self.is_setup = True
        return str(self.log_file)

    def is_logging_setup(self) -> bool:
        """Kiem tra xem logging da duoc setup chua"""
        return self.is_setup

    def use_existing_logger(self):
        """Su dung logger da duoc setup truoc do"""
        if self.is_setup:
            return self.logger
        else:
            # Neu chua setup, lay logger hien tai neu co
            existing_logger = logging.getLogger()
            if existing_logger.handlers:
                self.logger = existing_logger
                return self.logger
        return None

    def get_or_create_logger(
        self, session_name: str = "pipeline", log_level: str = "INFO"
    ) -> str:
        """
        Lay logger hien tai hoac tao moi neu chua co
        """
        # Kiem tra xem da co logger nao duoc setup chua
        root_logger = logging.getLogger()
        if root_logger.handlers:
            # Da co logger, su dung logger hien tai
            self.logger = root_logger
            self.is_setup = True
            # Tim file log hien tai
            for handler in root_logger.handlers:
                if hasattr(handler, "baseFilename"):
                    self.log_file = Path(handler.baseFilename)
                    return str(self.log_file)
            return "existing_logger"

        # Chua co logger, tao moi
        return self.setup_logging(session_name, log_level)

    def log_step_start(self, step_name: str, step_description: str = ""):
        """Log bat dau mot buoc trong pipeline"""
        if not self.logger:
            return

        self.logger.info("-" * 60)
        self.logger.info(f"BAT DAU BUOC: {step_name}")
        if step_description:
            self.logger.info(f"Mo ta: {step_description}")
        self.logger.info(f"Thoi gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("-" * 60)

    def log_step_end(
        self,
        step_name: str,
        success: bool = True,
        duration: float = None,
        additional_info: Dict[str, Any] = None,
    ):
        """Log ket thuc mot buoc trong pipeline"""
        if not self.logger:
            return

        status = "THANH CONG" if success else "THAT BAI"
        self.logger.info("-" * 60)
        self.logger.info(f"KET THUC BUOC: {step_name}")
        self.logger.info(f"Trang thai: {status}")
        if duration:
            self.logger.info(f"Thoi gian thuc thi: {duration:.2f} giay")
        if additional_info:
            for key, value in additional_info.items():
                self.logger.info(f"{key}: {value}")
        self.logger.info("-" * 60)

    def log_error(
        self, error_msg: str, exception: Exception = None, step_name: str = ""
    ):
        """Log loi voi thong tin chi tiet"""
        if not self.logger:
            return

        self.logger.error("!" * 60)
        self.logger.error(f"LOI: {error_msg}")
        if step_name:
            self.logger.error(f"Buoc: {step_name}")
        if exception:
            self.logger.error(
                f"Exception: {type(exception).__name__}: {str(exception)}"
            )
        self.logger.error("!" * 60)

    def log_system_info(self):
        """Log thong tin he thong"""
        if not self.logger:
            return

        import platform
        import psutil

        self.logger.info("THONG TIN HE THONG:")
        self.logger.info(f"OS: {platform.system()} {platform.release()}")
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"CPU: {psutil.cpu_count()} cores")
        self.logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

        # GPU info neu co
        try:
            import torch

            if torch.cuda.is_available():
                self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                self.logger.info(
                    f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
                )
            else:
                self.logger.info("GPU: Khong co")
        except ImportError:
            self.logger.info("GPU: Khong the kiem tra (PyTorch chua cai dat)")

    def log_session_end(self, success: bool = True, summary: Dict[str, Any] = None):
        """Log ket thuc session"""
        if not self.logger:
            return

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        self.logger.info("=" * 80)
        self.logger.info("KET THUC SESSION")
        self.logger.info(
            f"Thoi gian ket thuc: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.logger.info(
            f"Tong thoi gian: {duration:.2f} giay ({duration/60:.1f} phut)"
        )
        self.logger.info(f"Trang thai: {'THANH CONG' if success else 'THAT BAI'}")

        if summary:
            self.logger.info("TOM TAT:")
            for key, value in summary.items():
                self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 80)

    def get_logger(self, name: str = None) -> logging.Logger:
        """Lay logger instance"""
        if name:
            return logging.getLogger(name)
        return self.logger or logging.getLogger(__name__)


# Global instance
unified_logger = UnifiedLogger()


def setup_unified_logging(
    session_name: str = "pipeline", log_level: str = "INFO"
) -> str:
    """
    Ham helper de thiet lap logging tong hop

    Args:
        session_name: Ten session
        log_level: Muc do logging

    Returns:
        Duong dan den file log
    """
    return unified_logger.get_or_create_logger(session_name, log_level)


def get_logger(name: str = None) -> logging.Logger:
    """Lay logger instance"""
    return unified_logger.get_logger(name)


def is_logging_setup() -> bool:
    """Kiem tra xem logging da duoc setup chua"""
    return unified_logger.is_logging_setup()


def use_existing_logger():
    """Su dung logger da duoc setup truoc do"""
    return unified_logger.use_existing_logger()


def log_step_start(step_name: str, step_description: str = ""):
    """Log bat dau buoc"""
    unified_logger.log_step_start(step_name, step_description)


def log_step_end(
    step_name: str,
    success: bool = True,
    duration: float = None,
    additional_info: Dict[str, Any] = None,
):
    """Log ket thuc buoc"""
    unified_logger.log_step_end(step_name, success, duration, additional_info)


def log_error(error_msg: str, exception: Exception = None, step_name: str = ""):
    """Log loi"""
    unified_logger.log_error(error_msg, exception, step_name)


def log_session_end(success: bool = True, summary: Dict[str, Any] = None):
    """Log ket thuc session"""
    unified_logger.log_session_end(success, summary)
