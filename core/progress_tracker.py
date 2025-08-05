#!/usr/bin/env python3
"""
Progress Utilities for Legal QA Pipeline
======================================

Cac utility de hien thi progress bar va thong tin chi tiet
cho pipeline training va evaluation.
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class ProgressTracker:
    """Theo doi tien do cua pipeline"""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        self.logger = logging.getLogger(__name__)

    def start_step(self, step_name: str, step_info: Dict):
        """Bat dau mot buoc moi"""
        self.current_step += 1
        step_start = time.time()

        # Get actual step ID from step_info
        step_id = step_info.get("id", str(self.current_step))

        # Log thong tin buoc voi separator ro rang
        self.logger.info("")
        self.logger.info("[START]" + "=" * 78 + "[START]")
        self.logger.info(f"[START] BUOC {step_id}/{self.total_steps}: {step_name}")
        self.logger.info("[START]" + "=" * 78 + "[START]")
        self.logger.info(
            f"[NOTE] Mo ta: {step_info.get('description', 'Khong co mo ta')}"
        )
        self.logger.info(f"[TOOL] Script: {step_info.get('script', 'Khong xac dinh')}")

        if step_info.get("args"):
            self.logger.info(f"[CONFIG] Arguments: {' '.join(step_info['args'])}")

        estimated_time = step_info.get("estimated_time", "Khong xac dinh")
        self.logger.info(f"[TIME] Thoi gian uoc tinh: {estimated_time}")

        status = (
            "[REQUIRED] Bat buoc"
            if step_info.get("required", True)
            else "[OPTIONAL] Tuy chon"
        )
        self.logger.info(f"[STATS] Trang thai: {status}")

        # Hien thi progress bar
        self._show_progress_bar()
        self.logger.info("[START]" + "=" * 78 + "[START]")
        self.logger.info("")

        return step_start

    def end_step(self, step_start: float, success: bool = True, step_info: Dict = None):
        """Ket thuc mot buoc"""
        elapsed = time.time() - step_start
        self.step_times.append(elapsed)

        # Get actual step ID from step_info
        step_id = (
            step_info.get("id", str(self.current_step))
            if step_info
            else str(self.current_step)
        )

        # Log ket qua voi separator ro rang
        self.logger.info("")
        if success:
            self.logger.info("[OK]" + "=" * 78 + "[OK]")
            self.logger.info(
                f"[OK] BUOC {step_id} HOAN THANH THANH CONG trong {elapsed:.1f}s"
            )
            self.logger.info("[OK]" + "=" * 78 + "[OK]")
        else:
            self.logger.info("[FAIL]" + "=" * 78 + "[FAIL]")
            self.logger.info(f"[FAIL] BUOC {step_id} THAT BAI sau {elapsed:.1f}s")
            self.logger.info("[FAIL]" + "=" * 78 + "[FAIL]")
        self.logger.info("")

        # Hien thi thong ke
        self._show_statistics()

    def _show_progress_bar(self):
        """Hien thi progress bar"""
        progress = self.current_step / self.total_steps
        bar_length = 50
        filled_length = int(bar_length * progress)

        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        percentage = progress * 100

        self.logger.info(
            f"[PROGRESS] TIEN DO: [{bar}] {percentage:.1f}% ({self.current_step}/{self.total_steps})"
        )

    def _show_statistics(self):
        """Hien thi thong ke"""
        if self.step_times:
            avg_time = sum(self.step_times) / len(self.step_times)
            total_elapsed = time.time() - self.start_time
            remaining_steps = self.total_steps - self.current_step

            if remaining_steps > 0:
                estimated_remaining = avg_time * remaining_steps
                eta = datetime.now() + timedelta(seconds=estimated_remaining)

                self.logger.info("[STATS] THONG KE:")
                self.logger.info(
                    f"   [TIME] Thoi gian trung binh/buoc: {avg_time:.1f}s"
                )
                self.logger.info(
                    f"   [TIME] Tong thoi gian da chay: {total_elapsed/60:.1f} phut"
                )
                self.logger.info(
                    f"   [TIME] Thoi gian uoc tinh con lai: {estimated_remaining/60:.1f} phut"
                )
                self.logger.info(
                    f"   [TARGET] Du kien hoan thanh: {eta.strftime('%H:%M:%S')}"
                )
                self.logger.info("")


class StepLogger:
    """Logger chuyen dung cho tung buoc"""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"step.{step_name}")
        self.start_time = time.time()

    def info(self, message: str):
        """Log thong tin"""
        self.logger.info(f"[BUOC {self.step_name}] {message}")

    def warning(self, message: str):
        """Log canh bao"""
        self.logger.warning(f"[BUOC {self.step_name}] [WARNING] {message}")

    def error(self, message: str):
        """Log loi"""
        self.logger.error(f"[BUOC {self.step_name}] [FAIL] {message}")

    def success(self, message: str):
        """Log thanh cong"""
        self.logger.info(f"[BUOC {self.step_name}] [OK] {message}")

    def step_complete(self, additional_info: str = ""):
        """Log hoan thanh buoc"""
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"[BUOC {self.step_name}] [OK] Hoan thanh trong {elapsed:.1f}s {additional_info}"
        )

    def step_start(self, message: str):
        """Log bat dau buoc"""
        self.logger.info(f"[BUOC {self.step_name}] [START] {message}")

    def step_progress(self, message: str):
        """Log tien do trong buoc"""
        self.logger.info(f"[BUOC {self.step_name}] [LIST] {message}")


def format_time(seconds: float) -> str:
    """Format thoi gian thanh string de doc"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} phut"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} gio"


def format_file_size(bytes_size: int) -> str:
    """Format kich thuoc file thanh string de doc"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def create_summary_report(steps_info: List[Dict], total_time: float) -> str:
    """Tao bao cao tong ket"""
    report = []
    report.append("=" * 80)
    report.append("[CHART] BAO CAO TONG KET PIPELINE")
    report.append("=" * 80)

    successful_steps = 0
    failed_steps = 0

    for step in steps_info:
        status = "[OK]" if step.get("success", False) else "[FAIL]"
        name = step.get("name", "Unknown")
        time_taken = step.get("time", 0)

        if step.get("success", False):
            successful_steps += 1
        else:
            failed_steps += 1

        report.append(f"{status} {name} ({format_time(time_taken)})")

    report.append("")
    report.append(f"[CHART] Thong ke:")
    report.append(f"   [OK] Thanh cong: {successful_steps}")
    report.append(f"   [FAIL] That bai: {failed_steps}")
    report.append(f"   [TIME] Tong thoi gian: {format_time(total_time)}")

    if successful_steps > 0:
        success_rate = (successful_steps / len(steps_info)) * 100
        report.append(f"   [CHART] Ty le thanh cong: {success_rate:.1f}%")

    report.append("=" * 80)

    return "\n".join(report)
