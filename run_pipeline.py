#!/usr/bin/env python3
"""
Legal QA Pipeline - Luong Toi Uu Cuc Dai (Maximum Optimized Version)
==========================================================

Pipeline hoan chinh cho he thong hoi-dap phap luat Viet Nam
Su dung kien truc Retrieval-Rerank voi Bi-Encoder + Cross-Encoder
Da duoc toi uu cuc dai voi 2 buoc chinh va logic thong minh toi da

Tac gia: LawBot Team
Phien ban: Maximum Optimized Pipeline v5.0
"""

import subprocess
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Import progress utilities
from core.progress_tracker import ProgressTracker, StepLogger, create_summary_report
from core.logging_system import (
    setup_unified_logging,
    log_step_start,
    log_step_end,
    log_error,
    log_session_end,
    get_logger,
)


class LegalQAPipeline:
    """Pipeline toi uu cuc dai cho Legal QA System voi 4 buoc chinh (v7.0)"""

    def __init__(self, skip_filtering: bool = False, include_dapt: bool = True):
        self.skip_filtering = skip_filtering
        self.include_dapt = include_dapt
        self.project_root = Path(__file__).parent
        self.scripts_dir = self.project_root / "scripts"

        # Setup logging
        self.setup_logging()

        # Dinh nghia 4 buoc chinh toi uu cuc dai
        self.pipeline_steps = self._define_maximum_optimized_pipeline_steps()

        # Progress tracker
        self.progress_tracker = ProgressTracker(len(self.pipeline_steps))

    def setup_logging(self):
        """Thiet lap logging chi tiet"""
        # Setup unified logging
        log_file = setup_unified_logging("pipeline", "INFO")
        self.logger = get_logger(__name__)
        self.logger.info(f"[START] Bat dau Legal QA Pipeline - Log file: {log_file}")

    def _define_maximum_optimized_pipeline_steps(self) -> List[Dict]:
        """Dinh nghia 4 buoc chinh toi uu cuc dai cho v8.0 (Integrated Optimized Training)"""
        steps = []

        # Bước 0: DAPT (tùy chọn)
        if self.include_dapt:
            steps.append(
                {
                    "id": "00",
                    "name": "Domain-Adaptive Pre-training (DAPT) - PhoBERT-Law",
                    "script": "00_adapt_model.py",
                    "description": "Chuyen môn hóa PhoBERT thành PhoBERT-Law cho pháp luật với tối ưu hóa cao cấp",
                    "required": False,  # Optional but recommended
                    "estimated_time": "60-120 phut",
                }
            )

        # Các bước bắt buộc - Tối ưu hóa tích hợp
        steps.extend(
            [
                {
                    "id": "01",
                    "name": "Environment & Data Processing & Readiness Check Pipeline",
                    "script": "01_check_environment.py",
                    "description": "Comprehensive environment check + Configuration validation + Model status check + Data validation + Processing + Splitting + Mapping validation với tối ưu hóa",
                    "required": True,
                    "estimated_time": "15-20 phut",
                },
                {
                    "id": "02",
                    "name": "Training Data Preparation Pipeline",
                    "script": "02_prepare_training_data.py",
                    "description": "Advanced Hard Negative Mining + Create triplets/pairs + Merge data + Augmentation + Save training files",
                    "required": True,
                    "estimated_time": "10-15 phut",
                },
                {
                    "id": "03",
                    "name": "Integrated Model Training & Comprehensive Evaluation Pipeline",
                    "script": "03_train_models.py",
                    "description": "Optimized Bi-Encoder training + FAISS index + Cross-Encoder training + Light Reranker training + Comprehensive Evaluation với đầy đủ metrics",
                    "required": True,
                    "estimated_time": "120-200 phut",
                },
            ]
        )

        return steps

    def run_step(self, step: Dict) -> bool:
        """Chay mot buoc trong pipeline voi logging chi tiet"""
        step_id = step["id"]
        step_name = step["name"]
        script_name = step["script"]
        args = step.get("args", [])
        step_description = step.get("description", "")

        # Log bat dau buoc
        log_step_start(step_name, step_description)

        # Bat dau tracking
        step_start = self.progress_tracker.start_step(step_name, step)
        start_time = time.time()

        # Tao step logger
        step_logger = StepLogger(step_id)

        # Duong dan script
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            error_msg = f"Script khong ton tai: {script_path}"
            step_logger.error(error_msg)
            log_error(error_msg, step_name=step_name)
            self.progress_tracker.end_step(step_start, success=False)
            duration = time.time() - start_time
            log_step_end(step_name, success=False, duration=duration)
            return False

        # Chay script voi real-time output
        try:
            cmd = [sys.executable, str(script_path)] + args
            step_logger.step_start(f"Bat dau chay script: {script_name}")
            step_logger.info(f"Command: {' '.join(cmd)}")

            # Chay voi real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Doc output real-time voi gioi han
            output_lines = []
            line_count = 0
            max_lines_to_log = 100  # Gioi han so dong log

            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    if output:
                        line_count += 1
                        output_lines.append(output)

                        # Chi log mot so dong dau va cuoi
                        if line_count <= 50 or line_count > len(output_lines) - 50:
                            step_logger.step_progress(f"Line {line_count}: {output}")
                        elif line_count == 51:
                            step_logger.step_progress(
                                f"... (skipping {len(output_lines) - 100} lines) ..."
                            )

            # Cho process hoan thanh
            return_code = process.poll()
            duration = time.time() - start_time

            if return_code == 0:
                step_logger.success(
                    f"Script hoan thanh thanh cong voi {line_count} dong output"
                )
                step_logger.step_complete(f"Generated {len(output_lines)} output lines")
                self.progress_tracker.end_step(step_start, success=True, step_info=step)

                # Log ket thuc thanh cong
                additional_info = {
                    "Output lines": line_count,
                    "Return code": return_code,
                    "Script": script_name,
                }
                log_step_end(
                    step_name,
                    success=True,
                    duration=duration,
                    additional_info=additional_info,
                )
                return True
            else:
                error_msg = f"Script that bai voi return code: {return_code}"
                step_logger.error(error_msg)
                step_logger.error(f"Output lines: {len(output_lines)}")
                log_error(error_msg, step_name=step_name)
                self.progress_tracker.end_step(
                    step_start, success=False, step_info=step
                )
                log_step_end(step_name, success=False, duration=duration)
                return False

        except Exception as e:
            error_msg = f"Loi khong mong muon: {e}"
            step_logger.error(error_msg)
            log_error(error_msg, exception=e, step_name=step_name)
            self.progress_tracker.end_step(step_start, success=False, step_info=step)
            duration = time.time() - start_time
            log_step_end(step_name, success=False, duration=duration)
            return False

    def run_pipeline(self, start_step: Optional[str] = None) -> bool:
        """Chay toan bo pipeline hoac tu buoc cu the voi logging chi tiet"""
        self.logger.info(
            "[TARGET] LEGAL QA PIPELINE - LUONG TOI UU TICH HOP (INTEGRATED OPTIMIZED V8.0)"
        )
        self.logger.info("=" * 80)

        if self.skip_filtering:
            self.logger.warning(
                "[WARNING] Bo qua filtering dataset (su dung du lieu hien co)"
            )

        # Tim buoc bat dau
        start_index = 0
        if start_step:
            for i, step in enumerate(self.pipeline_steps):
                if step["id"] == start_step:
                    start_index = i
                    self.logger.info(f"[TARGET] Bat dau tu buoc {start_step}")
                    break
            else:
                self.logger.error(f"[FAIL] Khong tim thay buoc {start_step}")
                return False

        total_steps = len(self.pipeline_steps) - start_index
        pipeline_start_time = time.time()

        # Thong ke
        successful_steps = 0
        failed_steps = 0
        skipped_steps = 0
        steps_info = []

        # Chay tung buoc
        for i, step in enumerate(self.pipeline_steps[start_index:], 1):
            step_start = time.time()

            if not self.run_step(step):
                failed_steps += 1
                step_info = {
                    "name": step["name"],
                    "success": False,
                    "time": time.time() - step_start,
                }
                steps_info.append(step_info)

                if step["required"]:
                    self.logger.error(
                        f"[ERROR] Pipeline dung lai tai buoc {step['id']} (bat buoc)"
                    )

                    # Tao bao cao tong ket
                    total_time = time.time() - pipeline_start_time
                    summary = create_summary_report(steps_info, total_time)
                    self.logger.info(summary)

                    # Log ket thuc session that bai
                    summary_dict = {
                        "Total steps": len(self.pipeline_steps),
                        "Successful": successful_steps,
                        "Failed": failed_steps,
                        "Skipped": skipped_steps,
                        "Total time": f"{total_time:.2f}s",
                    }
                    log_session_end(success=False, summary=summary_dict)
                    return False
                else:
                    self.logger.warning(
                        f"[WARNING] Bo qua buoc {step['id']} (khong bat buoc)"
                    )
                    skipped_steps += 1
            else:
                successful_steps += 1
                step_info = {
                    "name": step["name"],
                    "success": True,
                    "time": time.time() - step_start,
                }
                steps_info.append(step_info)

        # Hoan thanh
        total_time = time.time() - pipeline_start_time
        self.logger.info("=" * 80)
        self.logger.info("[SUCCESS] PIPELINE HOAN THANH!")
        self.logger.info(
            f"[STATS] Thong ke: Thanh cong {successful_steps}, That bai {failed_steps}, Bo qua {skipped_steps}"
        )
        self.logger.info("[WIN] He thong Legal QA v8.0 da san sang su dung!")
        self.logger.info(
            "[NOTE] Pipeline da duoc toi uu voi 4 buoc chinh (Integrated Optimized Training) va evaluation toan dien!"
        )

        # Tao bao cao tong ket
        summary = create_summary_report(steps_info, total_time)
        self.logger.info(summary)

        # Log ket thuc session thanh cong
        summary_dict = {
            "Total steps": len(self.pipeline_steps),
            "Successful": successful_steps,
            "Failed": failed_steps,
            "Skipped": skipped_steps,
            "Total time": f"{total_time:.2f}s",
        }
        log_session_end(success=True, summary=summary_dict)

        return True

    def show_steps(self):
        """Hien thi danh sach cac buoc voi thong tin chi tiet"""
        self.logger.info(
            "[LIST] DANH SACH CAC BUOC PIPELINE (INTEGRATED OPTIMIZED TRAINING V8.0):"
        )
        self.logger.info("=" * 80)

        for i, step in enumerate(self.pipeline_steps, 1):
            status = "[REQUIRED]" if step["required"] else "[OPTIONAL]"
            args_str = f" {' '.join(step.get('args', []))}" if step.get("args") else ""
            estimated_time = step.get("estimated_time", "Khong xac dinh")

            self.logger.info(f"{status} Buoc {i:02d} ({step['id']}): {step['name']}")
            self.logger.info(f"   [NOTE] {step['description']}")
            self.logger.info(f"   [TOOL] {step['script']}{args_str}")
            self.logger.info(f"   [TIME] {estimated_time}")
            self.logger.info("")

        self.logger.info("[REQUIRED] = Bat buoc | [OPTIONAL] = Tuy chon")
        self.logger.info(
            "[NOTE] Pipeline da duoc toi uu voi 4 buoc chinh (Integrated Optimized Training), evaluation toan dien, va best practices"
        )


def main():
    """Ham chinh"""
    parser = argparse.ArgumentParser(
        description="[START] Legal QA Pipeline v8.0 - Integrated Optimized Training (State-of-the-Art)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du su dung:
  python run_pipeline.py                    # Chay toan bo pipeline (bao gom DAPT)
  python run_pipeline.py --no-dapt          # Chay pipeline khong bao gom DAPT
  python run_pipeline.py --step 02          # Chay tu buoc 02
  python run_pipeline.py --skip-filtering   # Bo qua buoc filtering dataset
  python run_pipeline.py --list-steps       # Xem danh sach buoc
        """,
    )

    parser.add_argument("--step", help="Chay tu buoc cu the (VD: 00, 01, 02, 03)")
    parser.add_argument(
        "--skip-filtering", action="store_true", help="Bo qua buoc filtering dataset"
    )
    parser.add_argument(
        "--no-dapt",
        action="store_true",
        help="Bo qua buoc DAPT (Domain-Adaptive Pre-training)",
    )
    parser.add_argument(
        "--list-steps", action="store_true", help="Hien thi danh sach cac buoc"
    )

    args = parser.parse_args()

    # Tao pipeline
    pipeline = LegalQAPipeline(
        skip_filtering=args.skip_filtering, include_dapt=not args.no_dapt
    )

    # Xu ly cac tuy chon
    if args.list_steps:
        pipeline.show_steps()
        return

    # Chay pipeline
    success = pipeline.run_pipeline(start_step=args.step)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
