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
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

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

# --- Checkpoint System ---
CHECKPOINT_FILE = Path("data/processed/pipeline_checkpoint.json")


def load_checkpoint():
    """Tải trạng thái pipeline từ file checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Could not read checkpoint file: {e}. Starting fresh.")
    return {"completed_steps": [], "failed_steps": [], "last_step": None}


def save_checkpoint(state):
    """Lưu trạng thái pipeline vào file checkpoint."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except IOError as e:
        print(f"Could not save checkpoint file: {e}")


def mark_step_complete(state, step_id):
    """Đánh dấu một bước đã hoàn thành."""
    if step_id not in state["completed_steps"]:
        state["completed_steps"].append(step_id)
    state["last_step"] = step_id
    save_checkpoint(state)


def mark_step_failed(state, step_id):
    """Đánh dấu một bước đã thất bại."""
    if step_id not in state["failed_steps"]:
        state["failed_steps"].append(step_id)
    state["last_step"] = step_id
    save_checkpoint(state)


def is_step_complete(state, step_id):
    """Kiểm tra xem một bước đã hoàn thành chưa."""
    return step_id in state["completed_steps"]


def get_next_step_to_run(state, pipeline_steps):
    """Tìm bước tiếp theo cần chạy dựa trên checkpoint."""
    completed_steps = set(state["completed_steps"])

    for step in pipeline_steps:
        if step["id"] not in completed_steps:
            return step["id"]

    return None  # Tất cả đã hoàn thành


class LegalQAPipeline:
    """Pipeline toi uu cuc dai cho Legal QA System voi 4 buoc chinh (v8.0) - WITH CHECKPOINT SUPPORT"""

    def __init__(
        self,
        skip_filtering: bool = False,
        include_dapt: bool = True,
        resume: bool = True,
    ):
        self.skip_filtering = skip_filtering
        self.include_dapt = include_dapt
        self.resume = resume
        self.project_root = Path(__file__).parent
        self.scripts_dir = self.project_root / "scripts"

        # Setup logging
        self.setup_logging()

        # Dinh nghia 4 buoc chinh toi uu cuc dai
        self.pipeline_steps = self._define_maximum_optimized_pipeline_steps()

        # Progress tracker
        self.progress_tracker = ProgressTracker(len(self.pipeline_steps))

        # Load checkpoint state
        self.checkpoint_state = (
            load_checkpoint()
            if resume
            else {"completed_steps": [], "failed_steps": [], "last_step": None}
        )

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
        """Chay mot buoc trong pipeline voi logging chi tiet va checkpoint support"""
        step_id = step["id"]
        step_name = step["name"]
        script_name = step["script"]
        args = step.get("args", [])
        step_description = step.get("description", "")

        # Kiểm tra checkpoint - nếu bước đã hoàn thành thì bỏ qua
        if is_step_complete(self.checkpoint_state, step_id):
            self.logger.info(
                f"[CHECKPOINT] Buoc {step_id} da hoan thanh truoc do, bo qua"
            )
            return True

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
            mark_step_failed(self.checkpoint_state, step_id)
            return False

        # Chay script
        try:
            import subprocess
            import os

            # Tao command
            cmd = [sys.executable, str(script_path)] + args
            step_logger.info(
                f"[BUOC {step_id}] [START] Bat dau chay script: {script_name}"
            )
            step_logger.info(f"[BUOC {step_id}] Command: {' '.join(cmd)}")

            # Chay script voi timeout va capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Doc output real-time
            output_lines = []
            line_count = 0
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    output_lines.append(output)
                    line_count += 1
                    step_logger.info(
                        f"[BUOC {step_id}] [LIST] Line {line_count}: {output}"
                    )

            # Lay return code
            return_code = process.poll()

            # Kiem tra ket qua
            if return_code == 0:
                step_logger.info(
                    f"[BUOC {step_id}] [OK] Script hoan thanh thanh cong voi {line_count} dong output"
                )
                step_logger.info(
                    f"[BUOC {step_id}] [OK] Hoan thanh trong {time.time() - start_time:.1f}s Generated {line_count} output lines"
                )
                log_step_end(
                    step_name,
                    True,
                    time.time() - start_time,
                    {"output_lines": line_count},
                )
                self.progress_tracker.end_step(step_start, success=True)

                # Mark step as complete in checkpoint
                mark_step_complete(self.checkpoint_state, step_id)

                return True
            else:
                step_logger.error(
                    f"[BUOC {step_id}] [FAIL] Script that bai voi return code: {return_code}"
                )
                step_logger.error(f"[BUOC {step_id}] [FAIL] Output lines: {line_count}")
                log_error(
                    f"Script that bai voi return code: {return_code}",
                    step_name=step_name,
                )
                self.progress_tracker.end_step(step_start, success=False)

                # Mark step as failed in checkpoint
                mark_step_failed(self.checkpoint_state, step_id)

                return False

        except Exception as e:
            error_msg = f"Loi khi chay script {script_name}: {str(e)}"
            step_logger.error(error_msg)
            log_error(error_msg, step_name=step_name)
            self.progress_tracker.end_step(step_start, success=False)

            # Mark step as failed in checkpoint
            mark_step_failed(self.checkpoint_state, step_id)

            return False

    def run_pipeline(self, start_step: Optional[str] = None) -> bool:
        """Chay toan bo pipeline hoac tu buoc cu the voi logging chi tiet va checkpoint support"""
        self.logger.info(
            "[TARGET] LEGAL QA PIPELINE - LUONG TOI UU TICH HOP (INTEGRATED OPTIMIZED V8.0) - WITH CHECKPOINT"
        )
        self.logger.info("=" * 80)

        # Hiển thị trạng thái checkpoint
        if self.resume and self.checkpoint_state["completed_steps"]:
            self.logger.info(
                f"[CHECKPOINT] Found checkpoint with {len(self.checkpoint_state['completed_steps'])} completed steps"
            )
            self.logger.info(
                f"[CHECKPOINT] Completed steps: {', '.join(self.checkpoint_state['completed_steps'])}"
            )
            if self.checkpoint_state["failed_steps"]:
                self.logger.info(
                    f"[CHECKPOINT] Failed steps: {', '.join(self.checkpoint_state['failed_steps'])}"
                )

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
        else:
            # Tự động tìm bước tiếp theo từ checkpoint
            next_step = get_next_step_to_run(self.checkpoint_state, self.pipeline_steps)
            if next_step:
                for i, step in enumerate(self.pipeline_steps):
                    if step["id"] == next_step:
                        start_index = i
                        self.logger.info(f"[CHECKPOINT] Resuming from step {next_step}")
                        break

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
                    self.logger.info(
                        f"[CHECKPOINT] Progress saved. You can resume from step {step['id']} later."
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

        # Xóa checkpoint khi hoàn thành
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            self.logger.info(
                "[CHECKPOINT] Pipeline completed successfully, checkpoint file removed."
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
        """Hien thi danh sach cac buoc trong pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("DANH SACH CAC BUOC TRONG PIPELINE:")
        self.logger.info("=" * 80)

        for i, step in enumerate(self.pipeline_steps, 1):
            status = (
                "[COMPLETED]"
                if is_step_complete(self.checkpoint_state, step["id"])
                else "[PENDING]"
            )
            required = "[REQUIRED]" if step["required"] else "[OPTIONAL]"
            self.logger.info(
                f"{i:2d}. {step['id']:2s} - {step['name']} {status} {required}"
            )
            self.logger.info(f"     Mo ta: {step['description']}")
            self.logger.info(f"     Thoi gian: {step['estimated_time']}")
            self.logger.info("")

    def clear_checkpoint(self):
        """Xóa checkpoint để chạy lại từ đầu"""
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            self.checkpoint_state = {
                "completed_steps": [],
                "failed_steps": [],
                "last_step": None,
            }
            self.logger.info(
                "[CHECKPOINT] Checkpoint cleared. Pipeline will start from beginning."
            )


def main():
    """Ham chinh"""
    import argparse

    parser = argparse.ArgumentParser(description="Legal QA Pipeline v8.0")
    parser.add_argument("--start-step", help="Bat dau tu buoc cu the (e.g., 02)")
    parser.add_argument(
        "--skip-filtering", action="store_true", help="Bo qua filtering dataset"
    )
    parser.add_argument("--no-dapt", action="store_true", help="Bo qua DAPT step")
    parser.add_argument(
        "--no-resume", action="store_true", help="Khong resume tu checkpoint"
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Xoa checkpoint va chay lai tu dau",
    )
    parser.add_argument(
        "--show-steps", action="store_true", help="Hien thi danh sach cac buoc"
    )

    args = parser.parse_args()

    # Tao pipeline
    pipeline = LegalQAPipeline(
        skip_filtering=args.skip_filtering,
        include_dapt=not args.no_dapt,
        resume=not args.no_resume,
    )

    # Clear checkpoint nếu được yêu cầu
    if args.clear_checkpoint:
        pipeline.clear_checkpoint()

    # Show steps nếu được yêu cầu
    if args.show_steps:
        pipeline.show_steps()
        return

    # Chay pipeline
    success = pipeline.run_pipeline(start_step=args.start_step)

    if success:
        print("\n" + "=" * 80)
        print("🎉 PIPELINE HOAN THANH THANH CONG!")
        print("🚀 He thong Legal QA v8.0 da san sang su dung!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ PIPELINE THAT BAI!")
        print("💡 Ban co the resume tu buoc bi loi bang cach chay:")
        print("   python run_pipeline.py --start-step <step_id>")
        print("=" * 80)


if __name__ == "__main__":
    main()
