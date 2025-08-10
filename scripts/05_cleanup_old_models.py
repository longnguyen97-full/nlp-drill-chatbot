#!/usr/bin/env python3
"""
Cleanup Old Models Script - Xóa models cũ
=========================================

Script này xóa các artifacts (models, index, checkpoints, training outputs)
để chuẩn bị cho việc train lại từ đầu. Hỗ trợ:
- --mode fast|quality: áp dụng cấu hình theo mode trước khi import config
- -y/--yes: chạy không hỏi lại
- --dry-run: chỉ hiển thị những gì sẽ xóa
- --all: xóa bổ sung training artifacts; có thể kèm --include-reports/--include-logs

Tác giả: LawBot Team
Phiên bản: Cleanup v2.0
"""

import sys
import os
import shutil
from pathlib import Path
from typing import List


# --- Early CLI pre-parse to set performance mode BEFORE importing config ---
def _apply_mode_from_cli_early():
    try:
        argv = sys.argv[1:]
        selected_mode = None
        for i, arg in enumerate(argv):
            if arg in ("--mode", "-mode") and i + 1 < len(argv):
                selected_mode = argv[i + 1]
                break
            if arg.startswith("--mode="):
                selected_mode = arg.split("=", 1)[1]
                break
            if arg.startswith("-mode="):
                selected_mode = arg.split("=", 1)[1]
                break
        if selected_mode:
            selected_mode = selected_mode.strip().lower()
            if selected_mode in ("fast", "quality"):
                os.environ["LAWBOT_PERFORMANCE_MODE"] = selected_mode
    except Exception:
        pass


_apply_mode_from_cli_early()

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent))

import config
from core.logging_system import get_logger

# Sử dụng logger đã được setup
logger = get_logger(__name__)


def _delete_path(p: Path, dry_run: bool) -> bool:
    try:
        if not p.exists():
            logger.info(f"ℹ️ Not found: {p}")
            return False
        if dry_run:
            logger.info(f"[DRY-RUN] Would delete: {p}")
            return False
        if p.is_file():
            p.unlink()
            logger.info(f"🗑️ Deleted file: {p}")
        else:
            shutil.rmtree(p)
            logger.info(f"🗑️ Deleted directory: {p}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not delete {p}: {e}")
        return False


def _gather_default_targets() -> List[Path]:
    return [
        # Models & index artifacts
        config.PHOBERT_LAW_PATH,
        config.BI_ENCODER_PATH,
        config.CROSS_ENCODER_PATH,
        config.LIGHT_RERANKER_PATH,
        config.FAISS_INDEX_PATH,
        config.INDEX_TO_AID_PATH,
        # Model checkpoints & training info
        config.BI_ENCODER_PATH / "checkpoints",
        config.CROSS_ENCODER_PATH / "checkpoints",
        config.LIGHT_RERANKER_PATH / "checkpoints",
        config.BI_ENCODER_PATH / "training_info.json",
        config.CROSS_ENCODER_PATH / "training_info.json",
        config.LIGHT_RERANKER_PATH / "training_info.json",
        # Pipeline checkpoint
        (config.DATA_PROCESSED_DIR / "pipeline_checkpoint.json"),
    ]


def _gather_training_artifacts() -> List[Path]:
    return [
        # Training data artifacts (processed)
        config.TRAIN_TRIPLETS_EASY_PATH,
        config.TRAIN_TRIPLETS_EASY_FOR_TRAINING_PATH,
        config.BI_ENCODER_VALIDATION_PATH,
        config.TRAIN_TRIPLETS_HARD_NEG_PATH,
        config.BI_ENCODER_TRAIN_MIXED_PATH,
        config.BI_ENCODER_TRAIN_AUGMENTED_PATH,
        config.TRAIN_PAIRS_PATH,
        config.TRAIN_PAIRS_HARD_NEG_PATH,
        config.TRAIN_PAIRS_MIXED_PATH,
        config.TRAIN_PAIRS_AUGMENTED_PATH,
        # Splits
        config.TRAIN_SPLIT_JSON_PATH,
        config.VAL_SPLIT_JSON_PATH,
    ]


def _gather_dir_contents(d: Path) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    return [p for p in d.glob("*")]


def cleanup_old_models(dry_run: bool = False, include_training_artifacts: bool = False,
                       include_reports: bool = False, include_logs: bool = False) -> bool:
    """Xóa các artifacts để chuẩn bị train lại."""
    logger.info("🧹 STARTING: Cleanup Old Models")
    logger.info("=" * 60)

    targets: List[Path] = []
    targets.extend(_gather_default_targets())

    if include_training_artifacts:
        targets.extend(_gather_training_artifacts())

    if include_reports:
        targets.extend(_gather_dir_contents(config.REPORTS_DIR))
    if include_logs:
        targets.extend(_gather_dir_contents(config.LOGS_DIR))

    # De-duplicate while keeping order
    seen = set()
    unique_targets: List[Path] = []
    for p in targets:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp not in seen:
            unique_targets.append(p)
            seen.add(rp)

    logger.info(f"Found {len(unique_targets)} targets to clean.")
    cleaned_count = 0
    for p in unique_targets:
        if _delete_path(p, dry_run=dry_run):
            cleaned_count += 1

    logger.info("=" * 60)
    if dry_run:
        logger.info(f"✅ Dry-run complete. {len(unique_targets)} items listed.")
    else:
        logger.info(f"✅ Cleanup completed! Deleted {cleaned_count} items")
    logger.info("=" * 60)

    return True


def verify_cleanup():
    """Kiểm tra xem cleanup đã thành công chưa."""
    logger.info("🔍 VERIFYING CLEANUP...")

    models_to_check = [
        config.PHOBERT_LAW_PATH,
        config.BI_ENCODER_PATH,
        config.CROSS_ENCODER_PATH,
        config.LIGHT_RERANKER_PATH,
        config.FAISS_INDEX_PATH,
        config.INDEX_TO_AID_PATH,
        (config.DATA_PROCESSED_DIR / "pipeline_checkpoint.json"),
    ]

    remaining_models = []

    for model_path in models_to_check:
        if model_path.exists():
            remaining_models.append(model_path)

    if remaining_models:
        logger.warning("⚠️ Some models still exist:")
        for model in remaining_models:
            logger.warning(f"  - {model}")
        return False
    else:
        logger.info("✅ All models successfully cleaned up!")
        return True


def main():
    """Hàm chính."""
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup LawBot artifacts for a fresh run")
    parser.add_argument("--mode", "-mode", choices=["fast", "quality"], help="Select performance mode for config paths")
    parser.add_argument("-y", "--yes", action="store_true", help="Do not prompt for confirmation")
    parser.add_argument("--dry-run", action="store_true", help="List what would be deleted without deleting")
    parser.add_argument("--all", action="store_true", help="Also delete processed training artifacts and splits")
    parser.add_argument("--include-reports", action="store_true", help="Also delete all files in reports directory")
    parser.add_argument("--include-logs", action="store_true", help="Also delete all files in logs directory")

    args = parser.parse_args()

    # Ensure env reflects --mode (redundant safety; early pre-parse already applied)
    if args.mode:
        os.environ["LAWBOT_PERFORMANCE_MODE"] = args.mode

    logger.info("🧹 Starting model cleanup...")
    logger.info(f"[CONFIG] PERFORMANCE_MODE: {config.PERFORMANCE_MODE}")

    if not args.yes:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: This will delete existing models and indices!")
        print("=" * 60)
        print("This includes at least:")
        print("  - PhoBERT-Law / Bi-Encoder / Cross-Encoder / Light Reranker")
        print("  - FAISS index & index_to_aid mapping")
        print("  - Model checkpoints & training_info.json")
        print("  - Pipeline checkpoint (data/processed/pipeline_checkpoint.json)")
        if args.all:
            print("  - Processed training artifacts & train/val splits")
        if args.include_reports:
            print("  - All files in reports/ directory")
        if args.include_logs:
            print("  - All files in logs/ directory")
        print("=" * 60)
        response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
        if response not in ("yes", "y"):
            logger.info("❌ Cleanup cancelled by user")
            return

    # Run cleanup
    success = cleanup_old_models(
        dry_run=args.dry_run,
        include_training_artifacts=args.all,
        include_reports=args.include_reports,
        include_logs=args.include_logs,
    )

    if success and not args.dry_run:
        # Verify cleanup
        verify_cleanup()
        logger.info("✅ Cleanup completed successfully!")
        logger.info("🚀 Ready for fresh training!")
    elif not success:
        logger.error("❌ Cleanup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
