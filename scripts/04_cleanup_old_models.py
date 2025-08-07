#!/usr/bin/env python3
"""
Cleanup Old Models Script - Xóa models cũ
=========================================

Script này xóa các models cũ để chuẩn bị cho việc train lại từ đầu.

Tác giả: LawBot Team
Phiên bản: Cleanup v1.0
"""

import sys
import os
import shutil
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent))

import config
from core.logging_system import get_logger

# Sử dụng logger đã được setup
logger = get_logger(__name__)


def cleanup_old_models():
    """Xóa các models cũ để chuẩn bị train lại."""
    logger.info("🧹 STARTING: Cleanup Old Models")
    logger.info("=" * 50)

    models_to_cleanup = [
        config.PHOBERT_LAW_PATH,
        config.BI_ENCODER_PATH,
        config.CROSS_ENCODER_PATH,
        config.LIGHT_RERANKER_PATH,
        config.FAISS_INDEX_PATH,
        config.INDEX_TO_AID_PATH,
    ]

    cleaned_count = 0

    for model_path in models_to_cleanup:
        if model_path.exists():
            try:
                if model_path.is_file():
                    model_path.unlink()
                    logger.info(f"🗑️ Deleted file: {model_path}")
                else:
                    shutil.rmtree(model_path)
                    logger.info(f"🗑️ Deleted directory: {model_path}")
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Could not delete {model_path}: {e}")
        else:
            logger.info(f"ℹ️ Not found: {model_path}")

    # Cleanup checkpoints
    checkpoint_dirs = [
        config.BI_ENCODER_PATH / "checkpoints",
        config.CROSS_ENCODER_PATH / "checkpoints",
        config.LIGHT_RERANKER_PATH / "checkpoints",
    ]

    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            try:
                shutil.rmtree(checkpoint_dir)
                logger.info(f"🗑️ Deleted checkpoint directory: {checkpoint_dir}")
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Could not delete checkpoint {checkpoint_dir}: {e}")

    # Cleanup training info files
    training_info_files = [
        config.BI_ENCODER_PATH / "training_info.json",
        config.CROSS_ENCODER_PATH / "training_info.json",
        config.LIGHT_RERANKER_PATH / "training_info.json",
    ]

    for info_file in training_info_files:
        if info_file.exists():
            try:
                info_file.unlink()
                logger.info(f"🗑️ Deleted training info: {info_file}")
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Could not delete {info_file}: {e}")

    logger.info("=" * 50)
    logger.info(f"✅ Cleanup completed! Deleted {cleaned_count} items")
    logger.info("=" * 50)

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
    logger.info("🧹 Starting model cleanup...")

    # Confirm with user
    print("\n" + "=" * 60)
    print("⚠️  WARNING: This will delete ALL existing models!")
    print("=" * 60)
    print("This includes:")
    print("  - PhoBERT-Law model")
    print("  - Bi-Encoder model")
    print("  - Cross-Encoder model")
    print("  - Light Reranker model")
    print("  - FAISS index")
    print("  - All checkpoints and training info")
    print("=" * 60)

    response = input("Are you sure you want to continue? (yes/no): ").lower().strip()

    if response not in ["yes", "y"]:
        logger.info("❌ Cleanup cancelled by user")
        return

    # Run cleanup
    success = cleanup_old_models()

    if success:
        # Verify cleanup
        verify_cleanup()
        logger.info("✅ Cleanup completed successfully!")
        logger.info("🚀 Ready for fresh training!")
    else:
        logger.error("❌ Cleanup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
