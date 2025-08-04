#!/usr/bin/env python3
"""
Dataset Filtering Utility
=========================

Script utility để lọc dataset trước khi chạy pipeline chính.
Chỉ chạy khi cần thiết để cải thiện chất lượng dữ liệu.

Usage:
    python scripts/utils/run_filter.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.utils.filter_dataset import main as filter_main


def main():
    """Main function for dataset filtering utility"""
    print("=" * 60)
    print("DATASET FILTERING UTILITY")
    print("=" * 60)
    print("Lưu ý: Chỉ chạy utility này khi cần lọc dữ liệu!")
    print("Pipeline chính sẽ tự động sử dụng dữ liệu đã được lọc.")
    print("=" * 60)

    # Run filtering
    success = filter_main()

    if success:
        print("\n✅ Dataset filtering completed successfully!")
        print("✅ Bạn có thể chạy pipeline chính với dữ liệu đã được lọc.")
    else:
        print("\n❌ Dataset filtering failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
