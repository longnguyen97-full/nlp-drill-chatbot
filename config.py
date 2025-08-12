#!/usr/bin/env python3
"""
Main Configuration for LawBot
============================

This file dynamically imports configuration settings based on the performance mode.
It supports FAST and QUALITY modes with separate configuration files.
"""

import os
import sys
import importlib
from typing import Dict, Any, Optional

# ============================================================================
# PERFORMANCE MODE DETECTION
# ============================================================================

def get_performance_mode() -> str:
    """
    Lấy chế độ hiệu năng từ biến môi trường. Mặc định là 'quality'.
    Chế độ này quyết định file cấu hình nào sẽ được load (fast vs. quality).
    """
    return os.environ.get("LAWBOT_PERFORMANCE_MODE", "quality").lower()

# ============================================================================
# DYNAMIC CONFIG IMPORT
# ============================================================================

def load_config():
    """
    Tải động các biến cấu hình từ `config_quality` hoặc `config_fast`
    dựa trên biến môi trường LAWBOT_PERFORMANCE_MODE.
    """
    mode = get_performance_mode()
    
    if mode == "fast":
        config_module_name = 'config_fast'
    elif mode == "quality":
        config_module_name = 'config_quality'
    else:
        raise ValueError(f"Chế độ hiệu năng không hợp lệ: '{mode}'. Phải là 'fast' hoặc 'quality'.")

    try:
        # Import module cấu hình tương ứng
        config_module = importlib.import_module(config_module_name)
        
        # Lấy tất cả các biến từ __all__ của module đó
        if hasattr(config_module, '__all__'):
            for var in config_module.__all__:
                globals()[var] = getattr(config_module, var)
        else:
            # Fallback nếu __all__ không được định nghĩa
            for attr in dir(config_module):
                if not attr.startswith('__'):
                    globals()[attr] = getattr(config_module, attr)
                    
    except ImportError:
        raise ImportError(f"Không thể import module cấu hình: {config_module_name}")

# Tải cấu hình khi module này được import lần đầu
load_config()

# Import các biến cấu hình cơ sở (base)
from config_base import *

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Để in tóm tắt cấu hình, bạn có thể tạo một hàm mới
    # hoặc gọi trực tiếp các biến đã được import.
    print(f"Chế độ hiệu năng hiện tại: {get_performance_mode().upper()}")
    # Ví dụ: in một vài biến cấu hình quan trọng
    # print(f"Bi-Encoder Batch Size: {BI_ENCODER_BATCH_SIZE}")
    # print(f"Cross-Encoder Max Length: {CROSS_ENCODER_MAX_LENGTH}")
    pass
