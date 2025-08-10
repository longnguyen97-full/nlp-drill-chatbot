#!/usr/bin/env python3
"""
Script để chuyển đổi giữa các performance mode của LawBot
Usage:
    python switch_performance_mode.py fast    # Chuyển sang mode test nhanh
    python switch_performance_mode.py quality # Chuyển sang mode tối ưu chất lượng
    python switch_performance_mode.py fast --immediate  # Áp dụng ngay lập tức
"""

import os
import sys
import subprocess
from pathlib import Path


def clear_config_cache():
    """Clear all config-related module cache to force reload"""
    modules_to_clear = ['config', 'config_fast', 'config_quality', 'config_base']
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]


def set_performance_mode(mode: str, immediate: bool = False):
    """Set performance mode environment variable"""
    if mode not in ["fast", "quality"]:
        print(f"❌ Invalid mode: {mode}")
        print("Valid modes: fast, quality")
        return False

    # Set environment variable for current session
    os.environ["LAWBOT_PERFORMANCE_MODE"] = mode
    print(f"✅ Performance mode set to: {mode.upper()} (current session)")

    # For Windows - set permanent environment variable
    if os.name == "nt":
        try:
            subprocess.run(
                f"setx LAWBOT_PERFORMANCE_MODE {mode}", shell=True, check=True
            )
            print(f"✅ Performance mode set to: {mode.upper()} (permanent)")
        except subprocess.CalledProcessError:
            print(
                f"⚠️  Could not set permanent environment variable, but current session is set to: {mode.upper()}"
            )
    else:
        # For Unix-like systems
        try:
            subprocess.run(
                f"export LAWBOT_PERFORMANCE_MODE={mode}", shell=True, check=True
            )
            print(f"✅ Performance mode set to: {mode.upper()}")
        except subprocess.CalledProcessError:
            print(
                f"⚠️  Could not set environment variable, but current session is set to: {mode.upper()}"
            )

    # If immediate mode is requested, show configuration summary
    if immediate:
        print("\n🔄 Loading configuration...")
        try:
            # Clear config cache first
            clear_config_cache()
            
            # Run a separate Python process to get the config summary
            result = subprocess.run(
                [sys.executable, "-c", "import config; config.print_config_summary()"],
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )

            if result.returncode == 0:
                print("✅ Configuration loaded successfully!")
                print("\n📊 Current configuration summary:")
                print(result.stdout)
            else:
                print(f"⚠️  Could not load configuration: {result.stderr}")
                print("💡 You may need to restart your Python session to see changes.")

        except Exception as e:
            print(f"⚠️  Could not load configuration: {e}")
            print("💡 You may need to restart your Python session to see changes.")

    return True


def show_current_mode():
    """Show current performance mode"""
    current_mode = os.getenv("LAWBOT_PERFORMANCE_MODE", "quality")
    print(f"🎯 Current performance mode: {current_mode.upper()}")

    # Show mode descriptions
    mode_descriptions = {
        "fast": {
            "description": "Test nhanh - Tối ưu cho việc kiểm thử hệ thống",
            "characteristics": [
                "• Epochs: 1 cho tất cả models",
                "• Batch size nhỏ hơn",
                "• Eval steps ít hơn",
                "• Validation split nhỏ hơn",
                "• Không sử dụng FP16",
                "• Memory usage thấp hơn",
            ],
        },
        "quality": {
            "description": "Tối ưu chất lượng - Tối ưu cho kết quả tốt nhất",
            "characteristics": [
                "• Epochs: 3-5 cho các models",
                "• Batch size lớn hơn",
                "• Eval steps nhiều hơn",
                "• Validation split đầy đủ",
                "• Sử dụng FP16 training",
                "• Memory usage cao hơn",
            ],
        },
    }

    desc = mode_descriptions.get(current_mode, {})
    if desc:
        print(f"\n📋 {desc['description']}")
        print("Đặc điểm:")
        for char in desc["characteristics"]:
            print(f"  {char}")


def show_usage():
    """Show usage information"""
    print("🚀 LawBot Performance Mode Switcher")
    print("=" * 50)
    print("Usage:")
    print("  python switch_performance_mode.py fast    # Chế độ test nhanh")
    print("  python switch_performance_mode.py quality # Chế độ tối ưu chất lượng")
    print(
        "  python switch_performance_mode.py fast --immediate  # Áp dụng ngay lập tức"
    )
    print("  python switch_performance_mode.py show    # Hiển thị mode hiện tại")
    print()
    print("Environment Variable:")
    print("  LAWBOT_PERFORMANCE_MODE=fast|quality")
    print()
    print("Examples:")
    print("  # Chuyển sang mode test nhanh (cần restart terminal)")
    print("  python switch_performance_mode.py fast")
    print()
    print("  # Chuyển sang mode test nhanh (áp dụng ngay lập tức)")
    print("  python switch_performance_mode.py fast --immediate")
    print()
    print("  # Chuyển sang mode tối ưu chất lượng")
    print("  python switch_performance_mode.py quality")
    print()
    print("  # Kiểm tra mode hiện tại")
    print("  python switch_performance_mode.py show")
    print()
    print("📁 Config Files:")
    print("  • config_base.py     - Config cơ bản chung")
    print("  • config_fast.py     - Config cho FAST mode")
    print("  • config_quality.py  - Config cho QUALITY mode")
    print("  • config.py          - Config chính (import động)")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        show_usage()
        return

    mode = sys.argv[1].lower()
    immediate = "--immediate" in sys.argv

    if mode == "show":
        show_current_mode()
    elif mode in ["fast", "quality"]:
        if set_performance_mode(mode, immediate):
            if not immediate:
                print(
                    "\n🔄 Để áp dụng thay đổi, hãy restart terminal hoặc chạy lại script với --immediate."
                )
                print("\n📊 Để xem cấu hình chi tiết:")
                print('  python -c "import config; config.print_config_summary()"')
    else:
        print(f"❌ Invalid mode: {mode}")
        show_usage()


if __name__ == "__main__":
    main()
