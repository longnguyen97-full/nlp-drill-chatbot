#!/usr/bin/env python3
"""
[START] LawBot - Simple Installation Script
=====================================

Script cai dat don gian, an toan cho LawBot
Tranh cac loi maturin va build dependencies
"""

import subprocess
import sys
import os


def install_package(package):
    """Cai dat mot package voi error handling."""
    print(f"[PACKAGE] Installing {package}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"[OK] {package} installed successfully")
            return True
        else:
            print(f"[FAIL] Failed to install {package}: {result.stderr}")
            return False
    except Exception as e:
        print(f"[FAIL] Error installing {package}: {e}")
        return False


def main():
    """Cai dat tat ca packages can thiet."""
    print("[START] LawBot - Simple Installation")
    print("=" * 40)

    # Danh sach packages an toan, khong can build
    packages = [
        "transformers",
        "sentence-transformers",
        "faiss-cpu",
        "scikit-learn",
        "numpy",
        "streamlit",
        "pandas",
        "datasets",
        "ijson",
        "tqdm",
        "accelerate",
        "python-dotenv",
        "psutil",
        "colorama",
        "pydantic",
        "click",
    ]

    failed_packages = []

    for package in packages:
        if not install_package(package):
            failed_packages.append(package)

    print("\n" + "=" * 40)

    if failed_packages:
        print(f"[FAIL] Failed packages: {', '.join(failed_packages)}")
        print("[IDEA] You may need to install these manually")
        return False
    else:
        print("[SUCCESS] All packages installed successfully!")
        print("\n[LIST] Next steps:")
        print("1. Install PyTorch: pip install torch torchvision torchaudio")
        print("2. Run: python scripts/00_check_environment.py")
        print("3. Run: python run_pipeline.py")
        return True


if __name__ == "__main__":
    main()
