#!/usr/bin/env python3
"""
Kiem tra moi truong LawBot
"""

import sys
import os
from pathlib import Path

# Them thu muc goc vao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import logging
from core import setup_logging

setup_logging()


def check_environment():
    """Kiem tra moi truong Python va dependencies"""
    logging.info("[SEARCH] Kiem tra moi truong Python...")

    # Kiem tra Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        logging.info(
            f"[OK] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
        )
    else:
        logging.error(
            f"[FAIL] Python version {python_version.major}.{python_version.minor} khong du. Can Python 3.8+"
        )
        return False

    # Kiem tra PyTorch
    try:
        import torch

        logging.info(f"[OK] PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logging.info(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("[NOTE] CUDA khong kha dung, se su dung CPU")
    except ImportError:
        logging.error("[FAIL] PyTorch chua duoc cai dat")
        return False

    # Kiem tra cac thu vien khac
    required_libs = [
        "transformers",
        "sentence_transformers",
        "faiss",
        "numpy",
        "pandas",
        "sklearn",
        "streamlit",
    ]

    for lib in required_libs:
        try:
            __import__(lib)
            logging.info(f"[OK] {lib} da duoc cai dat")
        except ImportError:
            logging.error(f"[FAIL] {lib} chua duoc cai dat")
            return False

    return True


def check_data_files():
    """Kiem tra cac file du lieu can thiet"""
    logging.info("[FILE] Kiem tra file du lieu...")

    required_files = [
        config.TRAIN_JSON_PATH,
        config.LEGAL_CORPUS_PATH,
        config.PUBLIC_TEST_JSON_PATH,
    ]

    for file_path in required_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logging.info(f"[OK] {file_path.name}: {size_mb:.2f} MB")
        else:
            logging.error(f"[FAIL] Thieu file: {file_path}")
            return False

    return True


def check_processed_files():
    """Kiem tra cac file da xu ly"""
    logging.info("[FILE] Kiem tra file da xu ly...")

    # Tao thu muc neu chua co
    config.DATA_PROCESSED_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.INDEXES_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    logging.info("[OK] Cac thu muc da duoc tao")
    return True


def check_models():
    """Kiem tra cac mo hinh da huan luyen"""
    logging.info("[MODEL] Kiem tra mo hinh...")

    models_to_check = [
        (config.BI_ENCODER_PATH, "Bi-Encoder model"),
        (config.CROSS_ENCODER_PATH, "Cross-Encoder model"),
    ]

    for model_path, description in models_to_check:
        if model_path.exists():
            size_mb = sum(
                f.stat().st_size for f in model_path.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            logging.info(f"[OK] {description}: {model_path} ({size_mb:.2f} MB)")
        else:
            logging.warning(f"[WARNING] Chua co mo hinh {description}: {model_path}")

    logging.info("[INFO] Cac mo hinh nay se duoc tao khi huan luyen")
    return True


def check_indexes():
    """Kiem tra FAISS index"""
    logging.info("[SEARCH] Kiem tra FAISS index...")

    indexes_to_check = [
        (config.FAISS_INDEX_PATH, "FAISS index"),
        (config.INDEX_TO_AID_PATH, "Index to AID mapping"),
    ]

    for index_path, description in indexes_to_check:
        if index_path.exists():
            file_size = index_path.stat().st_size / (1024 * 1024)
            logging.info(f"[OK] {description}: {index_path} ({file_size:.2f} MB)")
        else:
            logging.warning(f"[WARNING] Chua co {description}: {index_path}")

    logging.info("[INFO] Cac index nay se duoc tao khi chay script build index")
    return True


def validate_data_format():
    """Kiem tra format du lieu"""
    logging.info("[LIST] Kiem tra format du lieu...")

    try:
        # Kiem tra train.json
        with open(config.TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
            import json

            train_data = json.load(f)

        if not isinstance(train_data, list):
            logging.error("[FAIL] train.json phai la mot list")
            return False

        logging.info(f"[OK] train.json co {len(train_data)} mau")

        # Kiem tra format cua moi mau
        required_keys = ["question", "relevant_laws"]
        for key in required_keys:
            if key not in train_data[0]:
                logging.error(f"[FAIL] Thieu key '{key}' trong train.json")
                return False

        logging.info("[OK] Format train.json hop le")

    except Exception as e:
        logging.error(f"[FAIL] Loi khi doc train.json: {e}")
        return False

    try:
        # Kiem tra legal_corpus.json
        with open(config.LEGAL_CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)

        if not isinstance(corpus_data, list):
            logging.error("[FAIL] legal_corpus.json phai la mot list")
            return False

        logging.info(f"[OK] legal_corpus.json co {len(corpus_data)} documents")

    except Exception as e:
        logging.error(f"[FAIL] Loi khi doc legal_corpus.json: {e}")
        return False

    return True


def main():
    """Ham chinh"""
    logging.info("[START] Bat dau kiem tra moi truong LawBot...")

    checks = [
        ("Environment", check_environment),
        ("Data Files", check_data_files),
        ("Processed Files", check_processed_files),
        ("Models", check_models),
        ("Indexes", check_indexes),
        ("Data Format", validate_data_format),
    ]

    all_passed = True

    for check_name, check_func in checks:
        try:
            logging.info(f"[SEARCH] {check_name.upper()}")
            if not check_func():
                all_passed = False
        except Exception as e:
            logging.error(f"[FAIL] Loi trong kiem tra {check_name}: {e}")
            all_passed = False

    if all_passed:
        logging.info("[SUCCESS] Tat ca kiem tra deu PASSED! Moi truong san sang.")
        logging.info("[OK] Ban co the bat dau chay cac script training.")
    else:
        logging.error("[FAIL] Co mot so van de can khac phuc truoc khi tiep tuc.")
        logging.info("[INFO] Vui long kiem tra cac loi tren va thu lai.")


if __name__ == "__main__":
    main()
