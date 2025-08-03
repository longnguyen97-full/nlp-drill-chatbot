#!/usr/bin/env python3
"""
Setup script for LawBot - Legal QA Pipeline
===========================================

A comprehensive legal question-answering system for Vietnamese law
using Retrieval-Rerank architecture with Bi-Encoder and Cross-Encoder models.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="lawbot",
    version="2.0.0",
    author="LawBot Team",
    author_email="lawbot@example.com",
    description="Legal QA Pipeline for Vietnamese Law",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lawbot-team/lawbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "deploy": [
            "docker>=5.0",
            "gunicorn>=20.0",
            "uvicorn>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "lawbot=run_pipeline:main",
            "lawbot-check=scripts.01_check_environment:main",
            "lawbot-train=scripts.09_train_bi_encoder:main",
            "lawbot-evaluate=scripts.12_evaluate_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "legal",
        "qa",
        "vietnamese",
        "law",
        "nlp",
        "retrieval",
        "rerank",
        "bi-encoder",
        "cross-encoder",
        "faiss",
        "transformers",
    ],
    project_urls={
        "Bug Reports": "https://github.com/lawbot-team/lawbot/issues",
        "Source": "https://github.com/lawbot-team/lawbot",
        "Documentation": "https://lawbot.readthedocs.io/",
    },
)
