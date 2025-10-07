#!/usr/bin/env python3
"""
Setup configuration for ToSTLinearEngine.

A PyTorch-based linear model engine with optional activations for advanced machine learning applications.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements_path = this_directory / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split('\n')

setup(
    name="ToSTLinearEngine",
    version="1.0.0",
    author="Alexis Adams",
    author_email="info@tostlinear.com",
    description="A PyTorch-based linear model engine with optional activations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/7tAxiom/ToSTLinearEngine",
    project_urls={
        "Documentation": "https://github.com/7tAxiom/ToSTLinearEngine#readme",
        "Source": "https://github.com/7tAxiom/ToSTLinearEngine",
        "Tracker": "https://github.com/7tAxiom/ToSTLinearEngine/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "examples": [
            "matplotlib",
            "jupyter",
        ],
    },
    entry_points={
        "console_scripts": [
            "tostlinear-demo=tostlinear.examples.example_basic:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tostlinear": ["*.txt", "*.md"],
    },
    keywords=[
        "pytorch",
        "machine-learning",
        "neural-networks",
        "linear-models",
        "deep-learning",
        "artificial-intelligence",
        "activation-functions",
    ],
    license="MIT",
    platforms=["any"],
    zip_safe=False,
)
