"""
Setup script for DiffML package.

This file is used for backward compatibility with older pip versions.
The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read version from __init__.py
version = {}
with open(os.path.join("src", "diffml", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

setup(
    name="diffml",
    version=version.get("__version__", "0.1.0"),
    author="DiogoRibeiro7",
    author_email="diogo.ribeiro@example.com",
    description="Differential Machine Learning for Quantitative Finance - 5-10x faster convergence for option pricing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DiogoRibeiro7/diffml",
    project_urls={
        "Documentation": "https://diffml.readthedocs.io",
        "Bug Tracker": "https://github.com/DiogoRibeiro7/diffml/issues",
        "Source Code": "https://github.com/DiogoRibeiro7/diffml",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: Jupyter",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.62.0",
        "pydantic>=2.0.0",
        "toml>=0.10.2",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "ruff>=0.0.261",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "dashboard": [
            "streamlit>=1.20.0",
            "plotly>=5.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=8.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "all": [
            # Combines all extras
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "ruff>=0.0.261",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "streamlit>=1.20.0",
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipywidgets>=8.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "diffml=diffml.cli:main",
            "diffml-dashboard=diffml.dashboard_cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "differential-machine-learning",
        "quantitative-finance",
        "option-pricing",
        "deep-learning",
        "automatic-differentiation",
        "greeks",
        "risk-management",
        "derivatives",
        "neural-networks",
        "pytorch",
    ],
)