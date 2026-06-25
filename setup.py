"""
Enterprise Hybrid RAG System
Setup configuration for package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        # Skip comments and empty lines
        if line and not line.startswith("#") and not line.startswith("-"):
            requirements.append(line)

setup(
    name="enterprise-hybrid-rag",
    version="1.0.0",
    author="ML Engineering Team",
    author_email="ml-team@enterprise.com",
    description="Production-grade Enterprise Hybrid RAG Search System for ML Documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enterprise-hybrid-rag",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/enterprise-hybrid-rag/issues",
        "Documentation": "https://enterprise-hybrid-rag.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=24.10.0",
            "isort>=5.13.2",
            "flake8>=7.1.1",
            "mypy>=1.13.0",
            "pre-commit>=4.0.1",
            "pytest>=8.3.4",
            "pytest-asyncio>=0.24.0",
            "pytest-cov>=6.0.0",
        ],
        "docs": [
            "mkdocs>=1.6.1",
            "mkdocs-material>=9.5.47",
            "mkdocstrings>=0.27.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-server=src.api.main:main",
            "rag-ingest=scripts.ingest:main",
            "rag-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
)
