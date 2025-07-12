from pathlib import Path

from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "This package provides several models and benchmarks to analyze your time series in a probabilistic fashion"

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ptsa",
    version=VERSION,
    author="Lukas Scholz",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=["numpy==1.26.4",
                      "tqdm==4.67.1",
                      "torch==2.6.0",
                      "scikit-learn==1.7.0",
                      "optuna==4.4.0",
                      "wandb==0.21.0",
                      "matplotlib==3.10.3"],

    keywords=['python'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.12',
    extras_require={'linting': [
        "pylint",
        "typing-extensions",
        "pre-commit",
        "types-tqdm",
        "pandas==2.2.2",
        ""
    ],
        'testing': [
            "pytest"]
    }
)