"""
Pytest configuration and shared fixtures for training tests.
"""

import os
import sys
import pytest

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as quick smoke tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark parametrized training tests as slow."""
    for item in items:
        # Mark full training tests as slow
        if "test_deterministic_training" in item.nodeid or \
           "test_probabilistic_training" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark smoke tests
        if "TestSmoke" in item.nodeid:
            item.add_marker(pytest.mark.smoke)


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def venv_python():
    """Return the path to the virtual environment Python."""
    venv_path = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
    if os.path.exists(venv_path):
        return venv_path
    return sys.executable
