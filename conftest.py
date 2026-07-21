import os
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "azure: live Azure tests (require RUN_AZURE_TESTS=1)")

def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_AZURE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_AZURE_TESTS=1 to run live Azure tests")
    for item in items:
        if "azure" in item.keywords:
            item.add_marker(skip)
