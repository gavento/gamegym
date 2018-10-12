
import sys
import os
import pytest

TESTS = os.path.dirname(__file__)
ROOT = os.path.dirname(TESTS)
sys.path.insert(0, os.path.join(ROOT, "gamegym"))


def pytest_configure(config):
    pytest._called_from_pytest = True


def pytest_unconfigure(config):
    del pytest._called_from_pytest