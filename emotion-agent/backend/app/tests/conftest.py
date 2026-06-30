"""Force mock model during pytest to avoid loading multi-GB checkpoints."""

import os

import pytest

os.environ.setdefault("MODEL_PROVIDER", "mock")
os.environ.setdefault("MODEL_FAIL_ON_ERROR", "true")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU and RUN_GPU_TESTS=1")
