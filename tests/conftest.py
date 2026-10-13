"""Shared test configuration: single-threaded numeric libraries keep the suite fast and deterministic."""

import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(scope="session", autouse=True)
def _single_threaded_numerics():
    with threadpool_limits(limits=1):
        yield
