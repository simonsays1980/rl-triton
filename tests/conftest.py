import os
import pytest


def pytest_collection_modifyitems(items):
    """Enable correctness warnings for non-perf tests."""
    for item in items:
        if item.get_closest_marker("perf") is None:
            item.user_properties.append(("correctness_warnings", True))


@pytest.fixture(autouse=True)
def set_correctness_warnings(request):
    is_perf = request.node.get_closest_marker("perf") is not None
    if not is_perf:
        old = os.environ.get("RL_TRITON_CORRECTNESS_WARNINGS")
        os.environ["RL_TRITON_CORRECTNESS_WARNINGS"] = "1"
        yield
        if old is None:
            del os.environ["RL_TRITON_CORRECTNESS_WARNINGS"]
        else:
            os.environ["RL_TRITON_CORRECTNESS_WARNINGS"] = old
    else:
        yield
