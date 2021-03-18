import pytest


def pytest_addoption(parser):
    parser.addoption("--full-run", action="store_true", help="run with all data")
    parser.addoption("--generations", action="store", default=2, type=int)



@pytest.fixture
def full_run(request):
    return request.config.getoption("--full-run")

@pytest.fixture
def generations(request):
    return request.config.getoption("--generations")