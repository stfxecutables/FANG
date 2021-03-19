import pytest


def pytest_addoption(parser):
    parser.addoption("--full-run", action="store_true", help="run with all data")
    parser.addoption("--mutation-prob", action="store", default=0.1, type=float, help="mutation probability")
    parser.addoption("--generations", action="store", default=2, type=int)


# names of these functions are the names of the arguments to use in tests
@pytest.fixture
def full_run(request):
    return request.config.getoption("--full-run")

@pytest.fixture
def mutation_prob(request):
    return request.config.getoption("--mutation-prob")

@pytest.fixture
def generations(request):
    return request.config.getoption("--generations")