import pytest

from deep_compend.cli.config import DefaultCLIParametersConfig


@pytest.fixture(scope="package")
def default_config():
    """Returns a dict with default parameters for CLI."""
    default_args = DefaultCLIParametersConfig(filepath="article.pdf")
    return default_args
