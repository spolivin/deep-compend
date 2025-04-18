import pytest

from deep_compend.cli import BUILT_IN_DEFAULTS as default_args
from deep_compend.cli import load_config


@pytest.fixture
def existent_config():
    """Returns a dict loaded from an existent config file."""
    config = load_config(config_path="configs/bart_large_config.json")
    return config


@pytest.fixture
def nonexistent_config():
    """Returns a dict loaded from non-existent config file."""
    config = load_config(config_path="configs/config.json")
    return config


@pytest.fixture
def unspecified_config():
    """Returns a dict loaded from unspecified config file."""
    config = load_config(config_path=None)
    return config


def test_normal_config_loading(existent_config):
    """Tests contents of existent config."""
    assert isinstance(existent_config, dict)
    assert existent_config != {}
    assert existent_config["model_path"] == "facebook/bart-large-cnn"
    assert existent_config["report_name"] == "summary_report_bart_large.txt"


def test_nonexistent_config_loading(nonexistent_config):
    """Tests contents of non-existent config."""
    assert isinstance(nonexistent_config, dict)
    assert nonexistent_config == {}


def test_unspecified_config_loading(unspecified_config):
    """Tests contents of unspecified config."""
    assert isinstance(unspecified_config, dict)
    assert unspecified_config == {}


def test_default_args_dict():
    """Tests contents of dict with default CLI arguments."""
    assert len(default_args) == 15
    assert default_args["num_beams"] == 4


def test_merging_with_cli_args_existent(existent_config):
    """Tests merging occurring during running CLI with extra CLI arguments (existing config)."""
    cli_args = {"report_name": "test_rep_name.txt"}
    new_config = {**default_args, **existent_config, **cli_args}
    assert new_config["report_name"] == "test_rep_name.txt"


def test_merging_without_cli_args_existent(existent_config):
    """Tests merging occurring during running CLI without extra CLI arguments (existing config)."""
    new_config = {**default_args, **existent_config}
    assert new_config["repetition_penalty"] == 1.4


def test_merging_with_cli_args_nonexistent(nonexistent_config):
    """Tests merging occurring during running CLI with extra CLI arguments (non-existent config)."""
    cli_args = {"report_name": "test_rep_name.txt"}
    new_config = {**default_args, **nonexistent_config, **cli_args}
    assert new_config["report_name"] == "test_rep_name.txt"
    assert new_config["repetition_penalty"] == 1.2


def test_merging_without_cli_args_nonexistent(nonexistent_config):
    """Tests merging occurring during running CLI without extra CLI arguments (non-existent config)."""
    new_config = {**default_args, **nonexistent_config}
    assert new_config["repetition_penalty"] == 1.2


def test_merging_with_cli_args_unspecified(unspecified_config):
    """Tests merging occurring during running CLI with extra CLI arguments (unspecified config)."""
    cli_args = {"report_name": "test_rep_name.txt"}
    new_config = {**default_args, **unspecified_config, **cli_args}
    assert new_config["report_name"] == "test_rep_name.txt"
    assert new_config["repetition_penalty"] == 1.2


def test_merging_without_cli_args_unspecified(unspecified_config):
    """Tests merging occurring during running CLI without extra CLI arguments (unspecified config)."""
    new_config = {**default_args, **unspecified_config}
    assert new_config["repetition_penalty"] == 1.2
