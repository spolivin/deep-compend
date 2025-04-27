from dataclasses import asdict

from deep_compend.cli.config import DefaultCLIParametersConfig


def test_default_config(default_config):
    """Tests contents of dict with default CLI arguments."""
    assert default_config.filepath == "article.pdf"
    assert default_config.num_beams == 4
    assert default_config.lora_adapters_path is None
    assert default_config.tokenizer_path is None


def test_default_config_as_dict(default_config):
    """Tests contents of dict with default CLI arguments."""
    config_asdict = asdict(default_config)
    assert config_asdict["filepath"] == "article.pdf"
    assert config_asdict["num_beams"] == 4
    assert config_asdict["lora_adapters_path"] is None
    assert config_asdict["tokenizer_path"] is None


def test_default_config_from_dict():
    """Tests `from_dict` method of `DefaultCLIParametersConfig`."""
    # Mocked dict with merged arguments from default arguments, config file and CLI
    merged_args = {
        "filepath": "article.pdf",
        "num_beams": 10,
        "max_output_tokens": 900,
        "foo": "bar",
    }
    # Updating default config values with new ones from merged dict
    config_updated = DefaultCLIParametersConfig.from_dict(merged_args)
    # Verifying if the values have been updated
    assert config_updated.num_beams == 10
    assert config_updated.max_output_tokens == 900
    # Verifying if the values do not include extra argument
    assert "foo" not in asdict(config_updated)
