from dataclasses import asdict

from deep_compend import SummaryGenerationConfig


def test_config_defaults():
    """Tests default config parameters."""
    config = SummaryGenerationConfig()
    assert config.min_length == 30
    assert config.max_length == 250
    assert config.num_beams == 4
    assert config.length_penalty == 1.0
    assert config.repetition_penalty == 1.2
    assert config.no_repeat_ngram_size == 3
    assert config.early_stopping is True


def test_config_custom():
    """Tests custom config parameters."""
    config = SummaryGenerationConfig(
        num_beams=10,
        max_length=450,
        no_repeat_ngram_size=2,
    )
    assert config.min_length == 30
    assert config.max_length == 450
    assert config.num_beams == 10
    assert config.length_penalty == 1.0
    assert config.repetition_penalty == 1.2
    assert config.no_repeat_ngram_size == 2
    assert config.early_stopping is True


def test_config_asdict():
    """Tests dict representation of config."""
    config = SummaryGenerationConfig(max_length=512)
    config_dict = asdict(config)
    assert config_dict["max_length"] == 512


def test_config_equality():
    """Tests the equality of same objects."""
    a = SummaryGenerationConfig()
    b = SummaryGenerationConfig()
    assert a == b


def test_config_inequality():
    """Tests the inequality of different objects."""
    a = SummaryGenerationConfig()
    b = SummaryGenerationConfig(max_length=500)
    assert a != b
