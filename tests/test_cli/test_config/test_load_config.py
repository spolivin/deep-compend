from deep_compend.cli.config import load_config


def test_load_config_existing(capsys):
    """Tests contents of an existing config."""
    existing_config = load_config(config_path="configs/bart_large_config.json")
    captured = capsys.readouterr()
    assert "Loading config from" in captured.out
    assert isinstance(existing_config, dict)
    assert existing_config["model_path"] == "facebook/bart-large-cnn"
    assert existing_config["report_name"] == "summary_report_bart_large.txt"
    assert existing_config["repetition_penalty"] == 1.4


def test_load_config_non_existent(capsys):
    """Tests contents of non-existent config."""
    non_existent_config = load_config(config_path="configs/config.json")
    captured = capsys.readouterr()
    assert "not found" in captured.out
    assert isinstance(non_existent_config, dict)
    assert non_existent_config == {}


def test_load_config_unspecified(capsys):
    """Tests contents of unspecified config."""
    unspecified_config = load_config(config_path=None)
    captured = capsys.readouterr()
    assert "not specified" in captured.out
    assert isinstance(unspecified_config, dict)
    assert unspecified_config == {}
