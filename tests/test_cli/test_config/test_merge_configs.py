from dataclasses import asdict

from deep_compend.cli.config import merge_configs


def test_merge_configs_1(default_config):
    """Tests `merge_configs` function when only config file is provided in CLI.

    Mocked command: `deep-compend summarize article.pdf --config=configs/config.json`
    """
    def_config = asdict(default_config)
    # Keys/values from mocked `configs/config.json` file
    config_file_args = {
        "report_name": "new_report_name.txt",
        "spacy_lang_model": "en_core_web_md",
    }
    # Arguments provided in CLI
    cli_args = {"filepath": "article.pdf", "config": "configs/config.json"}
    # Merging arguments
    merged_config = merge_configs(def_config, config_file_args, cli_args)
    assert isinstance(merged_config, dict)
    assert merged_config["filepath"] == "article.pdf"
    assert merged_config["config"] == "configs/config.json"
    # Verifying that default arguments are overridden by config file values
    assert merged_config["report_name"] == "new_report_name.txt"
    assert merged_config["spacy_lang_model"] == "en_core_web_md"


def test_merge_configs_2(default_config):
    """Tests `merge_configs` function when config file and one extra CLI argument is provided in CLI.

    Mocked command: `deep-compend summarize article.pdf --config=configs/config.json --spacy-lang-model=en_core_web_lg`
    """
    def_config = asdict(default_config)
    config_file_args = {
        "report_name": "new_report_name.txt",
        "spacy_lang_model": "en_core_web_md",
    }
    cli_args = {
        "filepath": "article.pdf",
        "config": "configs/config.json",
        "spacy_lang_model": "en_core_web_lg",
    }
    merged_config = merge_configs(def_config, config_file_args, cli_args)
    assert isinstance(merged_config, dict)
    assert merged_config["filepath"] == "article.pdf"
    assert merged_config["config"] == "configs/config.json"
    # Verifying that extra CLI argument overrides the same argument from config file
    assert merged_config["report_name"] == "new_report_name.txt"
    assert merged_config["spacy_lang_model"] == "en_core_web_lg"


def test_merge_configs_3(default_config):
    """Tests `merge_configs` function when only positional argument is used.

    Mocked command: `deep-compend summarize article.pdf`
    """
    def_config = asdict(default_config)
    # Not specifying config file
    config_file_args = {}
    # Using only one positional argument
    cli_args = {"filepath": "article.pdf"}
    merged_config = merge_configs(def_config, config_file_args, cli_args)
    assert isinstance(merged_config, dict)
    # Verifying that same default arguments are preserved
    assert merged_config["filepath"] == "article.pdf"
    assert merged_config["config"] is None
    assert merged_config["report_name"] == "summary_report.txt"
    assert merged_config["spacy_lang_model"] == "en_core_web_sm"


def test_merge_configs_4(default_config):
    """Tests `merge_configs` function when using one extra CLI argument.

    Mocked command: `deep-compend summarize article.pdf --num-beams=9`
    """
    def_config = asdict(default_config)
    config_file_args = {}
    cli_args = {"filepath": "article.pdf", "num_beams": 9}
    merged_config = merge_configs(def_config, config_file_args, cli_args)
    assert isinstance(merged_config, dict)
    assert merged_config["filepath"] == "article.pdf"
    assert merged_config["config"] is None
    assert merged_config["num_beams"] == 9
    assert merged_config["spacy_lang_model"] == "en_core_web_sm"
