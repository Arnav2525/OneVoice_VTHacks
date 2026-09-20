

from onevoice.utils.config import AppConfig, load_config

def test_load_config_parses_sections(tmp_path):
    path = tmp_path / "conf.yaml"
    path.write_text(
        "hardware:\n"
        "  device: cuda\n"
        "backend:\n"
        "  name: speechbrain\n"
        "experiment:\n"
        "  id: smoke\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert isinstance(config, AppConfig)
    assert config.hardware.device == "cuda"
    assert config.backend.name == "speechbrain"
    assert config.experiment.id == "smoke"

def test_load_config_empty_file_yields_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    config = load_config(path)
    assert isinstance(config, AppConfig)

def test_extra_keys_allowed():
    config = AppConfig.model_validate({"benchmark": {"duration_s": 30, "extra": True}})
    assert config.benchmark.duration_s == 30
