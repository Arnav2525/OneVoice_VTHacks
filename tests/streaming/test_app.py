

from onevoice.separation.passthrough import PassthroughSeparator
from onevoice.streaming.app import (
    build_pipeline_from_config,
    build_separator,
    load_experiment_config,
    main,
)

def test_load_experiment_config_none_is_empty():
    assert load_experiment_config(None) == {}

def test_load_experiment_config_reads_yaml(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("backend:\n  name: passthrough\naudio:\n  sample_rate: 8000\n")
    config = load_experiment_config(str(path))
    assert config["backend"]["name"] == "passthrough"
    assert config["audio"]["sample_rate"] == 8000

def test_build_separator_unknown_backend_degrades_to_passthrough(caplog):
    with caplog.at_level("ERROR"):
        sep = build_separator({"backend": {"name": "does-not-exist"}})
    assert isinstance(sep, PassthroughSeparator)
    assert any("passthrough" in r.getMessage() for r in caplog.records)

def test_build_separator_passthrough():
    sep = build_separator({"backend": {"name": "passthrough"}})
    assert sep.get_status()["is_real_separation"] is False

def test_build_pipeline_from_config_defaults_to_mock():
    config = {"backend": {"name": "passthrough"}, "audio": {"sample_rate": 16000}}
    pipeline, chunk_ms = build_pipeline_from_config(config)
    assert chunk_ms > 0
    pipeline.start()
    try:
        assert pipeline.is_running
    finally:
        pipeline.stop()
    assert not pipeline.is_running

def test_main_runs_and_stops_cleanly(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("backend:\n  name: passthrough\n")
    code = main(
        ["--config", str(path), "--duration", "1", "--telemetry-interval", "0.3"]
    )
    assert code == 0
