from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from demo.envfile import load_env  # noqa: E402


def _clean(monkeypatch, *names):
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_loads_allowed_keys_and_strips_quotes_and_comments(tmp_path, monkeypatch):
    _clean(monkeypatch, "ELEVENLABS_API_KEY", "GEMINI_API_KEY", "ONEVOICE_X")
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        'ELEVENLABS_API_KEY="abc123"\n'
        "export GEMINI_API_KEY='g-key'\n"
        "ONEVOICE_X=plain # trailing note\n",
        encoding="utf-8",
    )
    assert load_env(env) == ["ELEVENLABS_API_KEY", "GEMINI_API_KEY", "ONEVOICE_X"]
    import os

    assert os.environ["ELEVENLABS_API_KEY"] == "abc123"
    assert os.environ["GEMINI_API_KEY"] == "g-key"
    assert os.environ["ONEVOICE_X"] == "plain"


def test_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("ELEVENLABS_API_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("ELEVENLABS_API_KEY=from-file\n", encoding="utf-8")
    assert load_env(env) == []
    assert os.environ["ELEVENLABS_API_KEY"] == "from-shell"


def test_unrelated_names_and_empty_values_are_ignored(tmp_path, monkeypatch):
    import os

    _clean(monkeypatch, "ELEVENLABS_API_KEY", "PATH_EVIL", "AWS_SECRET_ACCESS_KEY")
    before = os.environ.get("PATH")
    env = tmp_path / ".env"
    env.write_text(
        "PATH=/tmp/evil\nAWS_SECRET_ACCESS_KEY=nope\nELEVENLABS_API_KEY=\n",
        encoding="utf-8",
    )
    assert load_env(env) == []
    assert os.environ.get("PATH") == before
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ
    assert "ELEVENLABS_API_KEY" not in os.environ


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "absent.env") == []
