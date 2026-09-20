from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PREFIXES = ("ELEVENLABS_", "GEMINI_", "ONEVOICE_")
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _value(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def load_env(path: Path | None = None) -> list[str]:
    env_path = path or REPO_ROOT / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []
    loaded = []
    for line in lines:
        match = _LINE.match(line)
        if not match or line.lstrip().startswith("#"):
            continue
        name, value = match.group(1), _value(match.group(2))
        if not name.startswith(ALLOWED_PREFIXES) or not value or name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded
