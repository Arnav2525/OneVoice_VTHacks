

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class StructuredLogger:

    def __init__(
        self,
        name: str,
        experiment_id: str = "default",
        git_commit: str | None = None,
        hardware_fingerprint: str | None = None,
        log_dir: str = "runs",
    ) -> None:
        self.name = name
        self.experiment_id = experiment_id
        self.git_commit = git_commit
        self.hardware_fingerprint = hardware_fingerprint
        self.log_dir = Path(log_dir) / experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

        self.json_file = self.log_dir / "logs.json"
        self.csv_file = self.log_dir / "logs.csv"
        self._init_csv()

        self.tb_dir = self.log_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)

    def _init_csv(self) -> None:
        if not self.csv_file.exists():
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "level",
                        "message",
                        "experiment_id",
                        "git_commit",
                        "hardware_fingerprint",
                    ]
                )

    def _get_base_context(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": self.experiment_id,
            "git_commit": self.git_commit,
            "hardware_fingerprint": self.hardware_fingerprint,
        }

    def log(self, level: int, message: str, **kwargs: Any) -> None:
        self.logger.log(level, message)

        context = self._get_base_context()
        context["level"] = logging.getLevelName(level)
        context["message"] = message
        context.update(kwargs)

        with open(self.json_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(context) + "\n")

        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    context["timestamp"],
                    context["level"],
                    context["message"],
                    context["experiment_id"],
                    context["git_commit"],
                    context["hardware_fingerprint"],
                ]
            )

    def info(self, message: str, **kwargs: Any) -> None:
        self.log(logging.INFO, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.log(logging.ERROR, message, **kwargs)
