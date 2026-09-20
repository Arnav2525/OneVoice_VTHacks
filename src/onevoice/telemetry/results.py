

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a core dependency
    yaml = None  # type: ignore[assignment]

def _config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]

class RunDirectory:

    def __init__(
        self,
        benchmark_name: str,
        config: dict[str, Any] | None = None,
        base_dir: str | Path = "runs",
    ) -> None:
        self.benchmark_name = benchmark_name
        self.config = config or {}
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        cfg_hash = _config_hash({"benchmark": benchmark_name, **self.config})
        self.run_id = f"{timestamp}-{benchmark_name}-{cfg_hash}"
        self.path = Path(base_dir) / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self._log_handler: logging.Handler | None = None

    def attach_logging(self, level: int = logging.INFO) -> None:

        handler = logging.FileHandler(self.path / "benchmark.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handler.setLevel(level)
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level > level:
            root.setLevel(level)
        self._log_handler = handler

    def detach_logging(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

    def write_config(self) -> None:
        payload = {"benchmark": self.benchmark_name, **self.config}
        target = self.path / "config.yaml"
        if yaml is not None:
            with open(target, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=True)
        else:  # pragma: no cover
            with open(self.path / "config.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)

    def write_hardware(self, hardware: dict[str, Any]) -> None:
        with open(self.path / "hardware.json", "w", encoding="utf-8") as f:
            json.dump(hardware, f, indent=2, default=str)

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        with open(self.path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        self._write_metrics_csv(metrics)

    def _write_metrics_csv(self, metrics: dict[str, Any]) -> None:
        flat = _flatten(metrics)
        with open(self.path / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in flat.items():
                writer.writerow([key, value])

    def write_report(self, report: str) -> None:
        with open(self.path / "report.txt", "w", encoding="utf-8") as f:
            f.write(report)

    def write_per_result_csv(
        self, rows: list[dict[str, Any]], filename: str = "per_result.csv"
    ) -> None:
        if not rows:
            return
        fieldnames = sorted({k for row in rows for k in row})
        with open(self.path / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, child))
    elif isinstance(data, (list, tuple)):
        for idx, value in enumerate(data):
            child = f"{prefix}[{idx}]"
            flat.update(_flatten(value, child))
    else:
        flat[prefix] = data
    return flat
