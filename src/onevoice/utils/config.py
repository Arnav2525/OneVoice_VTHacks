

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

class DemoConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

class AppConfig(BaseModel):
    hardware: HardwareConfig = HardwareConfig()
    backend: BackendConfig = BackendConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    demo: DemoConfig = DemoConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()

def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return AppConfig()
    return AppConfig.model_validate(data)
