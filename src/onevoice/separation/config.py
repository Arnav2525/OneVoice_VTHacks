

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

_VALID_PRECISIONS = {"float32", "float16", "bfloat16"}

class SeparationConfig(BaseModel):

    model_config = ConfigDict(extra="allow")

    name: str = "passthrough"
    model_id: str | None = None
    checkpoint: str | None = None
    device: str = "auto"
    precision: str = "float32"
    batch_size: int = 1
    chunk_size: int | None = None
    sample_rate: int = 16000
    resample: bool = True
    warmup: bool = True
    warmup_iterations: int = 2
    timeout_ms: float = 1000.0
    fallback_on_error: bool = True
    params: dict[str, Any] = {}

    @field_validator("name")
    @classmethod
    def _lower_name(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("precision")
    @classmethod
    def _check_precision(cls, value: str) -> str:
        aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        value = aliases.get(value.lower(), value.lower())
        if value not in _VALID_PRECISIONS:
            raise ValueError(
                f"precision must be one of {sorted(_VALID_PRECISIONS)}, got '{value}'"
            )
        return value

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"auto", "cpu", "cuda"} and not value.startswith("cuda:"):
            raise ValueError(
                f"device must be 'auto', 'cpu', 'cuda', or 'cuda:N', got '{value}'"
            )
        return value

    @field_validator("batch_size")
    @classmethod
    def _check_batch(cls, value: int) -> int:
        if value < 1:
            raise ValueError("batch_size must be >= 1")
        return value

    @field_validator("timeout_ms")
    @classmethod
    def _check_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_ms must be > 0")
        return value

    @classmethod
    def from_raw(cls, raw: Any) -> SeparationConfig:

        if raw is None:
            return cls()
        if isinstance(raw, SeparationConfig):
            return raw
        if isinstance(raw, str):
            return cls(name=raw)
        if isinstance(raw, BaseModel):
            return cls.model_validate(raw.model_dump())
        if isinstance(raw, dict):
            return cls.model_validate(raw)
        raise TypeError(f"Cannot build SeparationConfig from {type(raw).__name__}")
