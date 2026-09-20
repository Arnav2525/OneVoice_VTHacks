

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from onevoice.separation.config import SeparationConfig
from onevoice.separation.exceptions import (
    BackendInitializationError,
    BackendOOMError,
)
from onevoice.separation.utilities import is_oom_error, select_device

logger = logging.getLogger(__name__)

Builder = Callable[[SeparationConfig, str], Any]
Warmup = Callable[[Any, SeparationConfig, str], None]

class BackendLoader:

    def __init__(
        self,
        config: SeparationConfig,
        builder: Builder,
        warmup: Warmup | None = None,
    ) -> None:
        self._config = config
        self._builder = builder
        self._warmup = warmup
        self._model: Any = None
        self._device: str = "cpu"
        self._lock = threading.Lock()
        self._load_failed = False
        self._load_time_ms = 0.0
        self._warmup_time_ms = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def load_time_ms(self) -> float:

        return self._load_time_ms

    @property
    def warmup_time_ms(self) -> float:

        return self._warmup_time_ms

    def get(self) -> Any:

        if self._model is not None:
            return self._model
        return self.load()

    def load(self) -> Any:

        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                raise BackendInitializationError(
                    f"backend '{self._config.name}' previously failed to load"
                )
            device = select_device(self._config.device)
            try:
                self._build_on(device)
            except BackendOOMError:
                if device.startswith("cuda"):
                    logger.warning("CUDA OOM building '%s'; retrying on CPU",
                                   self._config.name)
                    self._build_on("cpu")
                else:
                    self._load_failed = True
                    raise
            except BackendInitializationError:
                self._load_failed = True
                raise
            except Exception as exc:  # noqa: BLE001 - normalize all builder faults
                self._load_failed = True
                if is_oom_error(exc) and device.startswith("cuda"):
                    logger.warning("CUDA OOM building '%s'; retrying on CPU",
                                   self._config.name)
                    self._load_failed = False
                    self._build_on("cpu")
                else:
                    raise BackendInitializationError(
                        f"failed to load backend '{self._config.name}': {exc}"
                    ) from exc
            return self._model

    def _build_on(self, device: str) -> None:
        build_start = time.perf_counter()
        try:
            model = self._builder(self._config, device)
        except FileNotFoundError as exc:
            raise BackendInitializationError(
                f"checkpoint not found for '{self._config.name}': {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            if is_oom_error(exc):
                raise BackendOOMError(str(exc)) from exc
            raise
        self._load_time_ms = (time.perf_counter() - build_start) * 1000.0
        self._model = model
        self._device = device
        logger.info(
            "backend '%s' loaded on %s in %.0f ms",
            self._config.name,
            device,
            self._load_time_ms,
        )
        if self._config.warmup and self._warmup is not None:
            self._run_warmup(device)

    def _run_warmup(self, device: str) -> None:
        assert self._warmup is not None
        iterations = max(1, self._config.warmup_iterations)
        warmup_start = time.perf_counter()
        try:
            for _ in range(iterations):
                self._warmup(self._model, self._config, device)
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._load_failed = True
            raise BackendInitializationError(
                f"warmup failed for '{self._config.name}': {exc}"
            ) from exc
        self._warmup_time_ms = (time.perf_counter() - warmup_start) * 1000.0
        logger.info(
            "backend '%s' warmed up (%d iters) in %.0f ms",
            self._config.name,
            iterations,
            self._warmup_time_ms,
        )

    def unload(self) -> None:

        with self._lock:
            self._model = None
            self._load_failed = False
            self._load_time_ms = 0.0
            self._warmup_time_ms = 0.0
            if self._device.startswith("cuda"):
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # pragma: no cover
                    pass
            logger.info("backend '%s' unloaded", self._config.name)
