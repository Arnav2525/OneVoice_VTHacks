

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

@dataclass
class VisualEncoderConfig:
    embed_dim: int = 64
    target_patch_size: int = 256

@dataclass
class RollingROIBuffer:

    max_len: int = 4
    embeddings: list[np.ndarray] = field(default_factory=list)

    def push(self, embedding: np.ndarray) -> None:
        self.embeddings.append(embedding)
        if len(self.embeddings) > self.max_len:
            self.embeddings.pop(0)

    def latest(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        return self.embeddings[-1]  # type: ignore[no-any-return]

    def mean(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)  # type: ignore[no-any-return]

class SimpleVisualEncoder:

    def __init__(self, config: VisualEncoderConfig | None = None) -> None:
        self._config = config or VisualEncoderConfig()
        rng = np.random.default_rng(42)
        self._proj = rng.standard_normal(
            (self._config.target_patch_size, self._config.embed_dim)
        ).astype(np.float32) / np.sqrt(self._config.target_patch_size)

    def encode(self, lip_patch: list[float] | np.ndarray | None) -> np.ndarray | None:
        if lip_patch is None:
            return None
        arr = np.asarray(lip_patch, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        if arr.size < self._config.target_patch_size:
            padded = np.zeros(self._config.target_patch_size, dtype=np.float32)
            padded[: arr.size] = arr
            arr = padded
        elif arr.size > self._config.target_patch_size:
            arr = arr[: self._config.target_patch_size]
        emb = arr @ self._proj
        norm = np.linalg.norm(emb) + 1e-6
        return (emb / norm).astype(np.float32)  # type: ignore[no-any-return]
