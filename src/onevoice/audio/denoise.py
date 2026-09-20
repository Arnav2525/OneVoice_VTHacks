

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from onevoice.core.models.audio_chunk import AudioChunk

_INPUT_SHAPES = {
    "mix": (1, 257, 1, 2),
    "conv_cache": (2, 1, 16, 16, 33),
    "tra_cache": (2, 3, 1, 1, 16),
    "inter_cache": (2, 1, 33, 16),
}
_OUTPUT_NAMES = ["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"]

def _load_session(model_path: Path, num_threads: int) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "GTCRN requires onnxruntime. Install the noise-suppression dependency "
            "before starting a session: python -m pip install onnxruntime"
        ) from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.inter_op_num_threads = 1
    try:
        return ort.InferenceSession(
            str(model_path),
            options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load GTCRN noise model {model_path}: {exc}"
        ) from exc

class GtcrnDenoiser:

    sample_rate = 16000
    frame_size = 512
    hop_size = 256
    delay_samples = 512
    delay_ms = 32.0

    def __init__(
        self,
        model_path: str | Path = "checkpoints/gtcrn/gtcrn_simple.onnx",
        num_threads: int = 1,
    ) -> None:
        if (
            isinstance(num_threads, bool)
            or not isinstance(num_threads, int)
            or num_threads < 1
        ):
            raise ValueError("GTCRN num_threads must be a positive integer")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"GTCRN noise model missing: {self.model_path}. "
                "Run python scripts/fetch_gtcrn.py before starting a session."
            )
        self._session = _load_session(self.model_path, num_threads)
        inputs = {item.name: item for item in self._session.get_inputs()}
        outputs = {item.name: item for item in self._session.get_outputs()}
        if set(inputs) != set(_INPUT_SHAPES) or set(outputs) != set(_OUTPUT_NAMES):
            raise ValueError("GTCRN model has an unsupported input/output interface")
        for name, shape in _INPUT_SHAPES.items():
            output_name = "enh" if name == "mix" else name + "_out"
            for item in (inputs[name], outputs[output_name]):
                if tuple(item.shape) != shape or item.type != "tensor(float)":
                    raise ValueError(
                        f"GTCRN model has an unsupported tensor: {item.name}"
                    )

        self._window = np.sqrt(np.hanning(self.frame_size + 1)[:-1]).astype(np.float32)
        self.reset()

        self._infer(np.zeros(self.frame_size, dtype=np.float32))
        self.reset()

    def reset(self) -> None:
        self._key: Any = None
        self._states = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in _INPUT_SHAPES.items()
            if name != "mix"
        }

        self._pending = np.zeros(self.hop_size, dtype=np.float32)
        self._overlap = np.zeros(self.frame_size, dtype=np.float32)
        self._weight = np.zeros(self.frame_size, dtype=np.float32)
        self._discard = self.hop_size
        self._ready = np.zeros(self.delay_samples, dtype=np.float32)

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(frame * self._window)
        model_input = np.stack((spectrum.real, spectrum.imag), axis=-1)
        model_input = model_input.astype(np.float32)[None, :, None, :]
        try:
            outputs = self._session.run(
                _OUTPUT_NAMES,
                {"mix": model_input, **self._states},
            )
        except Exception as exc:
            raise RuntimeError(f"GTCRN noise suppression failed: {exc}") from exc
        if len(outputs) != len(_OUTPUT_NAMES):
            raise RuntimeError("GTCRN returned an unexpected number of output tensors")
        for (name, shape), output in zip(_INPUT_SHAPES.items(), outputs):
            if output.shape != shape or not np.isfinite(output).all():
                raise RuntimeError(f"GTCRN returned invalid {name} audio/state")
        self._states = dict(zip(list(_INPUT_SHAPES)[1:], outputs[1:]))
        enhanced = outputs[0][0, :, 0]
        time_frame = np.fft.irfft(
            enhanced[:, 0] + 1j * enhanced[:, 1],
            n=self.frame_size,
        )
        return time_frame.astype(np.float32) * self._window

    def process(self, chunk: AudioChunk) -> AudioChunk:
        if chunk.sample_rate != self.sample_rate or chunk.channels != 1:
            raise ValueError("GTCRN requires 16000 Hz mono audio")
        samples = np.asarray(chunk.data, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError("GTCRN received non-finite audio")
        key = (
            chunk.metadata.get("target_track_id"),
            chunk.metadata.get("ui_selection_epoch"),
        )
        if key != self._key:
            self.reset()
            self._key = key
        self._pending = np.concatenate((self._pending, samples))
        generated = []
        offset = 0
        try:
            while self._pending.size - offset >= self.frame_size:
                frame = self._pending[offset : offset + self.frame_size]
                self._overlap += self._infer(frame)
                self._weight += self._window**2
                hop = np.divide(
                    self._overlap[: self.hop_size],
                    self._weight[: self.hop_size],
                    out=np.zeros(self.hop_size, dtype=np.float32),
                    where=self._weight[: self.hop_size] > 1e-8,
                )
                if self._discard:
                    self._discard -= self.hop_size
                else:
                    generated.append(hop)
                self._overlap[: self.hop_size] = self._overlap[self.hop_size :]
                self._overlap[self.hop_size :] = 0
                self._weight[: self.hop_size] = self._weight[self.hop_size :]
                self._weight[self.hop_size :] = 0
                offset += self.hop_size
        except Exception:
            self.reset()
            raise
        self._pending = self._pending[offset:].copy()
        if generated:
            self._ready = np.concatenate((self._ready, *generated))
        if self._ready.size < samples.size:
            self.reset()
            raise RuntimeError("GTCRN output buffer underrun")
        output = self._ready[: samples.size].copy()
        self._ready = self._ready[samples.size :].copy()
        return replace(
            chunk,
            data=output.tolist(),
            metadata={
                **chunk.metadata,
                "noise_suppression": "gtcrn",
                "denoise_delay_samples": self.delay_samples,
                "denoise_delay_ms": self.delay_ms,
            },
        )
