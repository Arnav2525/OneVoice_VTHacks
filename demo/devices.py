

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)
AudioKind = Literal["input", "output"]

@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    hostapi: str
    hostapi_index: int
    input_channels: int
    output_channels: int
    default_sample_rate: float

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name} [{self.hostapi}]"

def _audio_backend() -> Any:
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Audio device discovery requires sounddevice/PortAudio. "
            'Install dependencies: python -m pip install -e ".[streaming]"'
        ) from exc
    return sounddevice

def _inventory(backend: Any) -> list[AudioDevice]:
    try:
        hostapis = backend.query_hostapis()
        devices = backend.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Could not enumerate audio devices: {exc}") from exc
    return [
        AudioDevice(
            index=i,
            name=str(device["name"]),
            hostapi=str(hostapis[int(device["hostapi"])]["name"]),
            hostapi_index=int(device["hostapi"]),
            input_channels=int(device["max_input_channels"]),
            output_channels=int(device["max_output_channels"]),
            default_sample_rate=float(device["default_samplerate"]),
        )
        for i, device in enumerate(devices)
    ]

def list_audio_devices() -> list[AudioDevice]:

    return _inventory(_audio_backend())

def format_device_list(devices: list[AudioDevice]) -> str:
    lines = ["Microphones (input devices):"]
    for device in devices:
        if device.input_channels:
            lines.append(f"  {device.label} ({device.input_channels} input channels)")
    lines.append("Headphones / speakers (output devices):")
    for device in devices:
        if device.output_channels:
            lines.append(f"  {device.label} ({device.output_channels} output channels)")
    lines.extend(
        [
            "",
            "Select the Logitech microphone by name: --input-device C270",
            "Or use the listed numeric IDs with --input-device and --output-device.",
            "IDs can change after reconnecting hardware. A name is preferred.",
            "Camera indices are separate: --camera-index 1 (try 2 if needed).",
            "Camera names are not enumerated; discovery does not open any devices.",
        ]
    )
    return "\n".join(lines)

def _same_endpoint_names(first: str, second: str) -> bool:

    first, second = first.casefold(), second.casefold()
    if first.strip() == second.strip():
        return True
    shorter, longer = sorted((first, second), key=len)
    return len(shorter) == 31 and longer.startswith(shorter)

def _check_one_endpoint(
    candidates: list[AudioDevice],
    value: str,
    kind: AudioKind,
) -> None:

    hosts = [device.hostapi_index for device in candidates]
    longest = max(candidates, key=lambda device: len(device.name))
    if len(set(hosts)) != len(hosts) or any(
        not _same_endpoint_names(device.name, longest.name) for device in candidates
    ):
        options = "; ".join(device.label for device in candidates)
        raise ValueError(
            f"More than one {kind} device matches {value!r}: {options}. "
            f"Use --{kind}-device with the exact numeric ID from --list-devices."
        )

def resolve_audio_device(
    value: int | str | None,
    *,
    kind: AudioKind,
    sample_rate: int = 16000,
    channels: int = 1,
) -> int | None:

    if kind not in ("input", "output"):
        raise ValueError("Audio device kind must be 'input' or 'output'")
    if value is None or (
        isinstance(value, str) and value.strip().casefold() == "default"
    ):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Invalid {kind} device {value!r}; use a name or numeric ID")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"The {kind} device name cannot be empty")
        if re.fullmatch(r"[+-]?\d+", value):
            value = int(value)
    if isinstance(value, int) and value < 0:
        raise ValueError(f"The {kind} device ID must be zero or greater")
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("Audio sample rate and channels must be positive")

    backend = _audio_backend()
    devices = _inventory(backend)
    capable = [
        device
        for device in devices
        if (device.input_channels if kind == "input" else device.output_channels)
        >= channels
    ]
    if isinstance(value, int):
        candidates = [device for device in capable if device.index == value]
    else:
        terms = value.casefold().split()
        candidates = [
            device
            for device in capable
            if all(
                term in f"{device.name} {device.hostapi}".casefold() for term in terms
            )
        ]

        if not candidates and value.casefold() == "logitech":
            candidates = [
                device for device in capable if "c270" in device.name.casefold()
            ]
        if candidates:
            _check_one_endpoint(candidates, value, kind)
    if not candidates:
        raise ValueError(
            f"Requested {kind} device {value!r} was not found "
            f"with {channels} channel(s). "
            "Reconnect it and run --list-devices, then choose its name or numeric ID. "
            "The system default will not be substituted."
        )

    priority = {"windows wasapi": 0, "windows directsound": 1, "mme": 2}
    candidates.sort(
        key=lambda device: (priority.get(device.hostapi.casefold(), 3), device.index)
    )
    check = (
        backend.check_input_settings
        if kind == "input"
        else backend.check_output_settings
    )
    errors = []
    for device in candidates:
        try:
            check(
                device=device.index,
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
            )
        except Exception as exc:
            errors.append(f"{device.label}: {exc}")
        else:
            logger.info(
                "Selected %s device %s at %s Hz",
                kind,
                device.label,
                sample_rate,
            )
            return device.index
    raise ValueError(
        f"The requested {kind} device does not support "
        f"{sample_rate} Hz / {channels} channel(s). "
        + "; ".join(errors)
        + ". Use --list-devices and select a compatible ID; "
        "no other device was substituted."
    )
