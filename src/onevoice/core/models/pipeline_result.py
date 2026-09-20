from dataclasses import dataclass

from .audio_chunk import AudioChunk
from .frame import Frame
from .target_selection import TargetSelection

@dataclass(frozen=True)
class PipelineResult:

    timestamp_ms: float
    separated_audio: AudioChunk | None
    processed_frame: Frame | None
    active_target: TargetSelection
    latency_metrics: dict[str, float]
