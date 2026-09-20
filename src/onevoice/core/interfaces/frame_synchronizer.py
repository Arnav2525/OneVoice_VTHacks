from typing import Protocol

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.frame import Frame

class FrameSynchronizer(Protocol):

    def push_audio(self, chunk: AudioChunk) -> None:

        ...

    def push_video(self, frame: Frame) -> None:

        ...

    def get_synchronized_pair(self) -> tuple[AudioChunk, Frame]:

        ...
