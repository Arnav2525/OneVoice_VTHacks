from typing import Protocol

from onevoice.core.models.audio_chunk import AudioChunk

class AudioSource(Protocol):

    def start(self) -> None:

        ...

    def stop(self) -> None:

        ...

    def read(self) -> AudioChunk:

        ...
