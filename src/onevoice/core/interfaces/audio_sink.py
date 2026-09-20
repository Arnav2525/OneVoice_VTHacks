from typing import Protocol

from onevoice.core.models.audio_chunk import AudioChunk

class AudioSink(Protocol):

    def start(self) -> None:

        ...

    def stop(self) -> None:

        ...

    def write(self, chunk: AudioChunk) -> None:

        ...
