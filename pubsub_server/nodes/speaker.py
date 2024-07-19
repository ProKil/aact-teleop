from typing import Any, AsyncIterator, Self
import pyaudio

from pubsub_server.messages.base import DataModel
from pubsub_server.messages.registry import DataModelFactory
from .base import Node
from .registry import NodeFactory
from pubsub_server.messages import Audio, Zero, Message


@NodeFactory.register("speaker")
class SpeakerNode(Node[Audio | DataModel, Zero]):
    def __init__(
        self,
        *,
        input_channel: str,
        interrupt_channel_types: dict[str, str] = {},
        redis_url: str,
    ):
        super().__init__(
            input_channel_types=[
                (input_channel, Audio),
                *(
                    (channel, DataModelFactory.registry[data_type])
                    for channel, data_type in interrupt_channel_types.items()
                ),
            ],
            output_channel_types=[],
            redis_url=redis_url,
        )
        self.input_channel = input_channel
        self.interrupt_channel_types = interrupt_channel_types
        self.audio = pyaudio.PyAudio()
        self.stream: pyaudio.Stream | None = None
        self.audio_bytes = b""

    def callback(
        self, _: bytes | None, frame_count: int, __: Any, ___: int
    ) -> tuple[bytes, int]:
        if len(self.audio_bytes) < frame_count * 2:
            return b"\x00" * frame_count * 2, pyaudio.paContinue
        else:
            data = self.audio_bytes[: frame_count * 2]
            self.audio_bytes = self.audio_bytes[frame_count * 2 :]
            return data, pyaudio.paContinue

    async def __aenter__(self) -> Self:
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            output=True,
            frames_per_buffer=1024,
            stream_callback=self.callback,
        )

        return await super().__aenter__()

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        return await super().__aexit__(_, __, ___)

    async def event_handler(
        self, input_channel: str, message: Message[Audio | DataModel]
    ) -> AsyncIterator[tuple[str, Message[Zero]]]:
        if input_channel == self.input_channel:
            if self.stream:
                assert isinstance(message.data, Audio)
                self.audio_bytes = message.data.audio
            else:
                raise ValueError(
                    "Stream is not initialized. Please use the async context manager."
                )
        elif input_channel in self.interrupt_channel_types:
            self.audio_bytes = b""
        else:
            yield "", Message[Zero](data=Zero())
