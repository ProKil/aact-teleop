from typing import Any, AsyncIterator, Self
import pyaudio
from pubsub_server import Node, Message, NodeFactory
from pubsub_server.messages import Audio, Zero


@NodeFactory.register("speaker")
class SpeakerNode(Node[Audio, Zero]):
    def __init__(
        self,
        input_channel: str,
    ):
        super().__init__(
            input_channel_types=[(input_channel, Audio)],
            output_channel_types=[],
        )
        self.input_channel = input_channel
        self.audio = pyaudio.PyAudio()
        self.stream: pyaudio.Stream | None = None

    async def __aenter__(self) -> Self:
        self.stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=44100, output=True
        )

        return await super().__aenter__()

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        return await super().__aexit__(_, __, ___)

    async def event_handler(
        self, input_channel: str, message: Message[Audio]
    ) -> AsyncIterator[tuple[str, Message[Zero]]]:
        if input_channel == self.input_channel:
            if self.stream:
                self.stream.write(message.data.audio)
            else:
                raise ValueError(
                    "Stream is not initialized. Please use the async context manager."
                )
        else:
            yield "", Message[Zero](data=Zero())
