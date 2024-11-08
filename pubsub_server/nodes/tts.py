import asyncio
from typing import Any, AsyncIterator, Self
from pubsub_server.messages.base import Message
from pubsub_server.messages.commons import Audio, Text
from .base import Node
from .registry import NodeFactory
from google.cloud import texttospeech
from google.api_core import exceptions
from google.api_core.client_options import ClientOptions


@NodeFactory.register("tts")
class TTSNode(Node[Text, Audio]):
    def __init__(
        self,
        input_channel: str,
        output_channel: str,
        api_key: str,
        rate: int,
        redis_url: str,
    ) -> None:
        super().__init__(
            input_channel_types=[(input_channel, Text)],
            output_channel_types=[(output_channel, Audio)],
            redis_url=redis_url,
        )
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.client = texttospeech.TextToSpeechAsyncClient(
            client_options=ClientOptions(api_key=api_key)  # type: ignore[no-untyped-call]
        )
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=rate
        )
        self.queue = asyncio.Queue[str]()
        self.shutdown = asyncio.Event()

    async def synthesize(self) -> None:
        while not self.shutdown.is_set():
            text = await self.queue.get()
            try:
                input_text = texttospeech.SynthesisInput(text=text)
                response = await self.client.synthesize_speech(
                    request={
                        "input": input_text,
                        "voice": self.voice,
                        "audio_config": self.audio_config,
                    }
                )
                await self.r.publish(
                    self.output_channel,
                    Message[Audio](
                        data=Audio(audio=response.audio_content)
                    ).model_dump_json(),
                )
            except exceptions.GoogleAPICallError as e:
                print(f"Error during speech synthesis: {e}")

    async def __aenter__(self) -> Self:
        self.task = asyncio.create_task(self.synthesize())
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown.set()
        return await super().__aexit__(exc_type, exc_value, traceback)

    async def event_handler(
        self, input_channel: str, text: Message[Text]
    ) -> AsyncIterator[tuple[str, Message[Audio]]]:
        if input_channel != self.input_channel:
            raise ValueError(f"Unexpected input channel: {input_channel}")
            yield "", Message[Audio](data=Audio(audio=b""))
        else:
            await self.queue.put(text.data.text)
