from logging import getLogger, basicConfig, INFO
from typing import Any, AsyncIterator, Self

import zmq
from aact import Node, NodeFactory, Message
from aact.messages import Image
from aact.messages.commons import Text

from zmq.asyncio import Context, Socket


@NodeFactory.register("video_streamer")
class VideoStreamerNode(Node[Image, Text]):
    def __init__(
        self,
        input_channel: str,
        output_channel: str,
        quest_controller_ip: str,
        quest_receiving_port: int = 12346,
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        super().__init__(
            input_channel_types=[(input_channel, Image)],
            output_channel_types=[(output_channel, Text)],
            redis_url=redis_url,
        )
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.quest_controller_ip = quest_controller_ip
        self.quest_receiving_port = quest_receiving_port
        self.logger = getLogger(__name__)
        basicConfig(level=INFO)

    def _connect_quest(self) -> Socket:
        context = Context()
        quest_socket = context.socket(zmq.PUSH)
        quest_socket.setsockopt(zmq.CONFLATE, 1)

        quest_socket.connect(
            f"tcp://{self.quest_controller_ip}:{self.quest_receiving_port}"
        )

        self.logger.info("Quest socket connected")

        return quest_socket

    async def event_handler(
        self, input_channel: str, image: Message[Image]
    ) -> AsyncIterator[tuple[str, Message[Text]]]:
        if input_channel == self.input_channel:
            await self.quest_socket.send(image.data.image)
            yield (
                self.output_channel,
                Message[Text](data=Text(text="Image sent to Quest")),
            )
        else:
            yield (
                self.output_channel,
                Message[Text](data=Text(text="Image sent to Quest")),
            )

    async def __aenter__(self) -> Self:
        self.quest_socket = self._connect_quest()
        return await super().__aenter__()

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        self.quest_socket.close()
        await super().__aexit__(_, __, ___)
