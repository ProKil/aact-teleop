from logging import getLogger
from typing import Any, AsyncIterator, Self

import zmq
from pubsub_server import Node, NodeFactory, Message
from pubsub_server.messages import Image
from pubsub_server.messages.commons import Text

from zmq.asyncio import Context, Socket


@NodeFactory.register("head_video_streamer")
class VideoStreamerNode(Node[Image, Text]):
    def __init__(
        self,
        input_channel: str = "meta_2_head_cam",
        output_channel: str = "head_video_stream",
        quest_controller_ip: str = "172.26.172.110",
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
        self.logger = getLogger(__name__)
        self.logger.debug("Initialized VideoStreamerNode")

    def _connect_quest(self) -> Socket:
        context = Context()
        quest_socket = context.socket(zmq.PUSH)

        quest_socket.connect(f"tcp://{self.quest_controller_ip}:12346")

        self.logger.debug("Socket Connected")

        return quest_socket

    async def event_handler(
        self, input_channel: str, head_image: Message[Image]
    ) -> AsyncIterator[tuple[str, Message[Text]]]:
        if input_channel == self.input_channel:
            await self.quest_socket.send(head_image.data.image)
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
