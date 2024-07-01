from typing import AsyncIterator
from pydantic import BaseModel
from pubsub_server import Node, NodeFactory
from pubsub_server.messages import Tick
from .data_classes import TargetPosition


class TargetPositionMessage(BaseModel):
    position: TargetPosition


@NodeFactory.register("stretch")
class StretchNode(Node[TargetPositionMessage | Tick, TargetPositionMessage]):
    def __init__(
        self,
        input_channels: list[str],
        output_channels: list[str],
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            input_type=TargetPositionMessage | Tick,
            output_type=TargetPositionMessage,
            redis_url=redis_url,
        )
        self.target_position: TargetPosition = TargetPosition()

    async def event_handler(
        self, message: TargetPositionMessage | Tick
    ) -> AsyncIterator[tuple[str, TargetPositionMessage]]:
        match message:
            case TargetPositionMessage(position=target_position):
                target_position
            case Tick():
                # Do something with tick
                pass
