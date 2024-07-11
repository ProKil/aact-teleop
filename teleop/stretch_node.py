import asyncio
from typing import Any, AsyncIterator, Self

from pydantic import ValidationError

from pubsub_server import Node, NodeFactory, Message
from pubsub_server.messages import Tick
from .data_classes import TargetPosition

from .stretch_control_loop import read_target_position, write_target_position


@NodeFactory.register("stretch")
class StretchNode(Node[TargetPosition | Tick, TargetPosition]):
    def __init__(
        self,
        input_channel: str,
        input_tick_channel: str,
        output_channel: str,
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channel_types=[
                (input_channel, TargetPosition),
                (input_tick_channel, Tick),
            ],
            output_channel_types=[
                (output_channel, TargetPosition),
            ],
            redis_url=redis_url,
        )
        self.input_tick_channel = input_tick_channel
        self.output_channel = output_channel
        self.tasks: list[asyncio.Task[None]] = []
        self.target_position: TargetPosition = TargetPosition()
        self.current_position: TargetPosition = TargetPosition()

    async def __aenter__(self) -> Self:
        return await super().__aenter__()

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        return await super().__aexit__(_, __, ___)

    async def event_handler(
        self, input_channel: str, input_message: Message[TargetPosition | Tick]
    ) -> AsyncIterator[tuple[str, Message[TargetPosition]]]:
        if input_channel == self.input_tick_channel:
            try:
                self.current_position = read_target_position(
                    "/dev/shm/current_position.json"
                )
            except FileNotFoundError:
                return
            except ValidationError:
                return
            yield (
                self.output_channel,
                Message[TargetPosition](data=self.current_position),
            )
        else:
            assert isinstance(input_message.data, TargetPosition)
            self.target_position = input_message.data
            write_target_position(self.target_position, "/dev/shm/target_position.json")
