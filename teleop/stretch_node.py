import asyncio
from typing import Any, AsyncIterator, Self

from pydantic import ValidationError

from pubsub_server import Node, NodeFactory, Message
from pubsub_server.messages import Tick
from .data_classes import TargetPosition


def read_target_position_replay(target_dict) -> TargetPosition:
    # target_position_json_str = json.dumps(target_dict)
    return TargetPosition.model_validate_json(target_dict)


def read_target_position(
    file_path: str = "/dev/shm/target_position.json",
) -> TargetPosition:
    with open(file_path, "r") as f:
        return TargetPosition.model_validate_json(f.read())


def write_target_position(
    target_position: TargetPosition, file_path: str = "/dev/shm/current_position.json"
) -> None:
    with open(file_path, "w") as f:
        f.write(target_position.model_dump_json())


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
