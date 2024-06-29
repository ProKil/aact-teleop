from typing import Any, AsyncGenerator, Generic, Self, Type, TypeVar
from pubsub_server.messages import Message

from abc import abstractmethod

from redis.asyncio import Redis

InputType = TypeVar("InputType", bound=Message)
OutputType = TypeVar("OutputType", bound=Message)


class Node(Generic[InputType, OutputType]):
    input_channels: list[str]
    output_channels: list[str]
    input_type: Type[InputType]
    output_type: Type[OutputType]

    def __init__(
        self,
        input_channels: list[str],
        output_channels: list[str],
        input_type: Type[InputType],
        output_type: Type[OutputType],
        redis_kwargs: dict[str, Any] = {},
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_type = input_type
        self.output_type = output_type

        try:
            self.r: Redis = Redis(**redis_kwargs)
        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                f"Could not connect to Redis with the provided arguments. {redis_kwargs}"
            )

        self.pubsub = self.r.pubsub()

    async def __enter__(self) -> Self:
        await self.pubsub.subscribe(*self.input_channels)
        return self

    async def __exit__(
        self,
    ) -> None:
        await self.pubsub.unsubscribe()
        self.r.close()

    async def _wait_for_input(
        self,
    ) -> AsyncGenerator[InputType, None]:
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                yield self.input_type.model_validate_strings(message["data"])

    async def event_loop(
        self,
    ) -> None:
        async for input_message in self._wait_for_input():
            output_channel, output_message = await self.event_handler(input_message)
            await self.r.publish(output_channel, output_message.model_dump_json())

    @abstractmethod
    async def event_handler(self, _: InputType) -> tuple[str, OutputType]:
        raise NotImplementedError("You must implement this method in your subclass.")
