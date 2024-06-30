from typing import Any, AsyncIterator, Generic, Self, Type, TypeVar
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
        redis_url: str = "redis://localhost:6379/0",
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_type = input_type
        self.output_type = output_type

        try:
            self.r: Redis = Redis.from_url(redis_url)
        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                f"Could not connect to Redis with the provided url. {redis_url}"
            )

        self.pubsub = self.r.pubsub()

    async def __aenter__(self) -> Self:
        await self.r.ping()
        await self.pubsub.subscribe(*self.input_channels)
        return self

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        await self.pubsub.unsubscribe()
        await self.r.close()

    async def _wait_for_input(
        self,
    ) -> AsyncIterator[InputType]:
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                yield self.input_type.model_validate_json(message["data"])

    async def event_loop(
        self,
    ) -> None:
        async for input_message in self._wait_for_input():
            async for output_channel, output_message in self.event_handler(
                input_message
            ):
                await self.r.publish(output_channel, output_message.model_dump_json())

    @abstractmethod
    async def event_handler(
        self, _: InputType
    ) -> AsyncIterator[tuple[str, OutputType]]:
        raise NotImplementedError("event_handler must be implemented in a subclass.")
        yield "", self.output_type()  # unreachable: dummy return value
