from typing import Any, AsyncIterator, Generic, Self, Type, TypeVar
from pydantic import BaseModel, ConfigDict

from abc import abstractmethod
from pubsub_server.messages import Message
from redis.asyncio import Redis

from pubsub_server.messages.base import DataModel

InputType = TypeVar("InputType", covariant=True, bound=DataModel)
OutputType = TypeVar("OutputType", covariant=True, bound=DataModel)


class Node(BaseModel, Generic[InputType, OutputType]):
    input_channel_types: dict[str, Type[InputType]]
    output_channel_types: dict[str, Type[OutputType]]
    redis_url: str
    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        input_channel_types: list[tuple[str, Type[InputType]]],
        output_channel_types: list[tuple[str, Type[OutputType]]],
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channel_types=dict(input_channel_types),
            output_channel_types=dict(output_channel_types),
            redis_url=redis_url,
        )

        self.r: Redis = Redis.from_url(redis_url)
        self.pubsub = self.r.pubsub()

    async def __aenter__(self) -> Self:
        try:
            await self.r.ping()
        except ConnectionError:
            raise ValueError(
                f"Could not connect to Redis with the provided url. {self.redis_url}"
            )
        await self.pubsub.subscribe(*self.input_channel_types.keys())
        return self

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        await self.pubsub.unsubscribe()
        await self.r.aclose()

    async def _wait_for_input(
        self,
    ) -> AsyncIterator[tuple[str, Message[InputType]]]:
        async for message in self.pubsub.listen():
            channel = message["channel"].decode("utf-8")
            if message["type"] == "message" and channel in self.input_channel_types:
                data = Message[self.input_channel_types[channel]].model_validate_json(  # type: ignore
                    message["data"]
                )
                yield channel, data
        raise Exception("Input channel closed unexpectedly")

    async def event_loop(
        self,
    ) -> None:
        async for input_channel, input_message in self._wait_for_input():
            async for output_channel, output_message in self.event_handler(
                input_channel, input_message
            ):
                await self.r.publish(output_channel, output_message.model_dump_json())
        raise Exception("Event loop exited unexpectedly")

    @abstractmethod
    async def event_handler(
        self, _: str, __: Message[InputType]
    ) -> AsyncIterator[tuple[str, Message[OutputType]]]:
        raise NotImplementedError("event_handler must be implemented in a subclass.")
        yield "", self.output_type()  # unreachable: dummy return value
