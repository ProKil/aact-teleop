import asyncio
from typing import AsyncIterator
import pytest
from pytest_mock import MockerFixture
import pytest_asyncio
from pubsub_server import Node, Message
from pydantic import BaseModel
import fakeredis
from redis.asyncio import Redis


class InputModel(BaseModel):
    data: str


class OutputModel(BaseModel):
    result: str


class _Node(Node[InputModel, OutputModel]):
    async def event_handler(
        self, input_channel: str, input_message: Message[InputModel]
    ) -> AsyncIterator[tuple[str, Message[OutputModel]]]:
        assert input_channel == "input_channel"
        assert input_message.data.data == "test"
        yield (
            "output_channel",
            Message[OutputModel](data=OutputModel(result="processed")),
        )


@pytest_asyncio.fixture
async def fake_redis() -> Redis:
    return fakeredis.aioredis.FakeRedis.from_url("redis://localhost:6379/0")


@pytest_asyncio.fixture
async def node(mocker: MockerFixture) -> AsyncIterator[_Node]:
    redis_url = "redis://localhost:6379/0"
    mocker.patch("pubsub_server.nodes.base.Redis", fakeredis.aioredis.FakeRedis)
    node_instance = _Node(
        input_channel_types=[("input_channel", InputModel)],
        output_channel_types=[("output_channel", OutputModel)],
        redis_url=redis_url,
    )
    yield node_instance
    # Teardown can be handled here if needed


@pytest.mark.asyncio
async def test_initialization(node: _Node) -> None:
    assert node.input_channel_types == {"input_channel": InputModel}
    assert node.output_channel_types == {"output_channel": OutputModel}
    assert node.redis_url == "redis://localhost:6379/0"


@pytest.mark.asyncio
async def test_connection(node: _Node) -> None:
    async with node:
        assert True


async def mock_input_message(fake_redis: Redis) -> None:
    await fake_redis.publish(
        "input_channel",
        Message[InputModel](data=InputModel(data="test")).model_dump_json(),
    )


async def check_output_message(fake_redis: Redis) -> None:
    async with fake_redis.pubsub() as pubsub:
        await pubsub.subscribe("output_channel")
        async for message in pubsub.listen():
            if message["type"] == "message":
                assert Message[OutputModel].model_validate_json(
                    message["data"]
                ) == Message[OutputModel](data=OutputModel(result="processed"))
                raise Exception("Successfully processed the message")


@pytest.mark.asyncio
async def test_event_loop(node: _Node, fake_redis: Redis) -> None:
    async with node:
        try:
            async with asyncio.timeout(1):
                await asyncio.gather(
                    node.event_loop(),
                    mock_input_message(fake_redis),
                    check_output_message(fake_redis),
                )
        except asyncio.TimeoutError:
            raise Exception("Timed out waiting for the output message")
        except Exception as e:
            assert str(e) == "Successfully processed the message"
