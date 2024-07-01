import asyncio
from typing import AsyncIterator

from pubsub_server.messages import Tick, Message
from .base import Node
from .registry import NodeFactory


@NodeFactory.register("tick")
class TickNode(Node[Message, Tick]):
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        super().__init__(
            input_channels=[],
            output_channels=[
                "tick/millis/1",
                "tick/millis/5",
                "tick/millis/10",
                "tick/millis/100",
                "tick/secs/1",
            ],
            input_type=Message,
            output_type=Tick,
            redis_url=redis_url,
        )

    async def event_loop(self) -> None:
        tick_count = 0
        while True:
            await self.r.publish(
                "tick/millis/1", Tick(tick=tick_count).model_dump_json()
            )
            if tick_count % 5 == 0:
                await self.r.publish(
                    "tick/millis/5", Tick(tick=tick_count).model_dump_json()
                )
            if tick_count % 10 == 0:
                await self.r.publish(
                    "tick/millis/10", Tick(tick=tick_count).model_dump_json()
                )
            if tick_count % 100 == 0:
                await self.r.publish(
                    "tick/millis/100", Tick(tick=tick_count).model_dump_json()
                )
            if tick_count % 1000 == 0:
                await self.r.publish(
                    "tick/secs/1", Tick(tick=tick_count).model_dump_json()
                )
            tick_count += 1
            await asyncio.sleep(0.0001)

    async def event_handler(self, _: Message) -> AsyncIterator[tuple[str, Tick]]:
        raise NotImplementedError("TickNode does not have an event handler.")
        yield "", Tick(tick=0)


async def _main() -> None:
    import os

    if "REDIS_URL" in os.environ:
        node = TickNode(redis_url=os.environ["REDIS_URL"])
    else:
        node = TickNode()
    async with node:
        await node.event_loop()


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_main())
    loop.close()
