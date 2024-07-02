import asyncio
from typing import AsyncIterator

from pubsub_server.messages import Tick, Message, Zero
from .base import Node
from .registry import NodeFactory


@NodeFactory.register("tick")
class TickNode(Node[Zero, Tick]):
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        super().__init__(
            input_channel_types=[],
            output_channel_types=[
                ("tick/millis/1", Tick),
                ("tick/millis/5", Tick),
                ("tick/millis/10", Tick),
                ("tick/millis/100", Tick),
                ("tick/secs/1", Tick),
            ],
            redis_url=redis_url,
        )

    async def event_loop(self) -> None:
        tick_count = 0
        while True:
            await self.r.publish(
                "tick/millis/1",
                Message[Tick](data=Tick(tick=tick_count)).model_dump_json(),
            )
            if tick_count % 5 == 0:
                await self.r.publish(
                    "tick/millis/5",
                    Message[Tick](data=Tick(tick=tick_count)).model_dump_json(),
                )
            if tick_count % 10 == 0:
                await self.r.publish(
                    "tick/millis/10",
                    Message[Tick](data=Tick(tick=tick_count)).model_dump_json(),
                )
            if tick_count % 100 == 0:
                await self.r.publish(
                    "tick/millis/100",
                    Message[Tick](data=Tick(tick=tick_count)).model_dump_json(),
                )
            if tick_count % 1000 == 0:
                await self.r.publish(
                    "tick/secs/1",
                    Message[Tick](data=Tick(tick=tick_count)).model_dump_json(),
                )
            tick_count += 1
            await asyncio.sleep(0.001)

    async def event_handler(
        self, _: str, __: Message[Zero]
    ) -> AsyncIterator[tuple[str, Message[Tick]]]:
        raise NotImplementedError("TickNode does not have an event handler.")
        yield "", Message[Tick](data=Tick(tick=0))
