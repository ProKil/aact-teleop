import asyncio
from typing import AsyncIterator, Self

from pubsub_server.messages import Tick, Message, Zero
from .base import Node
from .registry import NodeFactory
import time


@NodeFactory.register("tick")
class TickNode(Node[Zero, Tick]):
    def __init__(
        self,
        millis_intervals: list[int] = [],
        secs_intervals: list[int] = [],
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channel_types=[],
            output_channel_types=[
                (f"tick/millis/{interval}", Tick) for interval in millis_intervals
            ]
            + [(f"tick/secs/{interval}", Tick) for interval in secs_intervals],
            redis_url=redis_url,
        )
        self.millis_intervals = millis_intervals
        self.secs_intervals = secs_intervals

    async def tick_at_given_interval(self, channel: str, interval: float) -> None:
        tick_count = 0
        last: float | None = None
        last_sleep = interval
        while True:
            await self.r.publish(
                channel, Message[Tick](data=Tick(tick=tick_count)).model_dump_json()
            )
            tick_count += 1
            now = time.time()
            if last is not None:
                last_sleep = last_sleep - (now - last - interval)
            await asyncio.sleep(last_sleep)
            last = now

    async def event_loop(self) -> None:
        await asyncio.gather(
            *(
                self.tick_at_given_interval(f"tick/millis/{interval}", interval / 1000)
                for interval in self.millis_intervals
            ),
            *(
                self.tick_at_given_interval(f"tick/secs/{interval}", interval)
                for interval in self.secs_intervals
            ),
        )

    async def __aenter__(self) -> Self:
        return await super().__aenter__()

    async def event_handler(
        self, _: str, __: Message[Zero]
    ) -> AsyncIterator[tuple[str, Message[Tick]]]:
        raise NotImplementedError("TickNode does not have an event handler.")
        yield "", Message[Tick](data=Tick(tick=0))
