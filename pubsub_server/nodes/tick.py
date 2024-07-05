import asyncio
from typing import AsyncIterator, Self

from pubsub_server.messages import Tick, Message, Zero
from .base import Node
from .registry import NodeFactory
import time


@NodeFactory.register("tick")
class TickNode(Node[Zero, Tick]):
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        super().__init__(
            input_channel_types=[],
            output_channel_types=[
                ("tick/millis/10", Tick),
                ("tick/millis/20", Tick),
                ("tick/millis/50", Tick),
                ("tick/millis/100", Tick),
                ("tick/secs/1", Tick),
            ],
            redis_url=redis_url,
        )
        self.task_queue: asyncio.Queue[tuple[str, Tick]] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None

    async def _send_tick(self) -> None:
        while True:
            channel, tick = await self.task_queue.get()
            await self.r.publish(channel, Message[Tick](data=tick).model_dump_json())
            self.task_queue.task_done()

    async def event_loop(self) -> None:
        tick_count = 0
        last: float | None = None
        last_sleep = 0.01
        while True:
            if tick_count % 10 == 0:
                await self.task_queue.put(("tick/millis/10", Tick(tick=tick_count)))
            if tick_count % 20 == 0:
                await self.task_queue.put(("tick/millis/20", Tick(tick=tick_count)))
            if tick_count % 50 == 0:
                await self.task_queue.put(("tick/millis/50", Tick(tick=tick_count)))
            if tick_count % 100 == 0:
                await self.task_queue.put(("tick/millis/100", Tick(tick=tick_count)))
            if tick_count % 1000 == 0:
                await self.task_queue.put(("tick/sec/1", Tick(tick=tick_count)))
            tick_count += 10
            now = time.time()
            if last is not None:
                last_sleep = last_sleep - (now - last - 0.01)
            await asyncio.sleep(last_sleep)
            last = now

    async def __aenter__(self) -> Self:
        self.task = asyncio.create_task(self._send_tick())
        return await super().__aenter__()

    async def event_handler(
        self, _: str, __: Message[Zero]
    ) -> AsyncIterator[tuple[str, Message[Tick]]]:
        raise NotImplementedError("TickNode does not have an event handler.")
        yield "", Message[Tick](data=Tick(tick=0))
