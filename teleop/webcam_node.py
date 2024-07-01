import asyncio
from logging import getLogger
import time
from typing import Any, AsyncIterator, Self

import cv2
from pubsub_server import Node, NodeFactory, Message
from pubsub_server.messages import Tick, Image


@NodeFactory.register("webcam")
class WebcamNode(Node[Tick, Image]):
    def __init__(
        self,
        output_channel: str,
        webcam_id: str,
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channel_types=[
                ("tick/millis/10", Tick),
            ],
            output_channel_types=[
                (output_channel, Image),
            ],
            redis_url=redis_url,
        )
        self.output_channel = output_channel
        self.logger = getLogger(__name__)
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.latest_frame: bytes | None = None
        self.frame_lock: asyncio.Lock = asyncio.Lock()
        self.task: asyncio.Task[None] | None = None
        self.webcam_id = webcam_id

    async def update_video_feed(self) -> None:
        """
        Use a different process to update the video feed.
        """
        camera = cv2.VideoCapture(self.webcam_id, cv2.CAP_ANY)
        self.logger.debug("Starting video feed.")

        while not self.shutdown_event.is_set():
            start_time = time.time()
            success, camera_frame = camera.read()
            if not success:
                continue

            camera_frame = cv2.rotate(camera_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ret, buffer = cv2.imencode(".jpg", camera_frame)
            self.latest_frame = buffer.tobytes()
            end_time = time.time()
            await asyncio.sleep(end_time - start_time - 1 / 30)

    async def __aenter__(self) -> Self:
        if self.task is not None:
            raise Exception("WebcamNode is already open.")
        self.task = asyncio.create_task(self.update_video_feed())
        return await super().__aenter__()

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        self.shutdown_event.set()
        self.task = None
        await super().__aexit__(_, __, ___)

    async def event_handler(
        self, _: str, __: Message[Tick]
    ) -> AsyncIterator[tuple[str, Message[Image]]]:
        if self.latest_frame is not None:
            self.logger.debug("Sending frame...")
            yield (
                self.output_channel,
                Message[Image](data=Image(image=self.latest_frame)),
            )
        else:
            return
