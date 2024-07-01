import asyncio
from typing import Any, AsyncIterator, Self
from pubsub_server import Node, NodeFactory
from pubsub_server.messages import Tick, Image, Message
from open_gopro import WiredGoPro, Params
from open_gopro.constants import WebcamStatus, WebcamError
import numpy.typing as npt
import numpy as np
from logging import getLogger

import cv2


@NodeFactory.register("gopro")
class GoProNode(Node[Tick, Image]):
    def __init__(
        self, output_channel: str, redis_url: str = "redis://localhost:6379/0"
    ):
        super().__init__(
            input_channels=["tick/millis/10"],
            output_channels=[output_channel],
            input_type=Tick,
            output_type=Image,
            redis_url=redis_url,
        )
        self.logger = getLogger(__name__)
        self.gopro: WiredGoPro | None = None
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.latest_frame: bytes | None = None
        self.frame_lock: asyncio.Lock = asyncio.Lock()
        self.task: asyncio.Task[None] | None = None

    async def capture_frames(self) -> None:
        cap: cv2.VideoCapture = cv2.VideoCapture("udp://0.0.0.0:8554")
        self.logger.info("Starting frame capture...")
        try:
            while not self.shutdown_event.is_set():
                success: bool
                frame: npt.NDArray[np.uint8]
                success, frame = cap.read()  # type: ignore[assignment]
                buffer: npt.NDArray[np.uint8]
                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes: bytes = buffer.tobytes()
                if not success:
                    self.logger.warning("Failed to read frame from camera.")
                    await asyncio.sleep(0.1)
                    continue
                async with self.frame_lock:
                    self.latest_frame = frame_bytes
                await asyncio.sleep(0.01)
        except Exception as e:
            self.logger.error(f"Error in capture_frames: {e}")
        finally:
            cap.release()

    async def __aenter__(self) -> Self:
        if self.gopro is not None:
            raise Exception("GoProNode is already open.")
        self.logger.info("Initializing GoPro connection...")
        self.gopro = WiredGoPro()
        await self.gopro.open()
        self.logger.info("GoPro connection established.")

        await self.gopro.http_command.wired_usb_control(control=Params.Toggle.DISABLE)
        await self.gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)

        # Start webcam
        current_status: WebcamStatus = (
            await self.gopro.http_command.webcam_status()
        ).data.status
        self.logger.debug(f"Current webcam status: {current_status}")
        if current_status not in {WebcamStatus.OFF, WebcamStatus.IDLE}:
            self.logger.info("Stopping existing webcam session...")
            await self.gopro.http_command.webcam_stop()

        self.logger.info("Starting webcam...")
        start_status: WebcamError = (
            await self.gopro.http_command.webcam_start()
        ).data.error
        if start_status != WebcamError.SUCCESS:
            raise Exception(f"Couldn't start webcam: {start_status}")

        # Wait for webcam to be ready
        while True:
            status: WebcamStatus = (
                await self.gopro.http_command.webcam_status()
            ).data.status
            self.logger.debug(f"Waiting for webcam... Current status: {status}")
            if status == WebcamStatus.IDLE:
                self.logger.info("Webcam is idle. Try starting again.")
                await self.gopro.http_command.webcam_start()
            if status == WebcamStatus.HIGH_POWER_PREVIEW:
                self.logger.info("Webcam is ready.")
                break
            await asyncio.sleep(0.5)

        # Start frame capture task
        self.task = asyncio.create_task(self.capture_frames())
        return await super().__aenter__()

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        if self.shutdown_event.is_set():
            self.logger.warning("Frame capture is already closed.")
        else:
            self.shutdown_event.set()
            self.logger.info("Shutting down Frame...")
            await asyncio.sleep(0.1)
        if self.gopro is not None:
            await self.gopro.close()
            self.gopro = None
            self.logger.info("Shutting down GoPro connection...")
        return await super().__aexit__(_, __, ___)

    async def event_handler(self, _: Message) -> AsyncIterator[tuple[str, Image]]:
        if self.latest_frame is not None:
            self.logger.debug("Sending frame...")
            yield self.output_channels[0], Image(image=self.latest_frame)
        else:
            return


async def _main() -> None:
    import os

    if "REDIS_URL" in os.environ:
        redis_url = os.environ["REDIS_URL"]
        node = GoProNode(
            output_channel="gopro/image",
            redis_url=redis_url,
        )
    else:
        node = GoProNode(output_channel="gopro/image")
    async with node:
        await node.event_loop()


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    asyncio.run(_main())
