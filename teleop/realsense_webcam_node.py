import asyncio
import time

import cv2
from pubsub_server import NodeFactory
from teleop.webcam_node import WebcamNode

import numpy as np

from stretch_body.hello_utils import setup_realsense_camera  # type: ignore


@NodeFactory.register("realsense")
class RealsenseWebcamNode(WebcamNode):
    def __init__(
        self,
        input_tick_channel: str,
        output_channel: str,
        webcam_id: str,
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(input_tick_channel, output_channel, webcam_id, redis_url)

    async def update_video_feed(self) -> None:
        """
        Use a different process to update the video feed.
        """

        self.logger.debug("Starting video feed.")

        # serial number first input
        pipeline = setup_realsense_camera(None, (1280, 720), (1280, 720), 30)

        while not self.shutdown_event.is_set():
            start_time = time.time()
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.rot90(np.asanyarray(color_frame.get_data()), -1)

            ret, buffer = cv2.imencode(".jpg", color_image)
            self.latest_frame = buffer.tobytes()
            end_time = time.time()
            await asyncio.sleep(end_time - start_time - 1 / 30)
