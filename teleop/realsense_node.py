import asyncio
import time

import cv2
from pubsub_server import NodeFactory
from teleop.webcam_node import WebcamNode

import numpy as np

from stretch_body.hello_utils import setup_realsense_camera  # type: ignore


@NodeFactory.register("realsense_cam")
class RealsenseWebcamNode(WebcamNode):
    def __init__(
        self,
        input_tick_channel: str,
        output_channel: str,
        webcam_id: str,
        redis_url: str = "redis://localhost:6379/0",
        serial_no: int | None = None,
        frame_width: int | None = 1280,
        frame_height: int | None = 720,
        frame_rate: int | None = 30,
    ):
        super().__init__(input_tick_channel, output_channel, webcam_id, redis_url)
        self.serial_no = serial_no
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_rate = frame_rate

    async def update_video_feed(self) -> None:
        """
        Use a different process to update the video feed.
        """

        self.logger.debug("Starting video feed.")

        # Recommended framerates
        # 1280 x 720, 5 fps
        # 848 x 480, 10 fps
        # 640 x 480, 30 fps

        # serial number first input
        pipeline = setup_realsense_camera(
            self.serial_no,
            (self.frame_width, self.frame_height),
            (self.frame_width, self.frame_height),
            self.frame_rate,
        )

        while not self.shutdown_event.is_set():
            start_time = time.time()
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            if self.serial_no is None:
                color_image = np.rot90(np.asanyarray(color_frame.get_data()), -1)
            else:
                color_image = np.asanyarray(color_frame.get_data())

            ret, buffer = cv2.imencode(".jpg", color_image)
            self.latest_frame = buffer.tobytes()
            end_time = time.time()
            await asyncio.sleep(end_time - start_time - 1 / 30)
