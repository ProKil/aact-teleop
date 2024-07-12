import asyncio
import time

import cv2
from pubsub_server import NodeFactory
from teleop.webcam_node import WebcamNode

import numpy as np

import pyrealsense2 as rs  # type: ignore


@NodeFactory.register("realsense_wrist")
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

        self.logger.debug("Starting wrist video feed.")

        # get the serial number of the wrist camera
        camera_info = [
            {
                "name": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
            }
            for device in rs.context().devices
        ]
        d405_info = None

        for info in camera_info:
            if info["name"].endswith("D405"):
                d405_info = info
        if d405_info is None:
            self.logger.debug("Error: wrist camera not found")
            return

        # serial number first input
        # pipeline = setup_realsense_camera(d405_info['serial_number'], (1280, 720), (1280, 720), 30)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(d405_info["serial_number"])

        # 1280 x 720, 5 fps
        # 848 x 480, 10 fps
        # 640 x 480, 30 fps

        width, height, fps = 640, 480, 15
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = pipeline.start(config)

        # self.logger.debug("wrist cam pipeline set up successfully")

        while not self.shutdown_event.is_set():
            start_time = time.time()
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            ret, buffer = cv2.imencode(".jpg", color_image)
            self.latest_frame = buffer.tobytes()
            end_time = time.time()
            await asyncio.sleep(end_time - start_time - 1 / 30)
