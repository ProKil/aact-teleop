import os
from typing import AsyncIterator
import dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from aact.nodes import Node
from aact.messages import Message, Image, Zero

from aiostream import stream
import numpy as np
import cv2


class VideoStreamingNode(Node[Image, Zero]):
    def __init__(
        self,
        input_channel: str = "wrist_cam",
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        super().__init__(
            input_channel_types=[(input_channel, Image)],
            output_channel_types=[],
            redis_url=redis_url,
        )

    async def event_handler(
        self, _: str, __: Message[Image]
    ) -> AsyncIterator[tuple[str, Message[Zero]]]:
        raise NotImplementedError("VideoStreamingNode does not have an event handler.")
        yield "", Message()


app = FastAPI()


@app.get("/video_feed")
async def video_feed(input_channel: str) -> StreamingResponse:
    async def generate() -> AsyncIterator[bytes]:
        async with VideoStreamingNode(
            input_channel=input_channel, redis_url=os.environ["REDIS_URL"]
        ) as node:
            async for _, message in node._wait_for_input():
                frame = message.data.image
                assert isinstance(frame, bytes)
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
                    )
                else:
                    pass

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


def combine_frame(image1: bytes | None, image2: bytes | None) -> bytes:
    # Decode byte streams to images using OpenCV
    if image1 is not None and image2 is not None:
        np_img1 = np.frombuffer(image1, np.uint8)
        np_img2 = np.frombuffer(image2, np.uint8)
        img1 = cv2.imdecode(np_img1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np_img2, cv2.IMREAD_COLOR)

        # Resize img2 to 0.5 times its original size
        img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

        # Ensure both images have the same height
        if img1.shape[0] != img2.shape[0]:
            # Pad img2 with white space horizontally to match width of img1
            pad_height = img1.shape[0] - img2.shape[0]
            img2 = cv2.copyMakeBorder(
                img2,
                int(pad_height / 2),
                int(pad_height / 2),
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        # Combine images horizontally
        combined_image = cv2.hconcat([img1, img2])

        # Encode the combined image to bytes
        _, combined_image_bytes = cv2.imencode(".jpg", combined_image)
        return combined_image_bytes.tobytes()
    elif image1 is not None and image2 is None:
        return image1
    elif image1 is None and image2 is not None:
        return image2
    else:
        raise ValueError("Both image1 and image2 are None")


@app.get("/combined_feed")
async def video_feed_2(input_channel_1: str, input_channel_2: str) -> StreamingResponse:
    async def generate() -> AsyncIterator[bytes]:
        image1: bytes | None = None
        image2: bytes | None = None
        async with VideoStreamingNode(
            input_channel=input_channel_1, redis_url=os.environ["REDIS_URL"]
        ) as node1:
            async with VideoStreamingNode(
                input_channel=input_channel_2, redis_url=os.environ["REDIS_URL"]
            ) as node2:
                async for input_channel, message in stream.merge(
                    node1._wait_for_input(), node2._wait_for_input()
                ):
                    if input_channel == input_channel_1:
                        image1 = message.data.image
                        assert isinstance(image1, bytes)
                    elif input_channel == input_channel_2:
                        image2 = message.data.image
                        assert isinstance(image2, bytes)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + combine_frame(image1, image2)
                        + b"\r\n\r\n"
                    )

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


def main() -> None:
    dotenv.load_dotenv()

    if "SSLKEYFILE" in os.environ and "SSLCERTFILE" in os.environ:
        uvicorn.run(
            "teleop.video_streaming_client:app",
            host="0.0.0.0",
            port=8443,
            ssl_keyfile=os.environ["SSLKEYFILE"],
            ssl_certfile=os.environ["SSLCERTFILE"],
            reload=False,
        )
    else:
        uvicorn.run("teleop.video_streaming_client:app", host="0.0.0.0", port=8003)


if __name__ == "__main__":
    main()
