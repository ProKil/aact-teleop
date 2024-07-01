import os
from typing import AsyncIterator
import dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from pubsub_server.nodes import Node
from pubsub_server.messages import Message, Image


class VideoStreamingNode(Node[Image, Message]):
    def __init__(
        self,
        input_channel: str = "wrist_cam",
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        super().__init__(
            input_channels=[input_channel],
            output_channels=[],
            input_type=Image,
            output_type=Message,
            redis_url=redis_url,
        )

    async def event_handler(self, _: Message) -> AsyncIterator[tuple[str, Message]]:
        raise NotImplementedError("VideoStreamingNode does not have an event handler.")
        yield "", Message()


app = FastAPI()


@app.get("/video_feed")
async def video_feed(input_channel: str) -> StreamingResponse:
    async def generate() -> AsyncIterator[bytes]:
        async with VideoStreamingNode(
            input_channel=input_channel, redis_url=os.environ["REDIS_URL"]
        ) as node:
            async for message in node._wait_for_input():
                frame = message.image
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
