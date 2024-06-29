import asyncio
import signal
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import cv2  # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from open_gopro import WiredGoPro, Params
from open_gopro.constants import WebcamStatus, WebcamError

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

# Global variables
gopro: Optional[WiredGoPro] = None
shutdown_event: asyncio.Event = asyncio.Event()
latest_frame: Optional[npt.NDArray[np.uint8]] = None
frame_lock: asyncio.Lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global gopro

    try:
        # Startup
        logger.info("Initializing GoPro connection...")
        gopro = WiredGoPro()
        await gopro.open()
        logger.info("GoPro connection established.")

        await gopro.http_command.wired_usb_control(control=Params.Toggle.DISABLE)
        await gopro.http_command.set_shutter(shutter=Params.Toggle.DISABLE)

        # Start webcam
        current_status: WebcamStatus = (
            await gopro.http_command.webcam_status()
        ).data.status
        logger.debug(f"Current webcam status: {current_status}")
        if current_status not in {WebcamStatus.OFF, WebcamStatus.IDLE}:
            logger.info("Stopping existing webcam session...")
            await gopro.http_command.webcam_stop()

        logger.info("Starting webcam...")
        start_status: WebcamError = (await gopro.http_command.webcam_start()).data.error
        if start_status != WebcamError.SUCCESS:
            raise Exception(f"Couldn't start webcam: {start_status}")

        # Wait for webcam to be ready
        while True:
            status: WebcamStatus = (
                await gopro.http_command.webcam_status()
            ).data.status
            logger.debug(f"Waiting for webcam... Current status: {status}")
            if status == WebcamStatus.HIGH_POWER_PREVIEW:
                logger.info("Webcam is ready.")
                break
            await asyncio.sleep(0.5)

        # Start frame capture task
        app.state.capture_task = asyncio.create_task(capture_frames())

        yield
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    finally:
        # Shutdown
        await cleanup()


async def cleanup() -> None:
    global gopro
    if hasattr(app.state, "capture_task"):
        app.state.capture_task.cancel()
        try:
            await app.state.capture_task
        except asyncio.CancelledError:
            pass
    if gopro:
        logger.info("Shutting down GoPro connection...")
        try:
            await gopro.http_command.webcam_stop()
            await gopro.http_command.webcam_exit()
            await gopro.close()
            logger.info("GoPro connection closed successfully.")
        except Exception as e:
            logger.error(f"Error during GoPro shutdown: {e}")


def signal_handler(sig: int, frame: Optional[object]) -> None:
    logger.info("Ctrl+C detected. Initiating graceful shutdown...")
    shutdown_event.set()


app: FastAPI = FastAPI(lifespan=lifespan)


async def capture_frames() -> None:
    global latest_frame
    cap: cv2.VideoCapture = cv2.VideoCapture("udp://0.0.0.0:8554")
    try:
        while not shutdown_event.is_set():
            success: bool
            frame: npt.NDArray[np.uint8]
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame from camera.")
                await asyncio.sleep(0.1)
                continue
            async with frame_lock:
                latest_frame = frame
            await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"Error in capture_frames: {e}")
    finally:
        cap.release()


async def generate_frames() -> AsyncGenerator[bytes, None]:
    global latest_frame
    while not shutdown_event.is_set():
        async with frame_lock:
            if latest_frame is None:
                await asyncio.sleep(0.1)
                continue
            frame: npt.NDArray[np.uint8] = latest_frame.copy()

        ret: bool
        buffer: npt.NDArray[np.uint8]
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes: bytes = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        # Add a small delay to control frame rate and CPU usage
        await asyncio.sleep(
            0.03
        )  # Adjust this value to balance between smoothness and CPU usage


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "GoPro Streaming Server"}


@app.get("/stream")
async def video_feed() -> StreamingResponse:
    if not gopro:
        logger.error("GoPro is not initialized.")
        raise HTTPException(status_code=500, detail="GoPro is not initialized")
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


async def run_app(host: str, port: int) -> None:
    config: uvicorn.Config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server: uvicorn.Server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvicorn

    signal.signal(signal.SIGINT, signal_handler)

    host: str = "0.0.0.0"
    port: int = 8000

    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    try:
        logger.info(f"Starting server on {host}:{port}")
        loop.run_until_complete(run_app(host, port))
    except asyncio.CancelledError:
        logger.info("Server shutdown initiated.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        loop.run_until_complete(cleanup())
        loop.close()

    logger.info("Application has been shut down gracefully.")
