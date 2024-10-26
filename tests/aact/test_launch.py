import asyncio
import signal
import subprocess
from pathlib import Path
import pytest

import os

from redis.asyncio import Redis

from aact import Message
from aact.messages import Float


@pytest.mark.asyncio
async def test_launch() -> None:
    # Start the subprocess in a new process group
    s = subprocess.Popen(
        [f"pubsub run-dataflow {Path(__file__).parent / 'test.toml'}"],
        shell=True,
        preexec_fn=os.setsid,
    )

    r = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    pubsub = r.pubsub()
    await pubsub.subscribe("random/number/1")
    try:
        async with asyncio.timeout(1):
            async for message in pubsub.listen():
                if message["type"] == "message":
                    if (
                        Message[Float].model_validate_json(message["data"]).data.value
                        > 0.9
                    ):
                        break
    except asyncio.TimeoutError:
        # Kill the process group if the test times out
        os.killpg(os.getpgid(s.pid), signal.SIGTERM)
        await r.aclose()
        raise Exception("Did not receive the expected message")
    finally:
        # Ensure the process group is killed in the finally block
        os.killpg(os.getpgid(s.pid), signal.SIGTERM)
        await r.aclose()
