import asyncio
import logging
import os
import signal
from typing import Any, TypeVar
from ..app import app
import typer

from pydantic import BaseModel, ConfigDict, Field
from pubsub_server import NodeFactory

from multiprocessing import Pool, log_to_stderr
from subprocess import Popen

import toml

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class NodeArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class NodeConfig(BaseModel):
    node_name: str
    run_in_subprocess: bool = Field(default=False)
    node_args: NodeArgs = Field(default_factory=NodeArgs)


class Config(BaseModel):
    redis_url: str = Field(
        default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    extra_modules: list[str] = Field(default_factory=lambda: list())
    nodes: list[NodeConfig]


async def _run_node(node_config: NodeConfig, redis_url: str) -> None:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting node {node_config}")
    try:
        async with NodeFactory.make(
            node_config.node_name,
            **node_config.node_args.model_dump(),
            redis_url=redis_url,
        ) as node:
            logger.info(f"Starting eventloop {node_config.node_name}")
            await node.event_loop()
    except Exception as e:
        logger.error(f"Error in node {node_config.node_name}: {e}")


def _sync_run_node(node_config: NodeConfig, redis_url: str) -> None:
    asyncio.run(_run_node(node_config, redis_url))


def _dict_without_key(d: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != key}


@app.command()
def run_dataflow(
    dataflow_toml: str = typer.Option(help="Configuration dataflow toml file"),
) -> None:
    logger = logging.getLogger(__name__)
    config = Config.model_validate(toml.load(dataflow_toml))
    logger.info(f"Starting dataflow with config {config}")
    # dynamically import extra modules
    for module in config.extra_modules:
        __import__(module)

    log_to_stderr(logging.DEBUG)

    subprocesses: list[Popen[bytes]] = []

    try:
        # Nodes that run w/ subprocess
        for node in config.nodes:
            if node.run_in_subprocess:
                command = f"pubsub run-node --node-config-json {repr(node.model_dump_json())} --redis-url {config.redis_url}"
                logger.info(f"executing {command}")
                node_process = Popen(
                    [command],
                    shell=True,
                    preexec_fn=os.setsid,  # Start the subprocess in a new process group
                )
                subprocesses.append(node_process)
        # Nodes that run w/ multiprocessing
        with Pool(processes=len(config.nodes)) as pool:
            pool.starmap_async(
                _sync_run_node,
                [
                    (node, config.redis_url)
                    for node in config.nodes
                    if not node.run_in_subprocess
                ],
            ).get()
    except Exception as e:
        logger.warning("Error in multiprocessing: ", e)
        for node_process in subprocesses:
            os.killpg(
                os.getpgid(node_process.pid), signal.SIGTERM
            )  # Kill the process group
    finally:
        # Ensure all subprocesses are terminated
        for node_process in subprocesses:
            os.killpg(os.getpgid(node_process.pid), signal.SIGTERM)
