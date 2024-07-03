import asyncio
import logging
import os
import signal
from typing import Any, TypeVar
from ..app import app
import typer

from pydantic import BaseModel, ConfigDict, Field
from pubsub_server import NodeFactory

from multiprocessing import Pool
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
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(_run_node(node_config, redis_url))
    except asyncio.CancelledError:
        logger = logging.getLogger(__name__)
        logger.info(f"Node {node_config.node_name} shutdown gracefully.")
    finally:
        loop.close()


@app.command()
def run_node(
    dataflow_toml: str = typer.Option(),
    node_name: str = typer.Option(),
    redis_url: str = typer.Option(),
) -> None:
    logger = logging.getLogger(__name__)
    config = Config.model_validate(toml.load(dataflow_toml))
    logger.info(f"Starting dataflow with config {config}")
    # dynamically import extra modules
    for module in config.extra_modules:
        __import__(module)

    for nodes in config.nodes:
        if nodes.node_name == node_name:
            _sync_run_node(nodes, redis_url)
            break


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

    # log_to_stderr(logging.DEBUG)

    subprocesses: list[Popen[bytes]] = []

    try:
        # Nodes that run w/ subprocess
        for node in config.nodes:
            if node.run_in_subprocess:
                command = f"pubsub run-node --dataflow-toml {dataflow_toml} --node-name {node.node_name} --redis-url {config.redis_url}"
                logger.info(f"executing {command}")
                node_process = Popen(
                    [command],
                    shell=True,
                    preexec_fn=os.setsid,  # Start the subprocess in a new process group
                )
                subprocesses.append(node_process)

        def _cleanup_subprocesses(
            signum: int | None = None, frame: Any | None = None
        ) -> None:
            for node_process in subprocesses:
                os.killpg(os.getpgid(node_process.pid), signal.SIGTERM)

        signal.signal(signal.SIGTERM, _cleanup_subprocesses)
        signal.signal(signal.SIGINT, _cleanup_subprocesses)

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

        # In case there is no nodes that are run w/ multiprocessing, wait for the subprocesses
        for node_process in subprocesses:
            node_process.wait()

    except Exception as e:
        logger.warning("Error in multiprocessing: ", e)
        _cleanup_subprocesses()
    finally:
        _cleanup_subprocesses()
