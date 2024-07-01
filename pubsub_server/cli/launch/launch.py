import asyncio
import logging
import os
from typing import Any, TypeVar
from ..app import app
import typer

from pydantic import BaseModel, ConfigDict, Field
from pubsub_server import NodeFactory, Message


from multiprocessing import Pool, log_to_stderr

import toml

InputType = TypeVar("InputType", bound=Message)
OutputType = TypeVar("OutputType", bound=Message)


class NodeConfig(BaseModel):
    node_name: str
    model_config = ConfigDict(extra="allow")


class Config(BaseModel):
    redis_url: str = Field(
        default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    nodes: list[NodeConfig]


async def _run_node(node_config: NodeConfig, redis_url: str) -> None:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting node {node_config}")
    try:
        async with NodeFactory.make(
            node_config.node_name,
            **_dict_without_key(dict(node_config), "node_name"),
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
def launch(
    dataflow_toml: str = typer.Option(help="Configuration dataflow toml file"),
) -> None:
    config = Config.model_validate(toml.load(dataflow_toml))
    log_to_stderr(logging.DEBUG)

    print(config)

    with Pool(processes=len(config.nodes)) as pool:
        pool.starmap_async(
            _sync_run_node, [(node, config.redis_url) for node in config.nodes]
        ).get()
