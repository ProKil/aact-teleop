import asyncio
import os
from typing import Any, TypeVar
from ..app import app
import typer

from pydantic import BaseModel, ConfigDict, Field
from pubsub_server import NodeFactory, Message

from multiprocessing import Pool

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
    async with NodeFactory.make(
        node_config.node_name,
        **_dict_without_key(dict(node_config.model_config), "extra"),
        redis_url=redis_url,
    ) as node:
        await node.event_loop()


def _sync_run_node(node_config: NodeConfig, redis_url: str) -> None:
    asyncio.run(_run_node(node_config, redis_url))


def _dict_without_key(d: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != key}


@app.command()
def launch(
    dataflow_toml: str = typer.Option(help="Configuration dataflow toml file"),
) -> None:
    config = Config.model_validate(toml.load(dataflow_toml))

    with Pool(processes=len(config.nodes)) as pool:
        pool.starmap_async(
            _sync_run_node, [(node, config.redis_url) for node in config.nodes]
        ).get()
