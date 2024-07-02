import logging
import toml
import typer
from ..app import app
from .launch import Config, _sync_run_node


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
