import typer
from ..app import app
from .launch import NodeConfig, _sync_run_node


@app.command()
def run_node(
    node_config_json: str = typer.Option(), redis_url: str = typer.Option()
) -> None:
    node_config = NodeConfig.model_validate_json(node_config_json)
    _sync_run_node(node_config, redis_url)
