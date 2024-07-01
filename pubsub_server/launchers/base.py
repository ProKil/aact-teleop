from multiprocessing import Pool
from pubsub_server.nodes import Node
from pubsub_server.messages import Message


async def run_node(node: Node[Message, Message]) -> None:
    async with node:
        await node.event_loop()


class Launcher(object):
    nodes: list[Node[Message, Message]]

    def __init__(self, nodes: list[Node[Message, Message]]) -> None:
        self.nodes = nodes

    def run(self) -> None:
        with Pool(len(self.nodes)) as pool:
            pool.map(run_node, self.nodes)
