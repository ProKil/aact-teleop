import asyncio
from typing import Any, AsyncIterator, Self


from pubsub_server.messages import Message

from pubsub_server.messages import Text, Image, Tick
from pubsub_server.nodes.registry import NodeFactory
from pubsub_server.nodes.base import Node

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


@NodeFactory.register("llm")
class LLMNode(Node[Text | Image | Tick, Text]):
    def __init__(
        self,
        text_input_channel: str,
        tick_input_channel: str,
        response_output_channel: str,
        action_output_channel: str,
        api_key: str,
        redis_url: str,
    ):
        super().__init__(
            input_channel_types=[
                (text_input_channel, Text),
                (tick_input_channel, Tick),
            ],
            output_channel_types=[
                (response_output_channel, Text),
                (action_output_channel, Text),
            ],
            redis_url=redis_url,
        )
        self.text_input_channel = text_input_channel
        self.tick_input_channel = tick_input_channel
        self.response_output_channel = response_output_channel
        self.action_output_channel = action_output_channel

        self.input_buffer: list[Text | Image] = []
        self.queue: asyncio.Queue[tuple[Text | Image, ...]] = asyncio.Queue()
        self.shutdown: asyncio.Event = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,  # type: ignore
        )
        self.memory: str = ""
        self.memory_buffer: list[tuple[str, str]] = []

    async def submit_to_llm(self, input_to_submit: tuple[Text | Image, ...]) -> None:
        if len(input_to_submit) > 0:
            messages: list[str | dict[Any, Any]] = []
            input_text = " ".join(
                [
                    input_item.text
                    for input_item in input_to_submit
                    if isinstance(input_item, Text)
                ]
            )
            messages.append(
                {
                    "type": "text",
                    "text": f"Here is the history of the conversation between you and the user: {self.memory_buffer}"
                    f"The user just said: {input_text}",
                }
            )
            response = await self.model.ainvoke(
                [
                    HumanMessage(content=messages),
                ]
            )
            await self.r.publish(
                self.response_output_channel,
                Message[Text](data=Text(text=response.content)).model_dump_json(),
            )
            self.memory_buffer.append(("human", input_text))
            self.memory_buffer.append(("assitant", str(response.content)))

    async def __aenter__(self) -> Self:
        return await super().__aenter__()

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        self.shutdown.set()
        return await super().__aexit__(_, __, ___)

    async def event_handler(
        self, input_channel: str, input_message: Message[Text | Image | Tick]
    ) -> AsyncIterator[tuple[str, Message[Text]]]:
        match input_channel:
            case self.text_input_channel:
                assert isinstance(input_message.data, Text)
                self.input_buffer.append(input_message.data)
            case self.tick_input_channel:
                if len(self.input_buffer) > 0:
                    self.task = asyncio.create_task(
                        self.submit_to_llm(tuple(self.input_buffer))
                    )
                    self.input_buffer = []
            case _:
                raise ValueError(f"Unexpected input channel: {input_channel}")
                yield "", Message[Text](data=Text(text=""))
