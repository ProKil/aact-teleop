import asyncio
import base64
from typing import Any, AsyncIterator, Self


from pubsub_server.messages import Message

from pubsub_server.messages import Text, Image, Tick
from pubsub_server.nodes.registry import NodeFactory
from pubsub_server.nodes.base import Node

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field


class RobotAction(BaseModel):
    action: str = Field(..., description="The verb of the action")
    adjective: str = Field(
        ...,
        description="The adjective that describes the object's color/material/shape/size.",
    )
    object: str = Field(
        ...,
        description="The object that the action is performed on. A simple noun, e.g. cup, door, drawer, block.",
    )
    location: str = Field(
        ...,
        description="A preposition phrase. E.g. on the table, in the drawer, under the bed.",
    )


class LLMResponse(BaseModel):
    observation_summary: str = Field(
        ..., description="Please summarize what you are seeing right now."
    )
    response_to_human: str = Field(
        ..., description="Please respond to the human's message."
    )
    action: RobotAction
    status_summary: str = Field(
        ...,
        description="Please summarize what task are you performing, but the humans' request is."
        "How much progress has been made towards your task and humans' task"
        "This will be used as a memory for the next step.",
    )


@NodeFactory.register("custom_llm")
class CustomLLMNode(Node[Text | Image | Tick, Text]):
    def __init__(
        self,
        text_input_channel: str,
        image_input_channel: str,
        tick_input_channel: str,
        response_output_channel: str,
        action_output_channel: str,
        api_key: str,
        redis_url: str,
    ):
        super().__init__(
            input_channel_types=[
                (text_input_channel, Text),
                (image_input_channel, Image),
                (tick_input_channel, Tick),
            ],
            output_channel_types=[
                (response_output_channel, Text),
                (action_output_channel, Text),
            ],
            redis_url=redis_url,
        )
        self.text_input_channel = text_input_channel
        self.image_input_channel = image_input_channel
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
        self.memory_buffer: list[tuple[str, str]] = []

    async def submit_to_llm(self, input_to_submit: tuple[Text | Image, ...]) -> None:
        if len(input_to_submit) > 0:
            messages: list[str | dict[Any, Any]] = []
            human_text = " ".join(
                [
                    input_item.text
                    for input_item in input_to_submit
                    if isinstance(input_item, Text)
                ]
            )
            messages.append(
                {
                    "type": "text",
                    "text": "What human says: " + human_text,
                }
            )
            image_count = 0
            for input_item in input_to_submit:
                match input_item:
                    case Image(image=image):
                        messages.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(image).decode('utf-8')}"
                                },
                            }
                        )
                        image_count += 1
            if image_count > 1:
                parser = PydanticOutputParser(pydantic_object=LLMResponse)
                chain = self.model | parser
                response = await chain.ainvoke(
                    [
                        SystemMessage(
                            content=f"You are currently inside a robot body. The images that you see are from the robot's camera. You are talking to a human. Your previous memory is {self.memory_buffer}."
                        ),
                        HumanMessage(content=messages),
                        HumanMessage(
                            content=f"Format instruction: {parser.get_format_instructions()}"
                        ),
                    ]
                )
                assert isinstance(response, LLMResponse)
                await self.r.publish(
                    self.response_output_channel,
                    Message[Text](
                        data=Text(text=response.response_to_human)
                    ).model_dump_json(),
                )
                await self.r.publish(
                    self.action_output_channel,
                    Message[Text](
                        data=Text(text=response.action.model_dump_json())
                    ).model_dump_json(),
                )
                self.memory_buffer += [
                    ("human", human_text),
                    ("observation", response.observation_summary),
                    ("robot", response.response_to_human),
                    ("action", response.action.model_dump_json()),
                    ("status", response.status_summary),
                ]

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
            case self.image_input_channel:
                assert isinstance(input_message.data, Image)
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
