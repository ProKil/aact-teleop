import asyncio
from pathlib import Path
from datetime import datetime
from typing import Any, AsyncIterator, Self
from logging import getLogger, basicConfig, INFO

from aact.messages.commons import DataEntry

from aact.nodes.base import Node
from aact.nodes.registry import NodeFactory
from aact.messages import DataModel, Zero, Message
from aact.messages.registry import DataModelFactory

from aiofiles import open
from aiofiles.threadpool.text import AsyncTextIOWrapper
from aiofiles.base import AiofilesContextManager


@NodeFactory.register("stretch_record")
class RecordNode(Node[DataModel, Zero]):
    def __init__(
        self,
        record_channel_types: dict[str, str],
        recordings_folder_path: str = "./recordings/default_recordings",
        jsonl_file_name: str = "default_recording.jsonl",
        redis_url: str = "redis://localhost:6379/0",
        add_datetime: bool = True,
    ):
        input_channel_types: list[tuple[str, type[DataModel]]] = []
        for channel, channel_type_string in record_channel_types.items():
            input_channel_types.append(
                (channel, DataModelFactory.registry[channel_type_string])
            )

        super().__init__(
            input_channel_types=input_channel_types,
            output_channel_types=[],
            redis_url=redis_url,
        )
        self.jsonl_file_name = jsonl_file_name
        self.jsonl_file_path = ""
        self.num_recordings = 0
        self.add_datetime = add_datetime
        self.recordings_folder_path = recordings_folder_path
        self.logger = getLogger(__name__)
        basicConfig(level=INFO)

        # if add_datetime:
        #     # add a datetime to jsonl_file_path before the extension. The file can have any extension.
        #     jsonl_file_path = (
        #         jsonl_file_path[: jsonl_file_path.rfind(".")]
        #         + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        #         + jsonl_file_path[jsonl_file_path.rfind(".") :]
        #     )

        self.aioContextManager: AiofilesContextManager[AsyncTextIOWrapper] | None = None
        self.json_file: AsyncTextIOWrapper | None = None
        self.write_queue: asyncio.Queue[DataEntry[DataModel]] = asyncio.Queue()
        self.write_task: asyncio.Task[None] | None = None

        self.recording = False

    async def __aenter__(self) -> Self:
        # self.aioContextManager = open(self.jsonl_file_path, "w")
        # self.json_file = await self.aioContextManager.__aenter__()
        # self.write_task = asyncio.create_task(self.write_to_file())
        return await super().__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.aioContextManager:
            await self.aioContextManager.__aexit__(exc_type, exc_value, traceback)

        if self.json_file:
            del self.json_file

        return await super().__aexit__(exc_type, exc_value, traceback)

    async def write_to_file(self) -> None:
        while self.json_file:
            data_entry = await self.write_queue.get()
            await self.json_file.write(data_entry.model_dump_json() + "\n")
            await self.json_file.flush()
            self.write_queue.task_done()

    async def open_new_record(self) -> None:
        # close existing file and write task
        if self.aioContextManager:
            await self.aioContextManager.__aexit__(None, None, None)
        del self.json_file

        # rename the new file
        new_filename = ""
        if self.add_datetime:
            new_filename = (
                self.jsonl_file_name[: self.jsonl_file_name.rfind(".")]
                + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
                + self.jsonl_file_name[self.jsonl_file_name.rfind(".") :]
            )
        else:
            new_filename = f"{self.jsonl_file_name[:-6]}_{self.num_recordings}.jsonl"
        full_filepath = Path(self.recordings_folder_path) / new_filename
        Path.mkdir(full_filepath.parent, parents=True, exist_ok=True)

        # open new json file and write task
        self.logger.info("Writing to json file: %s", full_filepath)
        self.aioContextManager = open(full_filepath, "w")
        self.json_file = await self.aioContextManager.__aenter__()
        self.write_task = asyncio.create_task(self.write_to_file())

    async def event_handler(
        self, input_channel: str, input_message: Message[DataModel]
    ) -> AsyncIterator[tuple[str, Message[Zero]]]:
        from teleop.data_classes import TargetPosition

        if input_channel in self.input_channel_types:
            # Use a flag to see whether the record button is on
            # If yes, just record all the future information
            if input_channel == "quest_control":
                target_position = Message[TargetPosition](data=input_message.data)
                if target_position.data.record_button:
                    if not self.recording:
                        self.logger.info("====== Starting recording ======")
                        await self.open_new_record()
                        self.recording = True
                        self.num_recordings += 1

                elif target_position.data.stop_record_button:
                    if self.recording:
                        self.logger.info("====== Stopping recording ======")
                        self.recording = False

            if self.recording:
                await self.write_queue.put(
                    DataEntry[self.input_channel_types[input_channel]](  # type: ignore[name-defined]
                        channel=input_channel, data=input_message.data
                    )
                )
        else:
            yield input_channel, Message[Zero](data=Zero())
