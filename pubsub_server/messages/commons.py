from typing import Any, Annotated

from pubsub_server.messages.registry import DataModelFactory
from .base import DataModel
from pydantic import PlainValidator, PlainSerializer, WithJsonSchema


@DataModelFactory.register("zero")
class Zero(DataModel):
    pass


@DataModelFactory.register("tick")
class Tick(DataModel):
    tick: int


@DataModelFactory.register("float")
class Float(DataModel):
    value: float


def hex_bytes_validator(o: Any) -> bytes:
    if isinstance(o, bytes):
        return o
    elif isinstance(o, bytearray):
        return bytes(o)
    elif isinstance(o, str):
        return bytes.fromhex(o)
    raise ValueError(f"Expected bytes, bytearray, or hex string, got {type(o)}")


HexBytes = Annotated[
    bytes,
    PlainValidator(hex_bytes_validator),
    PlainSerializer(lambda b: b.hex()),
    WithJsonSchema({"type": "string"}),
]


@DataModelFactory.register("image")
class Image(DataModel):
    image: HexBytes
