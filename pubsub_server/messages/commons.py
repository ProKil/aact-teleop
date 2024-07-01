from typing import Any, Annotated
from .base import Message
from pydantic import BaseModel, PlainValidator, PlainSerializer, WithJsonSchema


class Zero(BaseModel):
    pass


class Tick(BaseModel):
    tick: int


TickMessage = Message[Tick]


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


class Image(BaseModel):
    image: HexBytes


ImageMessage = Message[Image]
