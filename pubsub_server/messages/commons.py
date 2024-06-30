from typing import Any, Annotated
from .base import Message
from pydantic import PlainValidator, PlainSerializer, WithJsonSchema


class Tick(Message):
    tick: int


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


class Image(Message):
    image: HexBytes
