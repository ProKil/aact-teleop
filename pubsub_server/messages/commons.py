from .base import Message


class Tick(Message):
    tick: int


class Image(Message):
    image: bytes
