from pydantic import BaseModel

from typing import Generic, TypeVar

T = TypeVar("T")


class Message(BaseModel, Generic[T]):
    data: T
