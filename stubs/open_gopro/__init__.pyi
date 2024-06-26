from enum import Enum
from typing import Any

class WiredGoPro:
    http_command: Any
    async def open(self) -> None: ...
    async def close(self) -> None: ...

class Params:
    class Toggle(Enum):
        ENABLE = 1
        DISABLE = 0
