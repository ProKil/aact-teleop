from functools import singledispatchmethod

class Joint:
    status: dict[str, float]

    def set_velocity(self, velocity: float, omega: float) -> None: ...
    @singledispatchmethod
    def move_to(
        self,
    ) -> None: ...
    @move_to.register
    def _(self, name: str, position: float) -> None: ...
    @move_to.register
    def _(
        self, position: float, v_m: float | None = None, a_m: float | None = None
    ) -> None: ...

class Robot:
    base: Joint
    arm: Joint
    lift: Joint
    end_of_arm: Joint
    head: Joint

    def startup(self) -> None: ...
    def stop(self) -> None: ...
    def push_command(self) -> None: ...
