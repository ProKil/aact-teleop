from pydantic import BaseModel, Field, AfterValidator
from typing import Annotated
from .utils import _normalize_angle

NormalizedAngle = Annotated[float, AfterValidator(lambda v: _normalize_angle(v))]


class TargetPosition(BaseModel):
    x: float = Field(default=0, description="x position.")
    y: float = Field(default=0, description="y position.")
    theta: NormalizedAngle = Field(default=0, description="orientation.")

    arm: Annotated[float, AfterValidator(lambda x: min(max(x, 0), 0.5))] = Field(
        default=0.25, description="The arm position."
    )
    lift: Annotated[float, AfterValidator(lambda x: min(max(x, 0.1), 1.09))] = Field(
        default=0.6, description="The lift position."
    )

    wrist_yaw: Annotated[
        float,
        AfterValidator(lambda x: min(max(x, -1.39), 3.14)),
    ] = Field(default=0, description="The wrist yaw position.")
    wrist_pitch: Annotated[
        float,
        AfterValidator(lambda x: min(max(x, -1.57), 0.57)),
    ] = Field(default=0, description="The wrist pitch position.")
    wrist_roll: NormalizedAngle = Field(
        default=0, description="The wrist roll position."
    )
    stretch_gripper: Annotated[
        float, AfterValidator(lambda x: min(max(x, -90), 90))
    ] = Field(default=0, description="The stretch gripper position.")
