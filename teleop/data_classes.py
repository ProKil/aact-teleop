from pydantic import Field, AfterValidator
from typing import Annotated

from pubsub_server.messages import DataModel, DataModelFactory
from .utils import _normalize_angle

NormalizedAngle = Annotated[float, AfterValidator(lambda v: _normalize_angle(v))]


@DataModelFactory.register("target_position")
class TargetPosition(DataModel):
    x: Annotated[float, AfterValidator(lambda x: min(max(x, -3), 3))] = Field(
        default=0, description="x position."
    )
    y: Annotated[float, AfterValidator(lambda x: min(max(x, -3), 3))] = Field(
        default=0, description="y position."
    )
    translation_speed: float = Field(default=0, description="translation speed.")
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
        float, AfterValidator(lambda x: min(max(x, -50), 90))
    ] = Field(default=0, description="The stretch gripper position.")
    head_tilt: Annotated[float, AfterValidator(lambda x: min(max(x, -1.57), 0.27))] = (
        Field(default=0, description="The head tilt position.")
    )
    head_pan: Annotated[float, AfterValidator(lambda x: min(max(x, -1.57), 1.57))] = (
        Field(default=0, description="The head pan position.")
    )
