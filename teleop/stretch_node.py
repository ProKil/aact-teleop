import asyncio
from copy import deepcopy
import time
from typing import Any, AsyncIterator, Self

from stretch_body.hello_utils import ThreadServiceExit
from pubsub_server import Node, NodeFactory, Message
from pubsub_server.messages import Tick
from .data_classes import TargetPosition

from stretch_body.robot import Robot
from .server import (
    ControlPolicyArguments,
    PlanTrajectoryArguments,
    Position,
    handle_exit,
    pid_control_policy,
    plan_trajectory,
)


@NodeFactory.register("stretch")
class StretchNode(Node[TargetPosition | Tick, TargetPosition]):
    def __init__(
        self,
        input_channel: str,
        input_tick_channel: str,
        output_channel: str,
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channel_types=[
                (input_channel, TargetPosition),
                (input_tick_channel, Tick),
            ],
            output_channel_types=[
                (output_channel, TargetPosition),
            ],
            redis_url=redis_url,
        )
        self.input_tick_channel = input_tick_channel
        self.output_channel = output_channel
        self.robot: Robot = Robot()
        self.tasks: list[asyncio.Task[None]] = []
        self.target_position: TargetPosition = TargetPosition()
        self.current_position: TargetPosition = TargetPosition()

    async def control_loop(self, robot: Robot) -> None:
        start_time = time.time()
        current_time: float = 0

        while True:
            new_current_time = time.time() - start_time
            delta_t = new_current_time - current_time
            current_time = new_current_time

            current_x, current_y, current_theta = (
                robot.base.status["x"],
                robot.base.status["y"],
                robot.base.status["theta"],
            )

            # if np.allclose((current_x, current_y, current_theta), move_to_position):
            #     await asyncio.sleep(1)
            #     continue

            next_step_position = plan_trajectory(
                PlanTrajectoryArguments(
                    target_position=Position(
                        x=self.target_position.x,
                        y=self.target_position.y,
                        theta=self.target_position.theta,
                    ),
                    current_position=Position(
                        x=current_x, y=current_y, theta=current_theta
                    ),
                )
            )
            x_d, y_d, _theta_d = (
                next_step_position.x,
                next_step_position.y,
                next_step_position.theta,
            )
            # Get the control inputs
            v, omega = pid_control_policy(
                ControlPolicyArguments(
                    desired_position=Position(
                        x=x_d, y=y_d, theta=self.target_position.theta
                    ),
                    current_position=Position(
                        x=current_x, y=current_y, theta=current_theta
                    ),
                    dt=delta_t,
                )
            )

            try:
                robot.base.set_velocity(self.target_position.translation_speed, omega)
                robot.arm.move_to(self.target_position.arm, v_m=2, a_m=2)
                robot.lift.move_to(self.target_position.lift, v_m=2, a_m=2)
                robot.end_of_arm.move_to("wrist_yaw", self.target_position.wrist_yaw)
                robot.end_of_arm.move_to(
                    "wrist_pitch", self.target_position.wrist_pitch
                )
                robot.end_of_arm.move_to("wrist_roll", self.target_position.wrist_roll)
                robot.end_of_arm.move_to(
                    "stretch_gripper", self.target_position.stretch_gripper
                )
                robot.head.move_to("head_tilt", self.target_position.head_tilt)
                robot.head.move_to("head_pan", self.target_position.head_pan)
                self.current_position = deepcopy(self.target_position)
                self.current_position.x = current_x
                self.current_position.y = current_y
                self.current_position.theta = current_theta
                robot.push_command()
                await asyncio.sleep(1 / 80)
            except (ThreadServiceExit, KeyboardInterrupt):
                print("Exiting control loop.")
                handle_exit(0, 0)

    async def __aenter__(self) -> Self:
        self.robot.startup()
        self.tasks.append(asyncio.create_task(self.control_loop(self.robot)))
        return await super().__aenter__()

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        self.robot.stop()
        return await super().__aexit__(_, __, ___)

    async def event_handler(
        self, input_channel: str, input_message: Message[TargetPosition | Tick]
    ) -> AsyncIterator[tuple[str, Message[TargetPosition]]]:
        if input_channel == self.input_tick_channel:
            yield (
                self.output_channel,
                Message[TargetPosition](data=self.current_position),
            )
        else:
            assert isinstance(input_message.data, TargetPosition)
            self.target_position = input_message.data
