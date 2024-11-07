import json
from logging import getLogger
import os
import time
from typing import Any, AsyncIterator, Self, cast, TextIO
from datetime import datetime

import numpy as np
import zmq
from aact import Node, NodeFactory, Message
from teleop.client import HeadsetControllerStates, get_euler
from teleop.utils import _normalize_angle
from .data_classes import TargetPosition

from zmq.asyncio import Context, Socket


@NodeFactory.register("quest_controller")
class QuestControllerNode(Node[TargetPosition, TargetPosition]):
    def __init__(
        self,
        input_channel: str,
        output_channel: str,
        quest_controller_ip: str = os.environ.get("QUEST_IP", ""),
        translation_speed: float = 0.5,
        gripper_length: float = 0.26,
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channel_types=[
                (input_channel, TargetPosition),
            ],
            output_channel_types=[
                (output_channel, TargetPosition),
            ],
            redis_url=redis_url,
        )
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.translation_speed = translation_speed
        self.gripper_length = gripper_length
        self.base_rotation_offset: float | None = None
        self.lift_offset: float = 0.0
        self.quaternion_offset: tuple[float, float, float, float] | None = None
        self.logger = getLogger(__name__)
        self.quest_controller_ip = quest_controller_ip
        self.delta_time = 0.0
        self.current_status: TargetPosition | None = None
        self.run_name = "Default_run"
        self.recording_file: TextIO | None = None

    def _connect_quest(self) -> Socket:
        context = Context()
        quest_socket = context.socket(zmq.PULL)
        # quest_socket.setsockopt(zmq.CONFLATE, 1)
        quest_socket.connect(f"tcp://{self.quest_controller_ip}:12345")

        return quest_socket

    def _convert_controller_states_to_control_signals(
        self, controller_states: HeadsetControllerStates
    ) -> TargetPosition:
        """
        Convert the controller states to control signals for the robot.

        There are several conversions that need to be done:

        1. The controller states are in the Unity left-hand axes: Z-forward, X-right, Y-up. While stretch uses
        right-hand axes, X-forward, Y-left, Z-up. We need to convert the axes.
        2. The controller rotation is in quaternion form. We need to convert it to Euler angles.
        3. We need to calculate the desired base rotation, so that the arm is pointing right.
        4. We need to calculate the desired forward/backward movement based on the projected movement of the headset
        onto the base rotation.
        5. We need to calculate the desired arm length projected to the floor.
        6. We need to calculate the desired arm lift.
        7. We need to calculate the desired wrist pitch, yaw, and roll.
        8. We need to calculate the desired gripper status.
        9. We need to consider the angle of the wrist.
        10. We need to get head tilt (pitch) and head pan (yaw).
        """

        # 1. Convert the axes
        controller_position = (
            controller_states.controller_position[2],
            -controller_states.controller_position[0],
        )
        head_position = (
            controller_states.headset_position[2],
            -controller_states.headset_position[0],
        )

        # 2. Convert the controller rotation to Euler angles
        # if controller_states.reset_button or self.quaternion_offset is None:
        #     self.quaternion_offset = controller_states.controller_rotation

        wrist_pitch, wrist_yaw, wrist_roll = get_euler(
            (
                controller_states.controller_rotation[3],
                controller_states.controller_rotation[0],
                controller_states.controller_rotation[1],
                controller_states.controller_rotation[2],
            )
        )

        # 3. Calculate the desired base rotation
        # This is the desired base rotation in the Unity space
        # We will calibrate it into the Stretch space later
        desired_base_rotation_in_unity_space = np.arctan2(
            controller_position[0] - head_position[0],
            -controller_position[1] + head_position[1],
        )

        # 4. Calculate the desired forward/backward movement
        desired_base_movement = (
            -controller_states.controller_thumbstick[0]
            * self.translation_speed
            * self.delta_time
        )

        # 4.a handle reset and calibration
        if controller_states.reset_button or self.base_rotation_offset is None:
            if self.current_status is not None:
                self.base_rotation_offset = (
                    desired_base_rotation_in_unity_space - self.current_status.theta
                )
                self.lift_offset = (
                    controller_states.controller_position[1] - self.current_status.lift
                )
            else:
                self.base_rotation_offset = 0.0
                self.lift_offset = 0.0

        desired_base_rotation_in_stretch_space = _normalize_angle(
            desired_base_rotation_in_unity_space - self.base_rotation_offset
        )

        # 5. Calculate the desired arm length projected to the floor
        arm_length_projected_to_floor = np.linalg.norm(
            [
                controller_position[0] - head_position[0],
                controller_position[1] - head_position[1],
            ]
        )

        # 6. We need to calculate the desired arm lift.
        arm_lift = controller_states.controller_position[1] - self.lift_offset

        # 7. We need to calculate the desired wrist pitch, yaw, and roll.
        wrist_pitch = -wrist_pitch
        wrist_yaw = _normalize_angle(
            -wrist_yaw - desired_base_rotation_in_unity_space + np.pi / 2
        )
        wrist_roll = -wrist_roll

        # 8. We need to calculate the desired gripper status.
        grip_status = 95 - float(controller_states.controller_trigger) * 190
        grip_status = np.exp(0.02764 * (grip_status + 95)) - 90

        # 9. Consider the angle of the wrist
        arm_length_gripper_subtracted = arm_length_projected_to_floor - 0.26 * np.cos(
            wrist_pitch
        ) * np.cos(wrist_yaw)
        arm_lift_gripper_subtracted = arm_lift - 0.26 * np.sin(wrist_pitch)

        # 10. Get head tilt (pitch) and head pan (yaw)
        head_tilt, head_pan, _ = get_euler(
            (
                controller_states.headset_rotation[3],
                controller_states.headset_rotation[0],
                controller_states.headset_rotation[1],
                controller_states.headset_rotation[2],
            )
        )

        head_tilt = -head_tilt
        head_pan = _normalize_angle(-head_pan - desired_base_rotation_in_unity_space)

        target_position = TargetPosition(
            translation_speed=desired_base_movement,
            theta=desired_base_rotation_in_stretch_space,
            lift=arm_lift_gripper_subtracted,
            arm=arm_length_gripper_subtracted,
            grip_status=grip_status,
            wrist_pitch=wrist_pitch,
            wrist_yaw=wrist_yaw,
            wrist_roll=wrist_roll,
            stretch_gripper=grip_status,
            head_tilt=head_tilt,
            head_pan=head_pan,
        )

        if controller_states.record_button:
            if self.recording_file is None:
                self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.logger.info(
                    "Recording started. File name: ", f"{self.run_name}.jsonl"
                )
                self.recording_file = open(f"{self.run_name}.jsonl", "w")
            self.recording_file.write(
                json.dumps(
                    dict(
                        controller_states=controller_states.model_dump(),
                        target_position=target_position.model_dump(),
                    )
                )
                + "\n"
            )
        else:
            if self.recording_file is not None:
                self.recording_file.close()
                self.recording_file = None
                self.logger.info("Recording stopped.")

        return target_position

    async def event_loop(self) -> None:
        # start control loop

        last_time = time.time()

        controller_states: HeadsetControllerStates = HeadsetControllerStates(
            headset_position=(0, 0, 0),
            headset_rotation=(0, 0, 0, 0),
            controller_position=(0, 0, 0),
            controller_rotation=(0, 0, 0, 0),
            controller_trigger=0,
            reset_button=False,
            record_button=False,
            safety_button=False,
            controller_thumbstick=(0, 0),
        )

        while True:
            message = await self.quest_socket.recv()
            now_time = time.time()
            self.delta_time = now_time - last_time
            last_time = now_time
            try:
                data = json.loads(message.decode())
            except json.JSONDecodeError:
                self.logger.info(f"The last controller states were {controller_states}")
                print("Terminating the connection...")
                return

            right_controller = cast(dict[str, str], data["RightController"])
            headset = cast(dict[str, str], data["Headset"])
            controller_states = HeadsetControllerStates(
                headset_position=tuple(
                    map(float, headset["HeadLocalPosition"].split(","))
                ),
                headset_rotation=tuple(
                    map(float, headset["HeadLocalRotation"].split(","))
                ),
                controller_position=tuple(
                    map(float, right_controller["RightLocalPosition"].split(","))
                ),
                controller_rotation=tuple(
                    map(float, right_controller["RightLocalRotation"].split(","))
                ),
                controller_trigger=float(right_controller["RightIndexTrigger"]),
                reset_button=bool(right_controller["RightB"]),
                record_button=bool(right_controller["RightA"]),
                safety_button=bool(right_controller["RightHandTrigger"]),
                controller_thumbstick=tuple(
                    map(float, right_controller["RightThumbstickAxes"].split(","))
                ),
            )

            if controller_states.controller_position == (
                0.0,
                0.0,
                0.0,
            ) and controller_states.controller_rotation == (0.0, 0.0, 0.0, 1.0):
                self.logger.warning("Abnormal controller states detected. Skipping...")
                continue

            # safety lock
            if not controller_states.safety_button:
                # self.logger.warning("Safety button not pressed. Skipping...")
                continue

            control_signals = self._convert_controller_states_to_control_signals(
                controller_states
            )

            await self.r.publish(
                self.output_channel,
                Message[TargetPosition](data=control_signals).model_dump_json(),
            )

    async def event_handler(
        self, input_channel: str, current_position: Message[TargetPosition]
    ) -> AsyncIterator[tuple[str, Message[TargetPosition]]]:
        if input_channel == self.input_channel:
            self.current_status = current_position.data
        else:
            yield (self.output_channel, current_position)

    async def __aenter__(self) -> Self:
        self.quest_socket = self._connect_quest()
        return await super().__aenter__()

    async def __aexit__(self, _: Any, __: Any, ___: Any) -> None:
        if self.recording_file is not None:
            self.recording_file.close()
        self.quest_socket.close()
        await super().__aexit__(_, __, ___)
