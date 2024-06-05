from typing import Tuple, cast
from pydantic import BaseModel
import zmq
import json
import requests
import numpy as np
import dotenv
import os

from .data_classes import TargetPosition
from .utils import _normalize_angle
from websockets.sync.client import connect, ClientConnection


def get_2d_rotation_from_quaternion(q: Tuple[float, float, float, float]) -> float:
    x, y, z, a = q
    return np.arctan2(2 * x * z - 2 * a * y, (1 - 2 * y * y - 2 * z * z))  # type: ignore


def get_euler(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """
    Convert a quaternion into pitch, yaw, and roll.

    Args:
    q: A numpy array of shape (4,) representing a quaternion [w, x, y, z]

    Returns:
    A tuple (pitch, yaw, roll)
    """
    w, x, y, z = q

    # Compute pitch (x-axis rotation)
    # 2*atan2(sqrt((a + x)**2 + (-y - z)**2), sqrt((a - x)**2 + (y - z)**2)) - pi/2
    pitch = _normalize_angle(
        2
        * np.arctan2(
            np.sqrt((w + x) ** 2 + (-y - z) ** 2), np.sqrt((w - x) ** 2 + (y - z) ** 2)
        )
        - np.pi / 2
    )

    # Compute yaw (y-axis rotation)
    # -atan2(-y - z, a + x) + atan2(y - z, a - x)
    yaw = _normalize_angle(-np.arctan2(-y - z, w + x) + np.arctan2(y - z, w - x))

    # Compute roll (z-axis rotation)
    # -atan2(-y - z, a + x) - atan2(y - z, a - x)
    roll = _normalize_angle(-np.arctan2(-y - z, w + x) - np.arctan2(y - z, w - x))

    return pitch, yaw, roll


def quaternion_multiply(
    q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    return (
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    )


class HeadsetControllerStates(BaseModel):
    headset_position: Tuple[float, float, float]
    headset_rotation: Tuple[float, float, float, float]
    controller_position: Tuple[float, float, float]
    controller_rotation: Tuple[float, float, float, float]
    controller_trigger: float
    reset_button: bool


class Client(object):
    # Connections
    quest_ip: str
    stretch_ip: str
    quest_socket: zmq.Socket[bytes]
    stretch_socket: ClientConnection | None

    # Memory
    old_head_position: Tuple[float, float] | None
    desired_base_position: Tuple[float, float] | None
    base_rotation_offset: float | None
    quaternion_offset: Tuple[float, float, float, float] | None

    def __init__(self, dry_run: bool = False) -> None:
        self.quest_ip = os.environ["QUEST_IP"]
        self.stretch_ip = os.environ["STRETCH_IP"] if not dry_run else ""
        self.quest_socket = self._connect_quest()
        self.stretch_socket = self._connect_stretch() if not dry_run else None

        self.old_head_position = None
        self.desired_base_position = None
        self.base_rotation_offset = None
        self.quaternion_offset = None

    def _connect_quest(self) -> zmq.Socket[bytes]:
        context = zmq.Context()
        quest_socket = context.socket(zmq.PULL)
        quest_socket.setsockopt(zmq.CONFLATE, 1)
        quest_socket.connect(f"tcp://{self.quest_ip}:12345")

        return quest_socket

    def _connect_stretch(self) -> ClientConnection:
        websocket_url = f"ws://{self.stretch_ip}:8000/move_to_ws"
        return cast(ClientConnection, connect(websocket_url).__enter__())  # mypy bug

    def __del__(self) -> None:
        if self.stretch_socket is not None:
            self.stretch_socket.send(TargetPosition().model_dump_json())
            self.stretch_socket.__exit__(None, None, None)

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

        # controller_states.controller_rotation = quaternion_multiply(
        #     (
        #         self.quaternion_offset[3],
        #         -self.quaternion_offset[0],
        #         -self.quaternion_offset[1],
        #         -self.quaternion_offset[2],
        #     ),
        #     (
        #         controller_states.controller_rotation[3],
        #         controller_states.controller_rotation[0],
        #         controller_states.controller_rotation[1],
        #         controller_states.controller_rotation[2],
        #     )
        # )
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
        if self.old_head_position is None:
            self.old_head_position = head_position
            desired_base_movement = 0.0
        else:
            head_vertical_movement = (
                head_position[0] - self.old_head_position[0],
                head_position[1] - self.old_head_position[1],
            )
            desired_base_movement = head_vertical_movement[0] * float(
                np.cos(desired_base_rotation_in_unity_space)
            ) + head_vertical_movement[1] * float(
                np.sin(desired_base_rotation_in_unity_space)
            )

        # 4.a handle reset and calibration
        if controller_states.reset_button or self.base_rotation_offset is None:
            if self.stretch_ip:
                current_base_status = requests.request(
                    "GET",
                    f"http://{self.stretch_ip}:8000/get_base_status",
                ).json()
                self.base_rotation_offset = (
                    desired_base_rotation_in_unity_space - current_base_status["theta"]
                )
            else:
                self.base_rotation_offset = 0.0

        desired_base_rotation_in_stretch_space = (
            desired_base_rotation_in_unity_space - self.base_rotation_offset
        )

        if self.desired_base_position is None:
            self.desired_base_position = (0.0, 0.0)

        self.desired_base_position = (
            self.desired_base_position[0]
            + desired_base_movement * np.cos(desired_base_rotation_in_stretch_space),
            self.desired_base_position[1]
            + desired_base_movement * np.sin(desired_base_rotation_in_stretch_space),
        )

        # 5. Calculate the desired arm length projected to the floor
        arm_length_projected_to_floor = np.linalg.norm(
            [
                controller_position[0] - head_position[0],
                controller_position[1] - head_position[1],
            ]
        )

        # 6. We need to calculate the desired arm lift.
        arm_lift = controller_states.controller_position[1]

        # 7. We need to calculate the desired wrist pitch, yaw, and roll.
        wrist_pitch = -wrist_pitch
        wrist_yaw = _normalize_angle(-wrist_yaw - desired_base_rotation_in_unity_space)
        wrist_roll = -wrist_roll

        # 8. We need to calculate the desired gripper status.
        grip_status = 95 - float(controller_states.controller_trigger) * 190
        grip_status = np.exp(0.02764 * (grip_status + 95)) - 90

        return TargetPosition(
            # x=self.desired_base_position[0],
            # y=self.desired_base_position[1],
            theta=desired_base_rotation_in_stretch_space,
            lift=arm_lift,
            arm=arm_length_projected_to_floor,
            grip_status=grip_status,
            wrist_pitch=wrist_pitch,
            wrist_yaw=wrist_yaw,
            wrist_roll=wrist_yaw,
            stretch_gripper=grip_status,
        )

    def event_loop(self) -> None:
        # start control loop

        _ = requests.request(
            "POST", f"http://{self.stretch_ip}:8000/start_control_loop"
        ).json()
        while True:
            message = self.quest_socket.recv()
            try:
                data = json.loads(message.decode())
            except json.JSONDecodeError:
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
            )

            if self.stretch_socket is not None:
                self.stretch_socket.send(
                    self._convert_controller_states_to_control_signals(
                        controller_states
                    ).model_dump_json()
                )


if __name__ == "__main__":
    dotenv.load_dotenv()

    client = Client()
    client.event_loop()

    del client
