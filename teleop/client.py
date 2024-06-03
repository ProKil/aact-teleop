import time
from typing import Tuple
import zmq
import json
import websocket
import requests
import numpy as np
import dotenv
import os

from .data_classes import TargetPosition
from .utils import _normalize_angle
from websockets.sync.client import connect


def get_2d_rotation_from_quaternion(q: Tuple[float, float, float, float]) -> float:
    x, y, z, a = q
    return np.arctan2(2 * x * z - 2 * a * y, (1 - 2 * y * y - 2 * z * z))


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


if __name__ == "__main__":
    dotenv.load_dotenv()

    stretch_ip = os.environ["STRETCH_IP"]
    quest_ip = os.environ["QUEST_IP"]

    """ ZMQ coniguration with quest"""
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(f"tcp://{quest_ip}:12345")

    """ Connect with move_to server"""
    websocket_url = f"ws://{stretch_ip}:8000/move_to_ws"
    with connect(websocket_url) as websocket:
        head_position_offset = None
        base_rotation_offset = None
        lift_height_offset = None
        wrist_offset = None

        requests.request(
            "POST",
            f"http://{stretch_ip}:8000/start_control_loop",
        )
        while True:
            start_time = time.time()
            # Receive message from remote server
            message = socket.recv()
            try:
                data = json.loads(message.decode())
            except json.JSONDecodeError:
                print("Terminating the connection...")
                websocket.send(TargetPosition().model_dump_json())
                websocket.close()
                socket.close()
                context.term()
                break

            # Deserialize the JSON message
            right_controller = data["RightController"]
            left_controller = data["LeftController"]
            xyz = right_controller["RightLocalPosition"]
            right_trigger_status = right_controller["RightIndexTrigger"]
            left_trigger_status = left_controller["LeftIndexTrigger"]
            right_safety = right_controller["RightHandTrigger"]
            left_safety = left_controller["LeftHandTrigger"]
            right_thumbstick_x, right_thumbstick_y = [
                float(x) for x in right_controller["RightThumbstickAxes"].split(",")
            ]
            right_button_a = right_controller["RightA"]
            right_button_b = right_controller["RightB"]

            # Get the controller rotation in quaternion form
            controller_rotation = [
                float(x) for x in right_controller["RightLocalRotation"].split(",")
            ]

            # Get the base rotation
            head_local_position = list(
                map(float, data["Headset"]["HeadLocalPosition"].split(","))
            )
            head_local_rotation = list(
                map(float, data["Headset"]["HeadLocalRotation"].split(","))
            )

            head_position = (head_local_position[2], -head_local_position[0])
            xyz = list(map(float, xyz.split(",")))
            controller_position = (xyz[2], -xyz[0])
            head_rotation = get_2d_rotation_from_quaternion(head_local_rotation)

            base_rotation = np.arctan2(
                controller_position[0] - head_position[0],
                -controller_position[1] + head_position[1],
            )

            arm_length_projected_to_floor = np.linalg.norm(
                [
                    controller_position[0] - head_position[0],
                    controller_position[1] - head_position[1],
                ]
            )
            arm_lift = xyz[1]

            if head_position_offset is None or right_button_b:
                current_base_status = requests.request(
                    "GET",
                    f"http://{stretch_ip}:8000/get_base_status",
                ).json()
                head_position_offset = (
                    head_position[0] - current_base_status["x"],
                    head_position[1] - current_base_status["y"],
                )

            head_position = (
                head_position[0] - head_position_offset[0],
                head_position[1] - head_position_offset[1],
            )

            if base_rotation_offset is None or right_button_b:
                current_base_status = requests.request(
                    "GET",
                    f"http://{stretch_ip}:8000/get_base_status",
                ).json()
                base_rotation_offset = base_rotation - current_base_status["theta"]

            base_rotation -= base_rotation_offset

            # The order of quaternion in unity is x, y, z, w
            # Check doc https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Quaternion.html
            wrist = get_euler(
                (
                    controller_rotation[3],
                    controller_rotation[0],
                    controller_rotation[1],
                    controller_rotation[2],
                )
            )

            grip_status = 95 - float(right_trigger_status) * 190
            grip_status = np.exp(0.02764 * (grip_status + 95)) - 90

            if wrist_offset is None or right_button_b:
                current_wrist_status = requests.request(
                    "GET",
                    f"http://{stretch_ip}:8000/get_end_of_arm_status",
                ).json()
                wrist_offset = (
                    wrist[0] + current_wrist_status["wrist_pitch"]["pos"],
                    wrist[1] + current_wrist_status["wrist_yaw"]["pos"],
                    wrist[2] + current_wrist_status["wrist_roll"]["pos"],
                )

            # websocket.send(
            #     TargetPosition(
            #         arm = arm_length_projected_to_floor,
            #         lift = arm_lift,
            #         theta=base_rotation,
            #         wrist_pitch = -wrist[0],
            #         wrist_yaw = _normalize_angle(-wrist[1] - base_rotation),
            #         wrist_roll = -wrist[2],
            #         stretch_gripper = grip_status,
            #     ).model_dump_json()
            # )

            # requests.request(
            #     "POST",
            #     f"http://{stretch_ip}:8000/end_of_arm_move_to",
            #     json={
            #         "wrist_yaw": _normalize_angle(-wrist[1] - base_rotation),
            #         "wrist_pitch": -wrist[0],
            #         "wrist_roll": -wrist[2],
            #         "stretch_gripper": grip_status,
            #     },
            # )

            end_time = time.time()
