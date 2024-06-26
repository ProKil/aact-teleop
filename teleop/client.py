import logging
import time
from typing import AsyncGenerator, Tuple, cast
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import uvicorn
import zmq
from zmq.asyncio import Socket, Context
import json
import requests
import numpy as np
import dotenv
import os

from teleop.data_classes import TargetPosition
from teleop.utils import _normalize_angle
from websockets.sync.client import connect, ClientConnection

import multiprocessing

from fastapi.templating import Jinja2Templates
from starlette.templating import _TemplateResponse
import asyncio


def get_2d_rotation_from_quaternion(q: Tuple[float, float, float, float]) -> float:
    x, y, z, a = q
    return np.arctan2(2 * x * z - 2 * a * y, (1 - 2 * y * y - 2 * z * z))  # type: ignore


def get_euler(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    w, x, y, z = q

    # roll: -atan2(y - z, a + x) + atan2(y + z, a - x),
    # pitch: 2*atan2(sqrt((a + x)**2 + (y - z)**2), sqrt((a - x)**2 + (y + z)**2)) - pi/2,
    # yaw: atan2(y - z, a + x) + atan2(y + z, a - x)

    roll = _normalize_angle(-np.arctan2(y - z, w + x) + np.arctan2(y + z, w - x))
    pitch = _normalize_angle(
        2
        * np.arctan2(
            np.sqrt((w + x) ** 2 + (y - z) ** 2), np.sqrt((w - x) ** 2 + (y + z) ** 2)
        )
        - np.pi / 2
    )
    yaw = _normalize_angle(np.arctan2(y - z, w + x) + np.arctan2(y + z, w - x))

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
    controller_thumbstick: Tuple[float, float] = Field(
        ..., description="thumbstick position. X-left/right, Y-forward/backward"
    )
    reset_button: bool
    record_button: bool


class Client(object):
    # Connections
    quest_ip: str
    stretch_ip: str
    quest_socket: Socket
    stretch_socket: ClientConnection | None

    # Memory
    base_rotation_offset: float | None
    quaternion_offset: Tuple[float, float, float, float] | None
    delta_time: float = 0.0

    # Configurations
    translation_speed: float = 10
    gripper_length: float = 0.26

    def __init__(self, dry_run: bool = False) -> None:
        self.quest_ip = os.environ["QUEST_IP"]
        self.stretch_ip = os.environ["STRETCH_IP"] if not dry_run else ""
        self.quest_socket = self._connect_quest()
        self.stretch_socket = self._connect_stretch() if not dry_run else None

        self.base_rotation_offset = None
        self.quaternion_offset = None

    def _connect_quest(self) -> Socket:
        context = Context()
        quest_socket = context.socket(zmq.PULL)
        quest_socket.setsockopt(zmq.CONFLATE, 1)
        quest_socket.connect(f"tcp://{self.quest_ip}:12345")

        return quest_socket

    def _connect_stretch(self) -> ClientConnection | None:
        websocket_url = f"ws://{self.stretch_ip}:8000/move_to_ws"
        return cast(ClientConnection, connect(websocket_url).__enter__())  # mypy bug

    def __del__(self) -> None:
        try:
            if self.stretch_socket:
                self.stretch_socket.send(TargetPosition().model_dump_json())
                self.stretch_socket.__exit__(None, None, None)
        except AttributeError as e:
            logging.warning(
                f"Partially initialized client: {e}. Skipping stretch socket cleanup."
            )

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
        desired_base_movement = (
            -controller_states.controller_thumbstick[0]
            * self.translation_speed
            * self.delta_time
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
        arm_lift = controller_states.controller_position[1]

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
            with open("log.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        dict(
                            controller_states=controller_states.model_dump(),
                            target_position=target_position.model_dump(),
                        )
                    )
                    + "\n"
                )

        return target_position

    async def event_loop(self) -> None:
        # start control loop

        last_time = time.time()

        if self.stretch_ip:
            _ = requests.request(
                "POST", f"http://{self.stretch_ip}:8000/start_control_loop"
            ).json()
            _ = requests.request(
                "POST", f"http://{self.stretch_ip}:8000/start_video_feed"
            ).json()
        while True:
            message = await self.quest_socket.recv()
            now_time = time.time()
            self.delta_time = now_time - last_time
            last_time = now_time
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
                record_button=bool(right_controller["RightA"]),
                controller_thumbstick=tuple(
                    map(float, right_controller["RightThumbstickAxes"].split(","))
                ),
            )

            control_signals = self._convert_controller_states_to_control_signals(
                controller_states
            )

            if self.stretch_socket is not None:
                self.stretch_socket.send(control_signals.model_dump_json())


def client_loop() -> None:
    """
    This function is used to receive the raw signals from the Quest and convert them into control signals for the robot.
    """
    print("running client loop")
    dotenv.load_dotenv()
    client = Client()
    asyncio.run(client.event_loop())
    del client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.background_process = multiprocessing.Process(target=client_loop)
    app.state.background_process.start()
    yield
    app.state.background_process.terminate()
    app.state.background_process.join()


templates = Jinja2Templates(directory="teleop/templates")
app = FastAPI(lifespan=lifespan)


@app.get("/video_feed")
def video_feed(request: Request) -> _TemplateResponse:
    dotenv.load_dotenv()
    return templates.TemplateResponse(
        "index.html.j2",
        {
            "ws_url": f"ws://{os.environ['STRETCH_IP']}:8000/video_feed_ws",
            "request": request,
        },
    )


def main() -> None:
    uvicorn.run("teleop.client:app", host="0.0.0.0", port=8001, reload=False)
