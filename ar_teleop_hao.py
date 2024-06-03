"""
This is our current implementation which consists of:
    1. set up ZMQ connection
    2. set up the initial position of controller on first iteration
    3. within the while True loop:
        a. read controller information from ZMQ
        b. generate new configuration
           call gripper_to_goal.update_goal to execute the new goal

We modified the original update_goal method to fit our new configuration format
"""

import time
from typing import Tuple
import zmq
import json
import argparse
import requests
import numpy as np

stretch_ip = "http://172.26.188.87"

"""
Parse the command line argument
1. The argument -i is the interval of responding the controller data
2. The argument -s is the speed of the robot, the choices include slow or fastest_stretch_2
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--interval", type=int, default=1, help="Interval in seconds"
)
parser.add_argument(
    "-s",
    "--speed",
    choices=["slow", "fastest_stretch_2"],
    default="fastest_stretch_2",
    help="Speed option (choices: slow, fastest_stretch_2)",
)
args = parser.parse_args()

""" ZMQ coniguration"""
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.setsockopt(zmq.CONFLATE, 1)

""" Quest configuration"""
# Quest 3 IP on CMU-DEVICE
socket.connect("tcp://172.26.172.110:12345")
# Quest 3 IP on Jensen's WIFI
# socket.connect("tcp://192.168.1.179:12345")

""" Robot configuration"""
robot_speed = args.speed
manipulate_on_ground = False
robot_allowed_to_move = True
using_stretch_2 = True

"""
Set up the initial configuration of the update goal
This initial movement is just to demonstrate the script has launched, 
    the robot can move, and to move the arm to a good initial position.
The argument are set based on user experience

1. joint_lift is set to 0.6, which is similar to the position of human
2. the right and left trigger are set to 0, to make sure the initialization of gripper remain the same place
3. the right safety and left safety are set to 1 to enable the safety lock so robot is able to move to initial position
4. joint_arm_10 is set to 
5. q1, q2, q3, q4 are the quaternion form
6. right_thumbstick_x is the base rotation, it is set to 0 in the initial configuration
7. right_thumbstick_y is the base translation, it is set to 0 in the initial configuration
"""
base_rotation = 0.0
initial_configuration = {
    "joint_mobile_base_rotation": base_rotation,
    "joint_lift": 0.6,
    "right_trigger_status": 0,  # range: 0~1
    "left_trigger_status": 0,  # range: 0~1
    "right_safety": 1,  # range: 0~1
    "left_safety": 1,  # range: 0~1
    "joint_arm_l0": 0.25,
    "q1": -0.04,
    "q2": 0.19,
    "q3": -0.08,
    "q4": 0.97,
    "right_thumbstick_x": 0,
    "right_thumbstick_y": 0,
    "right_button_a": False,
    "right_button_b": False,
}

def _normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].

    Args:
        angle: The angle to normalize.

    Returns:
        The normalized angle.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_2d_rotation_from_quaternion(q):
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
    sinp = 2 * (w * x + y * z)
    cosp = 1 - 2 * (x * x + y * y)
    pitch = np.arctan2(sinp, cosp)

    # Compute yaw (y-axis rotation)
    siny = 2 * (w * y - z * x)
    yaw = np.arcsin(siny)

    # Compute roll (z-axis rotation)
    sinr = 2 * (w * z + x * y)
    cosr = 1 - 2 * (y * y + z * z)
    roll = np.arctan2(sinr, cosr)

    return pitch, yaw, roll


# Sleep after reaching position to ensure stability
time.sleep(5)

response_cnt = 0
interval = args.interval
f = open("log.jsonl", "w")
head_position_offset = None
base_rotation_offset = None
lift_height_offset = None
wrist_offset = None

requests.request(
    "POST",
    f"{stretch_ip}:8000/start_control_loop",
)
while True:
    start_time = time.time()
    # Receive message from remote server
    message = socket.recv()
    try:
        data = json.loads(message.decode())
    except json.JSONDecodeError:
        print("Terminating the connection...")
        socket.close()
        context.term()
        break

    if response_cnt % interval == 0:
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
                f"{stretch_ip}:8000/get_base_status",
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
                f"{stretch_ip}:8000/get_base_status",
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

        # requests.request(
        #     "POST",
        #     f"{stretch_ip}:8000/move_to",
        #     json={
        #         "x": 0,
        #         "y": 0,
        #         "theta": base_rotation,
        #     }
        # )

        # print(arm_length_projected_to_floor, arm_lift)

        requests.request(
            "POST",
            f"{stretch_ip}:8000/arm_move_to",
            json={
                "arm": arm_length_projected_to_floor,
                "lift": arm_lift,
            }
        )

        if wrist_offset is None or right_button_b:
            current_wrist_status = requests.request(
                "GET",
                f"{stretch_ip}:8000/get_end_of_arm_status",
            ).json()
            wrist_offset = (
                wrist[0] + current_wrist_status["wrist_pitch"]["pos"],
                wrist[1] + current_wrist_status["wrist_yaw"]["pos"],
                wrist[2] + current_wrist_status["wrist_roll"]["pos"],
            )


        # requests.request(
        #     "POST",
        #     f"{stretch_ip}:8000/end_of_arm_move_to",
        #     json={
        #         "wrist_yaw": _normalize_angle(-wrist[1] - base_rotation),
        #         "wrist_pitch": -wrist[0],
        #         "wrist_roll": -wrist[2],
        #         "stretch_gripper": grip_status,
        #     },
        # )

    response_cnt += 1
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
