import asyncio
import json
import math
import sys
import time
from typing import List, Tuple
from typing_extensions import Annotated
import numpy as np
from stretch_body import robot as rb
from stretch_body.hello_utils import ThreadServiceExit
from pydantic import BaseModel, Field, AfterValidator

from fastapi import BackgroundTasks, FastAPI, WebSocket
from contextlib import asynccontextmanager
import signal

from .data_classes import TargetPosition
from .utils import _normalize_angle


@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot
    # Load the ML model
    print("starting robot")
    robot = rb.Robot()
    robot.startup()
    yield
    # Clean up the ML models and release the resources
    robot.stop()


app = FastAPI(lifespan=lifespan)


# Function to handle graceful shutdown
def handle_exit(sig, frame):
    print("Gracefully shutting down...")
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


robot = None


def _choose_closest_angle(
    candidate_angles: List[float], reference_angle: float
) -> float:
    """
    Choose the angle that is closest to a reference angle.

    Args:
        angle1: The first angle.
        angle2: The second angle.
        reference_angle: The reference angle.
    Returns:
        The closest angle to the reference angle.
    """
    normalized_angles = [_normalize_angle(angle) for angle in candidate_angles]
    normalized_reference_angle = _normalize_angle(reference_angle)

    return min(
        normalized_angles,
        key=lambda angle: np.abs(normalized_reference_angle - angle),
    )


class Position(BaseModel):
    x: float = Field(..., description="x position.")
    y: float = Field(..., description="y position.")
    theta: Annotated[float, AfterValidator(lambda v: _normalize_angle(v))] = Field(
        ..., description="orientation."
    )


class ArmAndLiftPosition(BaseModel):
    arm: Annotated[float, AfterValidator(lambda x: min(max(x, 0), 0.5))] = Field(
        ..., description="The arm position."
    )
    lift: Annotated[float, AfterValidator(lambda x: min(max(x, 0.1), 1.09))] = Field(
        ..., description="The lift position."
    )


class EndOfArmPosition(BaseModel):
    wrist_yaw: float = Field(..., description="The wrist yaw position.")
    wrist_pitch: float = Field(..., description="The wrist pitch position.")
    wrist_roll: float = Field(..., description="The wrist roll position.")
    stretch_gripper: float = Field(..., description="The stretch gripper position.")


class PlanTrajectoryArguments(BaseModel):
    """
    Arguments for the plan_trajectory function.
    """

    target_position: Position = Field(..., description="The target position.")
    current_position: Position = Field(..., description="The current position.")


def plan_trajectory(args: PlanTrajectoryArguments) -> Position:
    """
    Trajectory planning function that computes the next step position to reach the desired position.

    Algorithm:
    - If the distance is large, move towards the desired position.
    - If the distance is small, rotate towards the desired orientation.

    Args:
        args (PlanTrajectoryArguments) : The arguments for the plan_trajectory function.

    Returns:
        NextStepPosition: The next step position to reach the desired position.
    """
    x_desired, y_desired, theta_desired = (
        args.target_position.x,
        args.target_position.y,
        args.target_position.theta,
    )
    x_current, y_current, theta_current = (
        args.current_position.x,
        args.current_position.y,
        args.current_position.theta,
    )

    # Compute the distance and angle to the desired position
    dx = x_desired - x_current
    dy = y_desired - y_current
    dtheta = _normalize_angle(theta_desired - theta_current)
    dL = math.sqrt(dx**2 + dy**2)

    # Compute the time to reach the desired position
    v_max = 1  # maximum linear velocity
    omega_max = 1  # maximum angular velocity

    t_L = dL / v_max + dtheta / omega_max  # time to reach the desired position
    Tu = 0.5  # time constant
    N = max(int(t_L / Tu), 1)  # number of steps

    if dL > 0.05:  # if the distance is large, move towards the desired position
        return Position(
            x=x_current + dx / N,
            y=y_current + dy / N,
            theta=_choose_closest_angle(
                [np.arctan2(dy, dx), np.arctan2(-dy, -dx)], theta_current
            ),
        )
    else:  # if the distance is small, rotate towards the desired orientation
        return Position(
            x=x_desired, y=y_desired, theta=_normalize_angle(theta_current + dtheta / N)
        )


class PIDController(object):
    """
    A simple PID controller implementation.
    """

    def __init__(self, Kp: float, Ki: float, Kd: float, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0
        self.output_limits = output_limits
        self.integral_decay = 0.99

    def update(self, error, dt):
        self.integral = self.integral * self.integral_decay + (error * dt)
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Apply output limits
        min_output, max_output = self.output_limits
        if min_output is not None:
            output = max(output, min_output)
        if max_output is not None:
            output = min(output, max_output)

        self.previous_error = error
        return output


# Initial PID parameters
Kp_x, Ki_x, Kd_x = 2.0, 0.0, 0.0
Kp_y, Ki_y, Kd_y = 4.0, 0.2, 0.0
Kp_theta, Ki_theta, Kd_theta = 2.0, 0.0, 0.0

min_v, max_v = -2.0, 2.0
min_omega, max_omega = -2.0, 2.0

# Define PID controllers with appropriate limits
pid_x = PIDController(Kp_x, Ki_x, Kd_x)
pid_y = PIDController(Kp_y, Ki_y, Kd_y)
pid_theta = PIDController(Kp_theta, Ki_theta, Kd_theta)


class ControlPolicyArguments(BaseModel):
    desired_position: Position = Field(..., description="The target position.")
    current_position: Position = Field(..., description="The current position.")
    dt: float = Field(..., description="The time step.")


def pid_control_policy(args: ControlPolicyArguments) -> Tuple[float, float]:
    """
    PID control policy that computes the control actions to reach the desired position.

    Args:
        args (ControlPolicyArguments): The arguments for the control policy.

    Returns:
        Tuple[float, float]: The control actions (v, omega).
    """
    global f
    # Compute errors
    e_x = args.desired_position.x - args.current_position.x
    e_y = args.desired_position.y - args.current_position.y
    e_theta = _normalize_angle(
        args.desired_position.theta - args.current_position.theta
    )

    # Normalize orientation error to the range [-pi, pi]
    e_theta = (e_theta + np.pi) % (2 * np.pi) - np.pi

    # Convert position errors to robot frame
    e_x_prime = e_x * math.cos(args.current_position.theta) + e_y * math.sin(
        args.current_position.theta
    )
    e_y_prime = -e_x * math.sin(args.current_position.theta) + e_y * math.cos(
        args.current_position.theta
    )

    # Update PID controllers
    control_x = pid_x.update(e_x_prime, args.dt)
    control_y = pid_y.update(e_y_prime, args.dt)
    control_theta = pid_theta.update(e_theta, args.dt)

    f.write(
        json.dumps(
            {
                "position": args.current_position.dict(),
                "desired_position": args.desired_position.dict(),
                "e_x_prime": e_x_prime,
                "e_y_prime": e_y_prime,
                "e_theta": e_theta,
                "control_x": control_x,
                "control_y": control_y,
                "control_theta": control_theta,
            }
        )
        + "\n"
    )

    # Combine control actions
    v = control_x
    omega = control_y + control_theta

    # Apply limits to control signals to prevent instability
    v = max(min_v, min(v, max_v))
    omega = max(min_omega, min(omega, max_omega))

    return v, omega


move_to_position = (0, 0, 0)
wrist_move_to_position = EndOfArmPosition(
    wrist_yaw=0, wrist_pitch=0, wrist_roll=0, stretch_gripper=0
)
arm_move_to_position = ArmAndLiftPosition(arm=0.25, lift=0.6)


target_position = TargetPosition()
control_loop_started = False


async def control_loop(robot: rb.Robot) -> None:
    global control_loop_started
    global target_position

    if control_loop_started:
        print("Control loop already started.")
    else:
        print("Starting control loop.")
        control_loop_started = True

    start_time = time.time()
    current_time = 0

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
                    x=target_position.x,
                    y=target_position.y,
                    theta=target_position.theta,
                ),
                current_position=Position(
                    x=current_x, y=current_y, theta=current_theta
                ),
            )
        )
        x_d, y_d, theta_d = (
            next_step_position.x,
            next_step_position.y,
            next_step_position.theta,
        )
        # Get the control inputs
        v, omega = pid_control_policy(
            ControlPolicyArguments(
                desired_position=Position(x=x_d, y=y_d, theta=theta_d),
                current_position=Position(
                    x=current_x, y=current_y, theta=current_theta
                ),
                dt=delta_t,
            )
        )

        try:
            robot.base.set_velocity(v, omega)
            robot.arm.move_to(target_position.arm, v_m=2, a_m=2)
            robot.lift.move_to(target_position.lift, v_m=2, a_m=2)
            robot.end_of_arm.move_to("wrist_yaw", target_position.wrist_yaw)
            robot.end_of_arm.move_to("wrist_pitch", target_position.wrist_pitch)
            robot.end_of_arm.move_to("wrist_roll", target_position.wrist_roll)
            robot.end_of_arm.move_to("stretch_gripper", target_position.stretch_gripper)
            robot.push_command()
            await asyncio.sleep(1 / 80)
        except (ThreadServiceExit, KeyboardInterrupt):
            print("Exiting control loop.")
            handle_exit(0, 0)


@app.get("/get_base_status")
async def get_base_status():
    return robot.base.status


@app.get("/get_end_of_arm_status")
async def get_end_of_arm_status():
    return robot.end_of_arm.status


@app.get("/get_arm_status")
async def get_arm_status():
    return robot.arm.status


@app.get("/get_lift_status")
async def get_lift_status():
    return robot.lift.status


@app.post("/move_to")
async def move_to(position: Position):
    global move_to_position
    move_to_position = (position.x, position.y, position.theta)
    return {"message": "Moving to position."}


@app.post("/arm_move_to")
async def arm_move_to(position: ArmAndLiftPosition):
    global target_position
    target_position.arm = position.arm
    target_position.lift = position.lift
    return {"message": "Moving arm to position."}


@app.post("/end_of_arm_move_to")
async def end_of_arm_move_to(position: EndOfArmPosition):
    global wrist_move_to_position
    wrist_move_to_position = position
    return {"message": "Moving wrist and gripper to position."}


@app.post("/start_control_loop")
async def start_control_loop(background_tasks: BackgroundTasks):
    if background_tasks.tasks:
        return {"message": "Control loop already started."}
    background_tasks.add_task(control_loop, robot)
    return {"message": "Control loop started."}


@app.websocket("/move_to_ws")
async def move_to_ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        global target_position
        target_position = TargetPosition(**json.loads(data))
