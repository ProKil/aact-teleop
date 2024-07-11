from copy import deepcopy
from pydantic import ValidationError
from stretch_body.hello_utils import ThreadServiceExit
from stretch_body.robot import Robot
import time

from .server import (
    ControlPolicyArguments,
    PlanTrajectoryArguments,
    Position,
    handle_exit,
    pid_control_policy,
    plan_trajectory,
)

from .data_classes import TargetPosition


def read_target_position(
    file_path: str = "/dev/shm/target_position.json",
) -> TargetPosition:
    with open(file_path, "r") as f:
        return TargetPosition.model_validate_json(f.read())


def write_target_position(
    target_position: TargetPosition, file_path: str = "/dev/shm/current_position.json"
) -> None:
    with open(file_path, "w") as f:
        f.write(target_position.model_dump_json())


if __name__ == "__main__":
    robot: Robot = Robot()
    robot.startup()

    start_time = time.time()
    current_time: float = 0

    target_position = TargetPosition()

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

        try:
            target_position = read_target_position()
        except FileNotFoundError:
            time.sleep(1)
            continue
        except ValidationError:
            time.sleep(1 / 80)
            continue

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
        x_d, y_d, _theta_d = (
            next_step_position.x,
            next_step_position.y,
            next_step_position.theta,
        )
        # Get the control inputs
        v, omega = pid_control_policy(
            ControlPolicyArguments(
                desired_position=Position(x=x_d, y=y_d, theta=target_position.theta),
                current_position=Position(
                    x=current_x, y=current_y, theta=current_theta
                ),
                dt=delta_t,
            )
        )

        try:
            robot.base.set_velocity(target_position.translation_speed, omega)
            robot.arm.move_to(target_position.arm, v_m=2, a_m=2)
            robot.lift.move_to(target_position.lift, v_m=2, a_m=2)
            robot.end_of_arm.move_to("wrist_yaw", target_position.wrist_yaw)
            robot.end_of_arm.move_to("wrist_pitch", target_position.wrist_pitch)
            robot.end_of_arm.move_to("wrist_roll", target_position.wrist_roll)
            robot.end_of_arm.move_to("stretch_gripper", target_position.stretch_gripper)
            robot.head.move_to("head_tilt", target_position.head_tilt)
            robot.head.move_to("head_pan", target_position.head_pan)
            current_position = deepcopy(target_position)
            current_position.x = current_x
            current_position.y = current_y
            current_position.theta = current_theta
            robot.push_command()
            write_target_position(current_position)
            time.sleep(1 / 80)
        except (ThreadServiceExit, KeyboardInterrupt):
            print("Exiting control loop.")
            robot.stop()
            handle_exit(0, 0)
