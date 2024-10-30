from pydantic import ValidationError
from stretch_body.hello_utils import ThreadServiceExit
import time


from .stretch_node import read_target_position_replay, write_target_position
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--filename",
    type=str,
    default="/home/hello-robot/stretch_teleop_server/recordtest_2024-10-30_17-25-11.jsonl",
    help="Path to the replay file",
)

args = parser.parse_args()

file_path = args.filename


if __name__ == "__main__":
    # robot: Robot = Robot()
    # robot.startup()

    # start_time = time.time()
    # current_time: float = 0

    # target_position = TargetPosition()

    with open(file_path, "r") as f:
        json_data = [json.loads(line) for line in f]
        json_data = [json.dumps(x["data"]) for x in json_data]

    for target_position_dict in json_data:
        # new_current_time = time.time() - start_time
        # delta_t = new_current_time - current_time
        # current_time = new_current_time

        # current_x, current_y, current_theta = (
        #     robot.base.status["x"],
        #     robot.base.status["y"],
        #     robot.base.status["theta"],
        # )

        try:
            target_position = read_target_position_replay(target_position_dict)
            write_target_position(target_position, "/dev/shm/target_position.json")
            time.sleep(1 / 80)
        except FileNotFoundError:
            print("File Not Found!")
            time.sleep(1)
            continue
        except ValidationError as e:
            print(e)
            time.sleep(1 / 80)
            continue
        except (ThreadServiceExit, KeyboardInterrupt):
            print("Exiting control loop.")
            # robot.stop()
            # handle_exit(0, 0)

        # next_step_position = plan_trajectory(
        #     PlanTrajectoryArguments(
        #         target_position=Position(
        #             x=target_position.x,
        #             y=target_position.y,
        #             theta=target_position.theta,
        #         ),
        #         current_position=Position(
        #             x=current_x, y=current_y, theta=current_theta
        #         ),
        #     )
        # )
        # x_d, y_d, _theta_d = (
        #     next_step_position.x,
        #     next_step_position.y,
        #     next_step_position.theta,
        # )
        # # Get the control inputs
        # v, omega = pid_control_policy(
        #     ControlPolicyArguments(
        #         desired_position=Position(x=x_d, y=y_d, theta=target_position.theta),
        #         current_position=Position(
        #             x=current_x, y=current_y, theta=current_theta
        #         ),
        #         dt=delta_t,
        #     )
        # )

        # try:
        #     # robot.base.set_velocity(target_position.translation_speed, omega)
        #     # robot.arm.move_to(target_position.arm, v_m=0.18, a_m=1)
        #     # robot.lift.move_to(target_position.lift, v_m=2, a_m=2)
        #     # robot.end_of_arm.move_to("wrist_yaw", target_position.wrist_yaw)
        #     # robot.end_of_arm.move_to("wrist_pitch", target_position.wrist_pitch)
        #     # robot.end_of_arm.move_to("wrist_roll", target_position.wrist_roll)
        #     # robot.end_of_arm.move_to("stretch_gripper", target_position.stretch_gripper)
        #     # robot.head.move_to("head_tilt", target_position.head_tilt)
        #     # robot.head.move_to("head_pan", target_position.head_pan)
        #     current_position = deepcopy(target_position)
        #     current_position.x = current_x
        #     current_position.y = current_y
        #     current_position.theta = current_theta
        #     # robot.push_command()
        #     write_target_position(current_position)
        #     time.sleep(1 / 80)
        # except (ThreadServiceExit, KeyboardInterrupt):
        #     print("Exiting control loop.")
        #     robot.stop()
        #     handle_exit(0, 0)
