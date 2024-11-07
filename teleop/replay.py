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
    with open(file_path, "r") as f:
        json_data = [json.loads(line) for line in f]
        json_data = [json.dumps(x["data"]) for x in json_data]

    for target_position_dict in json_data:
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
