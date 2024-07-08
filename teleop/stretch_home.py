from stretch_body import robot as rb
import time

robot = rb.Robot()
robot.startup()
robot.home()
time.sleep(2)
robot.stop()
