import json
import math
import os
import time

from pynput import keyboard as kb

from lerobot.robots.so_follower.so_follower import SO100Follower
from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig


KP = 0.5
CONTROL_FREQ = 50
XY_STEP = 0.001
PAN_STEP = 0.3
PITCH_STEP = 0.3
ROLL_STEP = 0.3
GRIP_STEP = 1.5
GRIP_MIN = 0.0
GRIP_MAX = 90.0


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r = math.sqrt(x**2 + y**2)
    r_max, r_min = l1 + l2, abs(l1 - l2)

    if r > r_max:
        x *= r_max / r;  y *= r_max / r;  r = r_max
    if 0 < r < r_min:
        x *= r_min / r;  y *= r_min / r;  r = r_min

    cos_t2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    theta2 = math.pi - math.acos(cos_t2)
    theta1 = math.atan2(y, x) + math.atan2(l2 * math.sin(theta2),
                                             l1 + l2 * math.cos(theta2))

    j2 = max(-0.1, min(3.45, theta1 + theta1_offset))
    j3 = max(-0.2, min(math.pi, theta2 + theta2_offset))

    return 90 - math.degrees(j2), math.degrees(j3) - 90


_pressed: set = set()

def _on_press(key):   _pressed.add(key)
def _on_release(key): _pressed.discard(key)

def _char(c): return any(getattr(k, "char", None) == c for k in _pressed)
def _name(n): return any(getattr(k, "name", None) == n for k in _pressed)


def main(robot_port="/dev/ttyACM0"):
    robot = SO100Follower(SO100FollowerConfig(port=robot_port, id="follower_arm", use_degrees=True))
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("failed to connect to SO101")

    home_path = os.path.join(os.path.dirname(__file__), "home_position.json")
    home = None
    if os.path.exists(home_path):
        with open(home_path) as f:
            home = json.load(f)
        print("moving to home position...")
        for _ in range(150):
            obs = robot.get_observation()
            current = {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}
            action = {f"{j}.pos": current[j] + 0.1 * (home[j] - current[j]) for j in home if j in current}
            robot.send_action(action)
            time.sleep(1.0 / 50)
        print("at home position\n")

    obs = robot.get_observation()
    current = {k.removesuffix(".pos"): float(v)
               for k, v in obs.items() if k.endswith(".pos")}

    target = dict(current)

    ee_x, ee_y = 0.1629, 0.1131
    pitch = target["wrist_flex"] + target["shoulder_lift"] + target["elbow_flex"]
    grip_val = target.get("gripper", 90.0)

    listener = kb.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    period = 1.0 / CONTROL_FREQ

    try:
        while True:
            t0 = time.perf_counter()

            moved = False
            if _char("w"): ee_x += XY_STEP;  moved = True
            if _char("s"): ee_x -= XY_STEP;  moved = True
            if _char("q"): ee_y += XY_STEP;  moved = True
            if _char("e"): ee_y -= XY_STEP;  moved = True
            if moved:
                target["shoulder_lift"], target["elbow_flex"] = inverse_kinematics(ee_x, ee_y)

            if _char("a"): target["shoulder_pan"] -= PAN_STEP
            if _char("d"): target["shoulder_pan"] += PAN_STEP

            if _name("up"):   pitch += PITCH_STEP
            if _name("down"): pitch -= PITCH_STEP

            if _name("left"):  target["wrist_roll"] += ROLL_STEP
            if _name("right"): target["wrist_roll"] -= ROLL_STEP

            target["wrist_flex"] = -target["shoulder_lift"] - target["elbow_flex"] + pitch

            if _name("space"): grip_val = min(grip_val + GRIP_STEP, GRIP_MAX)
            if _char("b"):     grip_val = max(grip_val - GRIP_STEP, GRIP_MIN)
            target["gripper"] = grip_val

            if _char("r") and home:
                target = dict(home)
                pitch = home["wrist_flex"] + home["shoulder_lift"] + home["elbow_flex"]
                ee_x, ee_y = 0.1629, 0.1131
                grip_val = home.get("gripper", 90.0)

            obs = robot.get_observation()
            current = {k.removesuffix(".pos"): float(v)
                       for k, v in obs.items() if k.endswith(".pos")}

            action = {
                f"{joint}.pos": current[joint] + KP * (tgt - current[joint])
                for joint, tgt in target.items()
                if joint in current
            }
            robot.send_action(action)

            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        print("stopping")
        listener.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
