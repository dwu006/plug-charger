import json
import math
import os
import time

from pynput import keyboard as kb

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep


HERE = os.path.dirname(__file__)
SAVE_PATH = os.path.join(HERE, "home_position.json")

KP = 0.5
CONTROL_FREQ = 50
XY_STEP = 0.001
PAN_STEP = 0.3
PITCH_STEP = 0.3
ROLL_STEP = 0.3

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"]


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
    r = math.sqrt(x**2 + y**2)
    r_max, r_min = l1 + l2, abs(l1 - l2)
    if r > r_max:
        x *= r_max / r;  y *= r_max / r;  r = r_max
    if 0 < r < r_min:
        x *= r_min / r;  y *= r_min / r;  r = r_min
    cos_t2 = max(-1.0, min(1.0, -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)))
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


def main():
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/ttyACM0", id="follower_arm", use_degrees=True
    ))
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("failed to connect")

    obs = robot.get_observation()
    current = {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}
    target = dict(current)

    ee_x, ee_y = 0.1629, 0.1131
    pitch = target["wrist_flex"] + target["shoulder_lift"] + target["elbow_flex"]
    gripper_closed = False
    space_prev = False

    if os.path.exists(SAVE_PATH):
        print(f"existing home: {SAVE_PATH}")

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
            if _name("up"):    pitch += PITCH_STEP
            if _name("down"):  pitch -= PITCH_STEP
            if _name("left"):  target["wrist_roll"] += ROLL_STEP
            if _name("right"): target["wrist_roll"] -= ROLL_STEP
            target["wrist_flex"] = -target["shoulder_lift"] - target["elbow_flex"] + pitch

            space_now = _name("space")
            if space_now and not space_prev:
                gripper_closed = not gripper_closed
                target["gripper"] = 0.0 if gripper_closed else 90.0
            space_prev = space_now

            obs = robot.get_observation()
            current = {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}
            action = {f"{j}.pos": current[j] + KP * (target[j] - current[j])
                      for j in target if j in current}
            robot.send_action(action)

            precise_sleep(max(period - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        obs = robot.get_observation()
        home = {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}
        with open(SAVE_PATH, "w") as f:
            json.dump(home, f, indent=2)
        print(f"\nhome position saved to {SAVE_PATH}")
    finally:
        listener.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
