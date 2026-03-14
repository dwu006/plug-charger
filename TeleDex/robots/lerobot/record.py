import json
import math
import os
import pathlib
import shutil
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from pynput import keyboard

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep


TASK = "pick up the block and place it in the container"
REPO_ID = "local/so101"
ROOT = "data"
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 600  # 20s at 30fps
ROBOT_PORT = "/dev/ttyACM0"
FPS = 30
ZOOM_FACTOR = 1.0

KP = 0.5
CONTROL_FREQ = FPS
XY_STEP = 0.001
PAN_STEP = 0.3
PITCH_STEP = 0.3
ROLL_STEP = 0.3
GRIP_STEP = 1.5
GRIP_MIN = 0.0
GRIP_MAX = 90.0

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


class EpisodeBuilder:
    _FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

    def __init__(self, fps, video_dir):
        self._fps = fps
        self._video_dir = pathlib.Path(video_dir)
        self._rgb_writer = None
        self._dep_writer = None
        self._rgb_path = None
        self._dep_path = None
        self._count = 0

    def start(self, episode_idx):
        rgb_dir = self._video_dir / "observation.images.rgb"
        depth_dir = self._video_dir / "observation.images.depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        fname = f"episode_{episode_idx:06d}.mp4"
        self._rgb_path = str(rgb_dir / fname)
        self._dep_path = str(depth_dir / fname)

        self._rgb_writer = cv2.VideoWriter(self._rgb_path, self._FOURCC, self._fps, (640, 480))
        self._dep_writer = cv2.VideoWriter(self._dep_path, self._FOURCC, self._fps, (640, 480))
        self._count = 0

    def add_frame(self, color_bgr, depth_u16):
        if self._rgb_writer is None:
            return
        self._rgb_writer.write(color_bgr)
        depth_u8 = cv2.convertScaleAbs(depth_u16, alpha=0.03)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        self._dep_writer.write(depth_color)
        self._count += 1

    def save(self):
        if self._rgb_writer:
            self._rgb_writer.release()
            self._dep_writer.release()
            self._rgb_writer = None
            self._dep_writer = None
        return self._rgb_path, self._dep_path

    def discard(self):
        if self._rgb_writer:
            self._rgb_writer.release()
            self._dep_writer.release()
            self._rgb_writer = None
            self._dep_writer = None
        for p in (self._rgb_path, self._dep_path):
            if p and os.path.exists(p):
                os.remove(p)
        self._count = 0

    def __len__(self):
        return self._count


def apply_digital_zoom(color_bgr, depth_u16, zoom):
    if zoom <= 1.0:
        return color_bgr, depth_u16
    h, w = color_bgr.shape[:2]
    if depth_u16.shape[:2] != (h, w):
        depth_u16 = cv2.resize(depth_u16, (w, h), interpolation=cv2.INTER_NEAREST)
    nw, nh = int(w / zoom), int(h / zoom)
    x1, y1 = (w - nw) // 2, (h - nh) // 2
    color_zoomed = cv2.resize(color_bgr[y1:y1+nh, x1:x1+nw], (w, h), interpolation=cv2.INTER_LINEAR)
    depth_zoomed = cv2.resize(depth_u16[y1:y1+nh, x1:x1+nw], (w, h), interpolation=cv2.INTER_NEAREST)
    return color_zoomed, depth_zoomed


def main():
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
    align = rs.align(rs.stream.color)
    rs_pipeline.start(rs_config)

    robot = SO100Follower(SO100FollowerConfig(
        port=ROBOT_PORT, id="follower_arm", use_degrees=True
    ))
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to SO101.")

    home_path = os.path.join(os.path.dirname(__file__), "home_position.json")
    home = json.load(open(home_path)) if os.path.exists(home_path) else None
    if home:
        print("moving to home position...")
        for _ in range(150):
            obs = robot.get_observation()
            current = {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}
            action = {f"{j}.pos": current[j] + 0.1 * (home[j] - current[j]) for j in home if j in current}
            robot.send_action(action)
            time.sleep(1.0 / 50)
        print("at home position\n")

    joint_names = [f"{m}.pos" for m in MOTOR_NAMES]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": joint_names},
        "action": {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": joint_names},
    }
    root_path = pathlib.Path(ROOT)

    tasks_parquet = root_path / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        dataset = LeRobotDataset(repo_id=REPO_ID, root=str(root_path), download_videos=False)
        existing = dataset.num_episodes
        print(f"resuming - {existing} episodes already in '{root_path}'")
    else:
        if root_path.exists():
            shutil.rmtree(root_path)
        dataset = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, root=str(root_path),
                                         robot_type=robot.name, features=features, use_videos=False)
        existing = 0
        print(f"new dataset created at '{root_path}'")

    ep_builder = EpisodeBuilder(fps=FPS, video_dir=root_path / "videos")

    _pressed = set()
    events = {"start": False, "save": False, "rerecord": False, "quit": False}

    def on_press(key):
        _pressed.add(key)
        try:
            ch = key.char.lower() if hasattr(key, "char") and key.char else None
        except Exception:
            ch = None
        if ch == "y":  events["save"] = True
        if ch == "r":  events["rerecord"] = True
        if hasattr(key, "name"):
            if key.name == "enter":  events["start"] = True
            if key.name == "esc":    events["quit"] = True

    def on_release(key):
        _pressed.discard(key)

    def _char(c): return any(getattr(k, "char", None) == c for k in _pressed)
    def _name(n): return any(getattr(k, "name", None) == n for k in _pressed)

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()

    obs = robot.get_observation()
    current = {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}
    target = dict(current)

    ee_x, ee_y = 0.1629, 0.1131
    pitch = target["wrist_flex"] + target["shoulder_lift"] + target["elbow_flex"]
    grip_val = target.get("gripper", 90.0)

    recording = False
    recorded = existing
    steps_recorded = 0
    max_reached = False
    prompt_shown = False

    last_color_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    last_depth_u16 = np.zeros((480, 640), dtype=np.uint16)

    print(f"task: '{TASK}'  |  {NUM_EPISODES} episodes  |  {MAX_STEPS_PER_EPISODE} max steps each\n")

    try:
        while recorded < NUM_EPISODES and not events["quit"]:
            if not recording and not prompt_shown:
                if max_reached:
                    print(f"\nep {recorded+1}/{NUM_EPISODES} - max steps reached. Y=save  R=discard")
                else:
                    print(f"ep {recorded+1}/{NUM_EPISODES} - press Enter to start recording")
                prompt_shown = True

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

            if _name("space"): grip_val = min(grip_val + GRIP_STEP, GRIP_MAX)
            if _char("b"):     grip_val = max(grip_val - GRIP_STEP, GRIP_MIN)
            target["gripper"] = grip_val

            robot_obs = robot.get_observation()
            current = {k.removesuffix(".pos"): float(v)
                       for k, v in robot_obs.items() if k.endswith(".pos")}
            action = {f"{j}.pos": current[j] + KP * (target[j] - current[j])
                      for j in target if j in current}
            robot.send_action(action)

            ok, frames_rs = rs_pipeline.try_wait_for_frames(timeout_ms=0)
            if ok:
                aligned = align.process(frames_rs)
                color_bgr = np.asanyarray(aligned.get_color_frame().get_data())
                depth_u16 = np.asanyarray(aligned.get_depth_frame().get_data())
                color_bgr, depth_u16 = apply_digital_zoom(color_bgr, depth_u16, ZOOM_FACTOR)
                last_color_bgr = color_bgr
                last_depth_u16 = depth_u16

                preview = color_bgr.copy()
                if recording:
                    cv2.circle(preview, (20, 20), 10, (0, 0, 255), -1)
                    cv2.putText(preview, f"REC {steps_recorded}", (38, 27),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(preview, "READY - Enter to record", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("RealSense", preview)
                cv2.waitKey(1)

            if recording:
                obs_state = np.array([current.get(m, 0.0) for m in MOTOR_NAMES], dtype=np.float32)
                act_state = np.array([target.get(m, 0.0) for m in MOTOR_NAMES], dtype=np.float32)

                if steps_recorded == 0:
                    missing = [m for m in MOTOR_NAMES if m not in current]
                    if missing:
                        print(f"\nWARNING: joints not found in obs: {missing}")

                dataset.add_frame({"observation.state": obs_state, "action": act_state, "task": TASK})
                ep_builder.add_frame(last_color_bgr, last_depth_u16)
                steps_recorded += 1

                filled = int(30 * steps_recorded / MAX_STEPS_PER_EPISODE)
                bar = "\u2588" * filled + "\u2591" * (30 - filled)
                print(f"\rep {recorded+1}: {steps_recorded}/{MAX_STEPS_PER_EPISODE} [{bar}]", end="", flush=True)

                if steps_recorded >= MAX_STEPS_PER_EPISODE:
                    print(f"\nep {recorded+1} - max steps reached")
                    recording = False
                    max_reached = True
                    prompt_shown = False

            if events["start"] and not recording:
                events["start"] = False
                recording = True
                prompt_shown = False
                max_reached = False
                steps_recorded = 0
                ep_builder.start(recorded)
                print(f"\nep {recorded+1} recording... Y=save  R=discard")

            if events["save"] and (recording or max_reached):
                events["save"] = False
                if steps_recorded == 0:
                    print("\nnothing recorded yet")
                else:
                    recording = False
                    max_reached = False
                    prompt_shown = False
                    episode_idx = recorded
                    recorded += 1
                    steps_recorded = 0
                    ep_builder.save()
                    dataset.save_episode()
                    dataset.finalize()
                    dataset = LeRobotDataset(repo_id=REPO_ID, root=str(root_path), download_videos=False)
                    print(f"\nep {episode_idx+1} saved")
                    print(f"ep {recorded}/{NUM_EPISODES} - press Enter for next\n")
                    if recorded % 10 == 0:
                        snap_name = f"data_{recorded}v3"
                        snap_path = root_path.parent / snap_name
                        if snap_path.exists():
                            shutil.rmtree(snap_path)
                        shutil.copytree(root_path, snap_path)
                        print(f"  snapshot copied to {snap_path}")
                    if home:
                        target = dict(home)
                        pitch = home["wrist_flex"] + home["shoulder_lift"] + home["elbow_flex"]
                        ee_x, ee_y = 0.1629, 0.1131
                        grip_val = home.get("gripper", 90.0)

            if events["rerecord"] and (recording or max_reached):
                events["rerecord"] = False
                recording = False
                max_reached = False
                prompt_shown = False
                steps_recorded = 0
                ep_builder.discard()
                dataset.clear_episode_buffer()
                print(f"\nep {recorded+1} discarded. press Enter to re-record\n")
                if home:
                    target = dict(home)
                    pitch = home["wrist_flex"] + home["shoulder_lift"] + home["elbow_flex"]
                    ee_x, ee_y = 0.1629, 0.1131

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        ep_builder.save()
        print("\nfinalizing dataset...")
        try:
            dataset.finalize()
            print("dataset finalized")
        except Exception as e:
            print(f"ERROR finalizing: {e}")
            print("last parquet chunk might be corrupt, earlier episodes are fine")
        rs_pipeline.stop()
        cv2.destroyAllWindows()
        robot.disconnect()
        kb_listener.stop()
        print(f"\ndone - {recorded} episodes saved to '{root_path}'")


if __name__ == "__main__":
    main()
