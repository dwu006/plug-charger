"""
Replay recorded episodes on the SO101 robot.

Reads actions from the LeRobot parquet dataset and sends them to the real robot.

Usage:
    python replay.py                      # replay all episodes
    python replay.py --episode 0          # replay a specific episode
    python replay.py --data data/         # custom data folder
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

# ── Configuration ─────────────────────────────────────────────────────────────
ROBOT_PORT = "/dev/ttyACM0"
DATA_ROOT  = "data"
FPS        = 30
# ──────────────────────────────────────────────────────────────────────────────

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_dataset(data_root: str) -> pd.DataFrame:
    pattern = os.path.join(data_root, "data", "chunk-*", "file-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found at: {pattern}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return df


def replay_episode(robot: SO100Follower, frames: pd.DataFrame, episode_idx: int) -> None:
    print(f"[Replay] Episode {episode_idx} — {len(frames)} frames")
    for _, row in frames.iterrows():
        t0 = time.perf_counter()
        action_values = np.array(row["action"], dtype=float)
        action = {f"{m}.pos": float(v) for m, v in zip(MOTOR_NAMES, action_values)}
        robot.send_action(action)
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    print(f"[Replay] Episode {episode_idx} done.\n")


def main(episode: int = -1, data_root: str = DATA_ROOT) -> None:
    df = load_dataset(data_root)
    episodes = sorted(df["episode_index"].unique())
    print(f"[Replay] Found {len(episodes)} episode(s): {episodes}")

    robot = SO100Follower(SO100FollowerConfig(port=ROBOT_PORT, id="follower_arm", use_degrees=True))
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to SO101.")

    to_play = [episode] if episode >= 0 else episodes

    try:
        for ep in to_play:
            if ep not in episodes:
                print(f"[Replay] Episode {ep} not found. Available: {episodes}")
                continue
            frames = df[df["episode_index"] == ep].reset_index(drop=True)
            input(f"[Replay] Ready to replay episode {ep}. Press Enter to start...")
            replay_episode(robot, frames, ep)
    except KeyboardInterrupt:
        print("\n[Replay] Interrupted.")
    finally:
        robot.disconnect()
        print("[Replay] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=-1, help="Episode index to replay (-1 = all)")
    parser.add_argument("--data",    type=str, default=DATA_ROOT, help="Path to data folder")
    args = parser.parse_args()
    main(episode=args.episode, data_root=args.data)
